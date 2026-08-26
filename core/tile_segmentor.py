"""
Stage 3 - Local Tiling Semantic Pass.

Performs high-resolution semantic segmentation across a sliding tile grid
(512x512 px tiles with 80 px overlap) to extract boundary polygons for:

    * water_bodies      - fish ponds, aquaculture basins, lakes, canals
    * tree_canopies     - dense crowns / orchards / woodland belts
    * agricultural_zones- active farming plots (any crop state)

Each tile is analyzed independently; normalized tile-local coordinates
([y_t, x_t] on a 0-1000 scale) are mapped back to absolute global pixel
space and materialized as Shapely Polygons. Raw polygon lists are returned
without area filtering - thresholding belongs to a later stage
(see config.MIN_*_AREA).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

import config
from schemas import SpatialPolygon, TileFeaturesExtraction
from core.input_engine import JPEG_QUALITY, encode_jpeg_base64

#: Embedded VLM prompt used for every tile.
TILE_SEGMENTATION_PROMPT: str = """You are an expert Remote Sensing Semantic Segmentation Engine. Extract 3 core spatial layers:
1. 'water_bodies': Closed polygons for all standing water, fish ponds, and aquaculture basins.
2. 'tree_canopies': Dense concave polygons (20-45 vertices) tightly hugging tree crowns, orchards, and woodland belts. Exclude roads and water.
3. 'agricultural_zones': Polygons for active farming land, rice paddies, or crop fields. Seasonally invariant: include GREEN crops, YELLOW/STRAW ripe fields, MUDDY/FLOODED paddies (đổ ải), and PLOWED BROWN soil. Exclude houses, residential compounds, dirt courtyards, yards, and roads.

OUTPUT FORMAT: every polygon must be a closed loop of 20-45 vertices emitted as normalized integer [y, x] pairs on a 0-1000 scale relative to the tile."""

#: Headroom for worst-case structured output on dense tiles.
MAX_COMPLETION_TOKENS: int = 8192


def _axis_positions(length: int, tile_size: int, overlap: int) -> tuple[list[int], int]:
    """Sliding-window start positions along one axis with edge clamping.

    Returns:
        Tuple of (sorted unique start positions, actual window size along
        the axis). The window size shrinks when the image is smaller than
        the requested tile size.
    """
    window = min(tile_size, max(length, 1))
    if length <= tile_size:
        return [0], window

    stride = tile_size - overlap
    positions = list(range(0, length - tile_size + 1, stride))
    clamped_end = length - tile_size
    if positions[-1] != clamped_end:
        positions.append(clamped_end)
    return positions, window


def generate_tile_grid(
    img_w: int,
    img_h: int,
    tile_size: int = config.TILE_SIZE,
    overlap: int = config.OVERLAP,
) -> list[dict]:
    """Compute the sliding-window tile grid covering the entire image.

    Args:
        img_w: Full image width in pixels.
        img_h: Full image height in pixels.
        tile_size: Nominal tile side length.
        overlap: Overlap between adjacent tiles.

    Returns:
        List of dicts ``{"tx", "ty", "tw", "th"}``: tile top-left corner and
        actual (edge-clamped) dimensions. Tiles never extend past image
        bounds and every pixel is covered by at least one tile.
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image dimensions: {img_w}x{img_h}")
    if not 0 <= overlap < tile_size:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < tile_size "
            f"(got overlap={overlap}, tile_size={tile_size})"
        )

    xs, tw = _axis_positions(img_w, tile_size, overlap)
    ys, th = _axis_positions(img_h, tile_size, overlap)

    tiles: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for ty in ys:
        for tx in xs:
            key = (tx, ty)
            if key in seen:
                continue
            seen.add(key)
            tiles.append({"tx": tx, "ty": ty, "tw": tw, "th": th})
    return tiles


def _ring_to_polygon(
    ring: list[list[int]], tx: int, ty: int, tw: int, th: int
) -> Polygon | None:
    """Convert a normalized tile-local [y, x] ring to a global-pixel Polygon.

    Uses the strict grounding math:
        gx = int(round(tx + (pt[1] / 1000.0) * tw))
        gy = int(round(ty + (pt[0] / 1000.0) * th))
    Self-intersections are repaired via ``buffer(0)``; degenerate rings
    yield None.
    """
    coords: list[tuple[int, int]] = []
    for pt in ring:
        gx = int(round(tx + (pt[1] / 1000.0) * tw))
        gy = int(round(ty + (pt[0] / 1000.0) * th))
        if not coords or (gx, gy) != coords[-1]:
            coords.append((gx, gy))

    if len(coords) < 3:
        return None
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or getattr(poly, "geom_type", "") != "Polygon":
        return None
    return poly


def _convert_rings(rings, tx: int, ty: int, tw: int, th: int) -> list[Polygon]:
    """Batch-convert normalized rings to valid global-pixel Polygons.

    Accepts both raw ``[y, x]`` ring lists and :class:`SpatialPolygon`
    model instances (unwrapped via ``.polygon_1000``).
    """
    normalized = [
        item.polygon_1000 if isinstance(item, SpatialPolygon) else item
        for item in rings
    ]
    polygons = [
        p for p in (_ring_to_polygon(ring, tx, ty, tw, th) for ring in normalized)
        if p is not None
    ]
    return polygons


def extract_tile_features(
    crop_np: np.ndarray,
    tx: int,
    ty: int,
    tw: int,
    th: int,
    client: Any,
    model_id: str,
) -> dict:
    """Run semantic segmentation on a single tile crop.

    Args:
        crop_np: RGB uint8 array of the tile crop.
        tx, ty: Tile top-left corner in absolute image pixels.
        tw, th: Tile dimensions in pixels.
        client: OpenAI-compatible client pointed at OpenRouter.
        model_id: Model identifier, e.g. config.DEFAULT_MODEL.

    Returns:
        Dict with per-class lists of Shapely Polygons in absolute pixel
        space, per-class polygon counts, and tile metadata.
    """
    rgb = np.ascontiguousarray(crop_np[:, :, :3])
    data_uri = f"data:image/jpeg;base64,{encode_jpeg_base64(rgb, quality=JPEG_QUALITY)}"

    response = client.beta.chat.completions.parse(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TILE_SEGMENTATION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        response_format=TileFeaturesExtraction,
        max_tokens=MAX_COMPLETION_TOKENS,
    )

    result = getattr(response.choices[0].message, "parsed", None)
    if not isinstance(result, TileFeaturesExtraction):
        raise RuntimeError(
            "Model did not return a parsed TileFeaturesExtraction; "
            "tile segmentation failed."
        )

    water = _convert_rings(result.water_bodies, tx, ty, tw, th)
    trees = _convert_rings(result.tree_canopies, tx, ty, tw, th)
    agri = _convert_rings(result.agricultural_zones, tx, ty, tw, th)

    return {
        "water_bodies": water,
        "tree_canopies": trees,
        "agricultural_zones": agri,
        "counts": {
            "water_bodies": len(water),
            "tree_canopies": len(trees),
            "agricultural_zones": len(agri),
        },
        "tile": {"tx": tx, "ty": ty, "tw": tw, "th": th},
    }

def run_tiling_segmentation(
    image_np: np.ndarray,
    client: Any,
    model_id: str,
) -> dict[str, list[Polygon]]:
    """Slide the tile grid over the full scene and collect raw polygons.

    Args:
        image_np: Full-scene RGB uint8 array ``(H, W, 3)``.
        client: OpenAI-compatible client pointed at OpenRouter.
        model_id: Model identifier.

    Returns:
        ``{"water_bodies": [...], "tree_canopies": [...],
        "agricultural_zones": [...]}`` with raw Shapely Polygons in absolute
        pixel space (no area filtering applied).
    """
    rgb = np.asarray(image_np)[:, :, :3]
    height, width = int(rgb.shape[0]), int(rgb.shape[1])

    collected: dict[str, list[Polygon]] = {
        "water_bodies": [],
        "tree_canopies": [],
        "agricultural_zones": [],
    }

    for tile in generate_tile_grid(width, height):
        tx, ty = tile["tx"], tile["ty"]
        tw, th = tile["tw"], tile["th"]
        crop = rgb[ty : ty + th, tx : tx + tw]
        features = extract_tile_features(crop, tx, ty, tw, th, client, model_id)
        for cls in collected:
            collected[cls].extend(features[cls])

    return collected


__all__ = [
    "TILE_SEGMENTATION_PROMPT",
    "MAX_COMPLETION_TOKENS",
    "generate_tile_grid",
    "extract_tile_features",
    "run_tiling_segmentation",
    "render_stage3_preview",
]

