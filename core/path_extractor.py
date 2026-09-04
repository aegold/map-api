"""
Stage 2 - Global Transportation Pass.

Extracts topological linear centerlines of all visible thoroughfares,
canal dikes, and field ridge access tracks from the FULL-SCENE image in a
single structured-output API call.

Pipeline:
    image -> Base64 JPEG -> GeminiRoadResult (Pydantic-validated)
          -> absolute pixel polylines [x, y] -> Shapely LineStrings

Centerlines are kept as natural direct polylines: no B-spline or smoothing
is applied, so straight roads remain straight.
"""

from __future__ import annotations

import base64
from typing import Any, Sequence, Union

import numpy as np
from PIL import Image
from shapely.geometry import LineString
from shapely.ops import linemerge, snap, unary_union

import config
from schemas import DetectedPath, GeminiRoadResult
from core.input_engine import (
    JPEG_QUALITY,
    SatelliteImage,
    encode_jpeg_base64,
    load_image,
    normalized_to_pixel,
)
from core.spatial_engine import (
    _to_linestrings,  # noqa: F401 -- re-exported for sanitize_road_network
    smooth_open_linestring,
)

#: Embedded VLM prompt for the global transportation pass (cadastral grade).
ROAD_EXTRACTION_PROMPT: str = """ROLE & OBJECTIVE:
You are an expert Cadastral Remote Sensing Surveyor and Photogrammetry Engineer.
Analyze this aerial image and trace the exact CENTERLINES (TÂM ĐƯỜNG) of all visible road networks, paved streets, planned residential grids, and village alleys.

CRITICAL RULES:
1. PRESERVE STAGGERED T-JUNCTIONS (NO FAKE 4-WAY CROSSINGS):
   - In residential villages, paths frequently form offset/staggered T-junctions rather than aligned 4-way intersections.
   - Do NOT artificially align, bridge, or extend a dead-end village alley across a through-road into another branch if they are physically offset or separated by residential yards/houses.
   - Terminate an alley exactly where it meets a house gate, yard, or dead end.
2. PRESERVE MAIN THOROUGHFARE CONTINUITY:
   - When secondary lanes branch off from a curved or diagonal main road, the main road's continuous curvature must NOT be kinked, bent, or artificially pulled toward the intersection point.
3. RESIDENTIAL SUBDIVISIONS & PLANNED GRIDS:
   - For planned residential developments, trace EVERY internal dividing street and access road between plots, not just the outer perimeter.
4. ASPHALT CENTERLINE ACCURACY:
   - Trace strictly along the middle of the roadbed. Do NOT snap to sidewalk curbs, edges, or drainage ditches.
5. COORDINATE FORMAT:
   - Return ordered integer coordinates in normalized [y, x] space (0-1000 scale)."""

ImageLike = Union[str, bytes, np.ndarray, Image.Image, SatelliteImage]

#: Headroom for worst-case structured output (many long centerlines).
MAX_COMPLETION_TOKENS: int = 8192


def _to_rgb_array(image: ImageLike) -> np.ndarray:
    """Normalize any accepted image-like input into an RGB uint8 array."""
    if isinstance(image, SatelliteImage):
        return image.rgb_array
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"Expected an HxWx>=3 array, got shape {arr.shape}.")
        return np.ascontiguousarray(arr[:, :, :3])
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    # Delegate path / bytes handling to the Stage 1 loader.
    return load_image(image).rgb_array  # type: ignore[arg-type]


def _build_data_uri(rgb_array: np.ndarray, quality: int = JPEG_QUALITY) -> str:
    """Encode an RGB array as a Base64 JPEG data URI for vision payloads."""
    b64 = encode_jpeg_base64(rgb_array, quality=quality)
    return f"data:image/jpeg;base64,{b64}"


def _centerline_to_pixels(
    centerline_1000: Sequence[Sequence[int]], width: int, height: int
) -> list[list[int]]:
    """Convert normalized [y, x] points to absolute pixel [x, y] pairs.

    Collapses consecutive duplicate pixels and requires at least 2 distinct
    points for a valid polyline.
    """
    pixels: list[list[int]] = []
    for coord in centerline_1000:
        px, py = normalized_to_pixel(coord, width, height)
        point = [int(px), int(py)]
        if not pixels or point != pixels[-1]:
            pixels.append(point)
    return pixels


def extract_road_network(
    image_pil: ImageLike,
    client: Any,
    model_id: str,
) -> tuple[list[dict], list[LineString]]:
    """Run the global transportation pass over a full-scene image.

    Sends the entire image to the vision model in ONE structured-output call
    and converts the returned normalized centerlines into absolute pixel
    polylines and Shapely LineStrings.

    Args:
        image_pil: PIL image, SatelliteImage, RGB ndarray, raw bytes, or path.
        client: An OpenAI-compatible client pointed at OpenRouter
            (see core.input_engine.get_openrouter_client).
        model_id: Model identifier, e.g. config.DEFAULT_MODEL.

    Returns:
        Tuple of:
            roads_list: list of dicts with keys ``path_id``, ``name``,
                ``centerline_1000`` (normalized [y, x]) and
                ``centerline_px`` (absolute [x, y] pixels).
            geometries: list of Shapely ``LineString`` objects in pixel space.

    Raises:
        RuntimeError: If the model returns no parsed structured result.
    """
    rgb_array = _to_rgb_array(image_pil)
    height, width = int(rgb_array.shape[0]), int(rgb_array.shape[1])
    data_uri = _build_data_uri(rgb_array)

    response = client.beta.chat.completions.parse(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ROAD_EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        response_format=GeminiRoadResult,
        max_tokens=MAX_COMPLETION_TOKENS,
    )

    result = getattr(response.choices[0].message, "parsed", None)
    if not isinstance(result, GeminiRoadResult):
        raise RuntimeError(
            "Model did not return a parsed GeminiRoadResult; "
            "structured output extraction failed."
        )

    roads_list: list[dict] = []
    geometries: list[LineString] = []
    for path in result.paths:
        pixels = _centerline_to_pixels(path.centerline_1000, width, height)
        if len(pixels) < 2:
            # Degenerate polyline (all points collapsed); skip.
            continue
        # B-Spline smoothing removes 90-degree staircase / Manhattan
        # artifacts while preserving endpoints (see spatial_engine).
        pixels = smooth_open_linestring(pixels)
        if len(pixels) < 2:
            continue
        geometry = LineString([tuple(p) for p in pixels])
        roads_list.append(
            {
                "path_id": path.path_id,
                "name": path.name,
                "centerline_1000": path.centerline_1000,
                "centerline_px": pixels,
            }
        )
        geometries.append(geometry)

    return roads_list, geometries


def _extend_endpoints(line: LineString, dist: float) -> LineString:
    """Extend both endpoints of an open line by ``dist`` px along its
    first/last segment bearing, so short junction gaps close on snap."""
    coords = list(line.coords)
    if len(coords) < 2 or dist <= 0:
        return line
    (x0, y0), (x1, y1) = coords[0], coords[1]
    dx, dy = x1 - x0, y1 - y0
    seg = (dx * dx + dy * dy) ** 0.5
    if seg > 1e-9:
        x0 -= dist * dx / seg
        y0 -= dist * dy / seg
        coords[0] = (x0, y0)
    (x0, y0), (x1, y1) = coords[-2], coords[-1]
    dx, dy = x1 - x0, y1 - y0
    seg = (dx * dx + dy * dy) ** 0.5
    if seg > 1e-9:
        x1 += dist * dx / seg
        y1 += dist * dy / seg
        coords[-1] = (x1, y1)
    return LineString(coords)


def sanitize_road_network(
    road_lines, snap_tolerance: float = 9.0
):
    """Close small gaps at junctions before handing roads to Stage 4.

    Iteratively snaps each line's nodes onto the union of previously
    processed lines within ``snap_tolerance`` pixels, then merges everything.
    This repairs open T/X junctions (ngã ba / ngã tư bị hở) caused by
    per-path extraction.

    Args:
        road_lines: Extracted road ``LineString`` objects in pixel space
            (a single geometry, a ``MultiLineString``, a list, or ``None``).
        snap_tolerance: Maximum snapping distance in pixels (default 9.0 —
            just enough to seal touching gaps without coercing two offset
            staggered T-junctions into a fake 4-way crossing).

    Returns:
        List of snapped/merged ``LineString`` objects ready for Stage 4
        (always a list; empty when no valid input).
    """
    lines = _to_linestrings(road_lines)
    if not lines:
        return []
    if len(lines) == 1:
        return [lines[0]]

    # Extend endpoints so short branch-to-main gaps close on snap.
    snapped = [_extend_endpoints(lines[0], snap_tolerance)]
    for line in lines[1:]:
        reference = unary_union(snapped)
        snapped.append(snap(_extend_endpoints(line, snap_tolerance),
                            reference, snap_tolerance))
    merged = linemerge(unary_union(snapped))
    pieces = list(getattr(merged, "geoms", [merged])) or list(snapped)
    return [p for p in pieces if not p.is_empty]


__all__ = [
    "ROAD_EXTRACTION_PROMPT",
    "ImageLike",
    "MAX_COMPLETION_TOKENS",
    "extract_road_network",
    "sanitize_road_network",
]
