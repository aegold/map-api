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

#: Embedded VLM prompt for the global transportation pass (cadastral grade).
ROAD_EXTRACTION_PROMPT: str = """You are an expert Cadastral Remote Sensing Surveyor.
Analyze this aerial image and trace the exact CENTERLINES (TÂM ĐƯỜNG) of ALL visible transportation routes:

ROAD TYPES TO CAPTURE (bắt toàn bộ):
- Paved / asphalt roads (đường nhựa), concrete roads (đường bê tông).
- Sand / gravel roads (đường cát / sỏi), dirt roads (đường đất).
- Field footpaths (bờ mòn nội đồng) and hill-mountain trails (đường mòn đồi núi).

RULES:
1. Place waypoints along the geometric CENTERLINE of each path - NEVER on shoulders, edges, or medians.
2. DENSE SAMPLING: emit a high density of waypoints so the polyline hugs every hairpin bend and S-curve (switchbacks) exactly.
3. LINEAR MOMENTUM: keep trajectories straight through tree shadows AND building occlusions (do NOT detour).
4. NO BRIDGING GAPS: absolutely do NOT connect or bridge two separate road segments that are cut off by dense forest, steep hills, or slopes. STOP drawing immediately when the drivable surface ends.
5. Intersecting paths must snap cleanly to the main road coordinates at junctions.
6. DO NOT hallucinate paths or crossing lines across featureless plain agricultural plots / crop beds.

OUTPUT FORMAT: for each path emit a dense sequence of 20 to 60 waypoints as ordered integer [y, x] pairs on a 0-1000 scale relative to the image."""

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


def sanitize_road_network(
    road_lines: Sequence[LineString], snap_tolerance: float = 12.0
):
    """Close small gaps at junctions before handing roads to Stage 4.

    Iteratively snaps each line's nodes onto the union of previously
    processed lines within ``snap_tolerance`` pixels, then merges everything.
    This repairs open T/X junctions (ngã ba / ngã tư bị hở) caused by
    per-path extraction.

    Args:
        road_lines: Extracted road ``LineString`` objects in pixel space.
        snap_tolerance: Maximum snapping distance in pixels (default 12.0).

    Returns:
        List of snapped/merged ``LineString`` objects ready for Stage 4
        (empty list when no valid input).
    """
    lines = [line for line in road_lines if line is not None and not line.is_empty]
    if not lines:
        return []
    if len(lines) == 1:
        return [lines[0]]

    snapped = [lines[0]]
    for line in lines[1:]:
        reference = unary_union(snapped)
        snapped.append(snap(line, reference, snap_tolerance))
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
