"""
Stage 1 - Input handling for the satellite GIS vectorization pipeline.

Responsibilities:
    * Load a satellite image from a file path or raw bytes.
    * Expose image dimensions ``(width, height)`` and the RGB NumPy array.
    * Encode frames as high-quality Base64 JPEG for vision model payloads.
    * Initialize an OpenAI-compatible client pointed at OpenRouter.
    * Transform model-normalized ``[y, x]`` coordinates (0-1000) into
      absolute image pixel space ``[x, y]``.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from PIL import Image

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_APP_TITLE,
    OPENROUTER_BASE_URL,
    OPENROUTER_SITE_URL,
)

ImageSource = Union[str, Path, bytes]

#: JPEG quality used when encoding payloads for the vision model.
JPEG_QUALITY: int = 95


@dataclass(frozen=True)
class SatelliteImage:
    """A loaded satellite frame with its dimensions and RGB pixel data."""

    rgb_array: np.ndarray  # uint8, shape (height, width, 3)

    @property
    def height(self) -> int:
        return int(self.rgb_array.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb_array.shape[1])

    @property
    def dimensions(self) -> tuple[int, int]:
        """Return ``(width, height)`` of the image."""
        return self.width, self.height

    def encode_base64_jpeg(self, quality: int = JPEG_QUALITY) -> str:
        """Encode this frame as a high-quality Base64 JPEG string."""
        return encode_jpeg_base64(self.rgb_array, quality=quality)


def load_image(source: ImageSource) -> SatelliteImage:
    """Load a satellite image from a file path or raw bytes.

    Uses PIL first; falls back to OpenCV for formats PIL may reject.
    Always returns RGB-ordered data.

    Args:
        source: Path to an image file or raw image bytes.

    Returns:
        SatelliteImage with an ``(H, W, 3)`` uint8 RGB NumPy array.

    Raises:
        TypeError: If ``source`` is neither a path nor bytes.
        ValueError: If the image cannot be decoded by any backend.
    """
    raw_bytes: bytes
    if isinstance(source, bytes):
        raw_bytes = source
    elif isinstance(source, (str, Path)):
        raw_bytes = Path(source).read_bytes()
    else:
        raise TypeError(
            f"Unsupported image source type: {type(source)!r}. "
            "Expected a file path (str/Path) or raw bytes."
        )

    rgb_array = _decode_with_pil(raw_bytes)
    if rgb_array is None:
        rgb_array = _decode_with_opencv(raw_bytes)
    if rgb_array is None:
        raise ValueError("Could not decode image with PIL or OpenCV.")

    if rgb_array.ndim != 3 or rgb_array.shape[2] < 3:
        raise ValueError(
            f"Expected an HxWx>=3 image, got shape {rgb_array.shape}."
        )

    return SatelliteImage(rgb_array=np.ascontiguousarray(rgb_array[:, :, :3]))


def _decode_with_pil(data: bytes) -> np.ndarray | None:
    """Decode raw bytes to RGB via PIL. Returns None on failure."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def _decode_with_opencv(data: bytes) -> np.ndarray | None:
    """Decode raw bytes to RGB via OpenCV (BGR->RGB). Returns None on failure."""
    try:
        import cv2  # Optional dependency; imported lazily.

        buffer = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def encode_jpeg_base64(rgb_array: np.ndarray, quality: int = JPEG_QUALITY) -> str:
    """Encode an RGB uint8 array to a high-quality Base64 JPEG string.

    Args:
        rgb_array: ``(H, W, 3)`` uint8 RGB array.
        quality: JPEG quality (default 95).

    Returns:
        Base64-encoded JPEG (no data-URI prefix).
    """
    if rgb_array.dtype != np.uint8:
        rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb_array, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def get_openrouter_client(api_key: str | None = None):
    """Create an OpenAI client pointed at the OpenRouter endpoint.

    Args:
        api_key: Explicit key; falls back to config/env when omitted.

    Returns:
        An ``openai.OpenAI`` instance configured for OpenRouter.

    Raises:
        RuntimeError: If no API key is available.
        ImportError: If the ``openai`` package is not installed.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'openai' package is required. Install it with: pip install openai"
        ) from exc

    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise RuntimeError(
            f"Missing OpenRouter API key. Set the '{OPENROUTER_API_KEY_ENV}' "
            "environment variable or pass api_key explicitly."
        )

    default_headers = {"HTTP-Referer": OPENROUTER_SITE_URL} if OPENROUTER_SITE_URL else {}
    default_headers["X-Title"] = OPENROUTER_APP_TITLE

    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, default_headers=default_headers)


def normalized_to_pixel(coord: Sequence[float], width: int, height: int) -> tuple[int, int]:
    """Convert a normalized ``[y, x]`` coordinate (0-1000) to absolute pixels.

    Args:
        coord: Ordered ``[y, x]`` pair on the 0-1000 scale.
        width: Absolute image width in pixels.
        height: Absolute image height in pixels.

    Returns:
        ``(pixel_x, pixel_y)`` in absolute image pixel space, clamped to
        valid bounds.
    """
    y_norm, x_norm = float(coord[0]), float(coord[1])
    pixel_x = int(round((x_norm / 1000.0) * width))
    pixel_y = int(round((y_norm / 1000.0) * height))
    pixel_x = max(0, min(width, pixel_x))
    pixel_y = max(0, min(height, pixel_y))
    return pixel_x, pixel_y


def coords_to_pixels(
    coords: Sequence[Sequence[float]], width: int, height: int
) -> list[tuple[int, int]]:
    """Batch-convert a polyline/polygon of normalized ``[y, x]`` points to pixels."""
    return [normalized_to_pixel(coord, width, height) for coord in coords]


__all__ = [
    "ImageSource",
    "SatelliteImage",
    "JPEG_QUALITY",
    "load_image",
    "encode_jpeg_base64",
    "get_openrouter_client",
    "normalized_to_pixel",
    "coords_to_pixels",
]
