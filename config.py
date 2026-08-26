"""
Central configuration for the satellite GIS vectorization pipeline.

All stages import their tunables from this module so behavior stays
consistent across the pipeline. Values can be overridden via a `.env`
file in the project root or plain environment variables (env vars win).
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv(filename: str = ".env") -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file next to this module.

    Existing environment variables are never overwritten. Supports blank
    lines, ``#`` comments, and optional surrounding quotes on values.
    """
    env_path = Path(__file__).resolve().parent / filename
    loaded: dict[str, str] = {}
    if not env_path.is_file():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


#: Keys loaded from `.env` (informational; effective values live in os.environ).
DOTENV_LOADED: dict[str, str] = _load_dotenv()


# ---------------------------------------------------------------------------
# OpenRouter API configuration
# ---------------------------------------------------------------------------
#: Default OpenRouter base URL (OpenAI-compatible endpoint). Can be overridden
#: via the OPENROUTER_BASE_URL environment variable.
OPENROUTER_BASE_URL: str = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

#: Environment variable name that holds the OpenRouter API key.
OPENROUTER_API_KEY_ENV: str = "OPENROUTER_API_KEY"

#: The API key is read lazily by stage code (get_openrouter_client) so that
#: importing this module never fails when the key is absent.
OPENROUTER_API_KEY: str | None = os.getenv(OPENROUTER_API_KEY_ENV)

#: Default model used for all vision / extraction calls.
DEFAULT_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemini-3.7-flash")

# Optional site attribution headers recommended by OpenRouter.
OPENROUTER_SITE_URL: str | None = os.getenv("OPENROUTER_SITE_URL")
OPENROUTER_APP_TITLE: str = os.getenv("OPENROUTER_APP_TITLE", "satellite-gis-vectorizer")

# ---------------------------------------------------------------------------
# Tiling parameters (pixels)
# ---------------------------------------------------------------------------
TILE_SIZE: int = int(os.getenv("TILE_SIZE", "512"))   # Tile side length in px.
OVERLAP: int = int(os.getenv("TILE_OVERLAP", os.getenv("OVERLAP", "80")))
#: Step between tile origins; defaults to TILE_SIZE - OVERLAP (= 432).
STRIDE: int = int(os.getenv("STRIDE", str(TILE_SIZE - OVERLAP)))

# ---------------------------------------------------------------------------
# Vectorization parameters
# ---------------------------------------------------------------------------
ROAD_BUFFER_PX: float = float(os.getenv("ROAD_BUFFER_PX", "3.0"))

# Minimum polygon area thresholds (pixels^2); smaller features are discarded.
MIN_WATER_AREA: int = int(os.getenv("MIN_WATER_AREA", "80"))
MIN_TREE_AREA: int = int(os.getenv("MIN_TREE_AREA", "120"))
MIN_AGRI_AREA: int = int(os.getenv("MIN_AGRI_AREA", "600"))

# ---------------------------------------------------------------------------
# REST service settings (Stage 5)
# ---------------------------------------------------------------------------
HOST: str = os.getenv("HOST", os.getenv("SERVICE_HOST", "0.0.0.0"))
PORT: int = int(os.getenv("PORT", os.getenv("SERVICE_PORT", "8000")))
DEBUG: bool = os.getenv(
    "DEBUG", os.getenv("SERVICE_DEBUG", "false")
).strip().lower() in ("1", "true", "yes", "on")

#: Maximum accepted upload size in bytes (25 MB).
MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024

__all__ = [
    "DOTENV_LOADED",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_API_KEY_ENV",
    "OPENROUTER_API_KEY",
    "DEFAULT_MODEL",
    "OPENROUTER_SITE_URL",
    "OPENROUTER_APP_TITLE",
    "TILE_SIZE",
    "OVERLAP",
    "STRIDE",
    "ROAD_BUFFER_PX",
    "MIN_WATER_AREA",
    "MIN_TREE_AREA",
    "MIN_AGRI_AREA",
    "HOST",
    "PORT",
    "DEBUG",
    "MAX_UPLOAD_BYTES",
]

