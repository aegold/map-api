"""
Stage 5 - FastAPI REST service wrapping the 4-stage vectorization pipeline.

Endpoints:
    POST /api/v1/extract/paths   - road network only (Stage 1 -> 2)
    POST /api/v1/extract/geo     - semantic layers only (Stage 1 -> 3 + topo)
    POST /api/v1/extract/master  - full pipeline (Stage 1 -> 4)
    GET  /health                 - service health + active model

All extraction endpoints return a standardized APIResponse envelope with the
structured payload and a Base64 JPEG preview image. All blocking work
(image decoding, VLM calls, Shapely topology, matplotlib rendering) runs in
thread pools via asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

import config
from contracts import (
    APIResponse,
    GeoExtractionPayload,
    MasterGISPayload,
    PathsExtractionPayload,
    PolygonFeature,
    RoadFeature,
)
from schemas import GeminiRoadResult, TileFeaturesExtraction
from utils.visualizer import (
    numpy_image_to_base64,
    preview_b64 as _preview_b64,
    render_geo_cleaned_preview,
    render_master_overlay,
    render_stage2_preview,
)
from core.input_engine import load_image
from core.path_extractor import extract_road_network
from core.tile_segmentor import run_tiling_segmentation
from core.spatial_engine import (
    CHAIKIN_ITERATIONS,
    chaikin_smooth,
    process_topology,
)

logger = logging.getLogger("stage5_api")


__all__ = [
    "APIResponse",
    "PathsExtractionPayload",
    "GeoExtractionPayload",
    "MasterGISPayload",
    "numpy_image_to_base64",
]

# --------------------------------------------------------------------------
# Synchronous pipeline workers (executed inside asyncio.to_thread)
# --------------------------------------------------------------------------
def _decode_upload(image_bytes: bytes):
    """Decode uploaded bytes; raise ValueError for corrupt/empty images."""
    if not image_bytes:
        raise ValueError("Uploaded file is empty.")
    try:
        return load_image(image_bytes)
    except Exception as exc:
        raise ValueError(f"Could not decode image: {exc}") from exc


def _extract_paths_pipeline(image_bytes: bytes, model_id: str) -> dict:
    """Stage 1 -> Stage 2: road centerlines + overlay preview."""
    sat_img = _decode_upload(image_bytes)
    client = get_openrouter_client()
    roads_list, road_geoms = extract_road_network(sat_img, client, model_id)

    fig = render_stage2_preview(np.asarray(sat_img.rgb_array), roads_list)
    preview_b64 = _preview_b64(fig)

    return {
        "summary": {
            "path_count": len(roads_list),
            "total_waypoints": sum(len(r["centerline_px"]) for r in roads_list),
        },
        "paths": [
            RoadFeature(
                path_id=road["path_id"],
                name=road["name"],
                coordinates_pixel=road["centerline_px"],
            )
            for road in roads_list
        ],
        "preview_image_base64": preview_b64,
    }


def _polygon_features(polygons, prefix: str) -> list[PolygonFeature]:
    """Convert cleaned Shapely polygons to Chaikin-smoothed features.

    The emitted ``polygon_pixel`` ring is the exact geometry rendered in the
    preview, keeping JSON and visualization strictly consistent.
    """
    feats = []
    for i, poly in enumerate(
        sorted(polygons, key=lambda p: p.area, reverse=True), start=1
    ):
        xs, ys = poly.exterior.xy
        raw_ring = [[int(round(x)), int(round(y))]
                    for x, y in zip(xs[:-1], ys[:-1])]
        ring = chaikin_smooth(raw_ring, iterations=CHAIKIN_ITERATIONS)
        feats.append(PolygonFeature(feature_id=i, polygon_pixel=ring))
    return feats


def _extract_geo_pipeline(image_bytes: bytes, model_id: str) -> dict:
    """Stage 1 -> Stage 3 + standard topological cleanup (NO roads).

    Cleanup pipeline: per-layer unary union -> water hole-punching ->
    Chaikin smoothing (1 iteration) -> area filtering. The returned JSON
    coordinates and the preview image both render the SAME cleaned and
    smoothed polygons. Agricultural zones remain unified contiguous blocks
    (no road-buffer splitting).
    """
    sat_img = _decode_upload(image_bytes)
    client = get_openrouter_client()
    raw_polygons = run_tiling_segmentation(
        np.asarray(sat_img.rgb_array), client, model_id
    )

    # Topological cleanup WITHOUT cadastral road splitting.
    topo = process_topology(
        raw_polygons["water_bodies"],
        raw_polygons["tree_canopies"],
        raw_polygons["agricultural_zones"],
        road_linestrings=[],
    )

    water = _polygon_features(topo["water"], prefix="W")
    trees = _polygon_features(topo["trees"], prefix="T")
    agri = _polygon_features(topo["agri_plots"], prefix="A")
    layers = {
        "water_bodies": water,
        "tree_canopies": trees,
        "agricultural_zones": agri,
    }

    fig = render_geo_cleaned_preview(
        np.asarray(sat_img.rgb_array), layers, hole_polys=topo["trees"]
    )
    preview_b64 = _preview_b64(fig)

    return {
        "summary": {
            "water_bodies_count": len(water),
            "tree_canopies_count": len(trees),
            "agricultural_plots_count": len(agri),
        },
        "water_bodies": water,
        "tree_canopies": trees,
        "agricultural_plots": agri,
        "preview_image_base64": preview_b64,
    }


def _master_pipeline(image_bytes: bytes, model_id: str) -> dict:
    """Full Stage 1 -> 4 pipeline + dual-panel master preview."""
    sat_img = _decode_upload(image_bytes)
    client = get_openrouter_client()

    roads_list, road_geoms = extract_road_network(sat_img, client, model_id)
    raw_polygons = run_tiling_segmentation(
        np.asarray(sat_img.rgb_array), client, model_id
    )
    topo = process_topology(
        raw_polygons["water_bodies"],
        raw_polygons["tree_canopies"],
        raw_polygons["agricultural_zones"],
        road_linestrings=road_geoms,
    )

    transportation = [
        RoadFeature(
            path_id=road["path_id"],
            name=road["name"],
            coordinates_pixel=road["centerline_px"],
        )
        for road in roads_list
    ]
    agri = _polygon_features(topo["agri_plots"], prefix="P")
    trees = _polygon_features(topo["trees"], prefix="T")
    water = _polygon_features(topo["water"], prefix="W")

    fig = render_master_overlay(
        np.asarray(sat_img.rgb_array), roads_list,
        topo["agri_plots"], topo["trees"], topo["water"],
    )
    preview_b64 = _preview_b64(fig)

    return {
        "summary": {
            "transportation_network_count": len(transportation),
            "agricultural_plots_count": len(agri),
            "tree_canopies_count": len(trees),
            "water_bodies_count": len(water),
        },
        "transportation_network": transportation,
        "agricultural_plots": agri,
        "tree_canopies": trees,
        "water_bodies": water,
        "preview_image_base64": preview_b64,
    }


# --------------------------------------------------------------------------
# FastAPI application
# --------------------------------------------------------------------------
def get_openrouter_client(api_key: str | None = None):
    """Factory wrapper around Stage 1 client init (monkeypatch-friendly)."""
    from core.input_engine import get_openrouter_client as _factory

    return _factory(api_key=api_key)


app = FastAPI(
    title="Satellite GIS Vectorization API",
    description="Gemini-powered satellite image vectorization pipeline "
    "(roads, semantic layers, master GIS dataset).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _read_upload(image: UploadFile) -> bytes:
    """Read and size-check the uploaded image bytes."""
    data = await image.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds the "
            f"{config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    return data


def _error_response(status_code: int, message: str, errors: list[str]) -> JSONResponse:
    """Standardized failure envelope."""
    return JSONResponse(
        status_code=status_code,
        content=APIResponse(success=False, message=message, errors=errors,
                            data=None).model_dump(),
    )


def _classify_upstream_error(exc: Exception) -> int | None:
    """Map OpenRouter/OpenAI transport failures to HTTP status codes.

    Returns:
        503 for rate limits (HTTP 429), 502 for timeouts / connection
        drops, or None when the exception is not an upstream error.
    """
    try:
        import openai
    except ImportError:  # pragma: no cover - openai is a hard dependency.
        return None
    if isinstance(exc, openai.RateLimitError):
        return 503
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return 502
    if isinstance(exc, openai.InternalServerError):
        return 502
    return None


async def _run_endpoint(worker, image_bytes: bytes, model_id: str):
    """Dispatch a sync pipeline worker to a thread with error mapping."""
    try:
        payload = await asyncio.to_thread(worker, image_bytes, model_id)
        return APIResponse(success=True, message="Extraction completed.",
                           errors=[], data=payload)
    except ValueError as exc:
        # Corrupt / unsupported image.
        logger.warning("Bad request: %s", exc)
        return JSONResponse(
            status_code=400,
            content=APIResponse(success=False, message=str(exc),
                                errors=[str(exc)]).model_dump(),
        )
    except RuntimeError as exc:
        logger.error("OpenRouter/model failure: %s", exc)
        return JSONResponse(
            status_code=500,
            content=APIResponse(success=False,
                                message="Model inference failed.",
                                errors=[str(exc)]).model_dump(),
        )
    except Exception as exc:  # Topology, upstream transport, or unexpected.
        upstream_status = _classify_upstream_error(exc)
        if upstream_status == 503:
            logger.error("OpenRouter rate-limited (429): %s", exc)
            return JSONResponse(
                status_code=503,
                content=APIResponse(
                    success=False,
                    message="Upstream model provider is rate-limited. "
                            "Retry later.",
                    errors=[str(exc)]).model_dump(),
            )
        if upstream_status == 502:
            logger.error("OpenRouter unavailable (timeout/connection): %s",
                         exc)
            return JSONResponse(
                status_code=502,
                content=APIResponse(
                    success=False,
                    message="Upstream model provider timed out or is "
                            "unreachable.",
                    errors=[str(exc)]).model_dump(),
            )
        logger.exception("Pipeline failure")
        return JSONResponse(
            status_code=500,
            content=APIResponse(success=False,
                                message="Internal pipeline error.",
                                errors=[str(exc)]).model_dump(),
        )


@app.post("/api/v1/extract/paths", response_model=APIResponse[PathsExtractionPayload])
async def extract_paths(image: UploadFile = File(...)):
    """Extract road network centerlines only (Stage 1 -> 2)."""
    image_bytes = await _read_upload(image)
    return await _run_endpoint(_extract_paths_pipeline, image_bytes, config.DEFAULT_MODEL)


@app.post("/api/v1/extract/geo", response_model=APIResponse[GeoExtractionPayload])
async def extract_geo(image: UploadFile = File(...)):
    """Extract semantic polygon layers only (Stage 1 -> 3 + topo cleanup).

    Cleanup = per-layer union + water hole-punching + Chaikin smoothing +
    area filtering. No road extraction or plot splitting: agricultural
    zones stay unified contiguous blocks. JSON coordinates and the preview
    render the exact same cleaned polygons.
    """
    image_bytes = await _read_upload(image)
    return await _run_endpoint(_extract_geo_pipeline, image_bytes, config.DEFAULT_MODEL)


@app.post("/api/v1/extract/master", response_model=APIResponse[MasterGISPayload])
async def extract_master(image: UploadFile = File(...)):
    """Run the full composite pipeline (Stage 1 -> 4) + dual-panel render."""
    image_bytes = await _read_upload(image)
    return await _run_endpoint(_master_pipeline, image_bytes, config.DEFAULT_MODEL)


@app.get("/health")
async def health():
    """Service health status and active model configuration."""
    return {
        "status": "healthy",
        "model": config.DEFAULT_MODEL,
        "tiling_grid": f"{config.TILE_SIZE}x{config.TILE_SIZE}",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host=config.HOST, port=config.PORT,
                reload=config.DEBUG)

