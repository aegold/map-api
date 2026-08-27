"""
Strict Pydantic schemas for deterministic structured outputs from Gemini
via OpenRouter.

Coordinate convention (used everywhere in this pipeline):
    * All model-facing coordinates are ordered pairs ``[y, x]`` on a
      normalized 0-1000 scale relative to the tile/image.
    * Conversion to absolute pixel space happens downstream
      (see core.input_engine.normalized_to_pixel).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Normalized coordinate scale used by the model.
COORD_SCALE: int = 1000

#: Inclusive bounds for the number of polygon vertices the model must emit.
MIN_POLYGON_VERTICES: int = 20
MAX_POLYGON_VERTICES: int = 45


def _validate_point_list(
    value: list[list[int]], min_points: int = 3, max_points: int | None = None
) -> list[list[int]]:
    """Validate a list of ``[y, x]`` integer coordinate pairs (0-1000 scale)."""
    cleaned: list[list[int]] = []
    for idx, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(
                f"Point {idx} must be an ordered [y, x] pair, got: {point!r}"
            )
        y, x = point
        if not isinstance(y, int) or isinstance(y, bool) or not isinstance(x, int) or isinstance(x, bool):
            raise ValueError(f"Point {idx} coordinates must be integers, got: {point!r}")
        if not (0 <= y <= COORD_SCALE and 0 <= x <= COORD_SCALE):
            raise ValueError(
                f"Point {idx} out of 0-{COORD_SCALE} range: [y={y}, x={x}]"
            )
        cleaned.append([int(y), int(x)])
    if len(cleaned) < min_points:
        raise ValueError(
            f"Expected at least {min_points} points, got {len(cleaned)}"
        )
    if max_points is not None and len(cleaned) > max_points:
        raise ValueError(
            f"Expected at most {max_points} points, got {len(cleaned)}"
        )
    return cleaned


class DetectedPath(BaseModel):
    """A single road/path centerline detected inside a tile.

    ``centerline_1000`` is a **dense** ordered polyline of ``[y, x]`` points
    on the 0-1000 normalized scale (20-60 waypoints) — dense enough to hug
    hairpin bends / S-curves — ordered from one end of the path to the other.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path_id: int = Field(..., description="Unique identifier of the detected path within the tile.")
    name: str = Field(..., description="Human-readable label of the path (e.g. 'highway', 'dirt track').")
    centerline_1000: list[list[int]] = Field(
        ...,
        description=(
            "Dense sequence of 20 to 60 waypoints [y, x] in 0-1000 scale, "
            "ordered along the path (dense enough for switchbacks/S-curves)."
        ),
    )

    @field_validator("path_id")
    @classmethod
    def _validate_path_id(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"path_id must be non-negative, got {v}")
        return v

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must be a non-empty string")
        return v

    @field_validator("centerline_1000")
    @classmethod
    def _validate_centerline(cls, v: list[list[int]]) -> list[list[int]]:
        # Dense waypoint requirement: enough points to hug switchbacks.
        return _validate_point_list(v, min_points=20, max_points=60)


class GeminiRoadResult(BaseModel):
    """Top-level structured output for road extraction on a single tile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paths: list[DetectedPath] = Field(
        default_factory=list,
        description="All road/path centerlines detected in the tile.",
    )


class SpatialPolygon(BaseModel):
    """A closed spatial feature polygon.

    ``polygon_1000`` is an ordered ring of 20-45 ``[y, x]`` vertices on the
    0-1000 normalized tile scale.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    polygon_1000: list[list[int]] = Field(
        ...,
        description=(
            f"Ordered polygon ring of {MIN_POLYGON_VERTICES}-{MAX_POLYGON_VERTICES} "
            "[y, x] vertices on a 0-1000 scale."
        ),
    )

    @field_validator("polygon_1000")
    @classmethod
    def _validate_polygon(cls, v: list[list[int]]) -> list[list[int]]:
        cleaned = _validate_point_list(v, min_points=MIN_POLYGON_VERTICES)
        if len(cleaned) > MAX_POLYGON_VERTICES:
            raise ValueError(
                f"polygon_1000 must have at most {MAX_POLYGON_VERTICES} vertices, "
                f"got {len(cleaned)}"
            )
        return cleaned


class TileFeaturesExtraction(BaseModel):
    """Top-level structured output for land-cover feature extraction on a tile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    water_bodies: list[SpatialPolygon] = Field(
        default_factory=list,
        description="Water bodies detected in the tile.",
    )
    tree_canopies: list[SpatialPolygon] = Field(
        default_factory=list,
        description="Tree canopy regions detected in the tile.",
    )
    agricultural_zones: list[SpatialPolygon] = Field(
        default_factory=list,
        description="Agricultural zones detected in the tile.",
    )


__all__ = [
    "COORD_SCALE",
    "MIN_POLYGON_VERTICES",
    "MAX_POLYGON_VERTICES",
    "DetectedPath",
    "GeminiRoadResult",
    "SpatialPolygon",
    "TileFeaturesExtraction",
]
