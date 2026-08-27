"""
Single Source of Truth for GIS data exchange across all pipeline modules
and REST API responses (Stage 0 unified API contracts).

Every public payload produced by the FastAPI service (Stage 5) must
serialize into these models.
"""

from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


# =====================================================================
# 1. BASE GEOMETRY ENTITY SCHEMAS
# =====================================================================
class RoadFeature(BaseModel):
    path_id: int
    name: str = Field(
        default="road",
        description="Path type identifier (e.g. 'Main Road', 'Canal Dike')",
    )
    coordinates_pixel: list[list[int]] = Field(
        description="List of [x, y] coordinates in absolute image pixel space"
    )


class PolygonGeometry(BaseModel):
    """Polygon geometry with hole preservation (Chaikin-smoothed rings)."""

    exterior: list[list[int]] = Field(
        description="Smoothed closed exterior ring as [[x, y], ...] pixel coordinates"
    )
    interiors: list[list[list[int]]] = Field(
        default_factory=list,
        description=(
            "List of interior hole rings (e.g. punched-out water inside "
            "tree canopies), each a [[x, y], ...] smoothed loop"
        ),
    )


class PolygonFeature(BaseModel):
    plot_id: int
    geometry: PolygonGeometry


# =====================================================================
# 2. LAYER-SPECIFIC PAYLOADS (WITH BASE64 PREVIEWS)
# =====================================================================
class PathsExtractionPayload(BaseModel):
    summary: dict[str, int] = Field(
        description="Count summary of extracted paths"
    )
    paths: list[RoadFeature]
    preview_image_base64: str = Field(
        description="Rendered JPEG preview of road centerlines encoded in Base64"
    )


class GeoExtractionPayload(BaseModel):
    summary: dict[str, int] = Field(
        description=(
            "Counts for 'agricultural_plots_count', 'tree_canopies_count',"
            " 'water_bodies_count'"
        )
    )
    agricultural_plots: list[PolygonFeature]
    tree_canopies: list[PolygonFeature]
    water_bodies: list[PolygonFeature]
    preview_image_base64: str = Field(
        description=(
            "Rendered JPEG preview of cleaned, smoothed, and hole-punched"
            " polygons in Base64"
        )
    )


# =====================================================================
# 3. MASTER COMPOSITE PAYLOAD
# =====================================================================
class MasterGISPayload(BaseModel):
    summary: dict[str, int] = Field(
        description=(
            "Object counts: 'transportation_network_count',"
            " 'agricultural_plots_count', 'tree_canopies_count',"
            " 'water_bodies_count'"
        )
    )
    transportation_network: list[RoadFeature]
    agricultural_plots: list[PolygonFeature]
    tree_canopies: list[PolygonFeature]
    water_bodies: list[PolygonFeature]
    preview_image_base64: str = Field(
        description="Dual-panel Master GIS preview encoded in Base64"
    )


# =====================================================================
# 4. UNIVERSAL API RESPONSE ENVELOPE
# =====================================================================
class APIResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[T] = None
    errors: Optional[list[str]] = None


__all__ = [
    "RoadFeature",
    "PolygonGeometry",
    "PolygonFeature",
    "PathsExtractionPayload",
    "GeoExtractionPayload",
    "MasterGISPayload",
    "APIResponse",
]
