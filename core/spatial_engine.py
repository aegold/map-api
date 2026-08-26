"""
Stage 4 - Topological Spatial Engine.

Resolves topological conflicts between the raw Stage 3 semantic layers,
splits contiguous agricultural plots along Stage 2 road corridors, smooths
boundaries with Chaikin's corner-cutting algorithm, exports the Master GIS
Dataset JSON, and renders the final dual-panel overlay.

Processing order:
    union per layer -> hole-punch trees/agri with water
    -> split agri plots by road corridor buffers -> area filtering
    -> Chaikin smoothing at export/render time.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

import config

#: Iterations of Chaikin corner cutting applied to polygon boundaries.
CHAIKIN_ITERATIONS: int = 1

#: Generalized corner-cutting ratio: each segment [P_i, P_i+1] is replaced
#: by points at (1 - t) and t along it, i.e. the 0.85 / 0.15 split.
CHAIKIN_CUT_T: float = 0.15


def chaikin_smooth(
    points: Sequence[Sequence[int]],
    iterations: int = CHAIKIN_ITERATIONS,
    t: float = CHAIKIN_CUT_T,
) -> list[list[int]]:
    """Apply generalized Chaikin corner-cutting to a closed ring of points.

    Each segment [P_i, P_i+1] is replaced by two points at ratios
    (1 - t) / t along it (default t = 0.15 -> 0.85 / 0.15 split):
        Q = (1 - t) * P_i + t * P_i+1
        R = t * P_i + (1 - t) * P_i+1
    The input is treated as a closed ring (the wrap-around segment from the
    last point back to the first is cut as well), so stairstepped pixel
    boundaries become smooth while the shape stays closed.

    Args:
        points: Ordered ``[x, y]`` vertices of a closed polygon.
        iterations: Number of corner-cutting passes (each doubles the
            vertex count).
        t: Cut ratio in (0, 0.5); 0.15 gives the standard 0.85/0.15 split.

    Returns:
        Smoothed ring as ``list[list[int]]``; degenerate inputs (<3 unique
        points) are returned unchanged.
    """
    pts: list[list[float]] = [[float(p[0]), float(p[1])] for p in points]
    if len(pts) < 3 or iterations < 1:
        return [[int(round(p[0])), int(round(p[1]))] for p in pts]

    for _ in range(iterations):
        new_pts: list[list[float]] = []
        n = len(pts)
        for i in range(n):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % n]
            # Skip zero-length segments to avoid duplicate points.
            if x0 == x1 and y0 == y1:
                continue
            new_pts.append([(1 - t) * x0 + t * x1, (1 - t) * y0 + t * y1])
            new_pts.append([t * x0 + (1 - t) * x1, t * y0 + (1 - t) * y1])
        pts = new_pts if new_pts else pts

    smoothed: list[list[int]] = []
    for p in pts:
        q = [int(round(p[0])), int(round(p[1]))]
        if not smoothed or q != smoothed[-1]:
            smoothed.append(q)
    # Collapse wrap-around duplicate after rounding.
    if len(smoothed) > 1 and smoothed[0] == smoothed[-1]:
        smoothed.pop()
    return smoothed


def smooth_polygon_rings(
    poly: Polygon, iterations: int = CHAIKIN_ITERATIONS
) -> tuple[list[list[int]], list[list[list[int]]]]:
    """Chaikin-smooth a polygon's exterior AND all interior hole rings.

    Args:
        poly: Shapely Polygon (may carry interior hole rings).
        iterations: Corner-cutting passes per ring.

    Returns:
        ``(exterior_ring, interior_rings)`` where every ring is a
        Chaikin-smoothed closed loop of ``[x, y]`` ints (without the
        duplicated closing vertex).
    """
    ext = [[int(round(x)), int(round(y))] for x, y in poly.exterior.coords[:-1]]
    exterior = chaikin_smooth(ext, iterations=iterations)
    interiors = []
    for ring in poly.interiors:
        pts = [[int(round(x)), int(round(y))] for x, y in ring.coords[:-1]]
        interiors.append(chaikin_smooth(pts, iterations=iterations))
    return exterior, interiors


def _iter_polygons(geom):
    """Yield individual Polygon pieces from any geometry."""
    if geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif hasattr(geom, "geoms"):  # MultiPolygon / GeometryCollection
        for g in geom.geoms:
            yield from _iter_polygons(g)


def process_topology(
    raw_water: Sequence[Polygon],
    raw_trees: Sequence[Polygon],
    raw_agri: Sequence[Polygon],
    road_linestrings: Sequence[LineString],
) -> dict:
    """Merge, hole-punch, road-split, and area-filter the raw semantic layers.

    Args:
        raw_water: Raw water body polygons (Stage 3).
        raw_trees: Raw tree canopy polygons (Stage 3).
        raw_agri: Raw agricultural zone polygons (Stage 3).
        road_linestrings: Road centerlines in pixel space (Stage 2).

    Returns:
        Dict with keys ``water``, ``trees``, ``agri_plots`` (lists of
        Shapely Polygons) and ``stats`` (per-layer counts before/after
        filtering).
    """
    # --- Union per layer ---------------------------------------------------
    merged_water = unary_union(list(raw_water)) if raw_water else Polygon()
    merged_trees = unary_union(list(raw_trees)) if raw_trees else Polygon()
    merged_agri = unary_union(list(raw_agri)) if raw_agri else Polygon()

    # --- Hole punching: water wins over vegetation/crops --------------------
    clean_trees = merged_trees.difference(merged_water) if not merged_trees.is_empty else merged_trees
    clean_agri = merged_agri.difference(merged_water) if not merged_agri.is_empty else merged_agri

    # --- Cadastral plot splitting along road corridors ----------------------
    if road_linestrings:
        road_corridor_buffers = unary_union(
            [line.buffer(config.ROAD_BUFFER_PX) for line in road_linestrings]
        )
        split_agri = (
            clean_agri.difference(road_corridor_buffers)
            if not clean_agri.is_empty
            else clean_agri
        )
    else:
        split_agri = clean_agri

    # --- Area filtering -----------------------------------------------------
    thresholds = {
        "water": config.MIN_WATER_AREA,
        "trees": config.MIN_TREE_AREA,
        "agri_plots": config.MIN_AGRI_AREA,
    }
    layers = {"water": merged_water, "trees": clean_trees, "agri_plots": split_agri}
    raw_counts = {
        "water": len(raw_water),
        "trees": len(raw_trees),
        "agri_plots": len(raw_agri),
    }

    result: dict = {}
    stats: dict = {}
    for key, geometry in layers.items():
        threshold = thresholds[key]
        pieces = [p for p in _iter_polygons(geometry)]
        kept = [p for p in pieces if p.area >= threshold]
        result[key] = sorted(kept, key=lambda p: p.area, reverse=True)
        stats[key] = {
            "raw_input": raw_counts[key],
            "after_topology": len(pieces),
            "after_area_filter": len(kept),
            "min_area_threshold": threshold,
        }

    result["stats"] = stats
    return result


def _polygon_to_ring(poly: Polygon, iterations: int = CHAIKIN_ITERATIONS) -> list[list[int]]:
    """Extract a Chaikin-smoothed closed ring of [x, y] ints from a Polygon."""
    xs, ys = poly.exterior.xy
    ring = [[int(round(x)), int(round(y))] for x, y in zip(xs[:-1], ys[:-1])]
    return chaikin_smooth(ring, iterations=iterations)


def export_master_gis(
    final_roads: Sequence[dict],
    final_plots: Sequence[Polygon],
    final_trees: Sequence[Polygon],
    final_water: Sequence[Polygon],
    output_json_path: str | Path,
    model_id: str | None = None,
) -> dict:
    """Export the Master GIS Dataset as structured JSON.

    Schema:
        {
          "metadata": {crs, generator, model, exported_at},
          "roads":  [{"road_id", "name", "centerline_px"}],
          "plots":  [{"plot_id", "polygon_px" (Chaikin-smoothed), "area_px2"}],
          "trees":  [{"tree_id", ...}], "water": [{"water_id", ...}]
        }

    Args:
        final_roads: Stage 2 roads_list dicts (path_id/name/centerline_px).
        final_plots: Final agricultural plot polygons.
        final_trees: Final tree canopy polygons (may contain water holes).
        final_water: Final water polygons.
        output_json_path: Destination JSON file path.
        model_id: Optional model id recorded in metadata.

    Returns:
        The dataset dict that was written to disk.
    """
    def features(geoms: Sequence[Polygon], prefix: str, id_field: str) -> list[dict]:
        out = []
        for i, geom in enumerate(sorted(geoms, key=lambda g: g.area, reverse=True), start=1):
            out.append({
                id_field: f"{prefix}{i}",
                "polygon_px": _polygon_to_ring(geom),
                "area_px2": round(float(geom.area), 2),
            })
        return out

    dataset = {
        "metadata": {
            "crs": "image-pixel",
            "generator": "satellite-gis-vectorizer stage4",
            "model": model_id or config.DEFAULT_MODEL,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "parameters": {
                "tile_size": config.TILE_SIZE,
                "overlap": config.OVERLAP,
                "stride": config.STRIDE,
                "road_buffer_px": config.ROAD_BUFFER_PX,
                "min_areas": {
                    "water": config.MIN_WATER_AREA,
                    "trees": config.MIN_TREE_AREA,
                    "agri": config.MIN_AGRI_AREA,
                },
                "chaikin_iterations": CHAIKIN_ITERATIONS,
            },
        },
        "roads": [
            {
                "road_id": f"R{idx}",
                "name": road.get("name", ""),
                "centerline_px": [[int(x), int(y)] for x, y in road["centerline_px"]],
            }
            for idx, road in enumerate(final_roads, start=1)
        ],
        "plots": features(final_plots, "P", "plot_id"),
        "trees": features(final_trees, "T", "tree_id"),
        "water": features(final_water, "W", "water_id"),
    }

    path = Path(output_json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2, ensure_ascii=False)
    return dataset


__all__ = [
    "CHAIKIN_ITERATIONS",
    "CHAIKIN_CUT_T",
    "chaikin_smooth",
    "smooth_polygon_rings",
    "process_topology",
    "export_master_gis",
]

