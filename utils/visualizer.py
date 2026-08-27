"""
Rendering routines & Base64 preview encoders for the GIS pipeline.

Single home for all matplotlib-based visualization and in-memory
NumPy -> JPEG Base64 encoding. Stage modules import from here instead of
embedding their own plotting boilerplate.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def numpy_image_to_base64(img_np: np.ndarray, quality: int = 90) -> str:
    """Encode an RGB(A) NumPy canvas to a Base64 JPEG string (in-memory).

    Args:
        img_np: ``(H, W, 3|4)`` uint8 array.
        quality: JPEG quality (default 90).

    Returns:
        Base64-encoded JPEG string without data-URI prefix.
    """
    arr = np.asarray(img_np)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(
        buffer, format="JPEG", quality=quality, optimize=True
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def numpy_to_base64_jpeg(img_np: np.ndarray, quality: int = 90) -> str:
    """Convert a NumPy/OpenCV canvas to a Base64 JPEG **data URL**.

    Returns ``data:image/jpeg;base64,<payload>`` with no disk I/O.
    """
    return f"data:image/jpeg;base64,{numpy_image_to_base64(img_np, quality=quality)}"


def figure_to_rgb(fig) -> np.ndarray:
    """Rasterize a matplotlib Figure into an RGB uint8 NumPy array."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[:, :, :3])


def preview_b64(fig, quality: int = 90) -> str:
    """Render a matplotlib Figure to a Base64 JPEG **data URL** (closes fig).

    Returns ``data:image/jpeg;base64,<payload>``.
    """
    import matplotlib.pyplot as plt

    rgb = figure_to_rgb(fig)
    plt.close(fig)
    return numpy_to_base64_jpeg(rgb, quality=quality)


# ===========================================================================
# Road network overlay (Stage 2)
# ===========================================================================
def render_stage2_preview(image_np: np.ndarray, roads_list, save_path=None):
    """Render road centerlines with thick white casing + red core + dots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, 12), facecolor="black")
    ax.imshow(np.asarray(image_np)[:, :, :3])

    for road in roads_list:
        pts = np.asarray(road["centerline_px"], dtype=float)
        xs, ys = pts[:, 0], pts[:, 1]
        label = f"#{road['path_id']} {road['name']}"
        ax.plot(xs, ys, color="white", linewidth=4, solid_capstyle="round",
                solid_joinstyle="round", zorder=2)
        ax.plot(xs, ys, color="red", linewidth=2, solid_capstyle="round",
                solid_joinstyle="round", zorder=3)
        ax.scatter(xs, ys, s=28, color="yellow", edgecolors="red",
                   linewidths=0.8, zorder=4)
        ax.annotate(label, xy=(xs[0], ys[0]), xytext=(4, -10),
                    textcoords="offset points", color="white", fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.55, pad=1.5),
                    zorder=5)

    legend_handles = [
        Line2D([0], [0], color="white", lw=6),
        Line2D([0], [0], color="red", lw=2.5),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="yellow",
               markeredgecolor="red", markersize=6, lw=0),
    ]
    ax.legend(legend_handles, ["casing", "road core", "waypoints"],
              loc="upper right", framealpha=0.5, fontsize=9)
    ax.set_title("Stage 2 - Global Transportation Network Preview",
                 color="white")
    ax.set_axis_off()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ===========================================================================
# Semantic layer overlays (Stage 3)
# ===========================================================================
def render_stage3_preview(
    image_np: np.ndarray,
    raw_polygons_dict: dict[str, list],
    tiles_info: list[dict] | None = None,
    save_path: str | None = None,
):
    """Render the tile grid and extracted semantic polygon overlays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    class_style = {
        "water_bodies": ("#00b4d8", "Water bodies"),
        "tree_canopies": ("#2d6a4f", "Tree canopies"),
        "agricultural_zones": ("#ffb703", "Agricultural zones"),
    }

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(np.asarray(image_np)[:, :, :3])

    if tiles_info:
        for tile in tiles_info:
            ax.add_patch(Rectangle(
                (tile["tx"], tile["ty"]), tile["tw"], tile["th"],
                fill=False, edgecolor="white", linewidth=1.0,
                linestyle="--", alpha=0.7, zorder=2,
            ))

    z = 3
    for cls, (color, _) in class_style.items():
        for poly in raw_polygons_dict.get(cls, []):
            geoms = getattr(poly, "geoms", [poly])  # buffer(0) may multipolygon-ize
            for g in geoms:
                xs, ys = g.exterior.xy
                ax.fill(xs, ys, color=color, alpha=0.35, zorder=z)
                ax.plot(xs, ys, color=color, linewidth=1.5, zorder=z + 1)
        z += 2

    legend = [Patch(facecolor=c, edgecolor=c, alpha=0.5, label=l)
              for c, l in class_style.values()]
    if tiles_info:
        legend.insert(0, Patch(facecolor="none", edgecolor="white",
                               linestyle="--", label="512px tile (+80px overlap)"))
    ax.legend(handles=legend, loc="upper right", framealpha=0.7)
    ax.set_title("Stage 3 - Tiling Semantic Segmentation Preview")
    ax.set_axis_off()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def render_geo_cleaned_preview(image_np: np.ndarray, layers: dict):
    """Render CLEANED + SMOOTHED geo layers over the source image.

    Draws exactly the coordinate lists contained in ``layers`` so the JSON
    response and the preview image always match — including interior hole
    rings serialized on each feature.

    Args:
        image_np: Base full-scene RGB array.
        layers: class name -> list of contract PolygonFeature models
            (with ``geometry.exterior`` / ``geometry.interiors``) or plain
            dicts carrying ``polygon_pixel`` / ``polygon_px`` keys.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    class_style = {
        "water_bodies": ("#1565c0", "Water bodies"),
        "tree_canopies": ("#1b5e20", "Tree canopies"),
        "agricultural_plots": ("#00e5ff", "Agricultural zones"),
        "agricultural_zones": ("#00e5ff", "Agricultural zones"),
    }
    outline_color = {"water_bodies": "#ff8c00"}  # Dark blue fill, orange outline.

    def _rings_of(feat):
        """Return (exterior_ring, interior_rings) for a feature."""
        geometry = getattr(feat, "geometry", None)
        if geometry is not None:
            return geometry.exterior, list(geometry.interiors)
        if isinstance(feat, dict):
            ring = feat.get("polygon_pixel") or feat.get("polygon_px") or []
            return ring, []
        return [], []

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(np.asarray(image_np)[:, :, :3])

    z = 3
    for cls, (color, _) in class_style.items():
        for feat in layers.get(cls, []):
            exterior, interiors = _rings_of(feat)
            if not exterior:
                continue
            pts = np.asarray(exterior, dtype=float)
            xs, ys = pts[:, 0], pts[:, 1]
            ax.fill(xs, ys, color=color, alpha=0.35, zorder=z)
            ax.plot(xs, ys, color=outline_color.get(cls, color),
                    linewidth=1.6, zorder=z + 1)
            # Serialized hole rings (punched-out water etc.).
            for hole in interiors:
                if len(hole) < 3:
                    continue
                hpts = np.asarray(hole, dtype=float)
                ax.plot(hpts[:, 0], hpts[:, 1], color="#00b4d8",
                        linewidth=1.2, linestyle=":", zorder=z + 2)
        z += 3

    ax.legend(handles=[
        Patch(facecolor=c, edgecolor=c, alpha=0.55, label=label)
        for c, label in class_style.values()
    ], loc="upper right", framealpha=0.7)
    ax.set_title("Geo Extraction - Cleaned & Smoothed Layers")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


__all__ = [
    "numpy_image_to_base64",
    "figure_to_rgb",
    "preview_b64",
    "render_stage2_preview",
    "render_stage3_preview",
    "render_geo_cleaned_preview",
    "render_master_overlay",
]


# ===========================================================================
# Master dual-panel overlay (Stage 4)
# ===========================================================================
def render_master_overlay(
    image_np,
    final_roads,
    final_plots,
    final_trees,
    final_water,
    save_path=None,
):
    """Render a dual-panel Master GIS comparison figure.

    Left panel: original image. Right panel: multi-layer overlay -
    cyan agricultural plots labeled P1, P2..., dark green tree canopies
    (with punched-out water holes drawn as interiors), blue ponds, and
    roads with white casing + red core.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(24, 13), facecolor="black")
    rgb = np.asarray(image_np)[:, :, :3]

    for ax in axes:
        ax.imshow(rgb)
        ax.set_axis_off()
        ax.set_facecolor("black")

    axes[0].set_title("Original Image", color="white", fontsize=14)

    ax = axes[1]
    for i, plot in enumerate(final_plots, start=1):
        xs, ys = plot.exterior.xy
        ax.fill(xs, ys, color="#00e5ff", alpha=0.40, zorder=3)
        ax.plot(xs, ys, color="#00b8d4", linewidth=1.8, zorder=4)
        if not plot.centroid.is_empty:
            ax.annotate(
                f"P{i}", xy=(plot.centroid.x, plot.centroid.y),
                ha="center", va="center", color="black", fontsize=10,
                fontweight="bold",
                bbox=dict(facecolor="#00e5ff", alpha=0.85, pad=1.5),
                zorder=7,
            )
    for geom in final_trees:
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, color="#1b5e20", alpha=0.55, zorder=5)
        ax.plot(xs, ys, color="#66bb6a", linewidth=1.2, zorder=6)
        for interior in geom.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, color="white", alpha=0.0, zorder=5)
            ax.plot(ix, iy, color="#00b4d8", linewidth=1.0,
                    linestyle=":", zorder=6)
    for geom in final_water:
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, color="#1565c0", alpha=0.75, zorder=6)
        ax.plot(xs, ys, color="#42a5f5", linewidth=1.2, zorder=7)
    for road in final_roads:
        pts = np.asarray(road["centerline_px"], dtype=float)
        xs, ys = pts[:, 0], pts[:, 1]
        ax.plot(xs, ys, color="white", linewidth=5.0,
                solid_capstyle="round", solid_joinstyle="round", zorder=8)
        ax.plot(xs, ys, color="red", linewidth=2.0,
                solid_capstyle="round", solid_joinstyle="round", zorder=9)

    ax.set_title("Master GIS Overlay", color="white", fontsize=14)
    legend_handles = [
        Patch(facecolor="#00e5ff", edgecolor="#00b8d4", alpha=0.6, label="Agri plots"),
        Patch(facecolor="#1b5e20", edgecolor="#66bb6a", alpha=0.7, label="Tree canopies"),
        Patch(facecolor="#1565c0", edgecolor="#42a5f5", alpha=0.8, label="Water"),
        Patch(facecolor="red", edgecolor="white", label="Roads"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.6)

    fig.suptitle("Master GIS Plot - Satellite Vectorization Result",
                 color="white", fontsize=16)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


