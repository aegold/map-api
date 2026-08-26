"""
End-to-end satellite image GIS vectorization pipeline (Stages 1-4).

Usage:
    py main.py --image path/to/satellite.jpg --output-dir ./output
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _timed(label: str, results: dict, fn, *args, **kwargs):
    """Run a pipeline step, log its duration and store outputs."""
    print(f"\n=== {label} ===")
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    results[label] = {"seconds": round(elapsed, 2)}
    print(f"[OK] {label} finished in {elapsed:.2f}s")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Satellite image -> Master GIS dataset vectorization "
        "pipeline using Gemini via OpenRouter."
    )
    parser.add_argument("--image", required=True,
                        help="Path to the input satellite image.")
    parser.add_argument("--output-dir", default="./output",
                        help="Directory for JSON export and previews "
                        "(default: ./output).")
    parser.add_argument("--model", default=None,
                        help="OpenRouter model id "
                        "(default from config.DEFAULT_MODEL).")
    parser.add_argument("--api-key", default=None,
                        help="OpenRouter API key "
                        "(falls back to OPENROUTER_API_KEY env var).")
    parser.add_argument("--skip-previews", action="store_true",
                        help="Skip PNG preview rendering.")
    return parser


def run_pipeline(image_path: str, output_dir: str, model_id: str,
                 api_key: str | None = None, skip_previews: bool = False) -> dict:
    """Execute Stages 1-4 over one image. Returns the timing/summary dict."""
    import config
    from core.input_engine import get_openrouter_client, load_image
    from core.path_extractor import extract_road_network
    from core.spatial_engine import (
        export_master_gis,
        process_topology,
    )
    from core.tile_segmentor import (
        generate_tile_grid,
        run_tiling_segmentation,
    )
    from utils.visualizer import render_master_overlay

    model_id = model_id or config.DEFAULT_MODEL
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timings: dict = {}
    total_start = time.perf_counter()

    # ------------------------------------------------ Stage 1: input ------
    sat_img = _timed("Stage 1 - Load image", timings, load_image, image_path)
    width, height = sat_img.dimensions
    print(f"    Image: {width}x{height}px")

    client = _timed(
        "Stage 1 - OpenRouter client", timings,
        lambda: get_openrouter_client(api_key=api_key),
    )

    # ------------------------------------------------ Stage 2: roads ------
    roads_list, road_geoms = _timed(
        "Stage 2 - Global transportation pass", timings,
        extract_road_network, sat_img, client, model_id,
    )
    print(f"    Roads detected: {len(roads_list)}")

    # ------------------------------------------------ Stage 3: tiles ------
    tiles = generate_tile_grid(width, height)
    print(f"    Tile grid: {len(tiles)} tiles "
          f"({config.TILE_SIZE}px, {config.OVERLAP}px overlap)")
    raw_polygons = _timed(
        "Stage 3 - Tiling semantic segmentation", timings,
        run_tiling_segmentation, sat_img.rgb_array, client, model_id,
    )
    for cls, polys in raw_polygons.items():
        print(f"    {cls}: {len(polys)} raw polygons")

    # ------------------------------------------------ Stage 4: topology ---
    def _run_topology():
        result = process_topology(
            raw_polygons["water_bodies"],
            raw_polygons["tree_canopies"],
            raw_polygons["agricultural_zones"],
            road_geoms,
        )
        json_path = str(out_dir / "master_gis_dataset.json")
        dataset = export_master_gis(
            roads_list, result["agri_plots"], result["trees"], result["water"],
            json_path, model_id=model_id,
        )
        print(f"    Final layers: {len(dataset['plots'])} plots, "
              f"{len(dataset['trees'])} tree zones, "
              f"{len(dataset['water'])} water bodies, "
              f"{len(dataset['roads'])} roads")
        print(f"    Exported: {json_path}")
        return dataset, result

    dataset, topo_result = _timed(
        "Stage 4 - Topology + Master GIS export", timings, _run_topology,
    )

    return {
        "timings": timings,
        "dataset": dataset,
        "topology": topo_result,
        "roads_list": roads_list,
        "raw_polygons": raw_polygons,
        "tiles": tiles,
        "out_dir": out_dir,
        "sat_img": sat_img,
    }


def _render_all_previews(context: dict) -> dict:
    """Render Stage 2/3 previews and the Master overlay; returns filenames."""
    import matplotlib.pyplot as plt
    from utils.visualizer import (
        render_master_overlay,
        render_stage2_preview,
        render_stage3_preview,
    )

    out_dir: Path = context["out_dir"]
    rgb = np.asarray(context["sat_img"].rgb_array)

    fig2 = render_stage2_preview(
        rgb, context["roads_list"],
        save_path=str(out_dir / "preview_stage2_roads.png"),
    )
    plt.close(fig2)
    fig3 = render_stage3_preview(
        rgb, context["raw_polygons"], context["tiles"],
        save_path=str(out_dir / "preview_stage3_semantic.png"),
    )
    plt.close(fig3)
    topo = context["topology"]
    fig4 = render_master_overlay(
        rgb, context["roads_list"], topo["agri_plots"], topo["trees"],
        topo["water"], save_path=str(out_dir / "master_overlay.png"),
    )
    plt.close(fig4)
    return {
        "stage2": "preview_stage2_roads.png",
        "stage3": "preview_stage3_semantic.png",
        "master": "master_overlay.png",
    }


def main(argv: list[str] | None = None) -> int:
    import config

    args = build_arg_parser().parse_args(argv)
    try:
        total_start = time.perf_counter()
        context = run_pipeline(
            image_path=args.image,
            output_dir=args.output_dir,
            model_id=args.model or config.DEFAULT_MODEL,
            api_key=args.api_key,
            skip_previews=True,
        )
        timings = context["timings"]

        if not args.skip_previews:
            print("\n=== Previews ===")
            start = time.perf_counter()
            previews = _render_all_previews(context)
            elapsed = time.perf_counter() - start
            timings["Previews"] = {"seconds": round(elapsed, 2)}
            for name in previews.values():
                print(f"    Saved: {context['out_dir'] / name}")
            print(f"[OK] Previews finished in {elapsed:.2f}s")

        total = time.perf_counter() - total_start
        timings["TOTAL"] = {"seconds": round(total, 2)}
        print(f"\n==============================")
        for step, info in timings.items():
            print(f"  {step}: {info['seconds']}s")
        print(f"\nPipeline complete -> {context['out_dir']}")
        return 0
    except Exception as exc:
        print(f"[ERROR] Pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

