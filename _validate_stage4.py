"""
Offline smoke test for Stage 4 (stage4_spatial.py + main.py).
No API key or network needed. Run: py _validate_stage4.py
"""

import json
import os
import tempfile

from shapely.geometry import LineString, Point, Polygon, box

import config
from core.spatial_engine import (
    chaikin_smooth,
    export_master_gis,
    process_topology,
)
from utils.visualizer import render_master_overlay

passed = []
failed = []


def report(name, ok, err=None):
    (passed if ok else failed).append((name, err))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {err!r}" if err else ""))


# --------------------------------------------------------- chaikin ---------
sq = [[0, 0], [100, 0], [100, 100], [0, 100]]
s1 = chaikin_smooth(sq, iterations=1)
# 0.85 / 0.15 generalized corner-cutting (t = 0.15):
# segment [A, B] yields points at 85% and 15% along it.
expected_8 = {(85, 0), (15, 0), (100, 15), (100, 85), (15, 100), (85, 100),
              (0, 85), (0, 15)}
got_8 = {tuple(p) for p in s1}
report(
    "Chaikin 1 iteration on square: exact 8 cut points (0.85/0.15)",
    len(s1) == 8 and got_8 == expected_8,
    f"got {sorted(got_8)}",
)
report("corner vertex eliminated", (100, 0) not in got_8)

s2 = chaikin_smooth(sq, iterations=2)
report("2 iterations double the vertex count", len(s2) == 16, f"len={len(s2)}")

from core.spatial_engine import smooth_polygon_rings

holed = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]).buffer(10).difference(
    __import__("shapely.geometry", fromlist=["Point"]).Point(50, 50).buffer(5)
)
ext_s, int_s = smooth_polygon_rings(holed)
report(
    "smooth_polygon_rings smooths exterior + interior hole rings",
    len(ext_s) >= 3
    and len(int_s) == len(holed.interiors)
    and all(len(r) >= 3 for r in int_s),
    f"ext={len(ext_s)}, holes={len(int_s)}",
)

report("iterations=0 returns rounded input", chaikin_smooth(sq, iterations=0) == sq)

degenerate = chaikin_smooth([[0, 0], [10, 10]])
report("degenerate ring (<3 pts) unchanged", degenerate == [[0, 0], [10, 10]])

jagged = [[0, 0], [10, 0], [10, 10], [20, 10], [20, 20], [30, 20], [30, 30]]
smoothed_jagged = chaikin_smooth(jagged, iterations=1)
report("stairstepped boundary gets extra vertices", len(smoothed_jagged) == 14)

# ------------------------------------------------------- topology ----------
agri_big = box(100, 100, 900, 900)
water_mid = box(200, 200, 400, 400)
trees_overlapping_water = box(150, 150, 450, 450)
road_v = LineString([(500, 0), (500, 1000)])

topo = process_topology(
    raw_water=[water_mid],
    raw_trees=[trees_overlapping_water],
    raw_agri=[agri_big],
    road_linestrings=[road_v],
)

report("union keeps single water body", len(topo["water"]) == 1)
report(
    "trees get hole punched where water overlaps",
    len(topo["trees"]) == 1 and len(topo["trees"][0].interiors) == 1,
    f"interiors={len(topo['trees'][0].interiors) if topo['trees'] else 'n/a'}",
)
tree_hole_area = abs(topo["trees"][0].area - trees_overlapping_water.area)
report("hole area equals water overlap", 39900 < tree_hole_area < 40100,
       f"delta={tree_hole_area:.1f}")

report("vertical road splits plot into 2", len(topo["agri_plots"]) == 2,
       f"n={len(topo['agri_plots'])}")
if len(topo["agri_plots"]) == 2:
    widths = sorted(p.bounds[2] - p.bounds[0] for p in topo["agri_plots"])
    report("split pieces flank the 6px corridor", 390 < widths[0] < 400
           and 390 < widths[1] < 400, f"widths={widths}")
report("stats block present with filter counts",
       all(k in topo.get("stats", {}) for k in ("water", "trees", "agri_plots")))

# Union merge: two overlapping agri squares -> one plot.
merged = process_topology(
    raw_water=[], raw_trees=[],
    raw_agri=[box(0, 0, 100, 100), box(90, 90, 190, 190)],
    road_linestrings=[],
)
report("overlapping polygons union into one", len(merged["agri_plots"]) == 1
       and abs(merged["agri_plots"][0].area - 19900) < 5,
       f"area={merged['agri_plots'][0].area if merged['agri_plots'] else 'n/a'}")

# --------------------------------------------------- area filtering --------
tiny_water = box(300, 300, 308, 308)          # 64 < MIN_WATER_AREA (80)
tiny_trees = box(600, 600, 610, 610)          # 100 < MIN_TREE_AREA (120)
tiny_agri = box(700, 700, 715, 715)           # 225 < MIN_AGRI_AREA (400)
filtered = process_topology(
    raw_water=[tiny_water], raw_trees=[tiny_trees], raw_agri=[tiny_agri],
    road_linestrings=[],
)
report(
    "area thresholds drop small fragments per layer",
    filtered["water"] == [] and filtered["trees"] == []
    and filtered["agri_plots"] == [],
)
report("filter stats recorded", all(
    filtered["stats"][k]["after_area_filter"] == 0
    for k in ("water", "trees", "agri_plots")
))
report("empty inputs tolerated", expect_ok := (
    process_topology([], [], [], [])["stats"] is not None))

# --- NEW (upgrade): morphological closing heals tile slivers ---------------
from core.spatial_engine import extract_clean_polygons

sliver_agri = [box(0, 0, 499.7, 300), box(500, 0, 1000, 300)]  # 0.3px crack
closed = process_topology(raw_water=[], raw_trees=[], raw_agri=sliver_agri,
                          road_linestrings=[])
report("buffer-closing fuses 0.3px tile crack into 1 plot",
       len(closed["agri_plots"]) == 1,
       f"pieces={len(closed['agri_plots'])}")

wide_gap = process_topology(raw_water=[], raw_trees=[],
                            raw_agri=[box(0, 0, 480, 300), box(520, 0, 1000, 300)],
                            road_linestrings=[])
report("closing does NOT fuse genuinely separate plots",
       len(wide_gap["agri_plots"]) == 2)

# --- GeometryCollection safety ---------------------------------------------
from shapely.geometry import GeometryCollection

gc = GeometryCollection([box(0, 0, 100, 200), LineString([(0, 0), (50, 50)])])
cleaned = extract_clean_polygons(gc.buffer(0) if not gc.is_valid else gc)
report("extract_clean_polygons skips non-polygon members", len(cleaned) == 1)

# --- NoneType/iteration hardening (production "NoneType not iterable") -----
from shapely.geometry import MultiLineString, LineString as _LS

agri = [box(0, 0, 1000, 500)]
for tag, roads in [
    ("None", None),
    ("single LineString", _LS([(100, 0), (100, 500)])),
    ("[None] list", [None]),
    ("MultiLineString", MultiLineString([[(100, 0), (100, 500)], [(300, 0), (300, 500)]])),
]:
    try:
        t = process_topology(raw_water=[], raw_trees=[], raw_agri=agri,
                             road_linestrings=roads)
        head = "OK"
    except Exception as e:
        head = f"CRASH {e!r}"
    report(f"process_topology handles roads={tag} ({head})",
           head == "OK", head)

# sanitize_road_network must never raise / must return a list for weird input
from core.path_extractor import sanitize_road_network as _srn

report("sanitize_road_network([None]) returns [] without crash",
       _srn([None]) == [])
report("sanitize_road_network(None) returns [] without crash",
       _srn(None) == [])


# ------------------------------------------------------ JSON export --------
roads_list = [
    {"path_id": 0, "name": "main road", "centerline_px": [[500, 0], [500, 1000]]},
    {"path_id": 1, "name": "dike track", "centerline_px": [[0, 500], [1000, 500]]},
]
tmp_dir = tempfile.mkdtemp()
json_path = os.path.join(tmp_dir, "master_gis_dataset.json")
dataset = export_master_gis(
    roads_list, topo["agri_plots"], topo["trees"], topo["water"],
    json_path, model_id=config.DEFAULT_MODEL,
)

with open(json_path, encoding="utf-8") as fh:
    loaded = json.load(fh)

report(
    "JSON file written with Master GIS schema keys",
    set(loaded.keys()) == {"metadata", "roads", "plots", "trees", "water"},
)
report("IDs assigned P1/T1/W1/R1",
       loaded["plots"][0]["plot_id"] == "P1"
       and loaded["trees"][0]["tree_id"] == "T1"
       and loaded["water"][0]["water_id"] == "W1"
       and loaded["roads"][0]["road_id"] == "R1")
report("features sorted by area descending", all(
    loaded[layer][i]["area_px2"] >= loaded[layer][i + 1]["area_px2"]
    for layer in ("plots", "trees", "water")
    for i in range(len(loaded[layer]) - 1)))
report(
    "exported boundaries are Chaikin-smoothed (vertex count grew)",
    len(loaded["plots"][0]["polygon_px"]) > 4,
    f"vertices={len(loaded['plots'][0]['polygon_px'])}",
)
report("metadata records model and parameters",
       loaded["metadata"]["model"] == config.DEFAULT_MODEL
       and loaded["metadata"]["parameters"]["road_buffer_px"] == config.ROAD_BUFFER_PX)
report("roads carry names and pixel centerlines",
       loaded["roads"][0]["name"] == "main road"
       and loaded["roads"][0]["centerline_px"] == [[500, 0], [500, 1000]])

# ------------------------------------------------------------ render -------
import numpy as np

img = np.random.default_rng(3).integers(0, 256, size=(1000, 1000, 3), dtype=np.uint8)
png_path = os.path.join(tmp_dir, "master_overlay.png")
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

fig = render_master_overlay(img, roads_list, topo["agri_plots"],
                            topo["trees"], topo["water"], save_path=png_path)
saved_ok = isinstance(fig, Figure) and os.path.isfile(png_path) \
    and os.path.getsize(png_path) > 0
plt.close(fig)
report("render_master_overlay saves PNG + returns Figure", saved_ok)

# ------------------------------------------------------------ main.py ------
import subprocess
import sys

help_run = subprocess.run(
    [sys.executable, "main.py", "--help"], capture_output=True, text=True,
)
report("main.py --help exits cleanly", help_run.returncode == 0
       and "--image" in help_run.stdout)

import main as main_mod

report("main module imports and exposes pipeline", callable(main_mod.run_pipeline))

print("\n==============================")
print(f"TOTAL: {len(passed)} passed, {len(failed)} failed")
if failed:
    raise SystemExit(1)
print("ALL CHECKS PASSED")

