"""
Offline smoke test for Stage 3 (stage3_polygons.py) - no API key or network
needed; the OpenAI client is mocked. Run: py _validate_stage3.py
"""

import os

import numpy as np
from shapely.geometry import Polygon

import config
from schemas import SpatialPolygon, TileFeaturesExtraction
from core.tile_segmentor import (
    TILE_SEGMENTATION_PROMPT,
    _ring_to_polygon,
    extract_tile_features,
    generate_tile_grid,
    run_tiling_segmentation,
)
from utils.visualizer import render_stage3_preview


passed = []
failed = []


def report(name, ok, err=None):
    (passed if ok else failed).append((name, err))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {err!r}" if err else ""))


def expect_raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


report("prompt embedded verbatim",
       TILE_SEGMENTATION_PROMPT.startswith("ROLE & OBJECTIVE:")
       and "Remote Sensing Semantic Segmentation Engine" in TILE_SEGMENTATION_PROMPT)

# --------------------------------------------------------- tile grid -------
def covers_all(tiles, w, h, step=10):
    """Sample-check every point is inside some tile."""
    for y in range(0, h, step):
        for x in range(0, w, step):
            if not any(
                t["tx"] <= x < t["tx"] + t["tw"] and t["ty"] <= y < t["ty"] + t["th"]
                for t in tiles
            ):
                return False
    return True


tiles_1k = generate_tile_grid(1000, 800)
report("1000x800 grid: full coverage", covers_all(tiles_1k, 1000, 800), str(len(tiles_1k)))
report(
    "1000x800 grid: expected windows",
    sorted({t['tx'] for t in tiles_1k}) == [0, 432, 488]
    and sorted({t['ty'] for t in tiles_1k}) == [0, 288],
    f"xs={sorted({t['tx'] for t in tiles_1k})}, ys={sorted({t['ty'] for t in tiles_1k})}",
)

for size in ((512, 512), (431, 700), (300, 250), (2048, 1536)):
    w, h = size
    report(f"grid coverage {w}x{h}", covers_all(generate_tile_grid(w, h), w, h))

small = generate_tile_grid(200, 150)
report(
    "small image -> single clamped tile",
    small == [{"tx": 0, "ty": 0, "tw": 200, "th": 150}],
    str(small),
)

report("invalid dims raise", expect_raises(lambda: generate_tile_grid(0, 100)))
report("overlap >= tile_size raises", expect_raises(lambda: generate_tile_grid(1000, 1000, overlap=512)))

# --------------------------------------------- ring -> polygon mapping -----
sq_ring = [[0, 0], [0, 1000], [1000, 1000], [1000, 0]]
poly = _ring_to_polygon(sq_ring, tx=100, ty=200, tw=512, th=512)
report(
    "ring->Polygon uses exact formula gx=tx+(x/1000)*tw",
    poly is not None
    and abs(poly.bounds[0] - (100 + 0)) < 1e-9
    and abs(poly.bounds[1] - (200 + 0)) < 1e-9
    and abs(poly.bounds[2] - (100 + 512.0)) < 1e-9
    and abs(poly.bounds[3] - (200 + 512.0)) < 1e-9,
)

bowtie = [[500, 0], [500, 1000], [0, 0], [0, 1000]] * 6  # self-intersecting
fixed = _ring_to_polygon(bowtie[:24], 0, 0, 512, 512)
report(".buffer(0) repairs self-intersection", fixed is not None and fixed.is_valid)

degenerate = _ring_to_polygon([[5, 5], [5, 5], [7, 7]], 0, 0, 512, 512)
report("degenerate ring dropped", degenerate is None)

# Audit A-3: tile-local grounding must emit strict int(round()) pixel coords.
p_int = _ring_to_polygon(
    [[0, 0], [0, 1000], [1000, 1000], [1000, 0]], tx=13, ty=29, tw=333, th=777
)
report(
    "A3: tile grounding emits strict int(round()) pixel coords",
    p_int is not None
    and all(v == int(v) for x, y in p_int.exterior.coords for v in (x, y))
    and p_int.bounds == (13.0, 29.0, 346.0, 806.0),
    str(p_int.bounds if p_int else None),
)

# -------------------------------------------------- mocked tile pass -------
class _Msg:
    def __init__(self, parsed):
        self.parsed = parsed


class _Choice:
    def __init__(self, parsed):
        self.message = _Msg(parsed)


class _Resp:
    def __init__(self, parsed):
        self.choices = [_Choice(parsed)]


class _Completions:
    def __init__(self, outer):
        self._outer = outer

    def parse(self, **kwargs):
        self._outer.calls.append(kwargs)
        return self._outer.response


class _Chat:
    def __init__(self, outer):
        self.completions = _Completions(outer)


class _Beta:
    def __init__(self, outer):
        self.chat = _Chat(outer)


class MockClient:
    """One canned TileFeaturesExtraction per parse() call."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []
        self.beta = _Beta(self)

    def next_result(self):
        return self.results[len(self.calls) % len(self.results)] if self.results else None

    @property
    def response(self):
        return _Resp(self.next_result())


def ring_at(cx_norm, cy_norm, r_norm=100, n=24):
    import math

    return [
        [
            int(round(cy_norm + r_norm * math.sin(2 * math.pi * i / n))),
            int(round(cx_norm + r_norm * math.cos(2 * math.pi * i / n))),
        ]
        for i in range(n)
    ]


results = [
    TileFeaturesExtraction(
        water_bodies=[SpatialPolygon(polygon_1000=ring_at(500, 500))],
        tree_canopies=[],
        agricultural_zones=[],
    ),
    TileFeaturesExtraction(
        water_bodies=[],
        tree_canopies=[SpatialPolygon(polygon_1000=ring_at(300, 700, r_norm=80))],
        agricultural_zones=[SpatialPolygon(polygon_1000=ring_at(700, 300, r_norm=150))],
    ),
]
client = MockClient(results)

H, W = 1000, 800
img = np.random.default_rng(7).integers(0, 256, size=(H, W, 3), dtype=np.uint8)

raw = run_tiling_segmentation(img, client, config.DEFAULT_MODEL)

n_tiles = len(generate_tile_grid(W, H))
report(
    "one parse() call per tile",
    len(client.calls) == n_tiles,
    f"calls={len(client.calls)}, tiles={n_tiles}",
)
first_kwargs = client.calls[0]
content = first_kwargs["messages"][0]["content"]
report(
    "payload: prompt + data URI + response_format=TileFeaturesExtraction",
    first_kwargs["model"] == config.DEFAULT_MODEL
    and first_kwargs["response_format"] is TileFeaturesExtraction
    and any(c["type"] == "text" and c["text"] == TILE_SEGMENTATION_PROMPT for c in content)
    and any(c["type"] == "image_url" and c["image_url"]["url"].startswith("data:image/jpeg;base64,") for c in content),
)

report(
    "raw dict has all 3 classes with expected polygon counts",
    set(raw.keys()) == {"water_bodies", "tree_canopies", "agricultural_zones"}
    and len(raw["water_bodies"]) == n_tiles // len(results)
    and len(raw["tree_canopies"]) == n_tiles // len(results)
    and len(raw["agricultural_zones"]) == n_tiles // len(results),
    str({k: len(v) for k, v in raw.items()}),
)

in_bounds = all(
    -1e-6 <= p.bounds[0] and p.bounds[2] <= W + 1e-6
    and -1e-6 <= p.bounds[1] and p.bounds[3] <= H + 1e-6
    for cls_polygons in (raw["water_bodies"], raw["tree_canopies"], raw["agricultural_zones"])
    for p in cls_polygons
)
report("all polygons within absolute image bounds", in_bounds)

report("unparsed response raises RuntimeError", expect_raises(
    lambda: extract_tile_features(img[:512, :512], 0, 0, 512, 512,
                                  MockClient([None]), "m")))

# ------------------------------------------------------------ preview ------
from matplotlib.figure import Figure
import tempfile
import matplotlib.pyplot as plt

tiles_info = generate_tile_grid(W, H)
tmp_png = os.path.join(tempfile.gettempdir(), "stage3_preview_test.png")
fig = render_stage3_preview(img, raw, tiles_info, save_path=tmp_png)
saved_ok = isinstance(fig, Figure) and os.path.isfile(tmp_png) and os.path.getsize(tmp_png) > 0
os.unlink(tmp_png)
plt.close(fig)
report("render_stage3_preview saves PNG + returns Figure", saved_ok)

fig2 = render_stage3_preview(img, {"water_bodies": [], "tree_canopies": [], "agricultural_zones": []}, None)
plt.close(fig2)
report("preview works without tiles_info / empty polygons", True)

print("\n==============================")
print(f"TOTAL: {len(passed)} passed, {len(failed)} failed")
if failed:
    for name, err in failed:
        print(f"  FAILED: {name}: {err}")
    raise SystemExit(1)
print("ALL CHECKS PASSED")

