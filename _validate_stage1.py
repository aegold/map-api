"""
Smoke test for Stage 1 foundation modules (config, schemas, stage1_input).
Run from the project root:  py _validate_stage1.py
"""

import base64
import io
import os
import tempfile

import numpy as np
from PIL import Image

import config
from schemas import (
    DetectedPath,
    GeminiRoadResult,
    SpatialPolygon,
    TileFeaturesExtraction,
    MIN_POLYGON_VERTICES,
    MAX_POLYGON_VERTICES,
)
from core.input_engine import (
    coords_to_pixels,
    encode_jpeg_base64,
    load_image,
    normalized_to_pixel,
)

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


# ---------------------------------------------------------------- config ----
try:
    assert config.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    assert config.DEFAULT_MODEL == "google/gemini-3.7-flash"
    assert config.TILE_SIZE == 512 and config.OVERLAP == 80
    assert config.STRIDE == 432, f"STRIDE={config.STRIDE}"
    assert config.ROAD_BUFFER_PX == 3.0
    assert config.MIN_WATER_AREA == 80
    assert config.MIN_TREE_AREA == 120
    assert config.MIN_AGRI_AREA == 400
    report("config values match spec", True)
except AssertionError as e:
    report("config values match spec", False, e)

# ---------------------------------------------------------------- images ----
W, H = 1234, 567
rng = np.random.default_rng(42)
synthetic = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)

tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
tmp.close()
Image.fromarray(synthetic).save(tmp.name)

img = load_image(tmp.name)
report(
    "load from path -> dims + pixel-exact RGB",
    img.dimensions == (W, H) and np.array_equal(img.rgb_array, synthetic),
    None if img.dimensions == (W, H) else f"dims={img.dimensions}",
)

raw = open(tmp.name, "rb").read()
os.unlink(tmp.name)
img_b = load_image(raw)
report("load from bytes -> identical to path load", np.array_equal(img_b.rgb_array, img.rgb_array))

report(
    "rgb dtype uint8 / shape (H,W,3)",
    img_b.rgb_array.dtype == np.uint8 and img_b.rgb_array.shape == (H, W, 3),
)

b64 = encode_jpeg_base64(img_b.rgb_array)
decoded = Image.open(io.BytesIO(base64.b64decode(b64)))
report(f"JPEG base64 round-trip ({len(b64)} chars)", decoded.size == (W, H) and decoded.mode == "RGB")

report("invalid source type rejected", expect_raises(lambda: load_image(12345)))

# ---------------------------------------------------------------- schemas ---
def ring(n):
    return [
        [int(round(500 + 400 * np.sin(a))), int(round(500 + 400 * np.cos(a)))]
        for a in np.linspace(0, 2 * np.pi, n, endpoint=False)
    ]


poly_ok = SpatialPolygon(polygon_1000=ring(30))
report("SpatialPolygon accepts 30 vertices", len(poly_ok.polygon_1000) == 30)

report(
    f"rejects <{MIN_POLYGON_VERTICES} and >{MAX_POLYGON_VERTICES} vertices",
    expect_raises(lambda: SpatialPolygon(polygon_1000=ring(MIN_POLYGON_VERTICES - 1)))
    and expect_raises(lambda: SpatialPolygon(polygon_1000=ring(MAX_POLYGON_VERTICES + 1))),
)


def oob_polygon():
    p = ring(25)
    p[0] = [1000, 1001]
    return SpatialPolygon(polygon_1000=p)


def non_pair_point():
    SpatialPolygon(polygon_1000=[[1, 2, 3]] * 25)


report("out-of-range coordinate rejected", expect_raises(oob_polygon))
report("malformed point rejected", expect_raises(non_pair_point))

path_ok = DetectedPath(path_id=1, name="main road",
                       centerline_1000=[[int(round(i % 1000)), int(round(i * 499 / 20))]
                                        for i in range(24)])
roads = GeminiRoadResult(paths=[path_ok])
report("DetectedPath / GeminiRoadResult valid", roads.paths[0].name == "main road"
       and len(roads.paths[0].centerline_1000) == 24)

# Sparse paths (as returned by the real model) are valid too - no 20-point minimum.
sparse_ok = DetectedPath(path_id=2, name="short track",
                         centerline_1000=[[124, 650], [151, 622], [836, 461]])
report("DetectedPath accepts sparse 3-point centerline", len(sparse_ok.centerline_1000) == 3)
report("DetectedPath accepts single-segment (2-point) centerline",
       len(DetectedPath(path_id=3, name="min",
                        centerline_1000=[[566, 595], [478, 560]]).centerline_1000) == 2)

feats = TileFeaturesExtraction(
    water_bodies=[SpatialPolygon(polygon_1000=ring(20))],
    tree_canopies=[],
    agricultural_zones=[SpatialPolygon(polygon_1000=ring(45))],
)
report(
    "TileFeaturesExtraction valid",
    len(feats.water_bodies) == 1 and feats.tree_canopies == [] and len(feats.agricultural_zones) == 1,
)

restored = GeminiRoadResult.model_validate_json(roads.model_dump_json())
report("Pydantic JSON round-trip", restored == roads)

import json as _json

_json.dumps(feats.model_dump())
report("plain-JSON serializable (response_format ready)", True)

report(
    "extra fields forbidden",
    expect_raises(lambda: DetectedPath(path_id=1, name="x",
                                       centerline_1000=[[0, 0], [1, 1]],
                                       bogus="nope")),
)

# ------------------------------------------------------- coordinate math ----
expected = (int(round((250 / 1000.0) * W)), int(round((500 / 1000.0) * H)))
report(
    "normalized_to_pixel formula ([y,x] -> (x,y))",
    normalized_to_pixel([500, 250], W, H) == expected
    and normalized_to_pixel([0, 0], W, H) == (0, 0)
    and normalized_to_pixel([1000, 1000], W, H) == (W, H),
)

pts = coords_to_pixels(ring(30), W, H)
report("coords_to_pixels batch helper", len(pts) == 30 and all(len(p) == 2 for p in pts))

# ---------------------------------------------------------------- client ----
from core.input_engine import get_openrouter_client

client = get_openrouter_client(api_key="test-key")
report("OpenAI client base_url -> OpenRouter", str(client.base_url).rstrip("/") == "https://openrouter.ai/api/v1")

saved_key = os.environ.pop("OPENROUTER_API_KEY", None)
import core.input_engine as s1_mod

s1_mod.OPENROUTER_API_KEY = None

missing_raised = False
try:
    s1_mod.get_openrouter_client()
except RuntimeError:
    missing_raised = True
finally:
    if saved_key:
        os.environ["OPENROUTER_API_KEY"] = saved_key
report("missing API key raises RuntimeError", missing_raised)

print("\n==============================")
print(f"TOTAL: {len(passed)} passed, {len(failed)} failed")
if failed:
    raise SystemExit(1)
print("ALL CHECKS PASSED")

