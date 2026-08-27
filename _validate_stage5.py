"""
Offline smoke test for Stage 5 (stage5_api.py) via FastAPI TestClient,
validating against the unified Stage 0 contracts (contracts.py).
No network needed - the OpenRouter client is mocked. Run: py _validate_stage5.py
"""

import base64
import io
import math
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image as PILImage
from schemas import DetectedPath, GeminiRoadResult, SpatialPolygon, TileFeaturesExtraction
from fastapi.testclient import TestClient

passed = []
failed = []


def report(name, ok, err=None):
    (passed if ok else failed).append((name, err))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {err!r}" if err else ""))


# ------------------------------------------------------------- mocks -------
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
        fmt = kwargs.get("response_format")
        if fmt is GeminiRoadResult:
            return _Resp(self._outer.roads_result)
        if fmt is TileFeaturesExtraction:
            idx = self._outer.tile_calls
            self._outer.tile_calls += 1
            return _Resp(self._outer.tile_results[idx % len(self._outer.tile_results)])
        raise AssertionError(f"Unexpected response_format {fmt}")


class _Chat:
    def __init__(self, outer):
        self.completions = _Completions(outer)


class _Beta:
    def __init__(self, outer):
        self.chat = _Chat(outer)


def ring_rect(x0, y0, x1, y1, n=40):
    """Ring of n points around a rectangle in 0-1000 space."""
    pts = []
    edges = [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]
    per_edge = max(2, n // 4)
    for ax, ay, bx, by in edges:
        for i in range(per_edge):
            t = i / per_edge
            pts.append([int(round(ay + (by - ay) * t)), int(round(ax + (bx - ax) * t))])
    return pts[:n]


def ring_circle(cx, cy, r, n=24):
    return [
        [int(round(cy + r * math.sin(2 * math.pi * i / n))),
         int(round(cx + r * math.cos(2 * math.pi * i / n)))]
        for i in range(n)
    ]


class SmartMockClient:
    """Dispatches canned results based on response_format type."""

    def __init__(self):
        self.calls = []
        self.tile_calls = 0
        dense_centerline = [
            [int(round(i * (1000 / 23))), 500] for i in range(24)  # 24 dense [y,x]
        ]
        self.roads_result = GeminiRoadResult(paths=[
            DetectedPath(path_id=0, name="main road",
                         centerline_1000=dense_centerline),
        ])
        self.tile_results = [
            TileFeaturesExtraction(
                water_bodies=[SpatialPolygon(polygon_1000=ring_circle(500, 500, 100))],
                tree_canopies=[SpatialPolygon(polygon_1000=ring_circle(500, 500, 480))],
                agricultural_zones=[SpatialPolygon(polygon_1000=ring_rect(50, 50, 950, 950))],
            ),
            TileFeaturesExtraction(
                water_bodies=[],
                tree_canopies=[SpatialPolygon(polygon_1000=ring_circle(700, 200, 80))],
                agricultural_zones=[SpatialPolygon(polygon_1000=ring_rect(50, 50, 950, 950))],
            ),
        ]
        self.beta = _Beta(self)


mock = SmartMockClient()

import config
import api.server as stage5_api
from contracts import (
    APIResponse as ContractAPIResponse,
    GeoExtractionPayload,
    MasterGISPayload,
    PathsExtractionPayload,
)

stage5_api.get_openrouter_client = lambda api_key=None: mock

client = TestClient(stage5_api.app)

W, H = 600, 400
img = np.random.default_rng(11).integers(0, 256, size=(H, W, 3), dtype=np.uint8)
buf = io.BytesIO()
PILImage.fromarray(img).save(buf, format="PNG")
png_bytes = buf.getvalue()

# ------------------------------------------------------------- tests -------
def is_jpeg_b64(b64_or_url: str) -> bool:
    """Accept raw Base64 JPEG or a full ``data:image/jpeg;base64,...`` URL."""
    payload = b64_or_url.split(",", 1)[1] if b64_or_url.startswith("data:") else b64_or_url
    try:
        raw = base64.b64decode(payload)
        return raw[:2] == b"\xff\xd8"
    except Exception:
        return False


def is_jpeg_data_url(value: str) -> bool:
    return value.startswith("data:image/jpeg;base64,") and is_jpeg_b64(value)


# ------------------------------------------- Pillar 1: VLM prompt audit ----
from core.path_extractor import ROAD_EXTRACTION_PROMPT as P2_PROMPT
from core.tile_segmentor import TILE_SEGMENTATION_PROMPT as P3_PROMPT

report(
    "S2 prompt: persona, TÂM ĐƯỜNG, shoulders/edges ban, occlusion momentum",
    P2_PROMPT.startswith("You are an expert Cadastral Remote Sensing Surveyor.")
    and "TÂM ĐƯỜNG" in P2_PROMPT
    and "shoulders" in P2_PROMPT and "medians" in P2_PROMPT
    and "building occlusions" in P2_PROMPT
    and "crossing lines" in P2_PROMPT,
)
report(
    "S2 prompt: [y, x] integer 0-1000 grounding directive",
    "[y, x]" in P2_PROMPT and "0-1000" in P2_PROMPT and "integer" in P2_PROMPT,
)
report(
    "S3 prompt: persona, 3 layers, concave 20-45 vertices",
    P3_PROMPT.startswith("You are an expert Remote Sensing Semantic Segmentation Engine.")
    and all(k in P3_PROMPT for k in ("water_bodies", "tree_canopies", "agricultural_zones"))
    and "concave polygons (20-45 vertices)" in P3_PROMPT,
)
report(
    "S3 prompt: seasonal invariance (GREEN/YELLOW/flooded/PLOWED BROWN/ĐỔ ẢI)",
    "GREEN" in P3_PROMPT and "YELLOW/STRAW" in P3_PROMPT
    and "flooded" in P3_PROMPT and "ĐỔ ẢI" in P3_PROMPT
    and "PLOWED BROWN" in P3_PROMPT,
)
report(
    "S3 prompt: negative constraints (residential compounds, yards, roads)",
    "residential compounds" in P3_PROMPT and "yards" in P3_PROMPT
    and "roads" in P3_PROMPT,
)
report(
    "S3 prompt: [y, x] integer 0-1000 grounding directive",
    "[y, x]" in P3_PROMPT and "0-1000" in P3_PROMPT,
)



r = client.get("/health")
report("GET /health 200", r.status_code == 200, str(r.status_code))
report(
    "health payload shape",
    r.json() == {"status": "healthy", "model": config.DEFAULT_MODEL,
                 "tiling_grid": "512x512"},
    r.text,
)

# --- paths endpoint (contracts.PathsExtractionPayload) ---
mock.tile_calls = 0
r = client.post("/api/v1/extract/paths",
                files={"image": ("img.png", png_bytes, "image/png")})
body = r.json()
report("paths: HTTP 200 + success envelope", r.status_code == 200 and body["success"] is True,
       r.text[:200])
d = body.get("data") or {}
report("paths: contract payload keys", set(d.keys()) == {
    "summary", "paths", "preview_image_base64"}, str(set(d.keys())))
report("paths: summary counts",
       d.get("summary", {}).get("path_count") == 1
       and d["summary"]["total_waypoints"] >= 20,
       str(d.get("summary")))
road0 = (d.get("paths") or [{}])[0]
report("paths: RoadFeature with int id + dense mapped centerline",
       road0.get("path_id") == 0
       and road0.get("name") == "main road"
       and road0.get("coordinates_pixel", [])[0] == [300, 0]
       and road0.get("coordinates_pixel", [])[-1] == [300, 400]
       and len(road0.get("coordinates_pixel", [])) == d["summary"]["total_waypoints"],
       str(road0)[:200])
PathsExtractionPayload.model_validate(d)
report("paths: validates against contracts.PathsExtractionPayload", True)
report("paths: preview is Base64 JPEG data URL",
       is_jpeg_data_url(d.get("preview_image_base64", "")))

# --- geo endpoint (contracts.GeoExtractionPayload) ---
mock.tile_calls = 0
r = client.post("/api/v1/extract/geo",
                files={"image": ("img.png", png_bytes, "image/png")})
body = r.json()
report("geo: HTTP 200 + success envelope", r.status_code == 200 and body["success"] is True,
       r.text[:200])
d = body.get("data") or {}
report("geo: contract payload keys", set(d.keys()) == {
    "summary", "agricultural_plots", "tree_canopies", "water_bodies",
    "preview_image_base64"}, str(set(d.keys())))
layers = {
    "water_bodies": d.get("water_bodies", []),
    "tree_canopies": d.get("tree_canopies", []),
    "agricultural_plots": d.get("agricultural_plots", []),
}
report("geo: summary counts match layer lengths", all(
    d["summary"][f"{k}_count"] == len(v) for k, v in layers.items()),
    str(d.get("summary")))
in_bounds = all(
    -1e-6 <= min(p[0] for p in f["geometry"]["exterior"])
    and max(p[0] for p in f["geometry"]["exterior"]) <= W
    and -1e-6 <= min(p[1] for p in f["geometry"]["exterior"])
    and max(p[1] for p in f["geometry"]["exterior"]) <= H
    for feats in layers.values() for f in feats
)
report("geo: polygons within absolute bounds", in_bounds)
GeoExtractionPayload.model_validate(d)
report("geo: validates against contracts.GeoExtractionPayload", True)
report("geo: preview is Base64 JPEG data URL",
       is_jpeg_data_url(d.get("preview_image_base64", "")))

# --- geo cleanup consistency: smoothed, hole-punched, unified agri -------
from core.tile_segmentor import run_tiling_segmentation as _rts
from core.spatial_engine import (
    CHAIKIN_ITERATIONS as _CI,
    chaikin_smooth as _cs,
    process_topology as _pt,
)

mock.tile_calls = 0
raw_replay = _rts(img.astype(np.uint8), mock, config.DEFAULT_MODEL)
expected_topo = _pt(
    raw_replay["water_bodies"], raw_replay["tree_canopies"],
    raw_replay["agricultural_zones"], road_linestrings=[],
)


def expected_ring(poly):
    xs, ys = poly.exterior.xy
    return _cs([[int(round(x)), int(round(y))] for x, y in zip(xs[:-1], ys[:-1])],
               iterations=_CI)


exp_water = sorted(expected_topo["water"], key=lambda p: p.area, reverse=True)
got_water = layers["water_bodies"][0]["geometry"]["exterior"]
report("geo: water JSON ring == Chaikin-smoothed cleaned geometry",
       got_water == expected_ring(exp_water[0]),
       f"n_got={len(got_water)}, n_exp={len(expected_ring(exp_water[0]))}")

exp_agri = sorted(expected_topo["agri_plots"], key=lambda p: p.area, reverse=True)
got_agri = layers["agricultural_plots"]
report(
    "geo: agri zones stay unified (no road split) and match cleaned geometry",
    len(got_agri) == 1
    and got_agri[0]["geometry"]["exterior"] == expected_ring(exp_agri[0]),
    f"count={len(got_agri)}",
)

# NEW (upgrade): hole preservation in serialized tree canopies.
exp_trees = sorted(expected_topo["trees"], key=lambda p: p.area, reverse=True)
tree_feats = {f["plot_id"]: f["geometry"] for f in layers["tree_canopies"]}
report(
    "geo: tree interiors preserved when water punches holes",
    any(len(g["interiors"]) >= 1
        for g in tree_feats.values()),
    str({k: len(v["interiors"]) for k, v in tree_feats.items()}),
)

# --- master endpoint (contracts.MasterGISPayload) ---
mock.tile_calls = 0
r = client.post("/api/v1/extract/master",
                files={"image": ("img.png", png_bytes, "image/png")})
body = r.json()
report("master: HTTP 200 + success envelope", r.status_code == 200 and body["success"] is True,
       r.text[:200])
d = body.get("data") or {}
report("master: contract payload keys", set(d.keys()) == {
    "summary", "transportation_network", "agricultural_plots",
    "tree_canopies", "water_bodies", "preview_image_base64"},
    str(set(d.keys())))
report("master: summary has all four count keys", set(d.get("summary", {})) == {
    "transportation_network_count", "agricultural_plots_count",
    "tree_canopies_count", "water_bodies_count"}, str(d.get("summary")))
report("master: road splits plots into >=2",
       len(d.get("agricultural_plots", [])) >= 2,
       f"plots={len(d.get('agricultural_plots', []))}")
report("master: transportation network present",
       d.get("summary", {}).get("transportation_network_count") == 1
       and d["transportation_network"][0]["coordinates_pixel"][0] == [300, 0]
       and d["transportation_network"][0]["coordinates_pixel"][-1] == [300, 400])
MasterGISPayload.model_validate(d)
report("master: validates against contracts.MasterGISPayload", True)
report("master: preview is Base64 JPEG data URL",
       is_jpeg_data_url(d.get("preview_image_base64", "")))

# --- error handling ---
r = client.post("/api/v1/extract/paths",
                files={"image": ("corrupt.png", b"not an image at all", "image/png")})
body = r.json()
report("corrupt image -> HTTP 400 failure envelope",
       r.status_code == 400 and body["success"] is False and body["errors"],
       r.text[:200])

r = client.post("/api/v1/extract/master",
                files={"image": ("empty.png", b"", "image/png")})
report("empty upload -> HTTP 400", r.status_code == 400 and r.json()["success"] is False)

stage5_api.get_openrouter_client = lambda api_key=None: (_ for _ in ()).throw(
    RuntimeError("no api key configured"))
r = client.post("/api/v1/extract/paths",
                files={"image": ("img.png", png_bytes, "image/png")})
report("client failure -> HTTP 500 failure envelope",
       r.status_code == 500 and r.json()["success"] is False, r.text[:200])
stage5_api.get_openrouter_client = lambda api_key=None: mock

# --- E: numpy_to_base64_jpeg data-URL helper ---
from utils.visualizer import numpy_to_base64_jpeg

data_url = numpy_to_base64_jpeg(img)
report(
    "E: numpy_to_base64_jpeg returns decodable JPEG data URL",
    data_url.startswith("data:image/jpeg;base64,")
    and is_jpeg_b64(data_url.split(",", 1)[1]),
)

# --- Pillar 3: upstream transport error mapping (429 -> 503, timeout -> 502)
import httpx
import openai


def _upstream_request():
    return httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def rate_limited_client():
    def _raise(api_key=None):
        raise openai.RateLimitError(
            "rate limited",
            response=httpx.Response(429, request=_upstream_request()),
            body=None,
        )
    return _raise


def timeout_client():
    def _raise(api_key=None):
        raise openai.APITimeoutError(request=_upstream_request())
    return _raise


stage5_api.get_openrouter_client = rate_limited_client()
r = client.post("/api/v1/extract/paths",
                files={"image": ("img.png", png_bytes, "image/png")})
report("upstream 429 -> HTTP 503 failure envelope",
       r.status_code == 503 and r.json()["success"] is False
       and "rate-limited" in r.json()["message"], r.text[:200])

stage5_api.get_openrouter_client = timeout_client()
r = client.post("/api/v1/extract/geo",
                files={"image": ("img.png", png_bytes, "image/png")})
report("upstream timeout -> HTTP 502 failure envelope",
       r.status_code == 502 and r.json()["success"] is False, r.text[:200])

stage5_api.get_openrouter_client = lambda api_key=None: mock

print("\n==============================")
print(f"TOTAL: {len(passed)} passed, {len(failed)} failed")
if failed:
    for name, err in failed:
        print(f"  FAILED: {name}: {err}")
    raise SystemExit(1)
print("ALL CHECKS PASSED")

