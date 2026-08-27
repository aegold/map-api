"""
Offline smoke test for Stage 2 (stage2_paths.py) - no API key or network
needed; the OpenAI client is mocked. Run: py _validate_stage2.py
"""

import os
import tempfile

import numpy as np
from PIL import Image
from shapely.geometry import LineString

import config
from schemas import DetectedPath, GeminiRoadResult
from core.path_extractor import (
    ROAD_EXTRACTION_PROMPT,
    _centerline_to_pixels,
    _to_rgb_array,
    extract_road_network,
)
from utils.visualizer import render_stage2_preview


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


report("prompt embedded verbatim", ROAD_EXTRACTION_PROMPT.startswith(
    "ROLE & OBJECTIVE:") and "TÂM ĐƯỜNG" in ROAD_EXTRACTION_PROMPT)
report("prompt: max-recall + continuity directives",
       all(k in ROAD_EXTRACTION_PROMPT for k in (
           "Photogrammetry Engineer", "EVERY visible road",
           "short dead-end spur trails", "MAXIMUM RECALL",
           "Do NOT omit faint", "MAINTAIN the trajectory and bridge the gap",
           "NO 90-DEGREE STEPS", "20 to 60 waypoints", "`paths`", "[y, x]")))

# ------------------------------------------------- input normalization -----
W, H = 800, 600
synthetic = np.zeros((H, W, 3), dtype=np.uint8)
synthetic[:, :, 1] = 120
pil_img = Image.fromarray(synthetic)

report("_to_rgb_array: ndarray", np.array_equal(_to_rgb_array(synthetic), synthetic))
report("_to_rgb_array: PIL Image", np.array_equal(_to_rgb_array(pil_img), synthetic))

# --------------------------------------------- coordinate conversion -------
px = _centerline_to_pixels([[0, 0], [500, 250], [1000, 1000]], W, H)
expected = [(0, 0), (int(round((250 / 1000.0) * W)), int(round(0.5 * H))), (W, H)]
report(
    "normalized [y,x] -> pixel [x,y] mapping",
    px == [[x, y] for x, y in expected],
)
dup = _centerline_to_pixels([[100, 100], [100, 100], [200, 300]], W, H)
report("duplicate consecutive points collapsed", len(dup) == 2)
deg = _centerline_to_pixels([[100, 100], [100, 100]], W, H)
report("degenerate polyline collapses to <2 pts", len(deg) < 2)

# ------------------------------------------------------ mock API client ----
class _Msg:
    def __init__(self, parsed):
        self.parsed = parsed


class _Choice:
    def __init__(self, parsed):
        self.message = _Msg(parsed)


class _Resp:
    def __init__(self, parsed):
        self.choices = [_Choice(parsed)]


class _ParseNS:
    def __init__(self, outer):
        self._outer = outer


class _Completions:
    def __init__(self, outer):
        self._outer = outer

    def parse(self, **kwargs):
        self._outer.last_kwargs = kwargs
        return self._outer.response


class _Chat:
    def __init__(self, outer):
        self.completions = _Completions(outer)


class _Beta:
    def __init__(self, outer):
        self.chat = _Chat(outer)


class MockClient:
    """Captures call kwargs and returns a canned structured result."""

    def __init__(self, parsed_result):
        self.response = _Resp(parsed_result)
        self.last_kwargs = None
        self.beta = _Beta(self)


def dense_line_yx(vertices_yx, n=24):
    """Resample a [y, x] polyline into n evenly spaced waypoints."""
    verts = [(float(v[0]), float(v[1])) for v in vertices_yx]
    if len(verts) < 2:
        return [[int(verts[0][0]), int(verts[0][1])]] * n
    seg_total = len(verts) - 1
    out = []
    for i in range(n):
        t = i / (n - 1)
        f = min(int(t * seg_total), seg_total - 1)
        u = t * seg_total - f
        (y0, x0), (y1, x1) = verts[f], verts[f + 1]
        out.append([int(round(y0 + (y1 - y0) * u)),
                    int(round(x0 + (x1 - x0) * u))])
    return out


straight = DetectedPath(path_id=0, name="main road",
                        centerline_1000=dense_line_yx(
                            [(100, 100), (500, 500), (900, 900)]))
bent = DetectedPath(path_id=1, name="canal dike",
                    centerline_1000=dense_line_yx(
                        [(50, 50), (500, 50), (950, 400)], n=30))
mocked = GeminiRoadResult(paths=[straight, bent])
client = MockClient(mocked)

roads, geoms = extract_road_network(pil_img, client, config.DEFAULT_MODEL)

# Schema audit: any waypoint count is accepted (>= 2 only). Reproduces the
# production 400 error: model returned 4..19 points -> must now be valid.
report("schema: sparse 19-point path accepted",
       not expect_raises(lambda: DetectedPath(
           path_id=0, name="road",
           centerline_1000=[[i * 50 % 1000, 500] for i in range(19)])))
report("schema: 4-point path accepted (real production case)",
       not expect_raises(lambda: DetectedPath(
           path_id=0, name="road",
           centerline_1000=[[566, 595], [478, 560], [413, 523], [362, 505]])))
report("schema: <2 points still rejected",
       expect_raises(lambda: DetectedPath(
           path_id=9, name="degenerate",
           centerline_1000=[[123, 123]])))

kwargs = client.last_kwargs
report(
    "single parse() call with model + response_format=GeminiRoadResult",
    kwargs is not None
    and kwargs.get("model") == config.DEFAULT_MODEL
    and kwargs.get("response_format") is GeminiRoadResult,
)

content = kwargs["messages"][0]["content"]
img_part = next(c for c in content if c["type"] == "image_url")
text_part = next(c for c in content if c["type"] == "text")
report(
    "payload has prompt text + base64 JPEG data URI",
    text_part["text"] == ROAD_EXTRACTION_PROMPT
    and img_part["image_url"]["url"].startswith("data:image/jpeg;base64,"),
)

report(
    "roads_list: 2 dense paths kept with expected ids",
    len(roads) == 2
    and roads[0]["path_id"] == 0
    and roads[1]["path_id"] == 1
    and set(roads[0].keys()) == {"path_id", "name", "centerline_1000", "centerline_px"},
)

r0, g0 = roads[0], geoms[0]
straight_px = r0["centerline_px"]
report(
    "pixel centerline matches exact formula (endpoints)",
    straight_px[0] == [int(round((100 / 1000.0) * W)), int(round((100 / 1000.0) * H))]
    and straight_px[-1] == [int(round((900 / 1000.0) * W)), int(round((900 / 1000.0) * H))],
    f"got {straight_px[:1]}..{straight_px[-1:]}",
)

# Straight input must stay perfectly straight -> all points collinear,
# i.e. LineString length equals endpoint distance (no spline warping).
endpoints_dist = LineString([tuple(straight_px[0]), tuple(straight_px[-1])]).length
report(
    "no spline warping: straight road stays straight",
    isinstance(g0, LineString) and abs(g0.length - endpoints_dist) < 1.0,
    f"len={g0.length}, endpoints={endpoints_dist}",
)

bent_expected_first = (int(round((50 / 1000.0) * W)), int(round((50 / 1000.0) * H)))
report(
    "bent polyline vertices preserved in order",
    list(geoms[1].coords)[0] == bent_expected_first,
    f"got {list(geoms[1].coords)[:2]}",
)

# --- NEW: sanitize_road_network junction repair ----------------------------
from core.path_extractor import sanitize_road_network

# Two collinear roads with a 10px gap at x=495..505 -> snap(12) must fuse.
line_a = LineString([(0, 500), (495, 500)])
line_b = LineString([(505, 500), (1000, 500)])
merged = sanitize_road_network([line_a, line_b])
report("sanitize: 10px junction gap fused into 1 piece", len(merged) == 1,
       f"pieces={len(merged)}")

# Gap larger than tolerance stays separate.
far_a = LineString([(0, 500), (450, 500)])
far_b = LineString([(550, 500), (1000, 500)])
merged_far = sanitize_road_network([far_a, far_b])
report("sanitize: >tolerance gap NOT bridged", len(merged_far) == 2)
report("sanitize: empty input safe", sanitize_road_network([]) == [])

# --- RECALL: short dead-end spur trails (2-3 pts, ~5px) must survive -------
spur_cases = [
    (0, [[100, 100], [105, 102]]),        # 2-point ~5px spur
    (1, [[100, 100], [104, 104], [108, 100]]),  # 3-point short spur
]
survived = True
for pid, spur in spur_cases:
    spur_res = GeminiRoadResult(paths=[
        DetectedPath(path_id=pid, name="spur",
                     centerline_1000=spur),
    ])
    spur_client = MockClient(spur_res)
    sr, sg = extract_road_network(synthetic, spur_client, "m")
    if len(sr) != 1 or len(sg) != 1:
        survived = False
    elif not (sg[0].is_valid and sg[0].length > 0):
        survived = False
report("recall: short 2-3pt spurs (~5px) are NOT dropped by pipeline",
       survived)

# --------------------------------------------------- error handling --------
class BadClient(MockClient):
    def __init__(self):
        super().__init__(GeminiRoadResult(paths=[]))
        self.response.choices[0].message.parsed = None


report("unparsed response raises RuntimeError", expect_raises(
    lambda: extract_road_network(synthetic, BadClient(), config.DEFAULT_MODEL)))

empty_client = MockClient(GeminiRoadResult(paths=[]))
roads_e, geoms_e = extract_road_network(synthetic, empty_client, "m")
report("no paths -> empty lists (not error)", roads_e == [] and geoms_e == [])

# ------------------------------------------------------- preview render ----
tmp_png = os.path.join(tempfile.gettempdir(), "stage2_preview_test.png")
fig = render_stage2_preview(synthetic, roads, save_path=tmp_png)
saved_ok = os.path.isfile(tmp_png) and os.path.getsize(tmp_png) > 0
os.unlink(tmp_png)
import matplotlib.pyplot as plt

plt.close(fig)
report("render_stage2_preview saves PNG + returns Figure", saved_ok)

print("\n==============================")
print(f"TOTAL: {len(passed)} passed, {len(failed)} failed")
if failed:
    for name, err in failed:
        print(f"  FAILED: {name}: {err}")
    raise SystemExit(1)
print("ALL CHECKS PASSED")

