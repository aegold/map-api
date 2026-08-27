# 📘 API_DOCS — Satellite GIS Vectorization Service

> Tài liệu tham khảo API ngắn gọn · Tiếng Việt · Cập nhật theo hợp đồng `contracts.py`

---

## 1. Giới Thiệu Nhanh (Quick Overview)

- **Base URL mặc định:** `http://localhost:8000` (hoặc domain server production).
- **Swagger UI tương tác:** [`/docs`](http://localhost:8000/docs).
- **Phong bì phản hồi chung:** mọi endpoint đều trả `APIResponse[T]` gồm 4 trường:

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `success` | `bool` | `true` khi xử lý thành công |
| `message` | `string` | Mô tả kết quả / lỗi |
| `data` | `object \| null` | Payload chính (theo từng endpoint) |
| `errors` | `list[string] \| null` | Danh sách chi tiết lỗi (nếu có) |

- **Quy ước ảnh preview:** trả về trực tiếp dạng **Data URL Base64** → bind thẳng vào `<img src="...">`, không cần decode thủ công:

```text
data:image/jpeg;base64,/9j/4AAQSkZJRg...
```

---

## 2. Danh Sách Các Endpoints

### Endpoint 1 — Trích Xuất Riêng Mạng Lưới Đường

**`POST /api/v1/extract/paths`**

- **Mô tả:** Trích xuất tim đường (`TÂM ĐƯỜNG`), đường mòn, đê kênh toàn cảnh (Stage 1 → Stage 2). Polyline giữ nguyên hình học thật, không nội suy cong.
- **Request:** `multipart/form-data`, key `image` (tệp PNG/JPG ≤ 25 MB).

**cURL mẫu:**

```bash
curl -X POST http://localhost:8000/api/v1/extract/paths \
  -F "image=@duong/dan/anh_ve_tinh.png"
```

**Phản hồi JSON mẫu:**

```json
{
  "success": true,
  "message": "Extraction completed.",
  "data": {
    "summary": {
      "path_count": 8,
      "total_waypoints": 126
    },
    "paths": [
      {
        "path_id": 0,
        "name": "Main Road",
        "coordinates_pixel": [[120, 45], [340, 180]]
      }
    ],
    "preview_image_base64": "data:image/jpeg;base64,..."
  },
  "errors": []
}
```

---

### Endpoint 2 — Trích Xuất Lớp Địa Vật Đã Làm Sạch Topo

**`POST /api/v1/extract/geo`**

- **Mô tả:** Trích xuất Ruộng / Cây / Ao hồ đã **hợp nhất (union)**, **đục lỗ ao trong tán cây**, **làm mượt Chaikin**, và **lọc diện tích tối thiểu**. KHÔNG trích xuất đường, KHÔNG cắt thửa.
- **Request:** `multipart/form-data`, key `image`.

**cURL mẫu:**

```bash
curl -X POST http://localhost:8000/api/v1/extract/geo \
  -F "image=@duong/dan/anh_ve_tinh.png"
```

**Phản hồi JSON mẫu:**

```json
{
  "success": true,
  "message": "Extraction completed.",
  "data": {
    "summary": {
      "water_bodies_count": 3,
      "tree_canopies_count": 4,
      "agricultural_plots_count": 5
    },
    "agricultural_plots": [
      {
        "plot_id": 1,
        "geometry": {
          "exterior": [[10, 20], [100, 20], [100, 80], [10, 20]],
          "interiors": []
        }
      }
    ],
    "tree_canopies": [
      { "plot_id": 1, "geometry": { "exterior": [[120, 40], [160, 55], [140, 90]], "interiors": [] } }
    ],
    "water_bodies": [
      { "plot_id": 1, "geometry": { "exterior": [[200, 210], [260, 220], [240, 270]], "interiors": [] } }
    ],
    "preview_image_base64": "data:image/jpeg;base64,..."
  },
  "errors": []
}
```

---

### Endpoint 3 — Pipeline Địa Chính Master (Đầy Đủ 4 Giai Đoạn)

**`POST /api/v1/extract/master`**

- **Mô tả:** Chạy trọn vẹn pipeline: union → đục lỗ ao → buffer tim đường `3px` → **cắt thửa ruộng theo hành lang đường** → Chaikin smoothing → render ảnh đối chiếu 2 bảng (gốc ↔ overlay).
- **Request:** `multipart/form-data`, key `image`.

**cURL mẫu:**

```bash
curl -X POST http://localhost:8000/api/v1/extract/master \
  -F "image=@duong/dan/anh_ve_tinh.png"
```

**Phản hồi JSON mẫu:**

```json
{
  "success": true,
  "message": "Extraction completed.",
  "data": {
    "summary": {
      "transportation_network_count": 8,
      "agricultural_plots_count": 14,
      "tree_canopies_count": 6,
      "water_bodies_count": 4
    },
    "transportation_network": [
      { "path_id": 0, "name": "Main Road", "coordinates_pixel": [[120, 45], [340, 180]] }
    ],
    "agricultural_plots": [
      { "plot_id": 1, "geometry": { "exterior": [[10, 20], [100, 20], [100, 80], [10, 20]], "interiors": [] } }
    ],
    "tree_canopies": [ { "plot_id": 1, "geometry": { "exterior": [[120, 40], [160, 55]], "interiors": [] } } ],
    "water_bodies": [ { "plot_id": 1, "geometry": { "exterior": [[200, 210], [260, 220]], "interiors": [] } } ],
    "preview_image_base64": "data:image/jpeg;base64,..."
  },
  "errors": []
}
```

---

### Endpoint 4 — Health Check

**`GET /health`**

```bash
curl -X GET http://localhost:8000/health
```

**Phản hồi:**

```json
{ "status": "healthy", "model": "google/gemini-3.7-flash", "tiling_grid": "512x512" }
```

---

## 3. Bảng Mã Lỗi HTTP

| Mã | Nguyên nhân | Ví dụ `message` |
|---|---|---|
| `400` | Ảnh rỗng / hỏng / sai định dạng | `"Could not decode image: ..."` |
| `413` | File vượt 25 MB | `"Uploaded file exceeds the 25 MB limit."` |
| `500` | Lỗi model / topology không xác định | `"Internal pipeline error."` |
| `502` | OpenRouter timeout hoặc mất kết nối | `"Upstream model provider timed out or is unreachable."` |

---

## 4. Hướng Dẫn Tích Hợp Frontend

### 4.1. JavaScript / React (fetch)

```javascript
async function vectorizeMaster(imageFile) {
  const formData = new FormData();
  formData.append("image", imageFile); // key bắt buộc: "image"

  const res = await fetch("http://localhost:8000/api/v1/extract/master", {
    method: "POST",
    body: formData, // KHÔNG set Content-Type thủ công, browser tự thêm boundary
  });

  const json = await res.json();
  if (!json.success) throw new Error(json.message);

  // Ảnh preview là Data URL -> bind thẳng vào <img>
  const previewUrl = json.data.preview_image_base64;

  // Bóc tách tọa độ polygon: exterior + lỗ thủng interior
  const plots = json.data.agricultural_plots.map((p) => p.geometry.exterior);
  const holes = json.data.tree_canopies
    .map((t) => t.geometry.interiors)
    .filter((h) => h.length > 0);

  return { previewUrl, plots, summary: json.data.summary };
}
```

Sử dụng trong JSX:

```jsx
const { previewUrl, plots } = await vectorizeMaster(file);
<img src={previewUrl} alt="GIS preview" />
```

### 4.2. Python (requests) — gọi API và lưu ảnh ra ổ cứng

```python
import base64
import json
import requests

API = "http://localhost:8000"

with open("anh_ve_tinh.png", "rb") as f:
    res = requests.post(f"{API}/api/v1/extract/master",
                        files={"image": f}, timeout=300)
res.raise_for_status()
body = res.json()
assert body["success"], body["errors"]

data = body["data"]
print("Summary:", data["summary"])

# Lưu ảnh preview (Data URL -> file .jpg)
header, b64_payload = data["preview_image_base64"].split(",", 1)
with open("master_preview.jpg", "wb") as f:
    f.write(base64.b64decode(b64_payload))

# Xuất tọa độ thửa ruộng ra GeoJSON đơn giản (hệ pixel, có lỗ thủng)
features = []
for p in data["agricultural_plots"]:
    ext = p["geometry"]["exterior"] + [p["geometry"]["exterior"][0]]
    rings = [ext]
    rings += [h + [h[0]] for h in p["geometry"]["interiors"]]
    features.append({
        "type": "Feature",
        "properties": {"plot_id": p["plot_id"]},
        "geometry": {"type": "Polygon", "coordinates": rings},
    })
with open("plots.geojson", "w", encoding="utf-8") as f:
    json.dump({"type": "FeatureCollection", "features": features}, f,
              ensure_ascii=False)
```

> ⚠️ **Lưu ý:** `timeout=300` nên đặt lớn vì pipeline gọi VLM nhiều lần với ảnh lớn.

---

## 5. Ghi Chú Quan Trọng

- Key form-data bắt buộc tên là **`image`**.
- Thứ tự tọa độ trả về là **`[x, y]` pixel tuyệt đối** (model nội bộ dùng `[y, x]` chuẩn hóa 0–1000, đã được pipeline chuyển đổi).
- Giới hạn dung lượng upload: **25 MB**.
- Tất cả timestamp/tham số cấu hình được ghi trong trường `metadata` khi xuất qua CLI (`main.py`).

| `503` | OpenRouter bị rate-limit (HTTP 429) | `"Upstream model provider is rate-limited. Retry later."` |

Mọi lỗi đều trả cùng phong bì chuẩn:

```json
{ "success": false, "message": "...", "data": null, "errors": ["chi tiết lỗi"] }
```

