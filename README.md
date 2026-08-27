# 🛰️ Satellite GIS Vectorization Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)](https://fastapi.tiangolo.com/)
[![Model](https://img.shields.io/badge/Model-gemini--3.7--flash-orange)](https://openrouter.ai/)

> **Ngôn ngữ tài liệu:** Tiếng Việt · Mã lệnh & CLI giữ nguyên chuẩn English/UNIX.

---

## 1. Giới Thiệu Tổng Quan (Project Overview)

**Sứ mệnh dự án:** Tự động hóa việc biên soạn bản đồ địa chính (_Cadastral Vectorization_) từ ảnh vệ tinh / ảnh hàng không độ phân giải cao thành các đối tượng vector GIS có topology sạch, bao gồm:

- **Đường giao thông (Centerline `LineString`):** trục tim đường, đê kênh, luống ruộng.
- **Thửa đất nông nghiệp (Cadastral Farm Plots):** được tách tự động theo hành lang đường.
- **Tán cây (Tree Canopies):** polygon lõm bám sát tán lá, có đục rỗng vùng nước.
- **Mặt nước (Aquaculture / Water Bodies):** ao nuôi, hồ, kênh mương.

### Công nghệ cốt lõi (Core Tech Stack)

| Thành phần            | Công nghệ                                                         |
| --------------------- | ----------------------------------------------------------------- |
| Vision Language Model | `google/gemini-3.7-flash` qua **OpenRouter** (SDK `openai`)       |
| API Framework         | **FastAPI** + Uvicorn (async, non-blocking)                       |
| Data Contracts        | **Pydantic v2** (`contracts.py`, `schemas.py`)                    |
| Spatial Engine        | **Shapely** — unary union, difference, buffer, LineString/Polygon |
| Xử lý ảnh             | **OpenCV**, **Pillow**, **NumPy**                                 |
| Render preview        | **Matplotlib** (backend headless `Agg`) + mã hóa Base64 Data URL  |
| Làm mượt biên         | **Thuật toán Chaikin corner-cutting** (tỉ lệ 0.85/0.15)           |

## 3. Cấu Trúc Thư Mục Dự Án (Project Layout)

```text
gis_pipeline/
├── core/
│   ├── input_engine.py       # Stage 1: Load ảnh, mã hóa Base64 & chuyển đổi tọa độ
│   ├── path_extractor.py     # Stage 2: Trích xuất mạng lưới đường toàn cảnh
│   ├── tile_segmentor.py     # Stage 3: Phân đoạn ngữ nghĩa theo lưới Tiling 512x512
│   └── spatial_engine.py     # Stage 4: Xử lý Topo, đục rỗng ao, cắt thửa & Chaikin
├── utils/
│   └── visualizer.py         # Render ảnh đồ họa & mã hóa Base64 Data URL
├── api/
│   └── server.py             # Stage 5: FastAPI REST Microservice
├── config.py                 # Cấu hình biến môi trường & tham số pipeline
├── contracts.py              # Single source of truth cho API Schemas & DTOs
├── schemas.py                # Pydantic Schemas cho VLM Structured Outputs
├── main.py                   # Script CLI chạy kiểm thử offline cục bộ
├── requirements.txt          # Danh sách thư viện phụ thuộc
├── Dockerfile                # Cấu hình đóng gói Docker container
├── docker-compose.yml        # Cấu hình chạy multi-container / production
├── .env.example              # Mẫu tệp cấu hình biến môi trường
└── README.md                 # Tài liệu hướng dẫn sử dụng
```

> Các tệp `_validate_stage1..5.py` (tùy chọn) là bộ kiểm thử hồi quy offline — có thể xóa trước khi triển khai.

---

## 4. Cài Đặt Môi Trường (Prerequisites & Installation)

### Yêu cầu hệ thống

- **Python 3.10+** (khuyến nghị 3.11)
- pip ≥ 23
- Kết nối Internet tới `openrouter.ai`

### Bước 1 — Tạo Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS (bash):**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Bước 2 — Cài đặt thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

### Bước 3 — Cấu hình biến môi trường

Sao chép mẫu cấu hình và điền khóa API:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

Mở `.env` và khai báo ít nhất `OPENROUTER_API_KEY`:

```env
# OpenRouter API Credentials
OPENROUTER_API_KEY=sk-or-v1-your-real-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=google/gemini-3.7-flash

# Tiling Grid & Geometry Parameters (giá trị mặc định khuyến nghị)
TILE_SIZE=512
TILE_OVERLAP=80
STRIDE=432
ROAD_BUFFER_PX=3.0

# Area Filter Thresholds (Pixel Area)
MIN_WATER_AREA=80
MIN_TREE_AREA=120
MIN_AGRI_AREA=400

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=False
```

> 💡 **Ghi chú:** biến môi trường hệ thống luôn có độ ưu tiên cao hơn giá trị trong `.env`.

---

## 5. Hướng Dẫn Vận Hành (Operation Guide)

### A. Chạy Kiểm Thử Offline CLI

Chạy toàn bộ pipeline Stages 1 → 4 trên một ảnh vệ tinh:

```bash
python main.py --image <duong_dan_anh_ve_tinh>
```

Tùy chọn bổ sung:

```bash
python main.py --image anh.jpg --output-dir ./output --model google/gemini-3.7-flash --skip-previews
```

**Sản phẩm đầu ra** (thư mục mặc định `./output/`):

| Tệp                           | Mô tả                                                |
| ----------------------------- | ---------------------------------------------------- |
| `master_gis_dataset.json`     | Dataset GIS tổng hợp đầy đủ (đường, thửa, cây, nước) |
| `preview_stage2_roads.png`    | Preview mạng lưới đường                              |
| `preview_stage3_semantic.png` | Preview phân đoạn ngữ nghĩa theo lưới tile           |
| `master_overlay.png`          | Ảnh so sánh song song: ảnh gốc ↔ Master GIS Overlay  |

### B. Khởi Động Server FastAPI Cục Bộ

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

- **Tài liệu API tương tác (Swagger UI):** <http://localhost:8000/docs>
- **Health check:** <http://localhost:8000/health>

### C. Triển Khai Production Bằng Docker / Docker Compose

Build và khởi động container ở nền:

```bash
docker compose up --build -d
```

Kiểm tra trạng thái và xem log:

```bash
docker compose ps
docker compose logs -f
```

Dừng dịch vụ:

```bash
docker compose down
```

---

## 6. Đặc Tả REST API & Ví Dụ cURL (API Reference)

Base URL mặc định: `http://localhost:8000`

### 6.1. `POST /api/v1/extract/paths`

**Chỉ trích xuất mạng lưới đường** (Stage 1 → Stage 2).
Trả về `APIResponse[PathsExtractionPayload]`: tọa độ trục tim đường + ảnh preview Base64.

```bash
curl -X POST "http://localhost:8000/api/v1/extract/paths" \
  -H "accept: application/json" \
  -F "image=@anh_ve_tinh.jpg"
```

### 6.2. `POST /api/v1/extract/geo`

**Chỉ trích xuất địa vật diện tích** (Stage 1 → Stage 3 → Topo cleanup: union, đục rỗng nước, Chaikin smoothing, lọc diện tích — **không** trích đường và **không** cắt thửa).
Trả về `APIResponse[GeoExtractionPayload]`: polygon đã làm sạch/làm mượt của 3 lớp + preview Base64 khớp 100% với JSON.

```bash
curl -X POST "http://localhost:8000/api/v1/extract/geo" \
  -H "accept: application/json" \
  -F "image=@anh_ve_tinh.jpg"
```

### 6.3. `POST /api/v1/extract/master`

**Pipeline tổng hợp đầy đủ** (Stages 1 → 4): union, đục rỗng, cắt thửa theo hành lang đường, Chaikin smoothing.
Trả về `APIResponse[MasterGISPayload]`: dataset Master GIS + ảnh so sánh song song Dual-Panel Base64.

```bash
curl -X POST "http://localhost:8000/api/v1/extract/master" \
  -H "accept: application/json" \
  -F "image=@anh_ve_tinh.jpg"
```

### 6.4. `GET /health`

Health check trạng thái dịch vụ:

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model": "google/gemini-3.7-flash",
  "tiling_grid": "512x512"
}
```

### Mã lỗi HTTP chuẩn hóa

| Mã    | Nguyên nhân                             | Hành vi                                 |
| ----- | --------------------------------------- | --------------------------------------- |
| `400` | Ảnh hỏng / rỗng / sai định dạng         | Envelope `success=false` kèm `errors[]` |
| `413` | Ảnh vượt giới hạn 25 MB                 | Tương tự trên                           |
| `500` | Lỗi inference model hoặc topology       | Log đầy đủ server-side                  |
| `502` | OpenRouter timeout / mất kết nối        | Thông báo upstream unreachable          |
| `503` | Bị rate-limit bởi OpenRouter (HTTP 429) | Thông báo retry sau                     |

---

## 7. Định Dạng Dữ Liệu Đầu Ra (Unified JSON Contract)

Mọi phản hồi đều bọc trong phong bì chuẩn `APIResponse[T]` (`contracts.py`):

```jsonc
{
  "success": true,
  "message": "Extraction completed.",
  "errors": [],
  "data": {
    "summary": {
      "transportation_network_count": 1,
      "agricultural_plots_count": 2,
      "tree_canopies_count": 1,
      "water_bodies_count": 1,
    },
    "transportation_network": [
      {
        "path_id": 0,
        "name": "main road",
        // Chuỗi điểm [x, y] pixel tuyệt đối
        "coordinates_pixel": [
          [300, 0],
          [300, 200],
          [300, 400],
        ],
      },
    ],
    "agricultural_plots": [
      {
        "plot_id": 1,
        // Vòng ngoài + lỗ thủng [x, y] pixel (đã Chaikin-smoothed, bảo toàn interiors)
        "geometry": {
          "exterior": [[100, 100], [140, 102], [180, 110], [100, 100]],
          "interiors": []
        },
      },
    ],
    "tree_canopies": [
      /* PolygonFeature ... */
    ],
    "water_bodies": [
      /* PolygonFeature ... */
    ],
    "preview_image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  },
}
```

### Quy ước hệ tọa độ

- Model VLM trả tọa độ chuẩn hóa **`[y, x]`** trên thang **`0–1000`**.
- Pipeline chuyển về pixel tuyệt đối **`[x, y]`** trước khi trả ra API:

```text
gx = int(round(tx + (pt_x_norm / 1000.0) * tile_width))   # Stage 3 (tile cục bộ)
gy = int(round(ty + (pt_y_norm / 1000.0) * tile_height))

px = int(round((coord_x_norm / 1000.0) * image_width))    # Stage 2 (toàn cảnh)
py = int(round((coord_y_norm / 1000.0) * image_height))
```

> ⚠️ **Lưu ý quan trọng:** thứ tự trục đầu ra API luôn là `[x, y]` (trục X trước), khác với thứ tự `[y, x]` mà model sinh ra.

---

## 8. Xử Lý Sự Cố Thường Gặp (Troubleshooting)

| Hiện tượng                                    | Nguyên nhân                                          | Cách xử lý                                                                                                                  |
| --------------------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `RuntimeError: Missing OpenRouter API key`    | Chưa khai báo khóa trong `.env` hoặc biến môi trường | Điền `OPENROUTER_API_KEY` hợp lệ vào `.env` rồi khởi động lại                                                               |
| HTTP `503 rate-limited`                       | OpenRouter trả mã 429 (hết hạn mức)                  | Chờ theo hướng dẫn header `Retry-After`, kiểm tra hạn tài khoản OpenRouter                                                  |
| HTTP `502 timeout`                            | Mạng chậm / OpenRouter quá tải                       | Kiểm tra kết nối; thử lại; xem log chi tiết trong container                                                                 |
| Ảnh preview không hiển thị                    | Frontend chưa hỗ trợ Data URL                        | Trích phần sau dấu `,` của chuỗi `data:image/jpeg;base64,...` khi decode                                                    |
| `ModuleNotFoundError: No module named 'core'` | Chạy lệnh không đúng từ thư mục gốc dự án            | `cd` về thư mục chứa `main.py` / `api/` rồi chạy lại                                                                        |
| Matplotlib crash trên Linux không GUI         | Thiếu backend headless                               | Đã cấu hình sẵn `Agg`; đảm bảo không ghi đè biến `MPLBACKEND`                                                               |
| Container lỗi import OpenCV                   | Thiếu thư viện hệ thống GL                           | Đã cài sẵn trong Dockerfile (`libgl1`/`libgl1-mesa-glx`, `libglib2.0-0`) — build lại bằng `docker compose build --no-cache` |

---

## 9. Giấy Phép (License)

Dự án phục vụ mục đích nội bộ nghiên cứu & triển khai địa chính — vui lòng liên hệ đội phát triển để biết thêm chi tiết cấp phép.
