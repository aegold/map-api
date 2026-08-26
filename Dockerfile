# ---------------------------------------------------------------------------
# Production container - Satellite GIS Vectorization API
# Lightweight python:3.11-slim image with the OpenCV system libraries.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# OpenCV system libraries (libgl1-mesa-glx on buster/bullseye, libgl1 on
# bookworm and later; libglib2.0-0 is always required).
RUN apt-get update \
    && (apt-get install -y --no-install-recommends libgl1-mesa-glx \
        || apt-get install -y --no-install-recommends libgl1) \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
