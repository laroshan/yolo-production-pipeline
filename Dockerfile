# ==============================================================================
# Stage 1: Build & Dependency Resolution
# ==============================================================================
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-cache YOLO weights in builder layer
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# ==============================================================================
# Stage 2: Hardened Runtime Container
# ==============================================================================
FROM python:3.10-slim AS runner

WORKDIR /app

# Install minimal OS dependencies for OpenCV / headless rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Security: Create unprivileged non-root user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local
COPY --from=builder /build/yolov8n.pt /app/yolov8n.pt

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production

# Copy application source code
COPY --chown=appuser:appgroup ./app /app/app
RUN mkdir -p /app/data && chown -R appuser:appgroup /app/data

USER appuser

EXPOSE 8000

# Docker Healthcheck
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
