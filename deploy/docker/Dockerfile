# ═══════════════════════════════════════════════════════════════
# P053 — Multi-Stage Docker Build
# ═══════════════════════════════════════════════════════════════
# Stage 1: Builder — install dependencies
# Stage 2: Runtime — slim production image
#
# Why multi-stage?
#   Builder stage has pip, gcc, build tools (~1.2 GB)
#   Runtime stage has only Python + wheels (~450 MB)
#   55-65% size reduction = faster pulls, smaller attack surface
#
# Usage:
#   docker build -t memory-yield-predictor:v1 -f deploy/docker/Dockerfile .
#   docker run -p 8000:8000 memory-yield-predictor:v1
# ═══════════════════════════════════════════════════════════════

# ─── Stage 1: Builder ───
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for numpy/scipy wheel compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-serve.txt


# ─── Stage 2: Runtime ───
FROM python:3.12-slim AS runtime

LABEL maintainer="AIML Engineering Lab"
LABEL project="P053 Memory Yield Predictor"
LABEL version="1.0"

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user (security best practice)
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app

# Copy application code
COPY src/config.py src/inference.py src/serve.py src/model.py src/focal_loss.py ./
COPY src/artifacts/ ./artifacts/

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with uvicorn
CMD ["python", "-m", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
