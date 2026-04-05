"""
P053 — FastAPI Production Serving Endpoint
============================================
Real-time inference API with Prometheus metrics, structured logging,
health checks, and request validation.

Endpoints:
    POST /predict          — Single wafer die prediction
    POST /predict/batch    — Batch prediction (up to 1024 dies)
    GET  /health           — Liveness probe (always 200)
    GET  /health/ready     — Readiness probe (model loaded?)
    GET  /metrics          — Prometheus metrics (scrape target)
    GET  /model/info       — Model metadata + version

Production features:
    - Prometheus counters: request count, latency histogram, prediction distribution
    - Structured JSON logging for ELK/CloudWatch
    - Input validation with Pydantic models
    - Graceful error handling (no stack traces to clients)
    - CORS middleware for dashboard integration

Interview talking points:
    "Our serving layer exposes Prometheus metrics that Grafana scrapes every 15s.
    We track p50/p95/p99 latency, prediction distribution (fail ratio should
    match production baseline ~0.6%), and throughput. If fail ratio spikes above
    1.5%, the Grafana alert fires and pages the on-call engineer."

Usage:
    uvicorn serve:app --host 0.0.0.0 --port 8000
    # or
    python serve.py
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel, Field, field_validator

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from config import SERVING, NUMERIC_FEATURES, CATEGORICAL_FEATURES, SPATIAL_FEATURES

# ═══════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
logger = logging.getLogger("p053.serve")

# ═══════════════════════════════════════════════════════════════
# Prometheus Metrics
# ═══════════════════════════════════════════════════════════════

REQUEST_COUNT = Counter(
    "p053_request_total", "Total prediction requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "p053_request_latency_seconds", "Request latency",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
PREDICTION_DISTRIBUTION = Counter(
    "p053_prediction_total", "Prediction outcomes",
    ["label"],
)
BATCH_SIZE_HIST = Histogram(
    "p053_batch_size", "Batch sizes of prediction requests",
    buckets=[1, 5, 10, 50, 100, 500, 1024],
)
MODEL_LOADED = Gauge(
    "p053_model_loaded", "Whether the model is loaded and ready",
)

# ═══════════════════════════════════════════════════════════════
# Pydantic Models (Request/Response Validation)
# ═══════════════════════════════════════════════════════════════

class WaferDieInput(BaseModel):
    """Single wafer die measurement for prediction."""
    test_temp_c: float = Field(..., ge=-50, le=200, description="Test temperature (°C)")
    cell_leakage_fa: float = Field(..., ge=0, description="Cell leakage current (fA)")
    retention_time_ms: float = Field(..., ge=0, description="Data retention time (ms)")
    row_hammer_threshold: float = Field(0.0, description="Row hammer activation threshold")
    disturb_margin_mv: float = Field(0.0, description="Disturb margin (mV)")
    adjacent_row_activations: float = Field(0.0, ge=0, description="Adjacent row activation count")
    rh_susceptibility: float = Field(0.0, ge=0, description="Row hammer susceptibility score")
    bit_error_rate: float = Field(0.0, ge=0, description="Raw bit error rate")
    correctable_errors_per_1m: float = Field(0.0, ge=0, description="Correctable errors per 1M bits")
    ecc_syndrome_entropy: float = Field(0.0, ge=0, description="ECC syndrome entropy")
    uncorrectable_in_extended: float = Field(0.0, ge=0, description="Uncorrectable errors in extended test")
    trcd_ns: float = Field(0.0, description="tRCD timing (ns)")
    trp_ns: float = Field(0.0, description="tRP timing (ns)")
    tras_ns: float = Field(0.0, description="tRAS timing (ns)")
    rw_latency_ns: float = Field(0.0, description="Read/write latency (ns)")
    idd4_active_ma: float = Field(0.0, ge=0, description="Active power (mA)")
    idd2p_standby_ma: float = Field(0.0, ge=0, description="Standby power (mA)")
    idd5_refresh_ma: float = Field(0.0, ge=0, description="Refresh power (mA)")
    gate_oxide_thickness_a: float = Field(0.0, ge=0, description="Gate oxide thickness (Å)")
    channel_length_nm: float = Field(0.0, ge=0, description="Channel length (nm)")
    vt_shift_mv: float = Field(0.0, description="Threshold voltage shift (mV)")
    block_erase_count: float = Field(0.0, ge=0, description="Block erase count")
    tester_id: str = Field("T001", description="Tester equipment ID")
    probe_card_id: str = Field("PC001", description="Probe card ID")
    chamber_id: str = Field("CH001", description="Chamber ID")
    recipe_version: str = Field("R1.0", description="Test recipe version")
    die_x: float = Field(0.0, description="Die X position on wafer")
    die_y: float = Field(0.0, description="Die Y position on wafer")
    edge_distance: float = Field(0.0, ge=0, description="Distance from wafer edge")

    def to_dict(self) -> dict:
        return self.model_dump()


class BatchInput(BaseModel):
    """Batch of wafer die measurements."""
    dies: list[WaferDieInput] = Field(..., min_length=1, max_length=1024)

    @field_validator("dies")
    @classmethod
    def validate_batch_size(cls, v):
        if len(v) > SERVING["max_batch_size"]:
            raise ValueError(f"Batch size {len(v)} exceeds max {SERVING['max_batch_size']}")
        return v


class PredictionResponse(BaseModel):
    """Single prediction result."""
    probability: float
    prediction: int
    label: str
    threshold: float
    model_version: str


class BatchResponse(BaseModel):
    """Batch prediction results."""
    predictions: list[PredictionResponse]
    n_total: int
    n_fail: int
    fail_rate: float
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_version: str
    architecture: str
    n_parameters: int
    n_tabular_features: int
    n_spatial_features: int
    threshold: float
    device: str


# ═══════════════════════════════════════════════════════════════
# Application
# ═══════════════════════════════════════════════════════════════

predictor = None
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global predictor
    logger.info("Loading model...")
    try:
        from inference import YieldPredictor
        predictor = YieldPredictor(
            threshold=SERVING["default_threshold"],
        )
        MODEL_LOADED.set(1)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        MODEL_LOADED.set(0)
    yield
    logger.info("Shutting down...")
    MODEL_LOADED.set(0)


app = FastAPI(
    title="P053 Memory Yield Predictor",
    description="Real-time DRAM yield prediction API — HybridTransformerCNN",
    version=SERVING["model_version"],
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(die: WaferDieInput):
    """Predict pass/fail for a single wafer die."""
    if predictor is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.time()
    try:
        result = predictor.predict_raw(die.to_dict())
        latency = time.time() - t0

        label = result["label"]
        PREDICTION_DISTRIBUTION.labels(label=label).inc()
        REQUEST_COUNT.labels(endpoint="/predict", status="ok").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)

        return PredictionResponse(
            probability=round(result["probability"], 6),
            prediction=result["prediction"],
            label=label,
            threshold=result["threshold"],
            model_version=SERVING["model_version"],
        )
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        logger.error("Prediction error: %s", e)
        raise HTTPException(500, "Prediction failed")


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(batch: BatchInput):
    """Predict pass/fail for a batch of wafer dies."""
    if predictor is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.time()
    try:
        raw_batch = [die.to_dict() for die in batch.dies]
        results = predictor.predict_batch_raw(raw_batch)
        latency = time.time() - t0

        n_fail = sum(1 for r in results if r["prediction"] == 1)
        n_total = len(results)

        BATCH_SIZE_HIST.observe(n_total)
        PREDICTION_DISTRIBUTION.labels(label="FAIL").inc(n_fail)
        PREDICTION_DISTRIBUTION.labels(label="PASS").inc(n_total - n_fail)
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="ok").inc()
        REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(latency)

        return BatchResponse(
            predictions=[
                PredictionResponse(
                    probability=round(r["probability"], 6),
                    prediction=r["prediction"],
                    label=r["label"],
                    threshold=predictor.threshold,
                    model_version=SERVING["model_version"],
                )
                for r in results
            ],
            n_total=n_total,
            n_fail=n_fail,
            fail_rate=round(n_fail / n_total, 6) if n_total else 0.0,
            latency_ms=round(latency * 1000, 2),
        )
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="error").inc()
        logger.error("Batch prediction error: %s", e)
        raise HTTPException(500, "Batch prediction failed")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness probe — always returns 200."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        model_version=SERVING["model_version"],
        uptime_seconds=round(time.time() - start_time, 1),
    )


@app.get("/health/ready")
async def readiness():
    """Readiness probe — returns 200 only if model is loaded."""
    if predictor is None:
        raise HTTPException(503, "Model not ready")
    return {"status": "ready", "model_version": SERVING["model_version"]}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Model metadata and configuration."""
    from config import MODEL_PARAMS, N_TABULAR, N_SPATIAL
    n_params = sum(p.numel() for p in predictor.model.parameters()) if predictor else 0
    return ModelInfoResponse(
        model_version=SERVING["model_version"],
        architecture="HybridTransformerCNN",
        n_parameters=n_params,
        n_tabular_features=N_TABULAR,
        n_spatial_features=N_SPATIAL,
        threshold=predictor.threshold if predictor else SERVING["default_threshold"],
        device=str(predictor.device) if predictor else "none",
    )


# ═══════════════════════════════════════════════════════════════
# Error Handlers
# ═══════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "serve:app",
        host=SERVING["host"],
        port=SERVING["port"],
        log_level="info",
        access_log=True,
    )
