"""
P053 — Central Configuration for Memory Yield Predictor
========================================================
Single source of truth for all paths, feature lists, model params,
drift thresholds, and serving configuration.

Why a central config?
    Production ML systems have 50+ configurable parameters scattered
    across training, serving, monitoring, and retraining code.
    A single config module prevents silent mismatches (e.g., training
    with 33 tabular features but serving expecting 36).
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════

# Handle both local dev (project_root/src/config.py) and Docker flat layout (/app/config.py)
_here = Path(__file__).resolve().parent
ROOT = _here.parent if (_here.parent / "src").is_dir() else _here
SRC_DIR = ROOT / "src" if (ROOT / "src").is_dir() else ROOT
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
ARTIFACTS_DIR = SRC_DIR / "artifacts"
MODELS_DIR = ROOT / "models"
DEPLOY_DIR = ROOT / "deploy"

# ═══════════════════════════════════════════════════════════════
# Feature Definitions (must match preprocess.py exactly)
# ═══════════════════════════════════════════════════════════════

NUMERIC_FEATURES = [
    "test_temp_c", "cell_leakage_fa", "retention_time_ms",
    "row_hammer_threshold", "disturb_margin_mv", "adjacent_row_activations",
    "rh_susceptibility", "bit_error_rate", "correctable_errors_per_1m",
    "ecc_syndrome_entropy", "uncorrectable_in_extended",
    "trcd_ns", "trp_ns", "tras_ns", "rw_latency_ns",
    "idd4_active_ma", "idd2p_standby_ma", "idd5_refresh_ma",
    "gate_oxide_thickness_a", "channel_length_nm", "vt_shift_mv",
    "block_erase_count",
]

CATEGORICAL_FEATURES = ["tester_id", "probe_card_id", "chamber_id", "recipe_version"]

SPATIAL_FEATURES = ["die_x", "die_y", "edge_distance"]

ENGINEERED_FEATURES = [
    "retention_temp_interaction", "leakage_retention_ratio", "edge_risk",
    "power_ratio", "ecc_burden", "timing_margin", "rh_risk_composite",
]

LOG_FEATURES = [
    "cell_leakage_fa", "retention_time_ms", "rh_susceptibility",
    "rw_latency_ns", "ecc_syndrome_entropy",
]

# Order must match preprocessing output (tabular first, spatial last)
ALL_FEATURE_NAMES = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES + SPATIAL_FEATURES
)
N_TABULAR = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) + len(ENGINEERED_FEATURES)
N_SPATIAL = len(SPATIAL_FEATURES)

TARGET = "is_fail"

# ═══════════════════════════════════════════════════════════════
# Model Architecture
# ═══════════════════════════════════════════════════════════════

MODEL_PARAMS = {
    "n_tabular": N_TABULAR,
    "n_spatial": N_SPATIAL,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 2,
    "cnn_out": 64,
    "dropout": 0.2,
}

# ═══════════════════════════════════════════════════════════════
# Training Hyperparameters
# ═══════════════════════════════════════════════════════════════

TRAINING = {
    "focal_alpha": 0.75,
    "focal_gamma": 2.0,
    "label_smoothing": 0.01,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 50,
    "patience": 12,
    "scheduler_t0": 10,
    "scheduler_tmult": 2,
}

# ═══════════════════════════════════════════════════════════════
# Serving Configuration
# ═══════════════════════════════════════════════════════════════

SERVING = {
    "host": "0.0.0.0",
    "port": 8000,
    "model_version": "v1",
    "default_threshold": 0.5,
    "max_batch_size": 1024,
    "request_timeout_s": 30,
}

# ═══════════════════════════════════════════════════════════════
# Drift Detection Thresholds
# ═══════════════════════════════════════════════════════════════

DRIFT = {
    # PSI (Population Stability Index) thresholds
    "psi_warning": 0.1,       # Minor shift — log warning
    "psi_critical": 0.2,      # Major shift — trigger retrain
    # KL divergence threshold per feature
    "kl_warning": 0.05,
    "kl_critical": 0.1,
    # Minimum features drifted to trigger retraining
    "min_drifted_features": 3,
    # Performance degradation threshold (AUC-PR drop)
    "aucpr_drop_threshold": 0.05,
    # Monitoring window
    "reference_window_days": 30,
    "analysis_window_days": 7,
}

# ═══════════════════════════════════════════════════════════════
# SageMaker Configuration
# ═══════════════════════════════════════════════════════════════

SAGEMAKER = {
    "region": "us-west-2",
    "instance_type_training": "ml.g5.2xlarge",
    "instance_type_inference": "ml.g4dn.xlarge",
    "model_package_group": "memory-yield-predictor",
    "endpoint_name": "memory-yield-predictor-prod",
    "s3_bucket": "aiml-memory-yield-predictor",
    "s3_prefix": "production",
}

# ═══════════════════════════════════════════════════════════════
# Business Impact Constants
# ═══════════════════════════════════════════════════════════════

BUSINESS = {
    "wafers_per_month": 50_000,
    "cost_per_wafer_usd": 1_200,
    "base_yield_pct": 99.38,
    "cost_per_false_negative_usd": 45_000,  # Missed defective die reaches customer
    "cost_per_false_positive_usd": 120,     # Unnecessary retest
}

# ═══════════════════════════════════════════════════════════════
# Simulation Timeline
# ═══════════════════════════════════════════════════════════════

SIMULATION = {
    "start_date": "2026-02-20",  # Day 1 of 40-day production simulation
    "total_days": 40,
    "rows_per_day": 5_000_000,   # Full mode
}

# ═══════════════════════════════════════════════════════════════
# MLflow Configuration
# ═══════════════════════════════════════════════════════════════
# Three deployment targets — controlled by environment variables:
#   LOCAL:  sqlite:///mlflow.db  (dev laptop, zero dependencies)
#   DOCKER: postgresql://mlflow:mlflow@postgres:5432/mlflow
#   AWS:    postgresql://mlflow:<secret>@mlflow-rds.xxx.rds.amazonaws.com:5432/mlflow
#
# Override with:  export MLFLOW_TRACKING_URI="postgresql://..."
#                 export MLFLOW_S3_ENDPOINT_URL="http://localstack:4566"
#                 export MLFLOW_ARTIFACT_ROOT="s3://p053-mlflow-artifacts/"

import os as _os

_default_tracking_uri = f"sqlite:///{ROOT / 'mlflow.db'}"

MLFLOW = {
    "experiment_name": "P053-Memory-Yield-Predictor",
    "tracking_uri": _os.environ.get("MLFLOW_TRACKING_URI", _default_tracking_uri),
    "artifact_root": _os.environ.get("MLFLOW_ARTIFACT_ROOT", str(ROOT / "mlartifacts")),
    "registered_model_name": "HybridTransformerCNN",
    # Tags applied to every run
    "default_tags": {
        "project": "P053",
        "domain": "semiconductor",
        "model_arch": "HybridTransformerCNN",
        "dataset": "DRAM-STFD-16M",
    },
}
