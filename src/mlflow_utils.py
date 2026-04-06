"""
P053 — MLflow Integration Utilities
=====================================
Central MLflow helpers for experiment tracking, model registry,
and artifact logging.

Why this module?
    Real MLOps projects log EVERY training run — params, metrics per
    epoch, model artifacts, plots, and data lineage. This module
    provides consistent logging across:
    - Interactive notebook training (NB02, NB03)
    - Automated retraining (simulation Day 31+)
    - Baseline model comparison runs
    - A/B test evaluation

MLflow Concepts Used:
    - Experiment: Group of related runs (e.g., "P053-Memory-Yield-Predictor")
    - Run: Single training execution (e.g., "A100-bfloat16-v4")
    - Artifact: Files saved with a run (model weights, plots, configs)
    - Model Registry: Version control for production models (v1, v2, ...)
    - Tags: Metadata for filtering runs (gpu_name, amp_dtype, etc.)
"""

import json
from pathlib import Path
from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient

from src.config import BUSINESS, DRIFT, MLFLOW, MODEL_PARAMS, TRAINING


def init_mlflow(experiment_name: Optional[str] = None) -> str:
    """Initialize MLflow tracking. Returns experiment_id.

    Sets tracking URI and creates/gets experiment.
    Call this ONCE at the start of any training script or notebook.
    """
    tracking_uri = MLFLOW['tracking_uri']
    mlflow.set_tracking_uri(tracking_uri)

    exp_name = experiment_name or MLFLOW["experiment_name"]
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            exp_name,
            tags=MLFLOW["default_tags"],
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(exp_name)
    return experiment_id


def start_training_run(
    run_name: str,
    gpu_name: str,
    amp_dtype: str,
    batch_size: int,
    learning_rate: float,
    extra_params: Optional[dict] = None,
    extra_tags: Optional[dict] = None,
) -> mlflow.ActiveRun:
    """Start a new MLflow training run with standard params logged.

    Usage:
        with start_training_run("A100-bfloat16-v4", ...) as run:
            for epoch in range(50):
                train_epoch(...)
                log_epoch_metrics(epoch, ...)
    """
    tags = {
        **MLFLOW["default_tags"],
        "gpu_name": gpu_name,
        "amp_dtype": amp_dtype,
        "run_type": "training",
    }
    if extra_tags:
        tags.update(extra_tags)

    run = mlflow.start_run(run_name=run_name, tags=tags)

    # Log all training hyperparameters
    params = {
        # Model architecture
        "model.n_tabular": MODEL_PARAMS["n_tabular"],
        "model.n_spatial": MODEL_PARAMS["n_spatial"],
        "model.d_model": MODEL_PARAMS["d_model"],
        "model.n_heads": MODEL_PARAMS["n_heads"],
        "model.n_layers": MODEL_PARAMS["n_layers"],
        "model.cnn_out": MODEL_PARAMS["cnn_out"],
        "model.dropout": MODEL_PARAMS["dropout"],
        # Training
        "train.focal_alpha": TRAINING["focal_alpha"],
        "train.focal_gamma": TRAINING["focal_gamma"],
        "train.label_smoothing": TRAINING["label_smoothing"],
        "train.lr": learning_rate,
        "train.weight_decay": TRAINING["weight_decay"],
        "train.epochs": TRAINING["epochs"],
        "train.patience": TRAINING["patience"],
        "train.batch_size": batch_size,
        # Hardware
        "hw.gpu_name": gpu_name,
        "hw.amp_dtype": amp_dtype,
        # Drift thresholds (for reproducibility)
        "drift.psi_critical": DRIFT["psi_critical"],
        "drift.min_drifted_features": DRIFT["min_drifted_features"],
    }
    if extra_params:
        params.update(extra_params)

    mlflow.log_params(params)
    return run


def log_epoch_metrics(
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_auc_pr: float,
    val_auc_pr: float,
    learning_rate: float,
    epoch_time_s: float,
    throughput: float,
    extra_metrics: Optional[dict] = None,
) -> None:
    """Log metrics for a single training epoch.

    Uses step=epoch so MLflow plots training curves automatically.
    """
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_auc_pr": train_auc_pr,
        "val_auc_pr": val_auc_pr,
        "learning_rate": learning_rate,
        "epoch_time_s": epoch_time_s,
        "throughput_samples_per_s": throughput,
    }
    if extra_metrics:
        metrics.update(extra_metrics)

    mlflow.log_metrics(metrics, step=epoch)


def log_evaluation_results(
    results: dict[str, dict],
    threshold: float,
    prefix: str = "",
) -> None:
    """Log final evaluation metrics for val/test/unseen splits.

    Args:
        results: {"val": {"f1": 0.12, "auc_pr": 0.05, ...}, "test": {...}, ...}
        threshold: Classification threshold used
        prefix: Optional prefix for metric names (e.g., "retrain." for v2 model)
    """
    for split_name, metrics in results.items():
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                key = f"{prefix}{split_name}.{metric_name}"
                mlflow.log_metric(key, value)

    mlflow.log_metric(f"{prefix}threshold", threshold)

    # Log business impact
    test_recall = results.get("test", {}).get("recall", 0)
    annual_savings = (
        test_recall
        * BUSINESS["wafers_per_month"]
        * BUSINESS["cost_per_wafer_usd"]
        * 0.006  # defect rate
        * 12
    )
    mlflow.log_metric(f"{prefix}business.annual_savings_usd", annual_savings)


def log_model_artifact(
    model_path: Path,
    artifact_subdir: str = "model",
) -> None:
    """Log a trained model file as an MLflow artifact."""
    mlflow.log_artifact(str(model_path), artifact_subdir)


def log_plot_artifact(
    plot_path: Path,
    artifact_subdir: str = "plots",
) -> None:
    """Log a plot PNG as an MLflow artifact."""
    mlflow.log_artifact(str(plot_path), artifact_subdir)


def log_training_summary(
    best_epoch: int,
    best_val_auc: float,
    total_time_min: float,
    avg_epoch_time_s: float,
    throughput: float,
    peak_vram_gb: float,
    epochs_run: int,
    train_rows: int,
) -> None:
    """Log final training summary metrics."""
    mlflow.log_metrics({
        "best_epoch": best_epoch,
        "best_val_auc_pr": best_val_auc,
        "total_train_time_min": total_time_min,
        "avg_epoch_time_s": avg_epoch_time_s,
        "peak_throughput_samples_per_s": throughput,
        "peak_vram_gb": peak_vram_gb,
        "epochs_run": epochs_run,
        "train_rows": train_rows,
    })


def register_model(
    run_id: str,
    version_description: str,
    alias: str = "",
    artifact_path: str = "model",
) -> Any:
    """Register a model version in MLflow Model Registry.

    Uses MLflow 3.x alias system (replaces deprecated stages).
    Uses create_model_version (works with log_artifact, not just log_model).

    Args:
        run_id: MLflow run ID containing the model artifact
        version_description: Human-readable version description
        alias: Alias like "champion", "challenger", "baseline"
        artifact_path: Artifact subdirectory name (default: "model")

    Returns:
        ModelVersion object
    """
    model_name = MLFLOW["registered_model_name"]
    client = MlflowClient()

    # Ensure registered model exists
    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        pass  # Already exists

    model_version = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/{artifact_path}",
        run_id=run_id,
        description=version_description,
    )

    if alias:
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=model_version.version,
        )

    return model_version


def retrolog_completed_run(
    run_name: str,
    benchmark_json_path: Path,
    model_path: Optional[Path] = None,
    plot_paths: Optional[list[Path]] = None,
    extra_tags: Optional[dict] = None,
) -> str:
    """Retro-log a completed training run from its benchmark JSON.

    Used to import existing T4/A100 results into MLflow tracking.

    Returns:
        run_id of the created MLflow run
    """
    with open(benchmark_json_path) as f:
        benchmark = json.load(f)

    gpu_name = benchmark.get("gpu_name", "unknown")
    amp_dtype = "bfloat16" if "bfloat16" in str(benchmark.get("amp_enabled", "")) else "float16"

    # Detect dtype from context
    if "A100" in gpu_name or "H100" in gpu_name:
        amp_dtype = "bfloat16"
    else:
        amp_dtype = "float16"

    tags = {
        **MLFLOW["default_tags"],
        "gpu_name": gpu_name,
        "amp_dtype": amp_dtype,
        "run_type": "retrolog",
    }
    if extra_tags:
        tags.update(extra_tags)

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        # Log params
        mlflow.log_params({
            "hw.gpu_name": gpu_name,
            "hw.gpu_vram_gb": benchmark.get("gpu_vram_gb", 0),
            "hw.amp_dtype": amp_dtype,
            "train.batch_size": benchmark.get("batch_size", 0),
            "train.epochs": benchmark.get("epochs_run", 0),
            "model.params": benchmark.get("model_params", 317633),
            "train.rows": benchmark.get("train_rows", 0),
        })

        # Log per-epoch metrics from history
        history = benchmark.get("history", {})
        if history:
            n_epochs = len(history.get("train_loss", []))
            epoch_times = benchmark.get("epoch_times_s", [])
            train_rows = benchmark.get("train_rows", 10_000_000)
            for ep in range(n_epochs):
                step_metrics = {}
                for key in ["train_loss", "val_loss", "train_auc_pr", "val_auc_pr"]:
                    if key in history and ep < len(history[key]):
                        step_metrics[key] = history[key][ep]
                if epoch_times and ep < len(epoch_times):
                    step_metrics["epoch_time_s"] = epoch_times[ep]
                    step_metrics["throughput_samples_per_s"] = train_rows / epoch_times[ep]
                if step_metrics:
                    mlflow.log_metrics(step_metrics, step=ep + 1)

        # Log final summary
        log_training_summary(
            best_epoch=benchmark.get("best_epoch", 0),
            best_val_auc=benchmark.get("results", {}).get("val", {}).get("auc_pr", 0),
            total_time_min=benchmark.get("total_train_time_min", 0),
            avg_epoch_time_s=benchmark.get("avg_epoch_time_s", 0),
            throughput=benchmark.get("throughput_samples_per_s", 0),
            peak_vram_gb=benchmark.get("peak_gpu_memory_gb", 0),
            epochs_run=benchmark.get("epochs_run", 0),
            train_rows=benchmark.get("train_rows", 0),
        )

        # Log evaluation results
        results = benchmark.get("results", {})
        if results:
            threshold = results.get("val", {}).get("threshold", 0.5)
            log_evaluation_results(results, threshold)

        # Log artifacts
        if model_path and model_path.exists():
            log_model_artifact(model_path)

        if plot_paths:
            for plot_path in plot_paths:
                if plot_path.exists():
                    log_plot_artifact(plot_path)

        # Log the benchmark JSON itself
        mlflow.log_artifact(str(benchmark_json_path), "benchmark")

        return run.info.run_id
