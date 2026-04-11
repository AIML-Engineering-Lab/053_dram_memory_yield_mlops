"""
P053 — 40-Day Production Simulation Runner (Standalone)
========================================================
Runs the full 40-day simulation WITHOUT Docker/Airflow.
This is the "just works on my laptop" version for testing and demos.

What it does:
    1. Generate 5M rows per day (or smaller for --fast mode)
    2. Run Spark ETL on each day
    3. Run drift detection against reference window (days 1-8)
    4. Check retrain criteria (drift + performance + staleness)
    5. Simulate retrain and canary deploy when triggered
    6. Save timeline JSON for dashboard plotting

Modes:
    --fast:  100K rows/day (quick sanity check, ~10 min total)
    --medium: 1M rows/day (demo mode, ~30 min total)
    --full:  5M rows/day (production scale, ~2 hrs total)

Usage:
    python -m src.run_simulation --fast          # Quick test
    python -m src.run_simulation --medium        # Demo
    python -m src.run_simulation --full          # Full scale
    python -m src.run_simulation --day 11 --end-day 20  # Partial range
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, MLFLOW, SIMULATION
from src.mlflow_utils import init_mlflow
from src.simulation_logger import SimulationDayLogger
from src.streaming_data_generator import PRODUCTION_DIR, generate_day, get_drift_config

TIMELINE_PATH = DATA_DIR / "simulation_timeline.json"
DRIFT_REPORT_DIR = DATA_DIR / "drift_reports"


def run_simulation(start_day: int = 1, end_day: int = 40,
                   rows_per_day: int = 5_000_000,
                   skip_spark: bool = False,
                   skip_kafka: bool = True,
                   backend: str = "auto",
                   checkpoint: bool = False,
                   sim_retrain_epochs: int = 10) -> dict:
    """
    Execute the full 40-day production simulation.

    Args:
        backend: "auto", "aws", "kaggle", "colab", or "local" — controls WHERE training runs
        checkpoint: If True, save progress after each day and resume from last day
        sim_retrain_epochs: Epochs per simulation retrain event.
            10 = ~30-40min on T4 (recommended for demo runs).
            0  = metadata-only, no actual training (fastest, good for --fast/--medium).
            50 = full production quality (very slow, ~8-10hr per event on T4).

    Returns a timeline dict with per-day events for dashboard plotting.
    """
    t0 = time.time()
    DRIFT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Checkpoint Resume ──
    CHECKPOINT_PATH = DATA_DIR / "simulation_checkpoint.json"
    if checkpoint and CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            ckpt = json.load(f)
        resume_day = ckpt.get("last_completed_day", 0) + 1
        if resume_day > start_day:
            print(f"  [CHECKPOINT] Resuming from day {resume_day} (last completed: {resume_day - 1})")
            start_day = resume_day

    # ── Compute Backend Selection ──
    backend_info = None
    if backend != "auto":
        try:
            from src.compute_backend import get_training_backend
            backend_info = get_training_backend(force_backend=backend)
        except ImportError:
            pass

    # Initialize MLflow for simulation logging
    init_mlflow()

    sim_start = datetime.strptime(SIMULATION["start_date"], "%Y-%m-%d")
    sim_end = sim_start + timedelta(days=end_day - 1)

    timeline = {
        "start_time": datetime.now().isoformat(),
        "simulation_start_date": SIMULATION["start_date"],
        "simulation_end_date": sim_end.strftime("%Y-%m-%d"),
        "rows_per_day": rows_per_day,
        "total_days": end_day - start_day + 1,
        "days": [],
    }

    retrain_history = []
    last_retrain_day = 0  # Staleness gate: min 30 days between retrains
    model_version = "v1_original"

    print(f"\n{'='*70}")
    print("P053 — 40-DAY PRODUCTION SIMULATION")
    print(f"{'='*70}")
    print(f"  Days:     {start_day}-{end_day}")
    print(f"  Timeline: {sim_start.strftime('%b %d, %Y')} → {sim_end.strftime('%b %d, %Y')}")
    print(f"  Rows/day: {rows_per_day:,}")
    print(f"  Total:    {(end_day-start_day+1) * rows_per_day:,} rows")
    print(f"  Spark:    {'enabled' if not skip_spark else 'disabled'}")
    print(f"  Kafka:    {'enabled' if not skip_kafka else 'disabled'}")
    if backend_info:
        print(f"  Backend:  {backend_info.name.value} ({backend_info.gpu_name})")
        print(f"  MLflow:   {backend_info.mlflow_uri[:60]}")
    else:
        print("  Backend:  auto (AWS → Colab → Local)")
    if checkpoint:
        print(f"  Checkpoint: enabled {'(resuming)' if start_day > 1 else '(fresh)'}")
    print(f"{'='*70}\n")

    for day in range(start_day, end_day + 1):
        day_t0 = time.time()
        cfg = get_drift_config(day)
        day_date = (sim_start + timedelta(days=day - 1)).strftime("%Y-%m-%d")
        print(f"  ▶ Day {day:2d}/40 | {day_date} | {cfg['scenario']}", flush=True)

        # Initialize comprehensive daily logger
        scale = "phase2"  # Default; overridden by SIMULATION_SCALE env var
        import os as _os
        scale = _os.environ.get("SIMULATION_SCALE", scale)
        day_logger = SimulationDayLogger(day=day, phase=scale)

        day_record = {
            "day": day,
            "date": (sim_start + timedelta(days=day - 1)).strftime("%Y-%m-%d"),
            "scenario": cfg["scenario"],
            "model_version": model_version,
            "events": [],
        }

        # ── STEP 1: Generate data ──
        gen_t0 = time.time()
        parquet_path = generate_day(day, n_rows=rows_per_day)
        day_record["parquet_mb"] = round(parquet_path.stat().st_size / 1e6, 1)
        print(f"    ✓ Data: {rows_per_day:,} rows | {day_record['parquet_mb']:.1f} MB | {time.time()-gen_t0:.1f}s", flush=True)
        day_logger.log_data_generation(
            rows=rows_per_day, parquet_mb=day_record["parquet_mb"],
            scenario=cfg["scenario"], elapsed_sec=round(time.time() - gen_t0, 1),
        )

        # ── GPU/Infra Selection ──
        try:
            from src.gpu_selector import get_gpu_decision_for_day
            model_params = int(_os.environ.get("MODEL_PARAMS", "317000"))
            current_instance = _os.environ.get("EC2_INSTANCE_TYPE", "g4dn.xlarge")
            gpu_decision = get_gpu_decision_for_day(day, model_params, current_instance)
            day_logger.log_infra_selection(
                model_params=model_params,
                estimated_vram_gb=gpu_decision["estimated_vram_gb"],
                selected_gpu=gpu_decision["selected_gpu"],
                instance_type=gpu_decision["selected_instance"],
                cost_per_hour=gpu_decision["cost_per_hour"],
                needs_switch=gpu_decision["needs_instance_switch"],
                reason=gpu_decision["action"],
            )
        except Exception as e:
            day_logger.logger.warning(f"[INFRA] GPU selector unavailable: {e}")

        # ── STEP 2: Kafka (optional) ──
        if not skip_kafka:
            try:
                from src.kafka_producer import create_producer, publish_day
                producer = create_producer()
                stats = publish_day(day, producer, batch_size=10_000)
                day_record["kafka_msg_per_sec"] = stats["throughput_msg_per_sec"]
                day_record["events"].append("kafka_published")
                day_logger.log_kafka("published", msg_per_sec=stats["throughput_msg_per_sec"])
            except (ImportError, Exception) as e:
                day_record["events"].append(f"kafka_skip: {e}")
                day_logger.log_kafka("skipped", error=str(e))

        # ── STEP 3: Spark ETL (optional) ──
        if not skip_spark:
            try:
                from src.spark_etl import run_etl
                etl_result = run_etl(day, day)
                day_record["spark_rows_per_sec"] = etl_result.get("throughput_rows_per_sec", 0)
                day_record["events"].append("spark_etl_complete")
                day_logger.log_spark_etl("completed", rows_per_sec=day_record["spark_rows_per_sec"])
            except (ImportError, Exception) as e:
                day_record["events"].append(f"spark_skip: {e}")
                day_logger.log_spark_etl("skipped", error=str(e))

        # ── STEP 4: Drift detection (standalone, no Spark) ──
        if day >= 9:  # Only check after reference window
            drift_result = _standalone_drift_check(day)
            day_record["drift"] = drift_result

            # Log drift to comprehensive daily log
            day_logger.log_drift_detection(
                features_critical=drift_result["features_critical"],
                features_warning=drift_result["features_warning"],
                feature_psi=drift_result.get("feature_psi", {}),
                drift_reliable=drift_result.get("drift_reliable", True),
                ref_rows=drift_result.get("ref_rows", 0),
                analysis_rows=drift_result.get("analysis_rows", 0),
                low_data_warning=drift_result.get("low_data_warning"),
            )

            if drift_result["features_critical"] >= 3:
                day_record["events"].append(f"drift_critical_{drift_result['features_critical']}_features")
            elif drift_result["features_warning"] > 0:
                day_record["events"].append(f"drift_warning_{drift_result['features_warning']}_features")
            else:
                day_record["events"].append("drift_clean")

            # ── STEP 5: Retrain check (3-criteria gate) ──
            days_since_retrain = day - last_retrain_day
            low_data_blocked = not drift_result.get("drift_reliable", True)
            should_retrain = (
                drift_result["features_critical"] >= 3
                and days_since_retrain >= 30
                and not low_data_blocked
            )
            if should_retrain:
                day_record["events"].append("RETRAIN_TRIGGERED")
                retrain_num = len(retrain_history) + 2  # v2, v3, ...
                model_version = f"v{retrain_num}_retrained_day{day}"
                day_record["model_version"] = model_version

                day_logger.log_retrain_decision(
                    should_retrain=True,
                    reason=f"{drift_result['features_critical']} critical features, "
                           f"{days_since_retrain}d since last retrain",
                    drift_critical=drift_result["features_critical"],
                    days_since_retrain=days_since_retrain,
                )

                # Execute training — only update staleness if training succeeds
                print(f"    🔁 RETRAIN triggered → {model_version} ({sim_retrain_epochs} epochs)", flush=True)
                retrain_status = _log_retrain_to_mlflow(
                    day, model_version, drift_result, days_since_retrain,
                    sim_retrain_epochs=sim_retrain_epochs)
                if retrain_status == "succeeded":
                    retrain_history.append({"day": day, "new_model": model_version})
                    last_retrain_day = day
                else:
                    # Retrain skipped or failed — keep champion and don't block future retries.
                    model_version = retrain_history[-1]["new_model"] if retrain_history else "v1_original"
                    day_record["model_version"] = model_version
                    if retrain_status == "failed":
                        day_record["events"].append("RETRAIN_FAILED")
                    else:
                        day_record["events"].append("RETRAIN_SKIPPED")
            elif drift_result["features_critical"] >= 3:
                reason = (f"staleness gate ({days_since_retrain}d < 30)"
                          if not low_data_blocked
                          else "low-data drift (tagged only, not actionable)")
                day_logger.log_retrain_decision(
                    should_retrain=False, reason=reason,
                    drift_critical=drift_result["features_critical"],
                    days_since_retrain=days_since_retrain,
                    staleness_blocked=(days_since_retrain < 30),
                    low_data_blocked=low_data_blocked,
                )
                day_record["events"].append(f"retrain_blocked_staleness_{days_since_retrain}d")
            else:
                day_logger.log_retrain_decision(
                    should_retrain=False,
                    reason=f"only {drift_result['features_critical']} critical features (need ≥3)",
                    drift_critical=drift_result["features_critical"],
                    days_since_retrain=days_since_retrain,
                )

        # Day 39: bad model deploy scenario
        if day == 39:
            day_record["events"].append("BAD_MODEL_DEPLOYED")
            day_record["events"].append("CANARY_FAILED")
            day_record["events"].append("ROLLBACK_TO_v2")
            rollback_to = retrain_history[-1]["new_model"] if retrain_history else "v1_original"
            day_logger.log_rollback(
                from_version=model_version, to_version=rollback_to,
                reason="Day 39 bad model deploy scenario — canary failed",
            )
            model_version = rollback_to

        if day == 40:
            day_record["events"].append("SYSTEM_RECOVERED")

        # ── S3 Upload (non-blocking) ──
        try:
            from src.s3_utils import upload_simulation_artifacts
            s3_result = upload_simulation_artifacts(
                day, str(DATA_DIR), skip_parquet=(rows_per_day < 1_000_000),
            )
            if s3_result.get("status") != "skipped":
                n_uploaded = len([k for k in s3_result if k != "status"])
                day_record["s3_uploaded"] = n_uploaded
                day_record["events"].append(f"s3_uploaded_{n_uploaded}_files")
                day_logger.log_s3_upload(n_files=n_uploaded, status="uploaded")
                print(f"    ☁ S3: {n_uploaded} files uploaded", flush=True)
            else:
                day_logger.log_s3_upload(n_files=0, status="skipped")
        except Exception as e:
            day_record["events"].append(f"s3_skip: {e}")
            day_logger.log_s3_upload(n_files=0, status="error", error=str(e))
            print(f"    ⚠ S3 skipped: {e}", flush=True)

        day_elapsed = time.time() - day_t0
        day_record["elapsed_sec"] = round(day_elapsed, 1)
        timeline["days"].append(day_record)

        # Finalize comprehensive daily log (writes .log + .json)
        day_logger.finalize()

        # ── Checkpoint Save ──
        if checkpoint:
            ckpt_data = {
                "last_completed_day": day,
                "timestamp": datetime.now().isoformat(),
                "backend": backend,
                "model_version": model_version,
                "retrain_history": retrain_history,
                "last_retrain_day": last_retrain_day,
            }
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(ckpt_data, f, indent=2)

        # Progress line
        events_str = ", ".join(day_record["events"][:4]) if day_record["events"] else "—"
        parquet_mb = day_record.get('parquet_mb', 0)
        print(f"  ✅ Day {day:2d} done | {day_elapsed:>6.1f}s | {events_str}", flush=True)

    # ── SAVE TIMELINE ──
    total_elapsed = time.time() - t0
    timeline["total_elapsed_sec"] = round(total_elapsed, 1)
    timeline["total_elapsed_min"] = round(total_elapsed / 60, 1)
    timeline["retrain_events"] = retrain_history

    with open(TIMELINE_PATH, "w") as f:
        json.dump(timeline, f, indent=2)

    # Print summary
    total_rows = (end_day - start_day + 1) * rows_per_day
    total_size_mb = sum(d["parquet_mb"] for d in timeline["days"])

    print(f"\n{'='*70}")
    print("SIMULATION COMPLETE")
    print(f"  Days:          {end_day - start_day + 1}")
    print(f"  Total rows:    {total_rows:,}")
    print(f"  Total Parquet: {total_size_mb:,.0f} MB")
    print(f"  Retrain events: {len(retrain_history)}")
    print(f"  Timeline:      {TIMELINE_PATH}")
    print(f"  Wall clock:    {total_elapsed/60:.1f} min")
    print(f"{'='*70}")

    return timeline


def _log_retrain_to_mlflow(day: int, model_version: str,
                           drift_result: dict, days_since_retrain: int,
                           sim_retrain_epochs: int = 10) -> str:
    """Execute REAL GPU training and log to MLflow.

    Returns one of: "succeeded", "skipped", or "failed".

    Replaces the old stub that only logged metadata.
    Now calls train.py which runs actual PyTorch training on GPU
    with full MLflow experiment tracking.

    Args:
        sim_retrain_epochs: Epochs for simulation retrains. Default 50 for
            production quality. On T4, each epoch takes ~530s, so 50 epochs
            ≈ 4.9h (with early stopping at ~33 epochs ≈ 4.9h).
            Set 0 to skip actual training and log metadata only.

    Falls back to metadata-only logging if train.py fails or
    preprocessed data is not available (e.g., fast mode).
    """
    import subprocess

    sim_start = datetime.strptime(SIMULATION["start_date"], "%Y-%m-%d")
    retrain_date = (sim_start + timedelta(days=day - 1)).strftime("%Y-%m-%d")

    run_name = f"retrain-day{day:02d}-{model_version}"

    # Attempt REAL GPU training
    data_path = PROJECT_ROOT / "data" / "preprocessed_full.npz"
    training_succeeded = False
    training_status = "failed"

    # T4/V100 float16 retrains: batch_size=512 (ED-004 v2 — 1024 still causes NaN spiral on T4).
    # A100/H100 (cc>=8): batch_size=4096, bfloat16 is safe.
    batch_size = "512"
    learning_rate = "2e-4"
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            cc = _torch.cuda.get_device_capability()[0]
            if cc >= 8:
                batch_size = "4096"
                learning_rate = "1e-3"
            print(
                f"    [RETRAIN] GPU compute capability {cc} → batch_size={batch_size}, lr={learning_rate}"
            )
    except Exception:
        pass

    if data_path.exists() and sim_retrain_epochs > 0:
        cmd = [
            sys.executable, "-m", "src.train",
            "--full",
            "--epochs", str(sim_retrain_epochs),
            "--batch-size", batch_size,
            "--lr", learning_rate,
            "--run-name", run_name,
            "--context", "simulation-retrain",
        ]
        print(f"    [RETRAIN] Launching GPU training ({sim_retrain_epochs} epochs)", flush=True)

        try:
            # Stream stdout/stderr to parent so epoch progress is visible in Colab
            result = subprocess.run(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=28800,  # 8h: 50 epochs × 530s/epoch ≈ 7.4h on T4
            )
            if result.returncode == 0:
                training_succeeded = True
                training_status = "succeeded"
                # Read benchmark for metrics
                import glob as _glob
                benchmark_files = sorted(
                    _glob.glob(str(PROJECT_ROOT / "data" / "benchmark_*.json")),
                    key=lambda f: Path(f).stat().st_mtime,
                    reverse=True,
                )
                if benchmark_files:
                    with open(benchmark_files[0]) as f:
                        benchmark = json.load(f)
                    print(f"    [RETRAIN] GPU training complete: "
                          f"AUC-PR={benchmark['results']['val']['auc_pr']:.4f}, "
                          f"GPU={benchmark['gpu_name']}, "
                          f"Time={benchmark['total_train_time_min']:.1f}min", flush=True)
                    mlflow_run_id = benchmark.get("mlflow_run_id")
                    if mlflow_run_id:
                        print(f"    [RETRAIN] MLflow run: {mlflow_run_id}")

                # Auto-upload model .pt to S3
                from src.config import ARTIFACTS_DIR
                model_pt = ARTIFACTS_DIR / "hybrid_best_full.pt"
                if model_pt.exists():
                    try:
                        from src.s3_utils import S3ArtifactManager
                        s3 = S3ArtifactManager()
                        s3_uri = s3.upload_model(day, str(model_pt), version_tag=model_version)
                        print(f"    ☁ Model uploaded to {s3_uri}", flush=True)
                    except Exception as e:
                        print(f"    ⚠ Model S3 upload failed: {e}", flush=True)
            else:
                print(f"    [RETRAIN] GPU training failed (exit {result.returncode}), "
                      f"falling back to metadata logging")
        except subprocess.TimeoutExpired:
            print("    [RETRAIN] Training timed out after 8h, falling back to metadata logging")
        except Exception as e:
            print(f"    [RETRAIN] Training error: {e}, falling back to metadata logging")
    elif sim_retrain_epochs == 0:
        training_status = "skipped"
        print("    [RETRAIN] sim_retrain_epochs=0: skipping GPU training, logging metadata only")
    else:
        training_status = "skipped"
        print(f"    [RETRAIN] No preprocessed data at {data_path}, logging metadata only")

    # Fallback: log metadata to MLflow if training didn't run
    if not training_succeeded:
        with mlflow.start_run(run_name=run_name, tags={
            **MLFLOW["default_tags"],
            "run_type": "simulation_retrain_metadata",
            "simulation_day": str(day),
            "retrain_date": retrain_date,
            "model_version": model_version,
            "trigger": "drift_detected",
            "training_executed": "false",
            "training_status": training_status,
        }) as run:
            mlflow.log_params({
                "simulation.day": day,
                "simulation.date": retrain_date,
                "simulation.days_since_last_retrain": days_since_retrain,
                "drift.features_critical": drift_result["features_critical"],
                "drift.features_warning": drift_result["features_warning"],
            })

            for feat, psi_val in drift_result.get("feature_psi", {}).items():
                mlflow.log_metric(f"psi.{feat}", psi_val)

            report_path = DRIFT_REPORT_DIR / f"drift_day_{day:02d}.json"
            if report_path.exists():
                mlflow.log_artifact(str(report_path), "drift_reports")

            mlflow.set_tag("mlflow.note.content",
                f"Retrain triggered on Day {day} ({retrain_date}). "
                f"{drift_result['features_critical']} features exceeded PSI>0.2. "
                f"Days since last retrain: {days_since_retrain}. "
                f"Model: {model_version}. "
                f"Training status: {training_status}. Metadata logged only."
            )

        print(f"    MLflow: logged retrain metadata {run.info.run_id[:8]}...")

    return training_status

def _standalone_drift_check(day: int) -> dict:
    """
    Lightweight drift check using pandas (no Spark needed).
    Computes PSI for key features against day 1 reference.
    """
    import numpy as np
    import pandas as pd

    ref_path = PRODUCTION_DIR / "day_01.parquet"
    analysis_path = PRODUCTION_DIR / f"day_{day:02d}.parquet"

    if not ref_path.exists() or not analysis_path.exists():
        return {"features_critical": 0, "features_warning": 0, "error": "missing files",
                "drift_reliable": False}

    # Sample for speed (100K rows is enough for PSI)
    ref_df = pd.read_parquet(ref_path).sample(n=min(100_000, 5_000_000), random_state=42)
    analysis_df = pd.read_parquet(analysis_path).sample(n=min(100_000, 5_000_000), random_state=42)

    # Low-data tagging: if either dataset has < 10K rows, drift detection
    # is statistically unreliable. Log to MLflow for transparency but
    # NEVER use for retrain decisions.
    MIN_RELIABLE_ROWS = 10_000
    low_data = len(ref_df) < MIN_RELIABLE_ROWS or len(analysis_df) < MIN_RELIABLE_ROWS

    key_features = [
        "test_temp_c", "cell_leakage_fa", "retention_time_ms",
        "gate_oxide_thickness_a", "vt_shift_mv", "trcd_ns",
    ]

    n_critical = 0
    n_warning = 0
    feature_psi = {}

    for feat in key_features:
        ref_vals = ref_df[feat].dropna().values
        analysis_vals = analysis_df[feat].dropna().values

        if len(ref_vals) < 100 or len(analysis_vals) < 100:
            continue

        # PSI with 10 quantile bins
        quantiles = np.linspace(0, 100, 11)
        bin_edges = np.percentile(ref_vals, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        eps = 1e-6
        ref_counts = np.histogram(ref_vals, bins=bin_edges)[0]
        analysis_counts = np.histogram(analysis_vals, bins=bin_edges)[0]

        ref_prop = ref_counts / max(ref_counts.sum(), 1) + eps
        analysis_prop = analysis_counts / max(analysis_counts.sum(), 1) + eps

        psi = float(np.sum((analysis_prop - ref_prop) * np.log(analysis_prop / ref_prop)))
        feature_psi[feat] = round(psi, 6)

        if psi > 0.2:
            n_critical += 1
        elif psi > 0.1:
            n_warning += 1

    # Save drift report
    # If low data, tag the drift but NEVER allow retrain
    drift_reliable = not low_data
    should_retrain = n_critical >= 3 and drift_reliable

    report = {
        "analysis_day": day,
        "features_critical": n_critical,
        "features_warning": n_warning,
        "feature_psi": feature_psi,
        "should_retrain": should_retrain,
        "drift_reliable": drift_reliable,
        "low_data_warning": (
            f"Only {min(len(ref_df), len(analysis_df)):,} rows — "
            f"drift tagged for transparency, excluded from retrain decisions"
        ) if low_data else None,
        "ref_rows": len(ref_df),
        "analysis_rows": len(analysis_df),
    }

    report_path = DRIFT_REPORT_DIR / f"drift_day_{day:02d}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 40-day production simulation")
    parser.add_argument("--day", type=int, default=1, help="Start day")
    parser.add_argument("--end-day", type=int, default=40, help="End day")

    # Scale modes
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument("--fast", action="store_true",
                             help="100K rows/day (quick test, ~10 min)")
    scale_group.add_argument("--medium", action="store_true",
                             help="1M rows/day (demo, ~30 min)")
    scale_group.add_argument("--full", action="store_true",
                             help="5M rows/day (production, ~2 hrs)")
    parser.add_argument("--rows", type=int, default=None,
                        help="Custom rows per day")

    # Component toggles
    parser.add_argument("--with-spark", action="store_true",
                        help="Enable Spark ETL (requires PySpark)")
    parser.add_argument("--with-kafka", action="store_true",
                        help="Enable Kafka publishing (requires broker)")

    # Compute backend
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "aws", "kaggle", "colab", "local"],
                        help="Training backend: aws | kaggle (auto API) | colab | local | "
                             "auto (tries: Colab-if-there → AWS → Kaggle → 2hr-wait → local)")
    parser.add_argument("--checkpoint", action="store_true",
                        help="Save progress after each day, resume if interrupted")
    parser.add_argument("--sim-retrain-epochs", type=int, default=10,
                        help="Epochs for each simulation retrain event. "
                             "10 = ~30-40min on T4 (default). "
                             "0 = metadata-only, fastest. "
                             "50 = full production quality.")

    args = parser.parse_args()

    # Determine rows per day
    if args.rows:
        rows = args.rows
    elif args.fast:
        rows = 100_000
    elif args.medium:
        rows = 1_000_000
    else:
        rows = 5_000_000

    run_simulation(
        start_day=args.day,
        end_day=args.end_day,
        rows_per_day=rows,
        skip_spark=not args.with_spark,
        skip_kafka=not args.with_kafka,
        backend=args.backend,
        checkpoint=args.checkpoint,
        sim_retrain_epochs=args.sim_retrain_epochs,
    )
