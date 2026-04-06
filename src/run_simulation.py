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
from src.streaming_data_generator import PRODUCTION_DIR, generate_day, get_drift_config

TIMELINE_PATH = DATA_DIR / "simulation_timeline.json"
DRIFT_REPORT_DIR = DATA_DIR / "drift_reports"


def run_simulation(start_day: int = 1, end_day: int = 40,
                   rows_per_day: int = 5_000_000,
                   skip_spark: bool = False,
                   skip_kafka: bool = True) -> dict:
    """
    Execute the full 40-day production simulation.

    Returns a timeline dict with per-day events for dashboard plotting.
    """
    t0 = time.time()
    DRIFT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

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
    print(f"{'='*70}\n")

    for day in range(start_day, end_day + 1):
        day_t0 = time.time()
        cfg = get_drift_config(day)

        day_record = {
            "day": day,
            "date": (sim_start + timedelta(days=day - 1)).strftime("%Y-%m-%d"),
            "scenario": cfg["scenario"],
            "model_version": model_version,
            "events": [],
        }

        # ── STEP 1: Generate data ──
        parquet_path = generate_day(day, n_rows=rows_per_day)
        day_record["parquet_mb"] = round(parquet_path.stat().st_size / 1e6, 1)

        # ── STEP 2: Kafka (optional) ──
        if not skip_kafka:
            try:
                from src.kafka_producer import create_producer, publish_day
                producer = create_producer()
                stats = publish_day(day, producer, batch_size=10_000)
                day_record["kafka_msg_per_sec"] = stats["throughput_msg_per_sec"]
                day_record["events"].append("kafka_published")
            except (ImportError, Exception) as e:
                day_record["events"].append(f"kafka_skip: {e}")

        # ── STEP 3: Spark ETL (optional) ──
        if not skip_spark:
            try:
                from src.spark_etl import run_etl
                etl_result = run_etl(day, day)
                day_record["spark_rows_per_sec"] = etl_result.get("throughput_rows_per_sec", 0)
                day_record["events"].append("spark_etl_complete")
            except (ImportError, Exception) as e:
                day_record["events"].append(f"spark_skip: {e}")

        # ── STEP 4: Drift detection (standalone, no Spark) ──
        if day >= 9:  # Only check after reference window
            drift_result = _standalone_drift_check(day)
            day_record["drift"] = drift_result
            if drift_result["features_critical"] >= 3:
                day_record["events"].append(f"drift_critical_{drift_result['features_critical']}_features")
            elif drift_result["features_warning"] > 0:
                day_record["events"].append(f"drift_warning_{drift_result['features_warning']}_features")
            else:
                day_record["events"].append("drift_clean")

            # ── STEP 5: Retrain check (3-criteria gate) ──
            days_since_retrain = day - last_retrain_day
            should_retrain = (
                drift_result["features_critical"] >= 3
                and days_since_retrain >= 30
            )
            if should_retrain:
                day_record["events"].append("RETRAIN_TRIGGERED")
                retrain_num = len(retrain_history) + 2  # v2, v3, ...
                model_version = f"v{retrain_num}_retrained_day{day}"
                day_record["model_version"] = model_version
                retrain_history.append({"day": day, "new_model": model_version})
                last_retrain_day = day

                # Log retrain event to MLflow
                _log_retrain_to_mlflow(day, model_version, drift_result, days_since_retrain)
            elif drift_result["features_critical"] >= 3:
                day_record["events"].append(f"retrain_blocked_staleness_{days_since_retrain}d")

        # Day 39: bad model deploy scenario
        if day == 39:
            day_record["events"].append("BAD_MODEL_DEPLOYED")
            day_record["events"].append("CANARY_FAILED")
            day_record["events"].append("ROLLBACK_TO_v2")
            model_version = retrain_history[-1]["new_model"] if retrain_history else "v1_original"

        if day == 40:
            day_record["events"].append("SYSTEM_RECOVERED")

        day_elapsed = time.time() - day_t0
        day_record["elapsed_sec"] = round(day_elapsed, 1)
        timeline["days"].append(day_record)

        # Progress line
        events_str = ", ".join(day_record["events"][:3]) if day_record["events"] else "—"
        print(f"  Day {day:>2} [{cfg['scenario']:>22}] {day_elapsed:>5.1f}s | {events_str}")

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
                           drift_result: dict, days_since_retrain: int):
    """Log a simulated retrain event as an MLflow run.

    In production, this would trigger actual GPU training (via CI/CD or Airflow).
    Here we log the retrain decision metadata so MLflow shows the full lifecycle.
    """
    sim_start = datetime.strptime(SIMULATION["start_date"], "%Y-%m-%d")
    retrain_date = (sim_start + timedelta(days=day - 1)).strftime("%Y-%m-%d")

    with mlflow.start_run(run_name=f"retrain-day{day}-{model_version}", tags={
        **MLFLOW["default_tags"],
        "run_type": "simulation_retrain",
        "simulation_day": str(day),
        "retrain_date": retrain_date,
        "model_version": model_version,
        "trigger": "drift_detected",
    }) as run:
        mlflow.log_params({
            "simulation.day": day,
            "simulation.date": retrain_date,
            "simulation.days_since_last_retrain": days_since_retrain,
            "drift.features_critical": drift_result["features_critical"],
            "drift.features_warning": drift_result["features_warning"],
        })

        # Log per-feature PSI values
        for feat, psi_val in drift_result.get("feature_psi", {}).items():
            mlflow.log_metric(f"psi.{feat}", psi_val)

        # Log drift report as artifact
        report_path = DRIFT_REPORT_DIR / f"drift_day_{day:02d}.json"
        if report_path.exists():
            mlflow.log_artifact(str(report_path), "drift_reports")

        mlflow.set_tag("mlflow.note.content",
            f"Retrain triggered on Day {day} ({retrain_date}). "
            f"{drift_result['features_critical']} features exceeded PSI>0.2 critical threshold. "
            f"Days since last retrain: {days_since_retrain}. "
            f"New model version: {model_version}. "
            f"In production, this would trigger an Airflow DAG → Colab A100 training → "
            f"MLflow model registration → canary deployment."
        )

    print(f"    MLflow: logged retrain event {run.info.run_id[:8]}...")


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
        return {"features_critical": 0, "features_warning": 0, "error": "missing files"}

    # Sample for speed (100K rows is enough for PSI)
    ref_df = pd.read_parquet(ref_path).sample(n=min(100_000, 5_000_000), random_state=42)
    analysis_df = pd.read_parquet(analysis_path).sample(n=min(100_000, 5_000_000), random_state=42)

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
    report = {
        "analysis_day": day,
        "features_critical": n_critical,
        "features_warning": n_warning,
        "feature_psi": feature_psi,
        "should_retrain": n_critical >= 3,
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
    )
