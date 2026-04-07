"""
P053 — Airflow DAG: Daily Yield Pipeline
=========================================
Runs once per simulated day. Orchestrates:
    1. Generate production data (streaming_data_generator)
    2. Publish to Kafka (kafka_producer)
    3. Run Spark ETL (spark_etl)
    4. Run drift detection (spark_drift_detector)
    5. Evaluate retrain criteria

This is the PRIMARY production DAG — runs 40 times in the simulation.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

PROJECT_ROOT = "/opt/airflow"
DATA_DIR = f"{PROJECT_ROOT}/data"
SRC_DIR = f"{PROJECT_ROOT}/src"

default_args = {
    "owner": "p053",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    dag_id="p053_daily_yield_pipeline",
    default_args=default_args,
    description="Daily DRAM yield monitoring pipeline",
    schedule_interval=None,  # Triggered by run_simulation
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["p053", "production", "daily"],
    params={"day_number": 1, "n_rows": 5000000},
)


# ─── TASK 1: Generate production data ──────────────────────
generate_data = BashOperator(
    task_id="generate_production_data",
    bash_command=(
        f"cd {PROJECT_ROOT} && "
        "python -m src.streaming_data_generator "
        "--day {{ params.day_number }} "
        "--rows {{ params.n_rows }}"
    ),
    dag=dag,
)


# ─── TASK 2: Publish to Kafka ──────────────────────────────
publish_kafka = BashOperator(
    task_id="publish_to_kafka",
    bash_command=(
        f"cd {PROJECT_ROOT} && "
        "python -m src.kafka_producer "
        "--day {{ params.day_number }} "
        "--batch-size 10000"
    ),
    dag=dag,
)


# ─── TASK 3: Spark ETL ──────────────────────────────────────
spark_etl = BashOperator(
    task_id="spark_etl",
    bash_command=(
        "spark-submit --master local[*] "
        f"--driver-memory 4g {SRC_DIR}/spark_etl.py "
        "--day {{ params.day_number }}"
    ),
    dag=dag,
)


# ─── TASK 4: Drift detection ──────────────────────────────
drift_detection = BashOperator(
    task_id="drift_detection",
    bash_command=(
        "spark-submit --master local[*] "
        f"--driver-memory 4g {SRC_DIR}/spark_drift_detector.py "
        "--ref-days 1-8 "
        "--analysis-day {{ params.day_number }}"
    ),
    dag=dag,
)


# ─── TASK 5: Check retrain criteria ──────────────────────
def _check_retrain_criteria(**context):
    """Read drift report and decide whether to trigger retrain."""
    day = context["params"]["day_number"]
    report_path = Path(DATA_DIR) / "drift_reports" / f"drift_day_{day:02d}.json"

    if not report_path.exists():
        print(f"[INFO] No drift report for day {day}, skipping retrain check")
        return "skip_retrain"

    with open(report_path) as f:
        report = json.load(f)

    should_retrain = report.get("should_retrain", False)
    critical = report.get("features_critical", 0)
    drift_reliable = report.get("drift_reliable", True)

    # Low-data guard: log for transparency, never retrain
    if not drift_reliable:
        low_data_msg = report.get("low_data_warning", "insufficient samples")
        print(f"[LOW-DATA] Day {day}: drift tagged for info only — {low_data_msg}")
        print(f"[LOW-DATA] {critical} critical features detected but NOT actionable")
        return "skip_retrain"

    # Staleness gate: need >= 30 days of data for retrain
    if day < 30:
        print(f"[INFO] Day {day} < 30: staleness gate blocks retrain (critical={critical})")
        return "skip_retrain"

    if should_retrain:
        print(f"[RETRAIN] Day {day}: {critical} critical features → triggering retrain!")
        return "trigger_retrain"
    else:
        print(f"[OK] Day {day}: {critical} critical features → no retrain needed")
        return "skip_retrain"


check_retrain = BranchPythonOperator(
    task_id="check_retrain_criteria",
    python_callable=_check_retrain_criteria,
    dag=dag,
)


# ─── TASK 6: Trigger retrain (conditional) ─────────────────
trigger_retrain = TriggerDagRunOperator(
    task_id="trigger_retrain",
    trigger_dag_id="p053_retrain_pipeline",
    conf={"day_number": "{{ params.day_number }}"},
    dag=dag,
)


# ─── TASK 7: Skip retrain ──────────────────────────────────
skip_retrain = EmptyOperator(
    task_id="skip_retrain",
    dag=dag,
)


# ─── TASK 8: Log completion + S3 upload ───────────────────
def _log_completion(**context):
    day = context["params"]["day_number"]

    # Upload day's artifacts to S3 (non-blocking)
    try:
        import sys
        sys.path.insert(0, "/opt/airflow")
        from src.s3_utils import upload_simulation_artifacts
        uploaded = upload_simulation_artifacts(day, f"{DATA_DIR}")
        if uploaded.get("status") != "skipped":
            n_files = len([k for k in uploaded if k != "status"])
            print(f"[S3] Day {day}: uploaded {n_files} artifacts to S3")
    except Exception as e:
        print(f"[S3] Day {day}: S3 upload failed (non-fatal): {e}")

    print(f"[DONE] Day {day} pipeline complete")


log_complete = PythonOperator(
    task_id="log_completion",
    python_callable=_log_completion,
    trigger_rule="none_failed_min_one_success",
    dag=dag,
)


# ─── DAG FLOW ────────────────────────────────────────────────
generate_data >> publish_kafka >> spark_etl >> drift_detection >> check_retrain
check_retrain >> [trigger_retrain, skip_retrain]
[trigger_retrain, skip_retrain] >> log_complete
