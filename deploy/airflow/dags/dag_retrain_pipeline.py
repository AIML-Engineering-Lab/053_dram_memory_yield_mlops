"""
P053 — Airflow DAG: Retrain Pipeline
=====================================
Triggered by the daily pipeline when all 3 retrain criteria are met:
    1. Drift:   ≥ 3 features with PSI > 0.2
    2. Performance: AUC-PR drop > 5% from baseline
    3. Staleness: ≥ 30 days since last training

Pipeline steps:
    1. Collect training data (sliding window: last 15-31 days)
    2. Run Spark ETL on training window
    3. Launch model training (simulated — in production this would be SageMaker)
    4. Evaluate new model on holdout set
    5. Canary deployment (compare new vs old)
    6. Promote or rollback

This DAG demonstrates the full automated retraining loop that a
principal engineer would design at Micron/Samsung.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import json
from pathlib import Path

PROJECT_ROOT = "/opt/airflow"
DATA_DIR = f"{PROJECT_ROOT}/data"

default_args = {
    "owner": "p053",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

dag = DAG(
    dag_id="p053_retrain_pipeline",
    default_args=default_args,
    description="Automated model retraining pipeline",
    schedule_interval=None,  # Only triggered by daily pipeline
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["p053", "retrain", "mlops"],
    params={"day_number": 31},
)


# ─── TASK 1: Collect training window ─────────────────────
def _determine_training_window(**context):
    """Select 15-day sliding window for retraining."""
    day = int(context["params"]["day_number"])

    # Use last 15 days of data for retraining
    window_size = 15
    start_day = max(1, day - window_size + 1)
    end_day = day

    context["ti"].xcom_push(key="train_start_day", value=start_day)
    context["ti"].xcom_push(key="train_end_day", value=end_day)
    print(f"[RETRAIN] Training window: days {start_day}-{end_day} ({end_day-start_day+1} days)")
    return {"start_day": start_day, "end_day": end_day}


determine_window = PythonOperator(
    task_id="determine_training_window",
    python_callable=_determine_training_window,
    dag=dag,
)


# ─── TASK 2: Spark ETL on training window ─────────────────
retrain_etl = BashOperator(
    task_id="retrain_spark_etl",
    bash_command=(
        "spark-submit --master local[*] "
        f"--driver-memory 4g {PROJECT_ROOT}/src/spark_etl.py "
        "--day {{ ti.xcom_pull(key='train_start_day') }} "
        "--end-day {{ ti.xcom_pull(key='train_end_day') }}"
    ),
    dag=dag,
)


# ─── TASK 3: Simulate model training ──────────────────────
def _simulate_training(**context):
    """
    Simulate model training on the new data window.
    In production, this would launch a SageMaker training job.
    Here we log what would happen and create a mock model artifact.
    """
    day = int(context["params"]["day_number"])
    start_day = context["ti"].xcom_pull(key="train_start_day")
    end_day = context["ti"].xcom_pull(key="train_end_day")

    # Simulate training metrics
    training_result = {
        "model_version": f"v2_retrained_day{day}",
        "training_window": f"days {start_day}-{end_day}",
        "training_rows": (end_day - start_day + 1) * 5_000_000,
        "epochs": 50,
        "best_epoch": 28,
        "val_auc_pr": 0.0485,  # Simulated — would be from actual training
        "val_f1": 0.0612,
        "training_time_minutes": 45,
        "gpu": "A100-SXM4-40GB",
        "dtype": "bfloat16",
    }

    # Save training result
    result_path = Path(DATA_DIR) / "retrain_results" / f"retrain_day_{day:02d}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(training_result, f, indent=2)

    print(f"[RETRAIN] Training complete: AUC-PR={training_result['val_auc_pr']:.4f}")
    context["ti"].xcom_push(key="val_auc_pr", value=training_result["val_auc_pr"])
    return training_result


simulate_training = PythonOperator(
    task_id="simulate_model_training",
    python_callable=_simulate_training,
    dag=dag,
)


# ─── TASK 4: Canary evaluation ──────────────────────────────
def _canary_evaluation(**context):
    """
    Compare new model against old model on next day's data.
    In production: route 10% traffic to new model, compare metrics.
    Here: simulate the comparison.
    """
    new_auc = context["ti"].xcom_pull(key="val_auc_pr")
    day = int(context["params"]["day_number"])

    # Simulated comparison
    old_auc = 0.0167  # The collapsed v3 model's best AUC-PR
    improvement = ((new_auc - old_auc) / max(old_auc, 0.001)) * 100

    # Day 39 = bad model deploy scenario
    if day >= 39:
        # Simulate bad model
        new_auc = 0.005
        improvement = ((new_auc - old_auc) / max(old_auc, 0.001)) * 100

    canary_result = {
        "old_model_auc_pr": old_auc,
        "new_model_auc_pr": new_auc,
        "improvement_pct": round(improvement, 1),
        "canary_passed": improvement > -10,  # Allow up to 10% degradation
    }

    result_path = Path(DATA_DIR) / "canary_results" / f"canary_day_{day:02d}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(canary_result, f, indent=2)

    context["ti"].xcom_push(key="canary_passed", value=canary_result["canary_passed"])

    if canary_result["canary_passed"]:
        print(f"[CANARY] PASSED: +{improvement:.1f}% improvement → promote model")
        return "promote_model"
    else:
        print(f"[CANARY] FAILED: {improvement:.1f}% degradation → rollback!")
        return "rollback_model"


canary_eval = BranchPythonOperator(
    task_id="canary_evaluation",
    python_callable=_canary_evaluation,
    dag=dag,
)


# ─── TASK 5a: Promote model ─────────────────────────────────
def _promote_model(**context):
    day = int(context["params"]["day_number"])
    print(f"[DEPLOY] Day {day}: New model promoted to production")
    # In real production: update SageMaker endpoint, update model registry


promote_model = PythonOperator(
    task_id="promote_model",
    python_callable=_promote_model,
    dag=dag,
)


# ─── TASK 5b: Rollback model ────────────────────────────────
def _rollback_model(**context):
    day = int(context["params"]["day_number"])
    print(f"[ROLLBACK] Day {day}: Canary failed! Rolling back to previous model")
    # In real production: revert SageMaker endpoint, alert on-call


rollback_model = PythonOperator(
    task_id="rollback_model",
    python_callable=_rollback_model,
    dag=dag,
)


# ─── TASK 6: Log completion ─────────────────────────────────
log_complete = PythonOperator(
    task_id="log_retrain_result",
    python_callable=lambda **ctx: print(f"[DONE] Retrain pipeline day {ctx['params']['day_number']} complete"),
    trigger_rule="none_failed_min_one_success",
    dag=dag,
)


# ─── DAG FLOW ────────────────────────────────────────────────
determine_window >> retrain_etl >> simulate_training >> canary_eval
canary_eval >> [promote_model, rollback_model]
[promote_model, rollback_model] >> log_complete
