"""
P053 — Airflow DAG: 40-Day Production Simulation Master
========================================================
Orchestrates the full 40-day production simulation by triggering
the daily pipeline DAG for each simulated day.

This is the "director" DAG:
    - Loops through days 1-40
    - Triggers p053_daily_yield_pipeline for each day
    - Waits for completion before moving to next day
    - Logs scenario metadata per day

Run this ONCE to execute the entire simulation.
Total runtime: ~2 hours (3 min per simulated day × 40 days)

Usage:
    1. Start the big data stack: docker compose up -d
    2. Open Airflow UI: http://localhost:8888 (admin/admin)
    3. Unpause and trigger 'p053_simulation_master'
    4. Watch the 40-day simulation unfold

OR run from CLI:
    airflow dags trigger p053_simulation_master
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import PythonOperator

TOTAL_DAYS = 40
ROWS_PER_DAY = 5_000_000

# Scenario descriptions for logging
SCENARIOS = {
    1: "steady", 2: "steady", 3: "steady", 4: "steady",
    5: "steady", 6: "steady", 7: "steady", 8: "steady",
    9: "false_alarm", 10: "auto_recover",
    11: "gradual_drift", 12: "gradual_drift", 13: "gradual_drift",
    14: "gradual_drift", 15: "gradual_drift", 16: "gradual_drift",
    17: "gradual_drift", 18: "gradual_drift",
    19: "sudden_shift", 20: "threshold_1",
    21: "continued_drift", 22: "continued_drift", 23: "continued_drift",
    24: "continued_drift", 25: "continued_drift",
    26: "threshold_2",
    27: "worsening", 28: "worsening", 29: "worsening", 30: "worsening",
    31: "RETRAIN_TRIGGER",
    32: "post_retrain", 33: "post_retrain", 34: "post_retrain", 35: "post_retrain",
    36: "second_drift", 37: "second_drift", 38: "second_drift",
    39: "bad_model_deploy", 40: "final_recovery",
}

default_args = {
    "owner": "p053",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 0,
}

dag = DAG(
    dag_id="p053_simulation_master",
    default_args=default_args,
    description="Master orchestrator: 40-day production simulation",
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["p053", "simulation", "master"],
)


def _log_day_start(day, **context):
    scenario = SCENARIOS.get(day, "unknown")
    print(f"\n{'='*60}")
    print(f"SIMULATION DAY {day:>2}/40 — {scenario.upper()}")
    print(f"  Rows: {ROWS_PER_DAY:,}")
    print(f"{'='*60}")


# Build tasks dynamically for all 40 days
prev_task = None

for day in range(1, TOTAL_DAYS + 1):
    # Log start
    log_start = PythonOperator(
        task_id=f"log_day_{day:02d}_start",
        python_callable=_log_day_start,
        op_kwargs={"day": day},
        dag=dag,
    )

    # Trigger daily pipeline
    trigger_daily = TriggerDagRunOperator(
        task_id=f"trigger_day_{day:02d}",
        trigger_dag_id="p053_daily_yield_pipeline",
        conf={"day_number": day, "n_rows": ROWS_PER_DAY},
        wait_for_completion=True,
        poke_interval=30,
        allowed_states=["success"],
        failed_states=["failed"],
        dag=dag,
    )

    log_start >> trigger_daily

    if prev_task:
        prev_task >> log_start

    prev_task = trigger_daily


# Final summary
def _simulation_complete(**context):
    print(f"\n{'='*60}")
    print(f"40-DAY SIMULATION COMPLETE!")
    print(f"  Total rows generated: {TOTAL_DAYS * ROWS_PER_DAY:,}")
    print(f"  Retrain events: Day 31")
    print(f"  Bad deploy: Day 39 → rollback")
    print(f"  Recovery: Day 40")
    print(f"{'='*60}")


final_log = PythonOperator(
    task_id="simulation_complete",
    python_callable=_simulation_complete,
    dag=dag,
)

prev_task >> final_log
