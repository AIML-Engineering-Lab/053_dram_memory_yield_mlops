#!/usr/bin/env python3
"""Build the P053 40-day production learning report as PDF-ready HTML."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "P053_40_Day_Production_Learning_Report.html"

ASSETS = [
    ("../assets/p53_37_simulation_summary.png", "40-day production run summary"),
    ("../assets/p53_33_drift_timeline.png", "Drift timeline heatmap"),
    ("../assets/p53_34_retrain_story.png", "Retrain, canary, and rollback story"),
    ("../assets/p53_38_psi_waterfall.png", "PSI feature waterfall"),
    ("../assets/p53_28_architecture.png", "End-to-end architecture"),
    ("../assets/p53_30_deployment_stack.png", "Deployment stack"),
    ("../assets/p53_39_a100_training_results.png", "A100 training results"),
    ("../assets/p53_40_hardware_benchmark.png", "Hardware benchmark"),
    ("../assets/p53_31_cost_analysis.png", "Cost analysis"),
    ("../assets/p53_41_a100_shap_importance.png", "Feature importance"),
]

LIVE_AWS_RECOVERY_EVENTS = [
    {
        "date": "2026-07-01",
        "area": "Day 30 retrain orchestration",
        "symptom": "Day 30 drift was real, but the retrain DAG did not complete and the champion stayed on Day 1.",
        "root_cause": "TriggerDagRunOperator triggered the retrain DAG without waiting for completion; EC2 cleanup could stop the host while retraining was still expected.",
        "fix": "Added wait_for_completion=True and poke_interval=60; retrain results now upload to S3 and update pipeline_state.json on canary pass.",
        "prevention": "Production state is updated only after canary-confirmed promotion; retrain and canary artifacts are stored in S3.",
    },
    {
        "date": "2026-07-01",
        "area": "GitHub Actions timeout",
        "symptom": "Manual Day 30 re-run was cancelled around the 260-minute mark.",
        "root_cause": "Workflow timeout was too short for a full GPU retrain day.",
        "fix": "Raised daily_pipeline.yml timeout-minutes to 480.",
        "prevention": "Retrain days now have an 8-hour envelope for ETL, training, canary, artifact sync, and cleanup.",
    },
    {
        "date": "2026-07-03",
        "area": "Spark ETL memory pressure",
        "symptom": "Day 31 failed with Airflow zombie detection for spark_etl.",
        "root_cause": "Spark local[2] with a 2 GB driver plus Docker services exceeded practical memory headroom on g4dn.xlarge.",
        "fix": "Changed Spark tasks to local[1], 1 GB driver memory, maxResultSize=256m, shuffle.partitions=4, retries=2, and execution_timeout=45m.",
        "prevention": "Spark now uses a smaller footprint and Airflow has retry/timeout controls instead of waiting on dead processes.",
    },
    {
        "date": "2026-07-04",
        "area": "EC2 disk saturation",
        "symptom": "Day 31 cron failed during source sync with tar: No space left on device.",
        "root_cause": "Anonymous Docker volumes accumulated Kafka/Airflow data across repeated daily runs and consumed 78 GB; root volume reached 100%.",
        "fix": "Deleted 26 anonymous Docker volumes, cleared local data already uploaded to S3, restarted containerd/Docker, and pruned Docker resources.",
        "prevention": "Created /etc/cron.daily/docker-cleanup and kept S3 as the durable artifact store.",
    },
    {
        "date": "2026-07-04",
        "area": "Day 32 accidental retrain trigger",
        "symptom": "Day 32 finished ETL and drift detection, then blocked in trigger_retrain while the child retrain DAG stayed queued.",
        "root_cause": "The daily gate only checked day >= 30, so drift after Day 30 could retrigger training immediately; the retrain DAG was also paused, leaving TriggerDagRunOperator waiting.",
        "fix": "Marked the accidental Day 32 child retrain run successful to unblock the parent daily DAG, then added a champion_updated_day cooldown read from S3 pipeline_state.json.",
        "prevention": "Days after a champion promotion now skip retrain until the model reaches the configured staleness window; queued child retrains are no longer part of normal Day 31-39 flow.",
    },
    {
        "date": "2026-07-04",
        "area": "Kafka container memory",
        "symptom": "Kafka exited with code 137 during the Day 32 recovery window and Day 33 memory climbed toward the 1 GB container cap.",
        "root_cause": "The Confluent Kafka JVM had a container memory limit but no explicit heap cap, so the broker could be killed under publish pressure.",
        "fix": "Raised the live Day 33 Kafka container limit to 1.5 GB and committed KAFKA_HEAP_OPTS=-Xms256m -Xmx512m with restart: unless-stopped for future runs.",
        "prevention": "Future dispatches run Kafka with bounded heap and automatic restart behavior while leaving enough host memory for Airflow and Spark.",
    },
    {
        "date": "2026-07-11",
        "area": "Final Day 40 closure",
        "symptom": "The live AWS daily pipeline completed all intended 40 days, but project reports and trackers still reflected the Day 33 interim recovery point.",
        "root_cause": "The recovery report was intentionally generated mid-run and had not yet been refreshed after the scheduled Day 34-40 GitHub Actions runs completed.",
        "fix": "Audited S3 pipeline_state.json, Day 29-40 artifacts, GitHub Actions run history, and AWS resource state; refreshed the report to Day 40 complete with the Day 30 v2 champion still active.",
        "prevention": "Treat S3 pipeline_state.json as the final source of truth before publishing status documents or cost-closeout notes.",
    },
    {
        "date": "2026-07-17",
        "area": "Post-Day40 schedule guard",
        "symptom": "Scheduled GitHub Actions runs continued after Day 40 and generated real Day 41-45 Parquet, drift, and summary artifacts.",
        "root_cause": "The completion check used exit 0 inside one step, but GitHub Actions continued to later steps that started RDS/EC2 and triggered Airflow.",
        "fix": "Disabled the cron schedule and added a should_run guard to every AWS-starting or Airflow-triggering workflow step.",
        "prevention": "Future runs require explicit workflow_dispatch, and complete/day>40 state skips expensive AWS steps instead of only logging completion.",
    },
]

DECISIONS = [
    ("AUC-PR as primary metric", "At roughly 1:159 defect imbalance, AUC-ROC can look healthy while missing the minority class. AUC-PR directly measures ranking quality for defects."),
    ("HybridTransformerCNN", "Per-feature tokenization models heterogeneous DRAM parameters; CNN layers capture spatial wafer patterns; fusion MLP combines process, test, and location signals."),
    ("Focal loss", "The model must learn from rare defects without synthetic oversampling noise. Focal loss emphasizes hard minority examples."),
    ("bfloat16 on A100", "float16 with GradScaler collapsed under focal-loss gradients; bfloat16 preserves exponent range and trained stably without GradScaler."),
    ("T4 for production retrains", "The 317K-parameter model fits comfortably on T4. A100 is reserved for very large training/data volumes."),
    ("Staleness gate", "Early drift is tagged and monitored, but retraining is blocked until the model has enough production age to avoid thrashing."),
    ("Canary before promotion", "A retrained model becomes champion only after canary evaluation; Day 39 deliberately tested rollback behavior."),
    ("S3 as durable system of record", "Local EC2 files are disposable. Models, data, drift reports, canary results, and pipeline state live on S3."),
]

COST_ROWS = [
    ("AWS EC2 g4dn.xlarge", "$0.526/hour", "T4 GPU host for Airflow-triggered retrains and production runs", "~$2.10 for 4 hours; ~$52.60 for 100 hours"),
    ("AWS RDS db.t3.micro", "~$0.018/hour", "PostgreSQL backend for MLflow when AWS is active", "~$13/month if left running; stopped between daily runs to reduce idle cost"),
    ("S3 artifacts", "Low single-digit USD/month at this scale", "Models, Parquet, drift reports, metrics, checkpoints", "40-day Parquet footprint in simulation: ~8.5 GB plus models/JSON"),
    ("Final residual AWS resources", "Billable until deleted or expired", "125 GiB gp3 EBS; 20 GiB RDS; 21 automated RDS snapshots; ~4.80 GB current S3 data with 387 object versions; 1 ECR repo; 1 associated public IPv4/EIP", "EC2/RDS/NAT compute is stopped. User chose to keep minimal evidence resources for now; Day 41-45 cleanup remains optional."),
    ("Colab fallback T4", "~1.36 CU/hour", "Fallback retraining if AWS GPU is unavailable", "100 CU pack was the budget choice for multiple T4 runs"),
    ("Colab A100", "~6.79 CU/hour", "Initial high-speed training and large simulation fallback", "Useful for >1 TB/day or heavy experimentation"),
    ("Estimated business upside", "$36M/year", "From avoided escaped DRAM defects", "67 defects/month avoided x $45K x 12 months"),
]


def load_json(relative_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / relative_path
    return json.loads(path.read_text()) if path.exists() else {}


def fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "-"


def fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def cell(value: Any) -> str:
    return escape("-" if value is None else str(value))


def badge(label: str, kind: str = "neutral") -> str:
    return f'<span class="badge badge-{kind}">{escape(label)}</span>'


def event_label(events: list[str]) -> str:
    joined = " ".join(events)
    if "ROLLBACK" in joined:
        return badge("Rollback", "danger")
    if "CANARY_FAILED" in joined:
        return badge("Canary failed", "danger")
    if "RETRAIN_TRIGGERED" in joined:
        return badge("Retrain", "accent")
    if "retrain_blocked" in joined:
        return badge("Blocked by staleness", "warn")
    if "drift_critical" in joined:
        return badge("Critical drift", "warn")
    if "drift_warning" in joined:
        return badge("Warning", "warn")
    if "drift_clean" in joined:
        return badge("Clean", "good")
    return badge("Baseline", "neutral")


def max_psi(day: dict[str, Any]) -> float | None:
    values = list(((day.get("drift") or {}).get("feature_psi") or {}).values())
    return max(values, default=None)


def dominant_feature(day: dict[str, Any]) -> str:
    values = (day.get("drift") or {}).get("feature_psi") or {}
    if not values:
        return "-"
    feature_name, feature_value = max(values.items(), key=lambda item: item[1])
    return f"{feature_name} ({fmt_float(feature_value, 2)})"


def scenario_story(day: dict[str, Any]) -> tuple[str, str]:
    scenario = day.get("scenario", "steady")
    critical = (day.get("drift") or {}).get("features_critical")
    if scenario == "steady":
        return "Normal production baseline", "Data was generated and uploaded with no drift report required in the warm-up window."
    if scenario == "false_alarm":
        return "False-alarm drift test", "Two features spiked, but the retrain gate held because the overall condition was not strong enough for promotion risk."
    if scenario == "auto_recover":
        return "Auto-recovery check", "The previous abnormal shift disappeared, confirming the monitor can distinguish transient movement from durable drift."
    if scenario == "gradual_drift":
        return "Gradual drift accumulation", f"Drift increased slowly; {critical or 0} critical features were tracked without immediate retraining while the staleness gate matured."
    if scenario in {"sudden_shift", "threshold_1", "threshold_2", "continued_drift", "worsening"}:
        return "Persistent process drift", f"The pipeline recorded {critical or 0} critical features and preserved the evidence needed for the Day 30 retrain decision."
    if scenario == "retrain_trigger":
        return "Post-promotion guardrail", "The newly promoted v2 model served inference; retraining was blocked again because the model age was only one day."
    if scenario == "post_retrain_recovery":
        return "Recovery monitoring", "The system monitored whether v2 reduced production risk while continuing to tag residual drift."
    if scenario == "second_drift":
        return "Second drift wave", "A second wave of drift was observed, but the staleness gate prevented excessive retraining."
    if scenario == "bad_model_deploy":
        return "Canary rollback test", "A deliberately bad candidate failed canary evaluation and the system rolled back to v2 automatically."
    if scenario == "final_recovery":
        return "Recovered final state", "The production run finished with v2 active and rollback behavior proven."
    return scenario.replace("_", " ").title(), "Daily pipeline completed with artifacts uploaded to S3."


def image_card(src: str, title: str) -> str:
    image_path = PROJECT_ROOT / src.replace("../", "")
    if not image_path.exists():
        return ""
    return f"""
      <figure class="image-card">
        <img src="{escape(src)}" alt="{escape(title)}"/>
        <figcaption>{escape(title)}</figcaption>
      </figure>
    """


def kpi_card(label: str, value: str, note: str, tone: str = "blue") -> str:
    return f"""
      <div class="kpi kpi-{tone}">
        <div class="kpi-label">{escape(label)}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{escape(note)}</div>
      </div>
    """


def build_master_table(days: list[dict[str, Any]], rows_per_day: int) -> str:
    rows = []
    for day in days:
        drift = day.get("drift") or {}
        events = day.get("events", [])
        rows.append(
            "<tr>"
            f"<td>{day['day']}</td>"
            f"<td>{cell(day.get('date'))}</td>"
            f"<td>{fmt_int(rows_per_day)}</td>"
            f"<td>{fmt_float(day.get('parquet_mb'), 1)}</td>"
            f"<td>{cell(day.get('scenario', '').replace('_', ' '))}</td>"
            f"<td>{cell(day.get('model_version'))}</td>"
            f"<td>{drift.get('features_critical', '-')}</td>"
            f"<td>{drift.get('features_warning', '-')}</td>"
            f"<td>{fmt_float(max_psi(day), 2)}</td>"
            f"<td>{cell(dominant_feature(day))}</td>"
            f"<td>{event_label(events)}</td>"
            f"<td>{cell(day.get('s3_uploaded'))}</td>"
            f"<td>{fmt_float(day.get('elapsed_sec'), 1)}</td>"
            "</tr>"
        )
    return """
      <table class="dense-table master-table">
        <thead>
          <tr>
            <th>Day</th><th>Date</th><th>Rows</th><th>Parquet MB</th><th>Scenario</th><th>Model</th>
            <th>Crit</th><th>Warn</th><th>Max PSI</th><th>Dominant PSI Feature</th><th>Gate/Event</th><th>S3 Files</th><th>Elapsed Sec</th>
          </tr>
        </thead>
        <tbody>
    """ + "\n".join(rows) + """
        </tbody>
      </table>
    """


def build_day_cards(days: list[dict[str, Any]], rows_per_day: int) -> str:
    cards = []
    for day in days:
        drift = day.get("drift") or {}
        title, narrative = scenario_story(day)
        events = ", ".join(day.get("events", [])) or "s3 upload"
        detail_bits = [
            f"Rows: {fmt_int(rows_per_day)}",
            f"Parquet: {fmt_float(day.get('parquet_mb'), 1)} MB",
            f"Model: {day.get('model_version', '-')}",
            f"S3 files: {day.get('s3_uploaded', '-')}",
        ]
        if drift:
            detail_bits.extend(
                [
                    f"Critical features: {drift.get('features_critical', '-')}",
                    f"Warning features: {drift.get('features_warning', '-')}",
                    f"Max PSI: {fmt_float(max_psi(day), 2)}",
                ]
            )
        cards.append(
            f"""
            <section class="day-card">
              <div class="day-card-head">
                <div><strong>Day {day['day']}</strong> <span>{escape(day.get('date', ''))}</span></div>
                {event_label(day.get('events', []))}
              </div>
              <h4>{escape(title)}</h4>
              <p>{escape(narrative)}</p>
              <p class="micro"><strong>Numbers:</strong> {escape(' | '.join(detail_bits))}</p>
              <p class="micro"><strong>Events:</strong> {escape(events)}</p>
            </section>
            """
        )
    return "\n".join(cards)


def table_from_rows(headers: list[str], rows: list[list[Any]], class_name: str = "") -> str:
    header_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    row_html = "\n".join(
        "<tr>" + "".join(f"<td>{cell(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<table class="{class_name}"><thead><tr>{header_html}</tr></thead><tbody>{row_html}</tbody></table>'


def build_html(output_path: Path) -> str:
    timeline = load_json("data/simulation_timeline.json")
    a100 = load_json("data/benchmark_A100-day1-initial.json") or load_json("data/benchmark_a100.json")
    t4 = load_json("data/benchmark_t4.json")
    data_profile = load_json("data/data_profile.json")
    days = timeline.get("days", [])
    rows_per_day = int(timeline.get("rows_per_day", 0))
    total_rows = rows_per_day * len(days)
    total_parquet_mb = sum(float(day.get("parquet_mb", 0)) for day in days)
    total_s3_files = sum(int(day.get("s3_uploaded", 0)) for day in days)
    elapsed_total = float(timeline.get("total_elapsed_min") or sum(float(day.get("elapsed_sec", 0)) for day in days) / 60)
    events = Counter(event for day in days for event in day.get("events", []))
    scenarios = Counter(day.get("scenario", "unknown") for day in days)
    drift_days = [day for day in days if day.get("drift")]
    critical_days = [day for day in days if (day.get("drift") or {}).get("features_critical", 0) >= 3]
    retrain_days = [day for day in days if "RETRAIN_TRIGGERED" in day.get("events", [])]
    rollback_days = [day for day in days if "ROLLBACK_TO_v2" in day.get("events", [])]

    train_profile = data_profile.get("train", {})
    a100_val = (a100.get("results") or {}).get("val", {})
    a100_test = (a100.get("results") or {}).get("test", {})
    a100_unseen = (a100.get("results") or {}).get("unseen", {})
    t4_val = (t4.get("results") or {}).get("val", {})
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    output_relative = output_path.relative_to(PROJECT_ROOT) if output_path.is_relative_to(PROJECT_ROOT) else output_path

    kpis = [
        kpi_card("Production simulation", f"{len(days)} days", "Feb 20 - Mar 31, 2026", "blue"),
        kpi_card("Rows processed", fmt_int(total_rows), f"{fmt_int(rows_per_day)} rows/day", "green"),
        kpi_card("Parquet output", f"{fmt_float(total_parquet_mb / 1024, 2)} GB", f"{fmt_float(total_parquet_mb, 1)} MB total", "blue"),
        kpi_card("Wall clock", f"{fmt_float(elapsed_total, 1)} min", "End-to-end 40-day A100 simulation", "green"),
        kpi_card("Drift reports", str(len(drift_days)), f"{len(critical_days)} critical drift days", "warn"),
        kpi_card("Retrains", str(len(retrain_days)), "Day 30 retrain -> v2", "accent"),
        kpi_card("Rollbacks", str(len(rollback_days)), "Day 39 canary failure rollback", "danger"),
        kpi_card("S3 uploads", fmt_int(total_s3_files), "Daily artifacts persisted", "blue"),
        kpi_card("Champion params", fmt_int(a100.get("model_params")), f"A100 val AUC-PR {fmt_float(a100_val.get('auc_pr'), 4)}", "green"),
        kpi_card("Infra recovery", "85 GB free", "EC2 disk after Docker volume cleanup", "accent"),
    ]

    model_rows = [
        ["A100 Val", fmt_float(a100_val.get("auc_pr"), 4), fmt_float(a100_val.get("auc_roc"), 4), fmt_float(a100_val.get("f1"), 4), pct(a100_val.get("recall")), pct(a100_val.get("precision")), fmt_int(a100_val.get("tp")), fmt_int(a100_val.get("fp")), fmt_int(a100_val.get("fn")), fmt_int(a100_val.get("tn"))],
        ["A100 Test", fmt_float(a100_test.get("auc_pr"), 4), fmt_float(a100_test.get("auc_roc"), 4), fmt_float(a100_test.get("f1"), 4), pct(a100_test.get("recall")), pct(a100_test.get("precision")), fmt_int(a100_test.get("tp")), fmt_int(a100_test.get("fp")), fmt_int(a100_test.get("fn")), fmt_int(a100_test.get("tn"))],
        ["A100 Unseen", fmt_float(a100_unseen.get("auc_pr"), 4), fmt_float(a100_unseen.get("auc_roc"), 4), fmt_float(a100_unseen.get("f1"), 4), pct(a100_unseen.get("recall")), pct(a100_unseen.get("precision")), fmt_int(a100_unseen.get("tp")), fmt_int(a100_unseen.get("fp")), fmt_int(a100_unseen.get("fn")), fmt_int(a100_unseen.get("tn"))],
        ["T4 Val", fmt_float(t4_val.get("auc_pr"), 4), fmt_float(t4_val.get("auc_roc"), 4), fmt_float(t4_val.get("f1"), 4), pct(t4_val.get("recall")), pct(t4_val.get("precision")), fmt_int(t4_val.get("tp")), fmt_int(t4_val.get("fp")), fmt_int(t4_val.get("fn")), fmt_int(t4_val.get("tn"))],
    ]
    incident_rows = [[item["date"], item["area"], item["symptom"], item["root_cause"], item["fix"], item["prevention"]] for item in LIVE_AWS_RECOVERY_EVENTS]
    decision_rows = [[title, reason] for title, reason in DECISIONS]
    cost_rows = [[item, unit, role, note] for item, unit, role, note in COST_ROWS]
    scenario_rows = [[name.replace("_", " "), count] for name, count in scenarios.most_common()]
    event_rows = [[name, count] for name, count in events.most_common(14)]
    image_grid = "\n".join(image_card(src, title) for src, title in ASSETS)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>P053 40-Day Production Learning Report</title>
<style>
:root {{ --ink:#172033; --muted:#5c667a; --line:#d9e1ee; --paper:#fff; --soft:#f5f8fc; --blue:#2457a6; --green:#1b7f5a; --amber:#a56613; --red:#aa2e35; --accent:#6e4aa8; }}
@page {{ size:A4; margin:1.45cm 1.25cm; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:"Aptos","Segoe UI",Arial,sans-serif; color:var(--ink); background:var(--paper); line-height:1.45; }}
a {{ color:var(--blue); }}
.page {{ max-width:1180px; margin:0 auto; padding:28px; }}
.cover {{ min-height:88vh; display:grid; align-content:center; border-bottom:4px solid var(--blue); page-break-after:always; }}
.eyebrow {{ text-transform:uppercase; letter-spacing:2.5px; color:var(--blue); font-size:12px; font-weight:800; }}
h1 {{ font-size:42px; line-height:1.05; margin:12px 0; color:#102447; }}
.subtitle {{ font-size:18px; color:var(--muted); max-width:900px; }}
.cover-grid,.two-col {{ display:grid; grid-template-columns:1.2fr .8fr; gap:24px; margin-top:28px; }}
.cover-box {{ border:1px solid var(--line); background:var(--soft); padding:18px; border-radius:10px; }}
.cover-box h3 {{ margin:0 0 10px; font-size:15px; }}
.toc a {{ display:block; padding:4px 0; text-decoration:none; }}
h2 {{ font-size:26px; margin:34px 0 12px; color:#123a72; border-bottom:2px solid var(--line); padding-bottom:7px; page-break-after:avoid; }}
h3 {{ font-size:18px; margin:22px 0 8px; color:#17335f; page-break-after:avoid; }}
h4 {{ margin:8px 0 4px; font-size:14px; }}
p {{ margin:8px 0; }}
.lead {{ font-size:16px; color:var(--muted); }}
.kpi-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:16px 0 22px; }}
.kpi {{ border:1px solid var(--line); border-top:5px solid var(--blue); border-radius:8px; padding:12px; background:#fff; min-height:104px; page-break-inside:avoid; }}
.kpi-green {{ border-top-color:var(--green); }} .kpi-warn {{ border-top-color:var(--amber); }} .kpi-danger {{ border-top-color:var(--red); }} .kpi-accent {{ border-top-color:var(--accent); }}
.kpi-label {{ color:var(--muted); font-weight:800; font-size:11px; text-transform:uppercase; letter-spacing:.8px; }}
.kpi-value {{ font-size:24px; font-weight:900; margin-top:6px; }}
.kpi-note {{ color:var(--muted); font-size:12px; margin-top:4px; }}
.callout {{ border-left:5px solid var(--blue); background:#f4f8ff; padding:12px 16px; border-radius:0 8px 8px 0; margin:12px 0; }}
.callout.warn {{ border-left-color:var(--amber); background:#fff8ec; }} .callout.good {{ border-left-color:var(--green); background:#effaf5; }}
table {{ width:100%; border-collapse:collapse; margin:12px 0 20px; font-size:13px; }}
th {{ background:#edf3fb; color:#123a72; text-align:left; font-weight:800; border:1px solid var(--line); padding:8px; }}
td {{ border:1px solid var(--line); padding:7px 8px; vertical-align:top; }}
tbody tr:nth-child(even) td {{ background:#fbfcfe; }}
.dense-table {{ font-size:11px; }} .incident-table {{ font-size:10.5px; }}
.badge {{ display:inline-block; border-radius:999px; padding:3px 8px; font-weight:800; font-size:10.5px; white-space:nowrap; }}
.badge-good {{ background:#e8f7ef; color:var(--green); }} .badge-warn {{ background:#fff0d5; color:var(--amber); }} .badge-danger {{ background:#ffe8e9; color:var(--red); }} .badge-accent {{ background:#f0e8ff; color:var(--accent); }} .badge-neutral {{ background:#eef2f7; color:#4e5b70; }}
.image-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
.image-card {{ margin:0; border:1px solid var(--line); border-radius:10px; overflow:hidden; background:#fff; page-break-inside:avoid; }}
.image-card img {{ display:block; width:100%; }} .image-card figcaption {{ padding:8px 10px; color:var(--muted); font-size:12px; background:#f8fafc; border-top:1px solid var(--line); }}
.flow {{ display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin:12px 0 20px; }}
.flow-step {{ border:1px solid var(--line); background:#fff; border-radius:8px; padding:10px; text-align:center; font-size:12px; }}
.flow-step strong {{ display:block; color:var(--blue); margin-bottom:4px; }}
.day-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }}
.day-card {{ border:1px solid var(--line); border-radius:9px; padding:12px; background:#fff; page-break-inside:avoid; }}
.day-card-head {{ display:flex; align-items:center; justify-content:space-between; gap:8px; color:var(--muted); font-size:12px; }}
.day-card-head strong {{ color:var(--ink); font-size:14px; }}
.micro {{ font-size:12px; color:var(--muted); }}
.print-note {{ color:var(--muted); font-size:12px; margin-top:24px; border-top:1px solid var(--line); padding-top:10px; }}
.page-break {{ page-break-before:always; }}
@media print {{ .page {{ padding:0; max-width:none; }} .cover {{ min-height:92vh; }} h2 {{ page-break-after:avoid; }} table,figure,.kpi,.day-card,.callout {{ page-break-inside:avoid; }} thead {{ display:table-header-group; }} }}
@media (max-width:860px) {{ .page {{ padding:16px; }} .cover-grid,.two-col,.image-grid,.day-grid {{ grid-template-columns:1fr; }} .kpi-grid {{ grid-template-columns:repeat(2,1fr); }} .flow {{ grid-template-columns:1fr; }} table {{ display:block; overflow-x:auto; }} }}
</style>
</head>
<body>
<main class="page">
  <section class="cover">
    <div class="eyebrow">AIML Engineering Lab | P053</div>
    <h1>DRAM Memory Yield Predictor<br/>40-Day Production Learning Report</h1>
    <p class="subtitle">Interview-ready and PDF-ready operating record for a full MLOps pipeline: 16M-row training, HybridTransformerCNN, Airflow, Kafka, Spark, MLflow, S3, AWS GPU retraining, canary evaluation, rollback, incident recovery, and cost discipline.</p>
    <div class="cover-grid">
      <div class="cover-box"><h3>What this document is</h3><p>A living HTML report that can be regenerated as future production days complete. It starts with high-level KPIs, then gives the full day-by-day ledger, followed by narrative learnings, issues, fixes, decisions, cost, and interview talking points.</p><p><strong>Generated:</strong> {generated_at}<br/><strong>Output:</strong> {escape(str(output_relative))}</p></div>
      <div class="cover-box toc"><h3>Navigation</h3><a href="#kpis">1. Executive KPIs</a><a href="#system">2. System and Model</a><a href="#master-table">3. 40-Day Master Table</a><a href="#daywise">4. Day-by-Day Story</a><a href="#live-ops">5. Live AWS Recovery</a><a href="#costs">6. Cost and Value</a><a href="#decisions">7. Engineering Decisions</a><a href="#visuals">8. Visual Evidence</a></div>
    </div>
  </section>

  <section id="kpis"><h2>1. Executive KPIs</h2><p class="lead">The project demonstrates an end-to-end production MLOps system for DRAM yield prediction. The key point for interviews: this is not just a model result; it is a full operational lifecycle with drift detection, retraining, canary promotion, rollback, cloud cost control, and failure recovery.</p><div class="kpi-grid">{''.join(kpis)}</div><div class="callout good"><strong>Headline:</strong> 200M rows were processed across a 40-day production simulation. Drift accumulated from Day 17, the staleness gate opened on Day 30, v2 was retrained and promoted, and a Day 39 canary failure correctly rolled back to v2.</div></section>

  <section id="system"><h2>2. System and Model</h2><div class="two-col"><div><h3>Dataset and Signal</h3><table><tbody><tr><th>Training rows</th><td>{fmt_int(train_profile.get('rows'))}</td></tr><tr><th>Columns</th><td>{train_profile.get('columns', '-')} total; {train_profile.get('numeric_features', '-')} numeric; {train_profile.get('categorical_features', '-')} categorical</td></tr><tr><th>Fail rate</th><td>{fmt_float(train_profile.get('fail_rate_pct'), 3)}%</td></tr><tr><th>Label noise</th><td>{fmt_int(train_profile.get('noisy_labels'))} noisy labels</td></tr><tr><th>Missing cells</th><td>{fmt_int(train_profile.get('missing_cells'))} ({fmt_float(train_profile.get('missing_pct'), 2)}%)</td></tr><tr><th>Spatial effect</th><td>Edge fail rate {fmt_float(train_profile.get('edge_fail_rate_pct'), 3)}% vs center {fmt_float(train_profile.get('center_fail_rate_pct'), 3)}%; ratio {fmt_float(train_profile.get('spatial_ratio'), 1)}x</td></tr></tbody></table></div><div><h3>Training and Model</h3><table><tbody><tr><th>Architecture</th><td>HybridTransformerCNN</td></tr><tr><th>Parameters</th><td>{fmt_int(a100.get('model_params'))}</td></tr><tr><th>A100 GPU</th><td>{cell(a100.get('gpu_name'))}, {fmt_float(a100.get('gpu_vram_gb'), 1)} GB VRAM</td></tr><tr><th>A100 throughput</th><td>{fmt_int(a100.get('throughput_samples_per_s'))} samples/sec</td></tr><tr><th>T4 throughput</th><td>{fmt_int(t4.get('throughput_samples_per_s'))} samples/sec</td></tr><tr><th>Mixed precision</th><td>bfloat16 on A100; float16 + GradScaler on T4 when stable</td></tr></tbody></table></div></div><h3>Model Metrics</h3>{table_from_rows(['Split','AUC-PR','AUC-ROC','F1','Recall','Precision','TP','FP','FN','TN'], model_rows, 'dense-table')}<div class="flow"><div class="flow-step"><strong>Generate</strong>Daily DRAM production rows</div><div class="flow-step"><strong>Stream</strong>Kafka producer/consumer path</div><div class="flow-step"><strong>Transform</strong>Spark ETL to Parquet/S3</div><div class="flow-step"><strong>Monitor</strong>PSI drift and staleness gate</div><div class="flow-step"><strong>React</strong>GPU retrain, canary, promote/rollback</div></div></section>

  <section id="master-table" class="page-break"><h2>3. 40-Day Master Table</h2><p class="lead">This table is the operational ledger: rows, data volume, drift counts, model version, gate outcome, artifacts, and runtime for every day.</p>{build_master_table(days, rows_per_day)}<div class="two-col"><div><h3>Scenario Mix</h3>{table_from_rows(['Scenario','Days'], scenario_rows)}</div><div><h3>Event Counts</h3>{table_from_rows(['Event','Count'], event_rows)}</div></div></section>

  <section id="daywise" class="page-break"><h2>4. Day-by-Day Story</h2><p class="lead">The audit table gives the numbers; these notes explain what happened, why each stage mattered, and what a reviewer should understand from the progression.</p><div class="day-grid">{build_day_cards(days, rows_per_day)}</div></section>

    <section id="live-ops" class="page-break"><h2>5. Live AWS Recovery and Current Production Learning</h2><div class="callout warn"><strong>Why this section exists:</strong> The clean 40-day simulation proves the design, while the July AWS daily run shows production reality: timeout budgets, Docker disk growth, Spark memory pressure, S3 state, retry hygiene, schedule guard mistakes, and final cloud shutdown. These are strong interview stories because they show operating judgment, not only modeling skill.</div>{table_from_rows(['Date','Area','Symptom','Root Cause','Fix','Prevention'], incident_rows, 'dense-table incident-table')}<h3>Current State Snapshot</h3><table><tbody><tr><th>Intended production scope</th><td>Day 1-40 complete; Day 30 v2 champion retained through the intended run.</td></tr><tr><th>Current S3 pipeline state</th><td>complete; current_day=46; last_completed_day=45; last_run=2026-07-16T02:05:57Z</td></tr><tr><th>Post-Day40 finding</th><td>Days 41-45 were real unintended scheduled runs caused by a workflow guard bug; cron is now disabled and AWS-starting steps are guarded by should_run.</td></tr><tr><th>Champion model in pipeline_state.json</th><td>s3://p053-mlflow-artifacts/models/day30_v2_retrained.pt</td></tr><tr><th>Artifact status</th><td>Days 29-45 have current-prefix production Parquet, drift reports, and summaries on S3; Days 41-45 are extra artifacts from the schedule leak.</td></tr><tr><th>Known live issues fixed</th><td>Spark zombie/OOM settings reduced; EC2 disk recovered from 100% full; missing reference data now restores from S3; Day 32 accidental retrain waits are blocked by champion cooldown; Kafka heap is capped and restartable; post-Day40 workflow schedule is disabled.</td></tr><tr><th>AWS shutdown status</th><td>EC2 g4dn.xlarge stopped; RDS db.t3.micro stopped; no NAT gateways available or pending.</td></tr><tr><th>Residual billable resources</th><td>125 GiB gp3 EBS volume; 20 GiB RDS storage; 21 automated RDS snapshots; ~4.80 GB current S3 objects with 387 versions; 1 ECR repo; 1 associated public IPv4/EIP.</td></tr><tr><th>Retention decision</th><td>Keep minimal retained AWS evidence resources for now. Day 41-45 extra artifacts can be deleted separately if desired.</td></tr></tbody></table></section>

  <section id="costs"><h2>6. Cost and Value Ledger</h2><p class="lead">The cost story is deliberately practical: use A100 only when its throughput is justified, use T4 for the 317K-parameter production retrains, stop idle RDS/EC2, and keep S3 as durable storage.</p>{table_from_rows(['Cost Item','Unit Cost / Estimate','Role','Planning Note'], cost_rows)}</section>

  <section id="decisions"><h2>7. Engineering Decisions and Interview Talking Points</h2>{table_from_rows(['Decision','Why It Matters'], decision_rows)}<div class="callout good"><strong>Best interview framing:</strong> Start with the business problem and imbalance, then explain the metric choice, the bfloat16 failure/fix, the drift gate, the Day 30 retrain, the Day 39 rollback, and finally the live AWS recovery incidents. That sequence shows both ML depth and production ownership.</div></section>

  <section id="visuals" class="page-break"><h2>8. Visual Evidence</h2><p class="lead">Existing project assets are embedded as evidence for PDF conversion. They keep the report visual without relying on external image generation.</p><div class="image-grid">{image_grid}</div></section>

  <section><h2>9. Update Plan for Future Days</h2><ol><li>Let daily pipeline finish new days and sync artifacts to S3.</li><li>Export or update local JSON artifacts: simulation timeline, live state, and metrics.</li><li>Run <code>python3 scripts/build_40day_learning_report.py</code>.</li><li>Open the HTML in a browser and print to PDF after the final days are complete.</li><li>Add final screenshots from GitHub Actions, S3, MLflow, Airflow, and Grafana if desired.</li></ol><p class="print-note">PDF tip: use browser print, A4, background graphics enabled, margins default or none. The stylesheet repeats table headers and avoids page breaks inside key cards.</p></section>
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the P053 40-day production learning report.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output HTML file path")
    args = parser.parse_args()
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html(output_path))
    print(f"Wrote {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
