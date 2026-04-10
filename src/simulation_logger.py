"""
P053 — Comprehensive Daily Simulation Logger
=============================================
Creates detailed, timestamped log files for each simulation day.

Every decision, selection, metric, and event is logged with:
    - ISO timestamps (down to millisecond)
    - Decision flags with reasoning
    - Infrastructure selections with justification
    - Data volumes and processing stats
    - Drift detection results and reliability flags
    - Retrain decisions with all criteria
    - Canary evaluation results
    - Cost tracking per day

Log files:
    data/logs/day_01.log  — Human-readable, per-day
    data/logs/day_01.json — Machine-readable, per-day
    data/logs/simulation_master.log — Full simulation timeline

Interview talking point:
    "Every production ML system needs audit logs. Our simulation
    logs every decision with timestamps: why drift was flagged,
    why retraining was skipped (staleness gate), which GPU was
    selected and why. This is the observability layer that
    separates demo projects from production systems."
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

LOG_DIR = Path(os.environ.get("P053_LOG_DIR",
    Path(__file__).resolve().parent.parent / "data" / "logs"))


class SimulationDayLogger:
    """Structured logger for a single simulation day."""

    def __init__(self, day: int, phase: str = "phase2"):
        self.day = day
        self.phase = phase
        self.start_time = datetime.now()
        self.events = []

        LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Human-readable log
        self.log_path = LOG_DIR / f"day_{day:02d}.log"
        self.json_path = LOG_DIR / f"day_{day:02d}.json"

        # Setup file logger
        self.logger = logging.getLogger(f"p053.day{day:02d}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        fh = logging.FileHandler(self.log_path, mode="w")
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

        # Console handler: WARNING+ only unless P053_LOG_VERBOSE is set
        # (file handler still captures all INFO/DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO if os.environ.get("P053_LOG_VERBOSE") else logging.WARNING)
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

        self._header()

    def _header(self):
        self.logger.info("=" * 72)
        self.logger.info(f"P053 SIMULATION — DAY {self.day:02d} ({self.phase.upper()})")
        self.logger.info(f"Started: {self.start_time.isoformat()}")
        self.logger.info("=" * 72)

    def _record(self, category: str, event: str, details: dict):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "day": self.day,
            "category": category,
            "event": event,
            **details,
        }
        self.events.append(entry)

    # ── Data Generation ──────────────────────────────────────

    def log_data_generation(self, rows: int, parquet_mb: float,
                            scenario: str, elapsed_sec: float):
        self.logger.info(f"[DATA-GEN] Scenario: {scenario}")
        self.logger.info(f"[DATA-GEN] Rows generated: {rows:,}")
        self.logger.info(f"[DATA-GEN] Parquet size: {parquet_mb:.1f} MB")
        self.logger.info(f"[DATA-GEN] Time: {elapsed_sec:.1f}s")
        self._record("data_generation", "completed", {
            "rows": rows,
            "parquet_mb": parquet_mb,
            "scenario": scenario,
            "elapsed_sec": elapsed_sec,
        })

    # ── Infrastructure Selection ─────────────────────────────

    def log_infra_selection(self, model_params: int, estimated_vram_gb: float,
                            selected_gpu: str, instance_type: str,
                            cost_per_hour: float, needs_switch: bool,
                            reason: str):
        self.logger.info(f"[INFRA] Model params: {model_params:,}")
        self.logger.info(f"[INFRA] Estimated VRAM: {estimated_vram_gb:.1f} GB")
        self.logger.info(f"[INFRA] Selected GPU: {selected_gpu} ({instance_type})")
        self.logger.info(f"[INFRA] Cost: ${cost_per_hour:.2f}/hr")
        if needs_switch:
            self.logger.warning(f"[INFRA] ⚠ INSTANCE SWITCH REQUIRED — {reason}")
        else:
            self.logger.info(f"[INFRA] Current instance sufficient — {reason}")
        self._record("infra_selection", "gpu_selected", {
            "model_params": model_params,
            "estimated_vram_gb": estimated_vram_gb,
            "selected_gpu": selected_gpu,
            "instance_type": instance_type,
            "cost_per_hour": cost_per_hour,
            "needs_switch": needs_switch,
            "reason": reason,
        })

    # ── Kafka ────────────────────────────────────────────────

    def log_kafka(self, status: str, msg_per_sec: float = 0,
                  error: Optional[str] = None):
        if error:
            self.logger.warning(f"[KAFKA] Skipped: {error}")
        else:
            self.logger.info(f"[KAFKA] Published: {msg_per_sec:,.0f} msg/sec")
        self._record("kafka", status, {
            "msg_per_sec": msg_per_sec, "error": error,
        })

    # ── Spark ETL ────────────────────────────────────────────

    def log_spark_etl(self, status: str, rows_per_sec: float = 0,
                      error: Optional[str] = None):
        if error:
            self.logger.warning(f"[SPARK] Skipped: {error}")
        else:
            self.logger.info(f"[SPARK] ETL throughput: {rows_per_sec:,.0f} rows/sec")
        self._record("spark_etl", status, {
            "rows_per_sec": rows_per_sec, "error": error,
        })

    # ── Drift Detection ──────────────────────────────────────

    def log_drift_detection(self, features_critical: int, features_warning: int,
                            feature_psi: dict, drift_reliable: bool,
                            ref_rows: int, analysis_rows: int,
                            low_data_warning: Optional[str] = None):
        self.logger.info(f"[DRIFT] Critical features: {features_critical}")
        self.logger.info(f"[DRIFT] Warning features: {features_warning}")
        self.logger.info(f"[DRIFT] Reliable: {drift_reliable}")
        self.logger.info(f"[DRIFT] Ref rows: {ref_rows:,} | Analysis rows: {analysis_rows:,}")

        if not drift_reliable:
            self.logger.warning(f"[DRIFT] ⚠ LOW-DATA: {low_data_warning}")
            self.logger.warning("[DRIFT] Result tagged for transparency only — NOT actionable")

        for feat, psi in sorted(feature_psi.items(), key=lambda x: x[1], reverse=True):
            level = "CRITICAL" if psi > 0.2 else ("WARNING" if psi > 0.1 else "OK")
            self.logger.info(f"[DRIFT]   {feat}: PSI={psi:.6f} [{level}]")

        self._record("drift_detection", "completed", {
            "features_critical": features_critical,
            "features_warning": features_warning,
            "feature_psi": feature_psi,
            "drift_reliable": drift_reliable,
            "ref_rows": ref_rows,
            "analysis_rows": analysis_rows,
            "low_data_warning": low_data_warning,
        })

    # ── Retrain Decision ─────────────────────────────────────

    def log_retrain_decision(self, should_retrain: bool, reason: str,
                             drift_critical: int = 0,
                             days_since_retrain: int = 0,
                             staleness_blocked: bool = False,
                             low_data_blocked: bool = False):
        if should_retrain:
            self.logger.info(f"[RETRAIN] ✓ TRIGGERED — {reason}")
        else:
            self.logger.info(f"[RETRAIN] ✗ SKIPPED — {reason}")

        flags = {
            "drift_threshold_met": drift_critical >= 3,
            "staleness_gate_passed": days_since_retrain >= 30,
            "data_reliable": not low_data_blocked,
        }
        self.logger.info(f"[RETRAIN] Flags: drift={flags['drift_threshold_met']}, "
                         f"staleness={flags['staleness_gate_passed']}, "
                         f"reliable={flags['data_reliable']}")
        self.logger.info(f"[RETRAIN] Days since last retrain: {days_since_retrain}")

        self._record("retrain_decision", "triggered" if should_retrain else "skipped", {
            "should_retrain": should_retrain,
            "reason": reason,
            "drift_critical": drift_critical,
            "days_since_retrain": days_since_retrain,
            "flags": flags,
        })

    # ── Training Execution ───────────────────────────────────

    def log_training_start(self, model_version: str, gpu: str,
                           batch_size: int, epochs: int):
        self.logger.info(f"[TRAIN] Starting: {model_version}")
        self.logger.info(f"[TRAIN] GPU: {gpu}, Batch: {batch_size}, Epochs: {epochs}")
        self._record("training", "started", {
            "model_version": model_version,
            "gpu": gpu,
            "batch_size": batch_size,
            "epochs": epochs,
        })

    def log_training_complete(self, model_version: str, val_auc_pr: float,
                              val_f1: float, training_time_min: float,
                              mlflow_run_id: str):
        self.logger.info(f"[TRAIN] ✓ Complete: {model_version}")
        self.logger.info(f"[TRAIN] AUC-PR={val_auc_pr:.4f}, F1={val_f1:.4f}")
        self.logger.info(f"[TRAIN] Time: {training_time_min:.1f} min")
        self.logger.info(f"[TRAIN] MLflow run: {mlflow_run_id}")
        self._record("training", "completed", {
            "model_version": model_version,
            "val_auc_pr": val_auc_pr,
            "val_f1": val_f1,
            "training_time_min": training_time_min,
            "mlflow_run_id": mlflow_run_id,
        })

    def log_training_failed(self, error: str):
        self.logger.error(f"[TRAIN] ✗ FAILED: {error}")
        self._record("training", "failed", {"error": error})

    # ── Canary Evaluation ────────────────────────────────────

    def log_canary(self, passed: bool, old_auc: float, new_auc: float,
                   improvement_pct: float, forced_failure: bool = False):
        status = "PASSED" if passed else "FAILED"
        self.logger.info(f"[CANARY] {status}: {old_auc:.4f} → {new_auc:.4f} ({improvement_pct:+.1f}%)")
        if forced_failure:
            self.logger.warning("[CANARY] ⚠ FORCED FAILURE SCENARIO (Day 39 test)")
        self._record("canary", "passed" if passed else "failed", {
            "old_auc_pr": old_auc,
            "new_auc_pr": new_auc,
            "improvement_pct": improvement_pct,
            "forced_failure": forced_failure,
        })

    # ── Rollback ─────────────────────────────────────────────

    def log_rollback(self, from_version: str, to_version: str, reason: str):
        self.logger.warning(f"[ROLLBACK] ⚠ {from_version} → {to_version}")
        self.logger.warning(f"[ROLLBACK] Reason: {reason}")
        self._record("rollback", "executed", {
            "from_version": from_version,
            "to_version": to_version,
            "reason": reason,
        })

    # ── S3 Upload ────────────────────────────────────────────

    def log_s3_upload(self, n_files: int, status: str,
                      error: Optional[str] = None):
        if error:
            self.logger.warning(f"[S3] Upload failed: {error}")
        else:
            self.logger.info(f"[S3] Uploaded {n_files} artifacts")
        self._record("s3_upload", status, {
            "n_files": n_files, "error": error,
        })

    # ── Day Complete ─────────────────────────────────────────

    def finalize(self):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info("=" * 72)
        self.logger.info(f"DAY {self.day:02d} COMPLETE — {elapsed:.1f}s total")
        self.logger.info("=" * 72)

        # Write JSON log
        day_log = {
            "day": self.day,
            "phase": self.phase,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "elapsed_sec": round(elapsed, 1),
            "events": self.events,
        }
        with open(self.json_path, "w") as f:
            json.dump(day_log, f, indent=2, default=str)

        # Append to master log
        master_log = LOG_DIR / "simulation_master.log"
        with open(master_log, "a") as f:
            f.write(f"\n{'='*72}\n")
            f.write(f"DAY {self.day:02d} | {self.phase} | "
                    f"{self.start_time.isoformat()} | {elapsed:.1f}s\n")
            for evt in self.events:
                f.write(f"  [{evt['category']}] {evt['event']} — "
                        f"{evt['timestamp']}\n")
            f.write(f"{'='*72}\n")

        return day_log
