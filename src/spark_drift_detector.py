"""
P053 — Spark-based Drift Detection Engine
==========================================
Computes PSI (Population Stability Index) and summary statistics at scale
using PySpark. This is the distributed equivalent of src/drift_detector.py
but handles 200M rows without memory issues.

Drift Detection Strategy (same as drift_detector.py):
    1. PSI per feature: Divide into 10 quantile bins, compare reference vs analysis
    2. Warning:  PSI > 0.1 (minor shift)
    3. Critical: PSI > 0.2 (major shift)
    4. Retrain gate: ≥ 3 features with PSI > 0.2

Reference Window: Days 1-8 (steady state = training distribution)
Analysis Window: Rolling 7-day window

Usage:
    spark-submit --master local[*] src/spark_drift_detector.py --ref-days 1-8 --analysis-day 15
    spark-submit --master local[*] src/spark_drift_detector.py --ref-days 1-8 --analysis-day 15 --end-day 40
"""

import argparse
import json
import time
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_DIR = str(PROJECT_ROOT / "data" / "production")
DRIFT_REPORT_DIR = PROJECT_ROOT / "data" / "drift_reports"
DRIFT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

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

PSI_WARNING = 0.1
PSI_CRITICAL = 0.2
MIN_DRIFTED_FEATURES = 3
N_BINS = 10
EPSILON = 1e-6  # Prevent log(0) in PSI


def create_spark(app_name: str = "P053_DriftDetector") -> SparkSession:
    """Create SparkSession."""
    return (SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def compute_psi_for_feature(ref_values: np.ndarray,
                            analysis_values: np.ndarray,
                            n_bins: int = N_BINS) -> float:
    """
    Compute PSI between reference and analysis distributions.

    PSI = Σ (P_analysis - P_reference) × ln(P_analysis / P_reference)

    Uses quantile-based binning on the reference distribution.
    """
    # Create bins from reference quantiles
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(ref_values, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Count proportions in each bin
    ref_counts = np.histogram(ref_values, bins=bin_edges)[0]
    analysis_counts = np.histogram(analysis_values, bins=bin_edges)[0]

    ref_proportions = ref_counts / max(ref_counts.sum(), 1) + EPSILON
    analysis_proportions = analysis_counts / max(analysis_counts.sum(), 1) + EPSILON

    # PSI formula
    psi = np.sum(
        (analysis_proportions - ref_proportions) *
        np.log(analysis_proportions / ref_proportions)
    )
    return float(psi)


def detect_drift(spark: SparkSession,
                 ref_start_day: int, ref_end_day: int,
                 analysis_day: int) -> dict:
    """
    Compare analysis day against reference window.

    Returns:
        Dict with per-feature PSI, drift severity, retrain recommendation
    """
    # Load reference data
    ref_paths = [
        str(Path(PRODUCTION_DIR) / f"day_{d:02d}.parquet")
        for d in range(ref_start_day, ref_end_day + 1)
        if (Path(PRODUCTION_DIR) / f"day_{d:02d}.parquet").exists()
    ]
    analysis_path = str(Path(PRODUCTION_DIR) / f"day_{analysis_day:02d}.parquet")

    if not ref_paths:
        raise FileNotFoundError(f"No reference data for days {ref_start_day}-{ref_end_day}")
    if not Path(analysis_path).exists():
        raise FileNotFoundError(f"No data for analysis day {analysis_day}")

    ref_df = spark.read.parquet(*ref_paths)
    analysis_df = spark.read.parquet(analysis_path)

    ref_count = ref_df.count()
    analysis_count = analysis_df.count()

    # Compute PSI per feature
    feature_results = []
    critical_count = 0
    warning_count = 0

    for feat in NUMERIC_FEATURES:
        # Collect feature values (sample if too large for driver memory)
        if ref_count > 2_000_000:
            ref_vals = np.array(
                ref_df.select(feat).sample(fraction=2_000_000/ref_count, seed=42)
                .filter(F.col(feat).isNotNull())
                .rdd.flatMap(lambda x: x).collect()
            )
        else:
            ref_vals = np.array(
                ref_df.select(feat).filter(F.col(feat).isNotNull())
                .rdd.flatMap(lambda x: x).collect()
            )

        if analysis_count > 2_000_000:
            analysis_vals = np.array(
                analysis_df.select(feat).sample(fraction=2_000_000/analysis_count, seed=42)
                .filter(F.col(feat).isNotNull())
                .rdd.flatMap(lambda x: x).collect()
            )
        else:
            analysis_vals = np.array(
                analysis_df.select(feat).filter(F.col(feat).isNotNull())
                .rdd.flatMap(lambda x: x).collect()
            )

        psi = compute_psi_for_feature(ref_vals, analysis_vals)

        severity = "none"
        if psi > PSI_CRITICAL:
            severity = "critical"
            critical_count += 1
        elif psi > PSI_WARNING:
            severity = "warning"
            warning_count += 1

        feature_results.append({
            "feature": feat,
            "psi": round(psi, 6),
            "severity": severity,
            "ref_mean": round(float(np.mean(ref_vals)), 4),
            "ref_std": round(float(np.std(ref_vals)), 4),
            "analysis_mean": round(float(np.mean(analysis_vals)), 4),
            "analysis_std": round(float(np.std(analysis_vals)), 4),
        })

    # Retrain decision
    should_retrain = critical_count >= MIN_DRIFTED_FEATURES

    report = {
        "analysis_day": analysis_day,
        "reference_window": f"days {ref_start_day}-{ref_end_day}",
        "reference_rows": ref_count,
        "analysis_rows": analysis_count,
        "features_critical": critical_count,
        "features_warning": warning_count,
        "features_clean": len(NUMERIC_FEATURES) - critical_count - warning_count,
        "should_retrain": should_retrain,
        "retrain_reason": (
            f"{critical_count} features with PSI > {PSI_CRITICAL} (threshold: {MIN_DRIFTED_FEATURES})"
            if should_retrain else
            f"Only {critical_count} critical features (need {MIN_DRIFTED_FEATURES})"
        ),
        "feature_details": sorted(feature_results, key=lambda x: x["psi"], reverse=True),
    }

    # Print summary
    status = "🔴 RETRAIN" if should_retrain else ("🟡 WARNING" if warning_count > 0 else "🟢 CLEAN")
    print(f"  Day {analysis_day:02d} | {status} | "
          f"Critical: {critical_count} | Warning: {warning_count} | "
          f"Top PSI: {feature_results[0]['feature'] if feature_results else 'N/A'} = "
          f"{max(r['psi'] for r in feature_results):.4f}")

    return report


def run_drift_scan(ref_start_day: int, ref_end_day: int,
                   analysis_start_day: int, analysis_end_day: int) -> list:
    """Run drift detection across a range of analysis days."""
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"P053 DRIFT SCAN — Reference: Days {ref_start_day}-{ref_end_day}")
    print(f"                  Analysis:  Days {analysis_start_day}-{analysis_end_day}")
    print(f"{'='*70}\n")

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    all_reports = []
    retrain_days = []

    for day in range(analysis_start_day, analysis_end_day + 1):
        path = Path(PRODUCTION_DIR) / f"day_{day:02d}.parquet"
        if not path.exists():
            print(f"  Day {day:02d} | SKIPPED (no data)")
            continue

        report = detect_drift(spark, ref_start_day, ref_end_day, day)
        all_reports.append(report)

        if report["should_retrain"]:
            retrain_days.append(day)

        # Save individual report
        report_path = DRIFT_REPORT_DIR / f"drift_day_{day:02d}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

    # Save summary
    summary = {
        "reference_window": f"days {ref_start_day}-{ref_end_day}",
        "analysis_window": f"days {analysis_start_day}-{analysis_end_day}",
        "total_days_scanned": len(all_reports),
        "retrain_triggered_days": retrain_days,
        "retrain_count": len(retrain_days),
        "wall_clock_sec": round(time.time() - t0, 1),
    }

    summary_path = DRIFT_REPORT_DIR / "drift_scan_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DRIFT SCAN COMPLETE")
    print(f"  Days scanned: {len(all_reports)}")
    print(f"  Retrain triggers: {retrain_days if retrain_days else 'None'}")
    print(f"  Reports saved: {DRIFT_REPORT_DIR}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'='*70}")

    spark.stop()
    return all_reports


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark-based drift detection")
    parser.add_argument("--ref-days", type=str, default="1-8",
                        help="Reference window (e.g., '1-8')")
    parser.add_argument("--analysis-day", type=int, required=True,
                        help="First analysis day")
    parser.add_argument("--end-day", type=int, default=None,
                        help="Last analysis day (default: same as --analysis-day)")
    args = parser.parse_args()

    ref_parts = args.ref_days.split("-")
    ref_start = int(ref_parts[0])
    ref_end = int(ref_parts[1])
    end_day = args.end_day or args.analysis_day

    run_drift_scan(ref_start, ref_end, args.analysis_day, end_day)
