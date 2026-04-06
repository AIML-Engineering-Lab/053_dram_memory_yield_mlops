"""
P053 — PySpark ETL Pipeline for Production DRAM Data
=====================================================
Reads raw Parquet files from data/production/ (or data/landing/)
and performs the SAME preprocessing as preprocess.py but at scale.

Pipeline stages:
    1. INGEST — Read Parquet partition(s) into Spark DataFrame
    2. CLEAN — Handle missing values (median impute for numerics, mode for categoricals)
    3. FEATURE ENGINEERING — Same 7 engineered features as config.py
    4. QUALITY CHECKS — Row counts, null audits, distribution stats
    5. OUTPUT — Write cleaned Parquet to data/processed_spark/

Why Spark not pandas?
    200M rows × 36 features × 8 bytes ≈ 57 GB raw memory.
    pandas needs 3-5× that for operations (170-285 GB).
    Spark processes 200M rows in 2 partitions per worker × 2 workers = ~90 seconds.
    pandas OOM-kills at ~10M rows on a 16 GB machine (see pandas_vs_spark_benchmark.py).

Usage:
    spark-submit --master local[*] src/spark_etl.py --day 1
    spark-submit --master local[*] src/spark_etl.py --day 1 --end-day 40
    spark-submit --master spark://localhost:7077 src/spark_etl.py --day 1 --end-day 40
"""

import argparse
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — Must match config.py exactly
# ═══════════════════════════════════════════════════════════════

# Get project root relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_DIR = str(PROJECT_ROOT / "data" / "production")
LANDING_DIR = str(PROJECT_ROOT / "data" / "landing")
OUTPUT_DIR = str(PROJECT_ROOT / "data" / "processed_spark")
DRIFT_DIR = str(PROJECT_ROOT / "data" / "drift_stats")

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


def create_spark(app_name: str = "P053_DRAM_ETL",
                 master: str = None) -> SparkSession:
    """Create SparkSession with production tuning."""
    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    return (builder
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.parquet.mergeSchema", "true")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


# ═══════════════════════════════════════════════════════════════
# STAGE 1: INGEST
# ═══════════════════════════════════════════════════════════════

def ingest_days(spark: SparkSession, start_day: int, end_day: int,
                source_dir: str = PRODUCTION_DIR) -> DataFrame:
    """Read one or more day Parquet files into a Spark DataFrame."""
    paths = []
    for day in range(start_day, end_day + 1):
        p = Path(source_dir) / f"day_{day:02d}.parquet"
        if p.exists():
            paths.append(str(p))
        else:
            print(f"  [WARN] Missing: {p}")

    if not paths:
        raise FileNotFoundError(f"No Parquet files found for days {start_day}-{end_day}")

    df = spark.read.parquet(*paths)
    n_rows = df.count()
    n_parts = df.rdd.getNumPartitions()
    print(f"  INGEST: {n_rows:>12,} rows | {len(paths)} files | {n_parts} partitions")
    return df


def ingest_landing(spark: SparkSession) -> DataFrame:
    """Read all micro-batch Parquet files from the Kafka consumer landing zone."""
    landing_path = Path(LANDING_DIR)
    paths = list(landing_path.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet files in {LANDING_DIR}")

    df = spark.read.parquet(str(landing_path / "*.parquet"))
    n_rows = df.count()
    print(f"  INGEST (landing): {n_rows:>12,} rows | {len(paths)} files")
    return df


# ═══════════════════════════════════════════════════════════════
# STAGE 2: CLEAN — Missing value imputation
# ═══════════════════════════════════════════════════════════════

def clean_data(df: DataFrame) -> DataFrame:
    """
    Handle missing values using median (numeric) and mode (categorical).
    Same logic as preprocess.py but distributed via Spark.
    """
    # Compute medians for numeric columns with nulls
    numeric_with_nulls = []
    for col in NUMERIC_FEATURES:
        null_count = df.filter(F.col(col).isNull()).count()
        if null_count > 0:
            numeric_with_nulls.append(col)

    if numeric_with_nulls:
        # approxQuantile for median imputation (much faster than exact on 200M rows)
        medians = df.stat.approxQuantile(numeric_with_nulls, [0.5], 0.01)
        for col_name, med_vals in zip(numeric_with_nulls, medians):
            median_val = med_vals[0] if med_vals else 0.0
            df = df.fillna({col_name: median_val})
            print(f"    Imputed {col_name}: median = {median_val:.4f}")

    # Mode imputation for categoricals
    for col_name in CATEGORICAL_FEATURES:
        null_count = df.filter(F.col(col_name).isNull()).count()
        if null_count > 0:
            mode_row = (df.filter(F.col(col_name).isNotNull())
                        .groupBy(col_name)
                        .count()
                        .orderBy(F.desc("count"))
                        .first())
            mode_val = mode_row[0] if mode_row else "UNKNOWN"
            df = df.fillna({col_name: mode_val})
            print(f"    Imputed {col_name}: mode = {mode_val}")

    cleaned_nulls = sum(
        df.filter(F.col(c).isNull()).count()
        for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES + SPATIAL_FEATURES
    )
    print(f"  CLEAN: {len(numeric_with_nulls)} numeric cols imputed | "
          f"{cleaned_nulls} remaining nulls")
    return df


# ═══════════════════════════════════════════════════════════════
# STAGE 3: FEATURE ENGINEERING — Same 7 features as preprocess.py
# ═══════════════════════════════════════════════════════════════

def add_engineered_features(df: DataFrame) -> DataFrame:
    """
    Add the same 7 engineered features from config.py:
        retention_temp_interaction, leakage_retention_ratio, edge_risk,
        power_ratio, ecc_burden, timing_margin, rh_risk_composite
    """
    df = (df
        # 1. retention_temp_interaction — Physics: retention degrades exponentially with temp
        .withColumn("retention_temp_interaction",
                    F.col("retention_time_ms") * F.col("test_temp_c"))
        # 2. leakage_retention_ratio — High leakage + low retention = failing die
        .withColumn("leakage_retention_ratio",
                    F.col("cell_leakage_fa") / F.greatest(F.col("retention_time_ms"), F.lit(0.001)))
        # 3. edge_risk — Edge dies fail 7× more (edge_distance near 0 = center, near 1 = edge)
        .withColumn("edge_risk",
                    F.col("edge_distance") * F.col("cell_leakage_fa"))
        # 4. power_ratio — Active-to-standby power ratio indicates thermal stress
        .withColumn("power_ratio",
                    F.col("idd4_active_ma") / F.greatest(F.col("idd2p_standby_ma"), F.lit(0.001)))
        # 5. ecc_burden — Combines error rate with correction overhead
        .withColumn("ecc_burden",
                    F.col("correctable_errors_per_1m") * F.col("ecc_syndrome_entropy"))
        # 6. timing_margin — How close to spec limit (lower = more risky)
        .withColumn("timing_margin",
                    (F.lit(18.0) - F.col("trcd_ns")) + (F.lit(18.0) - F.col("trp_ns")))
        # 7. rh_risk_composite — Row hammer risk with disturb margin context
        .withColumn("rh_risk_composite",
                    F.col("rh_susceptibility") * (F.lit(250.0) - F.col("disturb_margin_mv")) / F.lit(250.0))
    )

    print("  FEATURE ENG: 7 engineered features added")
    return df


# ═══════════════════════════════════════════════════════════════
# STAGE 4: QUALITY CHECKS
# ═══════════════════════════════════════════════════════════════

def quality_checks(df: DataFrame, day_label: str = "all") -> dict:
    """Run data quality checks and return stats."""
    n_rows = df.count()

    # Fail rate per day
    if "day_number" in df.columns:
        fail_stats = (df.groupBy("day_number")
                      .agg(
                          F.count("*").alias("total"),
                          F.sum("is_fail").alias("n_fail"),
                          F.mean("is_fail").alias("fail_rate"),
                      )
                      .orderBy("day_number")
                      .collect())
    else:
        fail_stats = []

    # Null audit
    null_counts = {}
    for col in df.columns:
        nc = df.filter(F.col(col).isNull()).count()
        if nc > 0:
            null_counts[col] = nc

    # Key distribution stats
    stats = (df.select(
        F.mean("test_temp_c").alias("mean_temp"),
        F.stddev("test_temp_c").alias("std_temp"),
        F.mean("cell_leakage_fa").alias("mean_leakage"),
        F.mean("retention_time_ms").alias("mean_retention"),
        F.mean("is_fail").alias("fail_rate"),
    ).collect()[0])

    report = {
        "day_label": day_label,
        "total_rows": n_rows,
        "fail_rate": float(stats["fail_rate"]) if stats["fail_rate"] else 0.0,
        "mean_temp": float(stats["mean_temp"]) if stats["mean_temp"] else 0.0,
        "mean_leakage": float(stats["mean_leakage"]) if stats["mean_leakage"] else 0.0,
        "null_columns": len(null_counts),
        "per_day_stats": [
            {"day": r["day_number"], "total": r["total"],
             "n_fail": r["n_fail"], "fail_rate": float(r["fail_rate"])}
            for r in fail_stats
        ],
    }

    print(f"  QUALITY: {n_rows:,} rows | fail_rate={report['fail_rate']:.4f} | "
          f"mean_temp={report['mean_temp']:.1f}°C | {len(null_counts)} cols w/ nulls")

    return report


# ═══════════════════════════════════════════════════════════════
# STAGE 5: OUTPUT — Write processed data
# ═══════════════════════════════════════════════════════════════

def write_output(df: DataFrame, output_dir: str = OUTPUT_DIR,
                 partition_by: str = "day_number") -> str:
    """Write processed DataFrame as partitioned Parquet."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    (df.repartition(8)
       .write
       .mode("overwrite")
       .partitionBy(partition_by)
       .parquet(output_dir))

    # Size check
    total_bytes = sum(f.stat().st_size for f in Path(output_dir).rglob("*.parquet"))
    total_mb = total_bytes / 1e6
    print(f"  OUTPUT: {output_dir} | {total_mb:,.0f} MB")
    return output_dir


# ═══════════════════════════════════════════════════════════════
# COMPUTE DRIFT STATISTICS (for spark_drift_detector.py)
# ═══════════════════════════════════════════════════════════════

def compute_drift_stats(df: DataFrame, day: int,
                        output_dir: str = DRIFT_DIR) -> str:
    """
    Compute per-feature distribution stats for a single day.
    Used by spark_drift_detector.py to compute PSI against reference.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stats_rows = []
    for col in NUMERIC_FEATURES:
        col_stats = df.select(
            F.lit(day).alias("day_number"),
            F.lit(col).alias("feature"),
            F.mean(col).alias("mean"),
            F.stddev(col).alias("stddev"),
            F.expr(f"percentile_approx({col}, 0.25)").alias("p25"),
            F.expr(f"percentile_approx({col}, 0.50)").alias("p50"),
            F.expr(f"percentile_approx({col}, 0.75)").alias("p75"),
            F.min(col).alias("min_val"),
            F.max(col).alias("max_val"),
        ).collect()[0]
        stats_rows.append(col_stats.asDict())

    spark = df.sparkSession
    stats_df = spark.createDataFrame(stats_rows)

    out_path = str(Path(output_dir) / f"day_{day:02d}_stats.parquet")
    stats_df.coalesce(1).write.mode("overwrite").parquet(out_path)
    print(f"  DRIFT STATS: {len(stats_rows)} features → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════
# MAIN ETL PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_etl(start_day: int, end_day: int, source: str = "production",
            master: str = None) -> dict:
    """
    Full ETL pipeline: Ingest → Clean → Feature Engineer → Quality Check → Output.
    """
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"P053 SPARK ETL — Days {start_day}-{end_day}")
    print(f"{'='*70}\n")

    spark = create_spark(master=master)
    spark.sparkContext.setLogLevel("WARN")

    # INGEST
    t1 = time.time()
    if source == "landing":
        df = ingest_landing(spark)
    else:
        df = ingest_days(spark, start_day, end_day)
    ingest_time = time.time() - t1

    # CLEAN
    t2 = time.time()
    df = clean_data(df)
    clean_time = time.time() - t2

    # FEATURE ENGINEERING
    t3 = time.time()
    df = add_engineered_features(df)
    fe_time = time.time() - t3

    # QUALITY CHECKS
    t4 = time.time()
    report = quality_checks(df, f"day_{start_day}-{end_day}")
    qc_time = time.time() - t4

    # OUTPUT
    t5 = time.time()
    write_output(df)
    output_time = time.time() - t5

    # DRIFT STATS (per-day)
    t6 = time.time()
    for day in range(start_day, end_day + 1):
        day_df = df.filter(F.col("day_number") == day)
        if day_df.head(1):
            compute_drift_stats(day_df, day)
    drift_time = time.time() - t6

    total_time = time.time() - t0

    summary = {
        "days": f"{start_day}-{end_day}",
        "total_rows": report["total_rows"],
        "fail_rate": report["fail_rate"],
        "timing": {
            "ingest_sec": round(ingest_time, 1),
            "clean_sec": round(clean_time, 1),
            "feature_eng_sec": round(fe_time, 1),
            "quality_check_sec": round(qc_time, 1),
            "output_sec": round(output_time, 1),
            "drift_stats_sec": round(drift_time, 1),
            "total_sec": round(total_time, 1),
        },
        "throughput_rows_per_sec": round(report["total_rows"] / max(total_time, 0.001)),
    }

    print(f"\n{'='*70}")
    print("ETL COMPLETE")
    print(f"  Rows:       {summary['total_rows']:>12,}")
    print(f"  Fail rate:  {summary['fail_rate']:.4f}")
    print(f"  Ingest:     {ingest_time:>6.1f}s")
    print(f"  Clean:      {clean_time:>6.1f}s")
    print(f"  Feature Eng:{fe_time:>6.1f}s")
    print(f"  QC:         {qc_time:>6.1f}s")
    print(f"  Output:     {output_time:>6.1f}s")
    print(f"  Drift Stats:{drift_time:>6.1f}s")
    print(f"  TOTAL:      {total_time:>6.1f}s ({summary['throughput_rows_per_sec']:,} rows/s)")
    print(f"{'='*70}")

    spark.stop()
    return summary


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PySpark ETL for DRAM production data")
    parser.add_argument("--day", type=int, default=1, help="Start day")
    parser.add_argument("--end-day", type=int, default=None, help="End day")
    parser.add_argument("--source", choices=["production", "landing"], default="production")
    parser.add_argument("--master", type=str, default=None,
                        help="Spark master URL (default: local[*])")
    args = parser.parse_args()

    end_day = args.end_day or args.day
    run_etl(args.day, end_day, source=args.source, master=args.master)
