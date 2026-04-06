"""
P053 — Data Profiling Report
Quick stats summary of the DRAM STDF dataset for documentation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

def profile_split(name, path):
    """Profile a single split and return stats dict."""
    df = pd.read_parquet(path)
    n = len(df)

    # Numeric features only
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude metadata columns
    exclude = {"is_fail", "is_fail_true", "label_is_noisy", "die_x", "die_y",
               "wafer_num", "uncorrectable_in_extended"}
    feature_cols = [c for c in num_cols if c not in exclude]

    stats = {
        "split": name,
        "rows": n,
        "columns": len(df.columns),
        "numeric_features": len(num_cols),
        "categorical_features": len(df.select_dtypes(include=["object"]).columns),
        "fail_count": int(df["is_fail"].sum()),
        "fail_rate_pct": round(100 * df["is_fail"].mean(), 3),
        "true_fail_count": int(df["is_fail_true"].sum()) if "is_fail_true" in df.columns else None,
        "noisy_labels": int(df["label_is_noisy"].sum()) if "label_is_noisy" in df.columns else None,
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": round(100 * df.isna().sum().sum() / (n * len(df.columns)), 2),
    }

    # Feature-level stats
    feature_stats = {}
    for col in feature_cols[:15]:  # Top 15 features
        s = df[col].dropna()
        feature_stats[col] = {
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "skewness": round(float(s.skew()), 4),
            "missing_pct": round(100 * df[col].isna().mean(), 2),
        }
    stats["features"] = feature_stats

    # Correlation: temp vs leakage
    sub = df[["test_temp_c", "cell_leakage_fa"]].dropna()
    stats["temp_leakage_correlation"] = round(float(sub.corr().iloc[0, 1]), 4)

    # Spatial: edge vs center fail rate
    sub = df[["edge_distance", "is_fail"]].dropna()
    edge_fail = sub[sub["edge_distance"] > 0.7]["is_fail"].mean()
    center_fail = sub[sub["edge_distance"] < 0.3]["is_fail"].mean()
    stats["edge_fail_rate_pct"] = round(100 * edge_fail, 3)
    stats["center_fail_rate_pct"] = round(100 * center_fail, 3)
    stats["spatial_ratio"] = round(edge_fail / max(center_fail, 1e-6), 1)

    # Root cause distribution
    if "root_cause" in df.columns:
        rc = df[df["is_fail"] == 1]["root_cause"].value_counts().to_dict()
        stats["root_cause_distribution"] = {k: int(v) for k, v in rc.items()}

    return stats


if __name__ == "__main__":
    print("=" * 60)
    print("  P053 — Data Profiling Report")
    print("=" * 60)

    all_stats = {}
    for name in ["train", "val", "test", "unseen", "sample"]:
        path = DATA / f"dram_stdf_{name}.parquet"
        if path.exists():
            print(f"\n  Profiling {name}...")
            stats = profile_split(name, path)
            all_stats[name] = stats
            print(f"    Rows: {stats['rows']:,} | Fails: {stats['fail_count']:,} ({stats['fail_rate_pct']}%)")
            print(f"    Missing: {stats['missing_pct']}% | Spatial ratio: {stats['spatial_ratio']}x")
            print(f"    Temp-leakage r: {stats['temp_leakage_correlation']}")

    # Save
    out_path = DATA / "data_profile.json"
    with open(out_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n  Profile saved to {out_path.name}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"  {'Split':<10} {'Rows':>12} {'Fails':>8} {'Fail%':>7} {'Missing%':>9} {'Spatial':>8} {'r(T,L)':>8}")
    print("-" * 80)
    for name, s in all_stats.items():
        if name == "sample":
            continue
        print(f"  {name:<10} {s['rows']:>12,} {s['fail_count']:>8,} {s['fail_rate_pct']:>6.3f}% {s['missing_pct']:>7.2f}% {s['spatial_ratio']:>6.1f}x {s['temp_leakage_correlation']:>7.4f}")
    print("=" * 80)
