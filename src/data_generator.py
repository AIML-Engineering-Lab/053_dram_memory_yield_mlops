"""
P053 — Cloud-Scale Memory Yield & Reliability Predictor
Synthetic DRAM STDF Data Generator

Generates realistic semiconductor memory test data with 10 real-world data quality
issues that make this NOT a textbook clean dataset:

1. Class imbalance     — 1:500 defect ratio (99.8% pass)
2. Missing values      — 3-8% NaN (probe contact failures)
3. Outliers            — 2% extreme values (equipment miscalibration)
4. Skewed distributions — Leakage is log-normal (long right tail)
5. Correlated noise    — Adjacent cells share noise (spatial)
6. Temporal drift      — Process shifts over 6-month window
7. Mixed types         — Numeric + categorical features
8. Multicollinearity   — Temperature ↔ leakage at r=0.85
9. Label noise         — 1-2% mislabeled (human inspection error)
10. Spatial patterns   — Edge die fail 3× more than center die
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import warnings
import json

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
DATA.mkdir(exist_ok=True)
ASSETS.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS — Realistic DRAM manufacturing parameters
# ═══════════════════════════════════════════════════════════════════
WAFER_DIAMETER_MM = 300
DIE_PITCH_X_MM = 8.5
DIE_PITCH_Y_MM = 6.2
MAX_DIE_X = int(WAFER_DIAMETER_MM / DIE_PITCH_X_MM)
MAX_DIE_Y = int(WAFER_DIAMETER_MM / DIE_PITCH_Y_MM)

ROOT_CAUSES = [
    "row_hammer",           # Repeated row activation disturbs adjacent rows
    "retention_degradation", # Cell charge leaks faster than spec at temperature
    "ecc_uncorrectable",    # Bit errors exceed ECC correction capability
    "contact_resistance",   # Via/contact degradation increases access time
    "gate_oxide_defect",    # Thin oxide breakdown causes leakage path
    "pass_transistor_weak", # Word-line transistor threshold shift
]

TESTER_IDS = [f"T{i:02d}" for i in range(1, 9)]        # 8 testers
PROBE_CARD_IDS = [f"PC{i:03d}" for i in range(1, 25)]  # 24 probe cards
CHAMBER_IDS = [f"CH{i:02d}" for i in range(1, 7)]       # 6 deposition chambers
RECIPE_VERSIONS = ["R3.2.1", "R3.2.2", "R3.3.0", "R3.3.1"]  # Process recipes
LOT_PREFIXES = ["LOT", "ENG", "QUAL"]

# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def wafer_center_distance(die_x, die_y):
    """Distance from die to wafer center (normalized 0-1)."""
    cx = MAX_DIE_X / 2.0
    cy = MAX_DIE_Y / 2.0
    max_r = np.sqrt(cx**2 + cy**2)
    r = np.sqrt((die_x - cx)**2 + (die_y - cy)**2)
    return r / max_r


def generate_lot_id(rng, n, day_offset):
    """Generate realistic lot IDs: PREFIX_YYMMDD_NNN."""
    prefix = rng.choice(LOT_PREFIXES, size=n)
    lot_num = rng.integers(1, 999, size=n)
    base_day = 20260101 + day_offset
    return np.array([
        f"{p}_{base_day + rng.integers(0, 30)}_{num:03d}"
        for p, num in zip(prefix, lot_num)
    ])


def inject_temporal_drift(feature_values, day_fraction, drift_magnitude=0.15):
    """
    Simulate 6-month process drift: chamber seasoning causes gradual shift.
    drift_magnitude: fraction of feature std that shifts over the window.
    Real-world: deposition thickness drifts 2-5% between chamber cleans.
    """
    drift = drift_magnitude * day_fraction  # Linear drift (simplification)
    std = np.std(feature_values)
    return feature_values + drift * std


def inject_spatial_correlation(values, die_x, die_y, rng, strength=0.3):
    """
    Adjacent dies on the same wafer share correlated noise.
    Real-world: lithography exposure field covers ~26mm × 33mm (multiple dies).
    Dies in the same field share systematic errors.
    """
    # Group by approximate field position (every 3 dies)
    field_x = die_x // 3
    field_y = die_y // 3
    field_id = field_x * 100 + field_y

    # Each field gets a shared noise component
    unique_fields = np.unique(field_id)
    field_noise = {f: rng.normal(0, strength * np.std(values)) for f in unique_fields}
    shared_noise = np.array([field_noise[f] for f in field_id])
    return values + shared_noise


# ═══════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════════

def generate_dram_data(n_samples, seed=42, day_offset=0, split_name="train"):
    """
    Generate n_samples of synthetic DRAM test data with all 10 data quality issues.

    Returns a DataFrame with ~35 features covering:
    - Electrical measurements (leakage, retention, timing)
    - Row hammer metrics (activation counts, disturb margins)
    - ECC metrics (bit error rates, syndrome patterns)
    - Equipment metadata (tester, probe card, chamber, recipe)
    - Spatial info (die position, wafer position)
    - Temporal info (test_date as day_offset for drift injection)
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()
    print(f"  Generating {n_samples:,} samples (split={split_name}, seed={seed})...")

    # ── DIE POSITION (spatial patterns) ──────────────────────────
    die_x = rng.integers(0, MAX_DIE_X, size=n_samples)
    die_y = rng.integers(0, MAX_DIE_Y, size=n_samples)
    edge_distance = wafer_center_distance(die_x, die_y)

    # Day within 6-month production window (0.0 = start, 1.0 = 6 months)
    day_fraction = rng.uniform(0, 1, size=n_samples)
    # Sort partially to create temporal structure (not perfectly sorted — real production)
    day_fraction = np.sort(day_fraction) + rng.normal(0, 0.02, size=n_samples)
    day_fraction = np.clip(day_fraction, 0, 1)

    # ── EQUIPMENT METADATA (categorical) ──────────────────────────
    tester_id = rng.choice(TESTER_IDS, size=n_samples)
    probe_card_id = rng.choice(PROBE_CARD_IDS, size=n_samples)
    chamber_id = rng.choice(CHAMBER_IDS, size=n_samples)
    recipe_version = rng.choice(RECIPE_VERSIONS, size=n_samples)
    lot_id = generate_lot_id(rng, n_samples, day_offset)
    wafer_num = rng.integers(1, 26, size=n_samples)  # 25 wafers per lot

    # ── TEMPERATURE (environmental) ───────────────────────────────
    # Test temperature: nominal 85°C with ±3°C chamber variation
    test_temp_c = rng.normal(85.0, 3.0, size=n_samples)
    # Some chambers run hotter (systematic offset)
    chamber_temp_offset = {c: rng.normal(0, 1.5) for c in CHAMBER_IDS}
    test_temp_c += np.array([chamber_temp_offset[c] for c in chamber_id])

    # ── CELL LEAKAGE CURRENT (log-normal — SKEWED) ────────────────
    # Real DRAM: leakage is 50-500 fA, with heavy right tail (defective cells)
    # Issue #4: Skewed distribution
    cell_leakage_fa = rng.lognormal(mean=4.6, sigma=0.35, size=n_samples)  # median ~100 fA
    # Issue #8: Multicollinearity — leakage increases with temperature (physics)
    # Real DRAM: leakage doubles every 8°C (Arrhenius). Strong coupling r~0.85
    # Copula approach: sort leakage to match temperature ranking, then add noise
    temp_ranks = np.argsort(np.argsort(test_temp_c))  # rank of each temperature
    leakage_sorted = np.sort(cell_leakage_fa)
    coupled_leakage = leakage_sorted[temp_ranks]  # Perfect rank coupling → r≈0.99
    # Blend with original random leakage to introduce decorrelation: r ≈ mix^2 * 1.0
    mix = 0.92  # Empirically gives r ≈ 0.85 for lognormal
    cell_leakage_fa = mix * coupled_leakage + (1 - mix) * cell_leakage_fa
    # Issue #5: Spatial correlation
    cell_leakage_fa = inject_spatial_correlation(cell_leakage_fa, die_x, die_y, rng, 0.25)
    # Issue #6: Temporal drift (chamber seasoning increases leakage)
    cell_leakage_fa = inject_temporal_drift(cell_leakage_fa, day_fraction, 0.12)

    # ── RETENTION TIME (temperature-dependent exponential decay) ────
    # Real DRAM: retention spec is 64ms at 85°C, measured values 40-120ms
    base_retention_ms = rng.normal(72.0, 15.0, size=n_samples)
    # Temperature dependency: retention halves every 7°C (Arrhenius)
    temp_factor = 2.0 ** ((85.0 - test_temp_c) / 7.0)
    retention_time_ms = base_retention_ms * temp_factor
    retention_time_ms = inject_temporal_drift(retention_time_ms, day_fraction, -0.08)  # Degrades over time
    retention_time_ms = inject_spatial_correlation(retention_time_ms, die_x, die_y, rng, 0.15)

    # ── ROW HAMMER METRICS ─────────────────────────────────────────
    # Number of activations before adjacent row bit flip
    # Real DRAM: threshold is ~100K-200K activations for modern DDR5
    row_hammer_threshold = rng.normal(150000, 30000, size=n_samples).astype(int)
    row_hammer_threshold = np.maximum(row_hammer_threshold, 10000)
    # Disturb margin: voltage margin before bit flip (lower = more vulnerable)
    disturb_margin_mv = rng.normal(180, 35, size=n_samples)
    # Adjacent row activation count in stress test
    adjacent_row_activations = rng.poisson(80000, size=n_samples)
    # Row hammer susceptibility score (derived: activations/threshold)
    rh_susceptibility = adjacent_row_activations / row_hammer_threshold

    # ── ECC METRICS ──────────────────────────────────────────────
    # Bit Error Rate (BER) — log-scale, most cells have very low BER
    bit_error_rate = rng.lognormal(mean=-18, sigma=2.0, size=n_samples)  # ~1e-8 typical
    # Correctable errors per 1M reads
    correctable_errors_per_1m = rng.poisson(lam=2.5, size=n_samples)
    # ECC syndrome entropy (higher = more diverse error patterns = worse)
    ecc_syndrome_entropy = rng.exponential(scale=0.3, size=n_samples)
    # Uncorrectable error flag in extended test
    uncorrectable_in_extended = (rng.random(size=n_samples) < 0.003).astype(int)

    # ── READ/WRITE TIMING ─────────────────────────────────────────
    # tRCD (Row-to-Column Delay): spec is 13.75ns ± process variation
    trcd_ns = rng.normal(13.75, 0.8, size=n_samples)
    # tRP (Row Precharge): spec is 13.75ns
    trp_ns = rng.normal(13.75, 0.6, size=n_samples)
    # tRAS (Row Active Time): spec is 32ns
    tras_ns = rng.normal(32.0, 1.5, size=n_samples)
    # Issue #3: Outliers — 2% of measurements have extreme timing (equipment glitch)
    outlier_mask = rng.random(size=n_samples) < 0.02
    trcd_ns[outlier_mask] += rng.normal(5.0, 2.0, size=outlier_mask.sum())
    trp_ns[outlier_mask] += rng.normal(3.0, 1.5, size=outlier_mask.sum())

    # Read/write latency jitter (bimodal: normal + occasional queue stalls)
    # Issue: bimodal distribution — 95% normal, 5% stalled
    rw_latency_ns = rng.normal(15.0, 1.0, size=n_samples)
    stall_mask = rng.random(size=n_samples) < 0.05
    rw_latency_ns[stall_mask] = rng.normal(45.0, 8.0, size=stall_mask.sum())

    # ── POWER METRICS ────────────────────────────────────────────
    # Active power (IDD4): spec ~300mA, varies with design
    idd4_active_ma = rng.normal(310, 25, size=n_samples)
    # Standby power (IDD2P): spec ~20mA
    idd2p_standby_ma = rng.normal(22, 4, size=n_samples)
    # Refresh power (IDD5): directly related to retention time
    idd5_refresh_ma = rng.normal(180, 20, size=n_samples)
    # Devices with poor retention need more frequent refresh → higher power
    poor_retention_mask = retention_time_ms < 55
    idd5_refresh_ma[poor_retention_mask] += rng.normal(40, 10, size=poor_retention_mask.sum())

    # ── PROCESS METRICS ──────────────────────────────────────────
    # Gate oxide thickness (Angstroms) — affects leakage
    gate_oxide_thickness_a = rng.normal(28.0, 0.8, size=n_samples)
    gate_oxide_thickness_a = inject_temporal_drift(gate_oxide_thickness_a, day_fraction, 0.06)
    # Channel length (nm) — shrinks with process advancement
    channel_length_nm = rng.normal(14.0, 0.5, size=n_samples)
    # Threshold voltage (mV) — VT shift indicates reliability issue
    vt_shift_mv = rng.normal(0, 15, size=n_samples)
    # Issue #6: temporal drift on VT (stress-induced degradation)
    vt_shift_mv = inject_temporal_drift(vt_shift_mv, day_fraction, 0.20)

    # ── BLOCK ERASE COUNT (NAND cross-reference) ─────────────────
    # For hybrid memory systems: NAND wear-out indicator
    block_erase_count = rng.poisson(lam=5000, size=n_samples)

    # ═══════════════════════════════════════════════════════════════
    # FAILURE LABEL GENERATION (REALISTIC)
    # ═══════════════════════════════════════════════════════════════

    # Base failure probability — very low (Issue #1: class imbalance 1:500)
    fail_prob = np.full(n_samples, 0.0003)

    # Increase failure probability based on physics-driven rules:
    # (Edge effect applied as multiplier AFTER all additive terms — see below)

    # 2. High leakage → fail
    leakage_z = (cell_leakage_fa - np.median(cell_leakage_fa)) / np.std(cell_leakage_fa)
    fail_prob += 0.003 * np.clip(leakage_z - 2.0, 0, 5)

    # 3. Low retention → fail
    fail_prob += 0.005 * (retention_time_ms < 45)
    fail_prob += 0.015 * (retention_time_ms < 35)

    # 4. Row hammer vulnerable → fail
    fail_prob += 0.004 * (rh_susceptibility > 0.7)
    fail_prob += 0.010 * (rh_susceptibility > 0.9)

    # 5. ECC issues → fail
    fail_prob += 0.008 * (correctable_errors_per_1m > 10)
    fail_prob += 0.050 * uncorrectable_in_extended

    # 6. Extreme timing → fail
    fail_prob += 0.003 * (trcd_ns > 17.0)

    # 7. High VT shift → fail (reliability screen)
    fail_prob += 0.004 * (np.abs(vt_shift_mv) > 40)

    # 8. Temporal: failure rate increases with drift
    fail_prob += 0.0005 * day_fraction  # Late production has slightly higher failure rate

    # Issue #10: Spatial pattern — edge dies have HIGHER defect density across ALL mechanisms
    # Real wafer: edge exclusion zone has 2-4× higher defect density (CMP non-uniformity,
    # lithography edge effects, implant dose roll-off). Applied as MULTIPLIER.
    edge_factor = 1.0 + 6.0 * np.clip((edge_distance - 0.55) / 0.45, 0, 1)  # 1× center → 7× extreme edge
    fail_prob *= edge_factor

    # Generate binary label
    fail_prob = np.clip(fail_prob, 0, 0.95)
    is_fail = (rng.random(size=n_samples) < fail_prob).astype(int)

    # Assign root cause for failures
    root_cause = np.where(is_fail == 0, "pass", "unknown")
    fail_indices = np.where(is_fail == 1)[0]
    if len(fail_indices) > 0:
        # Assign based on dominant failure mechanism
        for idx in fail_indices:
            scores = {
                "row_hammer": rh_susceptibility[idx] * 2,
                "retention_degradation": max(0, (50 - retention_time_ms[idx]) / 20),
                "ecc_uncorrectable": correctable_errors_per_1m[idx] / 5 + uncorrectable_in_extended[idx] * 3,
                "contact_resistance": max(0, (trcd_ns[idx] - 15) / 2),
                "gate_oxide_defect": max(0, leakage_z[idx] - 2) * 1.5,
                "pass_transistor_weak": max(0, (np.abs(vt_shift_mv[idx]) - 30) / 10),
            }
            root_cause[idx] = max(scores, key=scores.get)

    # Issue #9: Label noise — realistic human inspection error
    # In real fabs, ~5% of failed-bin devices are mislabeled (inspector fatigue)
    # and ~0.1% of pass devices are wrongly marked fail (false alarms)
    is_fail_noisy = is_fail.copy()
    # Flip 5% of real fails to pass (missed defects)
    fail_indices_arr = np.where(is_fail == 1)[0]
    if len(fail_indices_arr) > 0:
        flip_fail = rng.random(size=len(fail_indices_arr)) < 0.05
        is_fail_noisy[fail_indices_arr[flip_fail]] = 0
    # Flip 0.1% of real passes to fail (false alarms)
    pass_indices_arr = np.where(is_fail == 0)[0]
    if len(pass_indices_arr) > 0:
        flip_pass = rng.random(size=len(pass_indices_arr)) < 0.001
        is_fail_noisy[pass_indices_arr[flip_pass]] = 1
    # Track which labels are noisy
    label_is_noisy = (is_fail != is_fail_noisy)

    # Fix root cause for noise-flipped samples
    # Samples that were pass but noise-flipped to fail get "label_error" root cause
    noise_flipped_to_fail = label_is_noisy & (is_fail == 0)
    root_cause[noise_flipped_to_fail] = "label_error"

    # ═══════════════════════════════════════════════════════════════
    # INJECT MISSING VALUES (Issue #2)
    # ═══════════════════════════════════════════════════════════════
    # Different features have different miss rates (realistic)
    # Probe contact issues cause sensor dropouts
    miss_rates = {
        "cell_leakage_fa": 0.03,
        "retention_time_ms": 0.04,
        "disturb_margin_mv": 0.06,  # Hard to measure reliably
        "bit_error_rate": 0.05,
        "trcd_ns": 0.02,
        "gate_oxide_thickness_a": 0.08,  # Inline metrology not always available
        "idd4_active_ma": 0.03,
        "vt_shift_mv": 0.07,  # Requires dedicated test (often skipped for throughput)
    }

    # ═══════════════════════════════════════════════════════════════
    # ASSEMBLE DATAFRAME
    # ═══════════════════════════════════════════════════════════════
    df = pd.DataFrame({
        # Spatial
        "die_x": die_x,
        "die_y": die_y,
        "edge_distance": np.round(edge_distance, 4),
        # Equipment
        "tester_id": tester_id,
        "probe_card_id": probe_card_id,
        "chamber_id": chamber_id,
        "recipe_version": recipe_version,
        "lot_id": lot_id,
        "wafer_num": wafer_num,
        # Temporal
        "day_fraction": np.round(day_fraction, 4),
        # Environmental
        "test_temp_c": np.round(test_temp_c, 2),
        # Electrical — cell level
        "cell_leakage_fa": np.round(cell_leakage_fa, 2),
        "retention_time_ms": np.round(retention_time_ms, 2),
        # Row hammer
        "row_hammer_threshold": row_hammer_threshold,
        "disturb_margin_mv": np.round(disturb_margin_mv, 2),
        "adjacent_row_activations": adjacent_row_activations,
        "rh_susceptibility": np.round(rh_susceptibility, 4),
        # ECC
        "bit_error_rate": bit_error_rate,
        "correctable_errors_per_1m": correctable_errors_per_1m,
        "ecc_syndrome_entropy": np.round(ecc_syndrome_entropy, 4),
        "uncorrectable_in_extended": uncorrectable_in_extended,
        # Timing
        "trcd_ns": np.round(trcd_ns, 3),
        "trp_ns": np.round(trp_ns, 3),
        "tras_ns": np.round(tras_ns, 3),
        "rw_latency_ns": np.round(rw_latency_ns, 3),
        # Power
        "idd4_active_ma": np.round(idd4_active_ma, 2),
        "idd2p_standby_ma": np.round(idd2p_standby_ma, 2),
        "idd5_refresh_ma": np.round(idd5_refresh_ma, 2),
        # Process
        "gate_oxide_thickness_a": np.round(gate_oxide_thickness_a, 2),
        "channel_length_nm": np.round(channel_length_nm, 2),
        "vt_shift_mv": np.round(vt_shift_mv, 2),
        # NAND cross-reference
        "block_erase_count": block_erase_count,
        # Labels
        "is_fail": is_fail_noisy,  # The noisy label (what production sees)
        "is_fail_true": is_fail,    # Ground truth (for analysis only — not for training)
        "root_cause": root_cause,
        "label_is_noisy": label_is_noisy,
        # Metadata
        "split": split_name,
    })

    # Inject missing values per feature
    for col, rate in miss_rates.items():
        mask = rng.random(size=n_samples) < rate
        df.loc[mask, col] = np.nan

    elapsed = time.time() - t0
    n_fail = is_fail.sum()
    n_noisy = label_is_noisy.sum()
    pct_fail = 100 * n_fail / n_samples
    print(f"    Rows: {n_samples:,} | Fails: {n_fail:,} ({pct_fail:.2f}%) | "
          f"Label noise: {n_noisy:,} | Missing: {sum(df.isna().sum()):,} cells | "
          f"Time: {elapsed:.1f}s")

    return df


def generate_all_splits():
    """Generate train/val/test/unseen splits and save as Parquet."""
    splits = {
        "train":  {"n": 10_000_000, "seed": 42,   "day_offset": 0},
        "val":    {"n":  2_000_000, "seed": 123,  "day_offset": 0},
        "test":   {"n":  2_000_000, "seed": 456,  "day_offset": 0},
        "unseen": {"n":  2_000_000, "seed": 999,  "day_offset": 180},  # Different seed + 6 months later
    }

    stats = {}
    for split_name, cfg in splits.items():
        print(f"\n{'='*60}")
        print(f"  Split: {split_name}")
        print(f"{'='*60}")

        df = generate_dram_data(
            n_samples=cfg["n"],
            seed=cfg["seed"],
            day_offset=cfg["day_offset"],
            split_name=split_name,
        )

        # Save as Parquet (partitioned by first 2 chars of lot_id for large-file management)
        out_path = DATA / f"dram_stdf_{split_name}.parquet"
        df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"    Saved: {out_path.name} ({size_mb:.1f} MB)")

        stats[split_name] = {
            "rows": len(df),
            "fails": int(df["is_fail"].sum()),
            "fail_rate": round(100 * df["is_fail"].mean(), 3),
            "missing_cells": int(df.isna().sum().sum()),
            "missing_pct": round(100 * df.isna().sum().sum() / (len(df) * len(df.columns)), 2),
            "noisy_labels": int(df["label_is_noisy"].sum()),
            "file_size_mb": round(size_mb, 1),
            "n_features": len(df.columns) - 4,  # Exclude split, is_fail_true, label_is_noisy, root_cause
        }

    # Save stats
    with open(DATA / "data_generation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved to data_generation_stats.json")
    print(f"  Total rows: {sum(s['rows'] for s in stats.values()):,}")
    total_mb = sum(s['file_size_mb'] for s in stats.values())
    print(f"  Total size: {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")

    return stats


def generate_small_sample(n=50_000, seed=42):
    """Generate a small sample for quick testing / EDA development."""
    print("\n  Generating small sample for EDA development...")
    df = generate_dram_data(n_samples=n, seed=seed, split_name="sample")
    out_path = DATA / "dram_stdf_sample.parquet"
    df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"    Saved: {out_path.name} ({size_mb:.1f} MB)")
    # Also save as CSV for quick inspection
    csv_path = DATA / "dram_stdf_sample.csv"
    df.head(1000).to_csv(csv_path, index=False)
    print(f"    CSV preview (1000 rows): {csv_path.name}")
    return df


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate DRAM STDF synthetic data")
    parser.add_argument("--full", action="store_true", help="Generate full 16M row dataset")
    parser.add_argument("--sample", action="store_true", help="Generate 50K sample for EDA")
    args = parser.parse_args()

    if args.full:
        generate_all_splits()
    elif args.sample:
        generate_small_sample()
    else:
        print("Usage: python data_generator.py --sample  (quick 50K)")
        print("       python data_generator.py --full    (full 16M rows, ~10 min)")
        print("\nGenerating sample by default...")
        generate_small_sample()
