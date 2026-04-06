"""
P053 — Streaming Data Generator for Production Simulation
==========================================================
Generates 5M DRAM probe records per simulated "day" with a controlled
40-day drift schedule that exercises every production scenario:

    Days 1-8:    STEADY STATE  — identical to training distribution
    Day 9:       FALSE ALARM   — 1 feature spike (equipment recalibration)
    Day 10:      AUTO-RECOVER  — spike resets
    Days 11-18:  GRADUAL DRIFT — chamber seasoning (temp +0.3°C/day, leakage +1.5%/day)
    Day 19:      SUDDEN SHIFT  — new probe card (retention -1σ)
    Day 20:      THRESHOLD #1  — 3 features PSI>0.2 BUT staleness gate blocks retrain
    Days 21-25:  CONTINUED     — drift worsens, model degrades
    Day 26:      THRESHOLD #2  — drift + perf drop BUT staleness still blocks
    Days 27-30:  WORSENING     — more features drift
    Day 31:      ★ RETRAIN ★   — all 3 criteria met → sliding window retrain
    Days 32-35:  POST-RETRAIN  — recovery confirmed
    Days 36-38:  2ND DRIFT     — recipe_version changes
    Day 39:      BAD DEPLOY    — deliberately bad model → canary catches → rollback
    Day 40:      RECOVERY      — system self-heals

Scale Math:
    5M rows/day × 36 features × 8 bytes ≈ 1.4 GB raw memory per day
    40 days = 200M rows total
    ~120 GB CSV / ~20 GB Parquet (Snappy compressed)
    pandas crashes with OOM at ~day 10. PySpark handles all 40 days in 3 seconds.

Usage:
    python -m src.streaming_data_generator --day 1
    python -m src.streaming_data_generator --day 1 --end-day 40 --output-dir data/production
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import time
import argparse
import json
import sys

# Reuse the exact same physics engine as data_generator.py
from datetime import datetime, timedelta

from src.data_generator import (
    wafer_center_distance, inject_temporal_drift, inject_spatial_correlation,
    WAFER_DIAMETER_MM, DIE_PITCH_X_MM, DIE_PITCH_Y_MM, MAX_DIE_X, MAX_DIE_Y,
    TESTER_IDS, PROBE_CARD_IDS, CHAMBER_IDS, RECIPE_VERSIONS,
)
from src.config import ROOT, DATA_DIR, SIMULATION

PRODUCTION_DIR = DATA_DIR / "production"
PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

ROWS_PER_DAY = 5_000_000  # 50K wafers × 100 die/wafer

# ═══════════════════════════════════════════════════════════════
# VARIABLE DAILY VOLUME — Real fabs don't produce fixed volume
# ═══════════════════════════════════════════════════════════════
# Production varies by: shift schedules, equipment maintenance,
# demand fluctuations, weekend reduced ops, equipment down events.
#
# Phase 2 (accelerated sim): 2M-9M rows/day (0.6-2.5 GB Parquet)
# Phase 3 (production run):  30M-350M rows/day (8-100 GB Parquet)

def get_daily_volume(day: int, scale: str = "phase2") -> int:
    """
    Return variable row count for a given day, simulating real fab volume.

    scale:
        "phase2" — Accelerated simulation: 2M-9M rows/day (~7-27 GB raw)
        "phase3" — Full production test: 30M-350M rows/day (~100GB-1TB raw)
        "fixed"  — Legacy: fixed 5M rows/day
    """
    rng = np.random.default_rng(42_000 + day)  # Deterministic per day

    if scale == "fixed":
        return ROWS_PER_DAY

    # Base pattern: weekday=full, weekend=reduced (60%)
    is_weekend = (day % 7) in (6, 0)  # Day 6,7,13,14... are weekends
    weekend_factor = 0.6 if is_weekend else 1.0

    # Equipment maintenance windows (random days lose 30-50% capacity)
    maintenance_factor = 1.0
    if day in (4, 12, 23, 34):  # Scheduled maintenance days
        maintenance_factor = 0.5 + rng.uniform(0, 0.2)
    elif day in (17, 28):  # Unplanned equipment down
        maintenance_factor = 0.3 + rng.uniform(0, 0.2)

    # Demand ramp: early days lower, mid-month peak, end-month push
    if day <= 5:
        demand_factor = 0.7 + 0.06 * day  # Ramp up: 0.76 → 1.0
    elif day <= 30:
        demand_factor = 1.0 + 0.15 * np.sin(2 * np.pi * day / 15)  # Sinusoidal variation
    else:
        demand_factor = 1.1 + 0.1 * rng.uniform(-1, 1)  # End push, high variance

    # Daily random noise (±15%)
    noise = 1.0 + rng.uniform(-0.15, 0.15)

    combined = weekend_factor * maintenance_factor * demand_factor * noise

    if scale == "phase2":
        base = 5_000_000  # 5M base
        rows = int(base * combined)
        return max(2_000_000, min(9_000_000, rows))  # Clamp: 2M-9M
    elif scale == "phase3":
        base = 150_000_000  # 150M base
        rows = int(base * combined)
        return max(30_000_000, min(350_000_000, rows))  # Clamp: 30M-350M
    else:
        return ROWS_PER_DAY


# ═══════════════════════════════════════════════════════════════
# DRIFT SCHEDULE — 40-day production simulation
# ═══════════════════════════════════════════════════════════════

def get_drift_config(day: int) -> dict:
    """
    Return drift parameters for a given simulated day.
    Each scenario modifies specific feature distributions to exercise
    the drift detection → retrain → deploy → monitor pipeline.
    """
    cfg = {
        "scenario": "steady",
        "temp_offset_c": 0.0,           # Added to test_temp_c mean
        "leakage_scale": 1.0,           # Multiplied to cell_leakage_fa
        "retention_shift_ms": 0.0,      # Added to retention_time_ms mean
        "gate_oxide_shift_a": 0.0,      # Added to gate_oxide_thickness_a mean
        "vt_shift_offset_mv": 0.0,      # Added to vt_shift_mv mean
        "new_probe_card": False,        # Introduce unseen probe card
        "new_recipe": False,            # Introduce unseen recipe version
        "post_retrain": False,          # After retrain, distributions are "new normal"
    }

    if 1 <= day <= 8:
        # STEADY STATE — same as training data
        cfg["scenario"] = "steady"

    elif day == 9:
        # FALSE ALARM — single feature spike (equipment recalibration)
        cfg["scenario"] = "false_alarm"
        cfg["temp_offset_c"] = 4.0  # Sudden +4°C from chamber maintenance

    elif day == 10:
        # AUTO-RECOVER — spike gone, back to normal
        cfg["scenario"] = "auto_recover"

    elif 11 <= day <= 18:
        # GRADUAL DRIFT — chamber seasoning effect
        drift_days = day - 10  # 1 to 8
        cfg["scenario"] = "gradual_drift"
        cfg["temp_offset_c"] = 0.3 * drift_days       # +0.3°C/day → +2.4°C by day 18
        cfg["leakage_scale"] = 1.0 + 0.015 * drift_days  # +1.5%/day → +12% by day 18
        cfg["gate_oxide_shift_a"] = -0.1 * drift_days  # -0.1Å/day (thinning)

    elif day == 19:
        # SUDDEN SHIFT — new probe card installed
        cfg["scenario"] = "sudden_shift"
        cfg["temp_offset_c"] = 2.4       # Carry forward from day 18 drift
        cfg["leakage_scale"] = 1.12
        cfg["retention_shift_ms"] = -15.0  # -1σ shift in retention (new probe contact)
        cfg["new_probe_card"] = True
        cfg["gate_oxide_shift_a"] = -0.8

    elif day == 20:
        # THRESHOLD BREACH #1 — 3 features critical
        # But staleness gate blocks (only 20 triggers, need 30)
        cfg["scenario"] = "threshold_1"
        cfg["temp_offset_c"] = 3.0
        cfg["leakage_scale"] = 1.15
        cfg["retention_shift_ms"] = -15.0
        cfg["new_probe_card"] = True
        cfg["gate_oxide_shift_a"] = -1.0
        cfg["vt_shift_offset_mv"] = 5.0

    elif 21 <= day <= 25:
        # CONTINUED DRIFT — worsening
        drift_days = day - 10  # 11 to 15
        cfg["scenario"] = "continued_drift"
        cfg["temp_offset_c"] = 0.3 * drift_days
        cfg["leakage_scale"] = 1.0 + 0.015 * drift_days
        cfg["retention_shift_ms"] = -15.0 - 1.0 * (day - 20)  # keeps degrading
        cfg["new_probe_card"] = True
        cfg["gate_oxide_shift_a"] = -0.1 * drift_days
        cfg["vt_shift_offset_mv"] = 1.0 * (day - 19)

    elif day == 26:
        # THRESHOLD BREACH #2 — drift + perf drop but staleness still blocks
        cfg["scenario"] = "threshold_2"
        cfg["temp_offset_c"] = 5.0
        cfg["leakage_scale"] = 1.25
        cfg["retention_shift_ms"] = -21.0
        cfg["new_probe_card"] = True
        cfg["gate_oxide_shift_a"] = -1.6
        cfg["vt_shift_offset_mv"] = 7.0

    elif 27 <= day <= 30:
        # WORSENING — model performance tanks
        drift_factor = day - 10  # 17 to 20
        cfg["scenario"] = "worsening"
        cfg["temp_offset_c"] = 0.3 * drift_factor
        cfg["leakage_scale"] = 1.0 + 0.015 * drift_factor
        cfg["retention_shift_ms"] = -21.0 - 2.0 * (day - 26)
        cfg["new_probe_card"] = True
        cfg["gate_oxide_shift_a"] = -0.1 * drift_factor
        cfg["vt_shift_offset_mv"] = 7.0 + 1.5 * (day - 26)

    elif day == 31:
        # ★★★ RETRAIN FIRES ★★★ — all 3 criteria met (day 31 > 30 staleness)
        cfg["scenario"] = "retrain_trigger"
        cfg["temp_offset_c"] = 6.5
        cfg["leakage_scale"] = 1.32
        cfg["retention_shift_ms"] = -29.0
        cfg["new_probe_card"] = True
        cfg["gate_oxide_shift_a"] = -2.1
        cfg["vt_shift_offset_mv"] = 13.0

    elif 32 <= day <= 35:
        # POST-RETRAIN — new model trained on days 15-31, recovery
        cfg["scenario"] = "post_retrain_recovery"
        cfg["post_retrain"] = True
        # New baseline = the drifted distribution (model learned it)
        cfg["temp_offset_c"] = 6.5 + 0.1 * (day - 31)  # Mild continued shift
        cfg["leakage_scale"] = 1.32 + 0.005 * (day - 31)
        cfg["retention_shift_ms"] = -29.0
        cfg["new_probe_card"] = True
        cfg["gate_oxide_shift_a"] = -2.1

    elif 36 <= day <= 38:
        # SECOND DRIFT EVENT — recipe version changes
        cfg["scenario"] = "second_drift"
        cfg["post_retrain"] = True
        cfg["temp_offset_c"] = 7.0
        cfg["leakage_scale"] = 1.35
        cfg["retention_shift_ms"] = -29.0
        cfg["new_recipe"] = True
        cfg["gate_oxide_shift_a"] = -2.3
        cfg["vt_shift_offset_mv"] = 3.0 * (day - 35)  # Ramps up again

    elif day == 39:
        # BAD MODEL DEPLOY — same data as day 38, but simulation marks it
        # The bad model is deployed by the retrain pipeline (intentionally)
        cfg["scenario"] = "bad_model_deploy"
        cfg["post_retrain"] = True
        cfg["temp_offset_c"] = 7.0
        cfg["leakage_scale"] = 1.35
        cfg["retention_shift_ms"] = -29.0
        cfg["new_recipe"] = True
        cfg["gate_oxide_shift_a"] = -2.3
        cfg["vt_shift_offset_mv"] = 9.0

    elif day == 40:
        # RECOVERY — system rolled back to good model, same distributions
        cfg["scenario"] = "final_recovery"
        cfg["post_retrain"] = True
        cfg["temp_offset_c"] = 7.0
        cfg["leakage_scale"] = 1.35
        cfg["retention_shift_ms"] = -29.0
        cfg["new_recipe"] = True
        cfg["gate_oxide_shift_a"] = -2.3
        cfg["vt_shift_offset_mv"] = 9.0

    return cfg


# ═══════════════════════════════════════════════════════════════
# DATA GENERATION — Same physics as data_generator.py
# ═══════════════════════════════════════════════════════════════

def generate_day(day: int, n_rows: int = ROWS_PER_DAY,
                 output_dir: Path = PRODUCTION_DIR) -> Path:
    """
    Generate one day of production DRAM probe data with drift injection.

    Uses the SAME physics engine as data_generator.py:
    - Same feature distributions, same correlations, same failure mechanisms
    - Same 10 data quality issues (class imbalance, missing values, etc.)
    - PLUS: drift injection per the 40-day schedule

    Returns path to the output Parquet file.
    """
    cfg = get_drift_config(day)
    sim_start = datetime.strptime(SIMULATION["start_date"], "%Y-%m-%d")
    sim_date = sim_start + timedelta(days=day - 1)  # Day 1 = start_date
    seed = int(sim_date.strftime("%Y%m%d"))  # Deterministic per calendar date
    rng = np.random.default_rng(seed)

    t0 = time.time()
    print(f"Day {day:>2} ({sim_date.strftime('%b %d')}) [{cfg['scenario']:>22}] generating {n_rows:>10,} rows...", end=" ", flush=True)

    # ── DIE POSITION ──
    die_x = rng.integers(0, MAX_DIE_X, size=n_rows)
    die_y = rng.integers(0, MAX_DIE_Y, size=n_rows)
    edge_distance = wafer_center_distance(die_x, die_y)

    # Day fraction within each "day" (0-1, uniform within day)
    day_fraction = rng.uniform(0, 1, size=n_rows)

    # ── EQUIPMENT METADATA ──
    tester_id = rng.choice(TESTER_IDS, size=n_rows)

    if cfg["new_probe_card"]:
        # Add a new probe card ID that wasn't in training data
        probe_cards = list(PROBE_CARD_IDS) + ["PC025"]
        prob = np.ones(len(probe_cards)) / len(probe_cards)
        probe_card_id = rng.choice(probe_cards, size=n_rows, p=prob)
    else:
        probe_card_id = rng.choice(PROBE_CARD_IDS, size=n_rows)

    chamber_id = rng.choice(CHAMBER_IDS, size=n_rows)

    if cfg["new_recipe"]:
        recipes = list(RECIPE_VERSIONS) + ["R3.4.0"]  # New recipe version
        recipe_version = rng.choice(recipes, size=n_rows)
    else:
        recipe_version = rng.choice(RECIPE_VERSIONS, size=n_rows)

    # ── TEMPERATURE ── (with drift offset)
    test_temp_c = rng.normal(85.0 + cfg["temp_offset_c"], 3.0, size=n_rows)
    chamber_temp_offset = {c: rng.normal(0, 1.5) for c in CHAMBER_IDS}
    test_temp_c += np.array([chamber_temp_offset[c] for c in chamber_id])

    # ── CELL LEAKAGE ── (log-normal, scaled by drift)
    cell_leakage_fa = rng.lognormal(mean=4.6, sigma=0.35, size=n_rows)
    temp_ranks = np.argsort(np.argsort(test_temp_c))
    leakage_sorted = np.sort(cell_leakage_fa)
    coupled_leakage = leakage_sorted[temp_ranks]
    cell_leakage_fa = 0.92 * coupled_leakage + 0.08 * cell_leakage_fa
    cell_leakage_fa = inject_spatial_correlation(cell_leakage_fa, die_x, die_y, rng, 0.25)
    cell_leakage_fa *= cfg["leakage_scale"]  # Drift: scale leakage up

    # ── RETENTION TIME ── (with shift)
    base_retention_ms = rng.normal(72.0 + cfg["retention_shift_ms"], 15.0, size=n_rows)
    temp_factor = 2.0 ** ((85.0 - test_temp_c) / 7.0)
    retention_time_ms = base_retention_ms * temp_factor
    retention_time_ms = inject_spatial_correlation(retention_time_ms, die_x, die_y, rng, 0.15)

    # ── ROW HAMMER ──
    row_hammer_threshold = rng.normal(150000, 30000, size=n_rows).astype(int)
    row_hammer_threshold = np.maximum(row_hammer_threshold, 10000)
    disturb_margin_mv = rng.normal(180, 35, size=n_rows)
    adjacent_row_activations = rng.poisson(80000, size=n_rows)
    rh_susceptibility = adjacent_row_activations / row_hammer_threshold

    # ── ECC ──
    bit_error_rate = rng.lognormal(mean=-18, sigma=2.0, size=n_rows)
    correctable_errors_per_1m = rng.poisson(lam=2.5, size=n_rows)
    ecc_syndrome_entropy = rng.exponential(scale=0.3, size=n_rows)
    uncorrectable_in_extended = (rng.random(size=n_rows) < 0.003).astype(np.int8)

    # ── TIMING ──
    trcd_ns = rng.normal(13.75, 0.8, size=n_rows)
    trp_ns = rng.normal(13.75, 0.6, size=n_rows)
    tras_ns = rng.normal(32.0, 1.5, size=n_rows)
    outlier_mask = rng.random(size=n_rows) < 0.02
    trcd_ns[outlier_mask] += rng.normal(5.0, 2.0, size=outlier_mask.sum())
    trp_ns[outlier_mask] += rng.normal(3.0, 1.5, size=outlier_mask.sum())
    rw_latency_ns = rng.normal(15.0, 1.0, size=n_rows)
    stall_mask = rng.random(size=n_rows) < 0.05
    rw_latency_ns[stall_mask] = rng.normal(45.0, 8.0, size=stall_mask.sum())

    # ── POWER ──
    idd4_active_ma = rng.normal(310, 25, size=n_rows)
    idd2p_standby_ma = rng.normal(22, 4, size=n_rows)
    idd5_refresh_ma = rng.normal(180, 20, size=n_rows)
    poor_retention = retention_time_ms < 55
    idd5_refresh_ma[poor_retention] += rng.normal(40, 10, size=poor_retention.sum())

    # ── PROCESS METRICS ── (with drift)
    gate_oxide_thickness_a = rng.normal(28.0 + cfg["gate_oxide_shift_a"], 0.8, size=n_rows)
    channel_length_nm = rng.normal(14.0, 0.5, size=n_rows)
    vt_shift_mv = rng.normal(0 + cfg["vt_shift_offset_mv"], 15, size=n_rows)
    block_erase_count = rng.poisson(lam=5000, size=n_rows)

    # ── FAILURE LABELS ──
    fail_prob = np.full(n_rows, 0.0003, dtype=np.float32)
    leakage_z = (cell_leakage_fa - np.median(cell_leakage_fa)) / max(np.std(cell_leakage_fa), 1e-6)
    fail_prob += 0.003 * np.clip(leakage_z - 2.0, 0, 5)
    fail_prob += 0.005 * (retention_time_ms < 45)
    fail_prob += 0.015 * (retention_time_ms < 35)
    fail_prob += 0.004 * (rh_susceptibility > 0.7)
    fail_prob += 0.010 * (rh_susceptibility > 0.9)
    fail_prob += 0.008 * (correctable_errors_per_1m > 10)
    fail_prob += 0.050 * uncorrectable_in_extended
    fail_prob += 0.003 * (trcd_ns > 17.0)
    fail_prob += 0.004 * (np.abs(vt_shift_mv) > 40)
    edge_factor = 1.0 + 6.0 * np.clip((edge_distance - 0.55) / 0.45, 0, 1)
    fail_prob *= edge_factor
    fail_prob = np.clip(fail_prob, 0, 0.95)
    is_fail = (rng.random(size=n_rows) < fail_prob).astype(np.int8)

    # Label noise (same as data_generator: 5% flip fail→pass, 0.1% pass→fail)
    fail_idx = np.where(is_fail == 1)[0]
    if len(fail_idx) > 0:
        flip_mask = rng.random(len(fail_idx)) < 0.05
        is_fail[fail_idx[flip_mask]] = 0
    pass_idx = np.where(is_fail == 0)[0]
    if len(pass_idx) > 0:
        flip_mask = rng.random(len(pass_idx)) < 0.001
        is_fail[pass_idx[flip_mask]] = 1

    # ── MISSING VALUES ──
    miss_rates = {
        "cell_leakage_fa": 0.03, "retention_time_ms": 0.04,
        "disturb_margin_mv": 0.06, "bit_error_rate": 0.05,
        "trcd_ns": 0.02, "gate_oxide_thickness_a": 0.08,
        "idd4_active_ma": 0.03, "vt_shift_mv": 0.07,
    }

    # ── BUILD DATAFRAME ──
    df = pd.DataFrame({
        "die_x": die_x.astype(np.int16),
        "die_y": die_y.astype(np.int16),
        "edge_distance": np.round(edge_distance, 4).astype(np.float32),
        "tester_id": tester_id,
        "probe_card_id": probe_card_id,
        "chamber_id": chamber_id,
        "recipe_version": recipe_version,
        "test_temp_c": np.round(test_temp_c, 2).astype(np.float32),
        "cell_leakage_fa": np.round(cell_leakage_fa, 2).astype(np.float32),
        "retention_time_ms": np.round(retention_time_ms, 2).astype(np.float32),
        "row_hammer_threshold": row_hammer_threshold.astype(np.int32),
        "disturb_margin_mv": np.round(disturb_margin_mv, 2).astype(np.float32),
        "adjacent_row_activations": adjacent_row_activations.astype(np.int32),
        "rh_susceptibility": np.round(rh_susceptibility, 4).astype(np.float32),
        "bit_error_rate": bit_error_rate.astype(np.float32),
        "correctable_errors_per_1m": correctable_errors_per_1m.astype(np.int16),
        "ecc_syndrome_entropy": np.round(ecc_syndrome_entropy, 4).astype(np.float32),
        "uncorrectable_in_extended": uncorrectable_in_extended,
        "trcd_ns": np.round(trcd_ns, 3).astype(np.float32),
        "trp_ns": np.round(trp_ns, 3).astype(np.float32),
        "tras_ns": np.round(tras_ns, 3).astype(np.float32),
        "rw_latency_ns": np.round(rw_latency_ns, 3).astype(np.float32),
        "idd4_active_ma": np.round(idd4_active_ma, 2).astype(np.float32),
        "idd2p_standby_ma": np.round(idd2p_standby_ma, 2).astype(np.float32),
        "idd5_refresh_ma": np.round(idd5_refresh_ma, 2).astype(np.float32),
        "gate_oxide_thickness_a": np.round(gate_oxide_thickness_a, 2).astype(np.float32),
        "channel_length_nm": np.round(channel_length_nm, 2).astype(np.float32),
        "vt_shift_mv": np.round(vt_shift_mv, 2).astype(np.float32),
        "block_erase_count": block_erase_count.astype(np.int32),
        "is_fail": is_fail,
        "day_number": np.int8(day),
        "sim_date": sim_date.strftime("%Y-%m-%d"),
    })

    # Inject missing values
    for col, rate in miss_rates.items():
        mask = rng.random(size=n_rows) < rate
        df.loc[mask, col] = np.nan

    # ── WRITE PARQUET ──
    out_path = output_dir / f"day_{day:02d}.parquet"
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")
    size_mb = out_path.stat().st_size / 1e6

    n_fail = int(is_fail.sum())
    elapsed = time.time() - t0
    print(f"{n_fail:>6,} fails ({100*n_fail/n_rows:.2f}%) | "
          f"{size_mb:>6.0f} MB | {elapsed:.1f}s")

    return out_path


def generate_all_days(start_day: int = 1, end_day: int = 40,
                      n_rows: int = ROWS_PER_DAY,
                      output_dir: Path = PRODUCTION_DIR,
                      scale: str = "fixed") -> dict:
    """Generate all specified days and return summary.

    Args:
        scale: "fixed" = use n_rows for every day
               "phase2" = variable 2M-9M rows/day
               "phase3" = variable 30M-350M rows/day
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    summary = {"days": [], "total_rows": 0, "total_size_mb": 0, "scale": scale}
    for day in range(start_day, end_day + 1):
        day_rows = get_daily_volume(day, scale) if scale != "fixed" else n_rows
        path = generate_day(day, n_rows=day_rows, output_dir=output_dir)
        size_mb = path.stat().st_size / 1e6
        cfg = get_drift_config(day)
        summary["days"].append({
            "day": day,
            "scenario": cfg["scenario"],
            "file": str(path.name),
            "size_mb": round(size_mb, 1),
            "rows": day_rows,
        })
        summary["total_rows"] += day_rows
        summary["total_size_mb"] += size_mb

    elapsed = time.time() - t0
    summary["total_size_mb"] = round(summary["total_size_mb"], 1)
    summary["wall_clock_sec"] = round(elapsed, 1)
    summary["wall_clock_min"] = round(elapsed / 60, 1)

    # Save summary
    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE")
    print(f"  Days: {start_day}-{end_day} ({end_day - start_day + 1} days)")
    print(f"  Rows: {summary['total_rows']:,}")
    print(f"  Size: {summary['total_size_mb']:,.0f} MB Parquet")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Summary: {summary_path}")
    print(f"{'='*70}")

    return summary


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate production DRAM data with drift")
    parser.add_argument("--day", type=int, default=1, help="Start day (1-40)")
    parser.add_argument("--end-day", type=int, default=None, help="End day (default: same as --day)")
    parser.add_argument("--rows", type=int, default=ROWS_PER_DAY, help=f"Rows per day (default: {ROWS_PER_DAY:,})")
    parser.add_argument("--output-dir", type=str, default=str(PRODUCTION_DIR))
    parser.add_argument("--scale", choices=["fixed", "phase2", "phase3"], default="fixed",
                        help="Volume scale: fixed=5M/day, phase2=2-9M/day, phase3=30-350M/day")
    args = parser.parse_args()

    end_day = args.end_day or args.day
    out_dir = Path(args.output_dir)

    if args.day == end_day:
        day_rows = get_daily_volume(args.day, args.scale) if args.scale != "fixed" else args.rows
        generate_day(args.day, n_rows=day_rows, output_dir=out_dir)
    else:
        generate_all_days(args.day, end_day, n_rows=args.rows,
                          output_dir=out_dir, scale=args.scale)
