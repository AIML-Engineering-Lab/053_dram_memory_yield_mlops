"""
P053 — Simulation Results Visualizer
=====================================
Reads simulation_timeline.json + drift reports and generates
LinkedIn-ready plots in assets/ directory.

Usage:
    python -m src.plot_simulation_results

Generates:
    p53_33_drift_timeline.png       — 40-day PSI heatmap + events
    p53_34_retrain_story.png        — Model version transitions + AUC timeline
    p53_35_failure_rate_drift.png   — Failure % climbing over 40 days
    p53_36_pandas_vs_spark.png      — The money demo (OOM vs success)
    p53_37_simulation_summary.png   — Executive summary card
    p53_38_psi_waterfall.png        — PSI per feature across key days
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ASSETS_DIR, DATA_DIR, SIMULATION

TIMELINE_PATH = DATA_DIR / "simulation_timeline.json"
DRIFT_REPORT_DIR = DATA_DIR / "drift_reports"

# ── Color palette (matches dashboard) ──
BLUE = "#1D4ED8"
LIGHT_BLUE = "#3B82F6"
GREEN = "#059669"
ORANGE = "#F59E0B"
RED = "#EF4444"
GRAY = "#6B7280"
DARK_BG = "#0d1117"
CARD_BG = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#c9d1d9"


def load_timeline() -> dict:
    with open(TIMELINE_PATH) as f:
        return json.load(f)


def load_drift_reports() -> dict:
    """Load all drift_day_XX.json files into a dict keyed by day number."""
    reports = {}
    for p in sorted(DRIFT_REPORT_DIR.glob("drift_day_*.json")):
        with open(p) as f:
            data = json.load(f)
            reports[data["analysis_day"]] = data
    return reports


# ═══════════════════════════════════════════════════════════════
# PLOT 1: 40-Day Drift Timeline Heatmap
# ═══════════════════════════════════════════════════════════════

def plot_drift_timeline(timeline: dict, drift_reports: dict):
    """PSI heatmap across 40 days × 6 features + event annotations."""
    features = [
        "test_temp_c", "cell_leakage_fa", "retention_time_ms",
        "gate_oxide_thickness_a", "vt_shift_mv", "trcd_ns",
    ]
    feature_labels = [
        "Temperature (°C)", "Cell Leakage (fA)", "Retention (ms)",
        "Gate Oxide (Å)", "Vt Shift (mV)", "tRCD (ns)",
    ]

    days = sorted(drift_reports.keys())
    if not days:
        print("  ⚠ No drift reports found, skipping drift timeline")
        return

    # Build PSI matrix
    psi_matrix = np.zeros((len(features), len(days)))
    for j, day in enumerate(days):
        rpt = drift_reports[day]
        for i, feat in enumerate(features):
            psi_matrix[i, j] = rpt.get("feature_psi", {}).get(feat, 0.0)

    fig, ax = plt.subplots(figsize=(16, 5), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)

    # Custom colormap: green → yellow → red
    cmap = LinearSegmentedColormap.from_list("drift",
        ["#064e3b", "#059669", "#fbbf24", "#ef4444", "#7f1d1d"])

    im = ax.imshow(psi_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=0.5,
                   interpolation="nearest")

    # X axis = days
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels([str(d) for d in days], fontsize=7, color=TEXT_COLOR)
    ax.set_xlabel("Simulation Day", fontsize=10, color=TEXT_COLOR)

    # Y axis = features
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(feature_labels, fontsize=9, color=TEXT_COLOR)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("PSI", color=TEXT_COLOR, fontsize=10)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=8)

    # Add threshold lines in colorbar
    for val, label, color in [(0.1, "Warning", ORANGE), (0.2, "Critical", RED)]:
        cbar.ax.axhline(y=val, color=color, linewidth=1.5, linestyle="--")

    # Annotate key events
    sim_start = datetime.strptime(SIMULATION["start_date"], "%Y-%m-%d")
    event_days_in_drift = {d: i for i, d in enumerate(days)}

    events_to_mark = []
    for d_record in timeline["days"]:
        day = d_record["day"]
        if day not in event_days_in_drift:
            continue
        idx = event_days_in_drift[day]
        for evt in d_record.get("events", []):
            if "RETRAIN" in evt:
                events_to_mark.append((idx, "★ RETRAIN", GREEN))
            elif "BAD_MODEL" in evt:
                events_to_mark.append((idx, "BAD DEPLOY", RED))
            elif "ROLLBACK" in evt:
                events_to_mark.append((idx, "ROLLBACK", ORANGE))

    for idx, label, color in events_to_mark:
        ax.axvline(x=idx, color=color, linewidth=1.5, linestyle="--", alpha=0.7)
        ax.annotate(label, (idx, -0.8), fontsize=7, color=color,
                    fontweight="bold", ha="center", annotation_clip=False)

    ax.set_title("40-Day Production Drift — PSI per Feature",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=15)

    plt.tight_layout()
    out = ASSETS_DIR / "p53_33_drift_timeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  ✓ {out.name}")


# ═══════════════════════════════════════════════════════════════
# PLOT 2: Failure Rate Over 40 Days
# ═══════════════════════════════════════════════════════════════

def plot_failure_rate(timeline: dict):
    """Show failure % climbing over 40 days with scenario annotations."""
    days_data = timeline["days"]
    if not days_data:
        print("  ⚠ No timeline data, skipping failure rate")
        return

    # Read each day's parquet to get fail rate
    from src.streaming_data_generator import PRODUCTION_DIR

    day_nums = []
    fail_rates = []
    dates = []
    scenarios = []

    for d in days_data:
        day = d["day"]
        pq_path = PRODUCTION_DIR / f"day_{day:02d}.parquet"
        if not pq_path.exists():
            continue

        df = pd.read_parquet(pq_path, columns=["is_fail"])
        rate = df["is_fail"].mean() * 100
        day_nums.append(day)
        fail_rates.append(rate)
        dates.append(d.get("date", ""))
        scenarios.append(d["scenario"])

    if not day_nums:
        print("  ⚠ No parquet files found, skipping failure rate")
        return

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)

    # Color bars by scenario severity
    colors = []
    for s in scenarios:
        s_lower = s.lower()
        if "retrain" in s_lower:
            colors.append(GREEN)
        elif "bad" in s_lower:
            colors.append(RED)
        elif "sudden" in s_lower or "worsening" in s_lower:
            colors.append(RED)
        elif "drift" in s_lower or "threshold" in s_lower or "continued" in s_lower:
            colors.append(ORANGE)
        elif "false" in s_lower:
            colors.append("#A855F7")  # Purple
        elif "recover" in s_lower or "post" in s_lower:
            colors.append(GREEN)
        else:
            colors.append(LIGHT_BLUE)

    ax.bar(day_nums, fail_rates, color=colors, alpha=0.85, edgecolor="none")

    # Horizontal reference line at baseline (Day 1)
    if fail_rates:
        ax.axhline(y=fail_rates[0], color=GRAY, linestyle="--", alpha=0.5, linewidth=1)
        ax.annotate(f"Baseline: {fail_rates[0]:.2f}%",
                    (1, fail_rates[0] + 0.05), fontsize=8, color=GRAY)

    # Mark retrain day
    for d in days_data:
        if any("RETRAIN" in e for e in d.get("events", [])):
            ax.axvline(x=d["day"], color=GREEN, linewidth=2, linestyle="--", alpha=0.8)
            ax.annotate("★ RETRAIN", (d["day"], max(fail_rates) * 0.95),
                        fontsize=9, color=GREEN, fontweight="bold", ha="center",
                        rotation=90)

    ax.set_xlabel("Simulation Day", fontsize=10, color=TEXT_COLOR)
    ax.set_ylabel("Failure Rate (%)", fontsize=10, color=TEXT_COLOR)
    ax.set_title("DRAM Die Failure Rate — 40-Day Production Simulation",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.spines[:].set_color(GRID_COLOR)

    # Legend
    legend_items = [
        mpatches.Patch(color=LIGHT_BLUE, label="Steady State"),
        mpatches.Patch(color=ORANGE, label="Drift/Threshold"),
        mpatches.Patch(color=RED, label="Critical/Bad Deploy"),
        mpatches.Patch(color=GREEN, label="Retrain/Recovery"),
        mpatches.Patch(color="#A855F7", label="False Alarm"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8,
              facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    out = ASSETS_DIR / "p53_35_failure_rate_drift.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  ✓ {out.name}")


# ═══════════════════════════════════════════════════════════════
# PLOT 3: Model Version Timeline + Events
# ═══════════════════════════════════════════════════════════════

def plot_retrain_story(timeline: dict):
    """Model lifecycle: v1 → retrain → v2 → bad deploy → rollback."""
    days_data = timeline["days"]
    if not days_data:
        return

    fig, ax = plt.subplots(figsize=(14, 4), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)

    day_nums = [d["day"] for d in days_data]
    model_versions = [d["model_version"] for d in days_data]

    # Map model versions to numeric values for plotting
    version_map = {}
    idx = 0
    for v in model_versions:
        if v not in version_map:
            version_map[v] = idx
            idx += 1

    version_nums = [version_map[v] for v in model_versions]
    version_colors = [LIGHT_BLUE if "v1" in v else GREEN if "v2" in v
                      else ORANGE for v in model_versions]

    # Plot as step function
    ax.fill_between(day_nums, version_nums, step="mid", alpha=0.3, color=LIGHT_BLUE)
    ax.step(day_nums, version_nums, where="mid", color=LIGHT_BLUE, linewidth=2)

    # Mark key events
    for d in days_data:
        for evt in d.get("events", []):
            if "RETRAIN" in evt:
                ax.axvline(x=d["day"], color=GREEN, linewidth=2, linestyle="--")
                ax.annotate("★ RETRAIN", (d["day"], max(version_nums) + 0.3),
                            fontsize=10, color=GREEN, fontweight="bold", ha="center")
            elif "BAD_MODEL" in evt:
                ax.axvline(x=d["day"], color=RED, linewidth=2, linestyle="--")
                ax.annotate("⚠ BAD DEPLOY", (d["day"], max(version_nums) + 0.5),
                            fontsize=9, color=RED, fontweight="bold", ha="center")
            elif "ROLLBACK" in evt:
                ax.annotate("↩ ROLLBACK", (d["day"], max(version_nums) + 0.1),
                            fontsize=9, color=ORANGE, fontweight="bold", ha="center")

    ax.set_yticks(range(len(version_map)))
    ax.set_yticklabels(list(version_map.keys()), fontsize=9, color=TEXT_COLOR)
    ax.set_xlabel("Simulation Day", fontsize=10, color=TEXT_COLOR)
    ax.set_title("Model Lifecycle — Automated Retrain, Bad Deploy & Rollback",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.spines[:].set_color(GRID_COLOR)

    plt.tight_layout()
    out = ASSETS_DIR / "p53_34_retrain_story.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  ✓ {out.name}")


# ═══════════════════════════════════════════════════════════════
# PLOT 4: PSI Waterfall — Key Days Comparison
# ═══════════════════════════════════════════════════════════════

def plot_psi_waterfall(drift_reports: dict):
    """PSI per feature for Days 1, 10, 20, 30, 31, 39."""
    key_days = [9, 15, 20, 25, 30, 35, 39]
    key_days = [d for d in key_days if d in drift_reports]

    if not key_days:
        print("  ⚠ No drift reports for key days, skipping PSI waterfall")
        return

    features = ["test_temp_c", "cell_leakage_fa", "retention_time_ms",
                "gate_oxide_thickness_a", "vt_shift_mv", "trcd_ns"]
    feat_short = ["Temp", "Leakage", "Retention", "Gate Ox", "Vt Shift", "tRCD"]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)

    x = np.arange(len(features))
    width = 0.8 / len(key_days)
    day_colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(key_days)))

    for i, day in enumerate(key_days):
        rpt = drift_reports[day]
        psi_vals = [rpt.get("feature_psi", {}).get(f, 0) for f in features]
        offset = (i - len(key_days) / 2 + 0.5) * width
        ax.bar(x + offset, psi_vals, width, label=f"Day {day}",
               color=day_colors[i], alpha=0.85, edgecolor="none")

    # Threshold lines
    ax.axhline(y=0.1, color=ORANGE, linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axhline(y=0.2, color=RED, linewidth=1.5, linestyle="--", alpha=0.7)
    ax.annotate("Warning (0.1)", (len(features) - 0.5, 0.105), fontsize=8, color=ORANGE)
    ax.annotate("Critical (0.2)", (len(features) - 0.5, 0.205), fontsize=8, color=RED)

    ax.set_xticks(x)
    ax.set_xticklabels(feat_short, fontsize=10, color=TEXT_COLOR)
    ax.set_ylabel("PSI", fontsize=10, color=TEXT_COLOR)
    ax.set_title("Feature Drift (PSI) — Key Simulation Days",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.spines[:].set_color(GRID_COLOR)
    ax.legend(loc="upper left", fontsize=8, ncol=2,
              facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    out = ASSETS_DIR / "p53_38_psi_waterfall.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  ✓ {out.name}")


# ═══════════════════════════════════════════════════════════════
# PLOT 5: Executive Summary Card
# ═══════════════════════════════════════════════════════════════

def plot_simulation_summary(timeline: dict, drift_reports: dict):
    """Executive summary — key numbers in a single card-style image."""
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    days_data = timeline["days"]
    total_rows = timeline["total_days"] * timeline["rows_per_day"]
    total_size = sum(d.get("parquet_mb", 0) for d in days_data)
    retrain_count = len(timeline.get("retrain_events", []))
    n_critical_days = sum(1 for d in drift_reports.values() if d.get("features_critical", 0) >= 3)
    elapsed_min = timeline.get("total_elapsed_min", 0)

    # Title
    ax.text(0.5, 0.95, "P053 — 40-Day Production Simulation Results",
            fontsize=20, fontweight="bold", color=LIGHT_BLUE,
            ha="center", va="top", transform=ax.transAxes)

    sim_start_date = timeline.get("simulation_start_date", SIMULATION["start_date"])
    sim_end_date = timeline.get("simulation_end_date", "")
    ax.text(0.5, 0.89, f"Timeline: {sim_start_date} → {sim_end_date}",
            fontsize=11, color=GRAY, ha="center", va="top", transform=ax.transAxes)

    # KPI cards
    kpis = [
        (f"{total_rows:,.0f}", "Total Records", LIGHT_BLUE),
        (f"{total_size / 1000:.1f} GB", "Parquet Data", LIGHT_BLUE),
        (f"{timeline['total_days']}", "Simulation Days", LIGHT_BLUE),
        (f"{elapsed_min:.0f} min", "Wall Clock", GREEN),
        (f"{retrain_count}", "Auto-Retrains", GREEN),
        (f"{n_critical_days}", "Critical Drift Days", RED),
    ]

    for i, (value, label, color) in enumerate(kpis):
        col = i % 3
        row = i // 3
        x = 0.18 + col * 0.32
        y = 0.72 - row * 0.22

        # Card background
        rect = plt.Rectangle((x - 0.12, y - 0.06), 0.24, 0.16,
                              facecolor=CARD_BG, edgecolor=GRID_COLOR,
                              linewidth=1, transform=ax.transAxes,
                              clip_on=False, zorder=2)
        ax.add_patch(rect)

        ax.text(x, y + 0.04, value, fontsize=22, fontweight="bold",
                color=color, ha="center", va="center", transform=ax.transAxes, zorder=3)
        ax.text(x, y - 0.03, label, fontsize=9, color=GRAY,
                ha="center", va="center", transform=ax.transAxes, zorder=3)

    # Scenario timeline bar at bottom
    ax.text(0.5, 0.30, "40-Day Scenario Timeline", fontsize=12,
            fontweight="bold", color=TEXT_COLOR, ha="center", transform=ax.transAxes)

    scenario_colors = {
        "STEADY STATE": LIGHT_BLUE, "FALSE ALARM": "#A855F7",
        "AUTO-RECOVER": GREEN, "GRADUAL DRIFT": ORANGE,
        "SUDDEN SHIFT": RED, "THRESHOLD #1": ORANGE,
        "CONTINUED DRIFT": ORANGE, "THRESHOLD #2": ORANGE,
        "WORSENING": RED, "★ RETRAIN": GREEN,
        "POST-RETRAIN": GREEN, "2ND DRIFT CYCLE": ORANGE,
        "BAD DEPLOY": RED, "RECOVERY": GREEN,
    }

    bar_y = 0.22
    bar_width = 0.018
    for d in days_data:
        day = d["day"]
        x_pos = 0.06 + (day - 1) * bar_width
        scenario = d["scenario"]
        color = scenario_colors.get(scenario, GRAY)
        rect = plt.Rectangle((x_pos, bar_y), bar_width * 0.9, 0.04,
                              facecolor=color, alpha=0.8,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        if day % 5 == 0 or day == 1:
            ax.text(x_pos + bar_width * 0.45, bar_y - 0.02, str(day),
                    fontsize=6, color=GRAY, ha="center", transform=ax.transAxes)

    # Footer
    ax.text(0.5, 0.05, "AIML Engineering Lab — Cloud-Scale Memory Yield Predictor",
            fontsize=9, color=GRAY, ha="center", transform=ax.transAxes)

    out = ASSETS_DIR / "p53_37_simulation_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  ✓ {out.name}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*50}")
    print("P053 — Generating Simulation Plots")
    print(f"{'='*50}\n")

    if not TIMELINE_PATH.exists():
        print(f"ERROR: {TIMELINE_PATH} not found.")
        print("Run the simulation first: python -m src.run_simulation --fast")
        sys.exit(1)

    timeline = load_timeline()
    drift_reports = load_drift_reports()

    print(f"Loaded timeline: {timeline['total_days']} days, "
          f"{timeline['rows_per_day']:,} rows/day")
    print(f"Drift reports: {len(drift_reports)} days\n")

    plot_drift_timeline(timeline, drift_reports)
    plot_failure_rate(timeline)
    plot_retrain_story(timeline)
    plot_psi_waterfall(drift_reports)
    plot_simulation_summary(timeline, drift_reports)

    print(f"\n✅ All plots saved to {ASSETS_DIR}/")


if __name__ == "__main__":
    main()
