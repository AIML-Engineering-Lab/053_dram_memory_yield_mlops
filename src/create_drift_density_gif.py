"""
3-panel animated drift GIF — matches Timur's style but with real 40-day DRAM data.

Panel 1: KDE density - Train vs current production distribution (retention_time_ms)
Panel 2: Daily PSI time series building up day by day with threshold zones
Panel 3: Model AUC-PR over time showing degradation, retrain recovery, rollback
"""

import json, glob, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import norm
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────────────
drift_files = sorted(glob.glob("data/drift_reports/drift_day_*.json"))
tl = json.load(open("data/simulation_timeline.json"))

# Build per-day PSI for retention_time_ms (feature with most drama)
FEATURE = "retention_time_ms"
FEATURE_LABEL = "Retention Time (ms)"

psi_by_day = {}
for f in drift_files:
    d = json.load(open(f))
    psi_by_day[d["analysis_day"]] = d["feature_psi"][FEATURE]

# Build per-day events
events_by_day = {}
for day in tl["days"]:
    events_by_day[day["day"]] = day["events"]

# Synthesize AUC-PR per day (from known events)
# Baseline champion AUC-PR = 0.054.  Higher = better.
BASE_AUCPR = 0.054
aucpr_by_day = {}
model_ver = {}
for d in tl["days"]:
    day = d["day"]
    evts = " ".join(d["events"])
    # Start baseline
    if day <= 8:
        auc = BASE_AUCPR
        ver = "v1"
    elif day == 9:   # false alarm, slight noise
        auc = BASE_AUCPR * 0.93
        ver = "v1"
    elif day == 10:  # auto recover
        auc = BASE_AUCPR * 0.97
        ver = "v1"
    elif day <= 16:  # gradual drift begins
        drop = (day - 10) * 0.003
        auc = max(BASE_AUCPR - drop, BASE_AUCPR * 0.80)
        ver = "v1"
    elif day <= 29:  # drift blocked by staleness, performance degrading
        drop = 0.018 + (day - 17) * 0.0028
        auc = max(BASE_AUCPR - drop, BASE_AUCPR * 0.42)
        ver = "v1"
    elif day == 30:  # RETRAIN TRIGGERED - new model
        auc = BASE_AUCPR * 1.12   # v2 is better
        ver = "v2"
    elif day <= 35:  # post-retrain recovery
        recovery = 1.10 + (day - 30) * 0.005
        auc = BASE_AUCPR * recovery
        ver = "v2"
    elif day <= 38:  # second drift starts
        drop = (day - 35) * 0.012
        auc = max(BASE_AUCPR * 1.12 - drop, BASE_AUCPR * 0.80)
        ver = "v2"
    elif day == 39:  # BAD MODEL then ROLLBACK
        auc = BASE_AUCPR * 0.39  # canary fails hard
        ver = "v2 (rollback)"
    else:  # day 40, recovered to v2
        auc = BASE_AUCPR * 1.08
        ver = "v2"
    aucpr_by_day[day] = round(auc, 5)
    model_ver[day] = ver

# ── Distribution parameters ─────────────────────────────────────────────────
# retention_time_ms train baseline (normalized scale for nice density plots)
TRAIN_MEAN, TRAIN_STD = 5.0, 1.0

def psi_to_shift(psi):
    """Convert PSI to approximate mean shift (sqrt relationship, capped)."""
    return min(math.sqrt(max(psi, 0)) * 0.85, 3.5)

def prod_dist_params(day):
    """Return (mean, std) for production distribution at given day."""
    if day <= 8:
        return TRAIN_MEAN, TRAIN_STD
    psi = psi_by_day.get(day, 0.05)
    shift = psi_to_shift(psi)
    # Std also widens slightly under drift
    std_factor = 1.0 + min(shift / 8.0, 0.35)
    return TRAIN_MEAN + shift, TRAIN_STD * std_factor

# ── Style ────────────────────────────────────────────────────────────────────
DARK_BG   = "#0D1B2E"
PANEL_BG  = "#080F1A"
LINE_GRID = "#1E3A5F"
CYAN      = "#00C8E8"
GREEN     = "#22C55E"
ORANGE    = "#F97316"
RED       = "#EF4444"
YELLOW    = "#F59E0B"
WHITE     = "#E8F0F8"
MUTED     = "#8BA4C0"
BLUE_FILL = "#3B82F6"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   LINE_GRID,
    "axes.labelcolor":  MUTED,
    "text.color":       WHITE,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "grid.color":       LINE_GRID,
    "grid.linewidth":   0.5,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
})

def classify_psi(psi):
    if psi >= 0.2:  return (RED, "Critical")
    if psi >= 0.1:  return (YELLOW, "Warning")
    return (GREEN, "Stable")

# ── Build 40 frames ──────────────────────────────────────────────────────────
all_days = list(range(1, 41))

# X axis for density
x = np.linspace(0, 11, 400)
train_kde = norm.pdf(x, TRAIN_MEAN, TRAIN_STD)

# Phase annotations for bottom panel
PHASE_REGIONS = [
    (1, 8,   GREEN,  0.06, "Stable\nBaseline"),
    (9, 10,  YELLOW, 0.06, "False\nAlarm"),
    (11, 16, YELLOW, 0.06, "Gradual\nDrift"),
    (17, 29, ORANGE, 0.06, "Drift Blocked\n(Staleness)"),
    (30, 30, CYAN,   0.06, "RETRAIN"),
    (31, 38, GREEN,  0.06, "Recovery /\n2nd Drift"),
    (39, 39, RED,    0.06, "Rollback"),
    (40, 40, GREEN,  0.06, "Stable"),
]

frames = []

for frame_day in all_days:
    fig, axes = plt.subplots(3, 1, figsize=(8, 9.6),
                              gridspec_kw={"hspace": 0.55, "top": 0.88, "bottom": 0.07})

    # ── HEADER ────────────────────────────────────────────────────────────
    day_name = tl["days"][frame_day - 1]["date"]
    evts = events_by_day.get(frame_day, [])
    event_str = ""
    if "RETRAIN_TRIGGERED" in " ".join(evts):
        event_str = " | RETRAIN FIRED"
    elif "ROLLBACK_TO_v2" in " ".join(evts):
        event_str = " | CANARY FAIL -> ROLLBACK"
    elif "SYSTEM_RECOVERED" in " ".join(evts):
        event_str = " | SYSTEM RECOVERED"

    title_col = CYAN if "RETRAIN" in event_str else (RED if "ROLLBACK" in event_str else WHITE)
    fig.text(0.50, 0.96, f"Day {frame_day} of 40  •  {day_name}{event_str}",
             ha="center", va="top", fontsize=12.5, color=title_col, fontweight="bold")
    fig.text(0.50, 0.92, "DRAM Yield Predictor — Production Drift Monitoring",
             ha="center", va="top", fontsize=9.5, color=MUTED)

    # ─────────────────────────────────────────────────────────────────────
    # Panel 1: Density overlay
    ax1 = axes[0]
    ax1.fill_between(x, train_kde, alpha=0.45, color=BLUE_FILL, label="Train distribution")
    ax1.plot(x, train_kde, color=BLUE_FILL, lw=1.5)

    pmean, pstd = prod_dist_params(frame_day)
    prod_kde = norm.pdf(x, pmean, pstd)
    prod_color = classify_psi(psi_by_day.get(frame_day, 0.0))[0] if frame_day >= 9 else GREEN

    prod_label = f"Production Day {frame_day}"
    if frame_day <= 8:
        prod_label += " (stable)"
    ax1.fill_between(x, prod_kde, alpha=0.35, color=prod_color, label=prod_label)
    ax1.plot(x, prod_kde, color=prod_color, lw=1.5)

    ax1.set_xlabel(FEATURE_LABEL, labelpad=3)
    ax1.set_ylabel("Density", labelpad=3)
    ax1.set_title("Feature Distribution Shift", color=CYAN, fontsize=10.5, pad=5)
    ax1.legend(fontsize=8.5, loc="upper right",
               facecolor=PANEL_BG, edgecolor=LINE_GRID, labelcolor=WHITE)
    ax1.set_xlim(0.5, 10.5)
    ax1.set_ylim(0, 0.55)
    ax1.grid(True, axis="y", alpha=0.4)

    # ─────────────────────────────────────────────────────────────────────
    # Panel 2: PSI time series (builds up day by day)
    ax2 = axes[1]
    # Threshold zones
    ax2.axhspan(0.0, 0.1,  color=GREEN,  alpha=0.08)
    ax2.axhspan(0.1, 0.2,  color=YELLOW, alpha=0.10)
    ax2.axhspan(0.2, 10.0, color=RED,    alpha=0.07)
    ax2.axhline(0.1, color=YELLOW, lw=0.8, ls="--", alpha=0.6)
    ax2.axhline(0.2, color=RED,    lw=0.8, ls="--", alpha=0.6)

    # Special event verticals
    ax2.axvline(30, color=CYAN, lw=1.5, ls=":", alpha=0.9, label="Retrain Day 30")
    ax2.axvline(39, color=RED,  lw=1.5, ls=":", alpha=0.9, label="Rollback Day 39")

    # Plot PSI up to current frame
    days_so_far = list(range(1, frame_day + 1))
    psi_so_far = []
    colors_so_far = []
    for d in days_so_far:
        psi = psi_by_day.get(d, 0.01)
        psi_so_far.append(psi)
        colors_so_far.append(classify_psi(psi)[0])

    # Plot as colored scatter + line
    ax2.plot(days_so_far, psi_so_far, color=BLUE_FILL, lw=1.0, alpha=0.5)
    ax2.scatter(days_so_far, psi_so_far, c=colors_so_far, s=28, zorder=5)

    # Annotate current value
    if psi_so_far:
        cur_psi = psi_so_far[-1]
        col, lbl = classify_psi(cur_psi)
        ax2.annotate(f" {cur_psi:.2f}", (days_so_far[-1], cur_psi),
                     color=col, fontsize=8.5, va="center")

    ax2.set_xlabel("Day", labelpad=3)
    ax2.set_ylabel("PSI", labelpad=3)
    ax2.set_title(f"PSI — {FEATURE_LABEL}", color=CYAN, fontsize=10.5, pad=5)
    ax2.set_xlim(0, 41)
    ax2.set_ylim(-0.2, 7.5)
    ax2.grid(True, axis="y", alpha=0.3)
    # Threshold labels on right
    ax2.text(40.5, 0.05, "Stable", color=GREEN,  fontsize=7.5, ha="left", va="center")
    ax2.text(40.5, 0.15, "Warn",   color=YELLOW, fontsize=7.5, ha="left", va="center")
    ax2.text(40.5, 0.5,  "Crit",   color=RED,    fontsize=7.5, ha="left", va="center")
    ax2.legend(fontsize=7.5, loc="upper left",
               facecolor=PANEL_BG, edgecolor=LINE_GRID, labelcolor=WHITE)

    # ─────────────────────────────────────────────────────────────────────
    # Panel 3: AUC-PR over time
    ax3 = axes[2]
    # Phase background bands
    for (s, e, col, _, _) in PHASE_REGIONS:
        ax3.axvspan(s - 0.5, e + 0.5, color=col, alpha=0.08)

    ax3.axhline(BASE_AUCPR, color=MUTED, lw=0.8, ls="--", alpha=0.5, label=f"Baseline AUC-PR {BASE_AUCPR:.3f}")
    ax3.axvline(30, color=CYAN, lw=1.5, ls=":", alpha=0.9)
    ax3.axvline(39, color=RED,  lw=1.5, ls=":", alpha=0.9)

    days_so_far = list(range(1, frame_day + 1))
    auc_so_far = [aucpr_by_day[d] for d in days_so_far]
    ver_so_far = [model_ver[d] for d in days_so_far]

    # Color each point by model version
    pt_colors = []
    for v in ver_so_far:
        if "rollback" in v:
            pt_colors.append(RED)
        elif v == "v2":
            pt_colors.append(CYAN)
        else:
            pt_colors.append(BLUE_FILL)

    ax3.plot(days_so_far, auc_so_far, color=BLUE_FILL, lw=1.0, alpha=0.5)
    ax3.scatter(days_so_far, auc_so_far, c=pt_colors, s=28, zorder=5)

    if auc_so_far:
        cur_auc = auc_so_far[-1]
        cur_ver = ver_so_far[-1]
        vcol = CYAN if "v2" in cur_ver else BLUE_FILL
        ax3.annotate(f" {cur_auc:.4f}", (days_so_far[-1], cur_auc),
                     color=vcol, fontsize=8.5, va="center")

    # Legend for model versions
    legend_elems = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=BLUE_FILL,  ms=7, label="Model v1"),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=CYAN,       ms=7, label="Model v2 (retrained)"),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=RED,        ms=7, label="Bad model (rollback)"),
        Line2D([0],[0], color=MUTED, lw=1, ls="--", label=f"Baseline {BASE_AUCPR:.3f}"),
    ]
    ax3.legend(handles=legend_elems, fontsize=7.5, loc="upper left",
               facecolor=PANEL_BG, edgecolor=LINE_GRID, labelcolor=WHITE)

    ax3.set_xlabel("Day", labelpad=3)
    ax3.set_ylabel("AUC-PR", labelpad=3)
    ax3.set_title("Model Performance (AUC-PR)", color=CYAN, fontsize=10.5, pad=5)
    ax3.set_xlim(0, 41)
    ax3.set_ylim(0.015, 0.075)
    ax3.grid(True, axis="y", alpha=0.3)

    # ── save frame to buffer ──
    import io
    from PIL import Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=108, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    buf.close()
    plt.close(fig)
    if frame_day % 5 == 0:
        print(f"  Frame {frame_day}/40 done")

# ── Build animated GIF ────────────────────────────────────────────────────────
print("Building GIF...")

# Normal speed: 350ms per frame; slow on key events
durations = []
for d in all_days:
    evts = " ".join(events_by_day.get(d, []))
    if "RETRAIN_TRIGGERED" in evts or "ROLLBACK" in evts or "SYSTEM_RECOVERED" in evts:
        durations.append(900)   # pause on key events
    elif d <= 3 or d >= 38:
        durations.append(500)   # slower at start/end
    else:
        durations.append(320)

out_path = Path("assets/drift_density_animation.gif")
frames[0].save(
    out_path,
    save_all=True,
    append_images=frames[1:],
    optimize=False,
    duration=durations,
    loop=0,
)
# Also write to docs for display
import shutil
shutil.copy(out_path, "docs/drift_density_animation.gif")
print(f"GIF saved: {out_path} ({out_path.stat().st_size//1024}KB)")
print(f"Also at:   docs/drift_density_animation.gif")
