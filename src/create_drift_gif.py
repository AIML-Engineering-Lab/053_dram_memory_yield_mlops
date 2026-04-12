"""
Create animated GIF of 40-day drift timeline for LinkedIn post.
Reveals day-by-day, highlights key events: staleness blocks, retrain, rollback, recovery.
Output: docs/drift_timeline.gif
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

# ── Load simulation data ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "data" / "simulation_timeline.json") as f:
    timeline = json.load(f)

FEATURES = [
    "retention_time_ms",
    "gate_oxide_thickness_a",
    "test_temp_c",
    "vt_shift_mv",
    "cell_leakage_fa",
    "trcd_ns",
]

FEATURE_LABELS = [
    "Retention Time",
    "Gate Oxide Thickness",
    "Test Temperature",
    "Vt Shift",
    "Cell Leakage",
    "tRCD",
]

DAYS = 40

# ── Build PSI matrix (features × days) ─────────────────────────────────────
psi_matrix = np.zeros((len(FEATURES), DAYS))
for day_data in timeline["days"]:
    d = day_data["day"] - 1  # 0-indexed
    if "drift" in day_data:
        psi_vals = day_data["drift"].get("feature_psi", {})
        for i, feat in enumerate(FEATURES):
            if feat in psi_vals:
                psi_matrix[i, d] = psi_vals[feat]

# ── Event annotations ───────────────────────────────────────────────────────
EVENTS = {
    9:  {"label": "⚠ False Alarm", "color": "#FFD700"},
    17: {"label": "🔒 Staleness Block", "color": "#FF8C00"},
    30: {"label": "🔁 RETRAIN v2", "color": "#00E5FF"},
    39: {"label": "🔴 CANARY FAIL → ROLLBACK", "color": "#FF4444"},
    40: {"label": "✅ System Recovered", "color": "#00FF88"},
}

# Day status for bottom band
def day_status(d):  # d is 1-indexed
    if d <= 8:
        return "steady", "#2E5A88"
    elif d == 9:
        return "false alarm", "#FFD700"
    elif 10 <= d <= 16:
        return "gradual drift", "#FFC107"
    elif 17 <= d <= 29:
        return "staleness block", "#FF6B35"
    elif d == 30:
        return "RETRAIN", "#00E5FF"
    elif 31 <= d <= 38:
        return "v2 serving", "#4CAF50"
    elif d == 39:
        return "ROLLBACK", "#FF4444"
    elif d == 40:
        return "recovered", "#00FF88"
    return "steady", "#2E5A88"


# ── PSI thresholds ──────────────────────────────────────────────────────────
PSI_MAX_DISPLAY = 7.0

# ── Color palette ───────────────────────────────────────────────────────────
BG_COLOR   = "#0a0e1a"
PANEL_COLOR = "#111827"
TITLE_COLOR = "#E0E6F0"
GRID_COLOR  = "#1f2937"

CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "drift_cmap",
    [
        (0.00, "#0D3B66"),   # 0.0  — stable deep blue
        (0.03, "#1565C0"),   # 0.2  — info blue
        (0.07, "#0288D1"),   # ~0.5 — notice
        (0.14, "#F9A825"),   # ~1.0 — warning yellow
        (0.28, "#EF6C00"),   # ~2.0 — warning orange
        (0.57, "#C62828"),   # ~4.0 — critical red
        (1.00, "#880E4F"),   # 7.0  — extreme magenta
    ],
)

# ── Setup figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 7), facecolor=BG_COLOR)
gs = fig.add_gridspec(
    3, 1,
    height_ratios=[0.55, 0.30, 0.15],
    hspace=0.12,
    top=0.88, bottom=0.06, left=0.12, right=0.92
)

ax_heat  = fig.add_subplot(gs[0])   # heatmap
ax_psi   = fig.add_subplot(gs[1])   # max-PSI line
ax_band  = fig.add_subplot(gs[2])   # event band

for ax in [ax_heat, ax_psi, ax_band]:
    ax.set_facecolor(PANEL_COLOR)

# ── Static colorbar axis ─────────────────────────────────────────────────────
cb_ax = fig.add_axes([0.935, 0.36, 0.013, 0.50])
cb_ax.set_facecolor(BG_COLOR)
norm = matplotlib.colors.Normalize(vmin=0, vmax=PSI_MAX_DISPLAY)
sm   = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cb_ax)
cbar.set_label("PSI", color=TITLE_COLOR, fontsize=8, labelpad=6)
cbar.ax.yaxis.set_tick_params(color=TITLE_COLOR, labelsize=7)
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TITLE_COLOR)
cbar.outline.set_edgecolor(GRID_COLOR)

# Threshold lines drawn in colorbar
for thresh, label, color in [(0.2, "warning", "#F9A825"), (0.5, "critical", "#C62828")]:
    cb_ax.axhline(y=thresh / PSI_MAX_DISPLAY, color=color, lw=1.0, alpha=0.7)

# ── Title  ────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.94, "40-Day DRAM Production Drift Timeline",
         ha='center', va='bottom', color=TITLE_COLOR,
         fontsize=13, fontweight='bold')
fig.text(0.5, 0.905, "Feature-level PSI heatmap · HybridTransformerCNN · 5M rows/day · 200M total",
         ha='center', va='bottom', color="#8899BB", fontsize=8.5)

# Day counter annotation
day_text = fig.text(0.92, 0.905, "Day 01 / 40",
                    ha='right', va='bottom', color="#A0C4FF",
                    fontsize=9, fontweight='bold')

# Legend patches
legend_items = [
    mpatches.Patch(color="#2E5A88", label="Stable (v1)"),
    mpatches.Patch(color="#FFD700", label="False Alarm"),
    mpatches.Patch(color="#FF6B35", label="Staleness Block"),
    mpatches.Patch(color="#00E5FF", label="Retrain → v2"),
    mpatches.Patch(color="#4CAF50", label="v2 Serving"),
    mpatches.Patch(color="#FF4444", label="Rollback"),
    mpatches.Patch(color="#00FF88", label="Recovered"),
]
fig.legend(handles=legend_items, loc='lower center',
           ncol=7, frameon=False,
           fontsize=7.5,
           labelcolor=TITLE_COLOR,
           bbox_to_anchor=(0.5, -0.01))

# ── Heatmap axes setup ────────────────────────────────────────────────────────
ax_heat.set_xlim(-0.5, DAYS - 0.5)
ax_heat.set_ylim(-0.5, len(FEATURES) - 0.5)
ax_heat.set_yticks(range(len(FEATURES)))
ax_heat.set_yticklabels(FEATURE_LABELS, color=TITLE_COLOR, fontsize=8)
# x-axis: show 1-indexed day numbers
x_tick_pos  = [0, 4, 9, 14, 19, 24, 29, 34, 39]
x_tick_lbls = ["1", "5", "10", "15", "20", "25", "30", "35", "40"]
ax_heat.set_xticks(x_tick_pos)
ax_heat.set_xticklabels(x_tick_lbls)
ax_heat.tick_params(axis='x', colors=TITLE_COLOR, labelsize=7.5)
ax_heat.tick_params(axis='y', colors=TITLE_COLOR, pad=3)
ax_heat.set_xlabel("Production Day", color=TITLE_COLOR, fontsize=8, labelpad=4)
for spine in ax_heat.spines.values():
    spine.set_edgecolor(GRID_COLOR)

# ── PSI line axes setup ───────────────────────────────────────────────────────
ax_psi.set_xlim(-0.5, DAYS - 0.5)
ax_psi.set_ylim(0, PSI_MAX_DISPLAY + 0.3)
ax_psi.set_ylabel("Max PSI", color=TITLE_COLOR, fontsize=7.5, labelpad=3)
ax_psi.tick_params(colors=TITLE_COLOR, labelsize=6.5)
ax_psi.axhline(0.2, color="#F9A825", lw=0.8, ls='--', alpha=0.5, label="warning 0.2")
ax_psi.axhline(0.5, color="#C62828", lw=0.8, ls='--', alpha=0.5, label="critical 0.5")
ax_psi.set_xticks([])
for spine in ax_psi.spines.values():
    spine.set_edgecolor(GRID_COLOR)
ax_psi.text(DAYS - 1, 0.22, "warn", color="#F9A825", fontsize=6, ha='right', va='bottom')
ax_psi.text(DAYS - 1, 0.52, "critical", color="#C62828", fontsize=6, ha='right', va='bottom')

# ── Band axes setup ───────────────────────────────────────────────────────────
ax_band.set_xlim(-0.5, DAYS - 0.5)
ax_band.set_ylim(0, 1)
ax_band.set_xticks(range(DAYS))
ax_band.set_xticklabels([str(i+1) if (i+1) % 5 == 0 or i == 0 else "" for i in range(DAYS)],
                         color=TITLE_COLOR, fontsize=6.5)
ax_band.set_yticks([])
ax_band.set_xlabel("Day", color=TITLE_COLOR, fontsize=7.5, labelpad=2)
for spine in ax_band.spines.values():
    spine.set_edgecolor(GRID_COLOR)

# ── Initialize drawing objects ────────────────────────────────────────────────
heatmap_cells = {}
for fi in range(len(FEATURES)):
    for di in range(DAYS):
        rect = plt.Rectangle([di - 0.5, fi - 0.5], 1, 1,
                              facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, lw=0.3)
        ax_heat.add_patch(rect)
        heatmap_cells[(fi, di)] = rect

band_rects = []
for di in range(DAYS):
    rect = plt.Rectangle([di - 0.5, 0], 1, 1,
                          facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, lw=0.3)
    ax_band.add_patch(rect)
    band_rects.append(rect)

# PSI line data
psi_line_data = np.zeros(DAYS)
max_psi_per_day = psi_matrix.max(axis=0)

(psi_line,) = ax_psi.plot([], [], color="#00E5FF", lw=1.5, alpha=0.9)
(psi_fill_holder,) = ax_psi.plot([], [], color="#00E5FF", alpha=0.0)  # placeholder

psi_fill = [None]  # mutable holder

# Event overlay in heatmap
event_annotations = {}

# ── Animation update ──────────────────────────────────────────────────────────
def update(frame):
    # frame goes 0..39
    nd = frame + 1  # days revealed so far

    for di in range(nd):
        d_status, d_color = day_status(di + 1)

        for fi in range(len(FEATURES)):
            psi_val = psi_matrix[fi, di]
            rgba = CMAP(norm(min(psi_val, PSI_MAX_DISPLAY)))
            # For days with no PSI data (days 1-8), show as very dark stable blue
            if psi_val == 0:
                rgba = CMAP(0.0)
            heatmap_cells[(fi, di)].set_facecolor(rgba)

        band_rects[di].set_facecolor(d_color)

    # Heatmap: redraw current day column with bright border
    if nd > 0:
        di = nd - 1
        for fi in range(len(FEATURES)):
            cell = heatmap_cells[(fi, di)]
            cell.set_edgecolor("#FFFFFF")
            cell.set_linewidth(0.8)
        # Remove border from previous day
        if nd > 1:
            di_prev = nd - 2
            for fi in range(len(FEATURES)):
                heatmap_cells[(fi, di_prev)].set_edgecolor(GRID_COLOR)
                heatmap_cells[(fi, di_prev)].set_linewidth(0.3)

    # PSI line
    psi_x = list(range(nd))
    psi_y = max_psi_per_day[:nd]
    psi_line.set_data(psi_x, psi_y)

    # Fill under PSI line
    if psi_fill[0] is not None:
        psi_fill[0].remove()
    if nd > 1:
        psi_fill[0] = ax_psi.fill_between(
            range(nd), max_psi_per_day[:nd], alpha=0.15, color="#00E5FF"
        )
    else:
        psi_fill[0] = None

    # Event annotations in heatmap
    for ann in list(event_annotations.values()):
        ann.remove()
    event_annotations.clear()

    if nd >= 30:
        ann = ax_heat.annotate(
            "▲ RETRAIN", xy=(29, len(FEATURES) - 0.3),
            xytext=(29, len(FEATURES) + 0.1),
            color="#00E5FF", fontsize=6.5, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color="#00E5FF", lw=1.0)
        )
        event_annotations[30] = ann

    if nd >= 39:
        ann2 = ax_heat.annotate(
            "▼ ROLLBACK", xy=(38, len(FEATURES) - 0.3),
            xytext=(38, len(FEATURES) + 0.1),
            color="#FF4444", fontsize=6.5, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color="#FF4444", lw=1.0)
        )
        event_annotations[39] = ann2

    if nd >= 40:
        ann3 = ax_heat.annotate(
            "✓ RECOVERED", xy=(39, len(FEATURES) - 0.3),
            xytext=(39, len(FEATURES) + 0.1),
            color="#00FF88", fontsize=6, ha='center', fontweight='bold',
        )
        event_annotations[40] = ann3

    # Update day counter
    day_text.set_text(f"Day {nd:02d} / 40")

    # Highlight today's band rect with white border
    for di in range(nd):
        band_rects[di].set_edgecolor(GRID_COLOR)
        band_rects[di].set_linewidth(0.3)
    band_rects[nd - 1].set_edgecolor("#FFFFFF")
    band_rects[nd - 1].set_linewidth(1.2)

    return list(heatmap_cells.values()) + band_rects + [psi_line, day_text]


anim = FuncAnimation(
    fig,
    update,
    frames=DAYS,
    interval=350,   # ms per frame  → ~14s for 40 frames
    blit=False,
    repeat=True,
    repeat_delay=2500,
)

# ── Save ─────────────────────────────────────────────────────────────────────
OUT_PATH = ROOT / "docs" / "drift_timeline.gif"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"Saving GIF to {OUT_PATH} ...")
writer = PillowWriter(fps=3, metadata={"loop": 0})
anim.save(str(OUT_PATH), writer=writer, dpi=110)
print(f"Done! Size: {OUT_PATH.stat().st_size / 1024:.0f} KB")
plt.close(fig)
