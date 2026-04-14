"""
Animated GIF showing HybridTransformerCNN architecture data flow.

6 frames showing data flowing through:
1. Raw features input (6 DRAM features)
2. Per-feature tokenization (embedding layer)
3. 1D-CNN local pattern extraction
4. Transformer self-attention global interactions
5. Classification head
6. Final prediction (pass/fail)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import io
from PIL import Image
from pathlib import Path

# ── Style ──
DARK_BG  = "#0D1B2E"
PANEL_BG = "#080F1A"
CYAN     = "#00C8E8"
GREEN    = "#22C55E"
ORANGE   = "#F97316"
RED      = "#EF4444"
YELLOW   = "#F59E0B"
WHITE    = "#E8F0F8"
MUTED    = "#8BA4C0"
BLUE     = "#3B82F6"
PURPLE   = "#A78BFA"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   DARK_BG,
    "text.color":       WHITE,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})

FEATURES = ["test_temp_c", "cell_leakage_fa", "retention_time_ms",
            "gate_oxide_a", "vt_shift_mv", "trcd_ns"]
FEAT_COLORS = [BLUE, CYAN, GREEN, ORANGE, YELLOW, PURPLE]


def draw_box(ax, x, y, w, h, color, label, sublabel=None, alpha=0.85, fontsize=12):
    """Draw a rounded rectangle with label."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.08", linewidth=1.5,
                         edgecolor=color, facecolor=color + "20",
                         zorder=3)
    ax.add_patch(box)
    ax.text(x, y + (0.015 if sublabel else 0), label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color, zorder=4)
    if sublabel:
        ax.text(x, y - 0.04, sublabel, ha="center", va="center",
                fontsize=9, color=MUTED, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color=MUTED, style="-|>", lw=1.5):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color,
                            lw=lw, mutation_scale=15, zorder=2)
    ax.add_patch(arrow)


def make_frame(stage, highlight_stage):
    """Create one frame of the architecture animation.
    stage 0: all gray, title only
    stage 1: features highlighted
    stage 2: tokenization highlighted
    stage 3: CNN highlighted
    stage 4: transformer highlighted
    stage 5: classifier highlighted
    stage 6: all lit, output
    """
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    # Title
    ax.text(0.5, 0.98, "HybridTransformerCNN Architecture", ha="center", va="top",
            fontsize=18, fontweight="bold", color=CYAN)
    ax.text(0.5, 0.935, "Per-feature tokenization + 1D-CNN + Transformer self-attention",
            ha="center", va="top", fontsize=11, color=MUTED)

    # Layout: vertical flow, 6 layers
    layers = [
        {"y": 0.83, "label": "Raw DRAM Features", "sub": "6 sensor measurements per chip",
         "color": BLUE, "stage": 1},
        {"y": 0.68, "label": "Per-Feature Tokenization", "sub": "Each feature gets its own learned embedding",
         "color": CYAN, "stage": 2},
        {"y": 0.53, "label": "1D-CNN (Local Patterns)", "sub": "Sliding filters capture adjacent feature interactions",
         "color": GREEN, "stage": 3},
        {"y": 0.38, "label": "Transformer Self-Attention", "sub": "Global interactions between ALL features",
         "color": ORANGE, "stage": 4},
        {"y": 0.23, "label": "Classification Head", "sub": "MLP with dropout",
         "color": PURPLE, "stage": 5},
        {"y": 0.08, "label": "PASS / FAIL", "sub": "FocalLoss trained output",
         "color": RED, "stage": 6},
    ]

    for i, layer in enumerate(layers):
        is_active = highlight_stage >= layer["stage"]
        is_current = highlight_stage == layer["stage"]
        color = layer["color"] if is_active else "#2A3A50"
        alpha = 1.0 if is_current else (0.7 if is_active else 0.3)

        # Main box
        box_w = 0.55 if is_current else 0.48
        box_h = 0.085 if is_current else 0.075
        facecolor = color + ("30" if is_current else "15" if is_active else "08")
        box = FancyBboxPatch((0.5 - box_w/2, layer["y"] - box_h/2), box_w, box_h,
                             boxstyle="round,pad=0.06", linewidth=2 if is_current else 1,
                             edgecolor=color if is_active else "#1E3A5F",
                             facecolor=facecolor, zorder=3)
        ax.add_patch(box)

        label_color = color if is_active else "#3A4A60"
        sub_color = MUTED if is_active else "#2A3A50"
        ax.text(0.5, layer["y"] + 0.012, layer["label"], ha="center", va="center",
                fontsize=13 if is_current else 11, fontweight="bold", color=label_color, zorder=4)
        ax.text(0.5, layer["y"] - 0.022, layer["sub"], ha="center", va="center",
                fontsize=9, color=sub_color, zorder=4)

        # Arrow to next layer
        if i < len(layers) - 1:
            next_y = layers[i+1]["y"]
            arrow_color = color if is_active and highlight_stage > layer["stage"] else "#1E3A5F"
            draw_arrow(ax, 0.5, layer["y"] - box_h/2 - 0.005,
                       0.5, next_y + 0.075/2 + 0.005, color=arrow_color, lw=1.8)

    # ── Stage-specific annotations ──
    if highlight_stage == 1:
        # Show 6 feature boxes
        for j, (feat, col) in enumerate(zip(FEATURES, FEAT_COLORS)):
            fx = 0.10 + j * 0.155
            fy = 0.83
            ax.text(fx, fy + 0.06, feat, ha="center", va="bottom",
                    fontsize=7.5, color=col, fontweight="600", rotation=30, zorder=5)
            ax.plot([fx], [fy + 0.055], marker="o", ms=6, color=col, zorder=5)

    elif highlight_stage == 2:
        # Show embedding arrows
        ax.text(0.88, 0.68, "Each feature\ngets its own\nembedding dim", ha="center", va="center",
                fontsize=9, color=CYAN, style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=CYAN+"15", edgecolor=CYAN+"40"))

    elif highlight_stage == 3:
        # CNN filter visualization
        ax.text(0.88, 0.53, "1D conv filters\nslide across\nfeature vector\n\nCaptures:\nleakage + retention\ncombined signals", ha="center", va="center",
                fontsize=9, color=GREEN, style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=GREEN+"15", edgecolor=GREEN+"40"))
        # Show filter sliding visual
        for k in range(3):
            ax.add_patch(FancyBboxPatch((0.12 + k*0.06, 0.53 - 0.02), 0.12, 0.04,
                                        boxstyle="round,pad=0.02", linewidth=1,
                                        edgecolor=GREEN + "80", facecolor=GREEN + "10",
                                        zorder=5))

    elif highlight_stage == 4:
        # Attention visualization
        ax.text(0.88, 0.38, "Self-attention\nconnects ALL\nfeatures globally\n\nvt_shift interacts\nwith trcd_ns even\nthough far apart", ha="center", va="center",
                fontsize=9, color=ORANGE, style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=ORANGE+"15", edgecolor=ORANGE+"40"))
        # Attention lines
        pts = [(0.22 + i*0.08, 0.38) for i in range(6)]
        for a in range(6):
            for b in range(a+1, 6):
                ax.plot([pts[a][0], pts[b][0]], [pts[a][1]+0.01, pts[b][1]-0.01],
                        color=ORANGE, alpha=0.25, lw=0.8, zorder=5)

    elif highlight_stage == 5:
        ax.text(0.88, 0.23, "Dense layers\nwith dropout\nfor regularization", ha="center", va="center",
                fontsize=9, color=PURPLE, style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PURPLE+"15", edgecolor=PURPLE+"40"))

    elif highlight_stage == 6:
        # Show final result
        ax.text(0.5, 0.015, "FocalLoss: easy majority samples get near-zero weight\n"
                "Model focuses on hard-to-detect defective chips",
                ha="center", va="center", fontsize=10, color=RED,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=RED+"15", edgecolor=RED+"40"))

    # Bottom credits
    ax.text(0.5, -0.02, "317K parameters  |  FT-Transformer / TabNet family  |  Outperforms pure ML baselines on sensor data",
            ha="center", va="top", fontsize=8.5, color=MUTED)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    buf.seek(0)
    frame = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return frame


# ── Build frames ──
print("Generating HybridTransformerCNN architecture GIF...")
frames = []
durations = []

# Stage 0: blank intro
frames.append(make_frame(0, 0))
durations.append(800)

# Stages 1-6: progressive highlight
for s in range(1, 7):
    frames.append(make_frame(s, s))
    durations.append(1200 if s <= 4 else 1000)  # longer on CNN/Transformer

# Stage 6 again (hold on final)
frames.append(make_frame(6, 6))
durations.append(1500)

# Full reveal (all lit)
# Make a final "all active" frame
fig, ax = plt.subplots(figsize=(10, 7.5))
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.axis("off")
ax.text(0.5, 0.98, "HybridTransformerCNN Architecture", ha="center", va="top",
        fontsize=18, fontweight="bold", color=CYAN)
ax.text(0.5, 0.935, "Per-feature tokenization + 1D-CNN + Transformer self-attention",
        ha="center", va="top", fontsize=11, color=MUTED)

layers_final = [
    (0.83, "Raw DRAM Features", "6 sensor measurements", BLUE),
    (0.68, "Per-Feature Tokenization", "Learned embeddings per feature", CYAN),
    (0.53, "1D-CNN (Local Patterns)", "Adjacent feature interactions", GREEN),
    (0.38, "Transformer Self-Attention", "Global feature interactions", ORANGE),
    (0.23, "Classification Head", "MLP + dropout", PURPLE),
    (0.08, "PASS / FAIL", "FocalLoss output", RED),
]
for i, (y, lbl, sub, col) in enumerate(layers_final):
    box = FancyBboxPatch((0.25, y - 0.04), 0.50, 0.08,
                         boxstyle="round,pad=0.06", linewidth=2,
                         edgecolor=col, facecolor=col + "25", zorder=3)
    ax.add_patch(box)
    ax.text(0.5, y + 0.01, lbl, ha="center", va="center",
            fontsize=12, fontweight="bold", color=col, zorder=4)
    ax.text(0.5, y - 0.02, sub, ha="center", va="center",
            fontsize=9, color=MUTED, zorder=4)
    if i < len(layers_final) - 1:
        draw_arrow(ax, 0.5, y - 0.045, 0.5, layers_final[i+1][0] + 0.045,
                   color=col, lw=2)

ax.text(0.5, -0.02, "317K parameters  |  FT-Transformer / TabNet family  |  Outperforms pure ML baselines on sensor data",
        ha="center", va="top", fontsize=8.5, color=MUTED)

buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
            facecolor=DARK_BG, edgecolor="none")
buf.seek(0)
all_frame = Image.open(buf).copy()
buf.close()
plt.close(fig)

frames.append(all_frame)
durations.append(2000)

# Save
out = Path("assets/hybrid_architecture_animation.gif")
frames[0].save(out, save_all=True, append_images=frames[1:],
               optimize=False, duration=durations, loop=0)
print(f"GIF saved: {out} ({out.stat().st_size // 1024}KB, {len(frames)} frames)")

# Also save final frame as static PNG for carousel
all_frame.save("assets/hybrid_architecture_static.png", "PNG")
print(f"Static PNG saved: assets/hybrid_architecture_static.png")
