"""
P053 — EDA Visualizations for DRAM STDF Dataset
Generates 4 publication-quality plots showing real-world data issues:
  1. Class distribution (extreme imbalance)
  2. Missing value heatmap by feature
  3. Outlier scatter (leakage vs retention)
  4. Spatial wafer map (edge-die failure pattern)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

BLUE = "#2563EB"
DARK_BLUE = "#1E3A5F"
GREEN = "#059669"
RED = "#DC2626"
AMBER = "#F59E0B"
GRAY = "#6B7280"
LIGHT_BG = "#F8FAFC"


def load_sample():
    """Load the 50K sample parquet (or fall back to full train)."""
    sample_path = DATA / "dram_stdf_sample.parquet"
    if sample_path.exists():
        return pd.read_parquet(sample_path)
    train_path = DATA / "dram_stdf_train.parquet"
    if train_path.exists():
        return pd.read_parquet(train_path).sample(50_000, random_state=42)
    print("ERROR: No data found. Run data_generator.py --sample first.")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# PLOT 1: Class Distribution
# ═══════════════════════════════════════════════════════════════════
def plot_class_distribution(df):
    """Bar chart showing extreme class imbalance + annotation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), 
                                     gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("white")

    # Left: Full scale bar chart
    counts = df["is_fail"].value_counts().sort_index()
    pass_count = counts.get(0, 0)
    fail_count = counts.get(1, 0)
    total = len(df)
    true_fail = df["is_fail_true"].sum() if "is_fail_true" in df.columns else fail_count

    bars = ax1.bar(
        ["PASS (Bin 1)", "FAIL (Bin 0)"],
        [pass_count, fail_count],
        color=[GREEN, RED],
        width=0.5,
        edgecolor="white",
        linewidth=2,
    )
    ax1.set_ylabel("Count")
    ax1.set_title("Class Distribution — Observed Labels", pad=12)
    ax1.set_facecolor(LIGHT_BG)

    # Annotate counts
    ax1.text(0, pass_count + total * 0.01, f"{pass_count:,}\n({100*pass_count/total:.2f}%)",
             ha="center", va="bottom", fontweight="bold", fontsize=12, color=GREEN)
    ax1.text(1, fail_count + total * 0.01, f"{fail_count:,}\n({100*fail_count/total:.2f}%)",
             ha="center", va="bottom", fontweight="bold", fontsize=12, color=RED)

    # Add imbalance ratio annotation
    ratio = pass_count / max(fail_count, 1)
    ax1.annotate(
        f"Imbalance Ratio: 1:{int(ratio)}",
        xy=(1, fail_count), xytext=(0.5, total * 0.6),
        fontsize=13, fontweight="bold", color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, lw=2),
        ha="center",
    )

    # Right: Label noise breakdown (pie)
    if "is_fail_true" in df.columns and "label_is_noisy" in df.columns:
        noisy_count = df["label_is_noisy"].sum()
        correct_fail = true_fail - noisy_count  # approximate
        noise_flip_to_fail = ((df["is_fail"] == 1) & (df["is_fail_true"] == 0)).sum()
        noise_flip_to_pass = ((df["is_fail"] == 0) & (df["is_fail_true"] == 1)).sum()

        sizes = [pass_count - noise_flip_to_fail, true_fail, noise_flip_to_fail, noise_flip_to_pass]
        labels = [
            f"True Pass\n{pass_count - noise_flip_to_fail:,}",
            f"True Fail\n{true_fail:,}",
            f"Pass>Fail (noise)\n{noise_flip_to_fail:,}",
            f"Fail>Pass (noise)\n{noise_flip_to_pass:,}",
        ]
        colors = [GREEN, RED, AMBER, "#94A3B8"]
        # Filter out zero counts
        nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
        if nonzero:
            sizes, labels, colors = zip(*nonzero)
            wedges, texts = ax2.pie(
                sizes, labels=None, colors=colors,
                startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2},
            )
            ax2.legend(wedges, labels, loc="center left", bbox_to_anchor=(0.85, 0.5),
                       fontsize=9, frameon=False)
        ax2.set_title("Label Noise Breakdown", pad=12)
    else:
        ax2.axis("off")

    plt.tight_layout()
    out = ASSETS / "p53_01_eda_class_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════
# PLOT 2: Missing Value Heatmap
# ═══════════════════════════════════════════════════════════════════
def plot_missing_heatmap(df):
    """Heatmap showing missing value patterns across features."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                     gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor("white")

    # Get features with missing values
    miss_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    miss_features = miss_pct[miss_pct > 0]

    if len(miss_features) == 0:
        ax1.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=16)
        ax1.axis("off")
        ax2.axis("off")
    else:
        # Left: Horizontal bar chart of missing percentages
        bars = ax1.barh(
            range(len(miss_features)),
            miss_features.values,
            color=[AMBER if v < 5 else RED for v in miss_features.values],
            edgecolor="white",
            linewidth=1,
        )
        ax1.set_yticks(range(len(miss_features)))
        ax1.set_yticklabels(miss_features.index, fontsize=10)
        ax1.set_xlabel("Missing (%)")
        ax1.set_title("Missing Value Rates by Feature", pad=12)
        ax1.set_facecolor(LIGHT_BG)

        # Add percentage labels
        for i, (feat, pct) in enumerate(miss_features.items()):
            ax1.text(pct + 0.15, i, f"{pct:.1f}%", va="center", fontsize=10,
                     fontweight="bold", color=RED if pct >= 5 else AMBER)

        # Right: Missing pattern matrix (sample of 500 rows × missing features)
        sample_idx = np.random.default_rng(42).choice(len(df), size=min(500, len(df)), replace=False)
        sample_idx.sort()
        missing_matrix = df.iloc[sample_idx][miss_features.index].isnull().astype(int).values

        cmap = LinearSegmentedColormap.from_list("miss", ["white", RED], N=2)
        ax2.imshow(missing_matrix, aspect="auto", cmap=cmap, interpolation="nearest")
        ax2.set_xticks(range(len(miss_features)))
        ax2.set_xticklabels([f.split("_")[0][:6] for f in miss_features.index],
                            rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Sample Index")
        ax2.set_title("Missing Pattern (500 rows)", pad=12)

    plt.tight_layout()
    out = ASSETS / "p53_02_eda_missing_values.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════
# PLOT 3: Outlier Scatter (Leakage vs Retention)
# ═══════════════════════════════════════════════════════════════════
def plot_outlier_scatter(df):
    """Scatter plot highlighting outliers in leakage vs retention space."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    # Drop NaN for these features
    sub = df[["cell_leakage_fa", "retention_time_ms", "trcd_ns", "is_fail", "test_temp_c"]].dropna()

    # Left: Leakage vs Retention colored by fail
    pass_mask = sub["is_fail"] == 0
    fail_mask = sub["is_fail"] == 1

    ax1.scatter(sub.loc[pass_mask, "cell_leakage_fa"],
                sub.loc[pass_mask, "retention_time_ms"],
                c=GREEN, alpha=0.15, s=5, label="Pass", rasterized=True)
    ax1.scatter(sub.loc[fail_mask, "cell_leakage_fa"],
                sub.loc[fail_mask, "retention_time_ms"],
                c=RED, alpha=0.9, s=30, label="Fail", zorder=5,
                edgecolors="white", linewidth=0.5)

    # Mark outlier region
    leakage_q3 = sub["cell_leakage_fa"].quantile(0.75)
    leakage_iqr = sub["cell_leakage_fa"].quantile(0.75) - sub["cell_leakage_fa"].quantile(0.25)
    outlier_thresh = leakage_q3 + 1.5 * leakage_iqr
    ax1.axvline(outlier_thresh, color=AMBER, linestyle="--", linewidth=2, alpha=0.8,
                label=f"Outlier threshold ({outlier_thresh:.0f} fA)")

    ax1.set_xlabel("Cell Leakage (fA)")
    ax1.set_ylabel("Retention Time (ms)")
    ax1.set_title("Leakage vs Retention — Outliers & Failures", pad=12)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_facecolor(LIGHT_BG)

    # Right: tRCD timing with outliers highlighted
    trcd = sub["trcd_ns"]
    q1, q3 = trcd.quantile(0.25), trcd.quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (trcd < q1 - 1.5 * iqr) | (trcd > q3 + 1.5 * iqr)
    normal_mask = ~outlier_mask

    ax2.hist(trcd[normal_mask], bins=80, color=BLUE, alpha=0.7, label="Normal", density=True)
    ax2.hist(trcd[outlier_mask], bins=30, color=RED, alpha=0.8, label="Outliers (equipment glitch)", density=True)
    ax2.axvline(q3 + 1.5 * iqr, color=AMBER, linestyle="--", linewidth=2, label="IQR boundary")
    ax2.set_xlabel("tRCD (ns)")
    ax2.set_ylabel("Density")
    ax2.set_title("tRCD Timing Distribution — 2% Equipment Outliers", pad=12)
    ax2.legend(fontsize=9)
    ax2.set_facecolor(LIGHT_BG)

    plt.tight_layout()
    out = ASSETS / "p53_03_eda_outliers.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════
# PLOT 4: Spatial Wafer Map (Edge-Die Failure Pattern)
# ═══════════════════════════════════════════════════════════════════
def plot_spatial_wafer_map(df):
    """Wafer map showing die positions colored by fail rate — edge effect."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    # Create wafer map: aggregate fail rate by (die_x, die_y)
    grid = df.groupby(["die_x", "die_y"]).agg(
        fail_rate=("is_fail", "mean"),
        count=("is_fail", "count"),
    ).reset_index()

    # Only plot cells with enough samples
    grid = grid[grid["count"] >= 3]

    # Left: Wafer map colored by fail rate
    # Normalize die positions to wafer coordinates
    max_x = df["die_x"].max()
    max_y = df["die_y"].max()
    cx, cy = max_x / 2, max_y / 2

    scatter = ax1.scatter(
        grid["die_x"], grid["die_y"],
        c=grid["fail_rate"] * 100,
        cmap="RdYlGn_r",
        s=40,
        marker="s",
        edgecolors="white",
        linewidth=0.3,
        vmin=0,
        vmax=max(grid["fail_rate"].quantile(0.95) * 100, 0.5),
    )
    # Draw wafer circle
    r = max(cx, cy) * 1.05
    wafer = Circle((cx, cy), r, fill=False, edgecolor=GRAY, linewidth=2, linestyle="--")
    ax1.add_patch(wafer)
    ax1.set_xlim(-2, max_x + 2)
    ax1.set_ylim(-2, max_y + 2)
    ax1.set_aspect("equal")
    ax1.set_xlabel("Die X")
    ax1.set_ylabel("Die Y")
    ax1.set_title("Wafer Fail Rate Map — Edge Die Effect", pad=12)
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar.set_label("Fail Rate (%)")
    ax1.set_facecolor(LIGHT_BG)

    # Right: Fail rate vs edge distance (binned)
    df_clean = df[["edge_distance", "is_fail"]].dropna()
    bins = np.linspace(0, 1, 21)
    df_clean["edge_bin"] = pd.cut(df_clean["edge_distance"], bins=bins)
    bin_stats = df_clean.groupby("edge_bin", observed=True).agg(
        fail_rate=("is_fail", "mean"),
        count=("is_fail", "count"),
    ).reset_index()
    bin_stats["bin_mid"] = bin_stats["edge_bin"].apply(lambda x: x.mid)

    # Only plot bins with enough samples
    bin_stats = bin_stats[bin_stats["count"] >= 20]

    ax2.bar(
        bin_stats["bin_mid"], bin_stats["fail_rate"] * 100,
        width=0.045,
        color=[RED if m > 0.7 else AMBER if m > 0.5 else BLUE 
               for m in bin_stats["bin_mid"]],
        edgecolor="white",
        linewidth=1,
    )
    ax2.set_xlabel("Edge Distance (0 = center, 1 = edge)")
    ax2.set_ylabel("Fail Rate (%)")
    ax2.set_title("Fail Rate vs Wafer Position — 3× at Edge", pad=12)
    ax2.set_facecolor(LIGHT_BG)

    # Annotate edge zone
    ax2.axvspan(0.7, 1.05, alpha=0.1, color=RED, label="Edge zone (>0.7)")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = ASSETS / "p53_04_eda_spatial_wafer.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  P053 — EDA Plot Generation")
    print("=" * 60)

    df = load_sample()
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    print()

    plot_class_distribution(df)
    plot_missing_heatmap(df)
    plot_outlier_scatter(df)
    plot_spatial_wafer_map(df)

    print()
    print(f"  All EDA plots saved to {ASSETS}/")
    print("=" * 60)
