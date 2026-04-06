"""
P053 — Model Comparison: Baselines vs Hybrid Transformer-CNN
=============================================================
Generates a comprehensive comparison visualization showing:
1. Bar chart: F1 / AUC-PR / Recall side-by-side
2. Confusion matrix grid (all 4 models × val)
3. Summary table image (for the report)
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"


def load_all_metrics():
    """Load baseline + hybrid metrics."""
    with open(DATA / "baseline_metrics_sample.json") as f:
        baselines = json.load(f)

    with open(DATA / "hybrid_model_metrics_sample.json") as f:
        hybrid = json.load(f)

    # Normalize into unified format
    models = {}
    for name, m in baselines.items():
        models[name] = m

    models["Hybrid Transformer-CNN"] = hybrid["results"]

    return models


def plot_comparison_bars(models):
    """Side-by-side bar chart of key metrics (val split)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    names = list(models.keys())
    colors = ["#3B82F6", "#F59E0B", "#8B5CF6", "#059669"]
    metrics_to_plot = [
        ("f1", "F1 Score", axes[0]),
        ("auc_pr", "AUC-PR", axes[1]),
        ("recall", "Recall", axes[2]),
    ]

    for metric_key, metric_name, ax in metrics_to_plot:
        vals = [models[n]["val"][metric_key] for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
        ax.set_title(metric_name, fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.4 + 0.01)
        ax.grid(axis="y", alpha=0.3)
        ax.set_facecolor("#F8FAFC")

    plt.suptitle("Model Comparison — Validation Set (50K Sample Data)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(ASSETS / "p53_21_model_comparison_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: p53_21_model_comparison_bars.png")


def plot_confusion_matrices(models):
    """2×2 confusion matrix grid for all models (val split)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    names = list(models.keys())

    for idx, (name, ax) in enumerate(zip(names, axes.ravel())):
        m = models[name]["val"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])

        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                        fontsize=14, fontweight="bold", color=color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pass", "Fail"])
        ax.set_yticklabels(["Pass", "Fail"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name}\nF1={m['f1']:.4f} | Recall={m['recall']:.4f}",
                     fontsize=11, fontweight="bold")

    plt.suptitle("Confusion Matrices — Validation Set (50K Sample)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(ASSETS / "p53_22_confusion_matrix_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: p53_22_confusion_matrix_comparison.png")


def plot_summary_table(models):
    """Publication-quality summary table as image."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    names = list(models.keys())
    metrics = ["f1", "auc_pr", "auc_roc", "precision", "recall", "tp", "fp", "fn"]
    headers = ["Model", "F1", "AUC-PR", "AUC-ROC", "Precision", "Recall", "TP", "FP", "FN"]

    cell_data = []
    for name in names:
        m = models[name]["val"]
        row = [name]
        for metric in metrics:
            v = m[metric]
            row.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        cell_data.append(row)

    # Find best F1 row
    f1_vals = [models[n]["val"]["f1"] for n in names]
    best_idx = np.argmax(f1_vals)

    table = ax.table(cellText=cell_data, colLabels=headers, loc="center",
                     cellLoc="center", colWidths=[0.20] + [0.10]*8)

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#1D4ED8")
        cell.set_text_props(color="white", fontweight="bold")

    # Highlight best row
    for j in range(len(headers)):
        cell = table[best_idx + 1, j]
        cell.set_facecolor("#F0FDF4")

    # Alternate row colors
    for i in range(len(names)):
        if i == best_idx:
            continue
        for j in range(len(headers)):
            cell = table[i + 1, j]
            cell.set_facecolor("#F8FAFC" if i % 2 == 0 else "white")

    plt.title("Model Comparison — Validation Metrics (50K Sample)\n",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ASSETS / "p53_23_model_comparison_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: p53_23_model_comparison_table.png")


def plot_cross_split(models):
    """Show generalization: val vs test vs unseen for each model."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    names = list(models.keys())
    splits = ["val", "test", "unseen"]
    split_colors = {"val": "#3B82F6", "test": "#F59E0B", "unseen": "#EF4444"}
    bar_width = 0.2

    for ax, metric_key, metric_name in zip(axes,
        ["f1", "auc_pr", "recall"],
        ["F1 Score", "AUC-PR", "Recall"]):

        x = np.arange(len(names))
        for si, split in enumerate(splits):
            vals = []
            for name in names:
                vals.append(models[name][split][metric_key])
            offset = (si - 1) * bar_width
            bars = ax.bar(x + offset, vals, bar_width, label=split.title(),
                         color=split_colors[split], alpha=0.85, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=7.5)
        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_facecolor("#F8FAFC")

    plt.suptitle("Generalization: Val → Test → Unseen (50K Sample)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(ASSETS / "p53_24_generalization_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: p53_24_generalization_comparison.png")


if __name__ == "__main__":
    print("=" * 60)
    print("P053 — Model Comparison Plots")
    print("=" * 60)

    models = load_all_metrics()
    print(f"\nModels: {list(models.keys())}")

    plot_comparison_bars(models)
    plot_confusion_matrices(models)
    plot_summary_table(models)
    plot_cross_split(models)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (Val Set — 50K Sample)")
    print("=" * 60)
    for name in models:
        m = models[name]["val"]
        print(f"  {name:30s} F1={m['f1']:.4f}  AUC-PR={m['auc_pr']:.4f}  Recall={m['recall']:.4f}")

    best_name = max(models, key=lambda n: models[n]["val"]["f1"])
    print(f"\n  BEST: {best_name} (by F1 on val)")
    print("\n  NARRATIVE: All models struggle with 1:150 class imbalance on 50K sample.")
    print("  This validates the PAIN POINT — we need full 16M dataset + focal loss + Transformer.")
