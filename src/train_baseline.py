"""
P053 — Baseline Model Training for DRAM Yield Prediction
=========================================================
Trains and compares 3 baseline models:
1. Logistic Regression (classic baseline)
2. XGBoost with scale_pos_weight (tree-based, handles imbalance)
3. LightGBM with is_unbalance (fast gradient boosting)

Key principle: Use AUC-PR (not AUC-ROC) as primary metric.
At 1:150 class imbalance, ROC-AUC is deceptively high even for bad models.
Precision-Recall AUC is the meaningful metric for rare-event detection.

Usage:
    python src/train_baseline.py                 # Train on sample
    python src/train_baseline.py --full          # Train on full 16M
    python src/train_baseline.py --full --smote  # Train on SMOTE-balanced data
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix, classification_report,
    roc_auc_score,
)
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
SRC = ROOT / "src"

# ═══════════════════════════════════════════════════════════════
# Metrics Helper
# ═══════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    """Compute all relevant metrics for imbalanced classification."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_pr = average_precision_score(y_true, y_pred_proba)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_pr": float(auc_pr),
        "auc_roc": float(auc_roc),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
    }


def find_best_threshold(y_true, y_pred_proba, metric="f1"):
    """Find threshold that maximizes F1 on validation data."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    return thresholds[min(best_idx, len(thresholds) - 1)]


# ═══════════════════════════════════════════════════════════════
# Model Training
# ═══════════════════════════════════════════════════════════════

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Logistic Regression baseline with class weights."""
    print("\n  Training Logistic Regression (class_weight='balanced')...")
    t0 = time.time()
    
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="saga",
        C=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    
    y_val_proba = model.predict_proba(X_val)[:, 1]
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.1f}s")
    
    return model, y_val_proba


def train_xgboost(X_train, y_train, X_val, y_val):
    """XGBoost with scale_pos_weight for class imbalance."""
    print("\n  Training XGBoost (scale_pos_weight)...")
    t0 = time.time()
    
    # Calculate imbalance ratio
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos = neg / max(pos, 1)
    print(f"  scale_pos_weight = {scale_pos:.1f}")
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="aucpr",
        early_stopping_rounds=20,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    y_val_proba = model.predict_proba(X_val)[:, 1]
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.1f}s (best iteration: {model.best_iteration})")
    
    return model, y_val_proba


def train_lightgbm(X_train, y_train, X_val, y_val):
    """LightGBM with is_unbalance flag."""
    print("\n  Training LightGBM (is_unbalance=True)...")
    t0 = time.time()
    
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        is_unbalance=True,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    
    y_val_proba = model.predict_proba(X_val)[:, 1]
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.1f}s (best iteration: {model.best_iteration_})")
    
    return model, y_val_proba


# ═══════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════

def plot_precision_recall_curves(results, split_name="val"):
    """Precision-Recall curves for all models — THE metric for imbalanced data."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    colors = {"LogisticRegression": "#3B82F6", "XGBoost": "#059669", "LightGBM": "#F59E0B"}
    
    for name, res in results.items():
        y_true = res[f"y_{split_name}"]
        y_proba = res[f"y_{split_name}_proba"]
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        auc_pr = average_precision_score(y_true, y_proba)
        
        ax.plot(recall, precision, color=colors.get(name, "#6B7280"), linewidth=2.5,
                label=f"{name} (AUC-PR={auc_pr:.4f})")
    
    # Baseline: random classifier
    baseline = results[list(results.keys())[0]][f"y_{split_name}"].mean()
    ax.axhline(y=baseline, color="#D1D5DB", linestyle="--", linewidth=1,
               label=f"Random (baseline={baseline:.4f})")
    
    ax.set_xlabel("Recall (Defect Detection Rate)", fontsize=12)
    ax.set_ylabel("Precision (Detection Accuracy)", fontsize=12)
    ax.set_title(f"Precision-Recall Curves — {split_name.title()} Set\n"
                 "AUC-PR is THE metric for 1:150 class imbalance (NOT ROC-AUC)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#F8FAFC")
    
    plt.tight_layout()
    plt.savefig(ASSETS / f"p53_15_baseline_pr_curves_{split_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: p53_15_baseline_pr_curves_{split_name}.png")


def plot_confusion_matrices(results, split_name="val"):
    """Confusion matrices for all baseline models."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx]
        metrics = res[f"metrics_{split_name}"]
        cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
        
        # Normalized confusion matrix
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                text_color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})",
                       ha="center", va="center", fontsize=11, color=text_color,
                       fontweight="bold")
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pass", "Fail"])
        ax.set_yticklabels(["Pass", "Fail"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name}\nF1={metrics['f1']:.3f} | Recall={metrics['recall']:.3f}",
                     fontsize=11, fontweight="bold")
    
    plt.suptitle(f"Confusion Matrices — Baseline Models ({split_name.title()} Set)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(ASSETS / f"p53_16_baseline_confusion_{split_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: p53_16_baseline_confusion_{split_name}.png")


def plot_feature_importance(model, feature_names, model_name="XGBoost"):
    """Feature importance plot from tree model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return
    
    indices = np.argsort(importances)[-15:]  # Top 15
    
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(indices)))
    ax.barh(range(len(indices)), importances[indices], color=colors, edgecolor="white")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"{model_name} — Top 15 Feature Importances\n"
                 "Edge distance & leakage dominate (matches semiconductor physics)",
                 fontsize=13, fontweight="bold")
    ax.set_facecolor("#F8FAFC")
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ASSETS / f"p53_17_baseline_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: p53_17_baseline_feature_importance.png")


def plot_model_comparison_table(results, feature_names):
    """Generate comparison table as image."""
    metrics_list = []
    for name, res in results.items():
        m_val = res["metrics_val"]
        m_test = res["metrics_test"]
        m_unseen = res["metrics_unseen"]
        metrics_list.append({
            "Model": name,
            "Val F1": f"{m_val['f1']:.4f}",
            "Val AUC-PR": f"{m_val['auc_pr']:.4f}",
            "Val Recall": f"{m_val['recall']:.4f}",
            "Test F1": f"{m_test['f1']:.4f}",
            "Test AUC-PR": f"{m_test['auc_pr']:.4f}",
            "Unseen F1": f"{m_unseen['f1']:.4f}",
            "Unseen AUC-PR": f"{m_unseen['auc_pr']:.4f}",
        })
    
    df_metrics = pd.DataFrame(metrics_list)
    
    fig, ax = plt.subplots(figsize=(14, 2.5))
    ax.axis("off")
    table = ax.table(cellText=df_metrics.values, colLabels=df_metrics.columns,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Color header
    for j in range(len(df_metrics.columns)):
        table[0, j].set_facecolor("#1D4ED8")
        table[0, j].set_text_props(color="white", fontweight="bold")
    
    # Highlight best values per column
    for j in range(1, len(df_metrics.columns)):
        vals = [float(df_metrics.iloc[i, j]) for i in range(len(df_metrics))]
        best_i = np.argmax(vals)
        table[best_i + 1, j].set_facecolor("#F0FDF4")
        table[best_i + 1, j].set_text_props(fontweight="bold")
    
    plt.title("Baseline Model Comparison — All Splits",
              fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(ASSETS / "p53_18_baseline_comparison_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: p53_18_baseline_comparison_table.png")


# ═══════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════

def train_baselines(use_full=False, use_smote=False):
    """Train all baseline models and generate comparison plots."""
    t0 = time.time()
    print("=" * 70)
    print("P053 — DRAM Yield Predictor: Baseline Model Training")
    print("=" * 70)

    # ─── Load preprocessed data ───
    suffix = "_full" if use_full else "_sample"
    smote_suffix = "_smote" if use_smote else ""
    data_path = DATA / f"preprocessed{suffix}{smote_suffix}.npz"
    
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run preprocess.py first.")
        return
    
    print(f"\n[LOAD] Loading preprocessed data: {data_path.name}")
    data = np.load(data_path, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    X_unseen = data["X_unseen"]
    y_unseen = data["y_unseen"]
    feature_names = list(data["feature_names"])
    
    print(f"  Train: {X_train.shape} ({y_train.sum():,} fails, {100*y_train.mean():.2f}%)")
    print(f"  Val:   {X_val.shape} ({y_val.sum():,} fails)")
    print(f"  Test:  {X_test.shape} ({y_test.sum():,} fails)")
    print(f"  Unseen:{X_unseen.shape} ({y_unseen.sum():,} fails)")
    
    # ─── Train models ───
    results = {}
    
    # 1. Logistic Regression
    lr_model, lr_val_proba = train_logistic_regression(X_train, y_train, X_val, y_val)
    lr_test_proba = lr_model.predict_proba(X_test)[:, 1]
    lr_unseen_proba = lr_model.predict_proba(X_unseen)[:, 1]
    
    # Find best threshold on val
    lr_threshold = find_best_threshold(y_val, lr_val_proba)
    print(f"  Best threshold (F1): {lr_threshold:.4f}")
    
    results["LogisticRegression"] = {
        "model": lr_model,
        "y_val": y_val, "y_val_proba": lr_val_proba,
        "y_test": y_test, "y_test_proba": lr_test_proba,
        "y_unseen": y_unseen, "y_unseen_proba": lr_unseen_proba,
        "metrics_val": compute_metrics(y_val, lr_val_proba, lr_threshold),
        "metrics_test": compute_metrics(y_test, lr_test_proba, lr_threshold),
        "metrics_unseen": compute_metrics(y_unseen, lr_unseen_proba, lr_threshold),
    }
    
    # 2. XGBoost
    xgb_model, xgb_val_proba = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_unseen_proba = xgb_model.predict_proba(X_unseen)[:, 1]
    
    xgb_threshold = find_best_threshold(y_val, xgb_val_proba)
    print(f"  Best threshold (F1): {xgb_threshold:.4f}")
    
    results["XGBoost"] = {
        "model": xgb_model,
        "y_val": y_val, "y_val_proba": xgb_val_proba,
        "y_test": y_test, "y_test_proba": xgb_test_proba,
        "y_unseen": y_unseen, "y_unseen_proba": xgb_unseen_proba,
        "metrics_val": compute_metrics(y_val, xgb_val_proba, xgb_threshold),
        "metrics_test": compute_metrics(y_test, xgb_test_proba, xgb_threshold),
        "metrics_unseen": compute_metrics(y_unseen, xgb_unseen_proba, xgb_threshold),
    }
    
    # 3. LightGBM
    lgb_model, lgb_val_proba = train_lightgbm(X_train, y_train, X_val, y_val)
    lgb_test_proba = lgb_model.predict_proba(X_test)[:, 1]
    lgb_unseen_proba = lgb_model.predict_proba(X_unseen)[:, 1]
    
    lgb_threshold = find_best_threshold(y_val, lgb_val_proba)
    print(f"  Best threshold (F1): {lgb_threshold:.4f}")
    
    results["LightGBM"] = {
        "model": lgb_model,
        "y_val": y_val, "y_val_proba": lgb_val_proba,
        "y_test": y_test, "y_test_proba": lgb_test_proba,
        "y_unseen": y_unseen, "y_unseen_proba": lgb_unseen_proba,
        "metrics_val": compute_metrics(y_val, lgb_val_proba, lgb_threshold),
        "metrics_test": compute_metrics(y_test, lgb_test_proba, lgb_threshold),
        "metrics_unseen": compute_metrics(y_unseen, lgb_unseen_proba, lgb_threshold),
    }
    
    # ─── Print comparison ───
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<22} {'Val F1':>8} {'Val AUC-PR':>11} {'Val Recall':>11} "
          f"{'Test F1':>8} {'Unseen F1':>10}")
    print("-" * 70)
    for name, res in results.items():
        mv = res["metrics_val"]
        mt = res["metrics_test"]
        mu = res["metrics_unseen"]
        print(f"{name:<22} {mv['f1']:>8.4f} {mv['auc_pr']:>11.4f} {mv['recall']:>11.4f} "
              f"{mt['f1']:>8.4f} {mu['f1']:>10.4f}")
    
    # ─── Generate plots ───
    print("\n[PLOTS] Generating visualization plots...")
    plot_precision_recall_curves(results, "val")
    plot_confusion_matrices(results, "val")
    
    # Feature importance from best tree model
    best_tree = "XGBoost" if results["XGBoost"]["metrics_val"]["auc_pr"] >= results["LightGBM"]["metrics_val"]["auc_pr"] else "LightGBM"
    plot_feature_importance(results[best_tree]["model"], feature_names, best_tree)
    plot_model_comparison_table(results, feature_names)
    
    # ─── Save models ───
    artifacts_path = SRC / "artifacts"
    artifacts_path.mkdir(exist_ok=True)
    
    for name, res in results.items():
        model_path = artifacts_path / f"baseline_{name.lower()}{suffix}.pkl"
        joblib.dump(res["model"], model_path)
        print(f"  Saved: {model_path.name}")
    
    # ─── Save metrics JSON ───
    metrics_json = {}
    for name, res in results.items():
        metrics_json[name] = {
            "val": res["metrics_val"],
            "test": res["metrics_test"],
            "unseen": res["metrics_unseen"],
        }
    
    metrics_path = DATA / f"baseline_metrics{suffix}{smote_suffix}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Saved: {metrics_path.name}")
    
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Baseline training complete in {elapsed:.1f}s")
    
    # Best model summary
    best_name = max(results, key=lambda n: results[n]["metrics_val"]["auc_pr"])
    best = results[best_name]["metrics_val"]
    print(f"\nBest baseline: {best_name}")
    print(f"  Val:  F1={best['f1']:.4f} | AUC-PR={best['auc_pr']:.4f} | Recall={best['recall']:.4f}")
    print(f"  Test: F1={results[best_name]['metrics_test']['f1']:.4f}")
    print(f"  Unseen: F1={results[best_name]['metrics_unseen']['f1']:.4f}")
    print(f"{'=' * 70}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P053 Baseline Training")
    parser.add_argument("--full", action="store_true", help="Use full dataset")
    parser.add_argument("--smote", action="store_true", help="Use SMOTE-balanced data")
    args = parser.parse_args()
    
    train_baselines(use_full=args.full, use_smote=args.smote)
