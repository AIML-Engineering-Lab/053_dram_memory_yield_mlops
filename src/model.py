"""
P053 — Hybrid Transformer-CNN Model for DRAM Yield Prediction
==============================================================
Architecture:
    ┌─────────────────────┐     ┌──────────────────────┐
    │  Tabular Features   │     │  Spatial Features     │
    │  (30 sensor values) │     │  (die_x, die_y, edge) │
    └─────────┬───────────┘     └──────────┬───────────┘
              │                            │
    ┌─────────▼───────────┐     ┌──────────▼───────────┐
    │  Transformer Block  │     │  1D-CNN Block         │
    │  - Self-attention    │     │  - Conv1d(3,32)       │
    │  - Feed-forward     │     │  - Conv1d(32,64)      │
    │  - LayerNorm        │     │  - GlobalAvgPool      │
    └─────────┬───────────┘     └──────────┬───────────┘
              │                            │
              └────────────┬───────────────┘
                           │
                ┌──────────▼───────────┐
                │   Fusion MLP         │
                │   - Concat(256+64)   │
                │   - Dense(320→128)   │
                │   - Dense(128→1)     │
                └──────────┬───────────┘
                           │
                     ┌─────▼─────┐
                     │  Sigmoid  │
                     │  (fail p) │
                     └───────────┘

Why this architecture?
    - Transformer: Captures complex non-linear interactions between sensor readings
      (e.g., leakage-temperature Arrhenius physics, timing margin cascades)
    - CNN: Captures spatial patterns on the wafer (edge die effect, systematic defects)
    - Fusion: Combines both signal types for the final prediction

Usage:
    python src/model.py                  # Train on sample data
    python src/model.py --full           # Train on full dataset
    python src/model.py --full --focal   # With focal loss
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import warnings

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
SRC = ROOT / "src"

# Import our focal loss
try:
    from src.focal_loss import FocalLoss, FocalLossWithLabelSmoothing
except ImportError:
    from focal_loss import FocalLossWithLabelSmoothing

# Device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════
# Model Architecture
# ═══════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """Single Transformer encoder block with self-attention."""

    def __init__(self, d_model, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class SpatialCNN(nn.Module):
    """1D-CNN for spatial wafer features (die_x, die_y, edge_distance)."""

    def __init__(self, in_features=3, out_features=64, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            # Treat spatial features as 1D signal
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pool
        )
        self.fc = nn.Sequential(
            nn.Linear(64, out_features),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (batch, 3) → (batch, 1, 3) for Conv1d
        x = x.unsqueeze(1)
        x = self.conv(x)         # (batch, 64, 1)
        x = x.squeeze(-1)        # (batch, 64)
        x = self.fc(x)           # (batch, out_features)
        return x


class HybridTransformerCNN(nn.Module):
    """Hybrid model combining Transformer for tabular + CNN for spatial features.
    
    Args:
        n_tabular: Number of tabular features (sensor readings)
        n_spatial: Number of spatial features (die_x, die_y, edge_distance) 
        d_model: Transformer hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of Transformer blocks
        cnn_out: CNN output dimension
        dropout: Dropout rate
    """

    def __init__(self, n_tabular=33, n_spatial=3, d_model=128, n_heads=4,
                 n_layers=2, cnn_out=64, dropout=0.2):
        super().__init__()

        # Tabular branch: project features → Transformer
        self.tabular_project = nn.Sequential(
            nn.Linear(n_tabular, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Create sequence of feature tokens for self-attention
        # Each feature becomes a "token" — Transformer learns feature interactions
        self.feature_embed = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_tabular, d_model) * 0.02)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Spatial CNN branch
        self.spatial_cnn = SpatialCNN(n_spatial, cnn_out, dropout)

        # Fusion MLP
        fusion_dim = d_model + cnn_out
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_tabular, x_spatial):
        """
        Args:
            x_tabular: (batch, n_tabular) sensor readings
            x_spatial: (batch, n_spatial) spatial features
        Returns:
            logits: (batch,) raw logits (pass through sigmoid externally)
        """
        batch_size = x_tabular.shape[0]

        # Transformer branch: each feature as a token
        # (batch, n_tabular) → (batch, n_tabular, 1) → (batch, n_tabular, d_model)
        tokens = self.feature_embed(x_tabular.unsqueeze(-1))
        tokens = tokens + self.pos_encoding

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Transformer layers
        for block in self.transformer_blocks:
            tokens = block(tokens)

        # Use CLS token output as tabular representation
        tabular_out = tokens[:, 0, :]  # (batch, d_model)

        # CNN branch
        spatial_out = self.spatial_cnn(x_spatial)  # (batch, cnn_out)

        # Fusion
        fused = torch.cat([tabular_out, spatial_out], dim=1)
        logits = self.fusion(fused).squeeze(-1)  # (batch,)

        return logits


# ═══════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════

def create_dataloaders(X_train, y_train, X_val, y_val, feature_names,
                       batch_size=512, oversample=True):
    """Create DataLoaders with optional oversampling for minority class."""

    # Split features into tabular and spatial
    spatial_cols = ["die_x", "die_y", "edge_distance"]
    spatial_idx = [feature_names.index(c) for c in spatial_cols if c in feature_names]
    tabular_idx = [i for i in range(len(feature_names)) if i not in spatial_idx]

    def split_features(X):
        X_tab = X[:, tabular_idx].astype(np.float32)
        X_spa = X[:, spatial_idx].astype(np.float32)
        return X_tab, X_spa

    X_train_tab, X_train_spa = split_features(X_train)
    X_val_tab, X_val_spa = split_features(X_val)

    train_ds = TensorDataset(
        torch.tensor(X_train_tab), torch.tensor(X_train_spa),
        torch.tensor(y_train.astype(np.float32))
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_tab), torch.tensor(X_val_spa),
        torch.tensor(y_val.astype(np.float32))
    )

    # For large datasets (>1M), WeightedRandomSampler is too slow.
    # Rely on focal loss alpha for class imbalance instead.
    use_sampler = oversample and len(y_train) < 1_000_000

    if use_sampler:
        class_counts = np.bincount(y_train.astype(int))
        weights = 1.0 / class_counts[y_train.astype(int)]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=0, pin_memory=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False)

    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=False)

    return train_loader, val_loader, len(tabular_idx), len(spatial_idx)


def train_one_epoch(model, loader, criterion, optimizer, device, log_every=500):
    """Train for one epoch with progress logging."""
    model.train()
    total_loss = 0
    n_batches = 0
    # Subsample predictions for AUC-PR (skip storing all 10M for speed)
    sample_preds = []
    sample_labels = []
    n_total = len(loader)
    t_start = time.time()

    for batch_idx, batch in enumerate(loader):
        x_tab, x_spa, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()
        logits = model(x_tab, x_spa)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # Store predictions from every 10th batch for AUC-PR estimate
        if batch_idx % 10 == 0:
            with torch.no_grad():
                preds = torch.sigmoid(logits).float().cpu().numpy()
                sample_preds.extend(preds)
                sample_labels.extend(labels.cpu().numpy())

        if log_every > 0 and (batch_idx + 1) % log_every == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (batch_idx + 1) * (n_total - batch_idx - 1)
            print(f"    batch {batch_idx+1}/{n_total} | loss={total_loss/n_batches:.4f} | "
                  f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s", flush=True)

    avg_loss = total_loss / n_batches
    all_preds = np.array(sample_preds)
    all_labels = np.array(sample_labels)

    auc_pr = average_precision_score(all_labels, all_preds) if all_labels.sum() > 0 else 0

    return avg_loss, auc_pr


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        x_tab, x_spa, labels = [b.to(device) for b in batch]

        logits = model(x_tab, x_spa)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        n_batches += 1

        preds = torch.sigmoid(logits).float().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / n_batches
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    auc_pr = average_precision_score(all_labels, all_preds) if all_labels.sum() > 0 else 0

    return avg_loss, auc_pr, all_preds, all_labels


def find_best_threshold(y_true, y_proba):
    """Find threshold maximizing F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    return thresholds[min(best_idx, len(thresholds) - 1)]


# ═══════════════════════════════════════════════════════════════
# Full Training Pipeline
# ═══════════════════════════════════════════════════════════════

def train_model(use_full=False, use_focal=True, epochs=50, lr=1e-3, batch_size=512):
    """Train the Hybrid Transformer-CNN model."""
    t0 = time.time()
    print("=" * 70)
    print("P053 — Hybrid Transformer-CNN Training")
    print(f"  Device: {DEVICE}")
    print(f"  Loss: {'Focal (alpha=0.75, gamma=2)' if use_focal else 'BCE with class weights'}")
    print("=" * 70)

    # ─── Load data ───
    suffix = "_full" if use_full else "_sample"
    data_path = DATA / f"preprocessed{suffix}.npz"

    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run preprocess.py first.")
        return

    data = np.load(data_path, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    X_unseen, y_unseen = data["X_unseen"], data["y_unseen"]
    feature_names = list(data["feature_names"])

    print(f"\n  Train: {X_train.shape} ({y_train.sum():.0f} fails, {100*y_train.mean():.2f}%)")
    print(f"  Val:   {X_val.shape} ({y_val.sum():.0f} fails)")
    print(f"  Features: {len(feature_names)}")

    # ─── Create dataloaders ───
    train_loader, val_loader, n_tab, n_spa = create_dataloaders(
        X_train, y_train, X_val, y_val, feature_names,
        batch_size=batch_size, oversample=True
    )
    print(f"  Tabular features: {n_tab}, Spatial features: {n_spa}")

    # ─── Model ───
    model = HybridTransformerCNN(
        n_tabular=n_tab, n_spatial=n_spa,
        d_model=128, n_heads=4, n_layers=2,
        cnn_out=64, dropout=0.2,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # ─── Loss ───
    if use_focal:
        criterion = FocalLossWithLabelSmoothing(alpha=0.75, gamma=2.0, smoothing=0.01)
    else:
        pos_weight = torch.tensor([(1 - y_train.mean()) / max(y_train.mean(), 1e-7)]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ─── Optimizer + Scheduler ───
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ─── Training loop ───
    history = {"train_loss": [], "val_loss": [], "train_auc_pr": [], "val_auc_pr": []}
    best_val_auc = 0
    best_epoch = 0
    patience = 10
    patience_counter = 0

    print(f"\n{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>10} {'Train AUC-PR':>13} {'Val AUC-PR':>11} {'LR':>10}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        log_every = 500 if use_full else 0
        train_loss, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, log_every=log_every)
        val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc_pr"].append(train_auc)
        history["val_auc_pr"].append(val_auc)

        lr_now = optimizer.param_groups[0]["lr"]

        improved = ""
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            improved = " *"
            # Save best model
            torch.save(model.state_dict(), SRC / "artifacts" / f"hybrid_model{suffix}.pt")
        else:
            patience_counter += 1

        if epoch <= 5 or epoch % 5 == 0 or improved:
            print(f"{epoch:>5} {train_loss:>11.4f} {val_loss:>10.4f} {train_auc:>13.4f} "
                  f"{val_auc:>11.4f} {lr_now:>10.6f}{improved}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (best: {best_epoch})")
            break

    # ─── Load best model ───
    model.load_state_dict(torch.load(SRC / "artifacts" / f"hybrid_model{suffix}.pt",
                                     weights_only=True))

    # ─── Evaluate on all splits ───
    print(f"\n{'=' * 70}")
    print("EVALUATION (best model from epoch {})".format(best_epoch))
    print("=" * 70)

    results = {}
    split_probas = {}

    spatial_cols = ["die_x", "die_y", "edge_distance"]
    spatial_idx = [feature_names.index(c) for c in spatial_cols if c in feature_names]
    tabular_idx = [i for i in range(len(feature_names)) if i not in spatial_idx]

    for split_name, X_split, y_split in [("val", X_val, y_val), ("test", X_test, y_test),
                                          ("unseen", X_unseen, y_unseen)]:
        X_tab = torch.tensor(X_split[:, tabular_idx].astype(np.float32)).to(DEVICE)
        X_spa = torch.tensor(X_split[:, spatial_idx].astype(np.float32)).to(DEVICE)

        model.eval()
        with torch.no_grad():
            # Process in batches to avoid OOM
            all_logits = []
            bs = 2048
            for i in range(0, len(X_tab), bs):
                logits = model(X_tab[i:i+bs], X_spa[i:i+bs])
                all_logits.append(logits.cpu())
            logits = torch.cat(all_logits)
            proba = torch.sigmoid(logits).float().numpy()

        split_probas[split_name] = proba
        threshold = find_best_threshold(y_split, proba) if split_name == "val" else results["val"]["threshold"]
        y_pred = (proba >= threshold).astype(int)

        metrics = {
            "precision": float(precision_score(y_split, y_pred, zero_division=0)),
            "recall": float(recall_score(y_split, y_pred, zero_division=0)),
            "f1": float(f1_score(y_split, y_pred, zero_division=0)),
            "auc_pr": float(average_precision_score(y_split, proba)),
            "auc_roc": float(roc_auc_score(y_split, proba)),
            "threshold": float(threshold),
        }

        cm = confusion_matrix(y_split, y_pred)
        tn, fp, fn, tp = cm.ravel()
        metrics.update({"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)})

        results[split_name] = metrics
        print(f"\n  {split_name.upper()}: F1={metrics['f1']:.4f} | AUC-PR={metrics['auc_pr']:.4f} | "
              f"Recall={metrics['recall']:.4f} | Precision={metrics['precision']:.4f}")
        print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # ─── Generate plots ───
    print("\n[PLOTS] Generating training plots...")
    _plot_training_curves(history, suffix)
    _plot_pr_comparison(y_val, split_probas["val"], results["val"], suffix)

    # ─── Save metrics ───
    metrics_out = {
        "model": "HybridTransformerCNN",
        "loss": "focal" if use_focal else "bce",
        "epochs_trained": best_epoch,
        "n_params": n_params,
        "best_val_auc_pr": best_val_auc,
        "results": results,
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }

    out_path = DATA / f"hybrid_model_metrics{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best Val AUC-PR: {best_val_auc:.4f} (epoch {best_epoch})")
    print(f"{'=' * 70}")

    return results, history


# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════

def _plot_training_curves(history, suffix):
    """Training loss and AUC-PR curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", linewidth=2, label="Train")
    ax1.plot(epochs, history["val_loss"], "r-", linewidth=2, label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#F8FAFC")

    ax2.plot(epochs, history["train_auc_pr"], "b-", linewidth=2, label="Train AUC-PR")
    ax2.plot(epochs, history["val_auc_pr"], "r-", linewidth=2, label="Val AUC-PR")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC-PR")
    ax2.set_title("AUC-PR per Epoch (Higher = Better)", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#F8FAFC")

    plt.suptitle("Hybrid Transformer-CNN — Training Curves", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(ASSETS / "p53_19_hybrid_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: p53_19_hybrid_training_curves.png")


def _plot_pr_comparison(y_true, y_proba, metrics, suffix):
    """PR curve for hybrid model."""
    fig, ax = plt.subplots(figsize=(8, 6))

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ax.plot(recall, precision, "b-", linewidth=2.5,
            label=f"Hybrid Transformer-CNN (AUC-PR={metrics['auc_pr']:.4f})")

    baseline = y_true.mean()
    ax.axhline(y=baseline, color="#D1D5DB", linestyle="--", linewidth=1,
               label=f"Random baseline ({baseline:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Hybrid Model (Val Set)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#F8FAFC")

    plt.tight_layout()
    plt.savefig(ASSETS / "p53_20_hybrid_pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: p53_20_hybrid_pr_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P053 Hybrid Model Training")
    parser.add_argument("--full", action="store_true", help="Use full dataset")
    parser.add_argument("--focal", action="store_true", default=True,
                        help="Use focal loss (default: True)")
    parser.add_argument("--no-focal", dest="focal", action="store_false",
                        help="Use BCE with class weights instead")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    args = parser.parse_args()

    train_model(use_full=args.full, use_focal=args.focal,
                epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
