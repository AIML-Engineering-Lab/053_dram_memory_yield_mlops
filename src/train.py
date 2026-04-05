"""
P053 — Production Training Script with MLflow Integration
==========================================================
Full training pipeline for HybridTransformerCNN with:
  - MLflow experiment tracking (params, metrics per epoch, artifacts)
  - AMP support (bfloat16 on A100, float16 on T4, no-amp on MPS)
  - GradScaler only for float16 (not needed for bfloat16)
  - Automatic hardware detection & optimal settings
  - Model checkpointing with MLflow artifact logging
  - Evaluation on val/test/unseen splits
  - Business impact calculation

Usage:
    # Local MPS (no AMP)
    python -m src.train

    # Full dataset on GPU
    python -m src.train --full --batch-size 4096

    # Retrain triggered by simulation (with run name)
    python -m src.train --full --run-name retrain-day31 --batch-size 4096

    # From Colab (auto-detects A100/T4)
    python -m src.train --full --batch-size 4096
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve, roc_auc_score,
)

from src.config import ROOT, DATA_DIR, ASSETS_DIR, ARTIFACTS_DIR, MODEL_PARAMS, TRAINING
from src.model import HybridTransformerCNN, create_dataloaders, find_best_threshold
from src.focal_loss import FocalLossWithLabelSmoothing
from src.mlflow_utils import (
    init_mlflow, start_training_run, log_epoch_metrics,
    log_evaluation_results, log_model_artifact, log_plot_artifact,
    log_training_summary,
)


# ═══════════════════════════════════════════════════════════════
# Hardware Detection
# ═══════════════════════════════════════════════════════════════

def detect_hardware():
    """Auto-detect GPU and return optimal training settings."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        cc = torch.cuda.get_device_capability(0)

        # A100/H100/A6000 (cc >= 8.0): use bfloat16, no GradScaler
        if cc[0] >= 8:
            amp_dtype = "bfloat16"
            use_amp = True
            use_gradscaler = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            # T4/V100 (cc < 8.0): float16 with GradScaler
            amp_dtype = "float16"
            use_amp = True
            use_gradscaler = True

        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_name = "Apple MPS"
        vram_gb = 0  # Shared memory
        amp_dtype = "float32"
        use_amp = False
        use_gradscaler = False
        device = torch.device("mps")
    else:
        gpu_name = "CPU"
        vram_gb = 0
        amp_dtype = "float32"
        use_amp = False
        use_gradscaler = False
        device = torch.device("cpu")

    return {
        "device": device,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 1),
        "amp_dtype": amp_dtype,
        "use_amp": use_amp,
        "use_gradscaler": use_gradscaler,
    }


# ═══════════════════════════════════════════════════════════════
# Training Loop with MLflow
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, hw):
    """Train for one epoch with AMP support."""
    model.train()
    total_loss = 0
    n_batches = 0
    sample_preds = []
    sample_labels = []
    n_total = len(loader)

    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16 if hw["amp_dtype"] == "bfloat16" else torch.float16)
        if hw["use_amp"]
        else torch.autocast("cpu", enabled=False)
    )

    for batch_idx, batch in enumerate(loader):
        x_tab, x_spa, labels = [b.to(device) for b in batch]

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx:
            logits = model(x_tab, x_spa)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 10 == 0:
            with torch.no_grad():
                preds = torch.sigmoid(logits).cpu().numpy()
                sample_preds.extend(preds)
                sample_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / n_batches
    auc_pr = (
        average_precision_score(np.array(sample_labels), np.array(sample_preds))
        if np.array(sample_labels).sum() > 0
        else 0
    )
    return avg_loss, auc_pr


@torch.no_grad()
def evaluate_split(model, X, y, feature_names, device, criterion, hw, batch_size=2048):
    """Evaluate model on a data split."""
    model.eval()

    spatial_cols = ["die_x", "die_y", "edge_distance"]
    spatial_idx = [feature_names.index(c) for c in spatial_cols if c in feature_names]
    tabular_idx = [i for i in range(len(feature_names)) if i not in spatial_idx]

    X_tab = torch.tensor(X[:, tabular_idx].astype(np.float32))
    X_spa = torch.tensor(X[:, spatial_idx].astype(np.float32))

    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16 if hw["amp_dtype"] == "bfloat16" else torch.float16)
        if hw["use_amp"]
        else torch.autocast("cpu", enabled=False)
    )

    all_logits = []
    total_loss = 0
    n_batches = 0

    for i in range(0, len(X_tab), batch_size):
        xt = X_tab[i:i + batch_size].to(device)
        xs = X_spa[i:i + batch_size].to(device)
        lb = torch.tensor(y[i:i + batch_size].astype(np.float32)).to(device)

        with amp_ctx:
            logits = model(xt, xs)
            loss = criterion(logits, lb)

        all_logits.append(logits.cpu())
        total_loss += loss.item()
        n_batches += 1

    logits = torch.cat(all_logits)
    proba = torch.sigmoid(logits).numpy()
    avg_loss = total_loss / n_batches
    auc_pr = average_precision_score(y, proba) if y.sum() > 0 else 0

    return avg_loss, auc_pr, proba


def run_training(args):
    """Full training pipeline with MLflow logging."""
    t_global = time.time()

    # ─── Hardware ───
    hw = detect_hardware()
    device = hw["device"]
    print("=" * 70)
    print("P053 — Production Training with MLflow")
    print(f"  GPU: {hw['gpu_name']} ({hw['vram_gb']} GB)")
    print(f"  AMP: {hw['amp_dtype']} | GradScaler: {hw['use_gradscaler']}")
    print(f"  Device: {device}")
    print("=" * 70)

    # ─── Load data ───
    suffix = "_full" if args.full else "_sample"
    data_path = DATA_DIR / f"preprocessed{suffix}.npz"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run: python -m src.preprocess {'--full' if args.full else ''}")
        return

    data = np.load(data_path, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    X_unseen, y_unseen = data["X_unseen"], data["y_unseen"]
    feature_names = list(data["feature_names"])

    print(f"\n  Train: {X_train.shape} ({y_train.sum():.0f} fails, {100 * y_train.mean():.2f}%)")
    print(f"  Val:   {X_val.shape} ({y_val.sum():.0f} fails)")
    print(f"  Test:  {X_test.shape}")
    print(f"  Unseen: {X_unseen.shape}")

    # ─── Dataloaders ───
    train_loader, val_loader, n_tab, n_spa = create_dataloaders(
        X_train, y_train, X_val, y_val, feature_names,
        batch_size=args.batch_size, oversample=True,
    )

    # ─── Model ───
    model = HybridTransformerCNN(
        n_tabular=n_tab, n_spatial=n_spa,
        d_model=MODEL_PARAMS["d_model"],
        n_heads=MODEL_PARAMS["n_heads"],
        n_layers=MODEL_PARAMS["n_layers"],
        cnn_out=MODEL_PARAMS["cnn_out"],
        dropout=MODEL_PARAMS["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # ─── Loss ───
    criterion = FocalLossWithLabelSmoothing(
        alpha=TRAINING["focal_alpha"],
        gamma=TRAINING["focal_gamma"],
        smoothing=TRAINING["label_smoothing"],
    )

    # ─── Optimizer + Scheduler ───
    lr = args.lr or TRAINING["lr"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=TRAINING["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # ─── GradScaler (float16 only) ───
    scaler = torch.amp.GradScaler("cuda") if hw["use_gradscaler"] else None

    # ═══════════════════════════════════════════════════════════
    # MLflow — Start Run
    # ═══════════════════════════════════════════════════════════
    init_mlflow()

    run_name = args.run_name or f"{hw['gpu_name'].split()[0]}-{hw['amp_dtype']}-train"
    run_ctx = start_training_run(
        run_name=run_name,
        gpu_name=hw["gpu_name"],
        amp_dtype=hw["amp_dtype"],
        batch_size=args.batch_size,
        learning_rate=lr,
        extra_params={
            "hw.vram_gb": hw["vram_gb"],
            "hw.use_gradscaler": str(hw["use_gradscaler"]),
            "train.rows": int(len(y_train)),
            "data.suffix": suffix,
        },
        extra_tags={
            "gpu_type": hw["gpu_name"],
            "run_context": args.context or "local",
        },
    )

    with run_ctx as run:
        print(f"\n  MLflow run: {run.info.run_id}")
        print(f"  Run name:   {run_name}")

        # ─── Training loop ───
        history = {"train_loss": [], "val_loss": [], "train_auc_pr": [], "val_auc_pr": []}
        epoch_times = []
        best_val_auc = 0
        best_epoch = 0
        patience_counter = 0
        model_save_path = ARTIFACTS_DIR / f"hybrid_best{suffix}.pt"

        print(f"\n{'Ep':>4} {'TrLoss':>8} {'VaLoss':>8} {'TrAUC':>8} {'VaAUC':>8} {'LR':>10} {'Time':>7}")
        print("-" * 70)

        for epoch in range(1, args.epochs + 1):
            t_epoch = time.time()

            train_loss, train_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler, hw,
            )

            # Evaluate on val loader
            val_loss, val_auc, _ = evaluate_split(
                model, X_val, y_val, feature_names, device, criterion, hw,
            )

            scheduler.step()
            epoch_time = time.time() - t_epoch
            epoch_times.append(epoch_time)
            throughput = len(y_train) / epoch_time

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_auc_pr"].append(train_auc)
            history["val_auc_pr"].append(val_auc)

            current_lr = optimizer.param_groups[0]["lr"]

            # ── MLflow: log epoch metrics ──
            log_epoch_metrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_auc_pr=train_auc,
                val_auc_pr=val_auc,
                learning_rate=current_lr,
                epoch_time_s=epoch_time,
                throughput=throughput,
            )

            improved = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                patience_counter = 0
                improved = " *"
                torch.save(model.state_dict(), model_save_path)
            else:
                patience_counter += 1

            if epoch <= 5 or epoch % 5 == 0 or improved:
                print(
                    f"{epoch:>4} {train_loss:>8.4f} {val_loss:>8.4f} "
                    f"{train_auc:>8.4f} {val_auc:>8.4f} {current_lr:>10.6f} "
                    f"{epoch_time:>6.1f}s{improved}"
                )

            if patience_counter >= TRAINING["patience"]:
                print(f"\nEarly stopping at epoch {epoch} (best: {best_epoch})")
                break

        # ─── Load best model ───
        model.load_state_dict(torch.load(model_save_path, weights_only=True, map_location=device))

        # ─── Evaluate all splits ───
        print(f"\n{'=' * 70}")
        print(f"EVALUATION — best model from epoch {best_epoch}")
        print("=" * 70)

        results = {}
        threshold = None

        for split_name, X_split, y_split in [
            ("val", X_val, y_val), ("test", X_test, y_test), ("unseen", X_unseen, y_unseen),
        ]:
            _, split_auc, proba = evaluate_split(
                model, X_split, y_split, feature_names, device, criterion, hw,
            )

            if split_name == "val":
                threshold = find_best_threshold(y_split, proba)

            y_pred = (proba >= threshold).astype(int)
            cm = confusion_matrix(y_split, y_pred)
            tn, fp, fn, tp = cm.ravel()

            results[split_name] = {
                "precision": float(precision_score(y_split, y_pred, zero_division=0)),
                "recall": float(recall_score(y_split, y_pred, zero_division=0)),
                "f1": float(f1_score(y_split, y_pred, zero_division=0)),
                "auc_pr": float(split_auc),
                "auc_roc": float(roc_auc_score(y_split, proba)),
                "threshold": float(threshold),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            }

            print(
                f"  {split_name.upper():>7}: AUC-PR={split_auc:.4f} | F1={results[split_name]['f1']:.4f} | "
                f"Recall={results[split_name]['recall']:.4f} | TP={tp} FP={fp} FN={fn} TN={tn}"
            )

        # ── MLflow: log evaluation results ──
        log_evaluation_results(results, threshold)

        # ── MLflow: log training summary ──
        total_time = time.time() - t_global
        peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        avg_epoch_time = np.mean(epoch_times)
        peak_throughput = len(y_train) / min(epoch_times) if epoch_times else 0

        log_training_summary(
            best_epoch=best_epoch,
            best_val_auc=best_val_auc,
            total_time_min=total_time / 60,
            avg_epoch_time_s=avg_epoch_time,
            throughput=peak_throughput,
            peak_vram_gb=peak_vram,
            epochs_run=len(epoch_times),
            train_rows=len(y_train),
        )

        # ── MLflow: log model artifact ──
        if model_save_path.exists():
            log_model_artifact(model_save_path)

        # ─── Save benchmark JSON ───
        benchmark = {
            "device": str(device),
            "gpu_name": hw["gpu_name"],
            "gpu_vram_gb": hw["vram_gb"],
            "amp_dtype": hw["amp_dtype"],
            "use_gradscaler": hw["use_gradscaler"],
            "batch_size": args.batch_size,
            "model_params": n_params,
            "train_rows": len(y_train),
            "epochs_run": len(epoch_times),
            "best_epoch": best_epoch,
            "avg_epoch_time_s": float(avg_epoch_time),
            "total_train_time_min": float(total_time / 60),
            "throughput_samples_per_s": float(peak_throughput),
            "peak_gpu_memory_gb": float(peak_vram),
            "results": results,
            "history": {k: [float(v) for v in vals] for k, vals in history.items()},
            "epoch_times_s": [float(t) for t in epoch_times],
            "mlflow_run_id": run.info.run_id,
        }

        benchmark_path = DATA_DIR / f"benchmark_{hw['gpu_name'].split()[0].lower()}.json"
        with open(benchmark_path, "w") as f:
            json.dump(benchmark, f, indent=2)

        import mlflow
        mlflow.log_artifact(str(benchmark_path), "benchmark")

        print(f"\n{'=' * 70}")
        print(f"Training complete in {total_time / 60:.1f} min")
        print(f"Best Val AUC-PR: {best_val_auc:.4f} (epoch {best_epoch})")
        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Benchmark saved: {benchmark_path}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="P053 — Train HybridTransformerCNN with MLflow")
    parser.add_argument("--full", action="store_true", help="Use full 16M dataset")
    parser.add_argument("--epochs", type=int, default=TRAINING["epochs"], help="Max epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (auto from config)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--context", type=str, default=None, help="Run context (local/colab/ci)")
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
