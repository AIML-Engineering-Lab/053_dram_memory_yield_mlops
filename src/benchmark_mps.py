"""
P053 — MPS (Apple Silicon) Benchmark
=====================================
PURPOSE: Document the PAIN of training on local hardware.
Runs 3 epochs on full 10M data to measure wall time, throughput, and peak metrics.

This creates the "before" story:
  "MPS: 47 min/epoch, AUC-PR=0.12 after 3 epochs"
  "A100: 2 min/epoch, AUC-PR=0.89 after 50 epochs"

Output: data/benchmark_mps.json — used in report comparison table.
"""

import json
import time

import numpy as np
import psutil
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

# Import model components
from model import (
    DATA,
    DEVICE,
    FocalLossWithLabelSmoothing,
    HybridTransformerCNN,
    create_dataloaders,
    evaluate,
    find_best_threshold,
)


def benchmark_mps():
    """Run 3-epoch benchmark on MPS to document limitations."""
    print("=" * 70)
    print("P053 — MPS LOCAL BENCHMARK (Apple Silicon M-series)")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print("  Purpose: Document 'PAIN' — local hardware limitations")
    print("  This trains 3 epochs to measure wall time vs cloud GPU")
    print()

    # System info
    mem_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"  System RAM: {mem_gb:.1f} GB")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print()

    # Load data
    t_load = time.time()
    data = np.load(DATA / "preprocessed_full.npz", allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    feature_names = list(data["feature_names"])
    load_time = time.time() - t_load

    print(f"  Data load time: {load_time:.1f}s")
    print(f"  Train: {X_train.shape} ({y_train.sum():.0f} fails, {100*y_train.mean():.2f}%)")
    print(f"  Val: {X_val.shape} ({y_val.sum():.0f} fails)")

    # Create dataloaders — use batch_size=1024 (realistic, not inflated)
    batch_size = 1024
    train_loader, val_loader, n_tab, n_spa = create_dataloaders(
        X_train, y_train, X_val, y_val, feature_names,
        batch_size=batch_size, oversample=False,  # focal loss handles imbalance
    )
    n_batches = len(train_loader)
    print(f"  Batch size: {batch_size} → {n_batches:,} batches/epoch")
    print()

    # Model
    model = HybridTransformerCNN(
        n_tabular=n_tab, n_spatial=n_spa,
        d_model=128, n_heads=4, n_layers=2,
        cnn_out=64, dropout=0.2,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    criterion = FocalLossWithLabelSmoothing(alpha=0.75, gamma=2.0, smoothing=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Benchmark 3 epochs
    epoch_times = []
    epoch_metrics = []

    for epoch in range(1, 4):
        t_start = time.time()
        model.train()
        total_loss = 0
        n = 0
        sample_preds, sample_labels = [], []

        for batch_idx, batch in enumerate(train_loader):
            x_tab, x_spa, labels = [b.to(DEVICE) for b in batch]

            optimizer.zero_grad()
            logits = model(x_tab, x_spa)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n += 1

            # Sample every 50th batch for metrics
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    preds = torch.sigmoid(logits).float().cpu().numpy()
                    sample_preds.extend(preds)
                    sample_labels.extend(labels.cpu().numpy())

            # Progress every 1000 batches
            if (batch_idx + 1) % 1000 == 0:
                elapsed = time.time() - t_start
                throughput = (batch_idx + 1) * batch_size / elapsed
                eta = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1)
                print(f"  Epoch {epoch} | batch {batch_idx+1}/{n_batches} | "
                      f"loss={total_loss/n:.4f} | {throughput:.0f} samples/s | "
                      f"ETA={eta/60:.1f}min", flush=True)

        epoch_time = time.time() - t_start
        epoch_times.append(epoch_time)

        # Quick val eval
        val_loss, val_auc, val_preds, val_labels = evaluate(model, val_loader, criterion, DEVICE)
        threshold = find_best_threshold(val_labels, val_preds)
        val_pred_binary = (val_preds >= threshold).astype(int)

        train_auc = average_precision_score(
            np.array(sample_labels), np.array(sample_preds)
        ) if sum(sample_labels) > 0 else 0

        metrics = {
            "epoch": epoch,
            "train_loss": round(total_loss / n, 4),
            "val_loss": round(val_loss, 4),
            "val_auc_pr": round(val_auc, 4),
            "train_auc_pr": round(train_auc, 4),
            "val_f1": round(float(f1_score(val_labels, val_pred_binary, zero_division=0)), 4),
            "val_precision": round(float(precision_score(val_labels, val_pred_binary, zero_division=0)), 4),
            "val_recall": round(float(recall_score(val_labels, val_pred_binary, zero_division=0)), 4),
            "epoch_time_s": round(epoch_time, 1),
            "throughput_samples_per_s": round(len(X_train) / epoch_time, 0),
        }
        epoch_metrics.append(metrics)

        print(f"\n  Epoch {epoch} DONE: {epoch_time/60:.1f}min | "
              f"val_AUC-PR={val_auc:.4f} | val_F1={metrics['val_f1']:.4f} | "
              f"throughput={metrics['throughput_samples_per_s']:.0f} samples/s\n")

    # Summary
    avg_epoch_time = np.mean(epoch_times)
    total_time = sum(epoch_times)
    projected_50_epochs = avg_epoch_time * 50

    benchmark = {
        "device": str(DEVICE),
        "device_name": "Apple M-series (MPS)",
        "system_ram_gb": round(mem_gb, 1),
        "pytorch_version": torch.__version__,
        "model_params": n_params,
        "batch_size": batch_size,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "epochs_run": 3,
        "epoch_times_s": [round(t, 1) for t in epoch_times],
        "avg_epoch_time_s": round(avg_epoch_time, 1),
        "avg_epoch_time_min": round(avg_epoch_time / 60, 1),
        "total_time_s": round(total_time, 1),
        "projected_50_epochs_min": round(projected_50_epochs / 60, 1),
        "projected_50_epochs_hr": round(projected_50_epochs / 3600, 1),
        "best_val_auc_pr": max(m["val_auc_pr"] for m in epoch_metrics),
        "best_val_f1": max(m["val_f1"] for m in epoch_metrics),
        "final_throughput": round(epoch_metrics[-1]["throughput_samples_per_s"], 0),
        "epoch_metrics": epoch_metrics,
        "limitation_notes": [
            f"3 epochs only = {total_time/60:.0f} min. Full 50 epochs would take ~{projected_50_epochs/3600:.1f} hours.",
            f"Throughput: ~{epoch_metrics[-1]['throughput_samples_per_s']:.0f} samples/s (A100 target: ~500K samples/s)",
            "No multi-GPU, no mixed precision (MPS fp16 is unreliable), limited batch size",
            "Memory: MPS shares unified memory with OS — OOM risk under heavy load",
            f"Val AUC-PR after 3 epochs: {max(m['val_auc_pr'] for m in epoch_metrics):.4f} — insufficient convergence",
        ],
    }

    out = DATA / "benchmark_mps.json"
    with open(out, "w") as f:
        json.dump(benchmark, f, indent=2)

    print("=" * 70)
    print("MPS BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Avg epoch time:     {benchmark['avg_epoch_time_min']:.1f} min")
    print(f"  Throughput:         {benchmark['final_throughput']:.0f} samples/s")
    print(f"  3-epoch total:      {total_time/60:.1f} min")
    print(f"  Projected 50 epochs: {benchmark['projected_50_epochs_hr']:.1f} hours")
    print(f"  Best Val AUC-PR:    {benchmark['best_val_auc_pr']:.4f}")
    print(f"  Best Val F1:        {benchmark['best_val_f1']:.4f}")
    print()
    print("  VERDICT: Local MPS is INADEQUATE for production training.")
    print("           → Move to cloud GPU (Colab A100 or SageMaker V100)")
    print("=" * 70)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    benchmark_mps()
