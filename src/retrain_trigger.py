"""
P053 — Automated Retraining Pipeline
======================================
Monitors drift metrics and triggers model retraining when data distribution
shifts beyond acceptable thresholds.

Production workflow:
    1. Drift detector runs daily (drift_detector.py)
    2. If 3+ features critical → this pipeline fires
    3. Pulls latest data from S3/Feature Store
    4. Retrains model with same hyperparameters
    5. Evaluates on held-out test set
    6. If new model improves AUC-PR → promotes to production
    7. If worse → alerts team, keeps current model

SageMaker integration:
    In production, this becomes a SageMaker Pipeline step:
    S3 data → Processing → Training → Evaluation → Conditional → Model Registry

Interview talking point:
    "Our retrain trigger evaluates 3 criteria: drift severity (PSI > 0.2 on 3+
    features), performance decay (AUC-PR dropped > 5%), and data freshness
    (> 30 days since last retrain). All three must be true — this prevents
    unnecessary retraining from transient distribution shifts."

Usage:
    python src/retrain_trigger.py --drift-report src/artifacts/drift_report.json
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score

logger = logging.getLogger(__name__)


class RetrainTrigger:
    """Evaluates whether model retraining is necessary.

    Three-criteria gate:
        1. Data drift: PSI critical on 3+ features
        2. Performance decay: AUC-PR dropped below threshold
        3. Staleness: Last retrain was > N days ago

    All three must be true to trigger retraining. This prevents
    unnecessary compute spend from transient distribution shifts.
    """

    def __init__(
        self,
        min_drifted_features: int = 3,
        aucpr_drop_threshold: float = 0.05,
        max_staleness_days: int = 30,
        baseline_aucpr: float = 0.0524,
    ):
        self.min_drifted_features = min_drifted_features
        self.aucpr_drop_threshold = aucpr_drop_threshold
        self.max_staleness_days = max_staleness_days
        self.baseline_aucpr = baseline_aucpr

    def evaluate(
        self,
        drift_report: dict,
        current_aucpr: Optional[float] = None,
        last_retrain_date: Optional[str] = None,
    ) -> dict:
        """Evaluate all three criteria and decide whether to retrain.

        Args:
            drift_report: Output from DriftDetector.detect()
            current_aucpr: Current model AUC-PR on recent data (None = skip check)
            last_retrain_date: ISO date of last retrain (None = skip staleness)

        Returns:
            dict with decision, reasoning, and individual criteria results
        """
        criteria = {}

        # Criterion 1: Data drift
        summary = drift_report.get("summary", {})
        n_critical = summary.get("n_features_critical", 0)
        drift_triggered = n_critical >= self.min_drifted_features
        criteria["drift"] = {
            "triggered": drift_triggered,
            "n_critical_features": n_critical,
            "threshold": self.min_drifted_features,
            "top_drifted": summary.get("top_drifted", [])[:5],
        }

        # Criterion 2: Performance decay
        if current_aucpr is not None:
            aucpr_drop = self.baseline_aucpr - current_aucpr
            perf_triggered = aucpr_drop > self.aucpr_drop_threshold
            criteria["performance"] = {
                "triggered": perf_triggered,
                "baseline_aucpr": round(self.baseline_aucpr, 4),
                "current_aucpr": round(current_aucpr, 4),
                "drop": round(aucpr_drop, 4),
                "threshold": self.aucpr_drop_threshold,
            }
        else:
            perf_triggered = True  # Conservative: if we can't measure, assume degraded
            criteria["performance"] = {
                "triggered": True,
                "reason": "No current AUC-PR available — assuming degraded",
            }

        # Criterion 3: Staleness
        if last_retrain_date:
            last_retrain = datetime.fromisoformat(last_retrain_date)
            days_since = (datetime.utcnow() - last_retrain).days
            stale_triggered = days_since > self.max_staleness_days
            criteria["staleness"] = {
                "triggered": stale_triggered,
                "days_since_retrain": days_since,
                "threshold_days": self.max_staleness_days,
            }
        else:
            stale_triggered = True
            criteria["staleness"] = {
                "triggered": True,
                "reason": "No retrain history — treating as stale",
            }

        # Decision: ALL three must be true
        should_retrain = drift_triggered and perf_triggered and stale_triggered

        decision = {
            "timestamp": datetime.utcnow().isoformat(),
            "should_retrain": should_retrain,
            "criteria": criteria,
            "reasoning": self._generate_reasoning(criteria, should_retrain),
        }

        if should_retrain:
            logger.warning("RETRAIN TRIGGERED: %s", decision["reasoning"])
        else:
            logger.info("Retrain NOT triggered: %s", decision["reasoning"])

        return decision

    def _generate_reasoning(self, criteria: dict, should_retrain: bool) -> str:
        """Generate human-readable reasoning for the decision."""
        parts = []
        for name, c in criteria.items():
            status = "YES" if c["triggered"] else "NO"
            parts.append(f"{name}={status}")

        prefix = "RETRAIN" if should_retrain else "HOLD"
        return f"{prefix} ({', '.join(parts)})"


class RetrainPipeline:
    """Executes the retraining workflow.

    Steps:
        1. Load latest data (reference + new production data)
        2. Preprocess with existing pipeline
        3. Train model with same hyperparameters
        4. Evaluate on held-out test set
        5. Compare with current production model
        6. Promote if better, alert if worse
    """

    def __init__(self, config: dict):
        self.config = config
        self.retrain_history: list[dict] = []

    def execute(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
        feature_names: list[str],
        current_model_aucpr: float,
    ) -> dict:
        """Execute full retraining pipeline.

        Returns:
            dict with training results, comparison, and promotion decision
        """
        from model import HybridTransformerCNN, create_dataloaders
        from focal_loss import FocalLossWithLabelSmoothing
        from config import MODEL_PARAMS, TRAINING, MODELS_DIR

        t0 = time.time()
        logger.info("Starting retraining: %d train, %d val, %d test",
                     len(X_train), len(X_val), len(X_test))

        # Create model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridTransformerCNN(**MODEL_PARAMS).to(device)

        criterion = FocalLossWithLabelSmoothing(
            alpha=TRAINING["focal_alpha"],
            gamma=TRAINING["focal_gamma"],
            label_smoothing=TRAINING["label_smoothing"],
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAINING["lr"],
            weight_decay=TRAINING["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=TRAINING["scheduler_t0"], T_mult=TRAINING["scheduler_tmult"],
        )

        # Create data loaders
        train_loader, val_loader = create_dataloaders(
            X_train, y_train, X_val, y_val, feature_names,
            batch_size=1024,
        )

        # Training loop (simplified — production uses SageMaker Training Job)
        best_aucpr = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(TRAINING["epochs"]):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                x_tab, x_spa, y_batch = [b.to(device) for b in batch]
                optimizer.zero_grad()
                logits = model(x_tab, x_spa)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validate
            model.eval()
            val_probs = []
            val_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    x_tab, x_spa, y_batch = [b.to(device) for b in batch]
                    logits = model(x_tab, x_spa)
                    probs = torch.sigmoid(logits).float().cpu().numpy()
                    val_probs.extend(probs)
                    val_labels.extend(y_batch.cpu().numpy())

            val_aucpr = average_precision_score(val_labels, val_probs)

            if val_aucpr > best_aucpr:
                best_aucpr = val_aucpr
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= TRAINING["patience"]:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        # Restore best model
        if best_state:
            model.load_state_dict(best_state)

        # Evaluate on test set
        model.eval()
        test_probs = []
        test_labels = []
        spatial_cols = ["die_x", "die_y", "edge_distance"]
        spatial_idx = [feature_names.index(c) for c in spatial_cols if c in feature_names]
        tabular_idx = [i for i in range(len(feature_names)) if i not in spatial_idx]

        X_test_tab = torch.tensor(X_test[:, tabular_idx], dtype=torch.float32, device=device)
        X_test_spa = torch.tensor(X_test[:, spatial_idx], dtype=torch.float32, device=device)

        with torch.no_grad():
            # Process in chunks
            chunk_size = 4096
            for start in range(0, len(X_test), chunk_size):
                end = min(start + chunk_size, len(X_test))
                logits = model(X_test_tab[start:end], X_test_spa[start:end])
                probs = torch.sigmoid(logits).float().cpu().numpy()
                test_probs.extend(probs)

        test_aucpr = average_precision_score(y_test, test_probs)
        test_preds = (np.array(test_probs) >= 0.5).astype(int)
        test_f1 = f1_score(y_test, test_preds)

        elapsed = time.time() - t0

        # Promotion decision
        improved = test_aucpr > current_model_aucpr
        promote = improved

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "training_time_s": round(elapsed, 1),
            "epochs_trained": epoch + 1,
            "best_val_aucpr": round(best_aucpr, 4),
            "test_aucpr": round(test_aucpr, 4),
            "test_f1": round(test_f1, 4),
            "current_model_aucpr": round(current_model_aucpr, 4),
            "improvement": round(test_aucpr - current_model_aucpr, 4),
            "promote": promote,
            "reasoning": (
                f"New model AUC-PR={test_aucpr:.4f} vs current={current_model_aucpr:.4f} → "
                f"{'PROMOTE' if promote else 'KEEP CURRENT'}"
            ),
        }

        if promote:
            # Save new model
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            save_path = MODELS_DIR / f"retrained_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save({
                "model_state_dict": best_state,
                "epoch": epoch + 1,
                "val_aucpr": best_aucpr,
                "test_aucpr": test_aucpr,
                "config": MODEL_PARAMS,
            }, save_path)
            result["model_path"] = str(save_path)
            logger.info("New model saved: %s (AUC-PR: %.4f → %.4f)",
                        save_path, current_model_aucpr, test_aucpr)
        else:
            logger.info("Retrained model NOT promoted (AUC-PR: %.4f ≤ %.4f)",
                        test_aucpr, current_model_aucpr)

        self.retrain_history.append(result)
        return result


if __name__ == "__main__":
    import argparse
    from config import DATA_DIR, ARTIFACTS_DIR

    parser = argparse.ArgumentParser(description="P053 Retrain Trigger")
    parser.add_argument("--drift-report", default=str(ARTIFACTS_DIR / "drift_report.json"))
    parser.add_argument("--baseline-aucpr", type=float, default=0.0524)
    parser.add_argument("--current-aucpr", type=float, default=None)
    parser.add_argument("--last-retrain", type=str, default=None)
    parser.add_argument("--execute", action="store_true", help="Actually retrain if triggered")
    args = parser.parse_args()

    # Load drift report
    with open(args.drift_report) as f:
        drift_report = json.load(f)

    # Evaluate trigger
    trigger = RetrainTrigger(
        baseline_aucpr=args.baseline_aucpr,
    )
    decision = trigger.evaluate(
        drift_report=drift_report,
        current_aucpr=args.current_aucpr,
        last_retrain_date=args.last_retrain,
    )

    print(f"\n{'='*60}")
    print(f"RETRAIN DECISION")
    print(f"{'='*60}")
    print(f"Should retrain: {decision['should_retrain']}")
    print(f"Reasoning:      {decision['reasoning']}")
    for name, c in decision["criteria"].items():
        print(f"  {name}: {'TRIGGERED' if c['triggered'] else 'OK'}")

    # Save decision
    decision_path = ARTIFACTS_DIR / "retrain_decision.json"
    with open(decision_path, "w") as f:
        json.dump(decision, f, indent=2, default=str)
    print(f"\nDecision saved: {decision_path}")

    # Execute retraining if triggered and --execute flag
    if decision["should_retrain"] and args.execute:
        print("\nExecuting retraining pipeline...")
        data = np.load(DATA_DIR / "preprocessed_full.npz")
        pipeline = RetrainPipeline(config={})
        result = pipeline.execute(
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_val=data["X_val"],
            y_val=data["y_val"],
            X_test=data["X_test"],
            y_test=data["y_test"],
            feature_names=data["feature_names"].tolist(),
            current_model_aucpr=args.baseline_aucpr,
        )
        print(f"\nRetrain result: {result['reasoning']}")
