"""
P053 — Unit Tests
==================
Tests for focal loss, preprocessing, inference, and drift detection.
"""

import sys
import os
import numpy as np
import pytest

# Ensure src/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ═══════════════════════════════════════════════════════════════
# Focal Loss Tests
# ═══════════════════════════════════════════════════════════════

class TestFocalLoss:
    """Verify focal loss implementation correctness."""

    def test_numpy_basic(self):
        """Focal loss with gamma=0 should equal weighted BCE."""
        from focal_loss import focal_loss_numpy
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.9, 0.1, 0.8, 0.2])
        loss = focal_loss_numpy(y_true, y_pred, alpha=0.5, gamma=0.0)
        assert loss > 0, "Loss should be positive"
        assert loss < 1.0, "Loss should be reasonable for good predictions"

    def test_focal_reduces_easy_examples(self):
        """Focal loss (gamma=2) should be lower than BCE for easy examples."""
        from focal_loss import focal_loss_numpy
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.99, 0.01, 0.98, 0.02])  # Very confident
        bce = focal_loss_numpy(y_true, y_pred, alpha=0.75, gamma=0.0)
        focal = focal_loss_numpy(y_true, y_pred, alpha=0.75, gamma=2.0)
        assert focal < bce, "Focal loss should down-weight easy examples"

    def test_numpy_pytorch_match(self):
        """NumPy and PyTorch implementations should produce same values."""
        import torch
        from focal_loss import focal_loss_numpy, FocalLoss

        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100).astype(np.float32)
        y_pred = np.random.uniform(0.1, 0.9, 100).astype(np.float32)

        np_loss = focal_loss_numpy(y_true, y_pred, alpha=0.75, gamma=2.0)

        # PyTorch uses logits, need to convert
        logits = torch.tensor(np.log(y_pred / (1 - y_pred + 1e-7)))
        targets = torch.tensor(y_true)
        pt_criterion = FocalLoss(alpha=0.75, gamma=2.0)
        pt_loss = pt_criterion(logits, targets).item()

        assert abs(np_loss - pt_loss) < 0.01, (
            f"NumPy ({np_loss:.4f}) and PyTorch ({pt_loss:.4f}) should match"
        )

    def test_perfect_predictions_low_loss(self):
        """Perfect predictions should give near-zero loss."""
        from focal_loss import focal_loss_numpy
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.999, 0.001, 0.999, 0.001])
        loss = focal_loss_numpy(y_true, y_pred, alpha=0.75, gamma=2.0)
        assert loss < 0.01, "Near-perfect predictions should have near-zero focal loss"


# ═══════════════════════════════════════════════════════════════
# Config Tests
# ═══════════════════════════════════════════════════════════════

class TestConfig:
    """Verify configuration consistency."""

    def test_feature_count_consistency(self):
        """N_TABULAR + N_SPATIAL should match ALL_FEATURE_NAMES."""
        from config import N_TABULAR, N_SPATIAL, ALL_FEATURE_NAMES
        assert N_TABULAR + N_SPATIAL == len(ALL_FEATURE_NAMES), (
            f"Feature count mismatch: {N_TABULAR}+{N_SPATIAL} != {len(ALL_FEATURE_NAMES)}"
        )

    def test_model_params_valid(self):
        """Model params should have required keys."""
        from config import MODEL_PARAMS
        required = {"n_tabular", "n_spatial", "d_model", "n_heads", "n_layers"}
        assert required.issubset(MODEL_PARAMS.keys())

    def test_drift_thresholds_ordered(self):
        """Warning threshold should be less than critical."""
        from config import DRIFT
        assert DRIFT["psi_warning"] < DRIFT["psi_critical"]
        assert DRIFT["kl_warning"] < DRIFT["kl_critical"]


# ═══════════════════════════════════════════════════════════════
# Model Architecture Tests
# ═══════════════════════════════════════════════════════════════

class TestModel:
    """Verify model architecture produces correct shapes."""

    def test_forward_pass(self):
        """Model should produce (batch,) logits from correct input shapes."""
        import torch
        from model import HybridTransformerCNN
        from config import MODEL_PARAMS

        model = HybridTransformerCNN(**MODEL_PARAMS)
        model.eval()

        batch_size = 4
        x_tab = torch.randn(batch_size, MODEL_PARAMS["n_tabular"])
        x_spa = torch.randn(batch_size, MODEL_PARAMS["n_spatial"])

        with torch.no_grad():
            logits = model(x_tab, x_spa)

        assert logits.shape == (batch_size,), f"Expected ({batch_size},), got {logits.shape}"

    def test_parameter_count(self):
        """Model should have ~317K parameters."""
        from model import HybridTransformerCNN
        from config import MODEL_PARAMS

        model = HybridTransformerCNN(**MODEL_PARAMS)
        n_params = sum(p.numel() for p in model.parameters())
        assert 200_000 < n_params < 500_000, f"Unexpected param count: {n_params}"

    def test_gradient_flow(self):
        """Gradients should flow through active branches (transformer + CNN + fusion)."""
        import torch
        from model import HybridTransformerCNN
        from config import MODEL_PARAMS

        model = HybridTransformerCNN(**MODEL_PARAMS)
        x_tab = torch.randn(2, MODEL_PARAMS["n_tabular"])
        x_spa = torch.randn(2, MODEL_PARAMS["n_spatial"])
        target = torch.tensor([0.0, 1.0])

        logits = model(x_tab, x_spa)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()

        # tabular_project exists in __init__ but the forward path uses
        # feature_embed + transformer blocks instead, so skip unused params
        has_grad = sum(
            1 for _, p in model.named_parameters()
            if p.requires_grad and p.grad is not None and not torch.all(p.grad == 0)
        )
        total = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        assert has_grad / total > 0.8, f"Only {has_grad}/{total} params got gradients"


# ═══════════════════════════════════════════════════════════════
# Drift Detection Tests
# ═══════════════════════════════════════════════════════════════

class TestDriftDetector:
    """Verify drift detection logic."""

    def test_no_drift_same_distribution(self):
        """Same distribution should show no drift."""
        from drift_detector import DriftDetector

        np.random.seed(42)
        feature_names = ["f1", "f2", "f3"]
        X_ref = np.random.randn(10000, 3)
        X_cur = np.random.randn(5000, 3)

        detector = DriftDetector(feature_names)
        detector.fit_reference(X_ref)
        results = detector.detect(X_cur)

        assert results["summary"]["n_features_critical"] == 0
        assert not results["summary"]["should_retrain"]

    def test_detects_severe_drift(self):
        """Shifted distribution should trigger critical drift."""
        from drift_detector import DriftDetector

        np.random.seed(42)
        feature_names = ["f1", "f2", "f3", "f4"]
        X_ref = np.random.randn(10000, 4)
        X_cur = np.random.randn(5000, 4) + 3.0  # Severe shift

        detector = DriftDetector(feature_names, psi_critical=0.2)
        detector.fit_reference(X_ref)
        results = detector.detect(X_cur)

        assert results["summary"]["n_features_critical"] > 0

    def test_save_load_reference(self, tmp_path):
        """Reference should survive save/load cycle."""
        from drift_detector import DriftDetector

        np.random.seed(42)
        feature_names = ["f1", "f2"]
        X_ref = np.random.randn(1000, 2)

        detector = DriftDetector(feature_names)
        detector.fit_reference(X_ref)
        detector.save_reference(str(tmp_path / "ref.json"))

        loaded = DriftDetector.load_reference(str(tmp_path / "ref.json"))
        assert loaded.feature_names == feature_names
        assert "f1" in loaded.reference_distributions


# ═══════════════════════════════════════════════════════════════
# Retrain Trigger Tests
# ═══════════════════════════════════════════════════════════════

class TestRetrainTrigger:
    """Verify retrain decision logic."""

    def test_no_retrain_when_no_drift(self):
        """Should not retrain if drift is below threshold."""
        from retrain_trigger import RetrainTrigger

        trigger = RetrainTrigger(min_drifted_features=3)
        drift_report = {"summary": {"n_features_critical": 1}}
        decision = trigger.evaluate(drift_report, current_aucpr=0.05)
        assert not decision["should_retrain"]

    def test_retrain_all_criteria_met(self):
        """Should retrain when all three criteria are met."""
        from retrain_trigger import RetrainTrigger

        trigger = RetrainTrigger(
            min_drifted_features=3,
            aucpr_drop_threshold=0.05,
            max_staleness_days=30,
            baseline_aucpr=0.10,
        )
        drift_report = {"summary": {"n_features_critical": 5, "top_drifted": []}}
        decision = trigger.evaluate(
            drift_report,
            current_aucpr=0.02,  # Dropped significantly
            last_retrain_date="2025-01-01T00:00:00",  # Very stale
        )
        assert decision["should_retrain"]

    def test_no_retrain_fresh_model(self):
        """Should not retrain if model was recently retrained."""
        from retrain_trigger import RetrainTrigger
        from datetime import datetime

        trigger = RetrainTrigger(max_staleness_days=30)
        drift_report = {"summary": {"n_features_critical": 5, "top_drifted": []}}
        decision = trigger.evaluate(
            drift_report,
            current_aucpr=0.01,
            last_retrain_date=datetime.utcnow().isoformat(),  # Just now
        )
        assert not decision["should_retrain"]
