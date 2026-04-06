"""
P053 — Focal Loss Implementation for Extreme Class Imbalance
=============================================================
From-scratch derivation → PyTorch implementation.

Focal Loss (Lin et al., 2017):
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Where:
    p_t = p     if y=1
    p_t = 1-p   if y=0

Parameters:
    alpha: Class weight factor (higher for minority class)
    gamma: Focusing parameter (higher = more focus on hard examples)
           gamma=0 → standard cross-entropy
           gamma=2 → recommended for most imbalanced problems
           gamma=5 → extreme focus on hard examples (our case: 1:150)

WHY focal loss for semiconductor yield?
    Standard BCE treats all misclassifications equally.
    At 1:150 imbalance, the model can achieve 99.3% accuracy by predicting ALL pass.
    Focal loss down-weights easy negatives (obvious pass devices) and focuses
    the gradient on the hard cases (borderline fail devices that look like passes).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════
# 1. NumPy Reference Implementation (from scratch)
# ═══════════════════════════════════════════════════════════════

def focal_loss_numpy(y_true, y_pred, alpha=0.75, gamma=2.0, eps=1e-7):
    """Pure NumPy focal loss — for understanding and verification.
    
    Args:
        y_true: (N,) binary labels {0, 1}
        y_pred: (N,) predicted probabilities [0, 1]
        alpha: Weight for positive class
        gamma: Focusing parameter
    
    Returns:
        Scalar mean focal loss
    """
    # Clip predictions for numerical stability
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # p_t = probability of correct class
    p_t = np.where(y_true == 1, y_pred, 1 - y_pred)

    # alpha_t = weight for this sample's class
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)

    # Focal modulating factor: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma

    # Cross-entropy term
    ce = -np.log(p_t)

    # Focal loss = alpha_t * focal_weight * CE
    loss = alpha_t * focal_weight * ce

    return loss.mean()


def focal_loss_gradient_numpy(y_true, y_pred, alpha=0.75, gamma=2.0, eps=1e-7):
    """Gradient of focal loss w.r.t. y_pred — needed for custom XGBoost/LightGBM.
    
    For XGBoost custom objective: returns (gradient, hessian).
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)

    # Gradient derivation:
    # d/dp FL = alpha_t * [-gamma * (1-p_t)^(gamma-1) * log(p_t) - (1-p_t)^gamma / p_t]
    # ... simplified for positive and negative cases

    # For positive class (y=1):
    # grad = alpha * [-gamma * (1-p)^(gamma-1) * log(p) - (1-p)^gamma / p]
    # For negative class (y=0):
    # grad = (1-alpha) * [-gamma * p^(gamma-1) * log(1-p) - p^gamma / (1-p)]
    # But we need grad w.r.t. raw sigmoid input, not p directly

    # Simpler: use the chain rule with sigmoid
    # If z = logit, p = sigmoid(z)
    # dFL/dz = dFL/dp * dp/dz = dFL/dp * p * (1-p)

    term1 = alpha_t * (1 - p_t) ** gamma * (gamma * p_t * np.log(np.maximum(p_t, eps)) + p_t - 1)
    grad = np.where(y_true == 1, -term1, term1)

    # Hessian (approximate)
    hess = np.abs(grad) * (1 - np.abs(grad)) + eps

    return grad, hess


# ═══════════════════════════════════════════════════════════════
# 2. PyTorch Implementation
# ═══════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """PyTorch Focal Loss for binary classification.
    
    Designed for extreme class imbalance (1:150+).
    Uses sigmoid internally — pass raw logits, not probabilities.
    """

    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (N,) raw model outputs (before sigmoid)
            targets: (N,) binary labels {0, 1}
        """
        # Sigmoid probability
        p = torch.sigmoid(logits)

        # Binary cross-entropy (numerically stable)
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")

        # p_t = probability of correct class
        p_t = p * targets + (1 - p) * (1 - targets)

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal modulating factor
        focal_weight = (1 - p_t) ** self.gamma

        # Final loss
        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLossWithLabelSmoothing(nn.Module):
    """Focal loss with label smoothing for noisy labels.
    
    Our data has 1-2% label noise from human inspection errors.
    Label smoothing (epsilon=0.01) helps prevent overfitting to noisy labels.
    """

    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.01, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        # Smooth labels: y_smooth = y * (1 - eps) + eps/2
        targets_smooth = targets.float() * (1 - self.smoothing) + self.smoothing / 2

        p = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction="none")
        p_t = p * targets.float() + (1 - p) * (1 - targets.float())
        alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())
        focal_weight = (1 - p_t) ** self.gamma
        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ═══════════════════════════════════════════════════════════════
# 3. XGBoost Custom Focal Objective
# ═══════════════════════════════════════════════════════════════

def xgb_focal_objective(alpha=0.75, gamma=2.0):
    """Custom focal loss objective for XGBoost.
    
    Returns a callable (y_pred, dtrain) -> (grad, hess).
    """
    def focal_obj(y_pred, dtrain):
        y_true = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid
        p = np.clip(p, 1e-7, 1 - 1e-7)

        # Gradient and hessian
        p_t = np.where(y_true == 1, p, 1 - p)
        alpha_t = np.where(y_true == 1, alpha, 1 - alpha)

        # Gradient of focal loss w.r.t. logit
        grad = alpha_t * (
            gamma * (1 - p_t) ** (gamma - 1) * np.log(np.maximum(p_t, 1e-7)) * p * (1 - p)
            + (1 - p_t) ** gamma * (p - y_true)
        )

        # Approximate hessian
        hess = np.abs(grad) * (1 - np.abs(grad))
        hess = np.maximum(hess, 1e-7)

        return grad, hess

    return focal_obj


def xgb_focal_eval(alpha=0.75, gamma=2.0):
    """Custom focal loss evaluation metric for XGBoost."""
    def focal_eval(y_pred, dtrain):
        y_true = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))
        loss = focal_loss_numpy(y_true, p, alpha=alpha, gamma=gamma)
        return "focal_loss", loss

    return focal_eval


# ═══════════════════════════════════════════════════════════════
# 4. Verification
# ═══════════════════════════════════════════════════════════════

def verify_focal_loss():
    """Verify NumPy and PyTorch implementations match."""
    np.random.seed(42)
    N = 1000
    y_true = np.random.binomial(1, 0.01, N).astype(np.float32)
    y_pred = np.random.uniform(0, 1, N).astype(np.float32)

    # NumPy
    loss_np = focal_loss_numpy(y_true, y_pred, alpha=0.75, gamma=2.0)

    # PyTorch
    logits = torch.tensor(np.log(y_pred / (1 - y_pred + 1e-7)))
    targets = torch.tensor(y_true)
    loss_pt = FocalLoss(alpha=0.75, gamma=2.0)(logits, targets).item()

    print(f"NumPy focal loss:   {loss_np:.6f}")
    print(f"PyTorch focal loss: {loss_pt:.6f}")
    print(f"Difference:         {abs(loss_np - loss_pt):.6f}")

    # Compare with standard BCE
    bce = focal_loss_numpy(y_true, y_pred, alpha=0.5, gamma=0.0)
    print(f"\nStandard BCE (gamma=0): {bce:.6f}")
    print(f"Focal (gamma=2):        {loss_np:.6f}")
    print(f"Focal reduces loss by:  {100*(1-loss_np/bce):.1f}% (down-weights easy examples)")

    # Show effect of gamma sweep
    print("\nGamma sweep (alpha=0.75):")
    for g in [0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        fl = focal_loss_numpy(y_true, y_pred, alpha=0.75, gamma=g)
        print(f"  gamma={g:.1f}: loss={fl:.6f}")


if __name__ == "__main__":
    verify_focal_loss()
