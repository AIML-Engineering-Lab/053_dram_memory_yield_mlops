"""
P053 — Production Inference Module
====================================
Loads model + preprocessing artifacts, runs real-time inference.

Designed for:
    1. FastAPI serving endpoint (single + batch)
    2. SageMaker inference container
    3. Drift detection reference predictions

Why separate from model.py?
    model.py defines architecture + training loop.
    inference.py handles the PRODUCTION path: load artifacts, preprocess raw
    input, run model, post-process output. This separation is standard in
    production ML — training code should never run in serving containers.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch

try:
    from src.config import (
        ARTIFACTS_DIR, MODELS_DIR, MODEL_PARAMS,
        NUMERIC_FEATURES, CATEGORICAL_FEATURES, SPATIAL_FEATURES,
        ENGINEERED_FEATURES, LOG_FEATURES, N_TABULAR, N_SPATIAL,
    )
    from src.model import HybridTransformerCNN
except ImportError:
    from config import (
        ARTIFACTS_DIR, MODELS_DIR, MODEL_PARAMS,
        NUMERIC_FEATURES, CATEGORICAL_FEATURES, SPATIAL_FEATURES,
        ENGINEERED_FEATURES, LOG_FEATURES, N_TABULAR, N_SPATIAL,
    )
    from model import HybridTransformerCNN

logger = logging.getLogger(__name__)


class YieldPredictor:
    """Production inference wrapper for the HybridTransformerCNN model.

    Handles:
        - Model weight loading (local .pt or SageMaker model dir)
        - Preprocessing artifact loading (scaler, encoders, winsorize bounds)
        - Raw feature preprocessing (impute → winsorize → log → encode → scale)
        - Batched GPU/CPU inference
        - Threshold-based classification
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        artifacts_suffix: str = "_full",
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        # Device selection
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.threshold = threshold
        self._load_model(model_path)
        self._load_artifacts(artifacts_suffix)
        logger.info(
            "YieldPredictor initialized: device=%s, threshold=%.2f, features=%d+%d",
            self.device, self.threshold, N_TABULAR, N_SPATIAL,
        )

    def _load_model(self, model_path: Optional[str]):
        """Load HybridTransformerCNN weights."""
        self.model = HybridTransformerCNN(**MODEL_PARAMS)

        if model_path is None:
            # Search for best available checkpoint
            candidates = [
                MODELS_DIR / "best_model.pt",
                ARTIFACTS_DIR / "hybrid_model_full.pt",
                ARTIFACTS_DIR / "hybrid_model_sample.pt",
            ]
            for p in candidates:
                if p.exists():
                    model_path = str(p)
                    break

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Loaded model from checkpoint: %s (epoch %d)",
                            model_path, checkpoint.get("epoch", -1))
            else:
                self.model.load_state_dict(checkpoint)
                logger.info("Loaded model state_dict: %s", model_path)
        else:
            logger.warning("No model weights found — using random initialization")

        self.model.to(self.device)
        self.model.eval()

    def _load_artifacts(self, suffix: str):
        """Load preprocessing artifacts (scaler, encoders, winsorize bounds)."""
        scaler_path = ARTIFACTS_DIR / f"scaler{suffix}.pkl"
        encoders_path = ARTIFACTS_DIR / f"encoders{suffix}.pkl"
        bounds_path = ARTIFACTS_DIR / f"winsorize_bounds{suffix}.pkl"

        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        self.encoders = joblib.load(encoders_path) if encoders_path.exists() else None
        self.bounds = joblib.load(bounds_path) if bounds_path.exists() else None

        if self.scaler is None:
            logger.warning("No scaler found at %s", scaler_path)

    def preprocess_raw(self, raw_features: dict) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess a single raw observation from STDF/MES data.

        Args:
            raw_features: dict with keys matching NUMERIC_FEATURES +
                          CATEGORICAL_FEATURES + SPATIAL_FEATURES

        Returns:
            (tabular_array, spatial_array) ready for model input
        """
        # Extract numeric features
        numeric_vals = []
        for feat in NUMERIC_FEATURES:
            val = raw_features.get(feat, np.nan)
            numeric_vals.append(float(val) if val is not None else np.nan)
        numeric_arr = np.array(numeric_vals, dtype=np.float32)

        # Impute missing with 0 (median would be applied in batch)
        nan_mask = np.isnan(numeric_arr)
        if nan_mask.any():
            numeric_arr[nan_mask] = 0.0

        # Winsorize
        if self.bounds:
            for i, feat in enumerate(NUMERIC_FEATURES):
                if feat in self.bounds:
                    lo, hi = self.bounds[feat]
                    numeric_arr[i] = np.clip(numeric_arr[i], lo, hi)

        # Log transform
        for feat in LOG_FEATURES:
            if feat in NUMERIC_FEATURES:
                idx = NUMERIC_FEATURES.index(feat)
                numeric_arr[idx] = np.log1p(max(numeric_arr[idx], 0))

        # Categorical encoding
        cat_vals = []
        for feat in CATEGORICAL_FEATURES:
            raw_val = str(raw_features.get(feat, "unknown"))
            if self.encoders and feat in self.encoders:
                le = self.encoders[feat]
                if raw_val in set(le.classes_):
                    cat_vals.append(float(le.transform([raw_val])[0]))
                else:
                    cat_vals.append(-1.0)
            else:
                cat_vals.append(0.0)
        cat_arr = np.array(cat_vals, dtype=np.float32)

        # Feature engineering
        eng_vals = self._engineer_single(raw_features, numeric_arr)

        # Spatial features
        spatial_arr = np.array([
            float(raw_features.get(f, 0.0)) for f in SPATIAL_FEATURES
        ], dtype=np.float32)

        # Combine tabular = numeric + categorical + engineered
        tabular_arr = np.concatenate([numeric_arr, cat_arr, eng_vals])

        # Scale
        if self.scaler:
            full_arr = np.concatenate([tabular_arr, spatial_arr]).reshape(1, -1)
            full_arr = self.scaler.transform(full_arr).flatten()
            tabular_arr = full_arr[:N_TABULAR]
            spatial_arr = full_arr[N_TABULAR:]

        return tabular_arr, spatial_arr

    def _engineer_single(self, raw: dict, numeric_arr: np.ndarray) -> np.ndarray:
        """Compute engineered features for a single observation."""
        def _get(name):
            idx = NUMERIC_FEATURES.index(name) if name in NUMERIC_FEATURES else -1
            return numeric_arr[idx] if idx >= 0 else float(raw.get(name, 0.0))

        retention = _get("retention_time_ms")
        temp = _get("test_temp_c")
        leakage = _get("cell_leakage_fa")
        edge_dist = float(raw.get("edge_distance", 0.0))
        idd4 = _get("idd4_active_ma")
        idd2p = _get("idd2p_standby_ma")
        ber = _get("bit_error_rate")
        ecc_ent = _get("ecc_syndrome_entropy")
        uncorr = _get("uncorrectable_in_extended")
        trcd = _get("trcd_ns")
        trp = _get("trp_ns")
        tras = _get("tras_ns")
        rh_sus = _get("rh_susceptibility")
        adj_act = _get("adjacent_row_activations")

        return np.array([
            retention * temp,                              # retention_temp_interaction
            leakage / (retention + 1e-8),                  # leakage_retention_ratio
            edge_dist ** 2,                                # edge_risk
            idd4 / (idd2p + 1e-8),                        # power_ratio
            ber + ecc_ent + uncorr,                        # ecc_burden
            trcd + trp - tras,                             # timing_margin
            rh_sus * np.log1p(adj_act),                    # rh_risk_composite
        ], dtype=np.float32)

    @torch.no_grad()
    def predict(self, tabular: np.ndarray, spatial: np.ndarray) -> dict:
        """Run inference on preprocessed features.

        Args:
            tabular: (N, N_TABULAR) or (N_TABULAR,) array
            spatial: (N, N_SPATIAL) or (N_SPATIAL,) array

        Returns:
            dict with keys: probability, prediction, threshold
        """
        if tabular.ndim == 1:
            tabular = tabular.reshape(1, -1)
            spatial = spatial.reshape(1, -1)

        t_tab = torch.tensor(tabular, dtype=torch.float32, device=self.device)
        t_spa = torch.tensor(spatial, dtype=torch.float32, device=self.device)

        logits = self.model(t_tab, t_spa)
        probs = torch.sigmoid(logits).cpu().numpy()

        predictions = (probs >= self.threshold).astype(int)

        return {
            "probabilities": probs.tolist(),
            "predictions": predictions.tolist(),
            "threshold": self.threshold,
            "n_predicted_fail": int(predictions.sum()),
            "n_total": len(predictions),
        }

    def predict_raw(self, raw_features: dict) -> dict:
        """End-to-end prediction from raw STDF features."""
        tabular, spatial = self.preprocess_raw(raw_features)
        result = self.predict(tabular, spatial)
        result["probability"] = result["probabilities"][0]
        result["prediction"] = result["predictions"][0]
        result["label"] = "FAIL" if result["prediction"] == 1 else "PASS"
        return result

    def predict_batch_raw(self, batch: list[dict]) -> list[dict]:
        """Batch prediction from raw features."""
        tabular_list, spatial_list = [], []
        for row in batch:
            tab, spa = self.preprocess_raw(row)
            tabular_list.append(tab)
            spatial_list.append(spa)

        tabular = np.stack(tabular_list)
        spatial = np.stack(spatial_list)
        result = self.predict(tabular, spatial)

        return [
            {
                "probability": result["probabilities"][i],
                "prediction": result["predictions"][i],
                "label": "FAIL" if result["predictions"][i] == 1 else "PASS",
            }
            for i in range(len(batch))
        ]
