"""
P053 — Preprocessing Pipeline for DRAM Yield Prediction
========================================================
Handles all 10 real-world data quality issues:
1. Missing values  → KNN imputer (numeric), mode (categorical)
2. Outliers        → IQR-based winsorization (keep data, clip extremes)
3. Skewed features → Log1p transform (leakage, retention, rh_susceptibility)
4. Class imbalance → SMOTE + Tomek links (optional — compare raw vs SMOTE vs focal)
5. Feature eng     → Interaction terms, spatial features, aggregate stats

Usage:
    python src/preprocess.py               # Preprocess sample (50K)
    python src/preprocess.py --full        # Preprocess all splits (16M)
    python src/preprocess.py --full --smote # With SMOTE oversampling on train only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
SRC = ROOT / "src"

# ═══════════════════════════════════════════════════════════════
# Feature Groups
# ═══════════════════════════════════════════════════════════════

# Numeric features to impute + scale
NUMERIC_FEATURES = [
    "test_temp_c", "cell_leakage_fa", "retention_time_ms",
    "row_hammer_threshold", "disturb_margin_mv", "adjacent_row_activations",
    "rh_susceptibility", "bit_error_rate", "correctable_errors_per_1m",
    "ecc_syndrome_entropy", "uncorrectable_in_extended",
    "trcd_ns", "trp_ns", "tras_ns", "rw_latency_ns",
    "idd4_active_ma", "idd2p_standby_ma", "idd5_refresh_ma",
    "gate_oxide_thickness_a", "channel_length_nm", "vt_shift_mv",
    "block_erase_count",
]

# Skewed features → log1p transform BEFORE scaling
LOG_FEATURES = ["cell_leakage_fa", "retention_time_ms", "rh_susceptibility",
                "rw_latency_ns", "ecc_syndrome_entropy"]

# Categorical features to encode
CATEGORICAL_FEATURES = ["tester_id", "probe_card_id", "chamber_id", "recipe_version"]

# Spatial features (keep raw + engineer new ones)
SPATIAL_FEATURES = ["die_x", "die_y", "edge_distance"]

# Target
TARGET = "is_fail"

# Meta columns to drop
META_COLS = ["is_fail_true", "root_cause", "label_is_noisy", "split",
             "lot_id", "wafer_num", "day_fraction"]


# ═══════════════════════════════════════════════════════════════
# 1. Missing Value Imputation
# ═══════════════════════════════════════════════════════════════

def impute_missing(df_train, df_val=None, df_test=None, df_unseen=None, n_neighbors=5):
    """KNN imputation for numeric features, mode for categorical.
    
    Fits on train, transforms all splits.
    For large datasets, uses median imputation (KNN is O(n²)).
    """
    n = len(df_train)
    use_knn = n <= 200_000  # KNN only viable for smaller datasets

    if use_knn:
        print(f"  KNN imputation (k={n_neighbors}) on {n:,} rows...")
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        num_cols = [c for c in NUMERIC_FEATURES if c in df_train.columns]
        df_train[num_cols] = imputer.fit_transform(df_train[num_cols])
        for df in [df_val, df_test, df_unseen]:
            if df is not None:
                df[num_cols] = imputer.transform(df[num_cols])
    else:
        print(f"  Median imputation on {n:,} rows (too large for KNN)...")
        num_cols = [c for c in NUMERIC_FEATURES if c in df_train.columns]
        medians = df_train[num_cols].median()
        df_train[num_cols] = df_train[num_cols].fillna(medians)
        for df in [df_val, df_test, df_unseen]:
            if df is not None:
                df[num_cols] = df[num_cols].fillna(medians)
        # Save medians for production inference
        imputer = medians

    # Categorical: fill with mode from training set
    for col in CATEGORICAL_FEATURES:
        if col in df_train.columns:
            mode_val = df_train[col].mode().iloc[0] if not df_train[col].mode().empty else "unknown"
            df_train[col] = df_train[col].fillna(mode_val)
            for df in [df_val, df_test, df_unseen]:
                if df is not None and col in df.columns:
                    df[col] = df[col].fillna(mode_val)

    return imputer


# ═══════════════════════════════════════════════════════════════
# 2. Outlier Treatment — IQR Winsorization
# ═══════════════════════════════════════════════════════════════

def winsorize_outliers(df_train, df_val=None, df_test=None, df_unseen=None, iqr_factor=3.0):
    """Clip extreme values to [Q1 - k*IQR, Q3 + k*IQR].
    
    Using iqr_factor=3.0 (not 1.5) because semiconductor data has legitimate
    tail values — equipment outliers are 2% but physics outliers are real.
    """
    print(f"  IQR winsorization (factor={iqr_factor})...")
    bounds = {}
    num_cols = [c for c in NUMERIC_FEATURES if c in df_train.columns]

    for col in num_cols:
        q1 = df_train[col].quantile(0.25)
        q3 = df_train[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        bounds[col] = (lower, upper)

        clipped = df_train[col].clip(lower, upper)
        n_clipped = (df_train[col] != clipped).sum()
        df_train[col] = clipped

        for df in [df_val, df_test, df_unseen]:
            if df is not None and col in df.columns:
                df[col] = df[col].clip(lower, upper)

    return bounds


# ═══════════════════════════════════════════════════════════════
# 3. Log Transform Skewed Features
# ═══════════════════════════════════════════════════════════════

def log_transform(df_train, df_val=None, df_test=None, df_unseen=None):
    """Apply log1p to highly skewed features.
    
    Log1p handles zeros safely. Applied BEFORE standardization.
    """
    print(f"  Log1p transform on {LOG_FEATURES}...")
    for col in LOG_FEATURES:
        if col in df_train.columns:
            # Ensure non-negative (clip to 0 first)
            df_train[col] = np.log1p(df_train[col].clip(lower=0))
            for df in [df_val, df_test, df_unseen]:
                if df is not None and col in df.columns:
                    df[col] = np.log1p(df[col].clip(lower=0))


# ═══════════════════════════════════════════════════════════════
# 4. Feature Engineering
# ═══════════════════════════════════════════════════════════════

def engineer_features(df):
    """Create domain-specific interaction and derived features.
    
    These are features a semiconductor engineer would create:
    - Retention × Temperature interaction (Arrhenius physics)
    - Leakage / Retention ratio (failure signature)
    - Edge risk score (spatial model input)
    - Power efficiency ratio
    - ECC burden composite
    - Timing margin composite
    """
    # Retention × Temperature interaction (Arrhenius: higher temp → worse retention)
    if "retention_time_ms" in df.columns and "test_temp_c" in df.columns:
        df["retention_temp_interaction"] = df["retention_time_ms"] * df["test_temp_c"]

    # Leakage-to-retention ratio (high leakage + low retention = failing cell)
    if "cell_leakage_fa" in df.columns and "retention_time_ms" in df.columns:
        df["leakage_retention_ratio"] = df["cell_leakage_fa"] / (df["retention_time_ms"] + 1e-8)

    # Edge risk: combine edge_distance with spatial features
    if "edge_distance" in df.columns:
        df["edge_risk"] = (df["edge_distance"] ** 2)  # Quadratic edge effect

    # Power efficiency: active power / standby power ratio
    if "idd4_active_ma" in df.columns and "idd2p_standby_ma" in df.columns:
        df["power_ratio"] = df["idd4_active_ma"] / (df["idd2p_standby_ma"] + 1e-8)

    # ECC burden: composite of error indicators
    if all(c in df.columns for c in ["bit_error_rate", "ecc_syndrome_entropy", "uncorrectable_in_extended"]):
        df["ecc_burden"] = (df["bit_error_rate"] + df["ecc_syndrome_entropy"] +
                           df["uncorrectable_in_extended"])

    # Timing margin: how close to spec limits
    if all(c in df.columns for c in ["trcd_ns", "trp_ns", "tras_ns"]):
        df["timing_margin"] = df["trcd_ns"] + df["trp_ns"] - df["tras_ns"]

    # Row hammer risk composite
    if all(c in df.columns for c in ["rh_susceptibility", "adjacent_row_activations"]):
        df["rh_risk_composite"] = df["rh_susceptibility"] * np.log1p(df["adjacent_row_activations"])

    return df


# ═══════════════════════════════════════════════════════════════
# 5. Categorical Encoding
# ═══════════════════════════════════════════════════════════════

def encode_categoricals(df_train, df_val=None, df_test=None, df_unseen=None):
    """Label encode categorical features. Fit on train, transform all."""
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col not in df_train.columns:
            continue
        le = LabelEncoder()
        # Fit on all known categories from train
        df_train[col] = df_train[col].astype(str)
        le.fit(df_train[col])
        df_train[col] = le.transform(df_train[col])
        encoders[col] = le

        for df in [df_val, df_test, df_unseen]:
            if df is not None and col in df.columns:
                df[col] = df[col].astype(str)
                # Handle unseen categories
                known = set(le.classes_)
                df[col] = df[col].map(lambda x, k=known, l=le: (
                    l.transform([x])[0] if x in k else -1
                ))

    return encoders


# ═══════════════════════════════════════════════════════════════
# 6. StandardScaler
# ═══════════════════════════════════════════════════════════════

def scale_features(df_train, df_val=None, df_test=None, df_unseen=None):
    """Standard scale all numeric features. Fit on train only."""
    feature_cols = [c for c in df_train.columns if c != TARGET]
    scaler = StandardScaler()
    num_cols = df_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
    for df in [df_val, df_test, df_unseen]:
        if df is not None:
            df[num_cols] = scaler.transform(df[num_cols])

    return scaler, feature_cols


# ═══════════════════════════════════════════════════════════════
# 7. SMOTE + Tomek (optional — train only)
# ═══════════════════════════════════════════════════════════════

def apply_smote(X_train, y_train, random_state=42):
    """SMOTE + Tomek links on training data only.
    
    For 10M rows, use random undersampling + SMOTE on subset.
    """
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import TomekLinks

    n = len(X_train)
    print(f"  SMOTE+Tomek on {n:,} samples (minority: {y_train.sum():,})...")

    if n > 500_000:
        # For very large datasets: undersample majority first, then SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        # Keep all minority + 10x minority from majority
        n_minority = y_train.sum()
        ratio = min(10 * n_minority / (n - n_minority), 1.0)
        rus = RandomUnderSampler(sampling_strategy=ratio, random_state=random_state)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        print(f"  After undersampling: {len(X_resampled):,} ({y_resampled.sum():,} minority)")
    else:
        X_resampled, y_resampled = X_train, y_train

    # Now SMOTE + Tomek
    smote_tomek = SMOTETomek(
        smote=SMOTE(sampling_strategy=0.3, random_state=random_state, k_neighbors=5),
        tomek=TomekLinks(),
        random_state=random_state
    )
    X_final, y_final = smote_tomek.fit_resample(X_resampled, y_resampled)
    print(f"  After SMOTE+Tomek: {len(X_final):,} ({y_final.sum():,} minority, "
          f"{100*y_final.mean():.1f}%)")

    return X_final, y_final


# ═══════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════

def preprocess_pipeline(use_full=False, use_smote=False):
    """Run the complete preprocessing pipeline.
    
    Returns preprocessed DataFrames or saves to parquet.
    """
    t0 = time.time()
    print("=" * 70)
    print("P053 — DRAM Yield Predictor: Preprocessing Pipeline")
    print("=" * 70)

    # ─── Load data ───
    if use_full:
        print("\n[1/7] Loading FULL dataset (16M rows)...")
        df_train = pd.read_parquet(DATA / "dram_stdf_train.parquet")
        df_val = pd.read_parquet(DATA / "dram_stdf_val.parquet")
        df_test = pd.read_parquet(DATA / "dram_stdf_test.parquet")
        df_unseen = pd.read_parquet(DATA / "dram_stdf_unseen.parquet")
    else:
        print("\n[1/7] Loading SAMPLE dataset (50K rows)...")
        df_all = pd.read_parquet(DATA / "dram_stdf_sample.parquet")
        # Split by split column
        df_train = df_all[df_all["split"] == "train"].copy().reset_index(drop=True)
        df_val = df_all[df_all["split"] == "val"].copy().reset_index(drop=True)
        df_test = df_all[df_all["split"] == "test"].copy().reset_index(drop=True)
        df_unseen = df_all[df_all["split"] == "unseen"].copy().reset_index(drop=True)
        if len(df_val) == 0:
            # Sample doesn't have splits — do 60/15/15/10
            from sklearn.model_selection import train_test_split
            df_train, df_rest = train_test_split(df_all, test_size=0.4, random_state=42,
                                                  stratify=df_all[TARGET])
            df_val, df_rest2 = train_test_split(df_rest, test_size=0.625, random_state=42,
                                                 stratify=df_rest[TARGET])
            df_test, df_unseen = train_test_split(df_rest2, test_size=0.4, random_state=42,
                                                   stratify=df_rest2[TARGET])
            for d in [df_train, df_val, df_test, df_unseen]:
                d.reset_index(drop=True, inplace=True)

    splits = {"train": df_train, "val": df_val, "test": df_test, "unseen": df_unseen}
    for name, df in splits.items():
        n_miss = df[NUMERIC_FEATURES].isnull().sum().sum()
        print(f"  {name}: {len(df):,} rows, {df[TARGET].sum():,} fails "
              f"({100*df[TARGET].mean():.2f}%), {n_miss:,} missing cells")

    # ─── Drop meta columns ───
    for df in splits.values():
        drop_cols = [c for c in META_COLS if c in df.columns]
        df.drop(columns=drop_cols, inplace=True)

    # ─── Step 2: Impute missing values ───
    print("\n[2/7] Imputing missing values...")
    imputer = impute_missing(df_train, df_val, df_test, df_unseen)
    for name, df in splits.items():
        remaining = df.isnull().sum().sum()
        print(f"  {name}: {remaining} NaN remaining")

    # ─── Step 3: Winsorize outliers ───
    print("\n[3/7] Winsorizing outliers...")
    bounds = winsorize_outliers(df_train, df_val, df_test, df_unseen, iqr_factor=3.0)

    # ─── Step 4: Log transform ───
    print("\n[4/7] Log-transforming skewed features...")
    log_transform(df_train, df_val, df_test, df_unseen)

    # ─── Step 5: Feature engineering ───
    print("\n[5/7] Engineering domain features...")
    for name, df in splits.items():
        splits[name] = engineer_features(df)
    df_train, df_val, df_test, df_unseen = (splits["train"], splits["val"],
                                             splits["test"], splits["unseen"])
    new_feats = [c for c in df_train.columns if c not in NUMERIC_FEATURES + CATEGORICAL_FEATURES +
                 SPATIAL_FEATURES + [TARGET] + META_COLS]
    print(f"  New features: {new_feats}")
    print(f"  Total features: {len(df_train.columns) - 1}")

    # ─── Step 6: Encode categoricals ───
    print("\n[6/7] Encoding categorical features...")
    encoders = encode_categoricals(df_train, df_val, df_test, df_unseen)

    # ─── Step 7: Scale features ───
    print("\n[7/7] Standardizing features...")
    scaler, feature_cols = scale_features(df_train, df_val, df_test, df_unseen)

    # ─── Separate X and y ───
    X_train = df_train.drop(columns=[TARGET]).values
    y_train = df_train[TARGET].values
    X_val = df_val.drop(columns=[TARGET]).values
    y_val = df_val[TARGET].values
    X_test = df_test.drop(columns=[TARGET]).values
    y_test = df_test[TARGET].values
    X_unseen = df_unseen.drop(columns=[TARGET]).values
    y_unseen = df_unseen[TARGET].values

    feature_names = [c for c in df_train.columns if c != TARGET]
    print(f"\n  Final feature count: {len(feature_names)}")
    print(f"  Feature names: {feature_names}")

    # ─── Optional SMOTE ───
    if use_smote:
        print("\n[SMOTE] Applying SMOTE + Tomek to training data...")
        X_train, y_train = apply_smote(X_train, y_train)

    # ─── Save preprocessed data ───
    suffix = "_full" if use_full else "_sample"
    smote_suffix = "_smote" if use_smote else ""
    out_path = DATA / f"preprocessed{suffix}{smote_suffix}.npz"

    np.savez_compressed(
        out_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        X_unseen=X_unseen, y_unseen=y_unseen,
        feature_names=np.array(feature_names),
    )

    # Save scaler + encoders for inference
    artifacts_path = SRC / "artifacts"
    artifacts_path.mkdir(exist_ok=True)
    joblib.dump(scaler, artifacts_path / f"scaler{suffix}.pkl")
    joblib.dump(encoders, artifacts_path / f"encoders{suffix}.pkl")
    joblib.dump(bounds, artifacts_path / f"winsorize_bounds{suffix}.pkl")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Preprocessing complete in {elapsed:.1f}s")
    print(f"  Output: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}, Unseen: {X_unseen.shape}")
    print(f"  Train fail rate: {100 * y_train.mean():.2f}%")
    print(f"{'=' * 70}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "X_unseen": X_unseen, "y_unseen": y_unseen,
        "feature_names": feature_names,
        "scaler": scaler,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P053 Preprocessing Pipeline")
    parser.add_argument("--full", action="store_true", help="Use full 16M dataset")
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE+Tomek to training")
    args = parser.parse_args()

    preprocess_pipeline(use_full=args.full, use_smote=args.smote)
