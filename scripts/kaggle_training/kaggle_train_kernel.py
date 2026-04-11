"""
P053 — Kaggle Training Kernel Script
======================================
This script runs INSIDE a Kaggle GPU session.
Triggered remotely via src/kaggle_backend.py (kaggle kernels push API).

What this script does:
    1. Read run_config.json (written by kaggle_backend.py before push)
    2. Install dependencies
    3. Clone the P053 GitHub repo
    4. Download preprocessed_full.npz from S3
    5. Run train.py with GPU (T4 or P100 depending on Kaggle selection)
    6. Upload model + benchmark + mlflow.db to S3

Required Kaggle environment variables (set at kaggle.com/Account > Environment Variables):
    AWS_ACCESS_KEY_ID       - S3 read/write
    AWS_SECRET_ACCESS_KEY   - S3 read/write
    AWS_DEFAULT_REGION      - us-west-2
    S3_BUCKET               - p053-mlflow-artifacts
    GITHUB_REPO_URL         - https://github.com/rajendarmuddasani/DRAM_Yield_Predictor_MLOps.git
    GITHUB_TOKEN            - (optional) if repo is private
"""

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

# ── Read run config ────────────────────────────────────────────────────────────
# This file is written by src/kaggle_backend.py before kernel push
CONFIG_PATH = Path(__file__).parent / "run_config.json"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        config = json.load(f)
else:
    # Defaults if running manually
    config = {}

RUN_NAME = config.get("run_name", "kaggle-manual-retrain")
EPOCHS = int(config.get("epochs", 10))
CONTEXT = config.get("context", "kaggle-retrain")
S3_BUCKET = config.get("s3_bucket", os.environ.get("S3_BUCKET", "p053-mlflow-artifacts"))
AWS_REGION = config.get("aws_region", os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
GITHUB_REPO = config.get(
    "github_repo",
    os.environ.get(
        "GITHUB_REPO_URL",
        "https://github.com/rajendarmuddasani/DRAM_Yield_Predictor_MLOps.git",
    ),
)

print("=" * 70)
print("P053 — Kaggle GPU Training Kernel")
print("=" * 70)
print(f"  Run name : {RUN_NAME}")
print(f"  Epochs   : {EPOCHS}")
print(f"  Context  : {CONTEXT}")
print(f"  S3 bucket: {S3_BUCKET}")
print(f"  Repo     : {GITHUB_REPO}")
print("=" * 70)

# ── Environment setup ──────────────────────────────────────────────────────────
os.environ.setdefault("AWS_DEFAULT_REGION", AWS_REGION)
os.environ.setdefault("S3_BUCKET", S3_BUCKET)
os.environ["COMPUTE_BACKEND"] = "kaggle"
os.environ["MODEL_PARAMS"] = "317000"
os.environ["SIMULATION_SCALE"] = "phase2"

# ── Install dependencies ───────────────────────────────────────────────────────
print("\n[SETUP] Installing dependencies...")
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "mlflow", "boto3", "pyarrow", "scikit-learn",
        "xgboost", "lightgbm", "ruff",
    ],
    check=False,
)
print("[SETUP] Done.")

# ── Clone repo ─────────────────────────────────────────────────────────────────
PROJECT_DIR = "/kaggle/working/053_memory_yield_predictor"

# If private repo, inject token into URL
github_token = os.environ.get("GITHUB_TOKEN", "")
if github_token and "github.com" in GITHUB_REPO and "@" not in GITHUB_REPO:
    # Transform https://github.com/... → https://TOKEN@github.com/...
    clone_url = GITHUB_REPO.replace("https://", f"https://{github_token}@")
else:
    clone_url = GITHUB_REPO

if not Path(PROJECT_DIR).exists():
    print(f"\n[SETUP] Cloning repo to {PROJECT_DIR}...")
    subprocess.run(["git", "clone", clone_url, PROJECT_DIR], check=True)
else:
    print(f"\n[SETUP] Updating existing clone at {PROJECT_DIR}...")
    subprocess.run(["git", "-C", PROJECT_DIR, "pull"], check=False)

os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
print(f"[SETUP] Working directory: {PROJECT_DIR}")

# ── Verify GPU ─────────────────────────────────────────────────────────────────
print("\n[GPU] Checking hardware...")
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        cc = torch.cuda.get_device_capability(0)
        bf16_ok = cc[0] >= 8
        print(f"[GPU] ✅ {gpu_name}  VRAM: {vram_gb:.1f}GB  CC: {cc[0]}.{cc[1]}")
        print(f"[GPU]    bfloat16: {'YES' if bf16_ok else 'NO (using float16 + GradScaler)'}")
    else:
        print("[GPU] ❌ No CUDA GPU found. Retrain requires GPU.")
        print("[GPU]    In Kaggle: Settings > Accelerator > GPU T4 x2")
        sys.exit(1)
except ImportError:
    print("[GPU] PyTorch not available. Something went wrong in setup.")
    sys.exit(1)

# ── Download preprocessed data from S3 ────────────────────────────────────────
print("\n[DATA] Checking preprocessed data...")
try:
    import boto3
    s3 = boto3.client("s3", region_name=AWS_REGION)

    data_path = Path(PROJECT_DIR) / "data" / "preprocessed_full.npz"
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"[DATA] Downloading preprocessed_full.npz from s3://{S3_BUCKET}/data/...")
        s3.download_file(S3_BUCKET, "data/preprocessed_full.npz", str(data_path))
        size_mb = data_path.stat().st_size / 1e6
        print(f"[DATA] ✅ Downloaded ({size_mb:.1f} MB)")
    else:
        size_mb = data_path.stat().st_size / 1e6
        print(f"[DATA] ✅ Already present ({size_mb:.1f} MB)")
except Exception as e:
    print(f"[DATA] ❌ Failed to get preprocessed data: {e}")
    print("[DATA]    Check AWS credentials in Kaggle env vars.")
    sys.exit(1)

# ── Run Training ───────────────────────────────────────────────────────────────
print(f"\n[TRAIN] Starting GPU training...")
print(f"[TRAIN]   Epochs: {EPOCHS}  Batch: 4096  Run: {RUN_NAME}")

cmd = [
    sys.executable, "-m", "src.train",
    "--full",
    "--epochs", str(EPOCHS),
    "--batch-size", "4096",
    "--run-name", RUN_NAME,
    "--context", CONTEXT,
]
print(f"[TRAIN] Command: {' '.join(cmd)}\n")

result = subprocess.run(cmd, cwd=PROJECT_DIR)

if result.returncode != 0:
    print(f"\n[TRAIN] ❌ Training failed (exit code {result.returncode})")
    sys.exit(result.returncode)

print(f"\n[TRAIN] ✅ Training complete!")

# ── Upload results to S3 ───────────────────────────────────────────────────────
print("\n[S3] Uploading artifacts...")

# Benchmark JSON (contains val AUC-PR, AUC-ROC, F1, run_id)
for f in glob.glob(str(Path(PROJECT_DIR) / "data" / "benchmark_*.json")):
    key = f"data/{Path(f).name}"
    s3.upload_file(f, S3_BUCKET, key)
    print(f"[S3] ✅ {key}")

# MLflow SQLite database
mlflow_db = Path(PROJECT_DIR) / "mlflow.db"
if mlflow_db.exists():
    key = f"mlflow/kaggle_{RUN_NAME}.db"
    s3.upload_file(str(mlflow_db), S3_BUCKET, key)
    print(f"[S3] ✅ {key}")

# Model checkpoints
models_dir = Path(PROJECT_DIR) / "models"
if models_dir.exists():
    for pt_file in models_dir.glob("*.pt"):
        key = f"models/{RUN_NAME}/{pt_file.name}"
        s3.upload_file(str(pt_file), S3_BUCKET, key)
        print(f"[S3] ✅ {key}")
else:
    print("[S3] ⚠️  No models/ dir found. train.py may save models elsewhere.")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("P053 Kaggle Training Kernel — COMPLETE")
print(f"  Run name : {RUN_NAME}")
print(f"  Artifacts: s3://{S3_BUCKET}/models/{RUN_NAME}/")
print("=" * 70)
