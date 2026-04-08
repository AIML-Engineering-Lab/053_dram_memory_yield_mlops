"""
P053 — Compute Backend Selector
=================================
AWS-first, Colab-fallback, Local-last-resort.

Purpose:
    Determines WHERE training runs: EC2 GPU, Google Colab GPU, or local.
    Non-training tasks (inference, drift check, ETL) always run locally.
    Only GPU training needs the backend decision.

Decision flow:
    1. Check if running ON Colab (COLAB_GPU env var) → use Colab directly
    2. Check if AWS EC2 is available (instance running + GPU quota OK)
    3. If AWS fails → fall back to Colab (user must run NB04)
    4. If Colab not available → local MacBook (MPS/CPU, slow)

MLflow tracking:
    - AWS EC2 → RDS PostgreSQL
    - Colab   → local SQLite (mlflow.db)
    - Local   → local SQLite or Docker PostgreSQL

Usage:
    from src.compute_backend import get_training_backend, TrainingBackend

    backend = get_training_backend()
    print(backend.name)        # "aws", "colab", or "local"
    print(backend.gpu_name)    # "T4", "A100", "MPS", "CPU"
    print(backend.mlflow_uri)  # "postgresql://..." or "sqlite:///mlflow.db"
"""

import logging
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BackendName(str, Enum):
    AWS = "aws"
    COLAB = "colab"
    LOCAL = "local"


@dataclass
class TrainingBackend:
    """Result of compute backend selection."""
    name: BackendName
    gpu_name: str
    gpu_vram_gb: float
    mlflow_uri: str
    instance_type: Optional[str] = None
    cost_per_hour: float = 0.0
    reason: str = ""
    supports_bf16: bool = False
    recommended_batch_size: int = 2048
    env_overrides: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "backend": self.name.value,
            "gpu_name": self.gpu_name,
            "gpu_vram_gb": self.gpu_vram_gb,
            "mlflow_uri": self.mlflow_uri,
            "instance_type": self.instance_type,
            "cost_per_hour": self.cost_per_hour,
            "reason": self.reason,
            "supports_bf16": self.supports_bf16,
            "recommended_batch_size": self.recommended_batch_size,
        }


def _detect_colab() -> bool:
    """Check if we're running inside Google Colab."""
    return "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ


def _detect_colab_gpu() -> tuple[str, float, bool]:
    """Detect which GPU is available on Colab.

    Returns:
        (gpu_name, vram_gb, supports_bf16)
    """
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            cc = torch.cuda.get_device_capability(0)
            bf16 = cc[0] >= 8  # Ampere+ (A100, A10G)
            return name, round(vram, 1), bf16
    except ImportError:
        pass
    return "Unknown", 0.0, False


def _check_aws_ec2() -> Optional[dict]:
    """Check if AWS EC2 GPU instance is running and reachable.

    Returns instance info dict or None if unavailable.
    """
    ec2_instance_id = os.environ.get("EC2_INSTANCE_ID", "")
    if not ec2_instance_id or ec2_instance_id.startswith("PENDING"):
        logger.info("[BACKEND] EC2 instance ID not configured or pending")
        return None

    try:
        import boto3
        ec2 = boto3.client("ec2", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
        resp = ec2.describe_instances(InstanceIds=[ec2_instance_id])
        reservations = resp.get("Reservations", [])
        if not reservations:
            return None

        instance = reservations[0]["Instances"][0]
        state = instance["State"]["Name"]

        if state != "running":
            logger.info("[BACKEND] EC2 %s is %s (not running)", ec2_instance_id, state)
            return None

        return {
            "instance_id": ec2_instance_id,
            "instance_type": instance["InstanceType"],
            "public_ip": instance.get("PublicIpAddress"),
            "state": state,
        }
    except Exception as e:
        logger.warning("[BACKEND] AWS EC2 check failed: %s", e)
        return None


def _get_mlflow_uri(backend: BackendName) -> str:
    """Get appropriate MLflow tracking URI for the backend."""
    if backend == BackendName.AWS:
        # Use RDS PostgreSQL when on AWS
        host = os.environ.get("MLFLOW_DB_HOST", "")
        port = os.environ.get("MLFLOW_DB_PORT", "5432")
        user = os.environ.get("MLFLOW_DB_USER", "mlflow")
        password = os.environ.get("MLFLOW_DB_PASSWORD", "")
        db = os.environ.get("MLFLOW_DB_NAME", "mlflow")
        if host and password:
            return f"postgresql://{user}:{password}@{host}:{port}/{db}"
        # Fall through to SQLite if RDS not configured
        logger.warning("[BACKEND] RDS not fully configured, falling back to SQLite")

    if backend == BackendName.COLAB:
        # Colab: use local SQLite (persists within session)
        return "sqlite:///mlflow.db"

    # Local: check if Docker PostgreSQL is running, else SQLite
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=mlflow-db", "--format", "{{.Status}}"],
            capture_output=True, text=True, timeout=5,
        )
        if "Up" in result.stdout:
            return "postgresql://mlflow:mlflow@localhost:5432/mlflow"
    except Exception:
        pass

    return "sqlite:///mlflow.db"


def get_training_backend(
    force_backend: Optional[str] = None,
    data_gb_per_day: float = 0.0,
) -> TrainingBackend:
    """Select the optimal training backend.

    Priority: AWS EC2 → Google Colab → Local MacBook

    Args:
        force_backend: Override automatic selection ("aws", "colab", "local")
        data_gb_per_day: Daily data volume in GB (>1000 = need A100)

    Returns:
        TrainingBackend with all details needed to dispatch training
    """
    # Force override
    if force_backend:
        forced = BackendName(force_backend)
        logger.info("[BACKEND] Forced to: %s", forced.value)
        if forced == BackendName.COLAB:
            return _build_colab_backend(data_gb_per_day)
        elif forced == BackendName.AWS:
            return _build_aws_backend()
        else:
            return _build_local_backend()

    # Auto-detect: are we already on Colab?
    if _detect_colab():
        logger.info("[BACKEND] Running on Google Colab — using Colab GPU directly")
        return _build_colab_backend(data_gb_per_day)

    # Try AWS first
    ec2_info = _check_aws_ec2()
    if ec2_info:
        logger.info("[BACKEND] AWS EC2 available: %s (%s)",
                    ec2_info["instance_id"], ec2_info["instance_type"])
        return _build_aws_backend(ec2_info)

    # AWS unavailable — check if user wants Colab fallback
    logger.info("[BACKEND] AWS unavailable — falling back to local. "
                "For GPU training, use NB04_colab_training.ipynb on Google Colab.")
    return _build_local_backend()


def _build_aws_backend(ec2_info: Optional[dict] = None) -> TrainingBackend:
    """Build AWS EC2 backend configuration."""
    instance_type = (ec2_info or {}).get("instance_type",
                                         os.environ.get("EC2_INSTANCE_TYPE", "g4dn.xlarge"))

    # Map instance type to GPU
    gpu_map = {
        "g4dn.xlarge": ("T4", 16.0, False, 0.526, 4096),
        "g4dn.2xlarge": ("T4", 16.0, False, 0.752, 4096),
        "p3.2xlarge": ("V100", 16.0, False, 3.06, 2048),
        "g5.2xlarge": ("A10G", 24.0, True, 1.212, 2048),
        "p4d.24xlarge": ("A100", 80.0, True, 32.77, 8192),
    }

    gpu_name, vram, bf16, cost, batch = gpu_map.get(
        instance_type, ("T4", 16.0, False, 0.526, 4096))

    return TrainingBackend(
        name=BackendName.AWS,
        gpu_name=gpu_name,
        gpu_vram_gb=vram,
        mlflow_uri=_get_mlflow_uri(BackendName.AWS),
        instance_type=instance_type,
        cost_per_hour=cost,
        reason=f"AWS EC2 {instance_type} with {gpu_name} GPU",
        supports_bf16=bf16,
        recommended_batch_size=batch,
        env_overrides={
            "MLFLOW_TRACKING_URI": _get_mlflow_uri(BackendName.AWS),
            "MLFLOW_S3_ENDPOINT_URL": "",
        },
    )


def _build_colab_backend(data_gb_per_day: float = 0.0) -> TrainingBackend:
    """Build Google Colab backend configuration."""
    if _detect_colab():
        gpu_name, vram, bf16 = _detect_colab_gpu()
    else:
        # Planning mode: recommend GPU based on data volume
        if data_gb_per_day > 1000:  # >1TB/day → need A100
            gpu_name, vram, bf16 = "A100-SXM4-40GB", 40.0, True
        else:
            gpu_name, vram, bf16 = "Tesla T4", 15.0, False

    # CU cost estimate (Colab Pro)
    colab_cu_map = {
        "T4": (1.36, 4096),
        "Tesla T4": (1.36, 4096),
        "V100": (2.72, 2048),
        "Tesla V100-SXM2-16GB": (2.72, 2048),
        "A100": (6.79, 8192),
        "A100-SXM4-40GB": (6.79, 8192),
    }

    # Match by prefix
    cu_per_hour, batch = 1.36, 4096  # default T4
    for key, (cu, bs) in colab_cu_map.items():
        if key in gpu_name:
            cu_per_hour, batch = cu, bs
            break

    return TrainingBackend(
        name=BackendName.COLAB,
        gpu_name=gpu_name,
        gpu_vram_gb=vram,
        mlflow_uri=_get_mlflow_uri(BackendName.COLAB),
        instance_type="colab",
        cost_per_hour=cu_per_hour,  # In CU, not USD
        reason=f"Google Colab with {gpu_name} ({vram}GB VRAM). "
               f"~{cu_per_hour} CU/hr.",
        supports_bf16=bf16,
        recommended_batch_size=batch,
        env_overrides={
            "MLFLOW_TRACKING_URI": "sqlite:///mlflow.db",
        },
    )


def _build_local_backend() -> TrainingBackend:
    """Build local MacBook backend configuration."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return TrainingBackend(
                name=BackendName.LOCAL,
                gpu_name="Apple MPS",
                gpu_vram_gb=0,  # Shared memory
                mlflow_uri=_get_mlflow_uri(BackendName.LOCAL),
                cost_per_hour=0.0,
                reason="Local MacBook with Apple MPS. Slow for large datasets.",
                supports_bf16=False,
                recommended_batch_size=1024,
            )
    except ImportError:
        pass

    return TrainingBackend(
        name=BackendName.LOCAL,
        gpu_name="CPU",
        gpu_vram_gb=0,
        mlflow_uri=_get_mlflow_uri(BackendName.LOCAL),
        cost_per_hour=0.0,
        reason="Local CPU only. Very slow — consider Colab for GPU training.",
        supports_bf16=False,
        recommended_batch_size=512,
    )


def dispatch_training(backend: TrainingBackend, day: int,
                      run_name: str, project_root: Path,
                      extra_args: Optional[list] = None) -> dict:
    """Execute training on the selected backend.

    For AWS and Local: runs train.py as subprocess.
    For Colab: runs train.py directly (already on Colab).

    Args:
        backend: Selected training backend
        day: Simulation day number
        run_name: MLflow run name
        project_root: Path to project root
        extra_args: Additional CLI args for train.py

    Returns:
        dict with training results or error info
    """
    import sys

    # Set environment overrides
    env = os.environ.copy()
    env.update(backend.env_overrides)
    env["COMPUTE_BACKEND"] = backend.name.value
    env["COMPUTE_GPU"] = backend.gpu_name

    cmd = [
        sys.executable, "-m", "src.train",
        "--full",
        "--batch-size", str(backend.recommended_batch_size),
        "--run-name", run_name,
        "--context", f"{backend.name.value}-retrain",
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("[BACKEND] Training dispatch: backend=%s, gpu=%s, cmd=%s",
                backend.name.value, backend.gpu_name, " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=7200,
            env=env,
        )

        if result.returncode == 0:
            # Read benchmark
            import glob
            import json
            benchmark_files = sorted(
                glob.glob(str(project_root / "data" / "benchmark_*.json")),
                key=lambda f: Path(f).stat().st_mtime,
                reverse=True,
            )
            if benchmark_files:
                with open(benchmark_files[0]) as f:
                    benchmark = json.load(f)
                return {
                    "status": "success",
                    "backend": backend.name.value,
                    "gpu": backend.gpu_name,
                    "val_auc_pr": benchmark["results"]["val"]["auc_pr"],
                    "mlflow_run_id": benchmark["mlflow_run_id"],
                    "train_time_min": benchmark["total_train_time_min"],
                    "benchmark": benchmark,
                }

            return {"status": "success", "backend": backend.name.value,
                    "note": "Training succeeded but no benchmark file found"}

        return {
            "status": "failed",
            "backend": backend.name.value,
            "exit_code": result.returncode,
            "stderr": result.stderr[-1000:] if result.stderr else "",
            "reason": f"train.py exited with code {result.returncode}",
        }

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "backend": backend.name.value,
                "reason": "Training timed out after 2 hours"}
    except Exception as e:
        return {"status": "error", "backend": backend.name.value,
                "reason": str(e)}


def dispatch_training_with_fallback(day: int, run_name: str,
                                    project_root: Path,
                                    data_gb_per_day: float = 0.0) -> dict:
    """Try AWS → Colab → Local, returning first success.

    This is the main entry point for the simulation runner.
    Non-training days don't call this — only retrain days.
    """
    results = []

    # Try 1: AWS
    aws_backend = get_training_backend(force_backend="aws", data_gb_per_day=data_gb_per_day)
    if aws_backend.name == BackendName.AWS:
        ec2_info = _check_aws_ec2()
        if ec2_info:
            logger.info("[FALLBACK] Attempt 1: AWS EC2 (%s)", aws_backend.gpu_name)
            result = dispatch_training(aws_backend, day, run_name, project_root)
            results.append(result)
            if result["status"] == "success":
                return result
            logger.warning("[FALLBACK] AWS failed: %s", result.get("reason", "unknown"))

    # Try 2: Colab (only works if already on Colab)
    if _detect_colab():
        colab_backend = _build_colab_backend(data_gb_per_day)
        logger.info("[FALLBACK] Attempt 2: Google Colab (%s)", colab_backend.gpu_name)
        result = dispatch_training(colab_backend, day, run_name, project_root)
        results.append(result)
        if result["status"] == "success":
            return result
        logger.warning("[FALLBACK] Colab failed: %s", result.get("reason", "unknown"))

    # Try 3: Local
    local_backend = _build_local_backend()
    logger.info("[FALLBACK] Attempt 3: Local (%s)", local_backend.gpu_name)
    result = dispatch_training(local_backend, day, run_name, project_root)
    results.append(result)

    if result["status"] == "success":
        return result

    # All failed
    return {
        "status": "all_failed",
        "attempts": results,
        "reason": "Training failed on all backends (AWS → Colab → Local)",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    print("P053 Compute Backend Selector")
    print("=" * 50)

    backend = get_training_backend()
    print(f"  Backend:    {backend.name.value}")
    print(f"  GPU:        {backend.gpu_name} ({backend.gpu_vram_gb} GB)")
    print(f"  MLflow URI: {backend.mlflow_uri}")
    print(f"  Cost/hr:    ${backend.cost_per_hour:.2f}")
    print(f"  bf16:       {backend.supports_bf16}")
    print(f"  Batch size: {backend.recommended_batch_size}")
    print(f"  Reason:     {backend.reason}")
    print()

    # Also show what each backend would look like
    for name in ["aws", "colab", "local"]:
        b = get_training_backend(force_backend=name)
        print(f"  [{name.upper():>5}] {b.gpu_name:>20} | {b.mlflow_uri[:40]:>40} | ${b.cost_per_hour:.2f}/hr")
