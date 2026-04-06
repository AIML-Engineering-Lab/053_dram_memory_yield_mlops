"""
P053 — S3 Artifact Management
===============================
Production-grade S3 integration for model artifacts, training data,
and experiment outputs.

Usage:
    from src.s3_utils import S3ArtifactManager

    s3 = S3ArtifactManager()
    s3.upload_model("models/retrain_day31/model.pt", local_path)
    s3.upload_training_data(day=31, parquet_path=path)
    s3.download_model("models/champion/model.pt", local_path)

Environment:
    AWS credentials via IAM role (EC2) or env vars:
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

Interview talking point:
    "All artifacts are stored on S3 with versioning enabled. Models go to
    s3://bucket/models/, training Parquets to s3://bucket/data/, and drift
    reports to s3://bucket/drift/. The MLflow artifact root also points to
    S3 so the tracking server is fully stateless."
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

S3_BUCKET = os.environ.get("S3_BUCKET", "p053-mlflow-artifacts")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")


def _get_s3_client():
    """Get boto3 S3 client. Raises ImportError if boto3 not installed."""
    import boto3
    return boto3.client("s3", region_name=AWS_REGION)


class S3ArtifactManager:
    """Manages S3 uploads/downloads for P053 MLOps pipeline."""

    def __init__(self, bucket: str = S3_BUCKET):
        self.bucket = bucket
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = _get_s3_client()
        return self._client

    def upload_file(self, local_path: str, s3_key: str) -> str:
        """Upload a single file to S3. Returns the S3 URI."""
        self.client.upload_file(str(local_path), self.bucket, s3_key)
        uri = f"s3://{self.bucket}/{s3_key}"
        logger.info("Uploaded %s → %s", local_path, uri)
        return uri

    def upload_model(self, day: int, model_path: str,
                     version_tag: str = "retrained") -> str:
        """Upload trained model artifact to S3."""
        filename = Path(model_path).name
        s3_key = f"models/day{day:02d}_{version_tag}/{filename}"
        return self.upload_file(model_path, s3_key)

    def upload_training_data(self, day: int, parquet_path: str) -> str:
        """Upload daily Parquet data to S3."""
        filename = Path(parquet_path).name
        s3_key = f"data/production/day_{day:02d}/{filename}"
        return self.upload_file(parquet_path, s3_key)

    def upload_drift_report(self, day: int, report_path: str) -> str:
        """Upload drift detection report to S3."""
        filename = Path(report_path).name
        s3_key = f"drift/day_{day:02d}/{filename}"
        return self.upload_file(report_path, s3_key)

    def upload_benchmark(self, day: int, benchmark_path: str) -> str:
        """Upload training benchmark JSON to S3."""
        filename = Path(benchmark_path).name
        s3_key = f"benchmarks/day{day:02d}_{filename}"
        return self.upload_file(benchmark_path, s3_key)

    def upload_directory(self, local_dir: str, s3_prefix: str,
                         pattern: str = "*") -> list[str]:
        """Upload all matching files in a directory to S3."""
        uploaded = []
        local_dir = Path(local_dir)
        for filepath in local_dir.glob(pattern):
            if filepath.is_file():
                s3_key = f"{s3_prefix}/{filepath.name}"
                uri = self.upload_file(str(filepath), s3_key)
                uploaded.append(uri)
        return uploaded

    def download_file(self, s3_key: str, local_path: str) -> str:
        """Download a file from S3."""
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, s3_key, str(local_path))
        logger.info("Downloaded s3://%s/%s → %s", self.bucket, s3_key, local_path)
        return local_path

    def download_champion_model(self, local_dir: str) -> Optional[str]:
        """Download the current champion model from S3."""
        try:
            resp = self.client.list_objects_v2(
                Bucket=self.bucket,
                Prefix="models/",
                MaxKeys=100,
            )
            if "Contents" not in resp:
                logger.warning("No models found in S3")
                return None

            # Find most recent champion model
            model_keys = [
                obj["Key"] for obj in resp["Contents"]
                if obj["Key"].endswith(".pt")
            ]
            if not model_keys:
                return None

            latest = sorted(model_keys)[-1]
            local_path = str(Path(local_dir) / Path(latest).name)
            return self.download_file(latest, local_path)
        except Exception as e:
            logger.error("Failed to download champion model: %s", e)
            return None

    def file_exists(self, s3_key: str) -> bool:
        """Check if a file exists in S3."""
        try:
            self.client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except Exception:
            return False


def upload_simulation_artifacts(day: int, data_dir: str,
                                skip_parquet: bool = False) -> dict:
    """
    Upload all artifacts for a simulation day to S3.

    Called after each day's pipeline completes.
    Returns dict of uploaded URIs.
    """
    try:
        s3 = S3ArtifactManager()
    except ImportError:
        logger.warning("boto3 not installed — skipping S3 upload")
        return {"status": "skipped", "reason": "boto3 not installed"}
    except Exception as e:
        logger.warning("S3 client init failed: %s — skipping upload", e)
        return {"status": "skipped", "reason": str(e)}

    data_dir = Path(data_dir)
    uploaded = {}

    # Upload Parquet data
    if not skip_parquet:
        parquet_path = data_dir / "production" / f"day_{day:02d}.parquet"
        if parquet_path.exists():
            uploaded["parquet"] = s3.upload_training_data(day, str(parquet_path))

    # Upload drift report
    drift_path = data_dir / "drift_reports" / f"drift_day_{day:02d}.json"
    if drift_path.exists():
        uploaded["drift_report"] = s3.upload_drift_report(day, str(drift_path))

    # Upload retrain results
    retrain_path = data_dir / "retrain_results" / f"retrain_day_{day:02d}.json"
    if retrain_path.exists():
        uploaded["retrain_result"] = s3.upload_file(
            str(retrain_path),
            f"retrain/day_{day:02d}/retrain_result.json",
        )

    # Upload canary results
    canary_path = data_dir / "canary_results" / f"canary_day_{day:02d}.json"
    if canary_path.exists():
        uploaded["canary_result"] = s3.upload_file(
            str(canary_path),
            f"canary/day_{day:02d}/canary_result.json",
        )

    return uploaded
