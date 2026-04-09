"""
P053 — Kaggle Kernels Training Backend
========================================
Triggers a Kaggle kernel to run GPU training when AWS EC2 is unavailable.
Kaggle provides 30hr/week free T4/P100 GPU — fully automatable via API.

Fallback position in chain: AWS EC2 → Kaggle (HERE) → Colab → Local Mac

One-time setup:
    1. pip install kaggle
    2. Get API key: kaggle.com → Account → Create New API Token
       Saves to ~/.kaggle/kaggle.json: {"username": "...", "key": "..."}
    3. Set KAGGLE_USERNAME in .env or shell
    4. Set AWS credentials as Kaggle env vars (kaggle.com → Account → Environment Variables):
         AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, S3_BUCKET
    5. Set GITHUB_REPO_URL as Kaggle env var if repo is private (add GitHub token)
    6. Run bash scripts/kaggle_training/setup_kaggle_env.sh to verify

Usage:
    from src.kaggle_backend import check_kaggle_available, trigger_training_kernel

    if check_kaggle_available():
        success = trigger_training_kernel(run_name="retrain-day15", epochs=10)
"""

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = PROJECT_ROOT / "scripts" / "kaggle_training"


# ── Availability check ────────────────────────────────────────────────────────

def check_kaggle_available() -> bool:
    """Check if Kaggle API is configured and a kernel script exists."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended  # noqa: F401
    except ImportError:
        logger.warning("[KAGGLE] kaggle package not installed. Run: pip install kaggle")
        return False

    username = get_kaggle_username()
    if not username:
        logger.warning("[KAGGLE] KAGGLE_USERNAME not set and ~/.kaggle/kaggle.json missing.")
        return False

    kernel_script = KERNEL_DIR / "kaggle_train_kernel.py"
    if not kernel_script.exists():
        logger.warning("[KAGGLE] Kernel script not found: %s", kernel_script)
        return False

    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()
        logger.info("[KAGGLE] API available for user: %s", username)
        return True
    except Exception as e:
        logger.warning("[KAGGLE] Auth failed: %s", e)
        return False


def get_kaggle_username() -> str:
    """Get Kaggle username from env or ~/.kaggle/kaggle.json."""
    username = os.environ.get("KAGGLE_USERNAME", "")
    if not username:
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            try:
                with open(kaggle_json) as f:
                    creds = json.load(f)
                username = creds.get("username", "")
            except Exception:
                pass
    return username


# ── Kernel trigger + polling ───────────────────────────────────────────────────

def _write_run_config(run_name: str, epochs: int, context: str) -> None:
    """Write per-run config to kernel dir so the kernel script reads it."""
    config = {
        "run_name": run_name,
        "epochs": epochs,
        "context": context,
        "s3_bucket": os.environ.get("S3_BUCKET", "p053-mlflow-artifacts"),
        "aws_region": os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
        "github_repo": os.environ.get(
            "GITHUB_REPO_URL",
            "https://github.com/AIML-Engineering-Lab/053_dram_memory_yield_mlops.git",
        ),
        "triggered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    config_path = KERNEL_DIR / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("[KAGGLE] Written run config to %s", config_path)


def trigger_training_kernel(
    run_name: str = "retrain",
    epochs: int = 10,
    context: str = "kaggle-retrain",
    max_wait_minutes: int = 120,
    poll_interval_seconds: int = 90,
) -> bool:
    """Push a Kaggle kernel and wait for GPU training to complete.

    Args:
        run_name: MLflow run name for this job (e.g. "retrain-day15-v2")
        epochs: Training epochs. Keep <=10 for simulation retrains (~30-40min on T4).
                Use 50 for production-quality retrains (~8-10hr on T4).
        context: Training context tag logged to MLflow.
        max_wait_minutes: Timeout. Kaggle T4 limit is 12hr/session.
        poll_interval_seconds: How often to check kernel status.

    Returns:
        True if kernel completed successfully and artifacts are on S3.
        False on error, timeout, or auth failure.
    """
    if not check_kaggle_available():
        return False

    from kaggle.api.kaggle_api_extended import KaggleApiExtended

    api = KaggleApiExtended()
    api.authenticate()

    username = get_kaggle_username()
    kernel_slug = os.environ.get("KAGGLE_KERNEL_SLUG", "p053-dram-yield-retrain")

    # Write per-run config (the kernel script reads this on startup)
    _write_run_config(run_name=run_name, epochs=epochs, context=context)

    # Push the kernel (starts a new run)
    print(f"\n  [KAGGLE] Pushing kernel {username}/{kernel_slug}")
    print(f"  [KAGGLE]   run_name={run_name}  epochs={epochs}  context={context}")
    print(f"  [KAGGLE]   Monitor: https://www.kaggle.com/code/{username}/{kernel_slug}")

    try:
        api.kernels_push_cli(folder=str(KERNEL_DIR))
        print(f"  [KAGGLE] ✅ Kernel pushed and queued.")
    except Exception as e:
        logger.error("[KAGGLE] Kernel push failed: %s", e)
        print(f"  [KAGGLE] ❌ Push failed: {e}")
        print(f"  [KAGGLE]   Possible fixes:")
        print(f"  [KAGGLE]   1. Check kernel-metadata.json has your Kaggle username")
        print(f"  [KAGGLE]   2. Verify ~/.kaggle/kaggle.json is valid")
        print(f"  [KAGGLE]   3. Run: bash scripts/kaggle_training/setup_kaggle_env.sh")
        return False

    # Poll for completion
    print(f"\n  [KAGGLE] Polling status (every {poll_interval_seconds}s, "
          f"timeout: {max_wait_minutes}min)...")
    start_time = time.time()
    deadline = start_time + (max_wait_minutes * 60)
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        time.sleep(poll_interval_seconds)

        try:
            kernel = api.kernel_status(username, kernel_slug)
            status = kernel.status
            elapsed_min = (time.time() - start_time) / 60

            # Print status every 5 polls or on final status
            if attempt % 5 == 0 or status in ("complete", "error", "cancelAcknowledged"):
                print(f"  [KAGGLE]   Status: {status:<20} ({elapsed_min:.0f}min elapsed)")

            if status == "complete":
                print(f"  [KAGGLE] ✅ Training complete after {elapsed_min:.0f}min!")
                _verify_s3_artifacts(run_name)
                return True

            if status in ("error", "cancelAcknowledged"):
                print(f"  [KAGGLE] ❌ Kernel ended with status: {status}")
                print(f"  [KAGGLE]   View logs: https://www.kaggle.com/code/{username}/{kernel_slug}")
                return False

        except Exception as e:
            logger.warning("[KAGGLE] Status poll failed (attempt %d): %s", attempt, e)

    elapsed_min = (time.time() - start_time) / 60
    print(f"  [KAGGLE] ⏰ Timeout after {elapsed_min:.0f}min (limit: {max_wait_minutes}min).")
    print(f"  [KAGGLE]   Kernel may still be running.")
    print(f"  [KAGGLE]   Check: https://www.kaggle.com/code/{username}/{kernel_slug}")
    return False


def _verify_s3_artifacts(run_name: str) -> None:
    """Check that training artifacts were uploaded to S3 by the kernel."""
    try:
        import boto3
        s3 = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
        bucket = os.environ.get("S3_BUCKET", "p053-mlflow-artifacts")
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=f"models/{run_name}/", MaxKeys=10)
        count = resp.get("KeyCount", 0)
        if count > 0:
            print(f"  [KAGGLE] ✅ S3 artifacts verified: {count} object(s) "
                  f"at s3://{bucket}/models/{run_name}/")
        else:
            print(f"  [KAGGLE] ⚠️  No S3 artifacts found at s3://{bucket}/models/{run_name}/")
            print(f"  [KAGGLE]   The kernel may not have uploaded results. Check kernel logs.")
    except Exception as e:
        logger.warning("[KAGGLE] S3 artifact verification failed: %s", e)


# ── Interactive wait helper (used by compute_backend fallback chain) ──────────

def interactive_wait_with_skip(message: str, wait_hours: float = 2.0) -> None:
    """Wait up to wait_hours, or until user presses Enter.

    Non-blocking: if stdin is not a TTY (CI/CD, subprocess), waits only 5s.
    Interactive: prints countdown every minute, Enter skips the wait.

    Args:
        message: What to display (e.g. "Kaggle failed. Options: ...")
        wait_hours: Maximum wait in hours before auto-proceeding.
    """
    wait_sec = int(wait_hours * 3600)
    border = "=" * 70

    print(f"\n{border}")
    print(message)
    if sys.stdin.isatty():
        print(f"\n  ⏱  Waiting up to {wait_hours:.0f}h for the situation to resolve.")
        print(f"  ↩  Press Enter at any time to skip waiting and continue.")
    else:
        print(f"  (Non-interactive mode — short pause only)")
    print(border)

    # Non-interactive (CI/CD, Airflow, piped): just pause briefly
    if not sys.stdin.isatty():
        time.sleep(5)
        return

    skipped = threading.Event()

    def _listen_for_enter():
        try:
            sys.stdin.readline()
            skipped.set()
        except Exception:
            pass

    listener = threading.Thread(target=_listen_for_enter, daemon=True)
    listener.start()

    start = time.time()
    report_interval = 300  # Print status every 5 minutes
    last_report = start

    while time.time() - start < wait_sec and not skipped.is_set():
        now = time.time()
        if now - last_report >= report_interval:
            elapsed_min = (now - start) / 60
            remaining_hr = (wait_sec - (now - start)) / 3600
            print(f"  [WAIT] {elapsed_min:.0f}min elapsed, "
                  f"{remaining_hr:.1f}hr remaining. Press Enter to skip.")
            last_report = now
        time.sleep(0.5)

    if skipped.is_set():
        print(f"  [SKIP] Proceeding to next fallback immediately.")
    else:
        print(f"  [TIMEOUT] {wait_hours:.0f}h wait expired. Proceeding to next fallback.")
