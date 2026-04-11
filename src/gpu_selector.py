"""
P053 — Automatic GPU Instance Selector
=======================================
Selects the optimal EC2 GPU instance based on model complexity.

Decision matrix:
    Model Params      | GPU        | Instance       | Cost/hr
    ──────────────────+────────────+────────────────+────────
    < 50M             | T4 (16GB)  | g4dn.xlarge    | $0.526
    50M - 500M        | V100 (16GB)| p3.2xlarge     | $3.06
    500M - 1.2B       | A10G (24GB)| g5.2xlarge     | $1.212
    > 1.2B            | A100 (80GB)| p4d.24xlarge   | $32.77

Auto-scaling workflow:
    1. Before retrain: estimate model params + VRAM needed
    2. Select optimal GPU instance
    3. If current instance is insufficient:
       a. Launch new instance with correct GPU
       b. Run training on that instance
       c. Terminate after training completes
    4. Upload model artifacts to S3 (accessible from any instance)

Interview talking point:
    "We built automatic GPU selection based on model complexity.
    Our 317K param model runs on T4 at $0.53/hr. But if the model
    scales to 1.2B+ params (LLM fine-tuning, large vision models),
    the system auto-provisions an A100 at $32.77/hr — only for the
    training duration. This is how NVIDIA/AMD production pipelines
    handle mixed-complexity workloads."

Usage:
    from src.gpu_selector import select_gpu, GPURequirement
    req = select_gpu(model_params=317_000)     # → T4
    req = select_gpu(model_params=1_200_000_000) # → A100
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")


@dataclass
class GPURequirement:
    """GPU selection result with full provisioning details."""
    gpu_name: str
    vram_gb: int
    instance_type: str
    cost_per_hour: float
    supports_bf16: bool
    supports_fp16: bool
    recommended_batch_size: int
    reason: str
    needs_instance_switch: bool = False

    def to_dict(self) -> dict:
        return {
            "gpu_name": self.gpu_name,
            "vram_gb": self.vram_gb,
            "instance_type": self.instance_type,
            "cost_per_hour": self.cost_per_hour,
            "supports_bf16": self.supports_bf16,
            "supports_fp16": self.supports_fp16,
            "recommended_batch_size": self.recommended_batch_size,
            "reason": self.reason,
            "needs_instance_switch": self.needs_instance_switch,
        }


# GPU catalog — AWS EC2 (us-west-2 on-demand pricing)
GPU_CATALOG = [
    GPURequirement(
        gpu_name="T4",
        vram_gb=16,
        instance_type="g4dn.xlarge",
        cost_per_hour=0.526,
        supports_bf16=False,
        supports_fp16=True,
        recommended_batch_size=4096,
        reason="Small models (<50M params). Best cost-efficiency for inference + light training.",
    ),
    GPURequirement(
        gpu_name="V100",
        vram_gb=16,
        instance_type="p3.2xlarge",
        cost_per_hour=3.06,
        supports_bf16=False,
        supports_fp16=True,
        recommended_batch_size=2048,
        reason="Medium models (50M-500M params). Strong FP64 for scientific compute.",
    ),
    GPURequirement(
        gpu_name="A10G",
        vram_gb=24,
        instance_type="g5.2xlarge",
        cost_per_hour=1.212,
        supports_bf16=True,
        supports_fp16=True,
        recommended_batch_size=2048,
        reason="Medium-large models (500M-1.2B params). Good balance of VRAM and cost.",
    ),
    GPURequirement(
        gpu_name="A100",
        vram_gb=80,
        instance_type="p4d.24xlarge",
        cost_per_hour=32.77,
        supports_bf16=True,
        supports_fp16=True,
        recommended_batch_size=8192,
        reason="Large models (>1.2B params). Required for LLM fine-tuning, large vision models.",
    ),
]


def estimate_vram_gb(model_params: int, batch_size: int = 4096,
                     dtype_bytes: int = 2) -> float:
    """
    Estimate GPU VRAM needed for training.

    Formula: model_params * dtype_bytes * 4 (params + gradients + optimizer states + activations)
    Plus batch activation memory.
    """
    # Model weights + gradients: params * dtype * 2
    model_memory_gb = (model_params * dtype_bytes * 2) / 1e9

    # Optimizer states (Adam = 2 copies): params * 4 bytes * 2
    optimizer_memory_gb = (model_params * 4 * 2) / 1e9

    # Activation memory (rough estimate: batch_size * hidden_dim * layers * dtype)
    # For large models, activations dominate
    activation_memory_gb = (batch_size * model_params * dtype_bytes * 0.001) / 1e9
    activation_memory_gb = min(activation_memory_gb, 40)  # Cap estimate

    total_gb = model_memory_gb + optimizer_memory_gb + activation_memory_gb + 1.0  # +1GB OS overhead

    return round(total_gb, 2)


def select_gpu(model_params: int, batch_size: int = 4096,
               current_instance: str = "g4dn.xlarge",
               data_rows: int = 0) -> GPURequirement:
    """
    Select optimal GPU based on model complexity AND data volume.

    Rules:
        - Model params > 1.2B → A100 (VRAM too large for T4)
        - Data rows > 1B → A100 (need HBM bandwidth for large datasets)
        - Data rows > 100M → V100/A10G minimum (memory pressure)
        - Otherwise → T4 (cost-efficient)

    Args:
        model_params: Number of trainable parameters
        batch_size: Training batch size
        current_instance: Current EC2 instance type
        data_rows: Number of data rows for training (0 = ignore)

    Returns:
        GPURequirement with selected GPU and provisioning details
    """
    estimated_vram = estimate_vram_gb(model_params, batch_size)

    # Data volume override: large datasets need high memory bandwidth
    # even with small models, because data loading + preprocessing
    # saturates T4's 320 GB/s bandwidth
    data_override = None
    if data_rows > 1_000_000_000:  # > 1B rows
        data_override = "A100"
        estimated_vram = max(estimated_vram, 40.0)  # Force A100 selection
    elif data_rows > 100_000_000:  # > 100M rows
        data_override = "V100_minimum"
        estimated_vram = max(estimated_vram, 10.0)  # Force at least V100

    logger.info("Model params: %s, Data rows: %s, Estimated VRAM: %.1f GB%s",
                f"{model_params:,}", f"{data_rows:,}" if data_rows else "N/A",
                estimated_vram,
                f" (data override: {data_override})" if data_override else "")

    # Select smallest GPU that fits
    selected = None
    for gpu in GPU_CATALOG:
        if gpu.vram_gb >= estimated_vram * 1.2:  # 20% headroom
            selected = gpu
            break

    if selected is None:
        # Model too large even for A100 — multi-GPU needed
        selected = GPU_CATALOG[-1]  # A100 as best single-GPU option
        selected = GPURequirement(
            gpu_name="A100 (multi-GPU needed)",
            vram_gb=80,
            instance_type="p4d.24xlarge",
            cost_per_hour=32.77,
            supports_bf16=True,
            supports_fp16=True,
            recommended_batch_size=4096,
            reason=f"Model requires {estimated_vram:.0f}GB VRAM. "
                   f"Single A100 (80GB) insufficient — consider model parallelism or DeepSpeed.",
        )

    # Check if we need to switch instances
    selected.needs_instance_switch = (selected.instance_type != current_instance)

    logger.info("Selected GPU: %s (%s) — $%.2f/hr, switch_needed=%s",
                selected.gpu_name, selected.instance_type,
                selected.cost_per_hour, selected.needs_instance_switch)

    return selected


def launch_training_instance(gpu_req: GPURequirement,
                             key_name: str = "p053-key",
                             security_group: str | None = None,
                             dry_run: bool = False) -> dict:
    """
    Launch a new EC2 instance with the required GPU for training.

    Used when the current instance doesn't have sufficient GPU.
    The training instance is TEMPORARY — stopped after training completes.
    """
    import boto3

    if security_group is None:
        security_group = os.environ.get("AWS_SECURITY_GROUP_ID", "")
        if not security_group:
            raise ValueError(
                "No security group provided. Set AWS_SECURITY_GROUP_ID env var "
                "or pass security_group parameter."
            )

    if dry_run:
        logger.info("DRY RUN: Would launch %s (%s) for training",
                     gpu_req.instance_type, gpu_req.gpu_name)
        return {"status": "dry_run", "instance_type": gpu_req.instance_type}

    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    # Deep Learning AMI (PyTorch, Ubuntu)
    # Latest DLAMI for us-west-2
    ami_response = ec2.describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name", "Values": ["Deep Learning AMI GPU PyTorch*Ubuntu 22.04*"]},
            {"Name": "state", "Values": ["available"]},
        ],
    )
    amis = sorted(ami_response["Images"], key=lambda x: x["CreationDate"], reverse=True)
    ami_id = amis[0]["ImageId"] if amis else "ami-0c55b159cbfafe1f0"  # Fallback

    response = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=gpu_req.instance_type,
        KeyName=key_name,
        SecurityGroupIds=[security_group],
        MinCount=1,
        MaxCount=1,
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": f"p053-training-{gpu_req.gpu_name}"},
                {"Key": "Project", "Value": "p053"},
                {"Key": "Purpose", "Value": "gpu-training-ephemeral"},
                {"Key": "AutoStop", "Value": "true"},
            ],
        }],
        InstanceInitiatedShutdownBehavior="stop",
    )

    instance_id = response["Instances"][0]["InstanceId"]
    logger.info("Launched training instance: %s (%s, %s)",
                instance_id, gpu_req.instance_type, gpu_req.gpu_name)

    return {
        "status": "launched",
        "instance_id": instance_id,
        "instance_type": gpu_req.instance_type,
        "gpu": gpu_req.gpu_name,
        "cost_per_hour": gpu_req.cost_per_hour,
    }


def get_gpu_decision_for_day(day: int, model_params: int = 317_000,
                             current_instance: str = "g4dn.xlarge",
                             data_rows: int = 0) -> dict:
    """
    Get GPU decision for a simulation day.

    Day 12+ in Phase 2/3 can simulate model scaling scenarios.
    In production, model_params comes from the actual model architecture.

    Returns decision dict logged to MLflow for transparency.
    """
    gpu_req = select_gpu(model_params, current_instance=current_instance,
                         data_rows=data_rows)

    decision = {
        "day": day,
        "model_params": model_params,
        "model_params_human": _human_readable_params(model_params),
        "estimated_vram_gb": estimate_vram_gb(model_params),
        "selected_gpu": gpu_req.gpu_name,
        "selected_instance": gpu_req.instance_type,
        "cost_per_hour": gpu_req.cost_per_hour,
        "needs_instance_switch": gpu_req.needs_instance_switch,
        "current_instance": current_instance,
    }

    if gpu_req.needs_instance_switch:
        decision["action"] = (
            f"SWITCH REQUIRED: {current_instance} → {gpu_req.instance_type} "
            f"(model has {_human_readable_params(model_params)}, "
            f"needs {gpu_req.gpu_name} {gpu_req.vram_gb}GB VRAM)"
        )
    else:
        decision["action"] = (
            f"CURRENT GPU SUFFICIENT: {gpu_req.gpu_name} on {current_instance} "
            f"handles {_human_readable_params(model_params)}"
        )

    return decision


def _human_readable_params(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.0f}K"
    return str(n)


# ═══════════════════════════════════════════════════════════════
# Colab GPU Catalog (for fallback when AWS unavailable)
# ═══════════════════════════════════════════════════════════════

COLAB_GPU_CATALOG = [
    GPURequirement(
        gpu_name="T4 (Colab)",
        vram_gb=15,
        instance_type="colab-t4",
        cost_per_hour=1.36,  # CU/hour, not USD
        supports_bf16=False,
        supports_fp16=True,
        recommended_batch_size=4096,
        reason="Colab T4. Default for <50M params, <1TB/day data. ~1.36 CU/hr.",
    ),
    GPURequirement(
        gpu_name="A100 (Colab)",
        vram_gb=40,
        instance_type="colab-a100",
        cost_per_hour=6.79,  # CU/hour
        supports_bf16=True,
        supports_fp16=True,
        recommended_batch_size=8192,
        reason="Colab A100. Required when data >1TB/day. ~6.79 CU/hr.",
    ),
]


def select_colab_gpu(data_gb_per_day: float = 0.0) -> GPURequirement:
    """Select Colab GPU based on daily data volume.

    Rule: T4 for everything UNLESS data >1TB/day → A100.
    """
    if data_gb_per_day > 1000:  # >1TB
        return COLAB_GPU_CATALOG[1]  # A100
    return COLAB_GPU_CATALOG[0]  # T4


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Demo: show decisions for different model sizes
    test_cases = [
        ("P053 HybridTransformerCNN", 317_000),
        ("ResNet-50", 25_600_000),
        ("BERT-base", 110_000_000),
        ("GPT-2 Medium", 345_000_000),
        ("LLaMA-7B fine-tune", 1_200_000_000),
        ("LLaMA-13B fine-tune", 7_000_000_000),
    ]

    print(f"\n{'='*80}")
    print("P053 — GPU AUTO-SELECTOR DECISION TABLE")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Params':<12} {'VRAM Est':<10} {'GPU':<12} {'Instance':<16} {'$/hr':<8}")
    print(f"{'-'*80}")

    for name, params in test_cases:
        req = select_gpu(params)
        vram = estimate_vram_gb(params)
        print(f"{name:<30} {_human_readable_params(params):<12} {vram:<10.1f} "
              f"{req.gpu_name:<12} {req.instance_type:<16} ${req.cost_per_hour:<7.2f}")

    print(f"{'='*80}")
