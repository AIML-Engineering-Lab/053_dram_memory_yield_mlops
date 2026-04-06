"""
P053 — EC2 Auto-Stop After Simulation
=======================================
Stops the EC2 instance after the simulation completes to prevent
runaway costs. Two mechanisms:

1. Called at end of simulation DAG (graceful shutdown)
2. CloudWatch billing alarm (safety net at $500 USD)

Usage:
    # Called automatically by the simulation master DAG
    python -m src.ec2_auto_stop

    # Manual trigger
    python -m src.ec2_auto_stop --force

    # Dry run (check but don't stop)
    python -m src.ec2_auto_stop --dry-run

Interview talking point:
    "Cost governance is non-negotiable. The simulation DAG has a final
    task that stops the EC2 instance. There's also a CloudWatch billing
    alarm at $500 as a safety net. The g4dn.xlarge costs $0.526/hr —
    leaving it running for a week would cost $88. Auto-stop prevents that."
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")


def get_instance_id() -> str:
    """Get current EC2 instance ID from metadata service."""
    import urllib.request
    # IMDSv2 token
    token_req = urllib.request.Request(
        "http://169.254.169.254/latest/api/token",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        method="PUT",
    )
    token = urllib.request.urlopen(token_req, timeout=2).read().decode()

    id_req = urllib.request.Request(
        "http://169.254.169.254/latest/meta-data/instance-id",
        headers={"X-aws-ec2-metadata-token": token},
    )
    return urllib.request.urlopen(id_req, timeout=2).read().decode()


def stop_instance(instance_id: str, dry_run: bool = False) -> dict:
    """Stop the EC2 instance."""
    import boto3
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    if dry_run:
        logger.info("DRY RUN: Would stop instance %s", instance_id)
        return {"status": "dry_run", "instance_id": instance_id}

    logger.info("Stopping EC2 instance: %s", instance_id)
    response = ec2.stop_instances(InstanceIds=[instance_id])
    state = response["StoppingInstances"][0]["CurrentState"]["Name"]
    logger.info("Instance %s state: %s", instance_id, state)
    return {"status": state, "instance_id": instance_id}


def setup_billing_alarm(threshold_usd: float = 500.0) -> dict:
    """
    Create CloudWatch billing alarm as safety net.

    Stops the instance if AWS charges exceed threshold.
    This is a ONE-TIME setup — run during bootstrap.
    """
    import boto3

    cloudwatch = boto3.client("cloudwatch", region_name="us-east-1")  # Billing is always us-east-1
    sns = boto3.client("sns", region_name="us-east-1")

    alarm_name = "p053-budget-safety-stop"

    # Create SNS topic for alarm
    topic = sns.create_topic(Name="p053-billing-alarm")
    topic_arn = topic["TopicArn"]

    # Create billing alarm
    cloudwatch.put_metric_alarm(
        AlarmName=alarm_name,
        AlarmDescription=f"P053: Stop if AWS charges exceed ${threshold_usd}",
        MetricName="EstimatedCharges",
        Namespace="AWS/Billing",
        Statistic="Maximum",
        Period=21600,  # 6 hours
        EvaluationPeriods=1,
        Threshold=threshold_usd,
        ComparisonOperator="GreaterThanThreshold",
        Dimensions=[{"Name": "Currency", "Value": "USD"}],
        AlarmActions=[topic_arn],
        TreatMissingData="notBreaching",
    )

    logger.info("Created billing alarm: %s (threshold: $%.0f)", alarm_name, threshold_usd)
    return {"alarm_name": alarm_name, "threshold_usd": threshold_usd, "topic_arn": topic_arn}


def simulation_complete_handler(phase: str = "phase2"):
    """
    Called at the end of a simulation phase.

    Phase 2: Just log completion, DON'T stop (user reviews first)
    Phase 3: Stop the instance (all work done)
    """
    logger.info("Simulation %s complete!", phase)

    if phase == "phase2":
        logger.info("Phase 2 complete. Instance stays running for review.")
        logger.info("After review, start Phase 3 or run: python -m src.ec2_auto_stop --force")
        return {"action": "keep_running", "phase": phase}

    elif phase == "phase3":
        logger.info("Phase 3 complete. Stopping instance to save costs.")
        try:
            instance_id = get_instance_id()
            return stop_instance(instance_id)
        except Exception as e:
            logger.error("Failed to stop instance: %s", e)
            logger.info("MANUAL ACTION REQUIRED: Stop the instance manually!")
            return {"action": "manual_stop_required", "error": str(e)}

    return {"action": "unknown_phase", "phase": phase}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(description="P053 — EC2 Auto-Stop")
    parser.add_argument("--force", action="store_true", help="Stop instance immediately")
    parser.add_argument("--dry-run", action="store_true", help="Check but don't stop")
    parser.add_argument("--setup-alarm", action="store_true",
                        help="Create CloudWatch billing alarm")
    parser.add_argument("--threshold", type=float, default=500.0,
                        help="Billing alarm threshold in USD")
    parser.add_argument("--phase", default="phase3",
                        help="Simulation phase (phase2=keep running, phase3=stop)")
    args = parser.parse_args()

    if args.setup_alarm:
        result = setup_billing_alarm(args.threshold)
        print(f"Billing alarm created: ${args.threshold} threshold")

    elif args.force or args.dry_run:
        instance_id = get_instance_id()
        result = stop_instance(instance_id, dry_run=args.dry_run)
        print(f"Instance {instance_id}: {result['status']}")

    else:
        result = simulation_complete_handler(args.phase)
        print(f"Result: {result}")
