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


def setup_billing_alarm(threshold_usd: float = 200.0) -> dict:
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


def stop_rds_instance(db_instance_id: str = "p053-mlflow-db", dry_run: bool = False) -> dict:
    """Stop the RDS instance to save costs between simulation runs."""
    import boto3
    rds = boto3.client("rds", region_name=AWS_REGION)

    if dry_run:
        logger.info("DRY RUN: Would stop RDS instance %s", db_instance_id)
        return {"status": "dry_run", "db_instance_id": db_instance_id}

    logger.info("Stopping RDS instance: %s", db_instance_id)
    try:
        rds.stop_db_instance(DBInstanceIdentifier=db_instance_id)
        logger.info("RDS instance %s stop initiated", db_instance_id)
        return {"status": "stopping", "db_instance_id": db_instance_id}
    except rds.exceptions.InvalidDBInstanceStateFault:
        logger.info("RDS instance %s already stopped or stopping", db_instance_id)
        return {"status": "already_stopped", "db_instance_id": db_instance_id}


def delete_nat_gateway(dry_run: bool = False) -> dict:
    """Delete NAT gateway to stop $0.045/hr idle charges."""
    import boto3
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    # Find NAT gateways tagged with p053
    response = ec2.describe_nat_gateways(
        Filters=[{"Name": "tag:Project", "Values": ["p053"]}]
    )
    gateways = [g for g in response["NatGateways"] if g["State"] in ("available",)]

    if not gateways:
        logger.info("No active p053 NAT gateways found")
        return {"status": "none_found"}

    results = []
    for gw in gateways:
        gw_id = gw["NatGatewayId"]
        if dry_run:
            logger.info("DRY RUN: Would delete NAT gateway %s", gw_id)
            results.append({"id": gw_id, "status": "dry_run"})
        else:
            ec2.delete_nat_gateway(NatGatewayId=gw_id)
            logger.info("Deleted NAT gateway: %s", gw_id)
            results.append({"id": gw_id, "status": "deleting"})

    return {"deleted": results}


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
        logger.info("Phase 3 complete. Stopping ALL resources to save costs.")
        results = {}

        # Stop RDS first (data preserved, restartable)
        try:
            results["rds"] = stop_rds_instance()
        except Exception as e:
            logger.error("Failed to stop RDS: %s", e)
            results["rds"] = {"status": "error", "error": str(e)}

        # Delete NAT gateway ($0.045/hr idle waste)
        try:
            results["nat_gateway"] = delete_nat_gateway()
        except Exception as e:
            logger.error("Failed to delete NAT gateway: %s", e)
            results["nat_gateway"] = {"status": "error", "error": str(e)}

        # Stop EC2 instance last (this will terminate our session)
        try:
            instance_id = get_instance_id()
            results["ec2"] = stop_instance(instance_id)
        except Exception as e:
            logger.error("Failed to stop instance: %s", e)
            logger.info("MANUAL ACTION REQUIRED: Stop the instance manually!")
            results["ec2"] = {"action": "manual_stop_required", "error": str(e)}

        return results

    return {"action": "unknown_phase", "phase": phase}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(description="P053 — EC2 Auto-Stop")
    parser.add_argument("--force", action="store_true", help="Stop instance immediately")
    parser.add_argument("--dry-run", action="store_true", help="Check but don't stop")
    parser.add_argument("--setup-alarm", action="store_true",
                        help="Create CloudWatch billing alarm")
    parser.add_argument("--threshold", type=float, default=200.0,
                        help="Billing alarm threshold in USD (default: $200)")
    parser.add_argument("--phase", default="phase3",
                        help="Simulation phase (phase2=keep running, phase3=stop all)")
    parser.add_argument("--stop-rds", action="store_true",
                        help="Stop RDS instance only")
    parser.add_argument("--delete-nat", action="store_true",
                        help="Delete NAT gateway only")
    args = parser.parse_args()

    if args.setup_alarm:
        result = setup_billing_alarm(args.threshold)
        print(f"Billing alarm created: ${args.threshold} threshold")

    elif args.stop_rds:
        result = stop_rds_instance(dry_run=args.dry_run)
        print(f"RDS: {result['status']}")

    elif args.delete_nat:
        result = delete_nat_gateway(dry_run=args.dry_run)
        print(f"NAT Gateway: {result}")

    elif args.force or args.dry_run:
        instance_id = get_instance_id()
        result = stop_instance(instance_id, dry_run=args.dry_run)
        print(f"Instance {instance_id}: {result['status']}")

    else:
        result = simulation_complete_handler(args.phase)
        print(f"Result: {result}")
