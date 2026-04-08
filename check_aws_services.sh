#!/bin/bash
# P053 — AWS Services Status Check
# Run this anytime to see what's running (and being billed)
# Usage: bash check_aws_services.sh

REGION="us-west-2"
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         P053 — AWS SERVICES STATUS CHECK                   ║"
echo "║         Region: $REGION                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ─── EC2 INSTANCES ────────────────────────────────────────────────
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│ EC2 INSTANCES                                               │"
echo "└─────────────────────────────────────────────────────────────┘"
EC2_OUT=$(aws ec2 describe-instances \
  --region "$REGION" \
  --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,State.Name,Tags[?Key==`Name`].Value|[0],PublicIpAddress]' \
  --output table 2>/dev/null)
if [ -z "$EC2_OUT" ]; then
  echo "  (none found)"
else
  echo "$EC2_OUT"
fi

echo ""
echo "  STATUS LEGEND:"
echo "  running   → BILLING at instance rate (COSTS MONEY)"
echo "  stopped   → No compute cost (EBS still charged ~0.10 USD/GB/month)"
echo "  terminated → Fully deleted"
echo ""
echo "  STOP COMMAND (replace INSTANCE_ID):"
echo "  aws ec2 stop-instances --instance-ids INSTANCE_ID --region $REGION"
echo ""

# ─── RDS INSTANCES ────────────────────────────────────────────────
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│ RDS INSTANCES                                               │"
echo "└─────────────────────────────────────────────────────────────┘"
RDS_OUT=$(aws rds describe-db-instances \
  --region "$REGION" \
  --query 'DBInstances[*].[DBInstanceIdentifier,DBInstanceClass,DBInstanceStatus,Engine,Endpoint.Address]' \
  --output table 2>/dev/null)
if [ -z "$RDS_OUT" ]; then
  echo "  (none found)"
else
  echo "$RDS_OUT"
fi

echo ""
echo "  STATUS LEGEND:"
echo "  available → BILLING at instance rate. p053-mlflow-db = \$0.018/hr"
echo "  stopped   → No compute cost (storage still ~\$0.115/GB/mo)"
echo "  ⚠️  AWS auto-restarts stopped RDS after 7 days!"
echo ""
echo "  STOP COMMAND:"
echo "  aws rds stop-db-instance --db-instance-identifier p053-mlflow-db --region $REGION"
echo ""
echo "  START COMMAND:"
echo "  aws rds start-db-instance --db-instance-identifier p053-mlflow-db --region $REGION"
echo ""

# ─── NAT GATEWAYS ─────────────────────────────────────────────────
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│ NAT GATEWAYS                                                │"
echo "└─────────────────────────────────────────────────────────────┘"
NAT_OUT=$(aws ec2 describe-nat-gateways \
  --region "$REGION" \
  --query 'NatGateways[*].[NatGatewayId,State,VpcId,SubnetId]' \
  --output table 2>/dev/null)
if [ -z "$NAT_OUT" ]; then
  echo "  (none found)"
else
  echo "$NAT_OUT"
fi

echo ""
echo "  STATUS LEGEND:"
echo "  available → BILLING \$0.045/hr + \$0.045/GB data processed"
echo "  deleted   → No cost"
echo ""
echo "  DELETE COMMAND (replace NAT_ID):"
echo "  aws ec2 delete-nat-gateway --nat-gateway-id NAT_ID --region $REGION"
echo ""

# ─── ELASTIC IPs ──────────────────────────────────────────────────
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│ ELASTIC IPs                                                 │"
echo "└─────────────────────────────────────────────────────────────┘"
EIP_JSON=$(aws ec2 describe-addresses --region "$REGION" --output json 2>/dev/null)
if [ -z "$EIP_JSON" ]; then
  echo "  (none allocated)"
else
  echo "$EIP_JSON" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for a in data.get('Addresses', []):
    svc = a.get('ServiceManaged', '')
    aid = a.get('AllocationId', '')
    ip  = a.get('PublicIp', '')
    asoc = a.get('AssociationId', 'None')
    iid = a.get('InstanceId', 'None')
    if svc:
        print(f'  [AUTO-MANAGED by {svc.upper()}] {ip} ({aid}) — do NOT release manually, AWS owns this')
    elif asoc == 'None':
        print(f'  [WARNING - BILLING 0.005 USD/hr] {ip} ({aid}) — unassociated, costs money!')
    else:
        print(f'  [OK - associated] {ip} ({aid}) — attached to {iid or "network interface"}')
"
fi

echo ""
echo "  NOTE: EIPs marked [AUTO-MANAGED] belong to RDS/NAT — do NOT release them."
echo "  Only release EIPs you created manually that show [WARNING]."
echo ""

# ─── S3 BUCKETS ───────────────────────────────────────────────────
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│ S3 BUCKETS                                                  │"
echo "└─────────────────────────────────────────────────────────────┘"
aws s3 ls --region "$REGION" 2>/dev/null || echo "  (none found or no permission)"
echo ""
echo "  COST: \$0.023/GB/month for standard storage"
echo "  CHECK SIZE:"
echo "  aws s3api list-objects-v2 --bucket p053-mlflow-artifacts --query 'sum(Contents[].Size)'"
echo ""

# ─── CloudWatch ALARMS ────────────────────────────────────────────
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│ CLOUDWATCH BILLING ALARMS                                   │"
echo "└─────────────────────────────────────────────────────────────┘"
aws cloudwatch describe-alarms \
  --region "us-east-1" \
  --alarm-types MetricAlarm \
  --query 'MetricAlarms[*].[AlarmName,StateValue,Threshold,ComparisonOperator]' \
  --output table 2>/dev/null || echo "  (Billing alarms in us-east-1 — check AWS console)"
echo ""

# ─── COST SUMMARY ─────────────────────────────────────────────────
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│ CURRENT MONTH SPEND (approximate)                           │"
echo "└─────────────────────────────────────────────────────────────┘"
aws ce get-cost-and-usage \
  --region "us-east-1" \
  --time-period Start="$(date +%Y-%m-01)",End="$(date +%Y-%m-%d)" \
  --granularity MONTHLY \
  --metrics "UnblendedCost" \
  --query 'ResultsByTime[0].Total.UnblendedCost' \
  --output table 2>/dev/null || echo "  (Cost Explorer may take 24hrs to show data. Check AWS Console → Billing)"
echo ""

# ─── SUMMARY ──────────────────────────────────────────────────────
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  QUICK STOP-EVERYTHING COMMANDS (for P053):                ║"
echo "║                                                            ║"
echo "║  RDS:  aws rds stop-db-instance \\                         ║"
echo "║          --db-instance-identifier p053-mlflow-db          ║"
echo "║                                                            ║"
echo "║  EC2:  aws ec2 stop-instances \\                           ║"
echo "║          --instance-ids <INSTANCE_ID>                     ║"
echo "║                                                            ║"
echo "║  CHECK AGAIN: bash check_aws_services.sh                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
