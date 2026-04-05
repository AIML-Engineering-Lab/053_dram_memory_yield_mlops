#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# P053 — AWS Infrastructure Setup
# ═══════════════════════════════════════════════════════════════
# Provisions all AWS resources needed for production deployment.
#
# Prerequisites:
#   - AWS CLI configured: aws configure
#   - Sufficient IAM permissions (RDS, S3, ECR, EC2, SageMaker)
#
# Usage:
#   chmod +x deploy/aws/setup_aws.sh
#   ./deploy/aws/setup_aws.sh
#
# Cost estimate (pay-as-you-go):
#   RDS db.t3.micro:    ~$15/month
#   S3 (< 5 GB):        ~$0.12/month
#   EC2 t3.xlarge:      ~$0.17/hr (~$120/month if always-on)
#   Total (always-on):  ~$135/month
#   Total (8 hrs/day):  ~$55/month
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-us-west-2}"
PROJECT="p053"
S3_BUCKET="${PROJECT}-mlflow-artifacts"
RDS_INSTANCE="${PROJECT}-mlflow"
ECR_REPO="053-memory-yield-predictor"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "=== P053 AWS Setup ==="
echo "Region:     $REGION"
echo "Account:    $ACCOUNT_ID"
echo ""

# ─── 1. S3 Bucket for MLflow Artifacts ───
echo "[1/4] Creating S3 bucket: $S3_BUCKET"
if aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null; then
    echo "  → Bucket already exists"
else
    aws s3 mb "s3://$S3_BUCKET" --region "$REGION"
    aws s3api put-bucket-versioning \
        --bucket "$S3_BUCKET" \
        --versioning-configuration Status=Enabled
    echo "  → Created with versioning enabled"
fi

# ─── 2. RDS PostgreSQL for MLflow Backend ───
echo "[2/4] Creating RDS instance: $RDS_INSTANCE"
if aws rds describe-db-instances --db-instance-identifier "$RDS_INSTANCE" 2>/dev/null; then
    echo "  → RDS instance already exists"
else
    echo "  Enter MLflow DB password (min 8 chars):"
    read -s MLFLOW_DB_PASSWORD
    aws rds create-db-instance \
        --db-instance-identifier "$RDS_INSTANCE" \
        --db-instance-class db.t3.micro \
        --engine postgres \
        --engine-version 16.4 \
        --master-username mlflow \
        --master-user-password "$MLFLOW_DB_PASSWORD" \
        --allocated-storage 20 \
        --storage-type gp3 \
        --db-name mlflow \
        --backup-retention-period 7 \
        --no-multi-az \
        --publicly-accessible \
        --region "$REGION"
    echo "  → Creating RDS (takes 5-10 min)..."
    aws rds wait db-instance-available --db-instance-identifier "$RDS_INSTANCE"
    
    ENDPOINT=$(aws rds describe-db-instances \
        --db-instance-identifier "$RDS_INSTANCE" \
        --query 'DBInstances[0].Endpoint.Address' \
        --output text)
    echo "  → RDS endpoint: $ENDPOINT"
fi

# ─── 3. ECR Repository ───
echo "[3/4] Creating ECR repository: $ECR_REPO"
if aws ecr describe-repositories --repository-names "$ECR_REPO" 2>/dev/null; then
    echo "  → ECR repo already exists"
else
    aws ecr create-repository \
        --repository-name "$ECR_REPO" \
        --image-scanning-configuration scanOnPush=true \
        --region "$REGION"
    echo "  → Created with scan-on-push"
fi

# ─── 4. Summary ───
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Fill in deploy/aws/.env with:"
ENDPOINT=$(aws rds describe-db-instances \
    --db-instance-identifier "$RDS_INSTANCE" \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text 2>/dev/null || echo "<pending>")
ECR_URI=$(aws ecr describe-repositories \
    --repository-names "$ECR_REPO" \
    --query 'repositories[0].repositoryUri' \
    --output text 2>/dev/null || echo "<pending>")
echo "  MLFLOW_DB_HOST=$ENDPOINT"
echo "  S3_BUCKET=$S3_BUCKET"
echo "  ECR_REPO=$ECR_URI"
echo ""
echo "Then deploy:"
echo "  docker compose -f deploy/aws/docker-compose-aws.yml up -d"
