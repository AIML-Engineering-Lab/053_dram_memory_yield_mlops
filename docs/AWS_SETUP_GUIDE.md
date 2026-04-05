# P053 — Complete AWS Setup Guide (First-Time AWS Users)

> **Budget:** ~$7-10 total for the 40-day simulation (no free tier)  
> **Time to setup:** ~30 minutes  
> **Region:** us-west-2 (Oregon) — cheapest for EC2 + S3 (latency irrelevant for batch simulation)  
> **Instance:** t3.medium ($0.0416/hr) for daily pipeline, g4dn.xlarge ($0.526/hr) for retrain only

---

## Table of Contents

1. [Create AWS Account (if not done)](#1-create-aws-account)
2. [Secure the Root Account (CRITICAL)](#2-secure-the-root-account)
3. [Create IAM User for CLI/CI-CD](#3-create-iam-user)
4. [Install AWS CLI on Mac](#4-install-aws-cli)
5. [Create S3 Bucket for Artifacts](#5-create-s3-bucket)
6. [Create ECR Repository for Docker Images](#6-create-ecr-repository)
7. [Launch EC2 Instance for 40-Day Simulation](#7-launch-ec2-instance)
8. [Set Up RDS PostgreSQL for MLflow](#8-set-up-rds-postgresql)
9. [Configure GitHub Actions Secrets](#9-configure-github-actions-secrets)
10. [Verify Everything Works](#10-verify-everything-works)
11. [Cost Control — Set Budget Alerts](#11-cost-control)
12. [Architecture Diagram](#12-architecture-diagram)

---

## 1. Create AWS Account

If you already have an account, skip to Step 2.

1. Go to https://aws.amazon.com/ → Click **"Create an AWS Account"**
2. Enter email, password, account name (e.g., `AIML-Engineering-Lab`)
3. Choose **Personal** account type
4. Enter payment method (credit/debit card — you won't be charged yet)
5. Verify phone number
6. Select **Basic Support** (free tier)
7. Sign in to Console: https://console.aws.amazon.com/

> **Free Tier Note:** If your account is >12 months old, free tier has expired. Main impact: RDS db.t3.micro costs ~$0.018/hr instead of free. We minimize this by stopping RDS when not in use. See Cost Control section for details.

---

## 2. Secure the Root Account (CRITICAL)

The root account has unlimited access. Secure it immediately.

### 2.1 Enable MFA on Root

1. Top-right corner → Click your account name → **Security credentials**
2. Under **Multi-factor authentication (MFA)** → Click **Assign MFA device**
3. Choose **Authenticator app** → Scan QR code with Google Authenticator / Authy
4. Enter two consecutive codes → Click **Add MFA**

### 2.2 Create Billing Alert

1. Search bar → Type **"Billing"** → Click **Billing and Cost Management**
2. Left sidebar → **Budgets** → **Create budget**
3. Choose **Zero spend budget** (alerts when ANY charge occurs)
4. Enter email → Create budget
5. Create another: **Monthly cost budget** → Set to **$50**

---

## 3. Create IAM User

**Never use root for CLI/CI-CD.** Create a dedicated IAM user.

### Step-by-Step (with screenshots context)

1. Search bar → Type **"IAM"** → Click **IAM**

2. Left sidebar → **Users** → **Create user**

3. **User details:**
   - User name: `p053-cicd-user`
   - ☑️ Check **"Provide user access to the AWS Management Console"** (optional — for debugging)
   - Select **"I want to create an IAM user"**
   - Auto-generated password, require reset → **Next**

4. **Set permissions:**
   - Select **"Attach policies directly"**
   - Search and check these 5 policies:
     - `AmazonEC2FullAccess`
     - `AmazonS3FullAccess`
     - `AmazonRDSFullAccess`
     - `AmazonEC2ContainerRegistryFullAccess`
     - `IAMReadOnlyAccess`
   - Click **Next** → **Create user**

5. **Create Access Key (for CLI + GitHub Actions):**
   - Click on the user name `p053-cicd-user`
   - Tab: **Security credentials**
   - Scroll to **Access keys** → **Create access key**
   - Use case: **Command Line Interface (CLI)**
   - ⚠️ AWS may recommend alternatives (SSO, CloudShell) — **ignore these**. We need access keys for GitHub Actions CI/CD. Click the acknowledgment checkbox.
   - ☑️ Check the confirmation box → **Next**
   - Description tag: `p053-local-and-cicd`
   - **IMPORTANT:** Copy both values NOW. You see the Secret only ONCE:

   ```
   Access Key ID:     AKIA...............
   Secret Access Key: wJal...............
   ```

   - Save these securely (password manager, NOT in plain text files)

---

## 4. Install AWS CLI

Open Terminal on your Mac:

```bash
# Install via Homebrew (Apple Silicon Mac — MUST use arch -arm64)
arch -arm64 brew install awscli

# Verify installation
aws --version
# Should show: aws-cli/2.x.x Python/3.x.x Darwin/...

# Configure with your IAM user credentials
aws configure
```

When prompted:
```
AWS Access Key ID [None]: AKIA............... (paste your key)
AWS Secret Access Key [None]: wJal............... (paste your secret)
Default region name [None]: us-west-2
Default output format [None]: json
```

Verify it works:
```bash
# Should return your IAM user info
aws sts get-caller-identity
```

Expected output:
```json
{
    "UserId": "AIDA...",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/p053-cicd-user"
}
```

**Save your Account ID** (the 12-digit number) — you'll need it for ECR.

---

## 5. Create S3 Bucket

S3 stores: MLflow artifacts, model weights, DVC-tracked data, simulation results.

```bash
# Create bucket (names must be globally unique)
aws s3 mb s3://p053-mlflow-artifacts --region us-west-2

# Enable versioning (protects against accidental deletes)
aws s3api put-bucket-versioning \
  --bucket p053-mlflow-artifacts \
  --versioning-configuration Status=Enabled

# Verify
aws s3 ls
```

Expected output:
```
2026-04-05 10:00:00 p053-mlflow-artifacts
```

**Cost:** $0.023/GB/month for S3 Standard. Our total data: ~5 GB → ~$0.12/month.

---

## 6. Create ECR Repository

ECR (Elastic Container Registry) stores our Docker images. GitHub CI/CD pushes here.

```bash
# Create repository
aws ecr create-repository \
  --repository-name 053-memory-yield-predictor \
  --region us-west-2 \
  --image-scanning-configuration scanOnPush=true

# Get your registry URL (save this)
aws ecr describe-repositories --repository-names 053-memory-yield-predictor \
  --query 'repositories[0].repositoryUri' --output text
```

Expected output:
```
123456789012.dkr.ecr.us-west-2.amazonaws.com/053-memory-yield-predictor
```

**Save this URI** — we'll need it for docker-compose and CI/CD.

**Cost:** 500 MB free storage/month. Our image is ~450 MB → free.

---

## 7. Launch EC2 Instance

This is where the 40-day simulation runs. We use **t3.medium** (2 vCPU, 4 GB RAM, $0.0416/hr ≈ $1/day).

### 7.1 Create a Key Pair (for SSH access)

```bash
# Create key pair and save the .pem file
aws ec2 create-key-pair \
  --key-name p053-key \
  --key-type rsa \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/p053-key.pem

# Set correct permissions (SSH requires this)
chmod 400 ~/.ssh/p053-key.pem
```

### 7.2 Create Security Group

```bash
# Create security group
aws ec2 create-security-group \
  --group-name p053-sg \
  --description "P053 MLOps simulation" \
  --region us-west-2

# Save the Security Group ID from the output (sg-0abc...)

# Allow SSH from your IP only
MY_IP=$(curl -s ifconfig.me)
aws ec2 authorize-security-group-ingress \
  --group-name p053-sg \
  --protocol tcp --port 22 \
  --cidr "${MY_IP}/32"

# Allow MLflow UI (port 5001) from your IP
aws ec2 authorize-security-group-ingress \
  --group-name p053-sg \
  --protocol tcp --port 5001 \
  --cidr "${MY_IP}/32"

# Allow Grafana UI (port 3000) from your IP
aws ec2 authorize-security-group-ingress \
  --group-name p053-sg \
  --protocol tcp --port 3000 \
  --cidr "${MY_IP}/32"

# Allow API (port 8000) from your IP
aws ec2 authorize-security-group-ingress \
  --group-name p053-sg \
  --protocol tcp --port 8000 \
  --cidr "${MY_IP}/32"

# Allow Airflow UI (port 8080)
aws ec2 authorize-security-group-ingress \
  --group-name p053-sg \
  --protocol tcp --port 8080 \
  --cidr "${MY_IP}/32"

# Allow Kafdrop (Kafka UI, port 9000)
aws ec2 authorize-security-group-ingress \
  --group-name p053-sg \
  --protocol tcp --port 9000 \
  --cidr "${MY_IP}/32"
```

### 7.3 Launch Instance

```bash
# Get latest Amazon Linux 2023 AMI ID
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=al2023-ami-2023*-x86_64" \
            "Name=state,Values=available" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text \
  --region us-west-2)

echo "Using AMI: $AMI_ID"

# Launch t3.medium (2 vCPU, 4 GB RAM)
# 30 GB disk for Docker images + data
aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type t3.medium \
  --key-name p053-key \
  --security-groups p053-sg \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=p053-simulation},{Key=Project,Value=P053}]' \
  --region us-west-2 \
  --count 1
```

### 7.4 Get Instance Details and Connect

```bash
# Get public IP (wait ~30 seconds for instance to start)
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=p053-simulation" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text

# SSH into the instance
ssh -i ~/.ssh/p053-key.pem ec2-user@<PUBLIC_IP>
```

### 7.5 Setup Docker on EC2

Run these commands AFTER SSH-ing into the instance:

```bash
# Install Docker
sudo dnf update -y
sudo dnf install -y docker git
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Log out and back in (for docker group)
exit
```

SSH back in, then:

```bash
# Verify Docker works
docker --version
docker-compose --version

# Login to ECR
aws configure  # Enter same credentials as local
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com

# Clone the repository
git clone https://github.com/AIML-Engineering-Lab/053_dram_memory_yield_mlops.git
cd 053_dram_memory_yield_mlops
```

### 7.6 Cost Control — Stop When Not Using

```bash
# STOP the instance (from your Mac, not from inside EC2)
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=p053-simulation" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

aws ec2 stop-instances --instance-ids "$INSTANCE_ID"

# START when you need it again
aws ec2 start-instances --instance-ids "$INSTANCE_ID"

# Note: Public IP changes after stop/start. Get new IP:
aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

**Cost:** t3.medium stopped = $0 (only EBS storage: $2.40/month for 30 GB gp3). Running = $0.0416/hr ≈ $1/day.

---

## 8. Set Up RDS PostgreSQL for MLflow

RDS provides a managed PostgreSQL database — no maintenance needed.

```bash
# Create DB subnet group (uses default VPC subnets)
# First, get your VPC and subnet IDs
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" \
  --query 'Vpcs[0].VpcId' --output text --region us-west-2)

SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" \
  --query 'Subnets[*].SubnetId' --output text --region us-west-2)

# Create subnet group
aws rds create-db-subnet-group \
  --db-subnet-group-name p053-db-subnet \
  --db-subnet-group-description "P053 MLflow DB subnets" \
  --subnet-ids $SUBNET_IDS \
  --region us-west-2

# Get EC2 security group ID
SG_ID=$(aws ec2 describe-security-groups --group-names p053-sg \
  --query 'SecurityGroups[0].GroupId' --output text --region us-west-2)

# Create RDS instance (db.t3.micro = free tier eligible)
aws rds create-db-instance \
  --db-instance-identifier p053-mlflow \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --engine-version "16.4" \
  --master-username mlflow \
  --master-user-password "MLfl0w_P053_Secure!" \
  --allocated-storage 20 \
  --storage-type gp3 \
  --vpc-security-group-ids "$SG_ID" \
  --db-subnet-group-name p053-db-subnet \
  --db-name mlflow \
  --backup-retention-period 7 \
  --no-multi-az \
  --no-publicly-accessible \
  --region us-west-2

echo "⏳ RDS takes 5-10 minutes to create..."
echo "   Check status: aws rds describe-db-instances --db-instance-identifier p053-mlflow --query 'DBInstances[0].DBInstanceStatus'"
```

Wait for status to be `available`, then get the endpoint:

```bash
# Get the RDS endpoint (hostname)
aws rds describe-db-instances \
  --db-instance-identifier p053-mlflow \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text --region us-west-2
```

Save this — it looks like: `p053-mlflow.xxxxxxxxxxxx.us-west-2.rds.amazonaws.com`

**Cost:** db.t3.micro = $0.018/hr. **Stop when not using** to save money:

```bash
# Stop RDS (from Mac) — takes ~5 min
aws rds stop-db-instance --db-instance-identifier p053-mlflow

# Start RDS when needed — takes ~5 min
aws rds start-db-instance --db-instance-identifier p053-mlflow
```

> **Note:** AWS auto-starts stopped RDS instances after 7 days. Set a calendar reminder to stop it again if you're pausing the project.

---

## 9. Configure GitHub Actions Secrets

This allows CI/CD to push Docker images to ECR and deploy.

1. Go to: https://github.com/AIML-Engineering-Lab/053_dram_memory_yield_mlops/settings/secrets/actions

2. Click **"New repository secret"** for each:

| Secret Name | Value | Where to find it |
|-------------|-------|-------------------|
| `AWS_ACCESS_KEY_ID` | `AKIA...` | Step 3 (IAM User) |
| `AWS_SECRET_ACCESS_KEY` | `wJal...` | Step 3 (IAM User) |
| `AWS_REGION` | `us-west-2` | We chose Oregon |
| `AWS_ACCOUNT_ID` | `123456789012` | Step 4 (`aws sts get-caller-identity`) |
| `RDS_ENDPOINT` | `p053-mlflow.xxx.us-west-2.rds.amazonaws.com` | Step 8 |
| `RDS_PASSWORD` | `MLfl0w_P053_Secure!` | Step 8 (the password you set) |

3. Verify secrets are listed (you can see names, not values)

---

## 10. Verify Everything Works

### From your Mac:

```bash
# S3 — upload a test file
echo "test" | aws s3 cp - s3://p053-mlflow-artifacts/test.txt
aws s3 ls s3://p053-mlflow-artifacts/
aws s3 rm s3://p053-mlflow-artifacts/test.txt

# ECR — check repo exists
aws ecr describe-repositories --repository-names 053-memory-yield-predictor

# EC2 — check instance
aws ec2 describe-instances --filters "Name=tag:Name,Values=p053-simulation" \
  --query 'Reservations[0].Instances[0].[InstanceId,State.Name,PublicIpAddress]' --output table

# RDS — check database
aws rds describe-db-instances --db-instance-identifier p053-mlflow \
  --query 'DBInstances[0].[DBInstanceStatus,Endpoint.Address]' --output table
```

### From EC2 (SSH in):

```bash
# Test RDS connection
sudo dnf install -y postgresql16
psql -h <RDS_ENDPOINT> -U mlflow -d mlflow -c "SELECT version();"
# Enter password: MLfl0w_P053_Secure!

# Test S3 access
aws s3 ls s3://p053-mlflow-artifacts/
```

---

## 11. Cost Control

### Set a Hard Budget Alarm

1. AWS Console → **Billing** → **Budgets**
2. Create budget: **Monthly cost budget** → Amount: **$50**
3. Alert at 80% ($40) and 100% ($50)
4. Add your email for notifications

### Expected Costs for 40-Day Simulation

| Resource | Usage | Cost |
|----------|-------|------|
| EC2 t3.medium | ~80 hrs (2 hrs/day × 40 days) | $3.33 |
| EC2 g4dn.xlarge | ~4 hrs (retrain × 3 sessions) | $2.10 |
| RDS db.t3.micro | ~80 hrs (stop when not using) | $1.44 |
| S3 (5 GB) | Storage + requests | $0.15 |
| ECR (500 MB) | Storage | $0.05 |
| Data transfer | Minimal (within region) | $0.10 |
| **Total** | | **~$7.17** |

### Emergency: Delete Everything

If costs spike, run this from your Mac:

```bash
# Stop EC2
aws ec2 terminate-instances --instance-ids $(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=p053-simulation" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)

# Delete RDS (skip final snapshot to avoid charges)
aws rds delete-db-instance \
  --db-instance-identifier p053-mlflow \
  --skip-final-snapshot

# Empty and delete S3 bucket
aws s3 rb s3://p053-mlflow-artifacts --force

# Delete ECR repo
aws ecr delete-repository \
  --repository-name 053-memory-yield-predictor --force
```

---

## 12. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       YOUR MAC (Local Development)                       │
│                                                                          │
│  ┌───────────────┐   ┌──────────────┐   ┌──────────────────────────┐   │
│  │ src/ code      │   │ Docker Stack │   │ aws configure            │   │
│  │ notebooks/     │──→│ (6 services) │   │ (IAM credentials)        │   │
│  │ tests/         │   │ localhost    │   └──────────┬───────────────┘   │
│  └───────┬───────┘   └──────────────┘              │                    │
│          │                                          │                    │
│          │ git push                                 │ aws cli            │
│          ▼                                          ▼                    │
└──────────┼──────────────────────────────────────────┼────────────────────┘
           │                                          │
┌──────────▼──────────┐                   ┌───────────▼───────────────────┐
│   GitHub Actions     │                   │        AWS Cloud               │
│                      │                   │                                │
│  ┌────────────────┐  │   Docker push     │  ┌─────────────────────────┐  │
│  │ test → build   │──┼──────────────────→│  │ ECR (Docker images)     │  │
│  │ (on every push)│  │                   │  └─────────────────────────┘  │
│  └────────────────┘  │                   │                                │
│  ┌────────────────┐  │   kubectl apply   │  ┌─────────────────────────┐  │
│  │ deploy         │──┼──────────────────→│  │ EC2 t3.medium           │  │
│  │ (on git tag)   │  │                   │  │ ┌──── docker-compose ──┐│  │
│  └────────────────┘  │                   │  │ │ Kafka+Spark+Airflow ││  │
│                      │                   │  │ │ API+MLflow+Grafana  ││  │
└──────────────────────┘                   │  │ └─────────────────────┘│  │
                                           │  └───────────┬─────────────┘  │
┌──────────────────────┐                   │              │                │
│   Google Colab A100   │                   │  ┌───────────▼─────────────┐  │
│                       │   model.pt → S3   │  │ S3 Bucket               │  │
│  NB03: Day 1 Training │──────────────────→│  │ - Model weights         │  │
│  (one-time only)      │                   │  │ - MLflow artifacts      │  │
│                       │                   │  │ - Simulation results    │  │
└───────────────────────┘                   │  └─────────────────────────┘  │
                                           │                                │
                                           │  ┌─────────────────────────┐  │
                                           │  │ RDS PostgreSQL          │  │
                                           │  │ MLflow tracking DB      │  │
                                           │  └─────────────────────────┘  │
                                           └────────────────────────────────┘
```

### 40-Day Flow on AWS

```
Day 1:  Colab A100 trains → model v1 → S3 → EC2 deploys → starts Airflow
Day 2-19: Daily: Airflow → generate 5M rows → Kafka → Spark ETL → inference → drift check ✅
Day 20: Drift detected (PSI > 0.10) → retrain DAG → model v2 → canary test → promote ✅
Day 21-30: Daily inference with v2 champion
Day 31: Severe drift (PSI > 0.20) → retrain DAG → model v3 → canary test → promote ✅
Day 32-38: Daily inference with v3
Day 39: Bad deploy simulation → canary FAILS → rollback to v3 → from-scratch retrain → v4
Day 40: Final day — all metrics logged, simulation complete
```

---

## What You Need to Give Me

After completing Steps 1-8, share these values so I can configure the project:

```
AWS_ACCOUNT_ID:    _____________ (12 digits)
AWS_REGION:        us-west-2
S3_BUCKET:         p053-mlflow-artifacts
ECR_REPO_URI:      _____________.dkr.ecr.us-west-2.amazonaws.com/053-memory-yield-predictor
RDS_ENDPOINT:      p053-mlflow._____________.us-west-2.rds.amazonaws.com
EC2_PUBLIC_IP:     _____________
```

I'll use these to:
1. Configure `.env.aws` for the EC2 stack
2. Update CI/CD to push to your ECR
3. Set up the Airflow DAGs with correct endpoints
4. Configure MLflow to use RDS + S3

---

*Created: April 5, 2026 — Step-by-step guide for first-time AWS users*
