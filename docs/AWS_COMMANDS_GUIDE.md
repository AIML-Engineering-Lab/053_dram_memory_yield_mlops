# AWS Commands Execution Guide — P053

> **Last Updated:** 2026-04-07  
> **Account:** 718036735422 · Region: us-west-2 (Oregon)  
> **Status:** Steps 1–8 complete. RDS created (stopped). EC2 g4dn.xlarge **blocked on GPU quota** (0→4 vCPUs).  
> **Instance:** g4dn.xlarge (4 vCPU, 16GB RAM, T4 GPU 16GB, $0.526/hr)  
> **Budget:** $1,000 SGD (~$740 USD)  
> **RDS Endpoint:** p053-mlflow-db.cxmsugggu12o.us-west-2.rds.amazonaws.com  
> **Security Group IP:** 119.234.92.99/32 (updated 2026-04-07)  
> **AMI:** ami-0e1c254c6491f13f2 (Amazon Linux 2023, kernel 6.18)

---

## Table of Contents

1. [Step 4 — AWS CLI Installation & Configuration](#step-4--aws-cli-installation--configuration)
2. [Step 5 — S3 Bucket Creation](#step-5--s3-bucket-creation)
3. [Step 6 — ECR Repository Creation](#step-6--ecr-repository-creation)
4. [Step 7.1 — EC2 Key Pair](#step-71--ec2-key-pair)
5. [Step 7.2 — Security Group & Firewall Rules](#step-72--security-group--firewall-rules)
6. [Cost Summary — What's Free, What's Not](#cost-summary)

---

## Step 4 — AWS CLI Installation & Configuration

### 4a. Install AWS CLI (macOS Apple Silicon)

```bash
arch -arm64 brew install awscli
```

| Part | Meaning |
|------|---------|
| `arch -arm64` | Forces Homebrew to run as native ARM64 (Apple Silicon). Without this, if Rosetta 2 is active, brew may fail with "Cannot install under Rosetta 2 in ARM default prefix" |
| `brew install awscli` | Installs the AWS CLI v2 package from Homebrew |

**Output:** `awscli 2.34.24` installed.  
**Alternative:** Download the `.pkg` installer from [AWS docs](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

### 4b. Verify Installation

```bash
aws --version
```

**Output:**
```
aws-cli/2.34.24 Python/3.13.3 Darwin/24.5.0 source/arm64
```

| Part | Meaning |
|------|---------|
| `aws-cli/2.34.24` | AWS CLI version installed |
| `Python/3.13.3` | Python bundled inside the CLI (separate from your system Python) |
| `Darwin/24.5.0` | macOS kernel version (Sequoia 15.5) |
| `source/arm64` | Running natively on Apple Silicon (not under Rosetta) |

### 4c. Configure Credentials

```bash
aws configure
```

**Prompts and what you entered:**

| Prompt | Value | Meaning |
|--------|-------|---------|
| `AWS Access Key ID` | `AKIA...` (hidden) | The public half of your IAM API key — like a username |
| `AWS Secret Access Key` | `****` (hidden) | The private half — like a password. **NEVER share or commit** |
| `Default region name` | `us-west-2` | All commands default to Oregon region unless overridden |
| `Default output format` | `json` | CLI responses come as JSON (alternatives: `table`, `text`, `yaml`) |

**Where it's stored:** `~/.aws/credentials` and `~/.aws/config`

### 4d. Verify Identity

```bash
aws sts get-caller-identity
```

**Output:**
```json
{
    "UserId": "AIDA6QCNLK6LUZRFR6HDI",
    "Account": "718036735422",
    "Arn": "arn:aws:iam::718036735422:user/p053-cicd-user"
}
```

| Field | Meaning |
|-------|---------|
| `UserId` | Unique internal AWS ID for this IAM user |
| `Account` | Your 12-digit AWS account number |
| `Arn` | Amazon Resource Name — the globally unique identifier. Format: `arn:aws:iam::<account>:user/<username>` |

**Why this matters:** Confirms your CLI is authenticated as the correct IAM user with the right permissions. If this fails, your access key is wrong or the user doesn't exist.

---

## Step 5 — S3 Bucket Creation

### 5a. Create Bucket

```bash
aws s3 mb s3://p053-mlflow-artifacts --region us-west-2
```

| Part | Meaning |
|------|---------|
| `aws s3 mb` | "make bucket" — creates a new S3 bucket |
| `s3://p053-mlflow-artifacts` | Bucket name. Must be **globally unique** across ALL AWS accounts worldwide |
| `--region us-west-2` | Physical location of the bucket (Oregon data center) |

**Output:** `make_bucket: p053-mlflow-artifacts`  
**Cost:** $0.023/GB/month for storage. Empty bucket = $0.

### 5b. Enable Versioning

```bash
aws s3api put-bucket-versioning \
  --bucket p053-mlflow-artifacts \
  --versioning-configuration Status=Enabled
```

| Part | Meaning |
|------|---------|
| `s3api` | Low-level S3 API (vs `s3` which is high-level convenience commands) |
| `put-bucket-versioning` | Enables version history on the bucket |
| `Status=Enabled` | Every object upload creates a new version instead of overwriting |

**Why versioning?** If we accidentally overwrite a model artifact, we can recover the previous version. Essential for MLOps — you never want to lose a trained model.

**No output** = success (S3 API returns empty on success).

### 5c. Verify Versioning

```bash
aws s3api get-bucket-versioning --bucket p053-mlflow-artifacts
```

**Output:**
```json
{
    "Status": "Enabled"
}
```

---

## Step 6 — ECR Repository Creation

```bash
aws ecr create-repository \
  --repository-name 053-memory-yield-predictor \
  --region us-west-2
```

| Part | Meaning |
|------|---------|
| `ecr create-repository` | Creates a private Docker image registry (like a private Docker Hub) |
| `--repository-name` | Name of the repo. Our Docker images get pushed here |

**Output:**
```json
{
    "repository": {
        "repositoryArn": "arn:aws:ecr:us-west-2:718036735422:repository/053-memory-yield-predictor",
        "registryId": "718036735422",
        "repositoryName": "053-memory-yield-predictor",
        "repositoryUri": "718036735422.dkr.ecr.us-west-2.amazonaws.com/053-memory-yield-predictor",
        "createdAt": "2025-06-28T...",
        "imageTagMutability": "MUTABLE",
        "imageScanningConfiguration": { "scanOnPush": false },
        "encryptionConfiguration": { "encryptionType": "AES256" }
    }
}
```

| Field | Meaning |
|-------|---------|
| `repositoryUri` | The full address to push Docker images to. Format: `<account>.dkr.ecr.<region>.amazonaws.com/<name>` |
| `imageTagMutability: MUTABLE` | You can overwrite the `latest` tag (vs IMMUTABLE where each push needs a unique tag) |
| `encryptionType: AES256` | Images encrypted at rest (free, default) |
| `scanOnPush: false` | Not scanning for vulnerabilities on push (can enable later) |

**Cost:** $0.10/GB/month for stored images. A typical Docker image is 1-3 GB.  
**Alternative:** Docker Hub (free for public images, $5/mo for private).

---

## Step 7.1 — EC2 Key Pair

```bash
aws ec2 create-key-pair \
  --key-name p053-key \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/p053-key.pem
```

| Part | Meaning |
|------|---------|
| `create-key-pair` | Generates an RSA key pair. AWS keeps the **public** key; you get the **private** key |
| `--key-name p053-key` | Name to identify this key pair in AWS console |
| `--query 'KeyMaterial'` | JMESPath filter — extracts only the private key text from the JSON response |
| `--output text` | Raw text instead of JSON (so the .pem file is clean) |
| `> ~/.ssh/p053-key.pem` | Saves private key to your SSH directory |

**Why?** SSH into EC2 requires key-based auth (no passwords). This is the private key you'll use: `ssh -i ~/.ssh/p053-key.pem ubuntu@<EC2_IP>`

```bash
chmod 400 ~/.ssh/p053-key.pem
```

| Part | Meaning |
|------|---------|
| `chmod 400` | Sets file permission to read-only for owner. SSH refuses to use keys with broader permissions (security requirement) |

**Output of `ls -la`:**
```
-r--------  1 rajendarmuddasani  staff  1674 Jun 28 14:23 p053-key.pem
```
`-r--------` = only owner can read. Perfect.

---

## Step 7.2 — Security Group & Firewall Rules

### 7.2a. Create Security Group

```bash
aws ec2 create-security-group \
  --group-name p053-sg \
  --description "P053 MLOps security group"
```

| Part | Meaning |
|------|---------|
| `create-security-group` | Creates a virtual firewall for EC2 instances |
| `--group-name p053-sg` | Identifier name |
| `--description` | Required — AWS won't create without a description |

**Output:**
```json
{
    "GroupId": "sg-0f11ba29c1155cba3"
}
```

The `GroupId` is used in all subsequent commands to reference this firewall.

### 7.2b. Get Your Public IP

```bash
curl -s https://checkip.amazonaws.com
```

**Output:** `121.6.66.58`

**Why?** We restrict ALL firewall rules to only allow connections from YOUR IP address. This means no one else on the internet can reach your EC2 instance.

### 7.2c. Add Firewall Rules (Ingress)

Each command opens a specific port on the firewall:

```bash
aws ec2 authorize-security-group-ingress \
  --group-id sg-0f11ba29c1155cba3 \
  --protocol tcp --port <PORT> --cidr 121.6.66.58/32
```

| Part | Meaning |
|------|---------|
| `authorize-security-group-ingress` | "Allow incoming traffic" on this rule |
| `--protocol tcp` | TCP protocol (all our services use TCP) |
| `--port <PORT>` | Which port to open |
| `--cidr 121.6.66.58/32` | Only allow this IP. The `/32` means "exactly this one IP address" (vs `/24` = 256 IPs, `/0` = ALL IPs) |

**The 6 ports we opened:**

| Port | Service | Why |
|------|---------|-----|
| 22 | SSH | Remote terminal access to EC2 instance |
| 5001 | MLflow UI | Track experiments, view model registry |
| 3000 | Grafana | Production monitoring dashboards |
| 8000 | FastAPI | Model inference API endpoint |
| 8080 | Airflow UI | View DAG runs, trigger workflows |
| 9000 | Kafdrop | Kafka topic viewer (see streaming data) |

**Output for each rule:**
```json
{
    "Return": true,
    "SecurityGroupRules": [{
        "SecurityGroupRuleId": "sgr-...",
        "IpProtocol": "tcp",
        "FromPort": <PORT>,
        "ToPort": <PORT>,
        "CidrIpv4": "121.6.66.58/32"
    }]
}
```

`"Return": true` = rule added successfully.

### 7.2d. Verify All Rules

```bash
aws ec2 describe-security-groups --group-ids sg-0f11ba29c1155cba3 \
  --query 'SecurityGroups[0].IpPermissions'
```

| Part | Meaning |
|------|---------|
| `describe-security-groups` | List all rules for a given security group |
| `--query 'SecurityGroups[0].IpPermissions'` | JMESPath filter — show only the ingress rules array |

Shows all 6 rules with ports and your IP confirmed.

> **⚠️ If your IP changes** (e.g., restart router, switch WiFi), run `curl -s https://checkip.amazonaws.com` and update the security group rules with the new IP.

---

## Cost Summary

### What We've Created — ALL FREE

| Resource | Monthly Cost | Status |
|----------|-------------|--------|
| IAM User + Policies | $0 | ✅ Created |
| S3 Bucket (empty) | $0 | ✅ Created |
| ECR Repository (empty) | $0 | ✅ Created |
| Key Pair | $0 | ✅ Created |
| Security Group | $0 | ✅ Created |

### What's Coming — CHARGES BEGIN

| Resource | Hourly Cost | Monthly (24/7) | Status |
|----------|-----------|-----------------|--------|
| EC2 t3.medium | $0.0416/hr | ~$30/mo | ❌ Step 7.3 |
| RDS db.t3.micro | $0.018/hr | ~$13/mo | ❌ Step 8 |
| S3 Storage (est. 40GB) | — | ~$0.92/mo | Usage-based |
| ECR Storage (est. 3GB) | — | ~$0.30/mo | Usage-based |
| Data Transfer (est.) | — | ~$2-5/mo | Usage-based |
| **Total estimated** | | **~$46/mo** | |

### S3 Pricing Reference

| Storage | Monthly Cost |
|---------|-------------|
| 1 GB | $0.023 |
| 100 GB | $2.30 |
| 1 TB (1000 GB) | $23.00 |
| 3 TB (3000 GB) | $69.00 |

> Our 40-day simulation generates ~1GB/day of Parquet data = ~40 GB total = **~$0.92/month**.
> 3000 GB would only happen with massive production datasets (not our case).

---

## Quick Reference — All AWS Resource IDs

```
Account ID:      718036735422
Region:          us-west-2
IAM User:        p053-cicd-user
S3 Bucket:       p053-mlflow-artifacts
ECR Repo URI:    718036735422.dkr.ecr.us-west-2.amazonaws.com/053-memory-yield-predictor
Security Group:  sg-0f11ba29c1155cba3
Key Pair:        p053-key (~/.ssh/p053-key.pem)
User IP:         121.6.66.58
EC2 Instance:    PENDING (Step 7.3)
RDS Endpoint:    PENDING (Step 8)
```

---

*P053 Memory Yield Predictor — AIML Engineering Lab*
