# AWS Infrastructure Guide — P053 Memory Yield Predictor

> Step-by-step guide to setting up AWS services for production deployment.
> **Budget:** $1,000 SGD (~$740 USD) total project budget.
> **Architecture:** ALL ON AWS — g4dn.xlarge with T4 GPU for training + inference.
> **Last Updated:** 2026-04-07 (Phase 0b complete, commit 072f877)  
> **Blocker:** GPU quota increase pending (0→4 vCPUs for G instances). RDS created & stopped.

---

## 1. What AWS Services We Use (and Why)

| Service | Instance | Purpose | Est. Cost |
|---------|----------|---------|-----------|
| **EC2** | g4dn.xlarge (4 vCPU, 16GB RAM, T4 GPU 16GB) | GPU training + 40-day sim + full Docker stack | $0.526/hr × ~100 hrs = ~$53 |
| **RDS PostgreSQL** | db.t3.micro | Production MLflow tracking database | $0.018/hr × 720 hrs = ~$13 |
| **S3** | p053-mlflow-artifacts | Models, data, drift reports, benchmarks | ~$5 |
| **ECR** | 053-memory-yield-predictor | Store Docker images (API, MLflow, Airflow-GPU) | ~$1 |
| **CloudWatch** | Billing alarm | Auto-stop at $500 threshold | Free |
| **IAM** | p053-cicd-user | Access control for all services | Free |

**Total estimated: ~$72 USD** — well within $740 USD budget.

### Why g4dn.xlarge (T4) instead of A100?
- Our model is 317K params — T4 handles it comfortably (6GB VRAM used)
- A100 requires p4d.24xlarge ($32.77/hr, 8 GPUs minimum) — 62× more expensive
- T4 retrain: ~45 min ($0.39). A100 retrain: ~15 min ($8.19). Not worth it.
- Day 1 initial training was done on Colab A100 (free). All retrains on T4.

### Auto-Stop Protection
- `src/ec2_auto_stop.py` stops EC2 after Phase 3 simulation completes
- CloudWatch billing alarm triggers at $500 spend
- EC2 user-data bootstraps everything — no manual SSH needed

---

## 2. Create an AWS Account

If you don't have one:
1. Go to https://aws.amazon.com/
2. Click "Create an AWS Account"
3. Enter email, password, account name
4. Add a payment method (credit/debit card)
5. Choose "Basic Support" (free tier)
6. You'll get 12 months of free tier for many services

---

## 3. Create an IAM User (DO NOT use root account)

**Why?** The root account has unlimited permissions. If those credentials leak,
someone could spin up $50,000 of GPU instances. IAM users have scoped permissions.

### Steps:
1. Sign in to AWS Console: https://console.aws.amazon.com/
2. Go to **IAM** (search "IAM" in top search bar)
3. Click **Users** → **Create user**
4. User name: `p053-cicd`
5. Check **"Provide user access to the AWS Management Console"** (optional — CLI-only is fine)
6. Click **Next**
7. Choose **"Attach policies directly"**
8. Search and attach these policies:
   - `AmazonEC2ContainerRegistryFullAccess` — push/pull Docker images
   - `AmazonS3FullAccess` — store MLflow artifacts + DVC data
   - `AmazonRDSFullAccess` — create/manage PostgreSQL database (only needed initially)
9. Click **Next** → **Create user**

### Get Access Keys:
1. Click on the newly created user `p053-cicd`
2. Go to **Security credentials** tab
3. Under **Access keys**, click **Create access key**
4. Choose **"Command Line Interface (CLI)"**
5. Check the acknowledgment box → **Next** → **Create access key**
6. **SAVE BOTH VALUES NOW** (you can never see the secret again):
   - `Access key ID`: Looks like `AKIAIOSFODNN7EXAMPLE`
   - `Secret access key`: Looks like `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`

### ⚠️ Security Rules:
- **NEVER** commit access keys to Git
- **NEVER** put them in code files
- Store them as **GitHub Secrets** (see Section 7) or in `.env` files that are `.gitignore`d
- Rotate keys every 90 days
- If you ever suspect a leak: IAM → User → Security credentials → Deactivate key immediately

---

## 4. Set Up ECR (Container Registry)

ECR stores your Docker images so CI/CD and EC2 can pull them.

### Via AWS Console:
1. Go to **ECR** (search "ECR" or "Elastic Container Registry")
2. Click **Create repository**
3. Visibility: **Private**
4. Repository name: `p053-memory-yield-predictor`
5. Tag immutability: **Disabled** (we'll use `latest` + semver tags)
6. Click **Create repository**
7. Note the **Repository URI**: `123456789012.dkr.ecr.us-west-2.amazonaws.com/p053-memory-yield-predictor`

### Via CLI (after installing AWS CLI):
```bash
# Install AWS CLI
brew install awscli

# Configure with your access keys
aws configure
# AWS Access Key ID: AKIAIOSFODNN7EXAMPLE
# AWS Secret Access Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
# Default region: us-west-2
# Default output format: json

# Create ECR repository
aws ecr create-repository \
    --repository-name p053-memory-yield-predictor \
    --region us-west-2
```

---

## 5. Set Up S3 Bucket (Artifacts + DVC)

### Create bucket:
1. Go to **S3** in AWS Console
2. Click **Create bucket**
3. Bucket name: `p053-mlflow-artifacts` (must be globally unique — add your initials if taken)
4. Region: `us-west-2` (same as other services)
5. **Block all public access**: ✅ Yes (keep default — private bucket)
6. Versioning: **Enable** (important for DVC and MLflow artifact versioning)
7. Click **Create bucket**

### Create a folder structure:
```
p053-mlflow-artifacts/
├── mlflow/          ← MLflow artifact storage
├── dvc/             ← DVC data files
└── models/          ← Production model weights
```

---

## 6. Set Up RDS PostgreSQL (Production MLflow Backend)

### Create database:
1. Go to **RDS** in AWS Console
2. Click **Create database**
3. Choose **Standard create**
4. Engine: **PostgreSQL** (version 16.x)
5. Template: **Free tier** (if available) or **Dev/Test**
6. Settings:
   - DB instance identifier: `p053-mlflow-db`
   - Master username: `mlflow`
   - Master password: (generate a strong one, save it!)
7. Instance type: `db.t4g.micro` (~$12/month)
8. Storage: 20 GB gp3 (minimum, auto-scaling enabled)
9. Connectivity:
   - VPC: Default
   - Public access: **Yes** (for initial setup — lock down later)
   - Security group: Create new → allow port 5432 from your IP
10. Click **Create database** (takes 5-10 minutes)
11. Note the **Endpoint**: `p053-mlflow-db.xxxx.us-west-2.rds.amazonaws.com`

### Connection string:
```
postgresql://mlflow:<password>@p053-mlflow-db.xxxx.us-west-2.rds.amazonaws.com:5432/mlflow
```

---

## 7. Add Secrets to GitHub

Your CI/CD pipeline needs AWS credentials to push Docker images and deploy.
**GitHub Secrets** encrypt the values — no one (including you) can read them after saving.

### Steps:
1. Go to your GitHub repo: `https://github.com/rajendarmuddasanu/DRAM_Chip_Yield_Prediction_Production_MLOps`
2. Click **Settings** (tab at top)
3. Left sidebar: **Secrets and variables** → **Actions**
4. Click **New repository secret** for each:

| Secret Name | Value | Used By |
|------------|-------|---------|
| `AWS_ACCESS_KEY_ID` | `AKIAIOSFODNN7EXAMPLE` | ECR login, S3 access |
| `AWS_SECRET_ACCESS_KEY` | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` | ECR login, S3 access |
| `AWS_REGION` | `us-west-2` | All AWS services |
| `AWS_ACCOUNT_ID` | `123456789012` (from ECR URI) | ECR push |
| `RDS_PASSWORD` | (your RDS master password) | MLflow PostgreSQL |

### How CI/CD Uses These:
In `.github/workflows/ci.yml`, the `ecr-push` job references these:
```yaml
- uses: aws-actions/configure-aws-credentials@v4
  with:
    aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
    aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    aws-region: ${{ secrets.AWS_REGION }}
```

The credentials are injected as environment variables during the GitHub Actions run.
They're never visible in logs (GitHub auto-masks them).

---

## 8. Install AWS CLI Locally

```bash
# macOS
brew install awscli

# Verify
aws --version

# Configure
aws configure
# Enter your access key ID, secret key, region (us-west-2), output format (json)

# Test connection
aws sts get-caller-identity
# Should return your account ID and user ARN
```

---

## 9. Push Docker Image to ECR (Manual Test)

Before CI/CD does it automatically, test the push manually:

```bash
# Login to ECR
aws ecr get-login-password --region us-west-2 | \
    docker login --username AWS --password-stdin \
    123456789012.dkr.ecr.us-west-2.amazonaws.com

# Tag your image
docker tag docker-api:latest \
    123456789012.dkr.ecr.us-west-2.amazonaws.com/p053-memory-yield-predictor:latest

# Push
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/p053-memory-yield-predictor:latest
```

---

## 10. Cost Monitoring

### Set up billing alerts:
1. Go to **Billing** → **Budgets**
2. Click **Create budget**
3. Budget type: **Cost budget**
4. Budget amount: **$50** (monthly)
5. Alert threshold: **80%** ($40) — sends email when you hit 80%
6. Add your email for notifications

### Monthly cost breakdown (approximate):
```
RDS t4g.micro:     $12.41/month
S3 (5 GB):          $0.12/month
ECR (2 images):     $0.50/month
Data transfer:      $1-3/month
EC2 t3.medium:    $15-20/month (only when running)
────────────────────────────
Total:             ~$30-35/month
```

### Cost-saving tips:
- **Stop RDS** when not actively using: RDS → Actions → Stop temporarily (auto-restarts after 7 days)
- **Stop EC2** when not running simulations
- **Delete old ECR images**: Keep only latest 3 tags
- **S3 lifecycle rules**: Move old artifacts to S3 Glacier after 30 days

---

## 11. Security Best Practices

1. **MFA on root account** — IAM → Root user → Security credentials → Enable MFA
2. **Least privilege** — Only give the IAM user permissions it needs
3. **Rotate keys** — Every 90 days: create new key → update secrets → delete old key
4. **VPC security groups** — Restrict RDS port 5432 to your IP only
5. **Encryption** — Enable S3 encryption (default SSE-S3), RDS encryption at rest
6. **CloudTrail** — Enable for audit logging (free for management events)

---

*Last updated: 2026-04-05*
