# P053 — Production Execution Tasks

> **Timeline:** Feb 1 – Mar 12, 2026 (40 working days)
> **Budget:** $50/month ($10 Colab Pro + $40 AWS)
> **GitHub:** [AIML-Engineering-Lab/053_dram_memory_yield_mlops](https://github.com/AIML-Engineering-Lab/053_dram_memory_yield_mlops)

---

## Summary

| Phase | Done | Remaining | Blocked On |
|-------|------|-----------|------------|
| P0 Cleanup | 10/11 | 1 | Folder rename (optional) |
| P1 Docker | 10/13 | 3 | User browser checks |
| P2 Notebook | 6/6 | 0 | — |
| P3 GitHub | 5/7 | 2 | User verifies repo + branch protection |
| P4 CI/CD | 5/9 | 4 | CI lint fixed (6b347ae); verify on Actions tab |
| P5 AWS Infra | 7/18 | 11 | EC2 launch + RDS (charges start) |
| P6 Colab A100 | 4/16 | 12 | **Session 1 DONE** ✅ 50ep/201.7min/AUC-ROC=0.816 |
| P7 40-Day Sim | 0/13 | 13 | EC2+RDS launch needed |
| P8 Report | 0/6 | 6 | Simulation data |
| P9 Content | 3/4 | 1 | User LinkedIn review |
| P10 Polish | 1/8 | 7 | Everything above |
| **Total** | **51/111** | **60** | |

---

## 40-Day Production Timeline

```
Feb 1-3    ┃ P5: AWS setup (EC2 + RDS + S3 + ECR)
Feb 4      ┃ P6: Upload NB03 to Colab, connect A100
Feb 4-5    ┃ P6: Session 1 — Day 1 initial training (50 epochs, ~90 min)
Feb 5      ┃ Deploy Day 1 model → Docker stack on EC2
Feb 6-25   ┃ P7: Daily inference on synthetic data (100K-1M rows/day)
           ┃     Day 5:  first health check
           ┃     Day 15: PSI shows early shift
           ┃     Day 20: moderate drift detected → Colab retrain Session 2
Feb 26     ┃ P6: Session 2 — Day 20 drift retrain (30 epochs, ~60 min)
Feb 27     ┃ Deploy retrained model v2
Feb 28-    ┃ P7: Continue daily inference
Mar 3      ┃     Day 31: severe drift → Colab retrain Session 3
Mar 3      ┃ P6: Session 3 — Day 31 severe drift (40 epochs)
Mar 4      ┃     Deploy model v3
Mar 5-8    ┃ P7: Recovery period
           ┃     Day 35: model degradation detected
Mar 9      ┃ P6: Session 4 — Day 39 from-scratch recovery (50 epochs)
Mar 10     ┃ Deploy champion model v4
Mar 11     ┃ P7: Full production run (5M rows/day × 40 = 200M total)
Mar 12     ┃ P8: Report + Dashboard update, P9: LinkedIn, P10: Final push
```

---

## Phase Breakdown — Remaining Tasks

### P1: Docker Stack (3 remaining)

| # | Task | Who | Status |
|---|------|-----|--------|
| 17 | Open http://localhost:5001 — verify 4 MLflow runs | User | ⬜ |
| 20 | Open http://localhost:3000 — Grafana (admin/admin) | User | ⬜ |
| 21 | Take Docker screenshots (MLflow UI, Grafana, docker ps) | User | ⬜ |

**Note:** Stack is running now. All 6 services healthy. Just need browser verification.

### P3: GitHub (2 remaining)

| # | Task | Who | Status |
|---|------|-----|--------|
| 36 | Verify README renders on GitHub | User | ⬜ |
| 37 | Protect main branch (Settings → Branches → Add rule) | User | ⬜ |

### P4: CI/CD (5 remaining)

| # | Task | Who | Status |
|---|------|-----|--------|
| 40 | Set GitHub Secrets: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION | User | ⬜ |
| 42 | Verify lint passes (GitHub Actions tab) | User | ⬜ |
| 43 | Verify tests pass (20/20) | User | ⬜ |
| 44 | Verify Docker build passes | User | ⬜ |
| 45 | Fix any CI failures | Copilot | ⬜ |

**How to set GitHub Secrets:**
1. Go to https://github.com/AIML-Engineering-Lab/053_dram_memory_yield_mlops/settings/secrets/actions
2. Click "New repository secret"
3. Add each: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` (us-west-2)

### P5: AWS Infrastructure (16 tasks)

**What I need from you:**
1. AWS Access Key ID
2. AWS Secret Access Key
3. AWS Region preference (us-west-2 recommended — cheapest)

| # | Task | Who | Est. |
|---|------|-----|------|
| 47 | Verify AWS CLI: `aws sts get-caller-identity` | User | 1 min |
| 48 | Run setup_aws.sh: Creates S3 + RDS + ECR | Both | 10 min |
| 49 | Wait for RDS to be available | Auto | 10 min |
| 50 | Configure RDS security group | User | 5 min |
| 51 | Create S3 bucket policy | Both | 5 min |
| 52 | Launch EC2 t3.medium (Amazon Linux 2023, 30 GB) | User | 5 min |
| 53 | Install Docker + Compose on EC2 | Both | 10 min |
| 54 | Configure EC2 security groups (22,5001,8000,3000,9090) | User | 5 min |
| 55 | Copy compose + configs to EC2 | Both | 5 min |
| 56 | Fill in .env on EC2 (RDS endpoint, S3 bucket) | Both | 5 min |
| 57 | Deploy compose on EC2 | Both | 5 min |
| 58 | Verify MLflow UI on AWS | Both | 2 min |
| 59 | Verify PostgreSQL on RDS | Both | 2 min |
| 60 | Verify S3 access | Both | 1 min |
| 61 | Run retrolog against AWS MLflow | Both | 5 min |
| 62 | Take AWS screenshots | User | 10 min |

### P6: Colab A100 Training (16 tasks)

| # | Task | Who | Est. |
|---|------|-----|------|
| 63 | Upload NB03 to Google Colab | User | 2 min |
| 64 | Connect A100 runtime | User | 1 min |
| 65 | Install deps on Colab | Auto | 3 min |
| 66 | Set MLFLOW_TRACKING_URI to AWS EC2 IP | User | 1 min |
| 67 | Verify Colab → AWS connection | User | 1 min |
| 68 | Upload preprocessed_full.npz (2.1 GB) to Colab | User | 5 min |
| 69 | **Session 1: Day 1 initial** (50 ep, lr=1e-3) | User | 90 min |
| 70 | Verify run in AWS MLflow | User | 2 min |
| 71 | Download Session 1 artifacts to Drive | User | 5 min |
| 72 | **Session 2: Day 20 drift retrain** (30 ep, lr=3e-4) | User | 60 min |
| 73 | Verify retrain in MLflow (A/B comparison) | User | 2 min |
| 74 | Register Day 31 model as v2 @challenger | User | 5 min |
| 75 | **Session 3: Day 31 severe** (40 ep, lr=1e-4) | User | 70 min |
| 76 | Demonstrate rollback in MLflow | Both | 5 min |
| 77 | **Session 4: Day 39 recovery** (50 ep, lr=5e-4) | User | 90 min |
| 78 | Copy all artifacts to project | Both | 5 min |

### P7: 40-Day Production Simulation (13 tasks)

| # | Task | Who | Est. |
|---|------|-----|------|
| 79 | SSH into EC2 | User | 1 min |
| 80 | Clone repo on EC2 | Both | 2 min |
| 81 | Install Python + deps | Both | 5 min |
| 82 | Run `--fast` (100K/day × 40 = 4M rows) | Both | 10 min |
| 83 | Verify drift events in AWS MLflow | User | 3 min |
| 84 | Run `--medium` (1M/day × 40 = 40M rows) | Both | 30 min |
| 85 | Verify Prometheus metrics | User | 3 min |
| 86 | Verify Grafana live dashboards | User | 5 min |
| 87 | Run `--full` (5M/day × 40 = 200M rows) | Auto | 2-3 hrs |
| 88 | Verify S3 artifacts | Both | 2 min |
| 89 | Export MLflow experiment summary | Both | 5 min |
| 90 | Take production run screenshots | User | 10 min |
| 91 | **STOP EC2 INSTANCE** (saves money!) | User | 1 min |

### P8: Report & Dashboard (6 tasks)

| # | Task | Who | Est. |
|---|------|-----|------|
| 92 | Update dashboard with AWS metrics + drift timeline | Copilot | 30 min |
| 93 | Add AWS screenshots to assets/ | Both | 5 min |
| 94 | Update report HTML (AWS section, 4-session A100) | Copilot | 30 min |
| 95 | Regenerate PDF (Playwright) | Copilot | 5 min |
| 96 | Update README (3-tier deployment, costs, sessions) | Copilot | 15 min |
| 97 | Visual QA of report | User | 10 min |

### P9: Content Hub (1 remaining)

| # | Task | Who | Status |
|---|------|-----|--------|
| 101 | Review LinkedIn post + comment tone | User | ⬜ |

### P10: Final Polish (7 remaining)

| # | Task | Who | Est. |
|---|------|-----|------|
| 103 | Final git commit | Copilot | 2 min |
| 104 | Push to GitHub | Copilot | 2 min |
| 105 | Tag release v2.0.0 | Copilot | 2 min |
| 106 | Verify CI/CD green | User | 5 min |
| 107 | Verify README renders | User | 3 min |
| 108 | Stop/delete AWS resources | User | 5 min |
| 109 | Archive in STATUS_TRACKER.csv | Copilot | 2 min |

---

## Data Flow Architecture

```
┌─── LOCAL (Mac) ─────────────────────────────────────┐
│  src/data_generator.py → 16M rows (2.1 GB NPZ)      │
│  DVC tracks data → S3 (pointer files in git)         │
│  Docker stack: API + MLflow + PostgreSQL + monitoring │
└──────────────────────────────────────────────────────┘
        │ upload 2.1 GB                    │ git push
        ▼                                  ▼
┌─── COLAB A100 ──────┐  ┌─── GITHUB ─────────────────┐
│  NB03 training       │  │  CI/CD: lint → test → build │
│  4 sessions, MLflow  │  │  Docker → GHCR + ECR        │
│  → logs to AWS MLflow│  └───────────────────────────┘
└──────────────────────┘
        │ MLflow metrics + artifacts
        ▼
┌─── AWS ──────────────────────────────────────────────┐
│  EC2: Docker stack (API, MLflow, Prometheus, Grafana) │
│  RDS: PostgreSQL (MLflow backend)                     │
│  S3: Artifacts + DVC data                             │
│  40-day simulation: 200M rows processed               │
└──────────────────────────────────────────────────────┘
```

**Synthetic data is generated locally** by `src/data_generator.py`. For the training set (16M rows = 2.1 GB), we generate it once, preprocess, save as NPZ, and upload to Colab. For the 40-day simulation, data is generated on-the-fly on EC2 (100K–5M rows/day). At 100GB/day scale, we'd need Spark on EMR — that's what our `deploy/docker-compose-bigdata.yml` Spark stack demonstrates.

---

## AWS Cost Estimate

| Service | Per Hour | 40-Day Total |
|---------|----------|-------------|
| EC2 t3.medium (run ~40 hrs) | $0.042 | $1.68 |
| RDS db.t3.micro | $0.017 | $13.00 |
| S3 (< 5 GB) | — | $0.12 |
| ECR (images) | — | $0.10 |
| **Total** | | **~$15** |

*Last updated: 2026-04-05*
