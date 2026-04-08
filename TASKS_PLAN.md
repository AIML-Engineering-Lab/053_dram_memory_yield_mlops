# P053 — Production Execution Tasks

> **Timeline:** Apr 4 – May 15, 2026 (40 working days)
> **Budget:** $1,000 SGD (~$740 USD) — g4dn.xlarge GPU on AWS
> **GitHub:** [AIML-Engineering-Lab/053_dram_memory_yield_mlops](https://github.com/AIML-Engineering-Lab/053_dram_memory_yield_mlops)
> **Architecture:** ALL ON AWS — Zero Colab, Zero MacBook training

---

## Summary

| Phase | Done | Remaining | Blocked On |
|-------|------|-----------|------------|
| P0 Cleanup + Phase 0 Wire-Up | 11/11 | 0 | ✅ COMPLETE (commit aeb83e2) |
| P0b GPU Selector + Drift Tags | 5/5 | 0 | ✅ COMPLETE (commit 072f877) |
| P1 Docker | 10/13 | 3 | User browser checks |
| P2 Notebook | 6/6 | 0 | — |
| P3 GitHub | 5/7 | 2 | User verifies repo + branch protection |
| P4 CI/CD | 5/9 | 4 | CI Run #15 triggered; verify on Actions tab |
| P5 AWS Infra (Code) | 18/20 | 2 | **GPU quota increase pending (0→4 vCPUs)** |
| P6 GPU Training | 1/4 | 3 | **Day 1 DONE** ✅. Retrains on g4dn.xlarge after quota |
| P7 40-Day Sim | 0/13 | 13 | EC2 launch needed (GPU quota) |
| P8 Report | 0/6 | 6 | Simulation data |
| P9 Content | 3/4 | 1 | User LinkedIn review |
| P10 Polish | 1/8 | 7 | Everything above |
| **Total** | **64/106** | **42** | **GPU quota approval** |

### Phase 0 Completion (commit aeb83e2 — pushed to GitHub)
- ✅ Replaced ALL fake/simulated code with REAL GPU training (train.py subprocess)
- ✅ S3ArtifactManager (boto3) — uploads models, Parquet, drift reports to S3
- ✅ docker-compose-bigdata-aws.yml — production stack for g4dn.xlarge (real S3/RDS, NVIDIA GPU)
- ✅ ec2-user-data.sh — full EC2 bootstrap (Docker, NVIDIA drivers, nvidia-container-toolkit)
- ✅ ec2_auto_stop.py — auto-stop EC2 + RDS + NAT gateway after Phase 3 + CloudWatch $200 billing alarm
- ✅ Variable daily volumes (1M–10M rows/day) via get_daily_volume()
- ✅ Real MLflow Model Registry operations (register, promote, rollback)
- ✅ 20/20 tests passing, lint clean, CI passing

### Phase 0b Completion (commit 072f877 — pushed to GitHub)
- ✅ gpu_selector.py — Auto GPU selection: T4 (<50M params) → A100 (>1.2B params or >1B rows)
- ✅ Low-data drift tagging — drift_reliable flag, tagged to MLflow, never triggers retrain
- ✅ RDS auto-stop — Phase 3 cleanup stops EC2 + RDS + deletes NAT gateway
- ✅ CloudWatch alarm → $200 USD (was $500)
- ✅ Comprehensive daily simulation logger (simulation_log.py)

### Phase 0b+ Completion: Colab Fallback (commit 6feff7a — pushed to GitHub)
- ✅ compute_backend.py — AWS→Colab→Local fallback chain with auto-detection
- ✅ gpu_selector.py — Colab GPU catalog added (T4 default, A100 for >1TB/day)
- ✅ run_simulation.py — --backend and --checkpoint flags for Colab resilience
- ✅ NB04_colab_training.ipynb — Colab notebook with T4/A100 support, S3 upload
- ✅ .github/copilot-instructions.md — standalone workspace context (13 sections)
- ✅ ED-043 documented (compute backend fallback decision)
- ✅ never_forget rules updated for Colab fallback (Rules 12-13)
- ✅ AWS appeal submitted for GPU quota re-evaluation

**Why:** AWS rejected GPU quota (0→4 vCPUs for G instances). Built Colab fallback in same session. Zero blocked days.

---

## 40-Day Production Timeline (AWS-Primary, Colab-Fallback Architecture)

```
Day 0      ┃ P5: Launch g4dn.xlarge EC2 + RDS PostgreSQL on AWS
           ┃     ec2-user-data.sh bootstraps: Docker, NVIDIA drivers, repo clone
           ┃     docker-compose-bigdata-aws.yml starts: Airflow+GPU, MLflow→RDS, Kafka, Spark
Day 1      ┃ P6: Deploy Day 1 champion model (trained on Colab A100, AUC-ROC=0.816)
           ┃     Register model v1 @champion in MLflow → S3
Day 2-19   ┃ P7: Airflow DAG runs daily simulation (1M-10M rows/day, variable)
           ┃     Kafka streaming → Spark ETL → batch predict → drift monitoring
           ┃     Day 5:  first health check (PSI baseline)
           ┃     Day 15: PSI shows early shift, warnings logged
Day 20     ┃ P7: Moderate drift detected (PSI > 0.10, 3+ critical features)
           ┃     Retrain triggered ON g4dn.xlarge T4 GPU (~45 min, 30 epochs)
           ┃     Canary test → promote v2 @champion → S3 upload
Day 21-30  ┃ P7: Continue daily inference with v2
Day 31     ┃ P7: Severe drift → Retrain v3 ON g4dn.xlarge (40 epochs)
Day 32-38  ┃ P7: Recovery period monitoring
Day 39     ┃ P7: Bad model deliberate deploy → rollback → retrain v4 (50 epochs)
Day 40     ┃ P7: Final production run. ec2_auto_stop.py triggers.
           ┃ P8: Report + Dashboard, P9: LinkedIn, P10: Final push
```

**Key change from earlier plan:** ALL retraining happens on AWS g4dn.xlarge (T4 GPU, $0.526/hr).
No more Colab for retrains. Only Day 1 initial training was done on Colab A100.

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

### P5: AWS Infrastructure (2 remaining — launch only)

**Phase 0 built all infrastructure code.** Remaining tasks are LAUNCH operations.

| # | Task | Who | Status |
|---|------|-----|--------|
| 47 | Verify AWS CLI: `aws sts get-caller-identity` | User | ✅ |
| 48 | S3 bucket created: `p053-mlflow-artifacts` | Both | ✅ |
| 49 | ECR repo created: `053-memory-yield-predictor` | Both | ✅ |
| 50 | EC2 key pair: `p053-key` | Both | ✅ |
| 51 | Security group: `sg-0f11ba29c1155cba3` | Both | ✅ |
| 52 | `.env.aws.template` updated | Copilot | ✅ |
| 53 | `docker-compose-bigdata-aws.yml` — production stack (GPU) | Copilot | ✅ |
| 54 | `Dockerfile.airflow-gpu` — PyTorch CUDA 12.1 | Copilot | ✅ |
| 55 | `ec2-user-data.sh` — g4dn.xlarge bootstrap | Copilot | ✅ |
| 56 | `s3_utils.py` — S3ArtifactManager (boto3) | Copilot | ✅ |
| 57 | `ec2_auto_stop.py` — auto-stop + billing alarm | Copilot | ✅ |
| 58 | Real GPU training wired into DAGs | Copilot | ✅ |
| 59 | Real MLflow Model Registry in promote/rollback | Copilot | ✅ |
| 60 | 20/20 tests passing, lint clean | Copilot | ✅ |
| 61 | **Launch g4dn.xlarge EC2 instance** | User | ⏳ GPU quota pending (appeal submitted) |
| 62 | **RDS PostgreSQL provisioned** | User | ✅ p053-mlflow-db (stopped, saving costs) |
| 61b | **Colab fallback: compute_backend.py** | Copilot | ✅ AWS→Colab→Local chain |
| 61c | **Colab fallback: NB04_colab_training.ipynb** | Copilot | ✅ T4/A100 with checkpoint |

**Cost estimate:** g4dn.xlarge @ $0.526/hr × ~100 hrs = ~$53. RDS db.t3.micro @ $0.018/hr × 720 hrs = ~$13. S3 ~$5. **Total: ~$71 USD**
**Colab fallback cost:** T4 @ 1.36 CU/hr × ~16 hrs = ~22 CU. Buy 100 CU PAYG (SGD 14.46).

### P6: GPU Training (3 remaining retrains on g4dn.xlarge)

| # | Task | Who | Status |
|---|------|-----|--------|
| 63 | **Day 1 initial training** (Colab A100, 50ep, 201.7min) | User | ✅ AUC-ROC=0.816, AUC-PR=0.054 |
| 64 | **Day 20 drift retrain** (g4dn.xlarge T4, 30ep, ~45min) | Auto (Airflow) | ⬜ |
| 65 | **Day 31 severe retrain** (g4dn.xlarge T4, 40ep, ~60min) | Auto (Airflow) | ⬜ |
| 66 | **Day 39 recovery retrain** (g4dn.xlarge T4, 50ep, ~90min) | Auto (Airflow) | ⬜ |

**Key change:** Retrains are FULLY AUTOMATED via `dag_retrain_pipeline.py` on AWS g4dn.xlarge.
If AWS unavailable, retrains run on Colab T4 via NB04 with `--checkpoint` for disconnect resilience.
`_execute_gpu_training()` calls `train.py` via subprocess on T4 GPU.

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

## Data Flow Architecture (AWS-Only)

```
┌─── LOCAL (Mac) — Development Only ──────────────────┐
│  src/ code development, tests, lint                        │
│  git push → GitHub → CI/CD → GHCR + ECR                    │
└────────────────────────────────────────────────┘
        │ git push                
        ▼                          
┌─── GITHUB ─────────────┐    ┌─── AWS g4dn.xlarge (T4 GPU) ────────────┐
│  CI: lint → test → build  │    │  Airflow (GPU scheduler)                   │
│  Docker → GHCR + ECR     │    │   └→ dag_simulation_master.py (40 days)     │
└──────────────────────┘    │   └→ dag_daily_yield_pipeline.py            │
                                │   └→ dag_retrain_pipeline.py (REAL T4 GPU) │
                                │  Kafka → Spark ETL → predict → drift      │
                                │  FastAPI serving + Prometheus metrics     │
                                │  ec2_auto_stop.py at Phase 3 completion   │
                                └──────┬────────┬────────────────────┘
                                       │        │
                                       ▼        ▼
                                ┌── S3 ──┐  ┌── RDS PostgreSQL ──┐
                                │ models │  │ MLflow tracking    │
                                │ data   │  │ experiment metadata│
                                │ drift  │  └───────────────────┘
                                └───────┘
```
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
