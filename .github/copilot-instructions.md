# P053 — DRAM Memory Yield MLOps — Complete Project Context

> **This file IS the project memory. When asked "what are we doing?" or "what's next?" — read this file.**
> **Last updated: 2026-04-12**

---

## 1. PROJECT OVERVIEW

**Title:** P053 — DRAM Memory Yield Predictor with Full MLOps Pipeline  
**GitHub:** `https://github.com/AIML-Engineering-Lab/053_dram_memory_yield_mlops` (PRIVATE)  
**GitHub (personal):** `https://github.com/rajendarmuddasani/DRAM_Yield_Predictor_MLOps` (PRIVATE)  
**Status:** 40-DAY A100 SIMULATION COMPLETE. Post-processing done. Cleanup & push phase.  
**Scope:** Production-grade MLOps pipeline demonstrating principal-level engineering

### What This Project Demonstrates
- **16M-row DRAM semiconductor dataset** with 1:160 class imbalance (yield defects)
- **HybridTransformerCNN** (317K params) with per-feature tokenization
- **Full MLOps stack:** Airflow orchestration, Kafka streaming, Spark ETL, MLflow tracking
- **40-day production simulation:** variable daily volumes (2M-350M rows), automatic drift detection, GPU retraining, canary evaluation, automatic rollback
- **Infrastructure:** Docker (6 services), Kubernetes manifests, Prometheus/Grafana monitoring
- **Cloud:** AWS (EC2 g4dn.xlarge + RDS + S3 + ECR) with Colab T4/A100 fallback

### Model Performance (Day 1 — A100 Training)
| Metric | Value |
|--------|-------|
| GPU | NVIDIA A100-SXM4-40GB (Colab Pro) |
| Dataset | 16M rows |
| Epochs | 50 (best at epoch 39) |
| Training Time | 201.7 minutes |
| Val AUC-ROC | 0.8157 |
| Val AUC-PR | 0.0543 |
| Val F1 | 0.127 |
| AMP | bfloat16 (NO GradScaler) |
| Model saved | `s3://p053-mlflow-artifacts/models/day1_champion.pt` (1.29 MB) |

### 40-Day Simulation Results (A100, Completed April 2026)
| Metric | Value |
|--------|-------|
| GPU | NVIDIA A100-SXM4-40GB (Colab Pro) |
| Total Days | 40 (Feb 20 - Mar 31, 2026) |
| Rows/Day | 5,000,000 |
| Total Rows | 200,000,000 |
| Wall Clock | 219.4 minutes |
| Retrains | 1 (Day 30, 50 epochs, bfloat16) |
| Canary Failures | 1 (Day 39, rollback to v2) |
| S3 Uploads | 40/40 days |
| Model on S3 | `s3://p053-mlflow-artifacts/models/day30_v2_retrained.pt` |

---

## 2. NON-NEGOTIABLE RULES

1. **AWS IS PRIMARY.** Colab is a FALLBACK only when AWS GPU is unavailable.
2. **GPU ALWAYS.** T4 minimum for all retraining. Never CPU training.
3. **Full day's data** for retraining. NOT a subset.
4. **Track all costs.** Budget-conscious decisions.
5. **All data/models/artifacts on S3.** Nothing stays only on MacBook.
6. **Auto-stop EC2 + RDS + NAT Gateway** after simulation completes.
7. **CloudWatch billing alarm at $200 USD.**
8. **Low-data drift is TAGGED** (MLflow) but **NEVER triggers retraining.**
9. **GPU auto-selector:** T4 for <50M params, A100 only for >1TB data per day.
10. **Document WHY** every decision — this is interview material.

---

## 3. COMPUTE BACKEND — AWS-FIRST, COLAB FALLBACK

### Architecture (compute_backend.py — NEW)

```
run_simulation.py (orchestrate 40-day sim)
    │
    ├─ NON-TRAINING days (inference, drift check, ETL)
    │   └─ Always AWS (EC2 g4dn.xlarge) or local MacBook
    │       → No GPU needed for inference/drift
    │
    └─ TRAINING days (retrain triggered by drift)
        ├─ TRY 1: AWS EC2 g4dn.xlarge (T4 GPU)
        │   └─ If EC2 available + GPU quota OK → train on EC2
        │
        ├─ TRY 2: Google Colab (T4 GPU via notebook)
        │   └─ If AWS fails → Colab notebook runs train.py
        │   └─ Artifacts uploaded to S3 via boto3
        │   └─ MLflow tracks to LOCAL SQLite (not RDS)
        │
        └─ TRY 3: Local MacBook (MPS/CPU — last resort)
            └─ Warning: slow, only for emergencies
```

### GPU Selection Rules
| Condition | GPU | Platform | Cost |
|-----------|-----|----------|------|
| Model <50M params + data <1TB/day | **T4** (16GB) | AWS g4dn.xlarge or Colab | $0.526/hr (AWS) or ~1.36 CU/hr (Colab) |
| Data >1TB/day | **A100** (80GB) | Colab A100 | ~6.79 CU/hr |
| AWS unavailable | **T4** | Colab T4 | ~1.36 CU/hr |
| Everything fails | **MPS/CPU** | MacBook | $0 (slow) |

### MLflow Tracking Strategy
| Backend | When | URI |
|---------|------|-----|
| **RDS PostgreSQL** | AWS EC2 is running | See `.env.aws` for connection string |
| **Local SQLite** | Colab or local fallback | `sqlite:///mlflow.db` |
| **Local PostgreSQL** | Docker compose local | `postgresql://mlflow:mlflow@localhost:5432/mlflow` |

> When AWS quota is approved, Colab runs can be retroactively registered in RDS MLflow.

### Colab Cost Estimate
| GPU | CU/hour | 4-hr run | 4 runs total | Cost (SGD) |
|-----|---------|----------|---------------|------------|
| T4 | ~1.36 | ~5.4 CU | ~22 CU | SGD 14.46 (100 CU pack) |
| A100 | ~6.79 | ~27 CU | ~108 CU | SGD 14.46 (100 CU pack, tight) |

**Recommendation:** Buy SGD 14.46 / 100 CU Pay-As-You-Go. Use T4 for retrains (~22 CU total).

### Colab Training Notebook (notebooks/NB04_colab_training.ipynb)
```
Cell 1: Clone repo, install deps
Cell 2: Set AWS secrets (S3 upload still works from Colab)
Cell 3: Verify GPU (T4 or A100 detection)
Cell 4: !python -m src.run_simulation --fast --backend colab --checkpoint (TEST first)
Cell 5: !python -m src.run_simulation --full --backend colab --checkpoint (PRODUCTION)
Cell 6: Results summary + S3 verification
Cell 7: Google Drive backup (optional)
```

**MacBook does NOT need to stay on** during Colab execution (Colab Pro sessions run in cloud). But Colab Pro disconnects after ~90 min idle — keep tab active or use the checkpoint/resume feature.

### How the 40-Day Simulation ACTUALLY Works (IMPORTANT)

The simulation is **ONE command** that runs ALL 40 days in a loop:
```bash
python -m src.run_simulation --full --backend colab --checkpoint
```

This single command does EVERYTHING automatically:
```
for day in range(1, 41):
    1. Generate daily data (1M-10M rows, variable)
    2. Run batch inference with current champion model
    3. Calculate drift metrics (PSI, feature importance shift)
    4. IF drift detected AND data >10K rows:
       → Trigger retrain on GPU (30-50 epochs)
       → Run canary evaluation
       → IF new model better: promote to champion
       → IF new model worse: rollback to previous
    5. Upload artifacts to S3
    6. Save checkpoint (resume-safe if Colab disconnects)
```

**You do NOT run 40 cells or 40 commands.** You start it ONCE and it completes all 40 days.
**If Colab disconnects:** Re-run the same command — `--checkpoint` resumes from the last completed day.

**On AWS (when quota approved):** Same command runs inside Airflow DAG — truly 100% automated, no human needed.
**On Colab:** 95% automated — you click "Run" once. Only manual step: re-run if disconnected.

---

## 4. AWS INFRASTRUCTURE STATUS

### Live Resources
| Service | Status | Details |
|---------|--------|---------|
| **S3** | ACTIVE | `p053-mlflow-artifacts` (versioned, day1_champion.pt uploaded) |
| **ECR** | ACTIVE | See deploy/aws/.env.aws for full URI |
| **RDS** | STOPPED | `p053-mlflow-db` (Postgres 17, db.t3.micro) — see .env.aws for endpoint |
| **EC2** | NOT LAUNCHED | GPU quota REJECTED, appeal submitted |
| **EIP** | SAFE | RDS-managed, do NOT release |

### AWS Account
- **Region:** us-west-2 (Oregon)
- **Security Group:** See `.env.aws` for IDs
- **Key Pair:** p053-key (~/.ssh/p053-key.pem)
- **Local IP (dynamic):** Check with `curl -s checkip.amazonaws.com` each session
- **Budget alarm:** CloudWatch billing alarm active

### GPU Quota Status
- **Requested:** "Running On-Demand G and VT instances" → 4 vCPUs
- **Status:** REJECTED (April 7). Appeal submitted with detailed use case.
- **Next step:** Wait for appeal response. If rejected again, use Colab fallback.

### ⚠️ RDS Auto-Restart Warning
AWS automatically restarts a stopped RDS instance after **7 days**. If waiting >7 days for quota, re-stop it:
```bash
aws rds stop-db-instance --db-instance-identifier p053-mlflow-db --region us-west-2
```

---

## 5. PROJECT PHASES & PROGRESS

### Phase Summary
| Phase | Description | Status |
|-------|-------------|--------|
| **P0** | Code fixes, wire real GPU training into DAGs | ✅ DONE |
| **P0b** | GPU selector, drift tagging, RDS auto-stop | ✅ DONE |
| **P1** | Docker stack verification (local) | ✅ DONE |
| **P2** | Notebooks (NB01-NB04) | ✅ DONE |
| **P3** | GitHub (README, CI green) | ✅ DONE |
| **P4** | CI/CD (GitHub Actions) | ✅ DONE |
| **P6** | 40-day A100 simulation | ✅ DONE (219.4 min, 200M rows) |
| **P8** | Report + charts regenerated | ✅ DONE |
| **P10** | Cleanup, archive, push | 🟡 IN PROGRESS |

**Simulation complete. Cleanup and push phase.**

### When GPU Quota Approved — Resume Sequence
1. Check IP: `curl -s checkip.amazonaws.com` → update security group if changed
2. Start RDS: `aws rds start-db-instance --db-instance-identifier p053-mlflow-db --region us-west-2`
3. Fill .env.aws password (REPLACE_WITH_YOUR_PASSWORD placeholder)
4. Launch EC2 g4dn.xlarge with ec2-user-data.sh bootstrap (~40 min)
5. Verify services: Airflow :8080, MLflow :5001, FastAPI :8000
6. Register Day 1 champion in MLflow Model Registry
7. Trigger simulation master DAG → MacBook can be closed — EC2 autonomous

### When Using Colab Fallback
1. Open NB04_colab_training.ipynb on Google Colab
2. Select T4 GPU runtime (or A100 for >1TB/day data)
3. Clone repo + install deps
4. Set AWS S3 credentials as Colab secrets
5. Run simulation with `--backend colab --checkpoint`
6. Artifacts auto-upload to S3
7. MLflow tracks to local SQLite
8. When AWS quota approved later, register Colab runs in RDS

---

## 6. KEY ENGINEERING DECISIONS

| ID | Decision | Why |
|----|----------|-----|
| ED-001 | 3-tier hardware: MPS → T4 → A100 | Cost-effective development cycle |
| ED-002 | FocalLoss (α=0.75, γ=2.0) | 1:160 imbalance — SMOTE creates synthetic noise |
| ED-003 | HybridTransformerCNN, per-feature tokenization | Each DRAM parameter has different distributions |
| ED-004 | bfloat16 on A100, float16+GradScaler on T4 | **float16 KILLED training** — GradScaler death spiral with FocalLoss. bfloat16's 8-bit exponent handles extreme gradients |
| ED-007 | AUC-PR as primary metric (not AUC-ROC) | At 1:160 imbalance, AUC-ROC inflates with TN count |
| ED-041 | GPU auto-selector | Automated T4/V100/A10G/A100 selection by model params |
| ED-042 | Low-data drift tagging | Tag don't retrain when <10K rows (drift unreliable) |
| ED-043 | Compute backend fallback | AWS→Colab→Local chain. Built when AWS rejected quota. |

### The bfloat16 Story (Key Interview Material)
- **float16 + GradScaler → DEATH SPIRAL:** Scale → 0, gradients → 0, loss collapses to constant
- **Root cause:** FocalLoss gradient magnitudes overflow float16's 5-bit exponent range
- **Fix:** bfloat16 (8-bit exponent) WITHOUT GradScaler — 11 locations across 7 files
- **Result:** AUC-PR went from collapsed 0.001 to 0.054 (matched T4 baseline)

---

## 7. FILE STRUCTURE REFERENCE

```
053_memory_yield_predictor/
├── .github/
│   ├── copilot-instructions.md    ← THIS FILE (project context)
│   └── workflows/ci.yml           ← GitHub Actions CI
├── assets/                         ← 44 PNGs + 9 carousel slides
├── data/                           ← DVC-tracked datasets (16M rows on S3)
├── deploy/
│   ├── airflow/dags/               ← 3 DAGs (daily, retrain, simulation master)
│   ├── aws/                        ← .env.aws, Dockerfile.airflow-gpu, docker-compose, ec2-user-data.sh
│   ├── docker/                     ← Local docker-compose (6 services), Grafana, Prometheus
│   └── k8s/                        ← Kubernetes manifests (deployment, canary, HPA)
├── docs/                           ← HTML report, PDFs, 8 technical guides
├── notebooks/                      ← NB01 (EDA), NB02 (GPU training), NB03 (simulation), NB04 (Colab fallback)
├── src/                            ← 33 Python modules
│   ├── train.py                    ← Training loop (auto-detects GPU: CUDA/MPS/CPU)
│   ├── model.py                    ← HybridTransformerCNN (317K params)
│   ├── gpu_selector.py             ← Auto GPU selection
│   ├── compute_backend.py          ← AWS → Colab → Local fallback (NEW)
│   ├── run_simulation.py           ← 40-day orchestrator
│   ├── s3_utils.py                 ← S3 artifact management
│   ├── ec2_auto_stop.py            ← Post-simulation cleanup
│   └── (28 more modules)
├── tests/                          ← 20/20 tests passing
├── web/dashboard.html              ← Plotly.js dark dashboard
├── MASTER_PLAN.md                  ← Execution plan (64/106 tasks)
├── TASKS_PLAN.md                   ← Detailed task tracker
├── never_forget_copilot_instructions.md ← Non-negotiable rules
├── check_aws_services.sh           ← AWS status checker
└── requirements.txt                ← 38 packages
```

---

## 8. DOCS FOLDER — GENERATED REPORTS & GUIDES

| File | Size | Description |
|------|------|-------------|
| Memory_Yield_Predictor_Report.html | 4.4 MB | Main project report (7 sections, 20 embedded plots) |
| Memory_Yield_Predictor_Report.pdf | 3.7 MB | PDF version |
| Big_Data_Concepts_Guide.html/pdf | 0.6 MB | Spark, Kafka, distributed systems |
| PyTorch_Advanced_Guide.html/pdf | 0.7 MB | Mixed precision, AMP, GradScaler |
| Cloud_Infrastructure_Guide.html/pdf | 0.6 MB | AWS architecture, costs, operations |
| ENGINEERING_DECISIONS.md | — | 42 numbered decisions with reasoning |
| REASONING_INFORMATION.md | 750+ lines | Complete GPU training saga |
| ACCELERATED_TRAINING_PLAN.md | — | Training schedule (which days retrain) |
| AWS_SETUP_GUIDE.md | — | Quick-start AWS checklist |
| AWS_COMMANDS_GUIDE.md | — | Reference commands |
| AWS_INFRASTRUCTURE_GUIDE.md | — | Detailed infra documentation |
| TERMINOLOGY_GUIDE.md | — | DRAM manufacturing terms |

---

## 9. CRITICAL FACTS (DO NOT FORGET)

1. **EIP** = RDS ServiceManaged — do NOT release, no separate charge
2. **AWS auto-restarts stopped RDS after 7 days** — re-stop if waiting >7 days
3. **MacBook NOT needed** during EC2 simulation (autonomous) or Colab training (cloud)
4. **Password:** `.env.aws` has a placeholder — fill in at deployment time. If password contains `@`, URL-encode as `%40`
5. **Security group IP is dynamic** — always check with `curl -s checkip.amazonaws.com` before connecting
6. **train.py CLI:** `--full` (16M dataset), `--epochs N`, `--batch-size N`, `--run-name NAME`, `--context local|colab|airflow-retrain`
7. **CI pipeline:** ruff lint + pytest 20/20 + Docker build — Run #14+ passes
8. **Colab Pro:** SGD 14.46/month, 100 CU/month. Buy 100 CU PAYG pack for ~22 CU of T4 training.
9. **Day 1 champion already on S3:** `s3://p053-mlflow-artifacts/models/day1_champion.pt`
10. **All sensitive IDs/endpoints stored in `.env.aws`** — never hardcode in source files

---

## 10. CONTENT HUB & LINKEDIN

- **Content hub:** `aiml-content-hub/053_memory_yield_predictor/` (post.txt, comment.txt, metadata.json)
- **n8n automation:** Reads `queue_groups.json` → finds `status: "ready"` → posts carousel to LinkedIn
- **Carousel slides:** `assets/carousel/slide_01-09.png`
- **P053 LinkedIn post:** Not yet scheduled (Phase 9)
- **NOTE:** queue_groups.json must be valid JSON — accidental keystrokes break n8n

---

## 11. WORKSPACE INDEPENDENCE NOTES

This project was developed within `/Users/rajendarmuddasani/AIML/55_LinkedIn/` workspace.  
To open as standalone workspace:
- Open `/Users/rajendarmuddasani/AIML/55_LinkedIn/053_memory_yield_predictor/` directly
- All code, tests, deploy configs are self-contained
- Python venv: create a local one (`python -m venv .venv && pip install -r requirements.txt`)
- The parent workspace's `.venv` at `../. venv/` also works if path is set
- AWS CLI must be configured separately (`aws configure`)
- Colab notebooks work independently (they clone the repo)

---

## 12. CONVERSATION HISTORY — KEY DECISIONS MADE

### Session 1-5 (Prior to April 2026)
- Built entire 053 codebase (33 src modules, 20 tests, 6 notebooks)
- Day 1 A100 training on Colab (50 epochs, 201.7 min)
- Debugged bfloat16 death spiral (4 training attempts)
- CI pipeline green (ruff + pytest)
- Generated all 44 plot assets

### Session 6 (April 5-6, 2026)
- Phase 0: Wired real GPU training into Airflow DAGs (commit `aeb83e2`)
- Phase 0b: GPU selector, drift tagging, RDS auto-stop (commit `072f877`)
- AWS Steps 0-8: Account, IAM, S3, ECR, key pair, security group, RDS

### Session 7 (April 7, 2026)
- RDS stopped to save costs
- GPU quota requested → REJECTED
- Updated all 8 tracker/guide files
- Generated 4 PDFs (Big Data, PyTorch, Cloud, Main Report)
- Created Cloud Infrastructure Guide from scratch
- Created check_aws_services.sh

### Session 8 (April 8, 2026)
- Fixed queue_groups.json (`in{` → `{` typo)
- Reverted g06 to "ready" (was accidentally posted)
- GPU quota appeal submitted
- Designed + BUILT Colab fallback architecture:
  - `src/compute_backend.py` — AWS→Colab→Local fallback chain
  - `src/gpu_selector.py` — Added Colab GPU catalog (T4/A100)
  - `src/run_simulation.py` — Added `--backend` and `--checkpoint` flags
  - `notebooks/NB04_colab_training.ipynb` — Complete Colab training notebook
  - ED-043 documented, TASKS_PLAN.md updated
- Created this context file for standalone workspace use
- Commits: `6feff7a` (feat: Colab fallback), `ae0170e` (docs: ED-043)
- Decision: T4 for all training, A100 only when data >1TB/day
- Decision: SQLite for Colab/local, RDS only when AWS active

### Session 9 (April 9, 2026)
- **ROOT CAUSE FOUND** for g05 LinkedIn mismatch:
  - g05 post.txt was edited locally but NEVER committed to GitHub
  - n8n reads from GitHub → posted the old shorter version (1,641 chars)
  - Fix: pushed improved version (2,932 chars) and reset g05 to ready
  - Removed filler line "18 plots. Interactive dashboard. Full report."
  - Commits: `e851879` (push improved post.txt), `eafe62c` (remove filler)
- Updated this context file with all Colab fallback details
- n8n content source: `grouped_posts/{{ folder }}/post.txt` from GitHub API
  - **LESSON:** Always `git add + commit + push` content changes before n8n trigger

---

## 13. NEXT STEPS (Priority Order)

### ✅ ALREADY DONE (Session 8-9)
- [x] `src/compute_backend.py` — AWS→Colab→Local fallback chain
- [x] `src/gpu_selector.py` — Colab GPU catalog (T4/A100)
- [x] `src/run_simulation.py` — `--backend` and `--checkpoint` flags
- [x] `notebooks/NB04_colab_training.ipynb` — Colab training notebook
- [x] `.github/copilot-instructions.md` — This standalone context file
- [x] ED-043 documented, TASKS_PLAN.md updated
- [x] g05 post.txt fixed and pushed to GitHub
- [x] All 20/20 tests passing

### 🔜 IMMEDIATE (User Action Required)
1. **Delete duplicate g05 LinkedIn posts** (old short version posted Apr 7 + Apr 8)
2. **Trigger n8n** to repost g05 with correct improved content
3. **Buy 100 CU PAYG** on Colab (SGD 14.46 for 100 CU)
4. **Run simulation on Colab:**
   - Open `NB04_colab_training.ipynb` on Google Colab
   - Select T4 GPU runtime
   - Run cells 1-4 (`--fast` first to verify)
   - Then run cell 5 (`--full` for production)
5. **Wait for AWS GPU quota appeal** — if approved, switch back to AWS EC2

### ⏳ AFTER SIMULATION COMPLETES
6. Take simulation screenshots (drift timeline, retrain events, model versions)
7. Update dashboard with real simulation metrics
8. Update report HTML with simulation results
9. Generate PDF with Playwright
10. Update README with final results
11. Content hub post for P053
12. Final commit + tag v2.0.0
13. Stop/delete AWS resources

---

## 14. OPENING 053 AS STANDALONE WORKSPACE

### How to Open
1. In VS Code: File → Open Folder → select `053_memory_yield_predictor/`
2. This `.github/copilot-instructions.md` will auto-load as project context
3. For Python: `python -m venv .venv && pip install -r requirements.txt`
4. Or use parent venv: `source ../55_LinkedIn/.venv/bin/activate` (if still accessible)

### What's Self-Contained
- All source code in `src/` (33 modules)
- All tests in `tests/` (20/20 passing)
- All deploy configs (`deploy/docker/`, `deploy/aws/`, `deploy/k8s/`, `deploy/airflow/`)
- All notebooks in `notebooks/` (NB01-NB04)
- All documentation in `docs/` (12 files)
- All assets in `assets/` (44 PNGs + 9 carousel slides)
- `.github/copilot-instructions.md` — THIS FILE has all project context
- `never_forget_copilot_instructions.md` — Non-negotiable rules
- `TASKS_PLAN.md` — Detailed task breakdown (64/106 done)
- `MASTER_PLAN.md` — High-level execution plan

### What Needs External Access
- **AWS CLI** — must be configured (`aws configure`)
- **Colab** — open NB04 in browser, not locally
- **Content Hub** — in sibling folder `../aiml-content-hub/` (separate repo)
- **n8n Automation** — runs on n8n cloud, reads from GitHub

### Key Commands (Quick Reference)
```bash
# Run tests
python -m pytest tests/ -x -q

# Check compute backend
python -m src.compute_backend

# GPU selector decision table
python -m src.gpu_selector

# Local simulation (fast test)
python -m src.run_simulation --fast --backend local --checkpoint

# AWS status check
bash check_aws_services.sh

# Check RDS status
aws rds describe-db-instances --db-instance-identifier p053-mlflow-db --query 'DBInstances[0].DBInstanceStatus' --region us-west-2
```

---

*This context file replaces the need for conversation history. Keep it updated after EVERY session.*
