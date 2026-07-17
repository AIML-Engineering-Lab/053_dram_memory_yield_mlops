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
| P5 AWS Infra (Code) | 20/20 | 0 | ✅ EC2/RDS/S3/GitHub Actions production path used; EC2/RDS stopped after run |
| P6 GPU Training | 4/4 | 0 | ✅ Day 1 A100 champion; Day 30 v2 retrain promoted; Day 39 rollback path verified |
| P7 40-Day Sim | 13/13 | 0 | ✅ Intended Day 1-40 complete; S3 now `current_day=46`, `last_completed_day=45`, `status=complete` after Day 41-45 schedule leak |
| P8 Report | 4/6 | 2 | Final HTML/report refreshed with July 17 schedule leak fix; PDF and optional screenshots still pending |
| P9 Content | 3/4 | 1 | User LinkedIn review |
| P10 Polish | 5/8 | 3 | AWS stopped; final commit/push/tag and residual-cost deletion decision remain |
| **Total** | **92/106** | **14** | **No execution blocker; only archive/publish decisions remain** |

## Final Closure Snapshot — 2026-07-17

| Check | Status |
|---|---|
| 40-day live run | ✅ Intended Day 1-40 complete. S3 `pipeline_state.json`: `current_day=46`, `last_completed_day=45`, `status=complete`, `last_run=2026-07-16T02:05:57Z` after unintended Day 41-45 scheduled runs. |
| Champion model | ✅ `s3://p053-mlflow-artifacts/models/day30_v2_retrained.pt`; `champion_updated_day=30`. |
| Latest scheduled runs | ✅ Days 34-40 completed successfully; Days 41-45 accidentally continued through GitHub Actions and are now blocked by disabling cron plus `should_run` guards. |
| Artifacts | ✅ Days 29-45 have current production Parquet, drift reports, and summaries; Days 41-45 are extra retained artifacts. |
| AWS shutdown | ✅ EC2 `g4dn.xlarge` stopped; RDS `db.t3.micro` stopped; no NAT gateways available or pending. |
| Cost caveat | ⚠️ Compute is stopped, but exact `$0` ongoing bill is not guaranteed. Residual inventory: 125 GiB gp3 EBS, 20 GiB RDS, 21 automated RDS snapshots, ~4.80 GB current S3 data with 387 versions, 1 ECR repo, and 1 associated public IPv4/EIP. User chose to keep minimal evidence resources for now. |

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

## 40-Day Production Timeline (Final Actual State)

```
Day 1      ┃ P6: Day 1 champion trained on Colab A100, uploaded to S3
Day 2-29   ┃ P7: Scheduled AWS daily pipeline runs inference, ETL, drift checks, and artifact sync
Day 30     ┃ P7: Drift/staleness gate opens → v2 retrain promoted to champion
Day 31     ┃ P7: Day 31 should use Day 30 v2 champion; recovered Spark memory and disk pressure issues
Day 32     ┃ P7: Fixed accidental retrain wait by adding champion_updated_day cooldown gate
Day 33     ┃ P7: Fixed Kafka heap/restart risk and completed recovery run
Day 34-38  ┃ P7: Normal scheduled AWS daily runs complete successfully
Day 39     ┃ P7: Canary failure/rollback path verified; champion remains v2
Day 40     ┃ P7: Final intended scheduled run completes; S3 state moves to current_day=41/status=complete
Day 41-45  ┃ Incident: scheduled workflow continued accidentally; real extra artifacts were produced
Jul 17     ┃ P10: Daily cron disabled and AWS-starting steps guarded; EC2 stopped, RDS stopped, NAT none
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
| 61 | **Launch/use g4dn.xlarge EC2 instance** | Auto/User | ✅ launched, used for live AWS daily run, now stopped |
| 62 | **RDS PostgreSQL provisioned** | User | ✅ p053-mlflow-db used for run window, now stopped |
| 61b | **Colab fallback: compute_backend.py** | Copilot | ✅ AWS→Colab→Local chain |
| 61c | **Colab fallback: NB04_colab_training.ipynb** | Copilot | ✅ T4/A100 with checkpoint |

**Final cost posture:** EC2 and RDS compute are stopped and no NAT gateways are active. Ongoing cost is not guaranteed to be exactly `$0`. Retained inventory kept for now: 125 GiB gp3 EBS, 20 GiB RDS, 21 automated RDS snapshots, ~4.80 GB current S3 data with 387 versions, 1 ECR repo, and 1 associated public IPv4/EIP.

### P6: GPU Training (Complete)

| # | Task | Who | Status |
|---|------|-----|--------|
| 63 | **Day 1 initial training** (Colab A100, 50ep, 201.7min) | User | ✅ AUC-ROC=0.816, AUC-PR=0.054 |
| 64 | **Day 30 drift retrain** (v2 promoted) | Auto | ✅ `day30_v2_retrained.pt` champion |
| 65 | **Day 31 v2 champion validation** | Auto + Copilot audit | ✅ Day 31+ uses Day 30 champion |
| 66 | **Day 39 canary/rollback path** | Auto | ✅ Bad model rejected; v2 retained |

**Key change:** Retrains are FULLY AUTOMATED via `dag_retrain_pipeline.py` on AWS g4dn.xlarge.
If AWS unavailable, retrains run on Colab T4 via NB04 with `--checkpoint` for disconnect resilience.
`_execute_gpu_training()` calls `train.py` via subprocess on T4 GPU.

### P7: 40-Day Production Simulation (Complete)

| # | Task | Who | Est. |
|---|------|-----|------|
| 79 | EC2 orchestration via GitHub Actions SSH | Auto | ✅ |
| 80 | Repo sync to EC2 | Auto | ✅ |
| 81 | AWS Docker/Python stack available | Auto | ✅ |
| 82 | Recovery/sanity runs for Day 31-33 | Both | ✅ |
| 83 | Verify champion/drift state in S3 | Copilot | ✅ |
| 84 | Days 34-40 scheduled daily runs | Auto | ✅ |
| 85 | Verify latest GitHub Actions runs | Copilot | ✅ |
| 86 | Verify S3 artifact prefixes | Copilot | ✅ |
| 87 | Final Day 40 completion | Auto | ✅ |
| 88 | Verify model artifacts | Copilot | ✅ |
| 89 | Regenerate learning report HTML | Copilot | ✅ |
| 90 | Capture CLI evidence | Copilot | ✅ |
| 91 | **STOP EC2/RDS/NAT** | Auto + Copilot audit | ✅ |

### P8: Report & Dashboard (Finalization)

| # | Task | Who | Est. |
|---|------|-----|------|
| 92 | Update dashboard with AWS metrics + drift timeline | Copilot | ⏳ left untouched because file already has unrelated local edits |
| 93 | Add AWS screenshots to assets/ | Both | 5 min |
| 94 | Update final learning report HTML | Copilot | ✅ |
| 95 | Regenerate PDF | Copilot | ⏳ |
| 96 | Update README final status | Copilot | ✅ |
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
| 108 | Stop/delete AWS resources | User | ✅ stopped; deletion decision pending for retained evidence |
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
