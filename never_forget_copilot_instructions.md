# P053 — NEVER FORGET COPILOT INSTRUCTIONS

> **⚠️ ATTACH THIS FILE TO EVERY CONVERSATION ABOUT P053.**
> **If you contradict anything here, the user will rightfully call you out.**

---

## NON-NEGOTIABLE RULES

1. **AWS IS PRIMARY. Colab is FALLBACK only when AWS GPU is unavailable.**
2. **GPU ALWAYS. g4dn.xlarge (T4 GPU) for ALL retraining. Colab T4 as fallback. A100 only when data >1TB/day.**
3. **Full day's data for retraining. NOT a subset. NOT a sample. The ENTIRE day's volume.**
4. **Budget: $1000 SGD (~$740 USD). PLAN BIG, DO BIG. No toy projects.**
5. **All data, models, artifacts stored on S3. Nothing on MacBook.**
6. **Target: Principal Data Scientist / Principal GenAI Engineer at AMD, NVIDIA, Micron, Qualcomm.**
7. **Document WHY: GPU vs CPU, Spark vs Pandas, big data vs toy data — for 10K+ LinkedIn followers.**
8. **Auto-stop EC2 + RDS + NAT Gateway after Phase 3 completes (all resources cleaned up).**
9. **CloudWatch billing alarm at $200 USD (safety net).**
10. **Low-data drift is TAGGED for transparency (MLflow) but NEVER triggers retraining.**
11. **Auto GPU selector: T4 for <50M params, A100 for >1TB/day data (src/gpu_selector.py).**
12. **Compute backend fallback: AWS EC2 → Google Colab → Local MacBook (src/compute_backend.py).**
13. **MLflow: RDS when AWS active, SQLite when Colab/local. Colab runs registered in RDS later.**

---

## ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                    AWS g4dn.xlarge (T4 GPU)                 │
│                 4 vCPU · 16 GB RAM · T4 16 GB               │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Airflow  │  │  Kafka   │  │  Spark   │  │   MLflow   │  │
│  │ (DAGs)   │  │ (stream) │  │  (ETL)   │  │ (tracking) │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘  │
│       │              │             │               │         │
│       ▼              ▼             ▼               ▼         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              PyTorch + T4 GPU Training                │   │
│  │         (Real train.py, NOT _simulate_training)       │   │
│  └──────────────────────────────────────────────────────┘   │
│       │              │             │               │         │
│       ▼              ▼             ▼               ▼         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │Prometheus│  │ Grafana  │  │ FastAPI  │  │   Redis    │  │
│  │(metrics) │  │ (viz)    │  │ (serve)  │  │  (cache)   │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │   S3     │ │   RDS    │ │   ECR    │
    │(artifacts│ │(Postgres │ │ (Docker  │
    │ models   │ │  MLflow) │ │  images) │
    │ Parquet) │ │          │ │          │
    └──────────┘ └──────────┘ └──────────┘
```

---

## INSTANCE STRATEGY

| Instance | Role | Cost/hr | When |
|----------|------|---------|------|
| **g4dn.xlarge** | Full stack + GPU retrain | $0.526 | Phase 2 + 3 |
| **RDS db.t3.micro** | MLflow PostgreSQL | $0.018 | During simulation (auto-stops after) |
| **S3** | ALL artifacts | $0.023/GB/mo | Always |
| **CloudWatch** | Billing alarm | $0 | $200 USD threshold |

**Why g4dn.xlarge for everything?** Real production teams don't switch instances mid-pipeline. One GPU-capable box runs the entire stack. Inference is batch (Spark), retraining is GPU (PyTorch on T4). This is how AMD/Micron operate — dedicated ML pipeline servers.

---

## PHASE 0 — CODE FIXES (Copilot, before AWS spend)

**Status: ✅ COMPLETED — committed as aeb83e2, pushed to GitHub**

The honest truth: Several critical components are FAKE right now.

| # | Task | What's Wrong | Fix |
|---|------|-------------|-----|
| 0.1 | Wire `train.py` into retrain DAG | `_simulate_training()` returns **hardcoded 0.0485 AUC-PR** | Replace with real `subprocess.run(["python", "-m", "src.train", ...])` calling real GPU training |
| 0.2 | Wire `train.py` into `run_simulation.py` | `_log_retrain_to_mlflow()` logs **metadata only, zero training** | Replace with actual `train.py` execution |
| 0.3 | Add boto3 S3 artifact upload | **Zero boto3 code in entire codebase** | Models → `s3://p053-mlflow-artifacts/models/`, plots → `s3://…/plots/`, Parquet → `s3://…/data/` |
| 0.4 | Update docker-compose for real AWS | Uses **LocalStack (fake S3)** | Point to real `s3://p053-mlflow-artifacts` + real RDS endpoint |
| 0.5 | EC2 bootstrap user-data script | **No EC2 setup automation** | Auto-install Docker, clone repo, start stack on instance launch |
| 0.6 | MLflow → real RDS PostgreSQL | **Not configured** | `MLFLOW_TRACKING_URI=postgresql://mlflow:<pw>@<rds-host>:5432/mlflow` |
| 0.7 | Auto-stop EC2 after Phase 3 | **No shutdown mechanism** | CloudWatch alarm on CPU < 5% for 30 min → stop instance, OR shutdown script at end of DAG |
| 0.8 | Local dry-run test (3 epochs) | Verify full pipeline end-to-end | Run locally with MPS/CPU, 1000 rows, confirm train→MLflow→S3 |
| 0.9 | Push + verify CI passes | — | Clean commit, all green |

---

## PHASE 1 — DAY 1 INITIAL TRAINING ✅ DONE

| Metric | Value |
|--------|-------|
| GPU | NVIDIA A100-SXM4-40GB (Colab Pro) |
| Data | 16M rows, preprocessed |
| Epochs | 50 |
| Time | 201.7 min |
| Val AUC-ROC | 0.8157 |
| Val AUC-PR | 0.0543 |
| Val F1 | 0.127 |
| Best Epoch | 39 |
| Model | `s3://p053-mlflow-artifacts/models/day1_champion.pt` (1.29 MB) |
| AMP | bfloat16 |

---

## PHASE 2 — ACCELERATED 40-DAY SIMULATION (g4dn.xlarge)

**Status: ⬜ NOT STARTED — Blocked on Phase 0**

### Data Volume Per Day

| Day Range | Volume/Day | GB/Day | Scenario | Key Event |
|-----------|-----------|--------|----------|-----------|
| 1-8 | 3-5M rows | 0.1-0.3 | Steady baseline | Reference window builds |
| 9-10 | 4-5M | 0.2 | False alarm | Auto-recover, PSI spike then normalize |
| 11-18 | 3-6M | 0.1-0.3 | Gradual drift | Chamber seasoning effect |
| 19-20 | 3-6M | 0.2-0.3 | Sudden shift | Threshold breach, 3× retrain criteria |
| 21-30 | 2-5M | 0.1-0.3 | Worsening | Staleness gate blocks premature retrain |
| **31** | **5M** | **0.3** | **RETRAIN** | **GPU training fires on T4, 30 epochs, full day data** |
| 32-38 | 3-6M | 0.2-0.3 | Post-retrain | Recovery + 2nd drift cycle |
| **39** | **5M** | **0.2** | **BAD DEPLOY** | **Canary catches → automatic ROLLBACK** |
| 40 | 6M | 0.3 | Recovery | Confirmed stable |
| **TOTAL** | **~170M** | **~8 GB** | | |

### Execution Steps

| # | Task | Who | Time | Cost |
|---|------|-----|------|------|
| 2.1 | Launch g4dn.xlarge EC2 | You (1 cmd) | 3 min | $0.526/hr starts |
| 2.2 | Create RDS PostgreSQL | You (1 cmd) | 10 min | $0.018/hr starts |
| 2.3 | SSH into EC2 | You | 2 min | $0 |
| 2.4 | Run bootstrap (Docker + deps) | Both | 15 min | $0 |
| 2.5 | Clone repo + configure .env | Both | 5 min | $0 |
| 2.6 | `docker compose up -d` full stack | Both | 10 min | $0 |
| 2.7 | Verify: Airflow/MLflow/Grafana UIs | You | 5 min | $0 |
| 2.8 | Register Day 1 model in MLflow | Both | 2 min | $0 |
| 2.9 | Trigger `dag_simulation_master` Phase 2 | You (1 click) | 1 min | $0 |
| 2.10 | Days 1-40 run automatically | Auto | 3-4 hrs | $0 |
| 2.11 | Take screenshots (MLflow/Grafana/Airflow/S3) | You | 15 min | $0 |
| 2.12 | Export metrics from MLflow | Both | 5 min | $0 |
| 2.13 | **STOP EC2 + RDS** | **You** | 2 min | saves $ |
| | **TOTAL** | | **~6 hrs** | **~$4 USD** |

### What Runs Each "Day" on EC2 (Automatic)

```
Airflow dag_simulation_master triggers dag_daily_yield_pipeline:
  1. streaming_data_generator → variable rows (2-9M)
  2. Kafka producer → topic: yield-predictions
  3. Spark ETL → clean + feature engineering + Parquet → S3
  4. Model inference (batch) → predictions + confidence scores
  5. Drift detector → PSI per feature + KL divergence
  6. MLflow → log metrics, drift scores, alerts
  7. [IF drift + staleness gate met]:
     → dag_retrain_pipeline triggers:
       a. Spark ETL on training window (full day data)
       b. train.py on T4 GPU (30 epochs, real training)
       c. MLflow → log model, metrics, artifacts → S3
       d. Canary evaluation (new vs old on holdout)
       e. [IF canary passes] → Promote to champion
       f. [IF canary fails] → ROLLBACK to previous champion
```

---

## ☕ BREAK — REVIEW PHASE 2 RESULTS

After Phase 2 completes:
1. Stop EC2 + RDS (save money)
2. Review MLflow: all runs, model versions, drift timeline
3. Review Grafana: throughput, latency, retrain events
4. Review S3: artifacts, Parquet structure, model registry
5. Identify issues / tuning needed before Phase 3
6. Decide: any architecture changes needed?

---

## PHASE 3 — PRODUCTION STRESS TEST (g4dn.xlarge, FULL 40 DAYS, TBs)

**Status: ⬜ NOT STARTED — Blocked on Phase 2 review**

### Data Volume Per Day (PRODUCTION SCALE)

| Day Range | Volume/Day | GB/Day | Scenario | Key Event |
|-----------|-----------|--------|----------|-----------|
| 1-8 | 30-100M rows | 2-3 | Steady baseline | Spark partition tuning proven |
| 9-10 | 80-100M | 2-3 | False alarm at scale | Memory pressure test |
| 11-18 | 50-120M | 2-4 | Gradual drift at scale | I/O throughput test |
| 19-20 | 80-140M | 3-4 | Sudden shift at scale | Kafka backpressure test |
| 21-30 | 50-150M | 2-4 | Worsening at scale | Disk management test |
| **31** | **200M** | **6** | **RETRAIN at prod scale** | **T4 GPU on 200M rows, real training** |
| 32-38 | 80-250M | 3-7 | Recovery at scale | S3 write throughput test |
| **39** | **200M** | **6** | **BAD DEPLOY at scale** | **Canary detects → ROLLBACK** |
| 40 | 350M | 9 | Final recovery | Peak volume test |
| **TOTAL** | **~5 BILLION** | **~247 GB** | | **TBs across S3** |

### Execution Steps

| # | Task | Who | Time | Cost |
|---|------|-----|------|------|
| 3.1 | Restart EC2 (same g4dn.xlarge) | You (1 cmd) | 2 min | $0.526/hr starts |
| 3.2 | Restart RDS | You (1 cmd) | 5 min | $0.018/hr starts |
| 3.3 | Reconfigure for Phase 3 scale | Both (1 cmd) | 5 min | $0 |
| 3.4 | Trigger `dag_simulation_master` Phase 3 | You (1 click) | 1 min | $0 |
| 3.5 | Days 1-40 run automatically | Auto | ~50 hrs | $0 |
| 3.6 | Take screenshots | You (SSH back) | 15 min | $0 |
| 3.7 | Export Phase 3 metrics | Both | 10 min | $0 |
| 3.8 | **EC2 AUTO-STOPS** (CloudWatch/Lambda) | Auto | 0 min | saves $ |
| | **TOTAL** | | **~55 hrs** | **~$30 USD** |

### What Phase 3 Proves (Interview Gold)

| Question Interviewer Asks | Your Answer |
|--------------------------|-------------|
| "How much data have you processed?" | "5 billion rows, 247 GB Parquet, across 40 production days" |
| "Why Spark and not Pandas?" | "Pandas OOM at 10M rows. We process 200M rows/day. Spark handles it in 45 min." |
| "Why GPU for retraining?" | "CPU takes 6 hours. T4 GPU does it in 40 minutes. At $0.53/retrain, GPU is 10x cheaper per compute-hour." |
| "How do you handle model drift?" | "PSI per feature, 3-criterion gate (drift + performance + staleness), automated Airflow retrain" |
| "What happens when a bad model deploys?" | "Day 39: canary evaluation caught 12% AUC-PR degradation, automatically rolled back in 3 minutes" |
| "How do you manage costs?" | "g4dn.xlarge spot instances, auto-stop on idle, S3 lifecycle policies. 40-day sim cost: $30" |
| "What's your MLOps stack?" | "Airflow orchestration, Kafka streaming, Spark ETL, MLflow tracking, Prometheus/Grafana monitoring, S3 artifact store, Docker-compose on EC2" |

---

## PHASE 4 — REPORT, DASHBOARD, CONTENT (Copilot, post-simulation)

**Status: ⬜ NOT STARTED — Blocked on Phase 3**

| # | Task | Who | Time |
|---|------|-----|------|
| 4.1 | Generate all simulation plots from MLflow data | Copilot | 1 hr |
| 4.2 | Update `docs/Report.html` with real AWS metrics | Copilot | 2 hrs |
| 4.3 | Generate PDF from HTML (Playwright) | Copilot | 10 min |
| 4.4 | Update `web/dashboard.html` with real Plotly charts | Copilot | 1 hr |
| 4.5 | Write "Why GPU not CPU" section with benchmarks | Copilot | 30 min |
| 4.6 | Write "Why Spark not Pandas" section with OOM proof | Copilot | 30 min |
| 4.7 | Write "5 Billion Rows: Lessons Learned" section | Copilot | 30 min |
| 4.8 | Update README with final architecture + results | Copilot | 30 min |
| 4.9 | LinkedIn post + first comment | Copilot | 30 min |
| 4.10 | Content-hub entry | Copilot | 15 min |
| 4.11 | Final commit + `git tag v2.0.0` + push | Copilot | 5 min |

---

## COST SUMMARY

| Item | Phase 2 | Phase 3 | Total |
|------|---------|---------|-------|
| g4dn.xlarge ($0.526/hr) | 6 hrs = $3.16 | 55 hrs = $28.93 | **$32.09** |
| RDS db.t3.micro ($0.018/hr) | 6 hrs = $0.11 | 55 hrs = $0.99 | **$1.10** |
| S3 storage (8 + 247 GB) | $0.18 | $5.68 | **$5.86** |
| S3 requests | $0.50 | $2.00 | **$2.50** |
| Data transfer | $0.50 | $2.00 | **$2.50** |
| **TOTAL** | **$4.45** | **$39.60** | **~$44 USD (~$60 SGD)** |

**Budget remaining: $940 SGD** — enough across for 15x reruns if needed.

---

## WHAT'S ALREADY REAL vs WHAT PHASE 0 MUST FIX

| Component | Currently | After Phase 0 |
|-----------|-----------|---------------|
| `_simulate_training()` | Returns hardcoded 0.0485 | Calls real `train.py` on GPU |
| `_log_retrain_to_mlflow()` | Metadata only | Real training + S3 upload |
| S3 integration | Zero boto3 | Full artifact lifecycle |
| docker-compose | LocalStack (fake) | Real AWS endpoints |
| EC2 bootstrap | Manual SSH | Automated user-data script |
| MLflow backend | SQLite (local) | RDS PostgreSQL |
| Auto-stop | None | CloudWatch alarm + shutdown |

---

## LEARNINGS TO DOCUMENT (for LinkedIn followers)

Throughout all phases, capture and document:

1. **Why GPU not CPU** — Show actual training time comparison. T4 = 40 min. CPU = 6 hrs. Same data.
2. **Why Spark not Pandas** — Show pandas `MemoryError` on 50M rows. Spark processes 200M in 45 min.
3. **Why Kafka not batch files** — Show streaming latency vs batch delay. Real-time drift detection.
4. **Why Airflow not cron** — Show DAG dependency management, retry logic, SLA monitoring.
5. **Why MLflow not manual tracking** — Show experiment comparison, model registry, artifact versioning.
6. **Why Docker not bare metal** — Show reproducibility, one-command deployment, environment isolation.
7. **Why S3 not local disk** — Show durability, cross-instance access, lifecycle policies.
8. **Why PSI not accuracy-only** — Show feature-level drift detection catches problems BEFORE accuracy drops.

---

*Created: 2026-04-07*
*Target: AMD, NVIDIA, Micron, Qualcomm — Principal Engineer level*
*Budget: $1000 SGD | Data: 5B+ rows | Stack: Full AWS MLOps*
