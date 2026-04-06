# 🔬 P053 — Memory Yield Predictor: End-to-End MLOps with MLflow

[![CI/CD](https://github.com/AIML-Engineering-Lab/053_dram_memory_yield_mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/AIML-Engineering-Lab/053_dram_memory_yield_mlops/actions)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.19-0194E2.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-6%20services-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Production-scale DRAM wafer yield prediction using HybridTransformerCNN on a 16M-row dataset with full MLOps pipeline, MLflow experiment tracking, and Model Registry.**

Predicts die-level failures before electrical test completion using 36 semiconductor process & test features. Demonstrates the complete MLOps lifecycle: EDA → baseline modeling → custom deep learning → MLflow experiment tracking → Model Registry → drift monitoring → 40-day production simulation → containerized deployment with Kafka + Spark + Airflow → SageMaker pipeline — the kind of end-to-end system a Principal Data Scientist / MLOps Engineer designs at Micron, Samsung, or SK Hynix.

---

## 📊 Key Results

| Model | Val AUC-PR | Test AUC-PR | Recall | Params | GPU | Training |
|-------|-----------|-------------|--------|--------|-----|----------|
| Logistic Regression | 0.0219 | — | 0.1724 | — | CPU | 2s |
| XGBoost | 0.0584 | — | 0.2759 | — | CPU | 45s |
| LightGBM | 0.0553 | — | 0.2672 | — | CPU | 12s |
| HybridTransformerCNN (T4) | 0.0524 | 0.0471 | 0.1779 | 317,633 | T4 | 33 epochs, 5h |
| **HybridTransformerCNN (A100) ✅ CHAMPION** | **0.0543** | **0.0497** | **0.1951** | **317,633** | **A100-SXM4-40GB** | **50 epochs, 201.7 min** |

> **Note:** AUC-PR is the ONLY valid metric for 1:159 class imbalance. Random baseline AUC-PR = 0.006 — our model is **9× better than random**. AUC-ROC 0.816 confirms strong discrimination.

### Day 1 Champion — Actual A100 Results (NB03, April 2026)

| Split | AUC-PR | AUC-ROC | F1 | Recall | Precision | Threshold |
|-------|--------|---------|-----|--------|-----------|-----------|
| Val | 0.0543 | 0.8157 | 0.1270 | 0.1951 | 0.0940 | 0.3555 |
| Test | 0.0497 | 0.7994 | 0.1185 | 0.1559 | 0.0926 | 0.3555 |
| Unseen | 0.0582 | 0.8148 | 0.1317 | 0.2108 | 0.0912 | 0.3555 |

> Unseen data (different seed, temporal shift) outperforms test — model generalizes beyond training distribution.

### Training Comparison: T4 vs A100

| Metric | T4 (Colab Free) | A100 (Colab Pro) | Speedup |
|--------|-----------------|------------------|---------|
| Throughput | 18,868 samples/s | 88,128 samples/s | **4.7×** |
| Epoch time | 530s | 241s (incl. full eval) | — |
| Total training | 291 min (33 ep) | 201.7 min (50 ep) | — |
| Peak VRAM | 6 GB / 14.6 GB | 1.8 GB / 40 GB | Efficient |
| AMP dtype | float16 + GradScaler | bfloat16 (no scaler) | Stable |
| Best epoch | 33 | 39 | — |

> Note: NB03 epoch time includes full 2M-row validation evaluation each epoch (correct production practice). Raw throughput was 88K samples/s.

### bfloat16 Fix Story (Engineering Decision ED-031)
- **v2 (float16):** Collapsed at epoch 5 — GradScaler scale→0 death spiral
- **v3 (float16+warmup):** Collapsed at epoch 7 — warmup delayed but didn't fix
- **v4 (bfloat16):** Stable 50 epochs, 8-bit exponent handles extreme gradients
- **ED-032 (numpy):** `torch.sigmoid(logits).cpu().numpy()` crashes bfloat16 — fix: add `.float()` before `.numpy()`

### Business Impact
- **50,000 wafers/month** × 0.62% fail rate = **310 defective wafers/month**
- **$45,000** cost per missed defect (customer return)
- A100 champion (17.3% test recall) catches 54 defects/month → **$29M annual savings**
- Estimated annual savings (benchmark model): **$748,305**

---

## 🏭 40-Day Production Lifecycle

### Phase 1 — Colab A100 (Day 1, one-time)

Train the champion model on Colab Pro A100 using `NB03_production_training.ipynb`:
- 16M rows, HybridTransformerCNN, FocalLoss, bfloat16 AMP
- Model weights → Google Drive → S3 → registered as v1 @champion

### Phase 2 — AWS EC2 (Days 2-40, fully automated)

Airflow orchestrates the entire lifecycle on EC2 with Kafka + Spark + MLflow:

| Day | Event | Automated Action |
|-----|-------|-----------------|
| 2-19 | Daily inference (5M rows/day) | Kafka → Spark ETL → predict → drift check → pass ✅ |
| 20 | Moderate drift detected (PSI > 0.10) | Retrain on AWS → v2 → canary → promote @champion |
| 21-30 | Daily inference with v2 | Drift stabilized |
| 31 | Severe drift detected (PSI > 0.20) | Retrain on AWS → v3 → canary → promote @champion |
| 32-38 | Daily inference with v3 | Monitoring |
| 39 | Canary deliberately fails | Rollback to v3, from-scratch → v4 |
| 40 | Simulation complete | Final metrics logged |

**Total:** 200M rows processed, 3 retrains, 1 rollback — zero manual intervention.

> See `notebooks/NB03_production_training.ipynb` for Day 1 training and `docs/AWS_SETUP_GUIDE.md` for deployment instructions.

---

## 📁 Project Structure

```
053_memory_yield_predictor/
├── assets/                          # 41 PNG plots (EDA, baselines, model, MLOps, A100)
├── data/
│   ├── memory_yield_full.csv        # 16M rows raw dataset
│   ├── preprocessed_full.npz        # 2.1 GB preprocessed (train/val/test/unseen)
│   ├── benchmark_a100.json          # A100 training benchmark (50 epochs)
│   ├── benchmark_t4.json            # T4 training benchmark (33 epochs)
│   ├── simulation_timeline.json     # 40-day simulation results
│   └── drift_reports/               # Per-day PSI drift reports
├── deploy/
│   ├── docker/
│   │   ├── Dockerfile               # Multi-stage build (450 MB)
│   │   ├── docker-compose.yml       # 6 services: API, Redis, Prometheus, Grafana, PostgreSQL, MLflow
│   │   ├── prometheus.yml           # Metrics scrape config
│   │   └── grafana_dashboard.json   # 8-panel monitoring dashboard
│   ├── aws/
│   │   ├── docker-compose-aws.yml   # AWS production stack (RDS + S3 + ECR)
│   │   ├── setup_aws.sh             # Provision RDS, S3, ECR (one command)
│   │   └── .env.aws.template        # Environment variables for AWS deployment
│   ├── docker-compose-bigdata.yml   # 14 services: Kafka, Spark, Airflow, LocalStack, PostgreSQL, MLflow
│   ├── k8s/
│   │   ├── deployment.yaml          # 2 replicas, rolling update, health probes
│   │   ├── hpa.yaml                 # Auto-scale 2→8 pods on CPU
│   │   ├── canary.yaml              # 90/10 canary rollout
│   │   └── ...
│   └── airflow/dags/                # 3 DAGs: daily pipeline, retrain, simulation master
├── docs/
│   ├── ENGINEERING_DECISIONS.md     # 35+ engineering decisions with interview answers
│   ├── ACCELERATED_TRAINING_PLAN.md # GPU training strategy (Colab + AWS)
│   ├── REASONING_INFORMATION.md     # Deep technical reasoning document
│   ├── Memory_Yield_Predictor_Report.html
│   └── Memory_Yield_Predictor_Report.pdf
├── notebooks/
│   ├── NB01_advanced_eda.ipynb      # 14-cell EDA with 14+ plots
│   ├── NB02_gpu_training_v4_A100.ipynb  # A100 bfloat16 production training
│   ├── NB02_gpu_training_Trained_with_T4.ipynb  # T4 baseline
│   └── NB03_production_training.ipynb  # 38-cell 4-session lifecycle (Colab A100)
├── src/
│   ├── config.py                    # Central config (features, params, MLflow, drift)
│   ├── data_generator.py            # 16M-row synthetic data with 10 quality issues
│   ├── preprocess.py                # Full preprocessing pipeline
│   ├── model.py                     # HybridTransformerCNN architecture
│   ├── train.py                     # MLflow-integrated training script (CLI)
│   ├── train_baseline.py            # LogReg + XGBoost + LightGBM baselines
│   ├── focal_loss.py                # FocalLoss (NumPy + PyTorch, NaN-safe)
│   ├── mlflow_utils.py              # MLflow experiment tracking & Model Registry
│   ├── retrolog_experiments.py      # Import completed runs into MLflow
│   ├── run_simulation.py            # 40-day production simulation with MLflow
│   ├── streaming_data_generator.py  # 14-scenario drift data generator
│   ├── drift_detector.py            # PSI + KL + KS triple drift detection
│   ├── spark_drift_detector.py      # Distributed drift detection (PySpark)
│   ├── spark_etl.py                 # Spark ETL pipeline
│   ├── retrain_trigger.py           # 3-criteria retrain gate
│   ├── serve.py                     # FastAPI (Prometheus integrated)
│   ├── inference.py                 # YieldPredictor class (batch + single)
│   ├── sagemaker_pipeline.py        # 9-step conditional SageMaker Pipeline
│   ├── kafka_producer.py            # Streaming data to Kafka topics
│   ├── kafka_consumer.py            # Real-time inference consumer
│   └── artifacts/                   # Model weights (A100, T4)
├── tests/
│   ├── test_core.py                 # 12 tests (model, focal loss, drift, config)
│   └── test_serve.py                # 8 tests (API endpoints, health, batch)
├── web/
│   └── dashboard.html               # Plotly.js dark-theme interactive dashboard
├── .github/workflows/ci.yml         # CI/CD: lint → test → Docker build → K8s deploy
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔧 Advanced Concepts Demonstrated

- **MLflow Experiment Tracking + Model Registry** — PostgreSQL-backed (not SQLite), S3 artifact storage, champion/baseline aliases
- **3-Tier Deployment Architecture** — Same codebase, env-var-driven: SQLite (local) → PostgreSQL+Docker → RDS+S3 (AWS)
- **AWS Production Stack** — RDS PostgreSQL, S3 artifacts, ECR container registry, setup script + compose
- **bfloat16 vs float16 debugging** — Diagnosed GradScaler death spiral on A100 (cc≥8.0), documented 3 failed runs + root cause
- **Extreme class imbalance handling** (1:159) with Focal Loss, AUC-PR metric selection, and threshold optimization
- **Custom hybrid architecture** — Transformer for tabular + CNN for spatial + Fusion MLP
- **40-day production simulation** — 14 drift scenarios, 3-criteria retrain gate, MLflow-logged retrain events
- **Big data stack** — Kafka streaming, Spark ETL, Airflow DAGs, LocalStack S3 emulation (14 Docker services)
- **3-tier GPU strategy** — MPS (pain benchmark) → T4 (baseline) → A100 (production)
- **Triple drift detection** — PSI + KL divergence + KS statistic across 36 features
- **Full Kubernetes deployment** — HPA auto-scaling, canary rollout, health probes
- **SageMaker Pipeline** — 9-step conditional pipeline with evaluation gate
- **Cost-aware engineering** — $45K/false negative vs $120/false positive, AWS cost estimates ($55-135/month)
- **35+ documented engineering decisions** — Each with rationale, alternatives, and interview answer
- **Enterprise-ready handoff** — AMD/Micron provides AWS account → change env vars → deploy (zero code changes)

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Data["Data Pipeline"]
        DG[Data Generator<br/>16M rows × 36 features] --> PP[Preprocessing<br/>Missing values, scaling,<br/>encoding, engineering]
        PP --> NPZ[preprocessed_full.npz<br/>2.1 GB]
    end

    subgraph Training["Model Training"]
        NPZ --> BL[Baselines<br/>LogReg, XGBoost, LightGBM]
        NPZ --> HM[HybridTransformerCNN<br/>317K params]
        HM --> |Tabular 33 features| TR[Transformer<br/>2 layers, 4 heads]
        HM --> |Spatial 3 features| CN[Conv1D<br/>64 channels]
        TR --> FU[Fusion MLP]
        CN --> FU
        FU --> FL[Focal Loss<br/>α=0.75, γ=2.0]
    end

    subgraph MLflow["MLflow + PostgreSQL"]
        FL --> ML[MLflow Runs<br/>Params, metrics, artifacts]
        ML --> PG[(PostgreSQL<br/>Backend Store)]
        ML --> S3[S3 Artifacts<br/>Model weights, plots]
        ML --> MR[Model Registry<br/>Champion / Baseline aliases]
        ML --> UI[MLflow UI<br/>localhost:5001]
    end

    subgraph Simulation["40-Day Production Simulation"]
        SG[Streaming Generator<br/>14 drift scenarios] --> DD[Drift Detector<br/>PSI + KL + KS]
        DD --> RG[Retrain Gate<br/>3-criteria decision]
        RG --> |retrain trigger| ML
    end

    subgraph BigData["Big Data Stack"]
        KF[Kafka<br/>Streaming ingestion] --> SP[Spark ETL<br/>Distributed processing]
        SP --> AF[Airflow DAGs<br/>Orchestration]
        LS[LocalStack S3<br/>AWS emulation] --> SP
    end

    subgraph Serving["Production Serving"]
        MR --> FA[FastAPI<br/>Prometheus metrics]
        FA --> DK[Docker Compose<br/>6 services + PostgreSQL]
        DK --> K8[Kubernetes<br/>HPA 2→8 pods]
        K8 --> PM[Prometheus + Grafana]
    end

    subgraph AWS["AWS Production"]
        DK -.-> RDS[(RDS PostgreSQL<br/>MLflow backend)]
        DK -.-> S3B[S3 Bucket<br/>Artifacts + models]
        DK -.-> ECR[ECR<br/>Container images]
    end

    subgraph GPU["GPU Strategy"]
        MPS[MPS M2 Pro<br/>3,200 s/s ❌] -.->|Pain point| T4[T4 Colab Free<br/>18,868 s/s]
        T4 -.->|Production| A100[A100 Colab Pro<br/>88,000 s/s ✓]
    end

    style Data fill:#1e3a5f,color:#fff
    style Training fill:#1e3a5f,color:#fff
    style MLflow fill:#7c2d12,color:#fff
    style Simulation fill:#065f46,color:#fff
    style BigData fill:#4c1d95,color:#fff
    style Serving fill:#065f46,color:#fff
    style AWS fill:#92400e,color:#fff
    style GPU fill:#4c1d95,color:#fff
```

---

## ⚙️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12+ | Core language |
| PyTorch | 2.3+ | HybridTransformerCNN (MPS + CUDA) |
| **MLflow** | **3.10+** | **Experiment tracking, Model Registry, PostgreSQL backend** |
| **PostgreSQL** | **16** | **MLflow backend store (Docker/AWS), replaces SQLite** |
| XGBoost | 2.0+ | Gradient boosting baseline |
| LightGBM | 4.0+ | Gradient boosting baseline |
| scikit-learn | 1.4+ | Preprocessing, metrics |
| FastAPI | 0.110+ | Model serving (Prometheus integrated) |
| Docker | 24.x | Multi-stage containerization (6+ services) |
| Kubernetes | 1.28+ | Production orchestration + HPA + canary |
| Apache Kafka | 3.6+ | Streaming data ingestion |
| Apache Spark | 3.5+ | Distributed ETL & drift detection |
| Apache Airflow | 2.8+ | Workflow orchestration (3 DAGs) |
| **AWS RDS** | **PostgreSQL 16** | **Managed MLflow backend (production)** |
| **AWS S3** | **—** | **MLflow artifact storage (model weights, plots)** |
| **AWS ECR** | **—** | **Container image registry** |
| AWS SageMaker | 2.x | 9-step training pipeline |
| LocalStack | 3.2 | AWS S3/SageMaker emulation (local Docker) |
| Prometheus | 2.x | Metrics collection |
| Grafana | 10.x | Monitoring dashboard |
| Google Colab Pro | — | A100 GPU training (bfloat16) |

---

## 🚀 Getting Started

### Local Development (SQLite backend — zero dependencies)
```bash
# Clone
git clone https://github.com/AIML-Engineering-Lab/053_memory_yield_predictor.git
cd 053_memory_yield_predictor

# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate 16M-row dataset (~5 minutes)
python src/data_generator.py

# Preprocess
python src/preprocess.py --full

# Train baselines (CPU, ~2 minutes)
python src/train_baseline.py

# MLflow-integrated training (auto-detects GPU)
python -m src.train --full --batch-size 4096

# Retro-log T4/A100 experiments (if you have benchmark JSONs)
python -m src.retrolog_experiments

# MLflow UI — view all experiments
mlflow ui --backend-store-uri "sqlite:///mlflow.db" --port 5001
# Open http://localhost:5001

# 40-day simulation (fast mode ~10 min)
python -m src.run_simulation --fast

# Serve locally
python src/serve.py
```

### Docker Stack (PostgreSQL backend — production-grade)
```bash
# Full stack: API + Redis + Prometheus + Grafana + PostgreSQL + MLflow
cd deploy/docker && docker compose up -d

# Open MLflow UI:     http://localhost:5001
# Open Grafana:       http://localhost:3000  (admin/admin)
# Open FastAPI docs:  http://localhost:8000/docs

# Point your training script at Docker MLflow:
export MLFLOW_TRACKING_URI="postgresql://mlflow:mlflow@localhost:5432/mlflow"
python -m src.train --full --batch-size 4096
```

### AWS Production (RDS + S3 — enterprise deployment)
```bash
# Provision AWS resources (RDS PostgreSQL, S3 bucket, ECR)
./deploy/aws/setup_aws.sh

# Fill in your AWS details
cp deploy/aws/.env.aws.template deploy/aws/.env
# Edit deploy/aws/.env with RDS endpoint, S3 bucket, ECR URI

# Deploy on EC2/ECS
docker compose -f deploy/aws/docker-compose-aws.yml --env-file deploy/aws/.env up -d

# Point Colab A100 training at AWS MLflow:
# export MLFLOW_TRACKING_URI="postgresql://mlflow:<pw>@<rds-host>:5432/mlflow"
```

### Tests
```bash
pytest tests/ -v  # 20 tests
```

---

## 📐 Engineering Decisions

See [docs/ENGINEERING_DECISIONS.md](docs/ENGINEERING_DECISIONS.md) for 35+ documented decisions including:
- ED-001: Why AUC-PR over ROC-AUC
- ED-003: Why Focal Loss over class weights
- ED-007: Why Transformer + CNN hybrid architecture
- ED-011: Why checkpoint resume is mandatory for cloud training
- ED-015: Why triple drift detection (PSI + KL + KS)
- ED-019: Why HPA + canary over blue-green deployment
- ED-031: Why bfloat16 on A100 (float16 GradScaler death spiral)
- ED-032: Why MLflow over Weights & Biases / Neptune
- ED-033: Why retro-log failed runs (interview story generation)
- ED-034: Why PostgreSQL over SQLite for MLflow backend
- ED-035: Why S3 for MLflow artifacts (not local filesystem)

See [docs/ACCELERATED_TRAINING_PLAN.md](docs/ACCELERATED_TRAINING_PLAN.md) for GPU training strategy and AWS cost estimates.

---

*AIML Engineering Lab · MIT License*
