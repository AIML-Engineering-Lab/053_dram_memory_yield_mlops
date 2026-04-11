# DRAM Memory Yield Predictor — End-to-End MLOps Pipeline

[![CI/CD](https://github.com/AIML-Engineering-Lab/053_dram_memory_yield_mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/AIML-Engineering-Lab/053_dram_memory_yield_mlops/actions)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.19-0194E2.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-6%20services-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Production-scale DRAM wafer yield prediction using a HybridTransformerCNN on 16M rows with full MLOps lifecycle: drift detection, automated retraining, canary deployment, and rollback — running on A100 GPU with bfloat16.**

Predicts die-level failures before electrical test completion using 36 semiconductor process and test features. The complete MLOps lifecycle is demonstrated end-to-end: EDA, baseline modeling, custom deep learning, MLflow experiment tracking, Model Registry, drift monitoring, a 40-day production simulation (200M rows), containerized deployment with Kafka + Spark + Airflow, and Kubernetes orchestration.

---

## Key Results

| Model | Val AUC-PR | Test AUC-PR | Recall | Params | GPU | Training |
|-------|-----------|-------------|--------|--------|-----|----------|
| Logistic Regression | 0.0219 | — | 0.1724 | — | CPU | 2s |
| XGBoost | 0.0584 | — | 0.2759 | — | CPU | 45s |
| LightGBM | 0.0553 | — | 0.2672 | — | CPU | 12s |
| HybridTransformerCNN (T4) | 0.0524 | 0.0471 | 0.1779 | 317,633 | T4 | 33 ep, 5h |
| **HybridTransformerCNN (A100)** | **0.0543** | **0.0497** | **0.1951** | **317,633** | **A100-SXM4-40GB** | **50 ep, 201.7 min** |

> AUC-PR is the correct metric at 1:159 class imbalance. Random baseline AUC-PR = 0.006 — the champion model is **9x better** than random. AUC-ROC = 0.816 confirms strong discrimination.

### A100 Champion (Day 1)

| Split | AUC-PR | AUC-ROC | F1 | Recall | Precision |
|-------|--------|---------|-----|--------|-----------|
| Val | 0.0543 | 0.8157 | 0.1270 | 0.1951 | 0.0940 |
| Test | 0.0497 | 0.7994 | 0.1185 | 0.1559 | 0.0926 |
| Unseen | 0.0582 | 0.8148 | 0.1317 | 0.2108 | 0.0912 |

### T4 vs A100 Comparison

| Metric | T4 (16 GB) | A100 (40 GB) | Speedup |
|--------|------------|--------------|---------|
| Throughput | 18,868 samples/s | 88,128 samples/s | **4.7x** |
| AMP dtype | float16 + GradScaler | bfloat16 (no scaler) | Stable |
| Best epoch | 33 | 39 | More room |
| Total time | 291 min (33 ep) | 201.7 min (50 ep) | Faster |

### Business Impact

- 50,000 wafers/month, 0.62% fail rate = 310 defective wafers/month
- $45,000 cost per missed defect (customer return)
- Champion catches 54 defects/month at 17.3% recall
- **Estimated annual savings: $29M**

---

## 40-Day Production Simulation (A100, Completed)

The simulation ran **200M rows across 40 days** on an A100 GPU in 219.4 minutes, with 1 automated retrain, 1 canary failure, and 1 rollback — all fully automated.

```mermaid
gantt
    title 40-Day Simulation Timeline
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Stable
    Days 1-8  Reference window       :done, 2026-02-20, 8d
    Days 9-16 Clean inference         :done, 2026-02-28, 8d

    section Drift Detected
    Days 17-29 Critical drift (3-5 features) :crit, 2026-03-09, 13d

    section Retrain
    Day 30 RETRAIN (50 ep, A100 bfloat16)    :crit, 2026-03-21, 1d

    section Post-Retrain
    Days 31-38 Monitoring v2 champion  :done, 2026-03-22, 8d
    Day 39 Bad model + canary fail + rollback :crit, 2026-03-30, 1d
    Day 40 System recovered            :done, 2026-03-31, 1d
```

| Day | Event | Action |
|-----|-------|--------|
| 1-8 | Reference window | Baseline data generation |
| 9-16 | Clean inference | No drift detected |
| 17-29 | Drift detected (3-5 critical features) | Staleness gate blocks retrain (< 30 days) |
| **30** | **Retrain triggered** | **50 epochs on A100, bfloat16, model v2 promoted** |
| 31-38 | Post-retrain monitoring | v2 champion, drift continues |
| **39** | **Bad model deployed** | **Canary failed, rollback to v2** |
| 40 | System recovered | Final state: v2 champion active |

All 40 days uploaded to S3. Model artifact: `s3://p053-mlflow-artifacts/models/day30_v2_retrained.pt`

### Simulation Charts

<p align="center">
  <img src="assets/p53_33_drift_timeline.png" width="48%" alt="Drift Timeline"/>
  <img src="assets/p53_34_retrain_story.png" width="48%" alt="Retrain Story"/>
</p>
<p align="center">
  <img src="assets/p53_37_simulation_summary.png" width="48%" alt="Simulation Summary"/>
  <img src="assets/p53_38_psi_waterfall.png" width="48%" alt="PSI Waterfall"/>
</p>

---

## System Architecture

```mermaid
graph TB
    subgraph DataPipeline["Data Pipeline"]
        DG["Data Generator<br/>16M rows x 36 features"] --> PP["Preprocessing<br/>Scaling, encoding, engineering"]
        PP --> NPZ["preprocessed_full.npz<br/>2.1 GB"]
    end

    subgraph Training["Model Training (A100 GPU)"]
        NPZ --> BL["Baselines<br/>LogReg, XGBoost, LightGBM"]
        NPZ --> HM["HybridTransformerCNN<br/>317K params"]
        HM --> FL["Focal Loss<br/>alpha=0.75, gamma=2.0"]
        FL --> AMP["bfloat16 AMP<br/>No GradScaler needed"]
    end

    subgraph MLflowStack["MLflow + Model Registry"]
        AMP --> ML["MLflow Tracking<br/>Params, metrics, artifacts"]
        ML --> MR["Model Registry<br/>Champion / Baseline aliases"]
        ML --> S3["S3 Artifacts<br/>Model weights + benchmarks"]
    end

    subgraph Simulation["40-Day Production Simulation"]
        SG["Streaming Generator<br/>14 drift scenarios"] --> DD["Drift Detector<br/>PSI across 6 key features"]
        DD --> RG["3-Criteria Retrain Gate<br/>Drift + Staleness + Data volume"]
        RG -->|trigger| RT["GPU Retrain<br/>50 epochs, auto S3 upload"]
        RT --> CE["Canary Evaluation"]
        CE -->|pass| PR["Promote Champion"]
        CE -->|fail| RB["Rollback to Previous"]
    end

    subgraph Serving["Production Serving"]
        MR --> FA["FastAPI<br/>Prometheus metrics"]
        FA --> DK["Docker Compose<br/>6 services"]
        DK --> K8["Kubernetes<br/>HPA 2-8 pods, canary"]
        K8 --> PM["Prometheus + Grafana"]
    end

    subgraph BigData["Big Data Stack (Docker)"]
        KF["Kafka<br/>Streaming ingestion"] --> SP["Spark ETL<br/>Distributed processing"]
        SP --> AF["Airflow DAGs<br/>3 orchestration pipelines"]
    end

    subgraph AWS["AWS Infrastructure"]
        RDS["RDS PostgreSQL<br/>MLflow backend"]
        S3B["S3 Bucket<br/>Artifacts + models"]
        ECR["ECR<br/>Container registry"]
    end

    style DataPipeline fill:#1e3a5f,color:#fff
    style Training fill:#1e3a5f,color:#fff
    style MLflowStack fill:#7c2d12,color:#fff
    style Simulation fill:#065f46,color:#fff
    style BigData fill:#4c1d95,color:#fff
    style Serving fill:#065f46,color:#fff
    style AWS fill:#92400e,color:#fff
```

---

## Drift Detection and Retrain Pipeline

```mermaid
flowchart LR
    A["Daily Data<br/>5M rows"] --> B["PSI Calculation<br/>6 key features"]
    B --> C{Critical features >= 3?}
    C -->|No| D["Log & Continue"]
    C -->|Yes| E{Days since<br/>last retrain >= 30?}
    E -->|No| F["Staleness Gate<br/>Blocks retrain"]
    E -->|Yes| G{Data volume<br/>>= 10K rows?}
    G -->|No| H["Low-data Tag<br/>No action"]
    G -->|Yes| I["RETRAIN<br/>50 epochs, A100"]
    I --> J["Canary Evaluation"]
    J -->|Better| K["Promote to Champion"]
    J -->|Worse| L["Rollback"]
    K --> M["Upload to S3"]
    L --> M

    style I fill:#b91c1c,color:#fff
    style K fill:#065f46,color:#fff
    style L fill:#92400e,color:#fff
```

---

## The bfloat16 Story (Engineering Decision ED-031)

Training on T4 GPU with float16 + GradScaler caused a **death spiral**: FocalLoss with extreme class imbalance (0.6% positives) produces gradient magnitudes that overflow float16's 5-bit exponent range. The GradScaler responds by halving its scale factor until it reaches zero, at which point all gradients become zero and training collapses.

| Attempt | AMP | Result |
|---------|-----|--------|
| v2 (float16) | float16 + GradScaler | Collapsed at epoch 5 (scale -> 0) |
| v3 (float16 + warmup) | float16 + GradScaler | Collapsed at epoch 7 |
| v4 (bfloat16, A100) | bfloat16, no GradScaler | Stable 50 epochs |

**Root cause:** bfloat16's 8-bit exponent (vs float16's 5-bit) handles the extreme gradient magnitudes from FocalLoss without overflow. No GradScaler needed.

---

## Project Structure

```
DRAM_Yield_Predictor_MLOps/
├── src/                              # 33 Python modules
│   ├── config.py                     # Central configuration
│   ├── data_generator.py             # 16M-row synthetic DRAM dataset
│   ├── preprocess.py                 # Full preprocessing pipeline
│   ├── model.py                      # HybridTransformerCNN (317K params)
│   ├── focal_loss.py                 # FocalLoss (NaN-safe, NumPy + PyTorch)
│   ├── train.py                      # MLflow-integrated GPU training (CLI)
│   ├── train_baseline.py             # LogReg + XGBoost + LightGBM baselines
│   ├── run_simulation.py             # 40-day production simulation orchestrator
│   ├── streaming_data_generator.py   # 14-scenario drift data generator
│   ├── drift_detector.py             # PSI + KL + KS triple drift detection
│   ├── retrain_trigger.py            # 3-criteria retrain gate
│   ├── gpu_selector.py               # Auto GPU selection (T4/A100)
│   ├── compute_backend.py            # AWS -> Colab -> Local fallback
│   ├── s3_utils.py                   # S3 artifact management
│   ├── serve.py                      # FastAPI with Prometheus metrics
│   ├── inference.py                  # YieldPredictor (batch + single)
│   ├── mlflow_utils.py               # MLflow tracking + Model Registry
│   ├── spark_etl.py                  # Spark ETL pipeline
│   ├── spark_drift_detector.py       # Distributed drift detection (PySpark)
│   ├── kafka_producer.py             # Streaming data to Kafka
│   ├── kafka_consumer.py             # Real-time inference consumer
│   ├── sagemaker_pipeline.py         # 9-step SageMaker Pipeline
│   └── simulation_logger.py          # Comprehensive day-level logging
├── notebooks/
│   ├── NB01_advanced_eda.ipynb       # 14-plot EDA with spatial wafer analysis
│   ├── NB02_gpu_training_v4_A100.ipynb   # A100 bfloat16 training
│   ├── NB03_production_training.ipynb    # Day 1 champion training
│   └── NB04_colab_training_A100.ipynb    # 40-day simulation notebook
├── deploy/
│   ├── docker/                       # Docker Compose (6 services), Prometheus, Grafana
│   ├── aws/                          # .env.aws, Dockerfile.airflow-gpu, EC2 bootstrap
│   ├── k8s/                          # Kubernetes: deployment, HPA, canary, service
│   ├── airflow/dags/                 # 3 DAGs: daily, retrain, simulation master
│   └── monitoring/                   # Prometheus + Grafana configs
├── docs/
│   ├── ENGINEERING_DECISIONS.md      # 42 decisions with interview-ready answers
│   ├── Memory_Yield_Predictor_Report.html  # Full project report
│   └── (8 more technical guides)
├── assets/                           # 45 PNG plots (EDA, baselines, simulation)
├── data/
│   ├── simulation_timeline.json      # 40-day A100 simulation results
│   ├── benchmark_*.json              # GPU training benchmarks
│   └── drift_reports/                # Per-day PSI drift reports
├── tests/                            # 20 tests (model, API, drift, config)
├── web/dashboard.html                # Plotly.js dark-theme interactive dashboard
├── .github/workflows/ci.yml          # CI: lint + test + Docker build
└── requirements.txt
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **ML/DL** | PyTorch 2.x, scikit-learn, XGBoost, LightGBM |
| **MLOps** | MLflow (PostgreSQL backend, S3 artifacts, Model Registry) |
| **Data** | Apache Spark, Apache Kafka, Pandas, NumPy |
| **Orchestration** | Apache Airflow (3 DAGs) |
| **Serving** | FastAPI, Prometheus, Grafana |
| **Infrastructure** | Docker (6+ services), Kubernetes (HPA, canary), AWS (RDS, S3, ECR, SageMaker) |
| **GPU** | NVIDIA A100 (bfloat16), T4 (float16), Apple MPS (benchmark) |
| **CI/CD** | GitHub Actions (ruff lint + pytest + Docker build) |

---

## Quick Start

### Local (SQLite backend)
```bash
git clone https://github.com/rajendarmuddasani/DRAM_Yield_Predictor_MLOps.git
cd DRAM_Yield_Predictor_MLOps

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate dataset + preprocess
python src/data_generator.py
python src/preprocess.py --full

# Train baselines (CPU, ~2 min)
python src/train_baseline.py

# Train HybridTransformerCNN (auto-detects GPU)
python -m src.train --full --batch-size 4096

# Run 40-day simulation (fast mode, ~10 min)
python -m src.run_simulation --fast

# MLflow UI
mlflow ui --backend-store-uri "sqlite:///mlflow.db" --port 5001

# Tests
pytest tests/ -v  # 20 tests
```

### Docker (PostgreSQL backend)
```bash
cd deploy/docker && docker compose up -d
# MLflow: http://localhost:5001
# Grafana: http://localhost:3000
# API docs: http://localhost:8000/docs
```

### AWS Production (RDS + S3)
```bash
cp deploy/aws/.env.aws.template deploy/aws/.env
# Edit with RDS endpoint, S3 bucket, ECR URI
docker compose -f deploy/aws/docker-compose-bigdata-aws.yml --env-file deploy/aws/.env up -d
```

---

## Engineering Decisions

See [docs/ENGINEERING_DECISIONS.md](docs/ENGINEERING_DECISIONS.md) for all 42 documented decisions. Key ones:

| ID | Decision | Why |
|----|----------|-----|
| ED-001 | AUC-PR over AUC-ROC | AUC-ROC inflates at 1:159 imbalance due to TNs |
| ED-003 | FocalLoss over SMOTE/class weights | SMOTE creates synthetic noise; FocalLoss focuses on hard examples |
| ED-004 | bfloat16 on A100 (no GradScaler) | float16 death spiral with FocalLoss extreme gradients |
| ED-007 | Transformer+CNN hybrid | Different feature distributions need separate processing |
| ED-015 | Triple drift detection | PSI alone misses distribution shape changes |
| ED-041 | GPU auto-selector | Automated T4/A100 selection by model params and data volume |
| ED-042 | Low-data drift tagging | Tag but never retrain when <10K rows |
| ED-043 | Compute backend fallback | AWS -> Colab -> Local chain for resilience |

---

*MIT License*
