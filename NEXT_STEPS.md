# P053 - Next Steps Tracker

**Status key:** TODO | IN-PROGRESS | DONE | BLOCKED

---

## Phase 0 - Publish (Do This First)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 0.1 | Post LinkedIn - upload carousel PDF as document | DONE | Posted Apr 18 2026 |
| 0.2 | Post LinkedIn - add comment.txt as first comment | TODO | Add technical deep-dive as first comment |
| 0.3 | Repo cleanup - remove planning/personal docs | DONE | Done Apr 18 2026 |
| 0.4 | Push clean repo to AIML-Engineering-Lab + personal | DONE | Deleted + recreated + fresh push |
| 0.5 | Update content hub post.txt with chosen style B | DONE | Updated Apr 18 2026 |

### Files to EXCLUDE from clean repo push
```
.github/copilot-instructions.md    <- personal session context
MASTER_PLAN.md                     <- internal planning
TASKS_PLAN.md                      <- internal task tracking
NEXT_STEPS.md                      <- this file
never_forget_copilot_instructions.md
check_aws_services.sh              <- AWS-specific ops
docs/ENGINEERING_DECISIONS.md      <- internal reasoning
docs/REASONING_INFORMATION.md      <- personal notes
docs/ACCELERATED_TRAINING_PLAN.md  <- internal plan
docs/AWS_COMMANDS_GUIDE.md
docs/AWS_SETUP_GUIDE.md
docs/AWS_INFRASTRUCTURE_GUIDE.md
mlruns/                            <- local MLflow sqlite runs
notebooks/checkpoints/             <- Colab checkpoint files
notebooks/P053_artifacts/          <- Colab output artifacts
notebooks/gpu_artifacts/
models/                            <- local model files
data/production/                   <- generated production data
data/drift_reports/                <- generated drift reports
```

### Files to KEEP in clean repo
```
src/               all 33 Python modules
tests/             20 tests
deploy/docker/     docker-compose, Dockerfile, grafana, prometheus
deploy/airflow/    3 DAGs
deploy/k8s/        manifests
deploy/aws/        Dockerfile.airflow-gpu (no .env files)
notebooks/         NB01-NB04 only (cleared outputs)
assets/            all 44 PNGs + carousel slides + GIFs
docs/carousel.html carousel
web/dashboard.html
README.md
requirements.txt
requirements-serve.txt
ruff.toml
LICENSE
.github/workflows/ci.yml
data/dram_stdf_sample.csv          <- sample only
data/*.dvc                         <- DVC pointers only
data/*.json                        <- benchmark/profile data
```

---

## Phase 1 - Docker (Local First, Then Cloud)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1.1 | `docker compose up -d` in deploy/docker | DONE | All 6 services up and healthy |
| 1.2 | Smoke test FastAPI `/predict` endpoint | DONE | Returns `{probability:0.99, label:FAIL}` |
| 1.3 | Verify MLflow tracking from container | DONE | MLflow UI at :5001, PostgreSQL backend |
| 1.4 | Verify Grafana dashboards load | DONE | Datasource UID fixed; panels populate after traffic |
| 1.5 | Fix config.py container path bug | DONE | Fixed `_here.parent` check for flat layout |
| 1.6 | Push image to ECR | TODO | `aws ecr get-login-password \| docker login` |
| 1.7 | Deploy on cloud - option A: AWS ECS Fargate | TODO | Serverless, no EC2 quota needed |
| 1.8 | Deploy on cloud - option B: SageMaker endpoint | TODO | ml.g4dn.xlarge, GPU inference |

---

## Phase 2 - Kafka + Spark (Actually Running)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 2.1 | Install Java 17 | DONE | openjdk@17 via brew, Apr 18 2026 |
| 2.2 | Start Kafka via docker-compose-bigdata.yml | TODO | `docker compose -f deploy/docker-compose-bigdata.yml up -d` |
| 2.3 | Run kafka_producer.py streaming test | TODO | 5M records per production day |
| 2.4 | Run kafka_consumer.py -> FastAPI pipeline | TODO | End-to-end streaming test |
| 2.5 | Run spark_etl.py on actual daily batch | TODO | Input: parquet, Output: processed features |
| 2.6 | Benchmark Spark vs pandas on 5M rows | DONE | Local 1-day test: pandas 1.3s vs Spark 1510s; Spark overhead dominates at this scale |

---

## Phase 3 - Airflow (DAGs Running)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 3.1 | Start Airflow via docker compose | TODO | Airflow webserver + scheduler + worker |
| 3.2 | Verify daily_inference DAG runs | TODO | Trigger manually for Day 1 test |
| 3.3 | Verify retrain_trigger DAG logic | TODO | Set low PSI threshold to force test trigger |
| 3.4 | Verify simulation_master DAG end-to-end | TODO | Full 3-day smoke test |

---

## Phase 4 - Kubernetes

| # | Task | Status | Notes |
|---|------|--------|-------|
| 4.1 | Install minikube locally | TODO | `brew install minikube` |
| 4.2 | Deploy FastAPI + Redis on K8s local | TODO | deploy/k8s/deployment.yaml |
| 4.3 | Test HPA autoscaling (2-8 pods) | TODO | Use load_test.py to trigger scale-up |
| 4.4 | Test canary deployment (90/10 split) | TODO | deploy/k8s/canary.yaml |
| 4.5 | Full 6-service stack on K8s local | TODO | Justify K8s service mesh + namespace isolation |
| 4.6 | K8s cloud - EKS or GKE | TODO | Needs AWS quota or Google Cloud |

---

## Phase 5 - AWS (Different Account / Spot)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 5.1 | New AWS account or IAM user in different account | TODO | Re-apply for g4dn quota |
| 5.2 | Try spot instances instead of on-demand | TODO | Spot has separate quota, often less contended |
| 5.3 | Start RDS + run MLflow on RDS PostgreSQL | TODO | `aws rds start-db-instance` when quota approved |
| 5.4 | Launch EC2 g4dn.xlarge with ec2-user-data.sh | TODO | Full autonomous run on EC2 |

---

## Phase 6 - 1TB Scale Training (Future)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 6.1 | Generate Day 1 dataset: 1TB STDF (~350GB parquet) | TODO | ~250M rows |
| 6.2 | Upload to S3 in partitioned parquet | TODO | `s3://p053-mlflow-artifacts/data/day1_v1/` |
| 6.3 | Build S3StreamingDataset (reads row groups, no full download) | TODO | Enables Colab/Kaggle training on 350GB |
| 6.4 | Train Day 1 v1 on streamed S3 data - Colab A100 | TODO | Baseline model on full scale |
| 6.5 | Generate Day 30 dataset: 1.2TB STDF (~420GB parquet) | TODO | ~300M rows with injected drift |
| 6.6 | Train Day 30 v2 retrain - verify drift recovery | TODO | Validate 3-gate logic at real scale |

---

## Phase 7 - SageMaker Endpoint (Cloud-Native Serving)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 7.1 | Package model as SageMaker inference container | TODO | Uses full CUDA PyTorch (not CPU-only) |
| 7.2 | Deploy ml.g4dn.xlarge real-time endpoint | TODO | `sagemaker.Predictor` API |
| 7.3 | Load test SageMaker endpoint | TODO | Target: <100ms p99 |
| 7.4 | Compare SageMaker vs Docker/K8s cost + latency | TODO | Document tradeoffs |

---

## Summary - What's Used vs Written vs Planned

| Component | Code Written | Actually Tested | Notes |
|-----------|-------------|-----------------|-------|
| HybridTransformerCNN | YES | YES (A100 Colab) | Champion model on S3 |
| MLflow (SQLite) | YES | YES | 4 runs retrologged |
| MLflow (RDS) | YES | YES | Container PostgreSQL backend running |
| DVC + S3 | YES | YES | Data pointers committed |
| FastAPI serving | YES | YES | /predict live, model_loaded:true |
| Redis cache | YES | YES | p053-redis healthy, TTL caching active |
| Docker 6-service | YES | YES | All 6 containers healthy Apr 18 2026 |
| Kafka producer/consumer | YES | NO | Java installed, broker not started yet |
| Spark ETL | YES | YES (benchmark only) | Java installed, benchmark running |
| Airflow 3 DAGs | YES | NO | Code done, never triggered |
| Prometheus metrics | YES | YES | Scraping :8000/metrics every 15s |
| Grafana dashboard | YES | YES (verify) | JSON provisioned, check :3000 |
| K8s manifests | YES | NO | HPA + canary written, minikube needed |
| GitHub Actions CI | YES | YES | 20/20 tests pass |
| AWS EC2 GPU | NO | NO | Quota rejected |
| SageMaker endpoint | PARTIAL | NO | Code started |
| 1TB scale training | NO | NO | Future phase |
