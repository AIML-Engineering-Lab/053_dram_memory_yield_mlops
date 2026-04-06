# P053 — Accelerated Training & Cloud Execution Plan
> **Goal:** Run actual A100 training on key drift days, not all 40 days.
> **Total GPU time:** ~8-10 hours on Colab A100 (not 40 × 85 min = 56 hours)

---

## 1. Strategy: Train on Drift Days Only

In production, the model only retrains when **drift is detected and retrain gates pass**.
Our 40-day simulation has ~2-3 retrain events. We train ONLY on those days.

### Simulation Timeline (from --medium run):

| Day | Scenario | Training? | Why | Status |
|-----|----------|-----------|-----|--------|
| 1 | Baseline (clean) | ✅ **Initial model** | First model, A100 50 epochs | ✅ DONE — 201.7 min, AUC-ROC=0.816, F1=0.127 |
| 2-8 | Clean reference | ❌ | No drift possible (reference window) | ⬜ |
| 9-14 | Early warnings | ❌ | Warnings only, < 3 critical features | ⬜ |
| 15-20 | Moderate drift | ❌ | Critical features building up | ⬜ |
| 21-25 | Equipment aging | ⚠️ Maybe | If 3+ features cross PSI > 0.2 | ⬜ |
| 26-30 | Process change | ✅ **Retrain v2** | Major drift, 30-day staleness passed | ⬜ |
| 31-35 | Post-retrain | ❌ | New model deployed, monitoring | ⬜ |
| 36-38 | Recovery | ❌ | System stable | ⬜ |
| 39 | Bad model deploy | ✅ **Retrain v3** | Deliberately deploy bad → rollback | ⬜ |
| 40 | Recovery | ❌ | System recovered | ⬜ |

**Result: 3 actual training runs instead of 40.**

---

## 2. Execution Plan

### Phase A: Local Simulation (your laptop, ~30 min)
```bash
# Run --medium simulation to identify exact drift days
python -m src.run_simulation --medium

# Check timeline for retrain events
python -c "
import json
t = json.load(open('data/simulation_timeline.json'))
for d in t['days']:
    if 'RETRAIN_TRIGGERED' in d.get('events', []):
        print(f'Day {d[\"day\"]}: {d[\"date\"]} — {d[\"scenario\"]}')"
```

### Phase B: Colab A100 Training (3 sessions, ~3 hours each)
Upload `NB03_production_training.ipynb` to Colab:

**Session 1 — Day 1 (Initial Model):**
```python
# On Colab A100:
!pip install mlflow psycopg2-binary torch ...
%env MLFLOW_TRACKING_URI=postgresql://mlflow:<pw>@<rds-host>:5432/mlflow

from src.train import run_training
run_training(args=Namespace(
    full=True, epochs=50, batch_size=4096, lr=1e-3,
    run_name="production-day1-initial", context="day1-baseline"
))
```

**Session 2 — Day 26-31 (Retrain v2 after drift):**
```python
# Same setup, but with drifted data
run_training(args=Namespace(
    full=True, epochs=30, batch_size=4096, lr=5e-4,  # lower LR, fewer epochs
    run_name="retrain-day31-drift", context="day31-post-drift"
))
```

**Session 3 — Day 39 (Bad model → rollback story):**
```python
# Deliberately train a bad model (wrong hyperparams)
run_training(args=Namespace(
    full=True, epochs=10, batch_size=256, lr=1e-2,  # bad config
    run_name="retrain-day39-bad-deploy", context="day39-bad-model"
))
# Then show rollback to v2 in MLflow registry
```

### Phase C: AWS Cloud (1-2 month subscription)
```bash
# Provision AWS infrastructure
./deploy/aws/setup_aws.sh

# Deploy stack
docker compose -f deploy/aws/docker-compose-aws.yml up -d

# Run FULL 40-day simulation against real AWS services
MLFLOW_TRACKING_URI="postgresql://mlflow:<pw>@<rds>:5432/mlflow" \
MLFLOW_ARTIFACT_ROOT="s3://p053-mlflow-artifacts/" \
python -m src.run_simulation --full

# Training happens on SageMaker or Colab → logs to same RDS MLflow
```

---

## 3. AWS Cost Estimate (1-2 months)

| Service | Spec | $/month | Notes |
|---------|------|---------|-------|
| RDS PostgreSQL | db.t3.micro, 20 GB | $15 | MLflow backend |
| S3 | < 5 GB artifacts | $0.12 | Model weights, plots |
| EC2 | t3.xlarge (stop when done) | $0.17/hr | Runs compose stack |
| **Total (always-on)** | | **~$135/month** | |
| **Total (8 hrs/day)** | | **~$55/month** | Stop EC2 at night |

**Colab Pro:** $10/month for A100 access (100 compute units)

**Total for 2 months:** ~$120-280 depending on usage pattern.

---

## 4. Enterprise Readiness Checklist

When AMD/Micron provides AWS account, you swap:
- [ ] `.env.aws.template` → fill with their RDS endpoint
- [ ] S3 bucket → their bucket (or new one in their account)
- [ ] ECR → their container registry
- [ ] IAM role → their role with SageMaker/S3/RDS access
- [ ] Synthetic data → real silicon fab data (same schema: 36 features + target)
- [ ] Colab training → SageMaker training jobs (pipeline already exists)

**Zero code changes needed.** Only environment variables and data source change.

---

## 5. What Makes This NOT a Toy Project

| Toy Project | This Project |
|-------------|-------------|
| SQLite for everything | PostgreSQL (Docker/AWS), SQLite (local dev only) |
| `model.pkl` on disk | MLflow Model Registry with aliases (@champion, @baseline) |
| Retrain manually | Drift-triggered retrain with 3-gate criteria |
| No artifact tracking | S3 artifact storage, versioned |
| Train once on laptop | Multi-GPU training (T4, A100), hardware auto-detection |
| No monitoring | Prometheus + Grafana + prediction distribution alerts |
| `pip install` | Docker Compose (5 services) + Kubernetes manifests |
| No data pipeline | Kafka → Spark → Airflow → SageMaker |
| No cost awareness | AWS cost estimates + spot instance strategy |

---

*Last updated: 2026-04-04*
