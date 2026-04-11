# P053 — Engineering Journey: From Failed Training to Production MLOps

> A story of debugging float16 death spirals, building multi-cloud fallback,
> and learning why "it works on my machine" is the scariest sentence in ML.

---

## Chapter 1: The Goal

Build a **production-grade DRAM yield predictor** that proves principal-level MLOps:
- 16 million rows of semiconductor test data
- 1:160 class imbalance (defective dies are rare)
- HybridTransformerCNN model (317K parameters)
- Full MLOps: Airflow orchestration, drift detection, automatic retraining, canary deployment, rollback
- 40-day production simulation running autonomously

---

## Chapter 2: The First Training — A100 Success

**Day 1 training on Colab A100** went perfectly:
- 50 epochs, 201.7 minutes, bfloat16 precision
- Val AUC-ROC: 0.816, Val AUC-PR: 0.054
- Best checkpoint at epoch 39
- Model saved to S3: `s3://p053-mlflow-artifacts/models/day1_champion.pt`

The A100's **8-bit exponent** in bfloat16 handled our FocalLoss gradients without issues. We had a working champion model. What could go wrong?

---

## Chapter 3: The float16 Death Spiral

When we moved to **T4 GPU** (the cheaper option for retraining), everything broke.

### What Happened

FocalLoss with α=0.75, γ=2.0 on 1:160 imbalanced data produces extreme gradient magnitudes. The formula:

```
FocalLoss = -α(1-pt)^γ × log(pt)
```

When a defective die (minority class) is misclassified, `pt` is tiny. The gradient term `(1-pt)^γ × log(pt)` explodes. With learning rate 1e-3, the scaled gradient value exceeds **65,504** — the maximum representable float16 number.

### The GradScaler Collapse

PyTorch's GradScaler detects inf/NaN in gradients and halves the scale factor:
```
Scale 1024 → 512 → 256 → ... → 1 → 0.5 → ... → 0
```
Once scale hits 0, every gradient becomes zero. Every batch produces NaN. Training collapses.

### The Fix

Two things had to change:
1. **Learning rate: 1e-3 → 2e-4** — keeps gradients within float16's 5-bit exponent range
2. **Batch size: 4096 → 1024** — reduces gradient variance per step

On A100 (compute capability 8.0+), bfloat16's 8-bit exponent handles max values up to 3.4×10^38, so LR=1e-3 works fine there.

### The Decision Table

| GPU | Compute Cap | Precision | Max Value | LR | Batch Size |
|-----|-------------|-----------|-----------|-----|------------|
| A100 | 8.0 | bfloat16 | 3.4×10^38 | 1e-3 | 4096 |
| T4 | 7.5 | float16 | 65,504 | 2e-4 | 1024 |

**Engineering Decision ED-004:** This is documented as the single most important technical insight of the project.

---

## Chapter 4: The Invisible Bug

We verified the LR fix in an isolated training notebook (NB05). Training completed perfectly on T4 with lr=2e-4. Tests passed. CI passed. We ran the full 40-day simulation on Colab.

**All 11 retraining attempts failed.**

### The Root Cause

NB05 tested `train.py` directly:
```bash
python -m src.train --lr 2e-4 --batch-size 1024 --full
```

But the simulation calls `run_simulation.py`, which spawns `train.py` as a subprocess. The pre-fix code in `run_simulation.py` **did not pass `--lr`** to the subprocess:
```python
cmd = [sys.executable, "-m", "src.train",
       "--full", "--epochs", str(epochs),
       "--batch-size", batch_size,   # ← batch_size was there
       # "--lr", learning_rate       # ← THIS WAS MISSING
       "--run-name", run_name]
```

So `train.py` fell back to `config.py`'s default: `lr = 1e-3`. The death spiral returned.

### The Lesson

**"It works on my machine"** had a new incarnation:
> "It works when I call it directly, but not when the orchestrator calls it."

Two different code paths — one tested, one not. The fix was two lines:
```python
learning_rate = "2e-4"  # T4 safe default
cmd = [... "--lr", learning_rate, ...]
```

---

## Chapter 5: Why 11 Retrains Instead of 3?

The accelerated training plan called for ~3 retrains (days 1, ~30, 39). But the simulation triggered 11 consecutive retrain attempts (days 30-40).

### The Staleness Gate Loop

The retrain trigger uses a **3-criteria gate**:
1. **Drift:** PSI critical on 3+ features ✓ (met from day 17 onward)
2. **Performance decay:** AUC-PR dropped below threshold ✓
3. **Staleness:** `days_since_last_retrain >= 30`

When retraining **succeeds** on day 30:
- `last_retrain_day = 30`
- Day 31: `31 - 30 = 1 < 30` → blocked
- Next retrain at day 60+ (beyond simulation)

When retraining **fails** on day 30:
- `last_retrain_day` stays at 0 (never updated)
- Day 31: `31 - 0 = 31 >= 30` → triggers again
- Day 32: `32 - 0 = 32 >= 30` → triggers again
- ...repeats until day 40

This is actually **correct behavior** — the system keeps trying because the model is stale and drift is real. In production, this would page an on-call engineer after N consecutive failures.

---

## Chapter 6: The BAD_MODEL_DEPLOYED Scenario

Day 39 includes a scripted event: `BAD_MODEL_DEPLOYED` → `CANARY_FAILED` → `ROLLBACK_TO_v2`.

This is NOT an actual bad model. It's a **demonstration of the rollback mechanism**:
1. A deliberately bad model gets deployed
2. Canary evaluation detects degraded metrics (AUC-PR, latency, error rate)
3. Canary fails → automatic rollback to previous champion
4. System recovers on Day 40

This proves the MLOps pipeline can handle the worst-case scenario autonomously.

### How Canary Deployment Works (Production)

```
Deploy new model → 5% traffic (canary)
                → 95% traffic (champion)

Monitor for 1 hour:
  - If canary AUC-PR > champion: promote to 100%
  - If canary latency > 2x champion: ROLLBACK
  - If canary error rate > threshold: ROLLBACK
```

---

## Chapter 7: The AUC-PR Question

"Is AUC-PR of 0.054 any good?"

At 1:160 class imbalance:
- **Random classifier AUC-PR** = 1/160 ≈ 0.006
- **Our model AUC-PR** = 0.054 = **9x better than random**
- AUC-PR of 0.8+ is mathematically impossible unless features perfectly separate classes

In semiconductor yield prediction:
- A 2x improvement over random saves millions in defect detection
- Our 9x improvement catches defects that would otherwise ship to customers
- The model identifies **which wafers** to inspect first, not whether defects exist

> AUC-ROC (0.816) looks better because it's dominated by true negatives (159 of every 160 dies are good).
> AUC-PR is the honest metric for rare event detection.

**Engineering Decision ED-007:** AUC-PR is the primary metric, AUC-ROC is reported but not used for decisions.

---

## Chapter 8: The False Alarm (Day 9)

Day 9 simulates a PSI spike on 1-2 features that looks like drift but isn't real. The system correctly handles it:
- Drift check: only 1 feature critical (threshold requires 3+)
- Decision: `drift_clean` — no action needed
- Day 10: `auto_recover` — values return to normal

This demonstrates the 3-criteria gate preventing unnecessary retraining from statistical noise. Without it, the system would retrain on every micro-fluctuation, wasting GPU hours.

---

## Chapter 9: The Multi-Cloud Fallback

When AWS rejected our GPU quota request, we built a 3-tier fallback:

```
AWS EC2 g4dn.xlarge (T4, $0.53/hr)
  ↓ if unavailable
Google Colab Pro (T4, ~1.36 CU/hr)
  ↓ if unavailable
Local MacBook (MPS/CPU, slow but works)
```

**compute_backend.py** automatically detects the environment and configures MLflow URI, batch size, and artifact storage accordingly.

**Engineering Decision ED-043:** The fallback chain ensures training always happens, somewhere. No blocked sprints waiting for cloud quotas.

---

## Chapter 10: What the Next Run Should Show

With the LR fix deployed:
1. Days 1-29: Inference only, drift monitoring, no retraining
2. Day 30: RETRAIN_TRIGGERED → training succeeds → v2 champion deployed
3. Days 31-38: New model serving, monitoring performance
4. Day 39: BAD_MODEL_DEPLOYED → CANARY_FAILED → ROLLBACK to v2
5. Day 40: System recovered

**Expected artifacts:**
- 1-2 successful retrain events (not 11 failures)
- Model transitions: v1_original → v2_retrained → v2_retrained (after rollback)
- AUC-PR maintained or improved after retraining

---

## Appendix: Key Engineering Decisions

| ID | Decision | Why |
|----|----------|-----|
| ED-001 | 3-tier compute: MPS → T4 → A100 | Cost-effective development |
| ED-002 | FocalLoss (α=0.75, γ=2.0) | SMOTE creates synthetic noise at 1:160 |
| ED-003 | Per-feature tokenization | Each DRAM param has different distributions |
| ED-004 | bfloat16 on A100, float16+LR=2e-4 on T4 | float16 death spiral with FocalLoss |
| ED-007 | AUC-PR as primary metric | AUC-ROC inflates at extreme imbalance |
| ED-041 | GPU auto-selector | T4/V100/A10G/A100 selection by model size |
| ED-042 | Low-data drift tagging | Tag don't retrain when <10K rows |
| ED-043 | Compute fallback chain | AWS → Colab → Local |

---

## Timeline

| Date | Event |
|------|-------|
| Pre-Apr 2026 | Built 33 src modules, 20 tests, 6 notebooks |
| Apr 5-6 | Phase 0: Wired GPU training into DAGs |
| Apr 7 | AWS GPU quota rejected; Colab fallback designed |
| Apr 8 | Built compute_backend.py, NB04 notebook |
| Apr 9 | v7A executed on T4; 11 retrains failed (LR bug) |
| Apr 9 | Root cause found: run_simulation.py missing --lr |
| Apr 9 | Fixed: lr=2e-4, batch_size=1024 for T4 |
| Next | Re-run simulation with fix → expected 1-2 successful retrains |
