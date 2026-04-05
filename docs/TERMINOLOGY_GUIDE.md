# P053 — Terminology & Concepts Guide

> Everything you need to know for interviews and building this project.  
> Organized by domain: DRAM Manufacturing → ML/MLOps → Big Data → Cloud/AWS → Deployment.

---

## Table of Contents

1. [DRAM Manufacturing & Features](#1-dram-manufacturing--features)
2. [Machine Learning Fundamentals](#2-machine-learning-fundamentals)
3. [MLflow & Experiment Tracking](#3-mlflow--experiment-tracking)
4. [MLOps & Production ML](#4-mlops--production-ml)
5. [Drift Detection & Monitoring](#5-drift-detection--monitoring)
6. [Model Deployment Strategies](#6-model-deployment-strategies)
7. [Big Data Stack (Kafka, Spark, Airflow)](#7-big-data-stack)
8. [Cloud & AWS Services](#8-cloud--aws-services)
9. [Docker & Containerization](#9-docker--containerization)
10. [CI/CD & DevOps](#10-cicd--devops)
11. [Our 40-Day Simulation — How It All Connects](#11-our-40-day-simulation)

---

## 1. DRAM Manufacturing & Features

### What Is DRAM?

**DRAM = Dynamic Random Access Memory.** The RAM in your computer, phone, and server. Companies like Micron, Samsung, and SK Hynix manufacture DRAM chips in semiconductor fabrication plants ("fabs").

**The business problem:** Each silicon wafer costs ~$1,200 to produce and contains hundreds of memory dies. Some dies are defective. If a defective die ships to a customer (e.g., Apple for iPhones), the cost is **$45,000** per defective unit (recall + reputation). If we falsely flag a good die as defective, it costs **$120** (unnecessary re-test).

**Our job:** Predict which dies will fail BEFORE they leave the fab using electrical test data.

### DRAM Test Features We Use (33 Features)

Our model uses 33 features from electrical tests performed on each die. Here's what each one means:

#### Electrical Test Parameters (23 features)

| Feature | Full Name | What It Measures | Why It Matters for Yield |
|---------|-----------|-----------------|-------------------------|
| `retention_time` | Data Retention Time | How long a memory cell holds its charge (ms) | Core reliability — if charge leaks too fast, data corrupts. Lower = failing die |
| `access_time` | Memory Access Time | Time to read/write one memory cell (ns) | Performance metric — slow access = potential defect in word/bit lines |
| `refresh_rate` | DRAM Refresh Rate | How often cells need recharging (µs) | DRAM is "dynamic" = charge leaks constantly. Higher refresh = weaker cells |
| `leakage_current` | Junction Leakage Current | Current flowing when transistor is OFF (nA) | Higher leakage = thinner gate oxide or contamination. Strong failure predictor |
| `threshold_voltage` | Transistor Threshold Voltage | Minimum voltage to turn ON a transistor (mV) | Variations indicate process inconsistency — too low → leakage, too high → no access |
| `write_voltage` | Write Operation Voltage | Voltage needed to write "1" to a cell (V) | Abnormal write voltage = capacitor degradation or wordline driver issues |
| `read_voltage` | Read Sense Voltage | Voltage difference detected during read (mV) | Small margins = unreliable reads. Die may pass today, fail tomorrow |
| `power_consumption` | Die Power Draw | Total current draw under test pattern (mW) | Outlier power = short circuits, excessive leakage, or stuck bits |
| `signal_noise_ratio` | Signal-to-Noise Ratio | Read signal strength vs noise floor (dB) | Low SNR = bit errors under stress. Critical for server-grade DRAM |
| `bit_error_rate` | Raw Bit Error Rate | Fraction of bits that read incorrectly | Direct quality metric — BER > threshold → die fails |
| `cell_capacitance` | Storage Capacitance | Charge storage capacity per cell (fF) | Smaller cells = less charge = harder to read = more errors |
| `resistance` | Interconnect Resistance | Metal line resistance (Ω) | High resistance = thin/broken metal lines = reliability risk |
| `temperature_sensitivity` | Thermal Coefficient | How much leakage changes per °C | Dies with high temp sensitivity fail in hot environments (servers, phones) |
| `voltage_margin` | Operating Voltage Margin | Buffer between operating and failure voltage (mV) | Thin margins = die works today but fails when conditions change slightly |
| `current_density` | Peak Current Density | Maximum current per metal cross-section (MA/cm²) | Electromigration risk — high current density slowly destroys metal lines |
| `oxide_thickness` | Gate Oxide Thickness | Insulating layer thickness (nm) | Thinner oxide = more leakage but faster transistors. Process control metric |
| `dopant_concentration` | Impurity Concentration | Atoms implanted per cm³ | Controls threshold voltage. Variations = process drift in ion implanter |
| `defect_density` | Defect Count per Area | Particle/defect count per cm² | Direct quality metric from wafer inspection. More defects = lower yield |
| `junction_depth` | Source/Drain Junction Depth | How deep the transistor junctions go (nm) | Deeper = more leakage, shallower = higher resistance. Process control |
| `gate_length` | Transistor Gate Length | Physical gate dimension (nm) | Critical dimension — variations affect speed, leakage, and yield |
| `interconnect_delay` | RC Delay | Signal propagation time through metal layers (ps) | Slow interconnect = timing failures in synchronous DRAM |
| `contact_resistance` | Via/Contact Resistance | Resistance at metal-to-silicon junctions (Ω) | High contact resistance = poor metallization = reliability failure |
| `metal_line_width` | Metal Trace Width | Width of aluminum/copper interconnects (nm) | Narrow lines = higher resistance + electromigration risk |

#### Categorical Features (4 features)

| Feature | Values | What It Represents |
|---------|--------|-------------------|
| `process_node` | 7nm, 10nm, 14nm | Manufacturing technology generation. Smaller = denser but harder to make |
| `wafer_position` | center, edge, mid | Where on the wafer this die sits. Edge dies have 3× higher failure rate |
| `fab_line` | A, B, C | Which production line made this wafer. Different equipment = different yield |
| `test_program` | std, extended, burn_in | Which test protocol was run. Burn-in is the most thorough |

#### Spatial Features (3 features)

| Feature | What It Measures | Why It Matters |
|---------|-----------------|----------------|
| `die_x` | X coordinate on wafer | Edge dies exposed to more contaminants during processing |
| `die_y` | Y coordinate on wafer | Combined with die_x, identifies "hot spots" on the wafer |
| `edge_distance` | Distance from wafer edge (mm) | **THE most important spatial feature.** Edge dies fail 3× more. Our CNN branch specifically captures this |

#### Engineered Features (7 features, created during preprocessing)

| Feature | Formula | Physics Insight |
|---------|---------|----------------|
| `retention_temp_interaction` | retention_time × temperature_sensitivity | Dies that lose charge fast AND are temperature-sensitive = double risk |
| `leakage_retention_ratio` | leakage_current / retention_time | High leakage + low retention = oxidation degradation pattern |
| `voltage_stress_index` | (write_voltage - read_voltage) / voltage_margin | How "stressed" the die is electrically. High stress = short lifespan |
| `spatial_risk_score` | f(edge_distance, die_x, die_y) | Combined spatial risk — accounts for edge, center, and gradient effects |
| `power_efficiency` | signal_noise_ratio / power_consumption | Good dies have high signal with low power. Bad dies waste power on leakage |
| `defect_impact` | defect_density × gate_length × oxide_thickness | Smaller features amplify defect impact — 1 particle on 7nm is worse than on 14nm |
| `process_stability` | std(threshold_voltage, dopant_concentration, junction_depth) | Low variance = stable process. High variance = equipment drifting |

### Why These Features Matter for Drift Detection

In a real fab, features don't stay constant:
- **Tool wear:** Etching chamber degrades → `gate_length` variance increases slowly (days 5-15)
- **Chemical batch change:** New photoresist lot → `oxide_thickness` shifts suddenly (day 20)
- **Seasonal:** Summer heat → `temperature_sensitivity` baseline shifts (days 25-35)
- **Contamination:** Particle event → `defect_density` spikes (day 31)

Our drift detection (PSI/KL/KS) monitors all 33 features daily. When enough features shift beyond thresholds, we retrain.

---

## 2. Machine Learning Fundamentals

### Loss Functions

| Term | Full Name | What It Does | When to Use |
|------|-----------|-------------|-------------|
| **BCE** | Binary Cross-Entropy | Standard loss for binary classification. Penalizes wrong predictions logarithmically | Balanced datasets (50/50 classes) |
| **Focal Loss** | Focal Loss (Lin et al., 2017) | Modified BCE that **down-weights easy examples**. Formula: $-\alpha(1-p_t)^\gamma \log(p_t)$ | **Imbalanced datasets** like ours (1:159 ratio). γ=2.0 makes the model focus on hard-to-classify failures |
| **Label Smoothing** | — | Instead of hard targets (0 or 1), use soft targets (0.01 or 0.99). Prevents overconfident predictions | Always good practice for production models. We use smoothing=0.01 |

### Our Loss: FocalLoss(α=0.75, γ=2.0, smoothing=0.01)

- **α=0.75**: Weight for positive class (failures). Since failures are rare, we weight them higher
- **γ=2.0**: Focusing parameter. At γ=0, Focal Loss = BCE. At γ=2, easy negatives (99% of data) get ~100× less weight
- **smoothing=0.01**: Soft targets to prevent the model from being 100% confident (which hurts calibration)

### Metrics

| Metric | What It Measures | Why We Use/Don't Use It |
|--------|-----------------|------------------------|
| **Accuracy** | % correct predictions | ❌ USELESS at 1:159 imbalance. Predicting "all pass" = 99.4% accuracy but catches 0 failures |
| **Precision** | Of predicted failures, % actually failing | Important — high FP = unnecessary re-tests ($120 each) |
| **Recall** | Of actual failures, % we caught | **MOST IMPORTANT** for safety. Missed failure = $45,000 |
| **F1 Score** | Harmonic mean of precision and recall | Balanced single metric. We optimize threshold for max F1 |
| **AUC-PR** | Area Under Precision-Recall Curve | **OUR PRIMARY METRIC.** Summarizes the entire precision-recall trade-off. More informative than AUC-ROC for imbalanced data |
| **AUC-ROC** | Area Under ROC Curve | Secondary metric. Less sensitive to class imbalance than AUC-PR |

### Why AUC-PR, Not AUC-ROC?

At 1:159 imbalance, AUC-ROC can look great (0.95+) while the model is useless at actually finding failures. AUC-PR focuses on the minority class (failures) and is much harder to game. A random model gets AUC-PR ≈ 0.006 (the fail rate). Our model gets ~0.05 = 8× better than random.

### Training Techniques

| Technique | What It Does | Our Setting |
|-----------|-------------|-------------|
| **Mixed Precision (AMP)** | Uses 16-bit floats for forward pass (faster, less memory) and 32-bit for gradients (stable) | bfloat16 on A100 (no overflow risk) |
| **GradScaler** | Scales loss up before backward pass to prevent underflow in float16 | ❌ Disabled on A100 (bfloat16 doesn't need it). Enabled on T4/V100 |
| **Gradient Clipping** | Limits gradient magnitude to prevent exploding gradients | max_norm=1.0 — critical with FocalLoss |
| **CosineAnnealing LR** | Learning rate follows a cosine curve from max to min | 1e-3 → 1e-6 over 50 epochs. Smooth decay |
| **Early Stopping** | Stop training when validation metric stops improving | patience=12 epochs. Saves GPU time, prevents overfitting |
| **WeightedRandomSampler** | Oversamples minority class so each batch is 50/50 | Without this, model sees ~25 failures per 4096-batch = too few to learn from |

### Model Architecture

| Component | Purpose | Parameters |
|-----------|---------|-----------|
| **Transformer Encoder** | Captures feature interactions via self-attention. "Which features matter together?" | 2 layers, 4 heads, d_model=128 |
| **CNN Branch** | Captures spatial patterns (edge dies fail more). 1D convolution over [die_x, die_y, edge_distance] | Conv1D: 3→32→64 channels |
| **Fusion MLP** | Combines Transformer and CNN outputs. Learns when to trust which branch | 192→128→64→1 with ReLU + dropout |

**Total: 317,633 parameters** — deliberately small for fast inference (<10ms per batch).

---

## 3. MLflow & Experiment Tracking

### What Is MLflow?

An open-source platform for managing the ML lifecycle: tracking experiments, packaging code, deploying models, and managing model versions.

### MLflow Components

| Component | What It Does | Our Usage |
|-----------|-------------|-----------|
| **Tracking** | Logs parameters, metrics per epoch, and artifacts for every training run | Every training session logs: LR, batch size, AUC-PR per epoch, loss curves, confusion matrices |
| **Model Registry** | Version control for models. Like GitHub but for .pt files. Supports aliases (@champion, @challenger) | v1 = Day 1, v2 = Day 20 retrain, v3 = Day 31, v4 = Day 39 recovery |
| **Artifacts** | Stores model files, plots, benchmark JSONs alongside runs | Model weights (.pt), training curves (.png), config (.json) |
| **Projects** | Packages code + environment for reproducibility | Not used directly — we use Docker instead |

### MLflow Key Concepts

| Term | Meaning | Example |
|------|---------|---------|
| **Experiment** | A collection of related runs | "P053-Production-Training" |
| **Run** | One training execution with params + metrics + artifacts | "A100-day1-champion" |
| **Parameter** | Input config (logged once) | `lr=0.001`, `batch_size=4096`, `amp_dtype=bfloat16` |
| **Metric** | Output measurement (logged per epoch or once) | `val_auc_pr=0.0523`, `train_loss=0.0341` |
| **Artifact** | File output | `hybrid_best_A100-day1-champion.pt` (model weights) |
| **Model Version** | Registered model with a version number | Version 1 → v1, Version 2 → v2 |
| **Alias** | Named pointer to a version | `@champion` points to v1 initially, moves to v2 after retrain |
| **Stage** (deprecated) | Old way of tracking model lifecycle | Replaced by aliases in MLflow 2.x |

### MLflow Backend Options

| Backend | For | Our Setup |
|---------|-----|-----------|
| **File store** | Learning/prototyping | Never used — too fragile |
| **SQLite** | Single-user, Colab training | NB03 on Colab uses local SQLite |
| **PostgreSQL** | Production, multi-user | Docker locally + RDS on AWS |
| **S3** | Artifact storage | Model weights, plots stored in S3 |

### Transfer Path

```
Colab (SQLite)  →  Google Drive (backup)  →  Mac (Docker PostgreSQL)  →  AWS (RDS PostgreSQL)
                                                                          ↓
                                              Artifacts always → S3 (s3://p053-mlflow-artifacts/)
```

---

## 4. MLOps & Production ML

### What Is MLOps?

**MLOps = ML + DevOps.** The practices for deploying, monitoring, and maintaining ML models in production. Like DevOps automates software deployment, MLOps automates ML deployment.

### MLOps Maturity Levels

| Level | Description | What We Demonstrate |
|-------|-------------|-------------------|
| **0 — Manual** | Jupyter notebook, copy-paste model to production | ❌ This is what tutorials teach. Not us. |
| **1 — ML Pipeline** | Automated training + evaluation, manual deploy | We have this via NB03 |
| **2 — CI/CD** | Automated training + testing + deployment | GitHub Actions → ECR → EC2 |
| **3 — Full MLOps** | Automated training + deploy + monitoring + retrain | **Our 40-day simulation = Level 3** |

### Key MLOps Concepts

| Concept | What It Is | Our Implementation |
|---------|-----------|-------------------|
| **Model Versioning** | Track every model ever trained, with lineage | MLflow Model Registry: v1, v2, v3, v4 |
| **Model Registry** | Central store for approved models | MLflow Registry with @champion/@challenger aliases |
| **Feature Store** | Centralized feature computation + serving | Not used (our features are computed in Spark ETL pipeline) |
| **Data Versioning** | Track dataset changes over time | DVC with S3 remote backend |
| **Experiment Tracking** | Log every training run's parameters + results | MLflow Tracking (PostgreSQL + S3) |
| **Model Monitoring** | Track model performance in production | Prometheus metrics → Grafana dashboards |
| **Automated Retraining** | Retrain when drift detected, no human needed | Airflow DAG triggers retrain when PSI > threshold |
| **A/B Testing** | Compare two models on live traffic | Canary deployment (10/90 split) in our simulation |
| **Rollback** | Revert to previous model version immediately | Day 39: canary fails → rollback to v3 → then from-scratch v4 |

---

## 5. Drift Detection & Monitoring

### What Is Data Drift?

When the statistical properties of production data change compared to training data. The model was trained on yesterday's data distribution — if today's data looks different, predictions become unreliable.

### Types of Drift

| Type | What Changes | Example | Detection |
|------|-------------|---------|-----------|
| **Data Drift** (Covariate Shift) | Input feature distributions change | Summer heat shifts temperature_sensitivity baseline | PSI, KL, KS on features |
| **Concept Drift** | The relationship between features and target changes | New defect type that existing features can't capture | Monitor prediction performance directly |
| **Label Drift** | Target variable distribution changes | New process reduces fail rate from 0.62% to 0.3% | Monitor predicted vs actual fail rates |

### Drift Detection Metrics

| Metric | Full Name | Formula (simplified) | Threshold | Interpretation |
|--------|-----------|---------------------|-----------|---------------|
| **PSI** | Population Stability Index | $\sum (p_i - q_i) \times \ln(p_i / q_i)$ where $p$ = training bins, $q$ = production bins | < 0.10 = stable, 0.10-0.20 = moderate, > 0.20 = severe | Measures how much a feature's histogram has shifted. Calculated per feature |
| **KL** | Kullback-Leibler Divergence | $\sum p(x) \times \ln(p(x) / q(x))$ | > 0.1 flag | Information-theoretic measure of distribution difference. Asymmetric: KL(P‖Q) ≠ KL(Q‖P) |
| **KS** | Kolmogorov-Smirnov Statistic | $\max |F_1(x) - F_2(x)|$ (max difference between CDFs) | p-value < 0.05 = drift | Non-parametric — works for any distribution shape. Best for detecting shifts in tails |

### How We Use Them (Daily Pipeline)

```
Each day:
  1. Generate 5M rows of production data
  2. For each of 33 features:
     - Compute PSI vs training distribution
     - Compute KS test p-value
  3. Decision:
     - All features PSI < 0.10 → ✅ No action
     - 4+ features PSI > 0.10  → ⚠️ Moderate drift → trigger retrain DAG
     - 7+ features PSI > 0.20  → 🔴 Severe drift → urgent retrain
```

### Why Multiple Metrics?

| Scenario | PSI Says | KS Says | Truth |
|----------|----------|---------|-------|
| Mean shifts, same shape | High | High | Real drift — both agree ✅ |
| Tails change, mean same | Low | High | Subtle drift — KS catches it, PSI misses it |
| Slight spread change | Moderate | Low | Minor variance change — PSI more sensitive to binning artifacts |

Using all three gives robust detection. In our simulation:
- **Day 20:** PSI > 0.10 on 4 features (tool wear accumulated). KS confirms. → Moderate retrain
- **Day 31:** PSI > 0.20 on 7 features (contamination event + seasonal). → Severe retrain
- **Day 39:** Deliberate bad model fails canary regardless of drift → Rollback + fresh retrain

---

## 6. Model Deployment Strategies

### Champion / Challenger

| Role | What It Means |
|------|--------------|
| **Champion** | The current production model. ALL real predictions go through it. Alias: `@champion` in MLflow |
| **Challenger** | A newly trained model being evaluated. Not yet trusted. Alias: `@challenger` |

**Promotion flow:** Train new model → register as challenger → canary test → if passes → promote to champion → old champion becomes archived.

### Deployment Strategies

| Strategy | Traffic Split | Risk | Speed | Our Usage |
|----------|-------------|------|-------|-----------|
| **Big Bang** | 0% → 100% instantly | 🔴 High — if new model is bad, 100% of predictions are wrong | Instant | ❌ Never in production |
| **Canary** | 10% new / 90% old → gradually increase | 🟢 Low — only 10% affected if bad | Slow (hours/days) | ✅ Day 20, 31 retrains |
| **Blue-Green** | Two full environments, switch DNS | 🟡 Medium — full switch but instant rollback | Fast (seconds) | Not used in our simulation |
| **Shadow** | New model runs in parallel but predictions not served | 🟢 None — zero production risk | Very slow | Not used (too costly for our budget) |
| **A/B Test** | 50/50 split, measure statistical significance | 🟡 Medium — 50% risk if bad | Slow (need statistical power) | Not used — canary is sufficient |

### Canary in Our Project

```
Day 20: Moderate drift detected
  1. Retrain on AWS → new model v2
  2. Deploy canary: v2 gets 10% of inference requests
  3. Compare v2 vs v1 metrics over 1000 samples:
     - v2 AUC-PR ≥ v1 AUC-PR × 0.95?  (not more than 5% worse)
     - v2 recall ≥ v1 recall × 0.90?     (safety critical)
  4. If YES → promote v2 to @champion, v1 archived
  5. If NO → kill v2, keep v1

Day 39: Deliberately bad model:
  1. Model v_bad intentionally poorly trained
  2. Canary fails (AUC-PR much worse)
  3. Rollback → v3 stays as @champion
  4. From-scratch retrain → v4 → canary passes → promote
```

### When Rollback Isn't Enough (Why From-Scratch Retrain?)

**Rollback = instant switch back to old model.** But sometimes the old model can't handle current data:

| Scenario | What Happened | Why Rollback Alone Fails | Fix |
|----------|-------------|------------------------|-----|
| **Regime change** | Fab switched from DDR4 to DDR5 production | v1-v3 trained on DDR4 data, DDR5 physics are different | From-scratch on DDR5 data |
| **Severe contamination** | Chemical leak permanently altered 70% of feature distributions | All previous models assumed old distributions | From-scratch on post-contamination data |
| **Accumulated drift** | Small shifts over 30 days compound. Each v1 retrain adapted slightly, but drift exceeds fine-tuning capacity | Fine-tuning from v3 inherits v3's biases which are now wrong | From-scratch with sliding window of last 30 days |
| **Catastrophic forgetting** | Fine-tuning on drifted data destroyed the model's original knowledge | v2 "forgot" how to handle normal patterns because it over-adapted to drifted data | From-scratch. This is what happens in our Day 39 |
| **Feature schema change** | New sensors added. Model expects 33 features, now there are 38 | Old model can't accept new input dimensions at all | Completely new architecture + training |

**In our Day 39:** The canary fails because accumulated drift from days 31-38 means fine-tuning from v3 produces a bad model. We rollback to v3 (still works fine for immediate predictions), then immediately launch a from-scratch retrain using the last 30 days of data → v4 @champion.

---

## 7. Big Data Stack

### Apache Kafka

| Aspect | Details |
|--------|---------|
| **What** | Distributed streaming platform. A message broker for real-time data |
| **Analogy** | Like a post office that never loses mail and delivers in order |
| **Key concepts** | **Topic** = named channel (like "dram_test_results"). **Producer** = writes data. **Consumer** = reads data. **Partition** = parallel lane within a topic |
| **Our usage** | Producer writes 5M test records/day → topic `dram-yield-data` → Consumer reads and writes to Parquet for Spark |
| **Why not just write files?** | Kafka provides: ordering guarantees, replay capability (re-read old data), back-pressure (if consumer is slow, producer waits), and fault tolerance (data replicated across partitions) |
| **Port** | 9092 (broker), 9000 (Kafdrop UI) |

### Kafdrop

A web UI for viewing Kafka topics, messages, and consumer lag. Like "phpMyAdmin" but for Kafka. Accessible at `http://<EC2_IP>:9000`.

### Apache Zookeeper

Kafka's coordination service. Manages which Kafka brokers are alive, which is the leader for each partition, etc. Runs on port 2181. You don't interact with it directly.

### Apache Spark

| Aspect | Details |
|--------|---------|
| **What** | Distributed computing engine for large-scale data processing. Splits data across multiple workers |
| **Analogy** | Like having 10 accountants split a million receipts and each processes 100K in parallel |
| **Key concepts** | **Driver** = coordinator. **Executor/Worker** = does the actual computation. **DataFrame** = distributed table (like pandas but across multiple machines). **Stage** = unit of parallel work |
| **Our usage** | 5-stage ETL pipeline: Read Parquet → Feature engineering → Quality checks → Drift detection → Write output |
| **Why not pandas?** | Pandas loads everything into RAM on one machine. With 5M rows/day × 33 features × 40 days, pandas would need 50+ GB RAM. Spark distributes across workers + only loads what it needs |
| **Ports** | 8080 (Spark master UI), 7077 (master RPC), 8081/8082 (worker UIs) |

### Our Spark ETL Pipeline (5 Stages)

```
Stage 1: Read     — Load Parquet from Kafka consumer output
Stage 2: Engineer — Compute 7 physics-based features (retention×temp, etc.)
Stage 3: Quality  — Null checks, range validation, outlier flagging
Stage 4: Drift    — PSI/KL/KS per feature vs training baseline
Stage 5: Write    — Save processed data + drift report to S3
```

### Apache Airflow

| Aspect | Details |
|--------|---------|
| **What** | Workflow orchestration platform. Schedules and monitors data pipelines |
| **Analogy** | Like a factory production scheduler that says "Run Step A at 6AM, then Step B after A finishes, then Step C if B's output is good" |
| **Key concepts** | **DAG** = Directed Acyclic Graph (workflow definition). **Task** = one unit of work. **Operator** = task template (PythonOperator, BashOperator, etc.). **Schedule** = when/how often to run |
| **Our usage** | 3 DAGs orchestrate the entire 40-day simulation |
| **Port** | 8080 (Airflow web UI — see DAG runs, task logs, trigger manual runs) |

### Our 3 Airflow DAGs

| DAG | Trigger | Tasks | Purpose |
|-----|---------|-------|---------|
| `dag_daily_yield_pipeline` | Daily (or triggered by master) | 8 tasks: generate data → Kafka produce → Spark ETL → drift check → decide retrain → trigger or skip | The daily heartbeat — data processing + monitoring |
| `dag_retrain_pipeline` | Triggered by daily DAG when drift detected | 6 tasks: determine training window → Spark ETL on window → train model → canary test → promote/rollback | Handles retraining + safe deployment |
| `dag_simulation_master` | Manual trigger (once) | Loops 40 days, triggering daily DAG each iteration | The master controller that drives the entire simulation |

### ETL

**ETL = Extract, Transform, Load.** The standard pattern for data pipelines:
- **Extract:** Pull raw data from source (Kafka topic)
- **Transform:** Clean, engineer features, validate quality
- **Load:** Write processed data to destination (S3/Parquet)

---

## 8. Cloud & AWS Services

### AWS Service Glossary

| Service | Full Name | What It Is | Analogy | Our Usage |
|---------|-----------|-----------|---------|-----------|
| **EC2** | Elastic Compute Cloud | Virtual servers (VMs) in the cloud. You pick CPU, RAM, storage, OS | Renting a computer by the hour | t3.medium runs the 40-day simulation (Kafka+Spark+Airflow) |
| **S3** | Simple Storage Service | Object storage — stores any file, any size, 99.999999999% durability | Google Drive but for code, unlimited capacity | Model weights, MLflow artifacts, processed data, DVC remote |
| **RDS** | Relational Database Service | Managed SQL databases (PostgreSQL, MySQL). AWS handles backups, patches, scaling | Having a DBA on call 24/7 | PostgreSQL for MLflow experiment tracking |
| **ECR** | Elastic Container Registry | Docker image registry, like Docker Hub but private on AWS | Private warehouse for your Docker images | Stores our built Docker image, GitHub CI/CD pushes here |
| **IAM** | Identity and Access Management | Permission system. Controls who can do what on your AWS account | Building access badges — different badges open different doors | `p053-cicd-user` with EC2+S3+RDS+ECR permissions |
| **VPC** | Virtual Private Cloud | Isolated network in AWS. Your own private network segment | Your office building's internal network | Default VPC — EC2 and RDS communicate privately |
| **AMI** | Amazon Machine Image | Snapshot of an OS + software. Template for launching EC2 instances | A "ghost image" for reinstalling a computer | Amazon Linux 2023 (lightweight, free) |

### IAM Concepts

| Term | What It Means |
|------|--------------|
| **Root account** | The email you signed up with. Has unlimited god-mode access. NEVER use for daily work |
| **IAM User** | A sub-account with limited permissions. `p053-cicd-user` is our IAM user |
| **Policy** | A JSON document defining what actions are allowed. `AmazonS3FullAccess` allows all S3 operations |
| **Access Key** | A long-lived credential (ID + Secret) for programmatic access. Like a username + password for the CLI |
| **MFA** | Multi-Factor Authentication. Second factor (phone app code) required to log in. CRITICAL for root |
| **Role** | Permissions attached to a service, not a user. EC2 instances can "assume" a role to access S3 |

### EC2 Instance Types

| Type | vCPU | RAM | Use Case | Cost/hr |
|------|------|-----|----------|---------|
| **t3.micro** | 2 | 1 GB | Testing, SSH jump box | $0.0104 |
| **t3.medium** | 2 | 4 GB | Our simulation (Docker + data pipeline) | $0.0416 |
| **t3.large** | 2 | 8 GB | If 4 GB isn't enough for Spark | $0.0832 |
| **g4dn.xlarge** | 4 | 16 GB + T4 GPU | Retraining on AWS | $0.526 |
| **g5.xlarge** | 4 | 16 GB + A10G GPU | Faster retraining | $1.006 |

### S3 Concepts

| Term | Meaning |
|------|---------|
| **Bucket** | Top-level container (like a drive). Globally unique name: `p053-mlflow-artifacts` |
| **Object** | Any file stored in S3. Identified by key (path): `models/v1/hybrid_best.pt` |
| **Versioning** | Keep all versions of every object. Accidental delete → restore previous version |
| **Region** | Where data is physically stored. `us-west-2` = Oregon data center |

### AWS Region Selection

| Region | Code | Distance from Singapore | Latency | EC2 Cost (t3.medium) |
|--------|------|----------------------|---------|---------------------|
| Singapore | ap-southeast-1 | 0 km | ~1 ms | $0.052/hr |
| Oregon (US-West) | us-west-2 | 15,000 km | ~180 ms | $0.0416/hr |
| Mumbai | ap-south-1 | 4,000 km | ~50 ms | $0.0456/hr |

**Why us-west-2 for us?** Our simulation has zero real-time users — latency is irrelevant. We're running batch jobs (Airflow → Kafka → Spark), not serving a website. us-west-2 is 20% cheaper than ap-southeast-1 and has the broadest service availability. For a production app serving Singaporean users, you'd choose ap-southeast-1.

---

## 9. Docker & Containerization

### Core Concepts

| Term | What It Is | Analogy |
|------|-----------|---------|
| **Container** | Lightweight isolated environment with your app + dependencies | A lunchbox — everything needed for the meal, no kitchen required |
| **Image** | Blueprint for a container. Immutable. Built from Dockerfile | A recipe — you can make unlimited identical lunchboxes from it |
| **Dockerfile** | Text file with instructions to build an image | The recipe card |
| **Docker Compose** | Tool for defining multi-container applications | A meal plan — "I need lunchbox A, lunchbox B, and lunchbox C, and they share a table" |
| **Volume** | Persistent storage that survives container restarts | The fridge — data persists even if you throw away the lunchbox |
| **Registry** | Server that stores Docker images (DockerHub, ECR, GHCR) | The cookbook library |

### Our Docker Stacks

| Stack | File | Services | RAM Needed | Purpose |
|-------|------|----------|-----------|---------|
| **Local (6 services)** | `deploy/docker/docker-compose.yml` | API, MLflow, PostgreSQL, Redis, Prometheus, Grafana | ~3 GB | Local development and testing |
| **Big Data (14 services)** | `deploy/docker-compose-bigdata.yml` | Above + Kafka, Zookeeper, Kafdrop, Spark master + 2 workers, Airflow (webserver + scheduler + postgres + init), LocalStack | ~10-12 GB | Full simulation on EC2 |
| **AWS (5 services)** | `deploy/aws/docker-compose-aws.yml` | API, Redis, MLflow, Prometheus, Grafana (uses external RDS + S3) | ~2 GB | Lightweight AWS deployment |

---

## 10. CI/CD & DevOps

### What Is CI/CD?

| Term | Full Name | What It Does |
|------|-----------|-------------|
| **CI** | Continuous Integration | Automatically test code on every push. "Does it still work?" |
| **CD** | Continuous Delivery/Deployment | Automatically build + deploy after tests pass. "Ship it!" |

### Our CI/CD Pipeline (GitHub Actions)

```
git push → GitHub Actions triggers →
  Job 1: test     → ruff lint + pytest (20/20 tests)
  Job 2: build    → Docker image → GHCR (GitHub Container Registry)
  Job 3: deploy   → kubectl apply (only on git tags like v1.0.0)
  Job 4: ecr-push → Push Docker image to AWS ECR (only on git tags)
```

### GitHub Secrets Types

| Type | Scope | Who Can Access | Our Choice |
|------|-------|---------------|------------|
| **Repository secrets** | Only this one repo (053_dram_memory_yield_mlops) | Only CI/CD for this repo | ✅ **We use this** — secrets are project-specific |
| **Environment secrets** | A named environment within a repo (e.g., "production", "staging") | Only jobs targeting that environment | Overkill for us — we have one environment |
| **Organization secrets** | All repos in the AIML-Engineering-Lab org | ALL repos in the org | ❌ Too broad — we don't want P053's AWS keys accessible from other project repos |

**Use Repository secrets.** Each project gets its own AWS credentials.

### Key DevOps Terms

| Term | What It Means |
|------|--------------|
| **Infrastructure as Code (IaC)** | Define infrastructure in files, not clicking UI buttons. Our docker-compose files, K8s manifests, setup_aws.sh |
| **GitOps** | Git is the source of truth. Push to main → auto-deploy. We use this via GitHub Actions |
| **Health Check** | Automated test that verifies a service is working. Our API has `/health` endpoint |
| **Scaling** | Adding more instances to handle more traffic. Our K8s HPA: 2→8 pods based on CPU |
| **Blue-Green deployment** | Two identical environments — switch traffic between them for zero-downtime deploys |
| **Rolling update** | Replace pods one at a time. Old pod down only after new pod is healthy |

---

## 11. Our 40-Day Simulation — How It All Connects

### Daily Inference Strategy

**Q: Do we inference on that day's data only?**

Yes. Each day, `streaming_data_generator.py` generates 5M NEW rows representing that day's manufacturing output. The model predicts which dies will fail. These predictions go to:
1. Prometheus (metrics: prediction distribution, latency)
2. MLflow (logged as artifacts)
3. Spark (drift detection against training baseline)

We do NOT re-inference on previous days' data. Each day's data is independent — just like a real fab producing new wafers daily.

### Retraining Strategy

**Q: When drift is detected, do we train on just that day or previous days too?**

Neither — we train on a **sliding window**:

| Retrain Event | Training Window | Why |
|--------------|----------------|-----|
| Day 20 (moderate drift) | Last 14 days (Day 7-20) | Recent enough to capture current distribution, long enough for the model to generalize |
| Day 31 (severe drift) | Last 21 days (Day 11-31) | Wider window because severe drift means we need more diverse examples |
| Day 39 (from scratch) | Last 30 days (Day 10-39) | Maximum window because we're starting fresh — need everything |

**Why not just today's data?** 5M rows from one day might have specific batch effects. A 14-30 day window smooths out day-to-day noise while capturing the current data distribution.

**Why not ALL data from Day 1?** Because the whole point is that old data is no longer representative. Day 1 data is "pre-drift" — training on it would teach the model outdated patterns.

### Versioning Strategy

**Q: Do we fine-tune and version every day?**

**No.** We do NOT retrain every day. The daily pipeline:

```
Every day:
  1. Generate 5M rows          ← always
  2. Kafka → Spark ETL         ← always
  3. Inference with @champion   ← always
  4. Drift detection            ← always
  5. IF drift > threshold:      ← only sometimes
       → Trigger retrain DAG
       → New model version
       → Canary test
       → Promote/reject
  6. ELSE: log metrics, done    ← most days
```

**Days 2-19:** No drift detected → NO retraining → v1 stays as @champion for 19 days.
**Day 20:** Drift threshold crossed → retrain → v2 created → canary passes → v2 becomes @champion.
**Days 21-30:** No additional drift → v2 handles it fine → no retraining.
**Day 31:** Severe drift → retrain → v3.
**Days 32-38:** v3 handles it.
**Day 39:** Deliberately bad model fails canary → rollback + fresh v4.

**Model version history:**
```
v1 (Day 1)  → @champion for 19 days
v2 (Day 20) → @champion for 11 days
v3 (Day 31) → @champion for 8 days
v4 (Day 39) → @champion for 2 days
```

**Total: 4 models in 40 days.** In a real fab, you might retrain monthly (12 models/year) unless drift events force more frequent updates.

### 100GB+ Data — Storage Strategy

For data larger than your laptop and Google Drive can hold:

| Data Type | Size | Where to Store | Cost |
|-----------|------|---------------|------|
| Training data (Day 1) | ~2 GB (preprocessed NPZ) | S3 + DVC | $0.05/mo |
| Daily production data (5M rows/day × 40 days) | ~40 GB total Parquet | S3 only (generated on EC2, stays on EC2/S3) | $0.92/mo |
| Model weights (4 versions) | ~5 MB total | S3 via MLflow | negligible |
| MLflow artifacts | ~500 MB | S3 | $0.01/mo |
| If you scale to 100GB+ | 100 GB | S3 | **$2.30/mo** |

**Key insight:** The 40GB of daily data is generated ON EC2 and processed ON EC2. It never touches your laptop or Google Drive. Only the initial training data (~2 GB) and model weights (~5 MB) need to move between Colab ↔ Drive ↔ S3.

For 100GB+ datasets:
```
Generate on EC2 → write directly to S3 → Spark reads from S3
                                         ↓
                               Never touches your laptop
```

S3 cost: $0.023/GB/month = $2.30/month for 100 GB. Cheapest cloud storage available.

---

*Last updated: April 5, 2026 — Comprehensive terminology guide for P053 DRAM Memory Yield Predictor*
