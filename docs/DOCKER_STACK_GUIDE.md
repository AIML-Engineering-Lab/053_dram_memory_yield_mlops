# Docker Stack Verification Guide

The local Docker Compose stack has 6 services. This guide covers everything from fixing the Docker CLI to verifying all services.

---

## Step 0 — Fix Docker CLI PATH (One-Time Fix)

Docker Desktop is installed and running, but the `docker` command is not in your terminal PATH. The old symlink at `/usr/local/bin/docker` points to a stale DMG mount. Fix it with **one of these two options:**

### Option A — Docker Desktop Settings (Recommended)

1. Open **Docker Desktop**
2. Click the **gear icon** (Settings) → **General**
3. Uncheck **"Install Docker CLI tools in the system PATH"** → Apply
4. Re-check **"Install Docker CLI tools in the system PATH"** → Apply & Restart
5. Quit and reopen your terminal

### Option B — Manual PATH in .zshrc

```bash
echo 'export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Verify the fix:

```bash
docker version --format '{{.Client.Version}}'
# Expected: 29.3.1 (or similar)

docker compose version
# Expected: Docker Compose version v5.1.1
```

> **If you don't do this fix**, use the full path as a workaround:
> `/Applications/Docker.app/Contents/Resources/bin/docker compose up -d`

---

## Step 1 — Start Docker Desktop

Open **Docker Desktop** from your Applications folder (the whale icon). Wait until the status bar shows "Engine running" (usually 20-30 seconds).

> **Note:** If containers from a previous session are already running (green dots), they auto-started because the compose file uses `restart: unless-stopped`. You can skip to Step 3 to verify, or proceed to Step 2 to rebuild.

---

## Step 2 — Start / Rebuild All 6 Services

```bash
cd /Users/rajendarmuddasani/AIML/55_LinkedIn/053_memory_yield_predictor/deploy/docker
```

**If containers are already running** (from Docker Desktop auto-restart): rebuild only the API container to pick up the matplotlib fix:

```bash
docker compose up -d --build api
```

**If no containers running** (fresh start):

```bash
docker compose up -d
```

Wait ~60 seconds for all containers to be healthy. Check status:

```bash
docker compose ps
```

Expected output (all Status = `Up ... (healthy)`):
```
NAME             STATUS                    PORTS
p053-api         Up 2 minutes (healthy)    0.0.0.0:8000->8000/tcp
p053-mlflow      Up 2 minutes (healthy)    0.0.0.0:5001->5000/tcp
p053-grafana     Up 2 minutes              0.0.0.0:3000->3000/tcp
p053-prometheus  Up 2 minutes              0.0.0.0:9090->9090/tcp
p053-mlflow-db   Up 2 minutes (healthy)    5432/tcp
p053-redis       Up 2 minutes (healthy)    0.0.0.0:6379->6379/tcp
```

---

## Step 3 — Open & Verify Each Service in Browser

### 3a. MLflow — http://localhost:5001

What you should see:
- MLflow UI with **Experiments** sidebar
- **Default** experiment should be visible

**To populate with A100 training results:**
```bash
cd /Users/rajendarmuddasani/AIML/55_LinkedIn/053_memory_yield_predictor
python3 -m src.retrolog_experiments
```

After retrolog, you should see:
- **Runs**: Day 1 A100 run with metrics: `val_auc_roc: 0.816`, `val_auc_pr: 0.054`, `val_f1: 0.127`
- **Params**: `model: HybridTransformerCNN`, `amp_dtype: bfloat16`, `epochs: 50`
- Click **Models** tab → register as `p053-yield-predictor`

---

### 3b. FastAPI (Inference API) — http://localhost:8000/docs

What you should see:
- Swagger UI with endpoints: `POST /predict`, `POST /predict/batch`, `GET /health`, `GET /model/info`

**Test a prediction:** click **POST /predict** → **Try it out** → paste:

```json
{
  "retention_time_ms": 45.2,
  "gate_oxide_thickness_a": 18.5,
  "vt_shift_mv": 8.1,
  "cell_leakage_fa": 12.3,
  "test_temp_c": 85.0,
  "trcd_ns": 13.75
}
```

Expected response (approximately):
```json
{"prediction": 0, "probability": 0.003, "label": "yield_pass", "model_version": "..."}
```

> **If you get 503 "Model not loaded":** The API container needs rebuilding — run `docker compose up -d --build api` from `deploy/docker/`. Check logs with `docker compose logs api`.

---

### 3c. Grafana — http://localhost:3000

Login: `admin` / `admin` (change password prompt — click **Skip**)

What you should see:
- Left sidebar → **Dashboards** → click **Browse** → look for **P053 DRAM Yield** dashboard
- If present: click to see inference latency, prediction count, drift metrics panels
- Go to **Explore** tab → datasource: `Prometheus` → query `up` → Execute → should show `1`

> **If no dashboards appear:** Go to **Dashboards → Import** → upload the JSON from `deploy/docker/grafana_dashboard.json`

---

### 3d. Prometheus — http://localhost:9090

What you should see:
- Query bar: type `up` and press **Execute** → all targets show value `1`
- Click **Status → Targets**: the FastAPI scrape target (`p053-api:8000/metrics`) should show `UP` in green

---

### 3e. Web Dashboard (Static — No Server Needed)

```bash
open /Users/rajendarmuddasani/AIML/55_LinkedIn/053_memory_yield_predictor/web/dashboard.html
```

Opens directly in browser. Shows 40-day simulation results: drift timeline, retrain events, model versions.

---

## Step 4 — Take Screenshots (For Portfolio/LinkedIn)

Open each service and take screenshots:
1. **MLflow** — experiments list with A100 training run
2. **FastAPI /docs** — Swagger UI with successful prediction response
3. **Grafana** — dashboard with inference metrics panels
4. **Prometheus** — targets page showing all UP
5. **Docker Desktop** — containers view (all 6 green dots)

---

## Stopping All Services

```bash
cd /Users/rajendarmuddasani/AIML/55_LinkedIn/053_memory_yield_predictor/deploy/docker
docker compose down
```

To also delete Postgres data volumes (full reset):
```bash
docker compose down -v
```

---

## Architecture Reference

```
Browser Requests
    │
    ▼
┌─────────────────┐    ┌───────────┐
│  FastAPI :8000   │◄──►│ Redis     │  (prediction cache, 5min TTL)
│  (p053-api)      │    │ :6379     │
└───────┬─────────┘    └───────────┘
        │ /metrics
        ▼
┌─────────────────┐    ┌───────────────┐
│ Prometheus :9090 │──►│ Grafana :3000  │
│ (scrapes 15s)    │    │ (dashboards)  │
└─────────────────┘    └───────────────┘

┌─────────────────┐    ┌───────────────┐
│ MLflow :5001     │──►│ PostgreSQL     │
│ (tracking UI)    │    │ (mlflow-db)   │
└─────────────────┘    └───────────────┘
```

**Docker Compose file:** `deploy/docker/docker-compose.yml`
**Dockerfile (multi-stage):** `deploy/docker/Dockerfile`
**Serving dependencies:** `requirements-serve.txt`
**Model files:** `src/artifacts/hybrid_best_a100.pt` (mounted read-only into container)

---

## Common Issues

| Issue | Fix |
|-------|-----|
| `docker: command not found` | See Step 0 — fix PATH or use full path `/Applications/Docker.app/Contents/Resources/bin/docker` |
| `docker-compose: command not found` | Modern Docker uses `docker compose` (space, no hyphen). Fix PATH per Step 0. |
| Port 5001 already in use | macOS AirPlay Receiver uses 5000/5001. Disable in System Settings → AirDrop & Handoff → AirPlay Receiver |
| FastAPI `/predict` returns 503 | Model not loaded. Rebuild: `docker compose up -d --build api`. Check logs: `docker compose logs api` |
| Grafana shows "No data" | Verify Prometheus targets are UP at http://localhost:9090/targets |
| MLflow 500 error on startup | PostgreSQL may not be ready. Wait 10s and refresh. Check: `docker compose logs mlflow-postgres` |
| Containers auto-started from last session | This is expected — `restart: unless-stopped` in compose. The state persists across Docker Desktop restarts. |

---
*Updated: April 13, 2026*
