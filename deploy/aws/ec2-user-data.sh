#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# P053 — EC2 g4dn.xlarge Bootstrap Script (User Data)
# ═══════════════════════════════════════════════════════════════
#
# Paste this as EC2 "User Data" when launching g4dn.xlarge.
# It installs everything needed and starts the full big data stack.
#
# Instance: g4dn.xlarge (4 vCPU, 16GB RAM, T4 GPU 16GB)
# AMI: Amazon Linux 2023 (al2023-ami-*)
# Cost: $0.526/hr on-demand
#
# What this script does:
#   1. Install Docker + Docker Compose
#   2. Install NVIDIA drivers + nvidia-container-toolkit
#   3. Clone the repo
#   4. Build GPU-enabled Airflow image
#   5. Generate preprocessed data (for retraining)
#   6. Start the full stack
#   7. Create completion marker for monitoring
#
# After launch, SSH in and monitor:
#   ssh -i ~/.ssh/p053-key.pem ec2-user@<public-ip>
#   tail -f /var/log/cloud-init-output.log
#
# Total setup time: ~15-20 minutes
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
exec > >(tee /var/log/p053-bootstrap.log) 2>&1

echo "═══════════════════════════════════════════════════════════════"
echo "P053 — Bootstrap starting at $(date)"
echo "═══════════════════════════════════════════════════════════════"

# ─── 1. System updates ───
echo "[1/8] Updating system packages..."
dnf update -y
dnf install -y git docker python3.11 python3.11-pip htop tmux jq

# ─── 2. Docker ───
echo "[2/8] Starting Docker..."
systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user

# Install Docker Compose plugin
DOCKER_COMPOSE_VERSION="v2.27.0"
mkdir -p /usr/local/lib/docker/cli-plugins
curl -SL "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-linux-x86_64" \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

echo "  Docker: $(docker --version)"
echo "  Compose: $(docker compose version)"

# ─── 3. NVIDIA drivers + container toolkit ───
echo "[3/8] Installing NVIDIA drivers..."
dnf install -y kernel-devel-$(uname -r) gcc

# NVIDIA driver (Amazon Linux 2023)
if ! command -v nvidia-smi &>/dev/null; then
    dnf install -y https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo 2>/dev/null || true
    dnf install -y nvidia-driver-latest-dkms cuda-toolkit-12-4
fi

# NVIDIA Container Toolkit
if ! command -v nvidia-container-toolkit &>/dev/null; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
        tee /etc/yum.repos.d/nvidia-container-toolkit.repo
    dnf install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected yet')"

# ─── 4. Clone repo ───
echo "[4/8] Cloning repository..."
PROJECT_DIR="/home/ec2-user/053_memory_yield_predictor"

if [ ! -d "$PROJECT_DIR" ]; then
    git clone https://github.com/AIML-Engineering-Lab/053_dram_memory_yield_mlops.git "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"
chown -R ec2-user:ec2-user "$PROJECT_DIR"

# ─── 5. Python environment ───
echo "[5/8] Setting up Python environment..."
python3.11 -m pip install --upgrade pip
python3.11 -m pip install -r requirements.txt

# ─── 6. Build Airflow GPU image ───
echo "[6/8] Building p053-airflow-gpu Docker image..."
docker build -t p053-airflow-gpu -f deploy/aws/Dockerfile.airflow-gpu .

# ─── 7. Preprocess data (if not already done) ───
echo "[7/8] Generating preprocessed data..."
if [ ! -f "data/preprocessed_full.npz" ]; then
    echo "  Generating training data (this takes a few minutes)..."
    cd "$PROJECT_DIR"
    python3.11 -m src.preprocess --full || echo "  WARNING: preprocess failed, may need manual run"
fi

# ─── 8. Start the stack ───
echo "[8/8] Starting big data stack..."
cd "$PROJECT_DIR"

# Check if .env exists
if [ ! -f "deploy/aws/.env" ]; then
    echo "  WARNING: deploy/aws/.env not found!"
    echo "  Copy .env.aws.template to .env and fill in secrets before starting."
    echo "  Then run: docker compose -f deploy/aws/docker-compose-bigdata-aws.yml --env-file deploy/aws/.env up -d"
else
    docker compose -f deploy/aws/docker-compose-bigdata-aws.yml \
        --env-file deploy/aws/.env up -d
fi

# ─── Completion marker ───
echo "P053_BOOTSTRAP_COMPLETE=$(date +%s)" > /home/ec2-user/.p053_bootstrap_status
chown ec2-user:ec2-user /home/ec2-user/.p053_bootstrap_status

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "P053 Bootstrap COMPLETE at $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Services:"
echo "  Airflow:     http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8888"
echo "  MLflow:      http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5001"
echo "  Spark UI:    http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8080"
echo "  Grafana:     http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):3000"
echo "  Kafka UI:    http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):9000"
echo "  Prometheus:  http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):9090"
echo ""
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'check after reboot')"
echo ""
echo "Next steps:"
echo "  1. Ensure deploy/aws/.env is configured"
echo "  2. Trigger simulation: airflow dags trigger p053_simulation_master"
