#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# P053 — Kaggle Backend Setup Guide
# ═══════════════════════════════════════════════════════════════
# Run this script ONCE to verify Kaggle is configured correctly.
# Usage: bash scripts/kaggle_training/setup_kaggle_env.sh
# ═══════════════════════════════════════════════════════════════

set -e

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " P053 — Kaggle Backend Setup Checker"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Check kaggle package ──────────────────────────────────────────────
echo "[1/6] Checking kaggle Python package..."
if python -m kaggle --version 2>/dev/null; then
    echo "      ✅ kaggle package installed"
else
    echo "      ❌ Not installed. Run: pip install kaggle"
    exit 1
fi

# ── Step 2: Check ~/.kaggle/kaggle.json ──────────────────────────────────────
echo ""
echo "[2/6] Checking ~/.kaggle/kaggle.json..."
if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    KAGGLE_USER=$(python -c "import json; d=json.load(open('$HOME/.kaggle/kaggle.json')); print(d['username'])" 2>/dev/null || echo "")
    if [ -n "$KAGGLE_USER" ]; then
        echo "      ✅ Found. Username: $KAGGLE_USER"
    else
        echo "      ❌ File exists but username not readable."
        exit 1
    fi
else
    echo "      ❌ Not found."
    echo ""
    echo "      How to get your API key:"
    echo "        1. Go to: https://www.kaggle.com/settings/account"
    echo "        2. Scroll to 'API' section → 'Create New API Token'"
    echo "        3. This downloads kaggle.json"
    echo "        4. Run: mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "        5. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# ── Step 3: Verify API auth ───────────────────────────────────────────────────
echo ""
echo "[3/6] Verifying Kaggle API authentication..."
if python -c "from kaggle.api.kaggle_api_extended import KaggleApiExtended; api=KaggleApiExtended(); api.authenticate(); print('OK')" 2>/dev/null | grep -q "OK"; then
    echo "      ✅ API authentication successful"
else
    echo "      ❌ API authentication failed. Check your kaggle.json."
    exit 1
fi

# ── Step 4: Check kernel-metadata.json ───────────────────────────────────────
echo ""
echo "[4/6] Checking kernel-metadata.json..."
KERNEL_META="scripts/kaggle_training/kernel-metadata.json"
if [ -f "$KERNEL_META" ]; then
    KERNEL_ID=$(python -c "import json; d=json.load(open('$KERNEL_META')); print(d['id'])" 2>/dev/null || echo "")
    if echo "$KERNEL_ID" | grep -q "YOUR_KAGGLE_USERNAME"; then
        echo "      ❌ Still has placeholder username."
        echo ""
        echo "      ACTION REQUIRED:"
        echo "        Edit $KERNEL_META"
        echo "        Replace 'YOUR_KAGGLE_USERNAME' with: $KAGGLE_USER"
        echo ""
        echo "        Or run this command:"
        echo "        sed -i '' 's/YOUR_KAGGLE_USERNAME/$KAGGLE_USER/g' $KERNEL_META"
        exit 1
    else
        echo "      ✅ kernel-metadata.json looks good. Kernel ID: $KERNEL_ID"
    fi
else
    echo "      ❌ kernel-metadata.json not found at $KERNEL_META"
    exit 1
fi

# ── Step 5: Check Kaggle environment variables ────────────────────────────────
echo ""
echo "[5/6] Checking required Kaggle environment variables..."
echo "      These must be set at: https://www.kaggle.com/settings/account"
echo "      (Scroll to 'Environment Variables' section)"
echo ""
echo "      Required variables:"
echo "        AWS_ACCESS_KEY_ID       — for S3 upload from kernel"
echo "        AWS_SECRET_ACCESS_KEY   — for S3 upload from kernel"
echo "        AWS_DEFAULT_REGION      — us-west-2"
echo "        S3_BUCKET               — p053-mlflow-artifacts"
echo "        GITHUB_REPO_URL         — https://github.com/AIML-Engineering-Lab/053_dram_memory_yield_mlops.git"
echo ""
echo "      ⚠️  These cannot be verified from local machine."
echo "         You must set them manually on Kaggle's website."

# ── Step 6: Test kernel push (dry-run) ───────────────────────────────────────
echo ""
echo "[6/6] Validating kernel directory structure..."
KERNEL_DIR="scripts/kaggle_training"
for required_file in "kaggle_train_kernel.py" "kernel-metadata.json"; do
    if [ -f "$KERNEL_DIR/$required_file" ]; then
        echo "      ✅ $required_file"
    else
        echo "      ❌ Missing: $KERNEL_DIR/$required_file"
        exit 1
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " ✅ Kaggle backend is configured!"
echo ""
echo " How to trigger a training job:"
echo "   python -c \""
echo "   from src.kaggle_backend import trigger_training_kernel"
echo "   trigger_training_kernel(run_name='test-retrain', epochs=3)"
echo "   \""
echo ""
echo " Add KAGGLE_USERNAME to your .env:"
echo "   echo 'KAGGLE_USERNAME=$KAGGLE_USER' >> .env"
echo ""
echo " Fallback chain is now:"
echo "   AWS EC2 → Kaggle Kernels (auto) → 2hr wait → Colab → 2hr wait → Local Mac"
echo "═══════════════════════════════════════════════════════════════"
