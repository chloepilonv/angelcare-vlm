#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# AngelCare — Deploy Training to Nebius (runs FROM your Mac)
# ─────────────────────────────────────────────────────────────
# Uploads training data to a Nebius A100 instance and runs
# the full QLoRA SFT pipeline.
#
# Usage:
#   bash scripts/deploy_training.sh <NEBIUS_IP>
#   bash scripts/deploy_training.sh <NEBIUS_IP> --push-to-hf --download-weights
#
# Environment:
#   SSH_USER     — SSH username on the Nebius instance (default: $USER)
#   PROJECT_DIR  — local project path (default: current directory)
#   HF_REPO      — HuggingFace repo for --push-to-hf (default: none, must set)
# ─────────────────────────────────────────────────────────────
set -euo pipefail

NEBIUS_IP=""
DOWNLOAD_WEIGHTS=false
REMOTE_FLAGS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --download-weights) DOWNLOAD_WEIGHTS=true; shift ;;
    --skip-baseline|--skip-eval|--push-to-hf)
      REMOTE_FLAGS="$REMOTE_FLAGS $1"; shift ;;
    *)
      if [ -z "$NEBIUS_IP" ]; then
        NEBIUS_IP="$1"; shift
      else
        echo "Unknown option: $1"; exit 1
      fi
      ;;
  esac
done

if [ -z "$NEBIUS_IP" ]; then
  echo "Usage: $0 NEBIUS_IP [--push-to-hf] [--download-weights] [--skip-baseline]"
  echo ""
  echo "Examples:"
  echo "  $0 51.250.X.X --push-to-hf"
  echo "  $0 51.250.X.X --push-to-hf --download-weights"
  echo "  $0 51.250.X.X --skip-baseline --push-to-hf"
  echo ""
  echo "Environment:"
  echo "  SSH_USER=ubuntu    Override SSH username (default: \$USER)"
  echo "  HF_REPO=user/repo  HuggingFace repo for weight upload"
  exit 1
fi

REMOTE_USER="${SSH_USER:-$USER}"
LOCAL_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# Pass HF_REPO to remote if set
if [ -n "${HF_REPO:-}" ]; then
  REMOTE_FLAGS="$REMOTE_FLAGS --hf-repo $HF_REPO"
fi

echo "================================================"
echo "  AngelCare — Training Deployment to $REMOTE_USER@$NEBIUS_IP"
echo "================================================"

# ── Test SSH ────────────────────────────────────────────
echo ""
echo "[1/4] Testing SSH connection..."
ssh -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE_USER@$NEBIUS_IP" "echo 'OK'" || {
  echo "ERROR: Cannot SSH to $REMOTE_USER@$NEBIUS_IP"
  exit 1
}

# ── Upload training data and scripts ───────────────────
echo ""
echo "[2/4] Uploading training data..."

rsync -avz --progress \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='.claude/' \
  --exclude='website/' \
  --exclude='node_modules/' \
  --exclude='videos/' \
  --exclude='Screenshot*' \
  "$LOCAL_DIR/" \
  "$REMOTE_USER@$NEBIUS_IP:~/angelcare/"

# Upload .env
if [ -f "$LOCAL_DIR/.env" ]; then
  scp "$LOCAL_DIR/.env" "$REMOTE_USER@$NEBIUS_IP:~/angelcare/.env"
fi

# ── Run training pipeline ──────────────────────────────
echo ""
echo "[3/4] Starting training pipeline on Nebius..."
ssh -t "$REMOTE_USER@$NEBIUS_IP" "bash ~/angelcare/scripts/setup_training.sh $REMOTE_FLAGS"

# ── Download weights ────────────────────────────────────
if [ "$DOWNLOAD_WEIGHTS" = true ]; then
  echo ""
  echo "[4/4] Downloading trained weights..."
  mkdir -p "$LOCAL_DIR/training_dataset/outputs/angelcare_sft"
  scp -r "$REMOTE_USER@$NEBIUS_IP:~/angelcare/training_dataset/outputs/angelcare_sft/final" \
    "$LOCAL_DIR/training_dataset/outputs/angelcare_sft/final"
  echo "Weights saved to training_dataset/outputs/angelcare_sft/final"
else
  echo ""
  echo "[4/4] Skipping weight download (use --download-weights to fetch)"
fi

echo ""
echo "================================================"
echo "  Training deployment complete!"
echo ""
echo "  REMINDER: Shut down the A100 instance!"
echo "  https://console.nebius.ai/"
echo "================================================"
