#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# AngelCare — Deploy to Nebius (runs FROM your Mac)
# ─────────────────────────────────────────────────────────────
# Uploads code to a Nebius GPU instance and starts the
# inference server with optional cascade and VSS.
#
# Usage:
#   bash scripts/deploy.sh <NEBIUS_IP> --cascade
#   bash scripts/deploy.sh <NEBIUS_IP> --cascade --with-vss
#   bash scripts/deploy.sh <NEBIUS_IP> --cascade --camera-ip <CAMERA_IP>
#
# Environment:
#   SSH_USER     — SSH username on the Nebius instance (default: $USER)
#   PROJECT_DIR  — local project path (default: current directory)
# ─────────────────────────────────────────────────────────────
set -euo pipefail

NEBIUS_IP=""
CAMERA_IP=""
REMOTE_FLAGS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --camera-ip) CAMERA_IP="$2"; shift 2 ;;
    --cascade|--with-vss)
      REMOTE_FLAGS="$REMOTE_FLAGS $1"; shift ;;
    --port)
      REMOTE_FLAGS="$REMOTE_FLAGS --port $2"; shift 2 ;;
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
  echo "Usage: $0 NEBIUS_IP [--cascade] [--with-vss] [--camera-ip IP]"
  echo ""
  echo "Examples:"
  echo "  $0 51.250.X.X --cascade"
  echo "  $0 51.250.X.X --cascade --camera-ip 192.168.1.42"
  echo "  $0 51.250.X.X --cascade --with-vss"
  echo ""
  echo "Environment:"
  echo "  SSH_USER=ubuntu    Override SSH username (default: \$USER)"
  echo "  PROJECT_DIR=./     Override local project path"
  exit 1
fi

REMOTE_USER="${SSH_USER:-$USER}"
LOCAL_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

echo "================================================"
echo "  AngelCare — Deploying to $REMOTE_USER@$NEBIUS_IP"
echo "================================================"

# ── Test SSH ────────────────────────────────────────────
echo ""
echo "[1/3] Testing SSH connection..."
ssh -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE_USER@$NEBIUS_IP" "echo 'OK'" || {
  echo "ERROR: Cannot SSH to $REMOTE_USER@$NEBIUS_IP"
  echo "Make sure the instance is running and your SSH key is configured."
  exit 1
}

# ── Upload code ─────────────────────────────────────────
echo ""
echo "[2/3] Uploading project..."
rsync -avz --progress \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='.claude/' \
  --exclude='training_dataset/' \
  --exclude='videos/' \
  --exclude='videos_compressed/' \
  --exclude='website/' \
  --exclude='node_modules/' \
  --exclude='Screenshot*' \
  "$LOCAL_DIR/" \
  "$REMOTE_USER@$NEBIUS_IP:~/angelcare/"

# Upload .env separately (contains tokens)
if [ -f "$LOCAL_DIR/.env" ]; then
  scp "$LOCAL_DIR/.env" "$REMOTE_USER@$NEBIUS_IP:~/angelcare/.env"
fi

# ── Run setup on Nebius ─────────────────────────────────
echo ""
echo "[3/3] Running setup on Nebius..."
ssh -t "$REMOTE_USER@$NEBIUS_IP" "bash ~/angelcare/scripts/setup_inference.sh $REMOTE_FLAGS"

# ── Post-deploy instructions ────────────────────────────
echo ""
echo "================================================"
echo "  Deployment complete!"
echo ""
echo "  To connect your camera, open a NEW terminal and run:"
if [ -n "$CAMERA_IP" ]; then
  echo "    ssh -R 8554:$CAMERA_IP:554 $REMOTE_USER@$NEBIUS_IP"
else
  echo "    ssh -R 8554:<CAMERA_IP>:554 $REMOTE_USER@$NEBIUS_IP"
fi
echo ""
echo "  Then on the dashboard → Livestream:"
echo "    Camera URL: rtsp://<user>:<pass>@localhost:8554/stream2"
echo "================================================"
