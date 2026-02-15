#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# AngelCare — Inference Setup (runs ON Nebius H100)
# ─────────────────────────────────────────────────────────────
# Sets up venv, installs deps, optionally starts VSS, launches
# Cloudflare tunnel, and starts the Flask app.
#
# Usage:
#   bash scripts/setup_inference.sh --cascade
#   bash scripts/setup_inference.sh --cascade --with-vss
# ─────────────────────────────────────────────────────────────
set -euo pipefail

WITH_VSS=false
CASCADE=false
PORT=5000

while [[ $# -gt 0 ]]; do
  case $1 in
    --with-vss) WITH_VSS=true; shift ;;
    --cascade)  CASCADE=true; shift ;;
    --port)     PORT="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "================================================"
echo "  AngelCare — Inference Setup"
echo "  cascade=$CASCADE  vss=$WITH_VSS  port=$PORT"
echo "================================================"

# ── Verify GPU ──────────────────────────────────────────
echo ""
echo "[1/6] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "ERROR: No GPU found. Is this the right instance?"
  exit 1
}

# ── Create venv ─────────────────────────────────────────
echo ""
echo "[2/6] Setting up Python environment..."
if [ ! -d ~/venv ]; then
  python3 -m venv ~/venv
fi
source ~/venv/bin/activate

# ── Install system deps ────────────────────────────────
echo ""
echo "[3/6] Installing system dependencies..."
sudo apt-get update -qq && sudo apt-get install -y -qq python3-dev ffmpeg > /dev/null

# ── Install Python deps ────────────────────────────────
echo ""
echo "[4/6] Installing Python dependencies..."
pip install -q -r ~/angelcare/requirements.txt

# ── HuggingFace login ──────────────────────────────────
if [ -f ~/angelcare/.env ]; then
  source <(grep -E '^(HF_TOKEN|HF_API_TOKEN|NGC_API_KEY)=' ~/angelcare/.env | sed 's/ //g')
fi

if [ -n "${HF_TOKEN:-${HF_API_TOKEN:-}}" ]; then
  echo ""
  echo "[5/6] Logging into HuggingFace..."
  python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN:-${HF_API_TOKEN:-}}')"
else
  echo ""
  echo "[5/6] Skipping HF login (no HF_TOKEN in .env)"
fi

# ── Deploy VSS (optional) ─────────────────────────────
if [ "$WITH_VSS" = true ]; then
  echo ""
  echo "[6/6] Deploying VSS stack..."
  if [ -n "${NGC_API_KEY:-}" ]; then
    export NGC_API_KEY HF_TOKEN
    bash ~/angelcare/deploy/setup_vss.sh
  else
    echo "ERROR: --with-vss requires NGC_API_KEY in .env"
    exit 1
  fi
else
  echo ""
  echo "[6/6] Skipping VSS (not requested)"
fi

# ── Start Cloudflare tunnel ────────────────────────────
echo ""
echo "Starting Cloudflare tunnel..."
if ! command -v cloudflared &> /dev/null; then
  echo "Installing cloudflared..."
  curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -o /usr/local/bin/cloudflared 2>/dev/null || \
  sudo curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -o /usr/local/bin/cloudflared
  sudo chmod +x /usr/local/bin/cloudflared
fi

# Start tunnel in background, capture URL
cloudflared tunnel --url http://localhost:$PORT &> /tmp/cloudflared.log &
TUNNEL_PID=$!
echo "Cloudflare tunnel starting (PID $TUNNEL_PID)..."

# Wait for tunnel URL
for i in {1..15}; do
  TUNNEL_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cloudflared.log 2>/dev/null | head -1 || true)
  if [ -n "$TUNNEL_URL" ]; then break; fi
  sleep 2
done

# ── Build Flask command ────────────────────────────────
FLASK_CMD="python3 ~/angelcare/app.py --host 0.0.0.0 --port $PORT"
[ "$CASCADE" = true ] && FLASK_CMD="$FLASK_CMD --cascade"
[ "$WITH_VSS" = true ] && FLASK_CMD="$FLASK_CMD --vss-url http://localhost:8100"

echo ""
echo "================================================"
echo "  Setup complete!"
echo ""
if [ -n "${TUNNEL_URL:-}" ]; then
  echo "  Cloudflare tunnel: $TUNNEL_URL"
  echo "  Paste this URL in angelcare.info → Livestream → Backend Connection"
else
  echo "  Cloudflare tunnel: starting... check /tmp/cloudflared.log"
fi
echo ""
echo "  Starting Flask: $FLASK_CMD"
echo "================================================"
echo ""

$FLASK_CMD
