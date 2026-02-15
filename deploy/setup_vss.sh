#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# AngelCare — VSS Single-GPU Deployment on Nebius
# ─────────────────────────────────────────────────────────────
# Deploys NVIDIA Video Search & Summarization (VSS) on a single
# H100/A100 (80 GB) GPU using Docker Compose.
#
# Prerequisites:
#   - Ubuntu 22.04 with NVIDIA driver ≥ 580, CUDA ≥ 13.0
#   - Docker ≥ 27.5, Docker Compose ≥ 2.32
#   - NVIDIA Container Toolkit ≥ 1.13.5
#   - NGC API key (https://org.ngc.nvidia.com/)
#   - HuggingFace token (https://huggingface.co/settings/tokens)
#
# Usage:
#   export NGC_API_KEY=<your-key>
#   export HF_TOKEN=<your-token>
#   bash deploy/setup_vss.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ── Validate environment ────────────────────────────────────
if [ -z "${NGC_API_KEY:-}" ]; then
  echo "ERROR: NGC_API_KEY is not set. Export it before running this script."
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set. Export it before running this script."
  exit 1
fi

echo "=== AngelCare VSS Deployment ==="
echo ""

# ── Check GPU ───────────────────────────────────────────────
echo "Checking GPU..."
if ! nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. Ensure NVIDIA drivers are installed."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── NGC Docker login ────────────────────────────────────────
echo "Logging into NGC container registry..."
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
echo ""

# ── Shared config ───────────────────────────────────────────
export LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim}"
mkdir -p "$LOCAL_NIM_CACHE"

DEVICE="0"  # single GPU

# ── 1. Deploy Llama 3.1 8B (LLM) ───────────────────────────
echo "=== Starting Llama 3.1 8B LLM NIM (port 8007) ==="
docker rm -f angelcare-llm 2>/dev/null || true
docker run -d \
  --name angelcare-llm \
  -u "$(id -u)" \
  --gpus "\"device=${DEVICE}\"" \
  --shm-size=16GB \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -p 8007:8000 \
  -e NIM_LOW_MEMORY_MODE=1 \
  -e NIM_RELAX_MEM_CONSTRAINTS=1 \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:1.12.0

# ── 2. Deploy Embedding NIM ────────────────────────────────
echo "=== Starting Embedding NIM (port 8006) ==="
docker rm -f angelcare-embed 2>/dev/null || true
docker run -d \
  --name angelcare-embed \
  -u "$(id -u)" \
  --gpus "\"device=${DEVICE}\"" \
  --shm-size=16GB \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -p 8006:8000 \
  -e NIM_SERVER_PORT=8000 \
  -e NIM_MODEL_PROFILE="f7391ddbcb95b2406853526b8e489fedf20083a2420563ca3e65358ff417b10f" \
  -e NIM_TRT_ENGINE_HOST_CODE_ALLOWED=1 \
  nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:1.9.0

# ── 3. Deploy Reranker NIM ─────────────────────────────────
echo "=== Starting Reranker NIM (port 8005) ==="
docker rm -f angelcare-rerank 2>/dev/null || true
docker run -d \
  --name angelcare-rerank \
  -u "$(id -u)" \
  --gpus "\"device=${DEVICE}\"" \
  --shm-size=16GB \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -p 8005:8000 \
  -e NIM_SERVER_PORT=8000 \
  -e NIM_MODEL_PROFILE="f7391ddbcb95b2406853526b8e489fedf20083a2420563ca3e65358ff417b10f" \
  nvcr.io/nim/nvidia/llama-3.2-nv-rerankqa-1b-v2:1.7.0

# ── 4. Wait for NIM containers to be ready ──────────────────
echo ""
echo "=== Waiting for NIM containers to initialize ==="
echo "(This can take several minutes on first run as models are downloaded)"
echo ""

wait_for_service() {
  local name="$1"
  local port="$2"
  local max_wait=600  # 10 minutes
  local elapsed=0

  printf "  Waiting for %s (port %s)..." "$name" "$port"
  while ! curl -sf "http://localhost:${port}/v1/health/ready" &>/dev/null; do
    sleep 10
    elapsed=$((elapsed + 10))
    if [ "$elapsed" -ge "$max_wait" ]; then
      echo " TIMEOUT after ${max_wait}s"
      echo "  Check logs: docker logs angelcare-${name}"
      exit 1
    fi
    printf "."
  done
  echo " ready! (${elapsed}s)"
}

wait_for_service "llm" 8007
wait_for_service "embed" 8006
wait_for_service "rerank" 8005

# ── 5. Clone VSS repo and launch engine ─────────────────────
VSS_DIR="${HOME}/video-search-and-summarization"

if [ ! -d "$VSS_DIR" ]; then
  echo ""
  echo "=== Cloning VSS repository ==="
  git clone https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization.git "$VSS_DIR"
fi

cd "${VSS_DIR}/deploy/docker/local_deployment_single_gpu"

# Write .env for VSS engine
cat > .env <<EOF
NGC_API_KEY=${NGC_API_KEY}
HF_TOKEN=${HF_TOKEN}
INSTALL_PROPRIETARY_CODECS=true
EOF

echo ""
echo "=== Starting VSS engine (docker compose) ==="
docker compose up -d

# ── 6. Wait for VSS API ─────────────────────────────────────
echo ""
printf "  Waiting for VSS API (port 8100)..."
elapsed=0
while ! curl -sf "http://localhost:8100/health" &>/dev/null; do
  sleep 10
  elapsed=$((elapsed + 10))
  if [ "$elapsed" -ge 600 ]; then
    echo " TIMEOUT"
    echo "  Check logs: docker compose logs"
    exit 1
  fi
  printf "."
done
echo " ready! (${elapsed}s)"

# ── Done ─────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  AngelCare VSS deployment complete!"
echo ""
echo "  VSS API:  http://localhost:8100"
echo "  VSS UI:   http://localhost:9100"
echo "  LLM NIM:  http://localhost:8007"
echo "=========================================="
