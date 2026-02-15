#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# AngelCare — Training Setup (runs ON Nebius A100)
# ─────────────────────────────────────────────────────────────
# Installs deps, generates dataset, runs baseline eval,
# trains QLoRA SFT, evaluates, and optionally pushes to HF.
#
# Usage:
#   bash scripts/setup_training.sh
#   bash scripts/setup_training.sh --skip-baseline --push-to-hf
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SKIP_BASELINE=false
SKIP_EVAL=false
PUSH_TO_HF=false
HF_REPO=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-baseline) SKIP_BASELINE=true; shift ;;
    --skip-eval)     SKIP_EVAL=true; shift ;;
    --push-to-hf)    PUSH_TO_HF=true; shift ;;
    --hf-repo)       HF_REPO="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "================================================"
echo "  AngelCare — Training Pipeline"
echo "  skip_baseline=$SKIP_BASELINE  push_to_hf=$PUSH_TO_HF"
echo "================================================"

# ── Verify GPU ──────────────────────────────────────────
echo ""
echo "[1/8] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Create venv ─────────────────────────────────────────
echo ""
echo "[2/8] Setting up Python environment..."
if [ ! -d ~/venv ]; then
  python3 -m venv ~/venv
fi
source ~/venv/bin/activate

# ── Install deps ────────────────────────────────────────
echo ""
echo "[3/8] Installing dependencies..."
sudo apt-get update -qq && sudo apt-get install -y -qq python3-dev > /dev/null
pip install -q torch torchvision transformers">=4.57.0" accelerate \
  huggingface_hub qwen-vl-utils decord trl peft bitsandbytes datasets

# ── HuggingFace login ──────────────────────────────────
if [ -f ~/angelcare/.env ]; then
  source <(grep -E '^(HF_TOKEN|HF_API_TOKEN)=' ~/angelcare/.env | sed 's/ //g')
fi

if [ -n "${HF_TOKEN:-${HF_API_TOKEN:-}}" ]; then
  echo ""
  echo "[4/8] Logging into HuggingFace..."
  python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN:-${HF_API_TOKEN:-}}')"
else
  echo "ERROR: No HF_TOKEN found in .env"
  exit 1
fi

# ── Generate dataset ────────────────────────────────────
echo ""
echo "[5/8] Generating dataset..."
cd ~/angelcare/training_dataset
python3 prepare_llava_dataset.py
python3 prepare_llava_dataset.py --stats-only

# ── Baseline evaluation ────────────────────────────────
if [ "$SKIP_BASELINE" = false ]; then
  echo ""
  echo "[6/8] Running baseline evaluation (~20-30 min)..."
  python3 evaluate.py \
    --model nvidia/Cosmos-Reason2-8B \
    --dataset angelcare_llava_test.json \
    --output eval_baseline.json
else
  echo ""
  echo "[6/8] Skipping baseline (--skip-baseline)"
fi

# ── Train ───────────────────────────────────────────────
echo ""
echo "[7/8] Training QLoRA SFT (~3 min)..."
bash train.sh

# ── Evaluate fine-tuned model ───────────────────────────
if [ "$SKIP_EVAL" = false ]; then
  echo ""
  echo "[8/8] Evaluating fine-tuned model..."
  python3 evaluate.py \
    --model outputs/angelcare_sft/final \
    --dataset angelcare_llava_test.json \
    --output eval_finetuned.json

  # Compare results
  echo ""
  echo "── Results ──────────────────────────────────"
  python3 -c "
import json
base = json.load(open('eval_baseline.json')) if not $SKIP_BASELINE else None
ft = json.load(open('eval_finetuned.json'))
if base:
    print(f'Zero-shot:  {base[\"exact_accuracy\"]*100:.1f}% exact | {base[\"risk_accuracy\"]*100:.1f}% risk')
print(f'Fine-tuned: {ft[\"exact_accuracy\"]*100:.1f}% exact | {ft[\"risk_accuracy\"]*100:.1f}% risk')
if base:
    print(f'Delta:      +{(ft[\"exact_accuracy\"]-base[\"exact_accuracy\"])*100:.1f}pp exact | +{(ft[\"risk_accuracy\"]-base[\"risk_accuracy\"])*100:.1f}pp risk')
"
else
  echo ""
  echo "[8/8] Skipping evaluation (--skip-eval)"
fi

# ── Push to HuggingFace ────────────────────────────────
if [ "$PUSH_TO_HF" = true ]; then
  if [ -z "$HF_REPO" ]; then
    echo "ERROR: --push-to-hf requires HF_REPO environment variable or --hf-repo flag"
    echo "Example: HF_REPO=username/model-name bash scripts/deploy_training.sh <IP> --push-to-hf"
    exit 1
  fi
  echo ""
  echo "Pushing weights to HuggingFace ($HF_REPO)..."
  python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='outputs/angelcare_sft/final',
    repo_id='$HF_REPO',
    commit_message='AngelCare QLoRA SFT on GMDCSA-24'
)
print('Pushed to https://huggingface.co/$HF_REPO')
"
fi

echo ""
echo "================================================"
echo "  Training complete!"
echo "  Weights: ~/angelcare/training_dataset/outputs/angelcare_sft/final"
echo ""
echo "  REMINDER: Shut down this instance when done!"
echo "================================================"
