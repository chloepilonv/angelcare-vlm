#!/bin/bash
# AngelCare — Post-Training (SFT) on Nebius
# ===========================================
# Fine-tunes Cosmos Reason 2 (8B) using QLoRA + TRL.
# Requires: 1x A100 (80GB) minimum.
#
# Setup:
#   1. pip install -r requirements.txt
#   2. huggingface-cli login
#   3. python prepare_llava_dataset.py
#   4. bash train.sh
#
# Expected time: ~30-60 min on 1x A100

set -e

echo "=== AngelCare Post-Training (SFT) ==="
echo ""

# Check dataset exists
if [ ! -f "angelcare_llava_train.json" ]; then
    echo "Dataset not found. Generating..."
    python3 prepare_llava_dataset.py
fi

TRAIN_COUNT=$(python3 -c "import json; print(len(json.load(open('angelcare_llava_train.json'))))")
TEST_COUNT=$(python3 -c "import json; print(len(json.load(open('angelcare_llava_test.json'))))" 2>/dev/null || echo "0")

echo "Train set: $TRAIN_COUNT samples"
echo "Test set:  $TEST_COUNT samples"
echo ""

# Check GPU
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "GPUs detected: $GPU_COUNT"
echo ""

echo "Starting SFT training (QLoRA + TRL)..."
echo ""

python3 train_trl.py "$@"

echo ""
echo "=== Training complete ==="
echo ""
echo "Next steps:"
echo "  python3 evaluate.py --model outputs/angelcare_sft/final --dataset angelcare_llava_test.json --output eval_finetuned.json"
