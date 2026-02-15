# AngelCare — Post-Training Guide

Step-by-step guide to fine-tune Cosmos Reason 2 (8B) on elder care fall detection data using QLoRA SFT on Nebius.

Based on the [Cosmos Cookbook Intelligent Transportation recipe](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/intelligent-transportation/post_training.html).

## Quick Start (Automated)

```bash
bash scripts/deploy_training.sh <NEBIUS_IP> --push-to-hf
```

This uploads data, installs deps, trains, evaluates, and pushes weights to HuggingFace — all in one command. See [DEPLOYMENT.md](../DEPLOYMENT.md) for details.

The steps below are the manual equivalent for reference and debugging.

## What You Need

| Resource | Spec |
|----------|------|
| GPU | **1x A100 80GB** |
| Nebius image | Container in VM → Cosmos Reason 8B |
| Time | ~3 minutes training + ~30 min setup/eval |
| Cost | ~$5-10 total |

## Step 1: Generate the Dataset (on your Mac)

```bash
cd ~/Documents/CODE/MISC/angelcare/training_dataset
python3 prepare_llava_dataset.py
```

This creates two files:
- `angelcare_llava_train.json` — ~134 samples (80%) for training
- `angelcare_llava_test.json` — ~33 samples (20%) for evaluation

The split is stratified — each class keeps its proportion in both sets.

Verify:
```bash
python3 prepare_llava_dataset.py --stats-only
```

## Step 2: Launch a Nebius Instance

1. Go to [console.nebius.ai](https://console.nebius.ai)
2. Create a **Container in VM** → select **Cosmos Reason 8B** image
3. GPU: 1x A100 80GB
4. Note the SSH command (e.g. `ssh <SSH_USER>@<NEBIUS_IP>`)

## Step 3: Upload Training Data

From your Mac:

```bash
rsync -avz --progress \
  ~/Documents/CODE/MISC/angelcare/training_dataset/ \
  <SSH_USER>@<NEBIUS_IP>:~/angelcare/training_dataset/

rsync -avz --progress \
  ~/Documents/CODE/MISC/angelcare/videos_compressed/ \
  <SSH_USER>@<NEBIUS_IP>:~/angelcare/videos_compressed/
```

## Step 4: Install Dependencies

```bash
ssh <SSH_USER>@<NEBIUS_IP>

python3 -m venv ~/venv
source ~/venv/bin/activate

# Python dev headers (needed for mamba-ssm compilation)
sudo apt-get update && sudo apt-get install -y python3-dev

# Install all deps
pip install torch torchvision transformers>=4.57.0 accelerate huggingface_hub \
  qwen-vl-utils decord trl peft bitsandbytes datasets

# HuggingFace login
python3 -c "from huggingface_hub import login; login(token='YOUR_HF_TOKEN')"
```

### Verify GPU

```bash
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)')"
```

## Step 5: Generate Dataset on Server

Video paths must be absolute and match the server filesystem:

```bash
cd ~/angelcare/training_dataset
python3 prepare_llava_dataset.py
```

Verify: should show ~134 train / ~33 test entries.

## Step 6: Baseline Evaluation

Evaluate the model *before* fine-tuning on the held-out test set:

```bash
python3 evaluate.py \
  --model nvidia/Cosmos-Reason2-8B \
  --dataset angelcare_llava_test.json \
  --output eval_baseline.json
```

This takes ~20-30 min (runs full inference on each video). Save the results.

## Step 7: Train

```bash
bash train.sh
```

What happens:
1. Loads Cosmos Reason 2 (8B) in 4-bit quantization (~5GB VRAM)
2. Attaches LoRA adapters (~40M trainable params, 0.5% of model)
3. Trains for 3 epochs on ~134 samples
4. Saves adapter weights to `outputs/angelcare_sft/final/`

**Training takes ~3 minutes on 1x A100.** This is fast because QLoRA only trains 0.5% of parameters.

### Training parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Method | QLoRA (4-bit + LoRA) | Prevents overfitting on 134 samples |
| LoRA rank | 16 | Enough capacity without memorizing |
| Epochs | 3 | Converges by epoch 2-3 |
| Batch size | 1 (grad accum 8) | Effective batch 8, fits in memory |
| Learning rate | 2e-5 | Standard for LoRA fine-tuning |

See [TECHNICAL.md](../TECHNICAL.md#post-training-sft) for detailed justification of every parameter.

## Step 8: Evaluate Fine-Tuned Model

```bash
python3 evaluate.py \
  --model outputs/angelcare_sft/final \
  --dataset angelcare_llava_test.json \
  --output eval_finetuned.json
```

### Compare results

```bash
python3 -c "
import json
base = json.load(open('eval_baseline.json'))
ft = json.load(open('eval_finetuned.json'))
print(f'Zero-shot:  {base[\"exact_accuracy\"]*100:.1f}% exact | {base[\"risk_accuracy\"]*100:.1f}% risk')
print(f'Fine-tuned: {ft[\"exact_accuracy\"]*100:.1f}% exact | {ft[\"risk_accuracy\"]*100:.1f}% risk')
"
```

## Step 9: Save the Weights

### Option A: Upload to HuggingFace

Requires a write-enabled HF token. Create the repo manually at https://huggingface.co/new first.

```bash
python3 -c "
from huggingface_hub import HfApi, login
login(token='YOUR_HF_WRITE_TOKEN')
api = HfApi()
api.upload_folder(
    folder_path='outputs/angelcare_sft/final',
    repo_id='your-username/your-model-name',
    commit_message='AngelCare QLoRA SFT on GMDCSA-24'
)
print('Done!')
"
```

### Option B: Download to your Mac

The adapter weights are only ~87MB:

```bash
# From your Mac
scp -r <SSH_USER>@<NEBIUS_IP>:~/angelcare/training_dataset/outputs/angelcare_sft/final \
  ~/Documents/CODE/MISC/angelcare/training_dataset/outputs/angelcare_sft/final
```

## Step 10: Shut Down

**Shut down the A100 instance** once weights are saved. Don't leave it running.

## Summary

```
Mac (free)                    Nebius A100 (~$5-10)              HuggingFace
──────────                    ────────────────────              ────────────
prepare_llava_dataset.py  →   rsync upload
                              eval baseline (~30 min)
                              train (~3 min)
                              eval fine-tuned (~30 min)
                              upload weights             →   your-username/your-model
                              shut down
```

## Citations

```bibtex
@article{alam2024,
  title={GMDCSA24: A Dataset for Human Fall Detection in Videos},
  author={Alam, Ekram and Sufian, Abu and Dutta, Paramartha and Leo, Marco and Hameed, I.A.},
  journal={Data in Brief},
  year={2024}
}

@data{DVN/75QPKK,
  title={FallVision: A benchmark video dataset for fall detection},
  publisher={Harvard Dataverse},
  doi={10.7910/DVN/75QPKK}
}
```
