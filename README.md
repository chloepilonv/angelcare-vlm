# AngelCare

**AI-powered safety monitoring for elderly people living alone.**

Uses video AI to detect falls, immobility, unsteady movement, and distress from standard home cameras — no wearables needed.

Built on [NVIDIA Cosmos Reason 2](https://huggingface.co/nvidia/Cosmos-Reason2-8B) with a [speculative cascade](https://arxiv.org/abs/2405.19261) to [NVIDIA Nemotron VL](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16) for uncertain clips.

**Live demo:** [angelcare.info](https://angelcare.info)

---

## Deploy (One Command)

### Inference / Livestream

```bash
bash scripts/deploy.sh <NEBIUS_IP> --cascade
```

Uploads code to a Nebius H100, installs deps, starts Cloudflare tunnel, launches Flask with both models.

Then connect your camera:
```bash
ssh -R 8554:<CAMERA_IP>:554 <SSH_USER>@<NEBIUS_IP>
```

Open the dashboard → Livestream → paste Cloudflare tunnel URL → enter camera RTSP URL → Start.

**Flags:**

| Flag | What it does |
|------|-------------|
| `--cascade` | Load both Cosmos (8B) + Nemotron VL (12B) |
| `--with-vss` | Deploy full [VSS](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/reason2/vss/inference.html) stack (requires NGC_API_KEY) |
| `--camera-ip IP` | Prints the exact SSH tunnel command for your camera |
| `--port N` | Flask port (default 5000) |

### Post-Training

```bash
HF_REPO=your-username/your-model bash scripts/deploy_training.sh <NEBIUS_IP> --push-to-hf
```

Uploads training data to a Nebius A100, trains QLoRA SFT (~3 min), evaluates, pushes weights to HuggingFace.

**Flags:**

| Flag | What it does |
|------|-------------|
| `--push-to-hf` | Upload trained weights to HuggingFace |
| `--download-weights` | Download weights back to your Mac |
| `--skip-baseline` | Skip baseline evaluation (saves ~30 min) |
| `--skip-eval` | Skip all evaluation (just train) |

---

## How It Works

1. Camera captures video → 10-second clips every 30 seconds
2. **Cosmos Reason 2 (8B)** classifies each clip
3. If uncertain → **Nemotron VL (12B)** verifies (cascade)
4. Risk detections trigger SMS alerts to caregivers

### Classification

| Risk | Classes |
|------|---------|
| CRITICAL | Fall Detected |
| HIGH | Immobility Alert, Distress Posture |
| MEDIUM | Unsteady Movement |
| SAFE | Normal Walking, Normal Sitting, Normal Daily Activity, Resting or Sleeping |

### Architecture

```
Camera → ffmpeg → Cosmos Reason 2 (8B) → Deferral gate → Dashboard + SMS
                                               ↓
                                        Nemotron VL (12B)
                                        (uncertain clips only)
```

---

## Local Development

```bash
# View pre-computed results (no GPU)
pip install -r requirements.txt
python app.py --no-model

# Analyze a video
python -m core.inference --video path/to/clip.mp4

# Batch analyze
python -m core.inference --video-dir videos/

# Cascade mode
python -m core.inference --video-dir videos/ --model cascade
```

---

## Project Structure

| Folder | What it contains |
|--------|-----------------|
| `core/` | AI model code — Cosmos Reason 2 inference, Nemotron VL inference, speculative cascade logic |
| `server/` | Server infrastructure — RTSP camera capture, NVIDIA VSS client, Twilio SMS alerts |
| `scripts/` | One-command deployment scripts for inference (Mac → H100) and training (Mac → A100) |
| `deploy/` | VSS Docker Compose setup for the full NVIDIA Video Search & Summarization stack |
| `website/` | Static frontend hosted on Vercel (dashboard, livestream, test videos) |
| `training_dataset/` | Post-training pipeline — dataset preparation, evaluation, QLoRA SFT training ([guide](training_dataset/TRAINING.md)) |

```
├── app.py                    # Flask entrypoint
├── core/
│   ├── inference.py          # Cosmos Reason 2 (8B) inference
│   ├── nemotron.py           # Nemotron VL (12B) inference
│   └── cascade.py            # Speculative cascade orchestrator
├── server/
│   ├── camera.py             # RTSP camera capture (ffmpeg)
│   ├── vss.py                # NVIDIA VSS API client
│   └── alerts.py             # Twilio SMS alerts
├── scripts/
│   ├── deploy.sh             # Mac → H100 (inference)
│   ├── setup_inference.sh    # Runs on H100
│   ├── deploy_training.sh    # Mac → A100 (training)
│   └── setup_training.sh     # Runs on A100
├── deploy/
│   └── setup_vss.sh          # VSS Docker Compose setup
├── website/                  # Static site (Vercel)
├── training_dataset/
│   ├── TRAINING.md           # Post-training guide
│   ├── prepare_llava_dataset.py
│   ├── evaluate.py
│   └── train.sh
└── requirements.txt
```

---

## References

- [Cosmos Cookbook — Post-Training (SFT)](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/post_training/overview.html)
- [Cosmos Cookbook — Data Curation](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/data_curation/overview.html)
- [Cosmos Cookbook — VSS Inference](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/reason2/vss/inference.html)
- [Speculative Cascade (ICLR 2025)](https://arxiv.org/abs/2405.19261)

## Privacy

Video is processed in memory and discarded — no raw footage is stored. See [Privacy Policy](website/privacy.html).

---

Built for the [NVIDIA Cosmos Cookoff Hackathon](https://luma.com/nvidia-cosmos-cookoff).
