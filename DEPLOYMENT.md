# AngelCare — Deployment Guide

Two workflows, each a single command from your Mac.

## Quick Start

### Inference / Livestream (H100)

```bash
bash scripts/deploy.sh <NEBIUS_IP> --cascade
```

Then open a second terminal for the camera:
```bash
ssh -R 8554:<CAMERA_IP>:554 <SSH_USER>@<NEBIUS_IP>
```

Go to [angelcare.info](https://angelcare.info) → Livestream → paste the Cloudflare URL → connect camera.

### Post-Training (A100)

```bash
bash scripts/deploy_training.sh <NEBIUS_IP> --push-to-hf
```

Runs the full pipeline: install deps → generate dataset → baseline eval → train → eval → push to HF.

---

## Workflow 1: Inference / Livestream

### What the script does

`scripts/deploy.sh` (runs from your Mac):
1. Tests SSH connection
2. Rsync uploads code to Nebius (excludes training data, videos, website)
3. Uploads `.env` (contains HF_TOKEN)
4. SSHs in and runs `scripts/setup_inference.sh`

`scripts/setup_inference.sh` (runs on Nebius):
1. Creates venv, installs system deps (python3-dev, ffmpeg)
2. `pip install -r requirements.txt`
3. Logs into HuggingFace
4. Optionally deploys VSS stack (`--with-vss`)
5. Starts Cloudflare tunnel (prints public URL)
6. Starts Flask with `--cascade` and/or `--vss-url`

### Options

```bash
# Cascade only (Cosmos + Nemotron, no VSS)
bash scripts/deploy.sh <NEBIUS_IP> --cascade

# Cascade + VSS (requires NGC_API_KEY in .env)
bash scripts/deploy.sh <NEBIUS_IP> --cascade --with-vss

# With camera IP (prints the exact SSH tunnel command)
bash scripts/deploy.sh <NEBIUS_IP> --cascade --camera-ip <CAMERA_IP>
```

### After deployment

1. The script prints a Cloudflare tunnel URL (e.g. `https://xxx.trycloudflare.com`)
2. Open [angelcare.info](https://angelcare.info) → **Livestream** tab
3. Paste the tunnel URL in **Backend Connection** → **Connect**
4. Open a second terminal for the camera reverse tunnel:
   ```bash
   ssh -R 8554:<CAMERA_IP>:554 <SSH_USER>@<NEBIUS_IP>
   ```
5. On the website, enter camera URL: `rtsp://<user>:<pass>@localhost:8554/stream2`
6. Click **Start** — events appear every ~30 seconds

### Instance type

| Mode | GPU | VRAM | Cost |
|------|-----|------|------|
| Cosmos-only | L40S | 24 GB | ~$1.50/hr |
| Cascade (Cosmos + Nemotron) | H100 | 80 GB | ~$3.50/hr |
| Cascade + VSS | H100 | 80 GB | ~$3.50/hr |

### Architecture

```
┌───────────┐          ┌─────────────┐     SSH -R     ┌──────────────────────┐
│ Tapo C200 │  RTSP    │  Your Mac   │ ─────────────► │  Nebius (H100)       │
│           │ ───────► │             │                │  Flask :5000         │
└───────────┘          └─────────────┘                │  Cosmos + Nemotron   │
                                                      │         │            │
                       ┌─────────────┐  Cloudflare    │  cloudflared         │
                       │  Browser    │ ◄────────────  │  tunnel :5000        │
                       │ angelcare   │  Tunnel        │                      │
                       │  .info      │                └──────────────────────┘
                       └─────────────┘
```

---

## Workflow 2: Post-Training

### What the script does

`scripts/deploy_training.sh` (runs from your Mac):
1. Tests SSH connection
2. Rsync uploads training data + videos to Nebius
3. Uploads `.env`
4. SSHs in and runs `scripts/setup_training.sh`
5. Optionally downloads weights back to Mac

`scripts/setup_training.sh` (runs on Nebius):
1. Creates venv, installs training deps (torch, trl, peft, bitsandbytes)
2. Logs into HuggingFace
3. Generates dataset (`prepare_llava_dataset.py`)
4. Baseline evaluation (~20-30 min)
5. Trains QLoRA SFT (~3 min)
6. Evaluates fine-tuned model
7. Compares results
8. Optionally pushes weights to HuggingFace

### Options

```bash
# Full pipeline + push to HF
bash scripts/deploy_training.sh 51.250.X.X --push-to-hf

# Skip baseline (already run before)
bash scripts/deploy_training.sh 51.250.X.X --skip-baseline --push-to-hf

# Download weights to Mac
bash scripts/deploy_training.sh 51.250.X.X --push-to-hf --download-weights

# Training only (no eval)
bash scripts/deploy_training.sh 51.250.X.X --skip-eval
```

### Instance type

| GPU | Training time | Cost |
|-----|--------------|------|
| 1x A100 80GB | ~3 min (+ ~30 min eval) | ~$5-10 total |

### Training parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Method | QLoRA (4-bit + LoRA) | Prevents overfitting on ~134 samples |
| LoRA rank | 16 | Enough capacity without memorizing |
| Epochs | 3 | Converges by epoch 2-3 |
| Batch size | 1 (grad accum 8) | Effective batch 8, fits in memory |
| Learning rate | 2e-5 | Standard for LoRA fine-tuning |

See [TRAINING.md](training_dataset/TRAINING.md) for the full step-by-step guide.

---

## Camera Setup (Tapo C200)

### One-time setup

1. Install the **Tapo** app → add the camera → connect to WiFi
2. **Camera Settings > Advanced > Camera Account** → create credentials
3. Note the camera IP from **Device Info** (e.g. `192.168.1.42`)

### RTSP streams

| Stream | URL | Resolution |
|--------|-----|------------|
| Main (1080p) | `rtsp://user:pass@IP:554/stream1` | 1920x1080 |
| Sub (360p) | `rtsp://user:pass@IP:554/stream2` | 640x360 |

**Use `stream2`** — lower resolution is faster and sufficient for safety classification.

### Test locally

```bash
ffplay "rtsp://<user>:<pass>@<CAMERA_IP>:554/stream2"
```

---

## Environment Variables

Create a `.env` file in the project root:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
NGC_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx    # only needed for VSS
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxx    # optional: SMS alerts
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_FROM_NUMBER=+1234567890
ALERT_TO_NUMBER=+1234567890
```

---

## Shut Down

Always shut down instances when done — H100 costs ~$3.50/hr.

1. `Ctrl+C` to stop Flask
2. Go to [Nebius Console](https://console.nebius.ai/) → Stop/Delete instance

---

## Troubleshooting

### Camera not connecting
- Check the SSH reverse tunnel is active (`ssh -R 8554:...`)
- Verify camera IP hasn't changed (check Tapo app)
- Test locally: `ffplay "rtsp://user:pass@IP:554/stream2"`
- Try `stream2` instead of `stream1`

### Model OOM
- Cascade needs ~40GB — only works on H100, not L40S
- Check with `nvidia-smi` for other processes using GPU

### SSH tunnel drops
```bash
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 \
  -R 8554:<CAMERA_IP>:554 <SSH_USER>@<NEBIUS_IP>
```

### mamba-ssm takes forever to install
It compiles CUDA kernels from source (~15-60 min). The compatibility patches in `app.py` and `core/nemotron.py` handle version mismatches with transformers.
