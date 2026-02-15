# AngelCare — Technical Documentation

Comprehensive technical reference covering model selection, inference architecture, post-training pipeline, and deployment decisions.

## Table of Contents

- [Models](#models)
- [Classification Schema](#classification-schema)
- [Inference Pipeline](#inference-pipeline)
- [Speculative Cascade](#speculative-cascade)
- [Video Processing](#video-processing)
- [Livestream Architecture](#livestream-architecture)
- [Post-Training (SFT)](#post-training-sft)
- [Dataset & Llava Format](#dataset--llava-format)
- [Evaluation](#evaluation)
- [VSS Integration](#vss-integration)
- [Deployment & Hardware](#deployment--hardware)

---

## Models

### Cosmos Reason 2 (8B) — Primary Model

| Property | Value |
|----------|-------|
| Model | `nvidia/Cosmos-Reason2-8B` |
| Architecture | Qwen3-VL (vision-language) |
| Parameters | 8 billion |
| Context window | 32,768 tokens |
| Input modality | Video + text |
| VRAM required | ~16 GB (fp16) / ~5 GB (4-bit) |

**Why this model?** Cosmos Reason 2 is a video-native language model designed for physical-world reasoning. Unlike image-based VLMs that process individual frames, Cosmos ingests video directly and reasons about temporal dynamics — critical for distinguishing a fall (sudden transition) from lying down (gradual, intentional). It was built on the [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B) architecture with additional physical-world pre-training from NVIDIA.

**Why 8B and not larger?** The 8B model fits on a single consumer GPU (L40S 24GB or even RTX 4090) while maintaining strong zero-shot performance on safety classification. Larger models would require multi-GPU setups, increasing deployment cost for a use case that needs to run continuously.

### Nemotron Nano V2 VL (12B) — Verifier Model

| Property | Value |
|----------|-------|
| Model | `nvidia/Nemotron-Nano-12B-v2-VL` |
| Architecture | Hybrid Mamba-Transformer |
| Parameters | 12 billion |
| VRAM required | ~24 GB (fp16) |

**Why this model?** Nemotron serves as the verification model in the speculative cascade. It uses a different architecture (Mamba-Transformer hybrid vs pure Transformer), providing an independent "second opinion" on ambiguous clips. Its slightly larger size gives it an edge on nuanced scenarios where Cosmos is uncertain.

**Why not use Nemotron for everything?** Nemotron is slower due to its larger size and doesn't have Cosmos's specialized physical-world pre-training. Using it only for uncertain clips gives us the best accuracy-to-latency tradeoff.

### Llama 3.1 8B — Summarization (VSS only)

Used within the NVIDIA VSS stack to aggregate clip-level captions into time-windowed safety summaries. Not used in standalone inference.

---

## Classification Schema

AngelCare classifies every video into exactly one of 8 categories across 4 risk levels:

| Risk Level | ID | Label | Definition | Action |
|------------|-----|-------|-----------|--------|
| CRITICAL | 0 | Fall Detected | Elder is falling, has fallen, or is on the ground unable to get up | Call emergency services immediately |
| HIGH | 1 | Immobility Alert | Elder has not moved for an unusually long period, appears unresponsive | Alert caregiver — check on elder |
| HIGH | 3 | Distress Posture | Signs of pain or discomfort (clutching body, hunched, calling for help) | Alert caregiver — possible pain or distress |
| MEDIUM | 2 | Unsteady Movement | Walking with visible instability, stumbling, or losing balance | Monitor closely for potential fall |
| SAFE | 4 | Normal Walking | Walking steadily and safely through the home | No action needed |
| SAFE | 5 | Normal Sitting | Seated comfortably in a chair, couch, or at a table | No action needed |
| SAFE | 6 | Normal Daily Activity | Routine activity (cooking, reading, watching TV, eating) | No action needed |
| SAFE | 7 | Resting or Sleeping | Lying down in bed or recliner in a normal resting position | No action needed |

**Design rationale:** The 4-level risk hierarchy (CRITICAL → HIGH → MEDIUM → SAFE) maps directly to urgency of response. CRITICAL triggers immediate emergency alert, HIGH notifies a caregiver, MEDIUM logs for review, SAFE is ignored. This hierarchy drives the deferral logic in the cascade and the SMS alerting thresholds.

---

## Inference Pipeline

### Prompt Engineering

The model receives two prompts:

1. **System prompt** — establishes the role ("Expert Elder Care Safety Monitor"), sets constraints (focus on the elder, prioritize risk, classify final state if transitions occur), and prevents common failure modes (treating household clutter as hazards, focusing on pets instead of the elder).

2. **User prompt** — provides the exact classification table with IDs, labels, definitions, and risk levels. Specifies the JSON output schema with required fields: `prediction_class_id`, `prediction_label`, `risk_level`, `video_description`, and `risk_assessment`.

**Why structured JSON output?** Forces the model to commit to a discrete class rather than generating ambiguous free-text. Enables programmatic risk-level extraction, direct accuracy measurement, and reliable alert triggering.

### Vision Processing

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `fps` | 4 | 4 frames/second captures enough temporal information for fall detection (a fall takes 1-3 seconds) while keeping token count manageable. Higher FPS would increase VRAM and latency without meaningful accuracy gains. |
| `PIXELS_PER_TOKEN` | 32² = 1024 | Standard Qwen3-VL token-to-pixel ratio. |
| `min_vision_tokens` | 256 | Ensures minimum visual resolution even for short clips. |
| `max_vision_tokens` | 8192 | Caps visual token count to prevent OOM on long videos. With 1024 px/token, this allows up to ~8M pixels per video. |
| `max_new_tokens` | 2048 | JSON output is typically 200-400 tokens. 2048 provides headroom for the model's chain-of-thought reasoning. |

### Model Loading

```python
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
)
```

- **`dtype=torch.float16`** — Half precision halves VRAM usage (~16GB vs ~32GB) with negligible quality loss for inference.
- **`device_map="auto"`** — Automatically distributes layers across available GPUs.
- **`attn_implementation="sdpa"`** — Scaled Dot-Product Attention (PyTorch native) is faster and more memory-efficient than the default attention implementation.

---

## Speculative Cascade

Inspired by ["Faster Cascades via Speculative Decoding" (ICLR 2025)](https://arxiv.org/abs/2405.19261).

```
Video → Cosmos Reason 2 (8B) → Deferral Gate → [if uncertain] → Nemotron VL (12B)
              ~10s                   ~0s                              ~15s
```

### How It Works

1. **Cosmos (drafter)** runs on every clip and produces a classification
2. **Deferral gate** checks the result for uncertainty signals
3. **Nemotron (verifier)** only runs when Cosmos is uncertain — its result replaces Cosmos's

### Deferral Signals

The gate triggers on any of these conditions:

| Signal | Condition | Rationale |
|--------|-----------|-----------|
| Parse error | `prediction_class_id == -1` | Model couldn't produce structured output — low confidence |
| Unknown risk | `risk_level == "UNKNOWN"` | Model didn't commit to a risk level |
| Medium risk | `risk_level == "MEDIUM"` | Ambiguous zone — could be a near-fall or normal movement |
| Short description | `len(video_description) < 30` | Minimal output suggests the model couldn't interpret the scene |
| Missing temporal | Risk is CRITICAL/HIGH but `temporal_segment` is null | Model flagged danger but can't pinpoint when — uncertain |

### Performance Characteristics

- **Best case** (clear SAFE or CRITICAL): Only Cosmos runs. Zero added latency.
- **Worst case** (ambiguous clip): Both models run sequentially. ~2x latency.
- **Typical deferral rate**: 15-30% of clips, depending on video quality and scene complexity.

On an H100 (80GB), both models stay preloaded in memory (~40GB total). On smaller GPUs (L40S 24GB), only one model fits at a time — the second model would need to be loaded on demand (slower).

---

## Video Processing

### Frame Sampling

Videos are sampled at **4 fps** regardless of source frame rate. For a 10-second clip, this produces 40 frames.

**Why 4 fps?** Falls happen over 1-3 seconds. At 4 fps, a 2-second fall produces 8 frames — enough to capture the full transition. Higher fps (e.g., 30) would add 7.5x more visual tokens with diminishing returns, increasing latency and VRAM usage proportionally.

### RTSP Camera Capture

For livestream monitoring, `server/camera.py` uses ffmpeg to grab clips from IP cameras:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_duration` | 10 seconds | Length of each captured clip |
| `interval` | 30 seconds | Time between capture starts |
| `output format` | H.264 MP4 | Universal, hardware-decodable |

**Why 10s clips every 30s?** A 10-second window is long enough to capture a complete fall event (approach → fall → post-fall state). The 30-second interval gives the model 20 seconds of processing headroom, ensuring real-time operation even on slower GPUs. Clips are saved to `captures/` for audit trail.

**Why ffmpeg, not OpenCV?** ffmpeg handles RTSP natively with robust reconnection, hardware decoding, and codec support. OpenCV's VideoCapture is unreliable with RTSP streams (drops frames, no reconnection, codec issues).

---

## Livestream Architecture

### Standalone Mode (Direct Camera)

```
Tapo C200 (RTSP) → ffmpeg capture → 10s MP4 clips → Cosmos Reason 2 → Risk Classification
                                                                              ↓
                                                                    Flask Dashboard + SMS
```

The Flask backend (`app.py`) spawns a background `CameraCapture` thread that periodically grabs clips. Each clip is analyzed by the loaded model, and results stream to the web dashboard via polling.

### VSS Mode (NVIDIA Video Search & Summarization)

```
RTSP Camera → NVIDIA VSS → Cosmos Reason 2 NIM (captioning)
                         → Llama 3.1 8B NIM (summarization)
                         → Embedding + Reranker NIMs (search)
                                    ↓
                         AngelCare Flask App (proxy) → SMS Alerts
```

VSS provides:
- **Automatic chunk management** — splits streams into analyzable segments
- **NIM-optimized inference** — TensorRT-accelerated, 5-10x faster than raw HuggingFace
- **Built-in summarization** — Llama 3.1 aggregates clip captions into time-windowed summaries
- **Search** — query past events ("show me all falls today") via embedding + reranker

VSS runs as a Docker Compose stack on a single H100/A100 (80GB). `server/vss.py` wraps the REST API, adding elder-safety-specific captioning prompts and regex-based risk extraction from free-text summaries.

---

## Post-Training (SFT)

Based on the [Cosmos Cookbook Intelligent Transportation recipe](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/intelligent-transportation/post_training.html).

### Why Post-Train?

Zero-shot Cosmos Reason 2 achieves strong baseline accuracy on elder safety classification. Post-training addresses:
- **Output format consistency** — eliminates parse errors by teaching the model our exact JSON schema
- **Edge case accuracy** — improves distinction between similar classes (e.g., "Resting or Sleeping" vs "Immobility Alert")
- **Domain adaptation** — home surveillance footage has different characteristics than Cosmos's pre-training data

### Method: QLoRA (Quantized Low-Rank Adaptation)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Quantization | 4-bit NF4 | Reduces base model VRAM from ~16GB to ~5GB, enabling training on a single GPU |
| LoRA rank (r) | 16 | Rank 16 provides enough capacity to learn output format and domain nuances without overfitting on 133 samples |
| LoRA alpha | 32 | Alpha = 2r is a standard heuristic that provides good learning signal |
| LoRA dropout | 0.05 | Light regularization against overfitting |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | All attention projections + MLP gates. Covers both attention patterns and feed-forward reasoning |
| Trainable parameters | ~40M (~0.5% of 8B) | Enough to learn format/domain, too few to overfit |
| Double quantization | Yes | Quantizes the quantization constants — saves ~0.4GB additional VRAM |

**Why QLoRA, not full SFT?** With only 133 training samples, full SFT (updating all 8B parameters) would memorize the dataset in < 1 epoch. QLoRA freezes the base model and only trains lightweight adapter layers (~0.5% of params), acting as a natural regularizer. It's the standard approach for fine-tuning large models on small datasets.

**Why not just LoRA (without quantization)?** The 8B model at fp16 uses ~16GB. With LoRA overhead (optimizer states, gradients), training would need ~40GB. 4-bit quantization drops the base model to ~5GB, leaving ample room for training state on a single A100 (80GB). QLoRA's quality loss vs full-precision LoRA is negligible ([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)).

### Training Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Epochs | 3 | With 133 samples, 3 epochs = 399 training steps. Enough to converge on format learning without overfitting. Loss curve shows convergence by epoch 2-3. |
| Batch size | 1 | Video inputs are large and variable-length. Batch size 1 avoids padding waste and OOM. |
| Gradient accumulation | 8 | Effective batch size of 8. Smooths gradients across diverse video types (fall vs sitting vs walking). |
| Learning rate | 2e-5 | Standard for LoRA fine-tuning. Matches the Cosmos Cookbook recipe. |
| Warmup steps | 10 | Brief warmup (~2.5% of steps) stabilizes early training. |
| Precision | bf16 | Bfloat16 training is numerically stable for gradient computation on A100/H100. |
| Gradient checkpointing | Yes | Trades compute for memory — recomputes activations during backward pass instead of storing them. Essential for video models. |
| Optimizer | AdamW (default) | Standard optimizer for transformer fine-tuning. |
| LR scheduler | Cosine (default) | Gradual decay prevents catastrophic forgetting in later epochs. |

### Training Time

| Hardware | Time (133 samples, 3 epochs) |
|----------|------------------------------|
| 1x A100 80GB | ~3 minutes |
| 1x L40S 24GB | ~10-15 minutes (estimated) |

Training is fast because QLoRA only computes gradients for ~40M parameters (0.5%), and the dataset is small. The Cosmos Cookbook's 2-hour estimate was for full SFT on 5,000 samples.

### Libraries

| Library | Purpose |
|---------|---------|
| [TRL](https://github.com/huggingface/trl) (Transformer Reinforcement Learning) | HuggingFace's training library for LLM fine-tuning. Provides `SFTTrainer` which handles dataset formatting, training loop, and checkpoint saving. Used as a drop-in replacement for NVIDIA's `cosmos-rl` (which requires NGC access). |
| [PEFT](https://github.com/huggingface/peft) (Parameter-Efficient Fine-Tuning) | Implements LoRA adapters. Wraps the base model, freezes original weights, and injects trainable low-rank matrices into specified layers. |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | Provides 4-bit NF4 quantization for the base model. Enables QLoRA by quantizing weights during loading while keeping adapter parameters in full precision. |

---

## Dataset & Llava Format

### Llava Format

The training data follows the [Llava conversation format](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/intelligent-transportation/post_training.html), which is the standard for vision-language SFT:

```json
{
  "id": "gmdc_Subject1_Fall_01",
  "video": "/absolute/path/to/video.mp4",
  "conversations": [
    {
      "from": "human",
      "value": "<video>\nAnalyze the video and classify the elder's safety status..."
    },
    {
      "from": "gpt",
      "value": "{\n  \"prediction_class_id\": 0,\n  \"prediction_label\": \"Fall Detected\",\n  ...}"
    }
  ]
}
```

Key design decisions:
- **`<video>` tag** in the human message signals where the video tokens are inserted
- **JSON string as the GPT response** — teaches the model to output valid, parseable JSON
- **Absolute paths** — video paths must be absolute because the training framework resolves them at load time. Paths must be regenerated if the dataset is moved to a different machine.

### Source Datasets

| Dataset | Videos | Classes | Source |
|---------|--------|---------|--------|
| GMDCSA-24 | 160 (79 fall + 81 ADL) | Fall, Walking, Sitting, Resting, Daily Activity | [Zenodo](https://doi.org/10.5281/zenodo.11489420) |
| Personal clips | 9 (2 fall + 7 safe) | Fall, Daily Activity | iPhone footage, real home |

**Label mapping:** GMDCSA-24 provides free-text descriptions per video. `prepare_llava_dataset.py` maps these to AngelCare classes using keyword matching:
- "sleeping", "lying", "bed" → Resting or Sleeping
- "sitting", "sit" → Normal Sitting
- "walking", "walk" → Normal Walking
- All other ADL → Normal Daily Activity
- All Fall category → Fall Detected

### Train/Test Split

**80/20 stratified split** — each class maintains its proportion in both sets. The split is seeded (`--seed 42`) for reproducibility. With 167 total entries, this gives ~134 train / ~33 test.

**Why stratified?** A random split on an imbalanced dataset (81 falls vs 8 walking) could put zero walking samples in the test set. Stratification guarantees at least 1 sample per class in both sets.

### Data Curation (Optional)

[NVIDIA Cosmos Curate](https://github.com/nvidia-cosmos/cosmos-curate) can optionally pre-process raw videos:
1. **Shot-aware splitting** — detect scene transitions within longer videos
2. **GPU transcoding** — normalize to consistent format/resolution
3. **Quality filtering** — remove low-motion or low-aesthetic clips
4. **Auto-captioning** — generate descriptions using Cosmos Reason 2

This is useful when ingesting raw footage (e.g., hours of camera recordings) but not needed for pre-segmented datasets like GMDCSA-24.

---

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **Exact label accuracy** | Prediction label exactly matches ground truth (e.g., "Fall Detected" == "Fall Detected") |
| **Risk-level accuracy** | Predicted risk level matches ground truth (more lenient — "Normal Walking" predicted as "Normal Sitting" would still be correct since both are SAFE) |
| **Per-class accuracy** | Breakdown by class — identifies which categories the model struggles with |

**Why both exact and risk-level?** For safety monitoring, the risk level matters more than the exact label. Confusing "Normal Walking" with "Normal Sitting" (both SAFE) is harmless. Confusing "Fall Detected" (CRITICAL) with "Normal Daily Activity" (SAFE) is dangerous. Risk-level accuracy captures this.

### Evaluation Protocol

1. **Baseline (zero-shot)** — evaluate the base Cosmos Reason 2 model on the held-out test set
2. **Fine-tuned** — evaluate the QLoRA-adapted model on the same test set
3. **Compare** — report improvement in both exact and risk-level accuracy

The test set is never seen during training. This prevents overfitting from inflating results.

---

## VSS Integration

NVIDIA Video Search & Summarization (VSS) provides production-grade video analysis as a Docker Compose stack.

### Components

| Service | Model | Role |
|---------|-------|------|
| Captioning NIM | Cosmos Reason 2 | Generates per-clip descriptions |
| Summarization NIM | Llama 3.1 8B | Aggregates captions into time-windowed summaries |
| Embedding NIM | NV-EmbedQA | Encodes captions for semantic search |
| Reranker NIM | NV-RerankQA | Re-ranks search results for relevance |

### How AngelCare Uses VSS

`server/vss.py` wraps the VSS REST API with elder-safety-specific prompts:

1. **Upload** — sends a video clip to VSS
2. **Caption** — VSS runs Cosmos Reason 2 with a safety-focused prompt: *"Describe what the elderly person is doing... End with a risk assessment: CRITICAL / HIGH / MEDIUM / SAFE"*
3. **Summarize** — VSS runs Llama to combine clip captions into a summary
4. **Risk extraction** — regex patterns parse CRITICAL/HIGH/MEDIUM from the free-text summary
5. **Alert** — if risk is CRITICAL or HIGH, trigger SMS via Twilio

### VSS vs Standalone

| Aspect | Standalone (`core/inference.py`) | VSS |
|--------|--------------------------------------|-----|
| Inference speed | ~30-60s per clip (raw HuggingFace) | ~5-10s per clip (TensorRT) |
| Summarization | None | Llama 3.1 time-windowed summaries |
| Search | None | Semantic search over past events |
| Setup complexity | `pip install` + model download | Docker Compose + NGC credentials |
| GPU requirement | 24GB+ (L40S) | 80GB (H100/A100) |

---

## Deployment & Hardware

### GPU Recommendations

| Use Case | GPU | VRAM | Notes |
|----------|-----|------|-------|
| Cosmos-only inference | L40S / RTX 4090 | 24 GB | Fits model in fp16 (~16GB) |
| Cascade (Cosmos + Nemotron) | H100 / A100 | 80 GB | Both models preloaded (~40GB) |
| VSS full stack | H100 / A100 | 80 GB | 4 NIM containers |
| QLoRA training | A100 | 80 GB | Base model 4-bit (~5GB) + training state |
| Full SFT training | 8x A100 | 640 GB total | Only needed for large datasets (5000+) |

### Nebius Deployment

AngelCare is designed for [Nebius AI Cloud](https://console.nebius.ai/):

- **Training**: Container in VM with Cosmos Reason 8B pre-installed. 1x A100 80GB. ~$3-4/hr, ~$10-12 per training session.
- **Inference**: L40S (24GB) for Cosmos-only, H100 (80GB) for cascade/VSS.
- **Access**: SSH tunnel (`ssh -L 5000:localhost:5000`) to access the Flask dashboard from your laptop.

### Cost Optimization

- Use the **Cosmos container image** on Nebius — model weights are pre-cached, no HuggingFace download needed (saves ~15 min)
- **Shut down training instances** immediately after weights are saved — upload to HuggingFace first, then terminate
- **QLoRA adapter weights are tiny** (~160MB vs ~16GB for full model) — fast to upload/download

---

## References

- [NVIDIA Cosmos Reason 2](https://huggingface.co/nvidia/Cosmos-Reason2-8B) — Base video LLM
- [Cosmos Cookbook — Post-Training Recipe](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/intelligent-transportation/post_training.html) — SFT recipe this project is based on
- [NVIDIA VSS Documentation](https://docs.nvidia.com/vss/latest/) — Video Search & Summarization
- [NVIDIA Cosmos Curate](https://github.com/nvidia-cosmos/cosmos-curate) — Video data curation
- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314) — QLoRA paper
- [Faster Cascades via Speculative Decoding (ICLR 2025)](https://arxiv.org/abs/2405.19261) — Speculative cascade paper
- [TRL — Transformer Reinforcement Learning](https://github.com/huggingface/trl) — SFT training library
- [PEFT — Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft) — LoRA implementation
- [GMDCSA-24 Dataset (Alam et al., 2024)](https://doi.org/10.5281/zenodo.11489420) — Fall detection dataset
- [FallVision Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/75QPKK) — Additional fall detection data
