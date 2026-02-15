"""
AngelCare — Nemotron Nano V2 VL Inference
==========================================
Video analysis using NVIDIA Nemotron Nano 12B v2 VL for elder safety
monitoring.  Used as the secondary (verifier) model in the speculative
cascade pipeline — only invoked when Cosmos Reason 2 produces an
uncertain result.

Nemotron VL requires pre-extracted video frames (unlike Cosmos which
accepts video paths directly), so this module handles frame extraction
via ffmpeg.

Requirements:
    pip install causal_conv1d mamba-ssm==2.2.5 open_clip_torch timm
    ffmpeg must be available on PATH
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

import torch
from PIL import Image

# Reuse the same prompts as Cosmos for consistent classification
from core.inference import SYSTEM_INSTRUCTIONS, USER_PROMPT_CONTENT


def extract_frames(video_path: str, fps: int = 1, max_frames: int = 32) -> list:
    """
    Extract frames from a video file using ffmpeg.

    Args:
        video_path: Path to the video file
        fps: Frames per second to extract
        max_frames: Maximum number of frames to return

    Returns:
        List of PIL Image objects
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        pattern = str(Path(tmp_dir) / "frame_%04d.jpg")
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={fps}",
            "-q:v", "2",
            "-frames:v", str(max_frames),
            pattern,
            "-loglevel", "quiet",
        ]
        subprocess.run(cmd, check=True)

        frame_paths = sorted(Path(tmp_dir).glob("frame_*.jpg"))
        return [Image.open(p).convert("RGB") for p in frame_paths]


def load_nemotron() -> tuple:
    """
    Load Nemotron Nano V2 VL model, tokenizer, and processor.

    Returns:
        Tuple of (model, tokenizer, processor)
    """
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    print("=== Loading Nemotron Nano V2 VL (12B) ===")
    model_path = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"

    # Patch: newer transformers expects all_tied_weights_keys on the model
    # but Nemotron's custom code only defines _tied_weights_keys.
    # Patch nn.Module so it's present before device_map inference runs.
    if not hasattr(torch.nn.Module, 'all_tied_weights_keys'):
        torch.nn.Module.all_tied_weights_keys = {}

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Enable video pruning for faster inference (75% token reduction)
    model.video_pruning_rate = 0.75

    print(f"Nemotron loaded on {torch.cuda.get_device_name(0)}")
    return model, tokenizer, processor


def analyze_video_nemotron(
    model, tokenizer, processor, video_path: str, fps: int = 1
) -> dict | None:
    """
    Analyze a video file using Nemotron Nano V2 VL.

    Args:
        model: Loaded Nemotron model
        tokenizer: Loaded tokenizer
        processor: Loaded processor
        video_path: Path to the video file
        fps: Frames per second to extract

    Returns:
        Dictionary with prediction, risk level, and recommended actions.
        Returns a fallback dict with "Parse Error" if JSON parsing fails.
    """
    frames = extract_frames(video_path, fps=fps)
    if not frames:
        return {
            "prediction_class_id": -1,
            "prediction_label": "No Frames",
            "risk_level": "UNKNOWN",
            "video_description": "Could not extract frames from video.",
        }

    # Build conversation — /no_think for direct structured output
    messages = [
        {"role": "system", "content": f"/no_think\n{SYSTEM_INSTRUCTIONS}"},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": ""},
                {"type": "text", "text": USER_PROMPT_CONTENT},
            ],
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt],
        videos=frames,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(
        pixel_values_videos=inputs.pixel_values_videos,
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=512,
    )

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Parse JSON — same recovery logic as Cosmos pipeline
    clean_json = output_text.strip().replace("```json", "").replace("```", "")
    clean_json = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', clean_json)

    try:
        return json.loads(clean_json)
    except json.JSONDecodeError:
        match = re.search(r'\{.*', clean_json, re.DOTALL)
        if match:
            fragment = match.group()
            if fragment.count('"') % 2 == 1:
                fragment += '"'
            open_braces = fragment.count('{') - fragment.count('}')
            open_brackets = fragment.count('[') - fragment.count(']')
            fragment += ']' * open_brackets + '}' * open_braces
            try:
                return json.loads(fragment)
            except json.JSONDecodeError:
                pass
        return {
            "prediction_class_id": -1,
            "prediction_label": "Parse Error",
            "risk_level": "UNKNOWN",
            "video_description": output_text[:500],
            "raw_output": output_text,
        }
