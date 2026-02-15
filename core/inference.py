"""
AngelCare — Elder Safety Monitoring with Cosmos Reason 2
=========================================================
Zero-shot video classification for detecting elder risk situations
using the same approach as the Cosmos Cookbook Worker Safety recipe.

Requirements:
    - NVIDIA GPU with 24GB+ VRAM
    - Python environment from cosmos-reason2 repo (see setup_nebius.sh)
    - HuggingFace auth token (for model download)

Usage:
    source ~/cosmos-reason2/.venv/bin/activate
    python -m core.inference --video path/to/video.mp4

    # Or process videos/bad/ and videos/good/ together:
    python -m core.inference --video-dir videos/
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import torch
import transformers

warnings.filterwarnings("ignore")

# ── Prompts ──────────────────────────────────────────────────

SYSTEM_INSTRUCTIONS = """
You are an expert Elder Care Safety Monitor analyzing home surveillance footage.
Your goal is to classify the video into EXACTLY ONE of the 8 classes defined below.

CRITICAL CONSTRAINTS:
1. FOCUS ON THE ELDER:
   - Identify the elderly person in the scene. Focus on their posture, movement, and behavior.
   - If multiple people are visible, focus on the oldest-looking individual.
2. CONTEXT AWARENESS:
   - The setting is a private home. Normal household clutter is NOT a hazard.
   - Pets, visitors, and caregivers may be present — they are not the subject.
3. SINGLE OUTPUT:
   - Choose the MOST PROMINENT behavior observed.
   - Prioritize RISK behaviors over SAFE behaviors if both are present.
4. TEMPORAL REASONING:
   - If the elder transitions between states (e.g., standing then falling), classify based on the FINAL state.
"""

USER_PROMPT_CONTENT = """
Analyze the video and output a JSON object. You MUST select the class ID and Label EXACTLY from the table below.

STRICT CLASSIFICATION TABLE (Use these exact IDs and Labels):

| ID | Label | Definition | Risk Level |
| :--- | :--- | :--- | :--- |
| 0 | Fall Detected | Elder is falling, has fallen, or is lying on the ground unable to get up. | CRITICAL |
| 1 | Immobility Alert | Elder has not moved for an unusually long period, appears unresponsive or stuck. | HIGH |
| 2 | Unsteady Movement | Elder is walking with visible instability, stumbling, or losing balance. | MEDIUM |
| 3 | Distress Posture | Elder shows signs of pain, distress, or discomfort (clutching body, hunched over, calling for help). | HIGH |
| 4 | Normal Walking | Elder is walking steadily and safely through the home. | SAFE |
| 5 | Normal Sitting | Elder is seated comfortably in a chair, couch, or at a table. | SAFE |
| 6 | Normal Daily Activity | Elder is engaged in routine activity (cooking, reading, watching TV, eating). | SAFE |
| 7 | Resting or Sleeping | Elder is lying down in a bed or recliner in a normal resting position. | SAFE |

INSTRUCTIONS:
1. Observe the elder's posture, movement, and behavior in the video.
2. Match it to the SINGLE BEST row in the table above.
3. Output the exact "ID" and "Label" from that row. Do not invent new labels.

OUTPUT FORMAT:
{
  "prediction_class_id": [Integer from Table],
  "prediction_label": "[Exact String from Table]",
  "risk_level": "[CRITICAL / HIGH / MEDIUM / SAFE]",
  "video_description": "[1-2 sentences MAX describing what the elder is doing]",
  "risk_assessment": {
    "is_at_risk": [true/false — true for CRITICAL, HIGH, or MEDIUM],
    "recommended_action": "[e.g., Call emergency services / Alert caregiver / Monitor closely / No action needed]",
    "temporal_segment": "[Start Time - End Time of the key moment] or null"
  }
}

IMPORTANT: Keep video_description to 1-2 sentences. Do NOT write paragraphs. Be brief.
"""

# ── Model loading ────────────────────────────────────────────

def load_model() -> tuple:
    """
    Load Cosmos-Reason2-8B model and processor.

    Returns:
        Tuple of (model, processor)
    """
    print("=== Loading Cosmos-Reason2-8B ===")
    model_name = "nvidia/Cosmos-Reason2-8B"

    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(model_name)

    PIXELS_PER_TOKEN = 32**2
    min_vision_tokens, max_vision_tokens = 256, 8192
    processor.image_processor.size = processor.video_processor.size = {
        "shortest_edge": min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": max_vision_tokens * PIXELS_PER_TOKEN,
    }

    print(f"Model loaded on {torch.cuda.get_device_name(0)}")
    return model, processor


# ── Inference ────────────────────────────────────────────────

def analyze_video(model, processor, video_path: str) -> dict | None:
    """
    Analyze a video file for elder safety risk using Cosmos Reason 2.

    Args:
        model: Loaded Qwen3VLForConditionalGeneration model
        processor: Loaded Qwen3VLProcessor for tokenization
        video_path: Path to the video file to analyze

    Returns:
        Dictionary with prediction, risk level, and recommended actions.
        Returns a fallback dict with "Parse Error" if JSON parsing fails.
    """
    # Use absolute path to avoid leaking folder names (e.g. "bad/") to the model
    abs_path = str(Path(video_path).resolve())

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCTIONS}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": abs_path},
                {"type": "text", "text": USER_PROMPT_CONTENT},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=4,
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    clean_json = output_text.strip().replace("```json", "").replace("```", "")

    # Sanitize invalid unicode escapes (e.g. \uXXXX where XXXX isn't valid hex)
    # This handles cases where the model outputs malformed escape sequences
    clean_json = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', clean_json)

    def _truncate_description(result: dict) -> dict:
        """Cap video_description to ~2 sentences to prevent rambling output."""
        desc = result.get("video_description", "")
        if len(desc) > 300:
            # Keep first two sentences
            sentences = re.split(r'(?<=[.!?])\s+', desc)
            result["video_description"] = " ".join(sentences[:2])
        return result

    try:
        return _truncate_description(json.loads(clean_json))
    except json.JSONDecodeError:
        # Try to salvage truncated JSON by closing open braces/brackets
        # This recovery strategy helps when the model hits token limits mid-response
        match = re.search(r'\{.*', clean_json, re.DOTALL)
        if match:
            fragment = match.group()
            # Close any unterminated strings, then close braces
            if fragment.count('"') % 2 == 1:
                fragment += '"'
            open_braces = fragment.count('{') - fragment.count('}')
            open_brackets = fragment.count('[') - fragment.count(']')
            fragment += ']' * open_brackets + '}' * open_braces
            try:
                return _truncate_description(json.loads(fragment))
            except json.JSONDecodeError:
                pass
        # Last resort: return raw text as a minimal result
        return {
            "prediction_class_id": -1,
            "prediction_label": "Parse Error",
            "risk_level": "UNKNOWN",
            "video_description": output_text[:500],
            "raw_output": output_text,
        }


# ── Main ─────────────────────────────────────────────────────

def collect_videos(path_str: str) -> list[Path]:
    """Collect video files from a path (file or directory)."""
    VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
    path = Path(path_str)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def run_batch(model, processor, video_paths: list[Path], base_dir: str = "videos") -> list[dict]:
    """
    Run inference on a list of videos and print results.

    Args:
        model: Loaded Cosmos Reason 2 model
        processor: Loaded processor
        video_paths: List of video file paths to analyze
        base_dir: Base directory name for display purposes

    Returns:
        List of analysis results with file paths and risk assessments
    """
    base_name = Path(base_dir).name
    transformers.set_seed(0)

    print(f"\n=== Analyzing {len(video_paths)} video(s) ===\n")

    results = []
    for path in video_paths:
        display_name = f"{path.parent.name}/{path.name}" if path.parent.name != base_name else path.name
        print(f"Processing: {display_name} ... ", end="", flush=True)
        try:
            result = analyze_video(model, processor, str(path))
            result["file"] = display_name
            result["expected"] = "risk" if "bad" in path.parent.name.lower() else "safe"
            results.append(result)

            risk = result.get("risk_level", "?")
            label = result.get("prediction_label", "?")
            marker = "🚨" if risk in ("CRITICAL", "HIGH") else "⚠️ " if risk == "MEDIUM" else "✅"
            print(f"{marker} [{risk}] {label}")

            if risk in ("CRITICAL", "HIGH"):
                print(f"    → {result.get('risk_assessment', {}).get('recommended_action', '')}")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"file": path.name, "error": str(e)})

    # Save results
    output_file = Path("angelcare_results.json")
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_file}")

    # Summary: compare predictions vs expected (bad=risk, good=safe)
    correct, total = 0, 0
    for r in results:
        if "error" in r:
            continue
        total += 1
        predicted_risk = r.get("risk_level", "SAFE") in ("CRITICAL", "HIGH", "MEDIUM")
        expected_risk = r.get("expected") == "risk"
        if predicted_risk == expected_risk:
            correct += 1

    if total > 0:
        print(f"\n=== Accuracy: {correct}/{total} ({100*correct/total:.0f}%) ===")

    return results


def run_batch_cascade(cascade_analyzer, video_paths: list[Path], base_dir: str = "videos") -> list[dict]:
    """
    Run speculative cascade inference on a list of videos.

    Args:
        cascade_analyzer: CascadeAnalyzer instance with both models loaded
        video_paths: List of video file paths to analyze
        base_dir: Base directory name for display purposes

    Returns:
        List of analysis results with cascade metadata
    """
    base_name = Path(base_dir).name
    transformers.set_seed(0)

    print(f"\n=== Cascade: Analyzing {len(video_paths)} video(s) ===")
    print(f"    Cosmos Reason 2 (8B) → deferral gate → Nemotron VL (12B)\n")

    results = []
    for path in video_paths:
        display_name = f"{path.parent.name}/{path.name}" if path.parent.name != base_name else path.name
        print(f"Processing: {display_name} ... ", end="", flush=True)
        try:
            result = cascade_analyzer.analyze(str(path))
            result["file"] = display_name
            result["expected"] = "risk" if "bad" in path.parent.name.lower() else "safe"
            results.append(result)

            risk = result.get("risk_level", "?")
            label = result.get("prediction_label", "?")
            model = result.get("model", "?")
            deferred = result.get("deferred", False)
            marker = "🚨" if risk in ("CRITICAL", "HIGH") else "⚠️ " if risk == "MEDIUM" else "✅"
            tag = f" [deferred → {model}]" if deferred else f" [{model}]"
            print(f"{marker} [{risk}] {label}{tag}")

            if risk in ("CRITICAL", "HIGH"):
                print(f"    → {result.get('risk_assessment', {}).get('recommended_action', '')}")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"file": path.name, "error": str(e)})

    # Save results
    output_file = Path("angelcare_results.json")
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_file}")

    # Cascade stats
    print(f"\n=== {cascade_analyzer.stats.summary()} ===")

    # Accuracy summary
    correct, total = 0, 0
    for r in results:
        if "error" in r:
            continue
        total += 1
        predicted_risk = r.get("risk_level", "SAFE") in ("CRITICAL", "HIGH", "MEDIUM")
        expected_risk = r.get("expected") == "risk"
        if predicted_risk == expected_risk:
            correct += 1

    if total > 0:
        print(f"=== Accuracy: {correct}/{total} ({100*correct/total:.0f}%) ===")

    return results


def main() -> None:
    """Main entry point for AngelCare inference CLI."""
    parser = argparse.ArgumentParser(description="AngelCare — Elder Safety Monitor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to a single video file")
    group.add_argument("--video-dir", type=str, help="Path to a directory of video clips")
    parser.add_argument("--model", type=str, default="cosmos",
                        choices=["cosmos", "nemotron", "cascade"],
                        help="Model to use: cosmos (default), nemotron, or cascade (both)")
    parser.add_argument("--interactive", action="store_true",
                        help="Keep model loaded and accept new commands after first run")
    args = parser.parse_args()

    # GPU check
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU found. Cannot run inference.")
        sys.exit(1)

    source = args.video or args.video_dir
    video_paths = collect_videos(source)
    if not video_paths:
        print("No video files found.")
        sys.exit(1)

    if args.model == "cascade":
        from core.cascade import CascadeAnalyzer
        from core.nemotron import load_nemotron

        model, processor = load_model()
        nem_model, nem_tokenizer, nem_processor = load_nemotron()
        cascade = CascadeAnalyzer(model, processor, nem_model, nem_tokenizer, nem_processor)
        run_batch_cascade(cascade, video_paths, source)
        return  # cascade mode doesn't support interactive

    if args.model == "nemotron":
        from core.nemotron import analyze_video_nemotron, load_nemotron

        nem_model, nem_tokenizer, nem_processor = load_nemotron()
        # Use run_batch-style loop with Nemotron
        transformers.set_seed(0)
        base_name = Path(source).name
        print(f"\n=== Nemotron VL: Analyzing {len(video_paths)} video(s) ===\n")
        results = []
        for path in video_paths:
            display_name = f"{path.parent.name}/{path.name}" if path.parent.name != base_name else path.name
            print(f"Processing: {display_name} ... ", end="", flush=True)
            try:
                result = analyze_video_nemotron(nem_model, nem_tokenizer, nem_processor, str(path))
                result["file"] = display_name
                result["model"] = "nemotron"
                results.append(result)
                risk = result.get("risk_level", "?")
                label = result.get("prediction_label", "?")
                marker = "🚨" if risk in ("CRITICAL", "HIGH") else "⚠️ " if risk == "MEDIUM" else "✅"
                print(f"{marker} [{risk}] {label}")
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({"file": path.name, "error": str(e)})
        Path("angelcare_results.json").write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to angelcare_results.json")
        return

    # Default: Cosmos
    model, processor = load_model()
    run_batch(model, processor, video_paths, source)

    # Interactive mode: keep model loaded, accept new paths
    if args.interactive:
        print("\n=== Interactive mode (model stays loaded) ===")
        print("Enter a video path or directory. Type 'q' to quit.\n")
        while True:
            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.lower() in ("q", "quit", "exit"):
                break
            if not user_input:
                continue
            paths = collect_videos(user_input)
            if not paths:
                print(f"No videos found at: {user_input}")
                continue
            run_batch(model, processor, paths, user_input)


if __name__ == "__main__":
    main()
