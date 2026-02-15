"""
AngelCare — Evaluate model accuracy (zero-shot vs fine-tuned)
==============================================================
Runs inference on a held-out set of videos and compares predictions
against ground truth labels from the Llava dataset.

Usage:
    # Evaluate zero-shot (base model)
    python evaluate.py --model nvidia/Cosmos-Reason2-8B --dataset angelcare_llava_train.json

    # Evaluate fine-tuned model
    python evaluate.py --model outputs/angelcare_sft/checkpoint --dataset angelcare_llava_train.json

    # Evaluate on specific videos only
    python evaluate.py --model nvidia/Cosmos-Reason2-8B --dataset angelcare_llava_train.json --max-samples 20
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import transformers

# Reuse inference logic from core/inference.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.inference import analyze_video, SYSTEM_INSTRUCTIONS, USER_PROMPT_CONTENT


def load_model(model_path: str):
    print(f"=== Loading model: {model_path} ===")
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(model_path)

    PIXELS_PER_TOKEN = 32**2
    min_vision_tokens, max_vision_tokens = 256, 8192
    processor.image_processor.size = processor.video_processor.size = {
        "shortest_edge": min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": max_vision_tokens * PIXELS_PER_TOKEN,
    }
    print(f"Model loaded on {torch.cuda.get_device_name(0)}")
    return model, processor


def main():
    parser = argparse.ArgumentParser(description="Evaluate AngelCare model accuracy")
    parser.add_argument("--model", required=True, help="Model path (HF repo or local checkpoint)")
    parser.add_argument("--dataset", required=True, help="Llava-format JSON dataset")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to evaluate (0=all)")
    parser.add_argument("--output", default="eval_results.json", help="Output results file")
    args = parser.parse_args()

    # Load dataset
    dataset = json.loads(Path(args.dataset).read_text())
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]

    print(f"Evaluating {len(dataset)} samples")

    # Load model
    model, processor = load_model(args.model)
    transformers.set_seed(0)

    # Run evaluation
    results = []
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}

    for i, entry in enumerate(dataset):
        video_path = entry["video"]
        ground_truth = json.loads(entry["conversations"][1]["value"])
        gt_label = ground_truth["prediction_label"]
        gt_risk = ground_truth["risk_level"]

        print(f"[{i+1}/{len(dataset)}] {Path(video_path).name} (GT: {gt_label})", end=" → ")

        try:
            prediction = analyze_video(model, processor, video_path)
            pred_label = prediction.get("prediction_label", "Parse Error")
            pred_risk = prediction.get("risk_level", "UNKNOWN")

            is_correct = pred_label == gt_label
            # Also count risk-level match (more lenient)
            risk_correct = pred_risk == gt_risk

            if is_correct:
                correct += 1
                print(f"CORRECT ({pred_label})")
            else:
                print(f"WRONG (predicted: {pred_label}, expected: {gt_label})")

            total += 1
            class_total[gt_label] = class_total.get(gt_label, 0) + 1
            if is_correct:
                class_correct[gt_label] = class_correct.get(gt_label, 0) + 1

            results.append({
                "id": entry["id"],
                "video": video_path,
                "ground_truth_label": gt_label,
                "ground_truth_risk": gt_risk,
                "predicted_label": pred_label,
                "predicted_risk": pred_risk,
                "exact_match": is_correct,
                "risk_match": risk_correct,
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"id": entry["id"], "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.model}")
    print(f"{'='*60}")
    print(f"Exact label accuracy: {correct}/{total} ({100*correct/total:.1f}%)" if total else "No results")

    risk_correct_count = sum(1 for r in results if r.get("risk_match"))
    print(f"Risk level accuracy: {risk_correct_count}/{total} ({100*risk_correct_count/total:.1f}%)" if total else "")

    print(f"\nPer-class accuracy:")
    for label in sorted(class_total.keys()):
        c = class_correct.get(label, 0)
        t = class_total[label]
        print(f"  {label:.<30} {c}/{t} ({100*c/t:.0f}%)")

    # Save
    output = {
        "model": args.model,
        "total_samples": total,
        "exact_accuracy": correct / total if total else 0,
        "risk_accuracy": risk_correct_count / total if total else 0,
        "per_class": {l: {"correct": class_correct.get(l, 0), "total": class_total[l]}
                      for l in class_total},
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
