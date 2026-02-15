"""
AngelCare — Dataset Preparation for Post-Training
===================================================
Converts fall detection datasets into Llava format for SFT with Cosmos Reason 2.

Datasets supported:
  - GMDC-SA24: 160 videos (79 fall + 81 ADL) with CSV descriptions
  - videos_compressed/: Personal clips in bad/good folders

Output: Llava-format JSON ready for cosmos-rl SFT pipeline.

Usage:
    python prepare_llava_dataset.py
    python prepare_llava_dataset.py --output custom_output.json
    python prepare_llava_dataset.py --normalize-videos normalized/
    python prepare_llava_dataset.py --stats-only
"""

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

# ── AngelCare classification schema (matches core/inference.py) ──

CLASSES = {
    "fall":       {"id": 0, "label": "Fall Detected",         "risk": "CRITICAL", "action": "Call emergency services immediately"},
    "immobility": {"id": 1, "label": "Immobility Alert",      "risk": "HIGH",     "action": "Alert caregiver — check on elder"},
    "unsteady":   {"id": 2, "label": "Unsteady Movement",     "risk": "MEDIUM",   "action": "Monitor closely for potential fall"},
    "distress":   {"id": 3, "label": "Distress Posture",      "risk": "HIGH",     "action": "Alert caregiver — possible pain or distress"},
    "walking":    {"id": 4, "label": "Normal Walking",        "risk": "SAFE",     "action": "No action needed"},
    "sitting":    {"id": 5, "label": "Normal Sitting",        "risk": "SAFE",     "action": "No action needed"},
    "daily":      {"id": 6, "label": "Normal Daily Activity", "risk": "SAFE",     "action": "No action needed"},
    "resting":    {"id": 7, "label": "Resting or Sleeping",   "risk": "SAFE",     "action": "No action needed"},
}

USER_PROMPT = (
    "Analyze the video and classify the elder's safety status. "
    "Output a JSON object with prediction_class_id, prediction_label, risk_level, "
    "video_description, and risk_assessment."
)


def make_llava_entry(video_id: str, video_path: str, class_key: str, description: str) -> dict:
    cls = CLASSES[class_key]
    answer = json.dumps({
        "prediction_class_id": cls["id"],
        "prediction_label": cls["label"],
        "risk_level": cls["risk"],
        "video_description": description,
        "risk_assessment": {
            "is_at_risk": cls["risk"] in ("CRITICAL", "HIGH", "MEDIUM"),
            "recommended_action": cls["action"],
        }
    }, indent=2)

    return {
        "id": video_id,
        "video": video_path,
        "conversations": [
            {"from": "human", "value": f"<video>\n{USER_PROMPT}"},
            {"from": "gpt", "value": answer},
        ]
    }


def map_gmdc_to_class(description: str, is_fall: bool) -> tuple[str, str]:
    """Map GMDC CSV description to AngelCare class key."""
    if is_fall:
        return "fall", description

    desc = description.lower()
    if any(w in desc for w in ["sleeping", "lying", "sleep", "bed to sitting"]):
        # "sleeping" or transitions involving lying down
        if "sitting" in desc and "sleeping" not in desc:
            return "sitting", description
        return "resting", description
    if any(w in desc for w in ["sitting", "sit"]):
        return "sitting", description
    if any(w in desc for w in ["walking", "walk"]):
        return "walking", description
    return "daily", description


def process_gmdc(base_dir: Path) -> list[dict]:
    """Process GMDCSA-24: 4 subjects × (Fall + ADL) with CSV descriptions."""
    entries = []
    for subject_dir in sorted(base_dir.iterdir()):
        if not subject_dir.is_dir() or subject_dir.name.startswith("."):
            continue

        for category in ["Fall", "ADL"]:
            csv_file = subject_dir / f"{category}.csv"
            video_dir = subject_dir / category
            if not csv_file.exists() or not video_dir.exists():
                continue

            is_fall = category == "Fall"

            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = (row.get("File Name") or "").strip()
                    description = (row.get("Description") or "").strip()
                    if not filename or not description:
                        continue

                    video_path = video_dir / filename
                    if not video_path.exists():
                        print(f"  WARN: missing {video_path}")
                        continue

                    class_key, desc = map_gmdc_to_class(description, is_fall)
                    video_id = f"gmdc_{subject_dir.name}_{category}_{video_path.stem}".replace(" ", "_")

                    entries.append(make_llava_entry(
                        video_id=video_id,
                        video_path=str(video_path.resolve()),
                        class_key=class_key,
                        description=desc,
                    ))
    return entries


def process_personal(base_dir: Path) -> list[dict]:
    """Process personal videos from videos_compressed/ (bad=fall, good=safe)."""
    entries = []
    for subfolder, class_key, desc in [
        ("bad", "fall", "Elder falling or in a risk situation"),
        ("good", "daily", "Elder performing normal daily activity"),
    ]:
        folder = base_dir / subfolder
        if not folder.exists():
            continue
        for video_file in sorted(folder.glob("*.mp4")):
            video_id = f"personal_{subfolder}_{video_file.stem}".replace(" ", "_")
            entries.append(make_llava_entry(
                video_id=video_id,
                video_path=str(video_file.resolve()),
                class_key=class_key,
                description=desc,
            ))
    return entries


def normalize_video(src: Path, dst: Path, width=640, fps=30) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True
    result = subprocess.run([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale={width}:-2",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an",
        str(dst),
    ], capture_output=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Prepare Llava dataset for AngelCare SFT")
    parser.add_argument("--output", default="angelcare_llava_train.json")
    parser.add_argument("--normalize-videos", default=None,
                        help="Normalize all videos to this directory (640px, 30fps)")
    parser.add_argument("--test-split", type=float, default=0.2,
                        help="Fraction of data to hold out for testing (default: 0.2)")
    parser.add_argument("--test-output", default="angelcare_llava_test.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).parent
    project_root = base.parent

    all_entries = []

    # 1. GMDC-SA24
    gmdc_dir = base / "gmdc"
    if gmdc_dir.exists():
        gmdc = process_gmdc(gmdc_dir)
        print(f"GMDC-SA24:       {len(gmdc):>4} entries")
        all_entries.extend(gmdc)
    else:
        print("GMDC-SA24:       not found, skipping")

    # 2. Personal videos
    personal_dir = project_root / "videos_compressed"
    if personal_dir.exists():
        personal = process_personal(personal_dir)
        print(f"Personal clips:  {len(personal):>4} entries")
        all_entries.extend(personal)
    else:
        print("Personal clips:  not found, skipping")

    # Stats
    class_counts = {}
    for entry in all_entries:
        answer = json.loads(entry["conversations"][1]["value"])
        label = answer["prediction_label"]
        class_counts[label] = class_counts.get(label, 0) + 1

    print(f"\n{'='*50}")
    print(f"Total: {len(all_entries)} entries\n")
    print("Class distribution:")
    for label, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 2)
        print(f"  {label:.<30} {count:>4}  {bar}")

    if args.stats_only:
        return

    # Normalize if requested
    if args.normalize_videos:
        norm_dir = Path(args.normalize_videos)
        print(f"\nNormalizing videos to {norm_dir}/...")
        updated = []
        for i, entry in enumerate(all_entries):
            src = Path(entry["video"])
            dst = norm_dir / f"{entry['id']}.mp4"
            print(f"  [{i+1}/{len(all_entries)}] {src.name}", end=" ")
            if normalize_video(src, dst):
                entry["video"] = str(dst.resolve())
                updated.append(entry)
                print("OK")
            else:
                print("FAIL")
        all_entries = updated

    # Stratified train/test split — keeps class proportions in both sets
    random.seed(args.seed)
    by_class = {}
    for entry in all_entries:
        label = json.loads(entry["conversations"][1]["value"])["prediction_label"]
        by_class.setdefault(label, []).append(entry)

    train_entries, test_entries = [], []
    for label, entries in by_class.items():
        random.shuffle(entries)
        n_test = max(1, round(len(entries) * args.test_split))
        test_entries.extend(entries[:n_test])
        train_entries.extend(entries[n_test:])

    random.shuffle(train_entries)
    random.shuffle(test_entries)

    print(f"\nSplit: {len(train_entries)} train / {len(test_entries)} test ({args.test_split:.0%} held out)")

    # Write
    out = Path(args.output)
    out.write_text(json.dumps(train_entries, indent=2))
    print(f"Written: {out} ({len(train_entries)} entries)")

    test_out = Path(args.test_output)
    test_out.write_text(json.dumps(test_entries, indent=2))
    print(f"Written: {test_out} ({len(test_entries)} entries)")


if __name__ == "__main__":
    main()
