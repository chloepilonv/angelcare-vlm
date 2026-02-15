"""
AngelCare — SFT Training with HuggingFace TRL
================================================
Fine-tunes Cosmos Reason 2 (8B) on the AngelCare fall detection dataset.
Uses TRL's SFTTrainer as a drop-in replacement for cosmos-rl.

Usage:
    python train_trl.py
    python train_trl.py --epochs 5 --lr 1e-5
    python train_trl.py --dataset custom.json --output outputs/custom
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


SYSTEM_PROMPT = (
    "You are an expert Elder Care Safety Monitor analyzing home surveillance footage. "
    "Classify the video into one of 8 safety categories and output structured JSON."
)


def format_entry(entry, processor):
    """Convert a Llava-format entry into a chat message for SFT."""
    user_text = entry["conversations"][0]["value"].replace("<video>", "").strip()
    assistant_text = entry["conversations"][1]["value"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": entry["video"], "fps": 4, "max_pixels": 81920},
                {"type": "text", "text": user_text},
            ],
        },
        {"role": "assistant", "content": assistant_text},
    ]
    return processor.apply_chat_template(messages, tokenize=False)


def main():
    parser = argparse.ArgumentParser(description="AngelCare SFT Training")
    parser.add_argument("--dataset", default="angelcare_llava_train.json")
    parser.add_argument("--output", default="outputs/angelcare_sft/final")
    parser.add_argument("--model", default="nvidia/Cosmos-Reason2-8B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    # Load dataset
    dataset = json.loads(Path(args.dataset).read_text())
    print(f"=== AngelCare SFT Training ===")
    print(f"Dataset: {len(dataset)} samples")
    print(f"Model:   {args.model}")
    print(f"Output:  {args.output}")
    print(f"Epochs:  {args.epochs}")
    print()

    # Load model with QLoRA (4-bit) to fit on single GPU
    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
    )

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "right"

    # LoRA config for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # Format dataset
    print("Formatting dataset...")
    train_texts = []
    for i, entry in enumerate(dataset):
        try:
            text = format_entry(entry, processor)
            train_texts.append({"text": text})
        except Exception as e:
            print(f"  WARN: skipping {entry['id']}: {e}")

    print(f"Formatted {len(train_texts)} / {len(dataset)} samples")

    # SFT config
    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        max_length=4096,
        dataset_text_field="text",
        remove_unused_columns=False,
    )

    # Train
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_texts),
        peft_config=lora_config,
        processing_class=processor.tokenizer,
    )

    trainer.train()

    # Save
    print(f"\nSaving model to {args.output}...")
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)

    print("\n=== Training complete ===")
    print(f"Checkpoint: {args.output}")
    print(f"\nTo evaluate:")
    print(f"  python evaluate.py --model {args.output} --dataset angelcare_llava_test.json")


if __name__ == "__main__":
    main()
