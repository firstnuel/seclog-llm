"""
Stage 1 decoder supervised fine-tuning (SFT) for Sec-LogLLM.

This script adapts Qwen/Qwen2.5-3B-Instruct to the AIT LLM-labeled dataset by
running LoRA SFT over individual log lines. It mirrors the workflow described in
`thesis_execution_plan_v2.md` Task 4.2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 1 decoder SFT.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/processed/train_labeled.jsonl"),
        help="JSONL file with LLM-labeled logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/checkpoints/stage1"),
        help="Directory for checkpoints and final adapter.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF identifier for the decoder backbone.",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of SFT epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Per-device batch size."
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for SFT.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length for SFTTrainer.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint directory to resume training from.",
    )
    return parser.parse_args()


def load_training_data(path: Path) -> Dataset:
    """Load JSONL and format it for TRL's SFTTrainer."""
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.strip()
            if not cleaned:
                continue
            entry = json.loads(cleaned)
            raw_log = entry.get("raw") or entry.get("raw_log") or ""
            user_msg = f"Analyze this log for security threats:\n{raw_log}"
            assistant_msg = json.dumps(
                {
                    "label": entry.get("label", "Normal"),
                    "mitre_t_code": entry.get("mitre_t_code"),
                    "mitre_technique": entry.get("mitre_technique"),
                    "reasoning": entry.get("reasoning", ""),
                },
                indent=2,
            )
            chat_text = (
                "<|im_start|>system\n"
                "You are a security analyst specializing in log analysis and threat detection.\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"{user_msg}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
                f"{assistant_msg}\n"
                "<|im_end|>"
            )
            examples.append({"text": chat_text})
    if not examples:
        raise RuntimeError(f"No training samples found in {path}")
    return Dataset.from_list(examples)


def tokenize_batch(batch, tokenizer, max_length):
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
        # padding="max_length",
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.train_path}")
    train_dataset = load_training_data(args.train_path)

    print(f"Loading model {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        report_to="none",
    )

    print("Tokenizing dataset...")
    tokenized_dataset = train_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,          # Pad to the longest sequence in the *batch*
            pad_to_multiple_of=8   # Optimizes memory for A100 GPUs
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting Stage 1 SFT...")
    resume_path = str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=resume_path)
    model.save_pretrained(args.output_dir / "final_adapter")
    tokenizer.save_pretrained(args.output_dir / "final_adapter")
    print("Stage 1 complete!")


if __name__ == "__main__":
    main()
