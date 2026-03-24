"""
Stage 2 projector alignment.

This script loads the Stage 1 LoRA adapter, freezes the decoder, and optimizes
the encoder + projector using the sparse alignment loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from peft import PeftModel

from src.config import ExperimentConfig
from src.models.sec_logllm import SecLogLLM
from src.utils.data_loader import SecLogCollator, SecLogDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 projector alignment.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/processed/train_labeled.jsonl"),
        help="JSONL/CSV file with training samples.",
    )
    parser.add_argument(
        "--stage1-adapter",
        type=Path,
        required=True,
        help="Directory containing the Stage 1 LoRA adapter (e.g., results/checkpoints/stage1_final/final_adapter).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/checkpoints/stage2"),
        help="Where to save Stage 2 checkpoints.",
    )
    parser.add_argument("--epochs", type=int, default=ExperimentConfig().stages["stage2"].epochs)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig().stages["stage2"].batch_size)
    parser.add_argument("--grad-accum", type=int, default=ExperimentConfig().stages["stage2"].gradient_accumulation_steps)
    parser.add_argument("--learning-rate", type=float, default=ExperimentConfig().stages["stage2"].learning_rate or 2e-4)
    parser.add_argument("--lambda-l1", type=float, default=0.001, help="Sparsity weight for alpha gates (keep small to avoid gate collapse).")
    parser.add_argument("--save-every", type=int, default=2000, help="Save checkpoint every N optimizer steps.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for debugging.")
    return parser.parse_args()


def format_targets(labels: List[str], metadata: List[dict]) -> List[str]:
    targets = []
    for label, meta in zip(labels, metadata):
        payload = {
            "label": label,
            "mitre_t_code": meta.get("mitre_t_code"),
            "mitre_technique": meta.get("mitre_technique"),
            "reasoning": meta.get("reasoning", ""),
        }
        targets.append(json.dumps(payload))
    return targets


def save_checkpoint(model, config, output_dir, tag):
    path = output_dir / f"stage2_{tag}.pt"
    torch.save(
        {
            "encoder_state": model.encoder.state_dict(),
            "projector_state": model.projector.state_dict(),
            "config": config.sec_logllm,
        },
        path,
    )
    print(f"\n  💾 Saved checkpoint: {path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SecLogDataset(args.train_path)
    if args.max_samples:
        dataset.samples = dataset.samples[: args.max_samples]
    collator = SecLogCollator()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    base_model = AutoModelForCausalLM.from_pretrained(ExperimentConfig().sec_logllm.decoder_model_name_or_path)
    peft_decoder = PeftModel.from_pretrained(base_model, args.stage1_adapter)
    peft_decoder.eval()
    for param in peft_decoder.parameters():
        param.requires_grad = False

    config = ExperimentConfig()
    model = SecLogLLM(config.sec_logllm)
    model.decoder_model = peft_decoder
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=config.optimizer.betas,
        weight_decay=config.optimizer.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(dataloader, desc=f"Stage 2 Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()
        for step, batch in enumerate(progress):
            log_sequences = batch["log_sequences"]
            labels = batch["labels"]
            metadata = batch["metadata"]
            targets = format_targets(labels, metadata)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                outputs = model(log_sequences=log_sequences, target_texts=targets)
                # Use decoder's next-token prediction loss to align projector
                decoder_loss = outputs.decoder_outputs.loss
                # Gentle sparsity regularization on alpha gates
                l1_loss = outputs.alphas.abs().mean()
                loss = decoder_loss + (args.lambda_l1 * l1_loss)

            scaler.scale(loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                alpha_mean = outputs.alphas.detach().mean().item()
                progress.set_postfix({"loss": loss.item(), "α": f"{alpha_mean:.4f}", "step": global_step})

                # Periodic checkpoint
                if global_step % args.save_every == 0:
                    save_checkpoint(model, config, args.output_dir, f"step_{global_step}")

            # Free memory to prevent OOM buildup
            del outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # End-of-epoch checkpoint
        save_checkpoint(model, config, args.output_dir, f"epoch_{epoch+1}")

    # Final checkpoint
    save_checkpoint(model, config, args.output_dir, "checkpoint")
    print(f"Stage 2 complete! Saved state to {args.output_dir}")


if __name__ == "__main__":
    main()

