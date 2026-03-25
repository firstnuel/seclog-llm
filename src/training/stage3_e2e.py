"""
Stage 3 end-to-end fine-tuning.

Loads the Stage 1 LoRA adapter + Stage 2 encoder/projector checkpoint,
then trains the projector + LoRA adapters with generation loss.
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
    parser = argparse.ArgumentParser(description="Stage 3 end-to-end fine-tuning.")
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
        help="Directory containing the Stage 1 LoRA adapter.",
    )
    parser.add_argument(
        "--stage2-checkpoint",
        type=Path,
        required=True,
        help="Stage 2 checkpoint with encoder/projector weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/checkpoints/stage3"),
        help="Where to save Stage 3 checkpoints.",
    )
    parser.add_argument("--epochs", type=int, default=ExperimentConfig().stages["stage3"].epochs)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig().stages["stage3"].batch_size)
    parser.add_argument("--grad-accum", type=int, default=ExperimentConfig().stages["stage3"].gradient_accumulation_steps)
    parser.add_argument("--learning-rate", type=float, default=ExperimentConfig().stages["stage3"].learning_rate or 1e-4)
    parser.add_argument("--save-every", type=int, default=2000, help="Save checkpoint every N optimizer steps.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume training from a Stage 3 checkpoint.")
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


def set_lora_trainable(model: torch.nn.Module) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        requires_grad = "lora" in name.lower()
        param.requires_grad = requires_grad
        if requires_grad:
            trainable += param.numel()
    return trainable


def save_checkpoint(model, config, output_dir, tag, global_step):
    path = output_dir / f"stage3_{tag}.pt"
    torch.save(
        {
            "encoder_state": model.encoder.state_dict(),
            "projector_state": model.projector.state_dict(),
            "decoder_lora_state": model.decoder_model.state_dict(),
            "config": config.sec_logllm,
            "global_step": global_step,
        },
        path,
    )
    print(f"\n  💾 Saved checkpoint: {path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SecLogDataset(
        args.train_path,
        sequence_keys=("log_sequence", "logs", "raw_logs", "raw_log", "raw", "message"),
    )
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

    config = ExperimentConfig()
    base_model = AutoModelForCausalLM.from_pretrained(config.sec_logllm.decoder_model_name_or_path)
    peft_decoder = PeftModel.from_pretrained(base_model, args.stage1_adapter)
    peft_decoder.train()

    model = SecLogLLM(config.sec_logllm)
    model.decoder_model = peft_decoder

    # Load weights: either resume from Stage 3 checkpoint or start from Stage 2
    if args.resume_from:
        print(f"Resuming from {args.resume_from}")
        resume_ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model.encoder.load_state_dict(resume_ckpt["encoder_state"])
        model.projector.load_state_dict(resume_ckpt["projector_state"])
        if "decoder_lora_state" in resume_ckpt:
            model.decoder_model.load_state_dict(resume_ckpt["decoder_lora_state"], strict=False)
            print(f"  Loaded decoder LoRA from resume checkpoint")
        resume_step = resume_ckpt.get("global_step", 0)
        del resume_ckpt
    else:
        checkpoint = torch.load(args.stage2_checkpoint, map_location="cpu", weights_only=False)
        model.encoder.load_state_dict(checkpoint["encoder_state"])
        model.projector.load_state_dict(checkpoint["projector_state"])
        resume_step = 0
        del checkpoint

    model.freeze_encoder()
    for param in model.projector.parameters():
        param.requires_grad = True

    lora_params = set_lora_trainable(model.decoder_model)
    if lora_params == 0:
        raise RuntimeError("No LoRA parameters found to train. Check adapter loading.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=config.optimizer.betas,
        weight_decay=config.optimizer.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    global_step = resume_step
    steps_to_skip = resume_step * args.grad_accum  # dataloader steps to skip
    if resume_step > 0:
        print(f"  Resuming from global_step={resume_step}, skipping {steps_to_skip} dataloader steps")

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(dataloader, desc=f"Stage 3 Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()
        for step, batch in enumerate(progress):
            # Skip already-processed steps when resuming
            if step < steps_to_skip:
                if step % 1000 == 0 and step > 0:
                    progress.set_postfix({"skipping": f"{step}/{steps_to_skip}"})
                continue
            log_sequences = batch["log_sequences"]
            labels = batch["labels"]
            metadata = batch["metadata"]
            targets = format_targets(labels, metadata)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                outputs = model(log_sequences=log_sequences, target_texts=targets)
                loss = outputs.decoder_outputs.loss

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
                    save_checkpoint(model, config, args.output_dir, f"step_{global_step}", global_step)

            # Free memory to prevent OOM buildup
            del outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # End-of-epoch checkpoint
        save_checkpoint(model, config, args.output_dir, f"epoch_{epoch+1}", global_step)

    print(f"Stage 3 complete! Saved checkpoints to {args.output_dir}")


if __name__ == "__main__":
    main()

