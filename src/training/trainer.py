from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from src.config import ExperimentConfig, StageTrainingConfig
from src.models.sec_logllm import SecLogLLM
from src.utils.data_loader import SecLogCollator, SecLogDataset


LOGGER = logging.getLogger(__name__)


class SecLogLLMTrainer:
    """
    Drives the three-stage Sec-LogLLM training regime.

    Responsibilities:
        * Build datasets/dataloaders from ExperimentConfig paths
        * Apply per-stage freezing rules (encoder/projector/decoder)
        * Run gradient-accumulated training loops with label formatting
    """

    def __init__(
        self,
        model: SecLogLLM,
        config: ExperimentConfig,
        *,
        device: Optional[torch.device] = None,
        train_dataset: Optional[SecLogDataset] = None,
        val_dataset: Optional[SecLogDataset] = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_dataset = train_dataset or self._load_dataset(self.config.data.train_path)
        self.val_dataset = val_dataset

        self.collator = SecLogCollator()
        self.global_step = 0
        self.global_updates = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, stages: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Run the requested training stages (defaults to all configured stages).
        Returns per-stage statistics such as average loss and number of updates.
        """
        stages = stages or list(self.config.stages.keys())
        history: Dict[str, Dict[str, float]] = {}
        for stage_name in stages:
            if stage_name not in self.config.stages:
                raise ValueError(f"Stage '{stage_name}' not defined in ExperimentConfig.")
            history[stage_name] = self._run_stage(stage_name, self.config.stages[stage_name])
        return history

    # ------------------------------------------------------------------
    # Stage execution
    # ------------------------------------------------------------------
    def _run_stage(self, name: str, stage_cfg: StageTrainingConfig) -> Dict[str, float]:
        LOGGER.info("Starting %s: epochs=%s, batch_size=%s", name, stage_cfg.epochs, stage_cfg.batch_size)
        self._apply_stage_freezing(stage_cfg)

        dataloader = self._build_dataloader(
            self.train_dataset,
            batch_size=stage_cfg.batch_size,
            shuffle=True,
        )

        optimizer = self._build_optimizer(stage_cfg)

        total_loss = 0.0
        total_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(stage_cfg.epochs):
            self.model.train()
            for step, batch in enumerate(dataloader):
                loss = self._forward_batch(batch)
                loss_val = loss.item()
                total_loss += loss_val
                total_steps += 1
                self.global_step += 1

                scaled_loss = loss / max(1, stage_cfg.gradient_accumulation_steps)
                scaled_loss.backward()

                if (
                    (step + 1) % stage_cfg.gradient_accumulation_steps == 0
                    or (step + 1) == len(dataloader)
                ):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.optimizer.max_grad_norm,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    self.global_updates += 1

        avg_loss = total_loss / max(1, total_steps)
        LOGGER.info("Finished %s: avg_loss=%.4f, steps=%s", name, avg_loss, total_steps)

        return {
            "avg_loss": avg_loss,
            "steps": float(total_steps),
            "updates": float(self.global_updates),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _forward_batch(self, batch: Dict[str, Sequence[str]]) -> torch.Tensor:
        log_sequences = batch["log_sequences"]
        labels = batch["labels"]
        metadata = batch["metadata"]

        target_texts = self._format_target_texts(labels, metadata)
        outputs = self.model(
            log_sequences=log_sequences,
            target_texts=target_texts,
        )

        decoder_outputs = getattr(outputs, "decoder_outputs", None)
        if decoder_outputs is None or decoder_outputs.loss is None:
            raise RuntimeError("Model did not return a loss; ensure decoder outputs include .loss.")
        return decoder_outputs.loss

    def _format_target_texts(self, labels: Sequence[str], metadata: Sequence[Dict[str, Any]]) -> List[str]:
        payloads = []
        for label, meta in zip(labels, metadata):
            data = {
                "label": label,
                "reasoning": meta.get("reasoning", ""),
                "mitre_t_code": meta.get("mitre_t_code"),
            }
            payloads.append(json.dumps(data))
        return payloads

    def _build_dataloader(
        self,
        dataset: SecLogDataset,
        *,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collator,
        )

    def _build_optimizer(self, stage_cfg: StageTrainingConfig) -> torch.optim.Optimizer:
        lr = stage_cfg.learning_rate or self.config.optimizer.learning_rate
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters found for this stage.")
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=self.config.optimizer.betas,
            weight_decay=self.config.optimizer.weight_decay,
        )

    def _apply_stage_freezing(self, stage_cfg: StageTrainingConfig) -> None:
        if stage_cfg.freeze_encoder:
            self.model.freeze_encoder()
        else:
            self.model.unfreeze_encoder()

        if stage_cfg.freeze_decoder:
            self.model.freeze_decoder()
        else:
            self.model.unfreeze_decoder()

        if hasattr(self.model, "projector"):
            self._set_module_trainable(self.model.projector, not stage_cfg.freeze_projector)

    @staticmethod
    def _set_module_trainable(module: torch.nn.Module, trainable: bool) -> None:
        for param in module.parameters():
            param.requires_grad = trainable

    def _load_dataset(self, path: Path) -> SecLogDataset:
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
        return SecLogDataset(path)
