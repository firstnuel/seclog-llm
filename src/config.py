from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .models.encoder import EncoderConfig
from .models.projector import ProjectorConfig
from .models.sec_logllm import SecLogLLMConfig


@dataclass
class DatasetPaths:
    """Centralizes file locations for train/val/eval splits."""

    root: Path = Path("data")
    processed: Path = Path("data/processed")
    mappings_dir: Path = Path("data/mappings")

    train_file: str = "train_labeled.jsonl"
    val_file: str = "val_labeled.jsonl"
    openssh_eval_file: str = "openssh_eval.jsonl"
    bgl_eval_file: str = "ood_bgl.jsonl"
    mitre_map_file: str = "mitre_codes.json"

    def processed_path(self, filename: str) -> Path:
        return self.processed / filename

    @property
    def train_path(self) -> Path:
        return self.processed_path(self.train_file)

    @property
    def val_path(self) -> Path:
        return self.processed_path(self.val_file)

    @property
    def openssh_eval_path(self) -> Path:
        return self.processed_path(self.openssh_eval_file)

    @property
    def bgl_eval_path(self) -> Path:
        return self.processed_path(self.bgl_eval_file)

    @property
    def mitre_mapping_path(self) -> Path:
        return self.mappings_dir / self.mitre_map_file


@dataclass
class OptimizerConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 200
    max_grad_norm: float = 1.0


@dataclass
class StageTrainingConfig:
    """
    Stage-specific hyperparameters for the three-phase fine-tuning routine.
    """

    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    freeze_encoder: bool
    freeze_projector: bool
    freeze_decoder: bool
    learning_rate: Optional[float] = None


@dataclass
class EvaluationConfig:
    openssh_sample_size: int = 10_000
    bgl_sample_size: int = 2_000
    hallucination_expect_null_mitre: bool = True


def _default_stage_hparams() -> Dict[str, StageTrainingConfig]:
    return {
        "stage1": StageTrainingConfig(
            epochs=1,
            batch_size=8,
            gradient_accumulation_steps=8,
            freeze_encoder=True,
            freeze_projector=True,
            freeze_decoder=False,
            learning_rate=1e-4,
        ),
        "stage2": StageTrainingConfig(
            epochs=2,
            batch_size=8,
            gradient_accumulation_steps=4,
            freeze_encoder=False,
            freeze_projector=False,
            freeze_decoder=True,
            learning_rate=2e-4,
        ),
        "stage3": StageTrainingConfig(
            epochs=1,
            batch_size=4,
            gradient_accumulation_steps=8,
            freeze_encoder=False,
            freeze_projector=False,
            freeze_decoder=False,
            learning_rate=1e-4,
        ),
    }


@dataclass
class ExperimentConfig:
    """
    Top-level configuration bundle consumed by trainers and scripts.
    """

    data: DatasetPaths = field(default_factory=DatasetPaths)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    sec_logllm: SecLogLLMConfig = field(
        default_factory=lambda: SecLogLLMConfig(
            encoder=EncoderConfig(),
            projector=ProjectorConfig(),
        )
    )
    stages: Dict[str, StageTrainingConfig] = field(default_factory=_default_stage_hparams)

    def get_stage(self, name: str) -> StageTrainingConfig:
        try:
            return self.stages[name]
        except KeyError as exc:  # pragma: no cover - convenience guard
            raise KeyError(f"Unknown training stage '{name}'. Available: {list(self.stages)}") from exc
