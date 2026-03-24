from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProjectorConfig:
    """
    Configuration for the alpha-gated projector.

    Attributes:
        input_dim: Size of encoder embeddings (e.g., SecBERT hidden dim).
        output_dim: Target embedding size for the decoder LM.
        gate_hidden_dim: Hidden size for the gating network; 0 disables hidden layer.
        dropout: Dropout probability applied to projected embeddings.
        use_layernorm: Apply LayerNorm after the linear projection.
        alpha_activation: Activation applied to gate logits ("sigmoid" or "softplus").
        alpha_temperature: Optional temperature scaling for gate logits.
    """

    input_dim: int = 768
    output_dim: int = 0  # Auto-set to decoder hidden size when 0
    gate_hidden_dim: int = 128
    dropout: float = 0.0
    use_layernorm: bool = False
    alpha_activation: str = "sigmoid"
    alpha_temperature: float = 1.0


class GatedProjector(nn.Module):
    """
    Projects SecBERT embeddings into the decoder token space while learning
    per-log importance weights ("alphas") used for interpretability.
    """

    def __init__(self, config: Optional[ProjectorConfig] = None) -> None:
        super().__init__()
        self.config = config or ProjectorConfig()

        self.projection = nn.Linear(self.config.input_dim, self.config.output_dim)
        self.layernorm = nn.LayerNorm(self.config.output_dim) if self.config.use_layernorm else None
        self.dropout = nn.Dropout(self.config.dropout) if self.config.dropout > 0 else None
        self.gate = self._build_gate_network()

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: Tensor of shape (batch, input_dim).

        Returns:
            gated_projection: Projected embeddings scaled by alpha.
            alphas: Importance weights with shape (batch, 1).
        """
        projected = self.projection(embeddings)
        if self.layernorm:
            projected = self.layernorm(projected)
        if self.dropout:
            projected = self.dropout(projected)

        alphas = self._compute_alpha(embeddings)
        gated = projected * alphas
        return gated, alphas

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_gate_network(self) -> nn.Module:
        layers = []
        if self.config.gate_hidden_dim > 0:
            layers.append(nn.Linear(self.config.input_dim, self.config.gate_hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Linear(self.config.gate_hidden_dim, 1))
        else:
            layers.append(nn.Linear(self.config.input_dim, 1))
        return nn.Sequential(*layers)

    def _compute_alpha(self, embeddings: torch.Tensor) -> torch.Tensor:
        logits = self.gate(embeddings)
        if self.config.alpha_temperature != 1.0:
            logits = logits / self.config.alpha_temperature

        if self.config.alpha_activation == "sigmoid":
            alpha = torch.sigmoid(logits)
        elif self.config.alpha_activation == "softplus":
            alpha = F.softplus(logits)
        else:
            raise ValueError(f"Unsupported alpha activation: {self.config.alpha_activation}")

        return alpha
