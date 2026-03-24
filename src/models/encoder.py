from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass
class EncoderConfig:
    """
    Configuration bundle for the SecBERT encoder wrapper.

    Attributes:
        model_name_or_path: Hugging Face repo ID or local checkpoint path.
        max_length: Number of tokens to keep per log sequence.
        padding: Tokenizer padding strategy.
        truncation: Whether to truncate overly long logs.
        pooling: Strategy for reducing token embeddings to a single vector.
        normalize: Apply L2 normalization to pooled embeddings.
        cache_dir: Optional cache location for HF downloads.
        gradient_checkpointing: Enable memory-saving gradient checkpoints.
        output_hidden_states: Return every hidden layer from the encoder.
    """

    model_name_or_path: str = "jackaduma/SecBERT"
    max_length: int = 256
    padding: Union[str, bool] = "max_length"
    truncation: bool = True
    pooling: str = "cls"  # Options: "cls", "mean"
    normalize: bool = False
    cache_dir: Optional[str] = None
    gradient_checkpointing: bool = False
    output_hidden_states: bool = False


@dataclass
class EncoderOutput:
    """Wrapper for everything downstream components need from the encoder."""

    pooled_embeddings: torch.Tensor
    token_embeddings: torch.Tensor
    attention_mask: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None


class SecBERTEncoder(nn.Module):
    """
    Wraps a Hugging Face encoder and tokenizer so the rest of Sec-LogLLM can
    treat log embeddings as a plug-and-play component.
    """

    def __init__(
        self,
        config: Optional[EncoderConfig] = None,
        *,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model: Optional[PreTrainedModel] = None,
        tokenizer_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.config = config or EncoderConfig()
        tokenizer_kwargs = tokenizer_kwargs or {}

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            cache_dir=self.config.cache_dir,
            use_fast=True,
            **tokenizer_kwargs,
        )
        if self.tokenizer.pad_token is None:
            pad_token = self.tokenizer.eos_token or self.tokenizer.cls_token or "[PAD]"
            self.tokenizer.add_special_tokens({"pad_token": pad_token})

        self.model = model or AutoModel.from_pretrained(
            self.config.model_name_or_path,
            cache_dir=self.config.cache_dir,
            output_hidden_states=self.config.output_hidden_states,
        )
        if tokenizer is None and hasattr(self.model, "resize_token_embeddings"):
            self.model.resize_token_embeddings(len(self.tokenizer))

        if self.config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        self.hidden_size = self.model.config.hidden_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> EncoderOutput:
        """
        Run a batch of already-tokenized sequences through the encoder.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=self.config.output_hidden_states,
        )

        token_embeddings = outputs.last_hidden_state
        pooled = self._pool(token_embeddings, attention_mask)
        if self.config.normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)

        hidden_states = outputs.hidden_states if self.config.output_hidden_states else None

        return EncoderOutput(
            pooled_embeddings=pooled,
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
            hidden_states=hidden_states,
        )

    @torch.inference_mode()
    def encode_texts(
        self,
        texts: Union[str, Sequence[str]],
        *,
        batch_size: Optional[int] = 16,
        device: Optional[torch.device] = None,
    ) -> EncoderOutput:
        """
        Convenience helper for inference-time encoding of raw log lines.
        """
        self.eval()
        if isinstance(texts, str):
            texts = [texts]
        if batch_size is None or batch_size <= 0:
            batch_size = len(texts)

        pooled_chunks: List[torch.Tensor] = []
        token_chunks: List[torch.Tensor] = []
        mask_chunks: List[torch.Tensor] = []
        hidden_storage: Optional[List[List[torch.Tensor]]] = None

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_inputs = self.tokenize(batch_texts, device=device)
            outputs = self.forward(**batch_inputs)

            pooled_chunks.append(outputs.pooled_embeddings)
            token_chunks.append(outputs.token_embeddings)
            mask_chunks.append(outputs.attention_mask)

            if outputs.hidden_states:
                if hidden_storage is None:
                    hidden_storage = [[] for _ in range(len(outputs.hidden_states))]
                for idx, layer_tensor in enumerate(outputs.hidden_states):
                    hidden_storage[idx].append(layer_tensor)

        pooled_embeddings = torch.cat(pooled_chunks, dim=0)
        token_embeddings = torch.cat(token_chunks, dim=0)
        attention_mask = torch.cat(mask_chunks, dim=0)

        hidden_states = None
        if hidden_storage:
            hidden_states = tuple(torch.cat(layer_chunks, dim=0) for layer_chunks in hidden_storage)

        return EncoderOutput(
            pooled_embeddings=pooled_embeddings,
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
            hidden_states=hidden_states,
        )

    def tokenize(
        self,
        texts: Union[str, Sequence[str]],
        *,
        device: Optional[torch.device] = None,
    ):
        """
        Tokenize raw text into tensors ready for the forward pass.
        """
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            list(texts),
            padding=self.config.padding,
            truncation=self.config.truncation,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        if device is None:
            return encoded

        return {key: tensor.to(device) for key, tensor in encoded.items()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.config.pooling == "cls":
            return token_embeddings[:, 0, :]
        if self.config.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
            summed = torch.sum(token_embeddings * mask, dim=1)
            lengths = mask.sum(dim=1).clamp(min=1e-6)
            return summed / lengths
        raise ValueError(f"Unsupported pooling strategy: {self.config.pooling}")
