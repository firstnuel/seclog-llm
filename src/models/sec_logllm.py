from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .encoder import EncoderConfig, SecBERTEncoder
from .projector import GatedProjector, ProjectorConfig


DEFAULT_PREAMBLE = (
    "<|im_start|>system\n"
    "You are Sec-LogLLM, a SOC copilot powered by Qwen. "
    "Reason about security logs and determine whether the activity is malicious.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Below is a sequence of enterprise log messages represented as embeddings. "
    "Analyze them carefully before answering."
)
DEFAULT_QUERY = (
    "Respond with JSON containing 'label' (Attack/Normal), "
    "'mitre_t_code' (or null), and a short 'reasoning'.\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@dataclass
class SecLogLLMConfig:
    """
    High-level configuration for the Sec-LogLLM architecture.

    Attributes:
        encoder: Settings for the SecBERT encoder wrapper.
        projector: Settings for the alpha-gated projector.
        decoder_model_name_or_path: Hugging Face identifier for the causal LM backbone.
        decoder_cache_dir: Optional HF cache directory.
        prompt_preamble: Prefix text injected before log embeddings (system + user lead-in).
        prompt_query: Suffix text injected after log embeddings (e.g., user close + assistant start).
        max_logs_per_sequence: Hard cap on number of log lines per sample.
        encoder_batch_size: Optional chunk size when encoding many log lines.
    """

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    decoder_model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct"
    decoder_cache_dir: Optional[str] = None
    prompt_preamble: str = DEFAULT_PREAMBLE
    prompt_query: str = DEFAULT_QUERY
    max_logs_per_sequence: int = 128
    encoder_batch_size: int = 16


@dataclass
class SecLogLLMOutput:
    """Structured outputs returned by SecLogLLM.forward."""

    decoder_outputs: object
    projected_embeddings: torch.Tensor
    alphas: torch.Tensor
    log_mask: torch.Tensor
    prompt_prefixes: List[str]
    prompt_suffixes: List[str]


class SecLogLLM(nn.Module):
    """
    Full Sec-LogLLM pipeline combining SecBERT, the alpha-gated projector,
    and a causal decoder (e.g., Llama). The model can be trained in three
    stages by selectively freezing encoder/projector/decoder parameters.
    """

    def __init__(
        self,
        config: Optional[SecLogLLMConfig] = None,
        *,
        encoder: Optional[SecBERTEncoder] = None,
        projector: Optional[GatedProjector] = None,
        decoder_model: Optional[PreTrainedModel] = None,
        decoder_tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        super().__init__()
        self.config = config or SecLogLLMConfig()

        self.encoder = encoder or SecBERTEncoder(self.config.encoder)
        self.decoder_tokenizer = decoder_tokenizer or AutoTokenizer.from_pretrained(
            self.config.decoder_model_name_or_path,
            cache_dir=self.config.decoder_cache_dir,
            use_fast=True,
        )
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

        self.decoder_model = decoder_model or AutoModelForCausalLM.from_pretrained(
            self.config.decoder_model_name_or_path,
            cache_dir=self.config.decoder_cache_dir,
        )

        decoder_hidden = self.decoder_model.config.hidden_size
        projector_config = self.config.projector
        if not projector_config.output_dim:
            projector_config.output_dim = decoder_hidden
        if projector_config.output_dim != decoder_hidden:
            raise ValueError(
                f"Projector output_dim ({projector_config.output_dim}) "
                f"must match decoder hidden size ({decoder_hidden})."
            )
        self.projector = projector or GatedProjector(projector_config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _decoder_dtype(self) -> torch.dtype:
        return next(self.decoder_model.parameters()).dtype

    def forward(
        self,
        *,
        log_sequences: Optional[Sequence[Sequence[str]]] = None,
        log_embeddings: Optional[torch.Tensor] = None,
        log_mask: Optional[torch.Tensor] = None,
        prompt_texts: Optional[Union[str, Sequence[str]]] = None,
        prompt_suffixes: Optional[Union[str, Sequence[str]]] = None,
        target_texts: Optional[Union[str, Sequence[str]]] = None,
    ) -> SecLogLLMOutput:
        """
        Args:
            log_sequences: Batch of log-line sequences (list of list of str).
            log_embeddings: Pre-computed encoder embeddings of shape (B, T, H_enc).
            log_mask: Binary mask for log_embeddings.
            prompt_texts: Optional per-sample prompt overrides.
            target_texts: Optional per-sample targets (used for teacher forcing).
        """
        if log_embeddings is None or log_mask is None:
            if log_sequences is None:
                raise ValueError("Either (log_embeddings, log_mask) or log_sequences must be provided.")
            log_embeddings, log_mask = self._encode_log_sequences(log_sequences)

        decoder_dtype = self._decoder_dtype()
        projected, alphas = self._project_log_embeddings(log_embeddings, log_mask)
        projected = projected.to(dtype=decoder_dtype)

        prefix_texts, suffix_texts = self._prepare_prompt_chunks(
            prompt_texts,
            prompt_suffixes,
            batch_size=log_embeddings.size(0),
        )
        prefix_embeds, prefix_mask, _ = self._tokenize_decoder_texts(prefix_texts)
        suffix_embeds, suffix_mask, _ = self._tokenize_decoder_texts(suffix_texts)
        prefix_embeds = prefix_embeds.to(dtype=decoder_dtype)
        suffix_embeds = suffix_embeds.to(dtype=decoder_dtype)

        target_embeds, target_mask, target_ids = self._tokenize_target_texts(target_texts, batch_size=log_embeddings.size(0))
        target_embeds = target_embeds.to(dtype=decoder_dtype)

        inputs_embeds = torch.cat([prefix_embeds, projected, suffix_embeds, target_embeds], dim=1)
        inputs_embeds = inputs_embeds.to(dtype=decoder_dtype)
        attention_mask = torch.cat(
            [
                prefix_mask,
                log_mask.to(prefix_mask.dtype),
                suffix_mask,
                target_mask,
            ],
            dim=1,
        )

        labels = None
        if target_ids.size(1) > 0:
            prefix_ignore = torch.full(
                (prefix_embeds.size(0), prefix_embeds.size(1)),
                -100,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            log_ignore = torch.full(
                (log_mask.size(0), log_mask.size(1)),
                -100,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            suffix_ignore = torch.full(
                (suffix_embeds.size(0), suffix_embeds.size(1)),
                -100,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            labels = torch.cat([prefix_ignore, log_ignore, suffix_ignore, target_ids], dim=1)

        decoder_outputs = self.decoder_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return SecLogLLMOutput(
            decoder_outputs=decoder_outputs,
            projected_embeddings=projected,
            alphas=alphas,
            log_mask=log_mask,
            prompt_prefixes=prefix_texts,
            prompt_suffixes=suffix_texts,
        )

    @torch.inference_mode()
    def generate(
        self,
        log_sequences: Sequence[Sequence[str]],
        *,
        prompt_texts: Optional[Union[str, Sequence[str]]] = None,
        prompt_suffixes: Optional[Union[str, Sequence[str]]] = None,
        max_new_tokens: int = 128,
        **generation_kwargs,
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Runs autoregressive generation for a batch of log sequences.

        Returns:
            decoded_texts: Best-effort decoding using the configured tokenizer.
            alphas: Alpha weights for each log line (B, T, 1).
        """
        self.eval()
        decoder_dtype = self._decoder_dtype()
        log_embeddings, log_mask = self._encode_log_sequences(log_sequences)
        projected, alphas = self._project_log_embeddings(log_embeddings, log_mask)
        projected = projected.to(dtype=decoder_dtype)

        prefix_texts, suffix_texts = self._prepare_prompt_chunks(
            prompt_texts,
            prompt_suffixes,
            batch_size=len(log_sequences),
        )
        prefix_embeds, prefix_mask, _ = self._tokenize_decoder_texts(prefix_texts)
        suffix_embeds, suffix_mask, _ = self._tokenize_decoder_texts(suffix_texts)
        prefix_embeds = prefix_embeds.to(dtype=decoder_dtype)
        suffix_embeds = suffix_embeds.to(dtype=decoder_dtype)

        inputs_embeds = torch.cat([prefix_embeds, projected, suffix_embeds], dim=1)
        inputs_embeds = inputs_embeds.to(dtype=decoder_dtype)
        attention_mask = torch.cat(
            [prefix_mask, log_mask.to(prefix_mask.dtype), suffix_mask],
            dim=1,
        )

        generated_ids = self.decoder_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )

        decoded_texts: List[str]
        if hasattr(self.decoder_tokenizer, "batch_decode"):
            decoded_texts = self.decoder_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            decoded_texts = ["<decoder tokenizer unavailable>"] * generated_ids.size(0)

        return decoded_texts, alphas

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_decoder(self) -> None:
        for param in self.decoder_model.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self) -> None:
        for param in self.decoder_model.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _encode_log_sequences(
        self,
        log_sequences: Sequence[Sequence[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode variable-length sequences of raw log strings into a padded tensor.
        """
        device = next(self.decoder_model.parameters()).device
        batch_size = len(log_sequences)
        encoded_sequences: List[torch.Tensor] = []
        lengths: List[int] = []

        for seq in log_sequences:
            trimmed = list(seq)[: self.config.max_logs_per_sequence]
            if not trimmed:
                trimmed = ["<EMPTY>"]

            tokenized = self.encoder.tokenize(trimmed, device=device)
            encoder_outputs = self.encoder(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                token_type_ids=tokenized.get("token_type_ids"),
            )
            encoded_sequences.append(encoder_outputs.pooled_embeddings)
            lengths.append(encoder_outputs.pooled_embeddings.size(0))

        max_len = max(lengths) if lengths else 1
        hidden_size = encoded_sequences[0].size(-1)
        padded = torch.zeros(batch_size, max_len, hidden_size, device=device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)

        for idx, (embeds, length) in enumerate(zip(encoded_sequences, lengths)):
            padded[idx, :length] = embeds
            mask[idx, :length] = 1

        return padded, mask

    def _project_log_embeddings(
        self,
        log_embeddings: torch.Tensor,
        log_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the alpha-gated projector over padded log embeddings.
        """
        batch, seq_len, hidden = log_embeddings.shape
        flattened = log_embeddings.view(batch * seq_len, hidden)
        projected_flat, alpha_flat = self.projector(flattened)

        proj_dim = projected_flat.size(-1)
        projected = projected_flat.view(batch, seq_len, proj_dim)
        alphas = alpha_flat.view(batch, seq_len, 1)

        mask = log_mask.unsqueeze(-1).to(projected.dtype)
        projected = projected * mask
        alphas = alphas * mask

        return projected, alphas

    def _prepare_prompt_chunks(
        self,
        prefix_texts: Optional[Union[str, Sequence[str]]],
        suffix_texts: Optional[Union[str, Sequence[str]]],
        *,
        batch_size: int,
    ) -> Tuple[List[str], List[str]]:
        prefixes = self._standardize_prompt_input(prefix_texts, batch_size, self.config.prompt_preamble)
        suffixes = self._standardize_prompt_input(suffix_texts, batch_size, self.config.prompt_query)
        return prefixes, suffixes

    @staticmethod
    def _standardize_prompt_input(
        value: Optional[Union[str, Sequence[str]]],
        batch_size: int,
        default: str,
    ) -> List[str]:
        if value is None:
            return [default] * batch_size
        if isinstance(value, str):
            return [value] * batch_size
        value_list = list(value)
        if len(value_list) != batch_size:
            raise ValueError(f"Expected {batch_size} prompt entries, got {len(value_list)}")
        return value_list

    def _tokenize_decoder_texts(
        self,
        texts: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenized = self.decoder_tokenizer(
            list(texts),
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        device = next(self.decoder_model.parameters()).device
        dtype = self._decoder_dtype()
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        embeds = self.decoder_model.get_input_embeddings()(input_ids).to(dtype=dtype)
        return embeds, attention_mask, input_ids

    def _tokenize_target_texts(
        self,
        target_texts: Optional[Union[str, Sequence[str]]],
        *,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.decoder_model.parameters()).device
        dtype = self._decoder_dtype()
        if target_texts is None:
            empty_embed = torch.zeros(batch_size, 0, self.decoder_model.config.hidden_size, device=device, dtype=dtype)
            empty_mask = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            empty_ids = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            return empty_embed, empty_mask, empty_ids

        if isinstance(target_texts, str):
            texts = [target_texts] * batch_size
        else:
            texts = list(target_texts)
            if len(texts) != batch_size:
                raise ValueError(f"Expected {batch_size} target texts, got {len(texts)}")

        embeds, attention_mask, input_ids = self._tokenize_decoder_texts(texts)
        return embeds, attention_mask, input_ids
