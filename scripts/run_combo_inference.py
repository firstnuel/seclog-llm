"""Batch inference helper for Sec-LogLLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import torch
from peft import PeftModel
from torch.serialization import safe_globals
from tqdm import tqdm
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.models.encoder import EncoderConfig
from src.models.projector import ProjectorConfig
from src.models.sec_logllm import SecLogLLM, SecLogLLMConfig


STRICT_JSON_SUFFIX = (
    "Respond with a single JSON object only. Do not output markdown, lists, or extra text. "
    "Use keys exactly: label, mitre_t_code, reasoning. "
    "label must be Attack or Normal; mitre_t_code can be null.\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def consume_jsonl(path: Path, *, max_samples: int | None = None) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_samples is not None and idx >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def log_sequences_from_record(record: dict) -> Sequence[Sequence[str]]:
    if "log_sequence" in record and isinstance(record["log_sequence"], list):
        return [record["log_sequence"]]
    raw = record.get("raw") or record.get("message")
    if raw:
        return [[str(raw)]]
    return [[""]]


def parse_response(response: str) -> tuple[dict, bool, str | None]:
    trimmed = response.strip()
    start = trimmed.find("{")
    if start >= 0:
        trimmed = trimmed[start:]

    end = trimmed.rfind("}")
    if end >= 0:
        trimmed = trimmed[: end + 1]

    try:
        payload = json.loads(trimmed)
        if isinstance(payload, dict):
            return payload, True, None
        return {"raw_response": response}, False, "response_json_is_not_object"
    except json.JSONDecodeError as exc:
        return {"raw_response": response}, False, f"json_decode_error: {exc.msg}"


def load_sec_logllm(stage1_adapter: Path, stage3_checkpoint: Path) -> SecLogLLM:
    config = ExperimentConfig()
    model = SecLogLLM(config.sec_logllm)

    base_decoder = AutoModelForCausalLM.from_pretrained(
        config.sec_logllm.decoder_model_name_or_path,
        trust_remote_code=True,
    )
    peft_decoder = PeftModel.from_pretrained(base_decoder, stage1_adapter)
    peft_decoder.eval()
    model.decoder_model = peft_decoder

    generation_config = model.decoder_model.generation_config
    generation_config.do_sample = False
    generation_config.temperature = 1.0
    generation_config.top_p = 1.0
    generation_config.top_k = 50

    decoder_dtype = next(model.decoder_model.parameters()).dtype
    model.encoder.to(dtype=decoder_dtype)
    model.projector.to(dtype=decoder_dtype)

    with safe_globals([SecLogLLMConfig, EncoderConfig, ProjectorConfig]):
        checkpoint = torch.load(stage3_checkpoint, map_location="cpu")
    model.encoder.load_state_dict(checkpoint["encoder_state"])
    model.projector.load_state_dict(checkpoint["projector_state"])

    # Load Stage 3 decoder LoRA weights (co-trained with encoder+projector)
    if "decoder_lora_state" in checkpoint:
        missing, unexpected = model.decoder_model.load_state_dict(
            checkpoint["decoder_lora_state"], strict=False
        )
        print(f"Loaded Stage 3 decoder LoRA: {len(checkpoint['decoder_lora_state'])} keys")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    else:
        print("WARNING: No decoder_lora_state in checkpoint, using Stage 1 LoRA only!")

    model.encoder.to(dtype=decoder_dtype)
    model.projector.to(dtype=decoder_dtype)

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sec-LogLLM on a JSONL dataset.")
    parser.add_argument("--stage1-adapter", type=Path, required=True, help="Path to Stage 1 LoRA adapter.")
    parser.add_argument("--stage3-checkpoint", type=Path, required=True, help="Stage 3 encoder+projector checkpoint (.pt).")
    parser.add_argument("--input", type=Path, required=True, help="JSONL file with logs.")
    parser.add_argument("--output", type=Path, default=Path("results/openssh_predictions.jsonl"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling; default is deterministic decoding.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    device = select_device()
    model = load_sec_logllm(args.stage1_adapter, args.stage3_checkpoint)
    model.to(device)

    predictions: list[dict] = []
    for record in tqdm(consume_jsonl(args.input, max_samples=args.max_samples), desc="Predicting"):
        sequences = log_sequences_from_record(record)
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "prompt_suffixes": STRICT_JSON_SUFFIX,
            "do_sample": args.do_sample,
        }
        if args.do_sample:
            generation_kwargs["temperature"] = args.temperature

        texts, alphas = model.generate(
            sequences,
            **generation_kwargs,
        )

        parsed, parse_ok, parse_error = parse_response(texts[0])
        prediction = {
            "raw": record.get("raw") or record.get("message"),
            "heuristic_label": record.get("heuristic_label", record.get("label")),
            "label": parsed.get("label"),
            "mitre_t_code": parsed.get("mitre_t_code"),
            "reasoning": parsed.get("reasoning"),
            "parse_ok": parse_ok,
            "parse_error": parse_error,
            "alpha": alphas[0].tolist(),
            "response_text": texts[0],
        }
        predictions.append(prediction)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction) + "\n")

    print(f"Wrote {len(predictions)} records to {args.output}")


if __name__ == "__main__":
    main()
