# Sec-LogLLM (Thesis Codebase)

Sec-LogLLM is a hybrid security log analysis pipeline that combines:
- a log encoder (SecBERT),
- an alpha-gated projector for log importance weighting,
- and a decoder LLM (Qwen + LoRA) for structured JSON predictions.

The repository supports three-stage training, batch inference, and baseline/metric utilities for evaluation.

## 1. Repository Layout

- `src/models/`
  - `encoder.py`: SecBERT wrapper and pooling logic.
  - `projector.py`: alpha-gated projector (`projection + gate -> alpha`).
  - `sec_logllm.py`: full model assembly, training forward pass, and generation.
- `src/training/`
  - `stage1_sft.py`: LoRA SFT on labeled logs.
  - `stage2_alignment.py`: freeze decoder, train encoder+projector.
  - `stage3_e2e.py`: train projector + LoRA jointly (end-to-end).
  - `trainer.py`: reusable multi-stage trainer abstraction.
  - `custom_loss.py`: sparse alignment loss helper.
- `src/utils/`
  - `data_loader.py`: dataset/collator for JSONL/CSV log data.
  - `llm_labeler.py`: OpenAI-powered label generation utility.
  - `ait_v1_parser.py`: parser for AIT-LDS v1.1 corpus.
  - `openssh_parser.py`: parser/feature extractor for OpenSSH logs.
  - `make_val_ground_truth_jsonl.py`: convert validation CSV to JSONL labels.
- `src/evaluation/`
  - `metrics.py`: classification/fidelity/hallucination metrics.
  - `baselines.py`: TF-IDF + RandomForest baseline.
- `scripts/`
  - `run_combo_inference.py`: batch inference over a JSONL file.
  - `run_eval_v2.sh`: one-at-a-time eval helper (`ait|openssh|bgl`).
  - `diagnose_pipeline.py`: targeted pipeline diagnostics.
- `data/processed/`: processed train/val/eval JSONL datasets.
- `results/`: checkpoints and generated predictions.

## 2. Environment Setup

### Requirements

Python 3.10+ recommended, CUDA-capable GPU recommended for training/inference.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Data Format Expectations

Most training/inference paths expect JSONL records with at least one of:
- `log_sequence` (list of log lines), or
- `raw` / `message` (single log line fallback).

Common fields used by training/evaluation:
- `label`: `Attack` or `Normal`
- `mitre_t_code`: MITRE code or null
- `reasoning`: short explanation
- `raw`: raw log text

## 4. Architecture Summary

Sec-LogLLM flow:
1. Logs are tokenized and encoded by SecBERT (`encoder.py`).
2. Per-log embeddings are projected to decoder hidden size and gated by alpha (`projector.py`).
3. Prompt prefix + projected log embeddings + prompt suffix are fed as `inputs_embeds` into Qwen (`sec_logllm.py`).
4. Training uses decoder loss; stage 2 optionally adds alpha sparsity regularization.

## 5. Training Workflow

### Stage 1: Decoder LoRA SFT

```bash
python -m src.training.stage1_sft \
  --train-path data/processed/train_labeled.jsonl \
  --output-dir results/checkpoints/stage1_final
```

Key output: `results/checkpoints/stage1_final/final_adapter/`

### Stage 2: Projector Alignment (decoder frozen)

```bash
python -m src.training.stage2_alignment \
  --stage1-adapter results/checkpoints/stage1_final/final_adapter \
  --output-dir results/checkpoints/stage2_v2 \
  --epochs 2 \
  --lambda-l1 0.001
```

Key output example: `results/checkpoints/stage2_v2/stage2_epoch_*.pt` or `stage2_checkpoint.pt`

### Stage 3: End-to-End (projector + LoRA)

```bash
python -m src.training.stage3_e2e \
  --stage1-adapter results/checkpoints/stage1_final/final_adapter \
  --stage2-checkpoint results/checkpoints/stage2_v2/stage2_checkpoint.pt \
  --output-dir results/checkpoints/stage3_v2 \
  --epochs 1
```

Key output example: `results/checkpoints/stage3_v2/stage3_epoch_1.pt`

## 6. Inference

### Single dataset run

```bash
python scripts/run_combo_inference.py \
  --stage1-adapter results/checkpoints/stage1_final/final_adapter \
  --stage3-checkpoint results/checkpoints/stage3_v2/stage3_epoch_1.pt \
  --input data/processed/openssh_eval.jsonl \
  --output results/openssh_v2_predictions.jsonl
```

### One-at-a-time helper

Use the executable helper script for standard v2 eval targets:

```bash
./scripts/run_eval_v2.sh ait
./scripts/run_eval_v2.sh openssh
./scripts/run_eval_v2.sh bgl
```

## 7. Prediction Output Format

`run_combo_inference.py` writes JSONL with fields like:
- `raw`
- `heuristic_label`
- `label`
- `mitre_t_code`
- `reasoning`
- `parse_ok`
- `parse_error`
- `alpha`
- `response_text`

This allows both classification evaluation and generation quality/debug checks.

## 8. Useful Utilities

- Build validation JSONL from AIT ground truth:

```bash
python -m src.utils.make_val_ground_truth_jsonl \
  --input data/processed/ait_val_subset.csv \
  --output data/processed/val_labeled.jsonl
```

- Parse OpenSSH logs into feature-rich dataframe (from Python):

```python
from src.utils.openssh_parser import OpenSSHParser
parser = OpenSSHParser("data/openssh")
df = parser.parse_file()
```

- LLM labeling pipeline requires `OPENAI_API_KEY`:

```bash
export OPENAI_API_KEY="..."
```

## 9. Large Files and Git Hygiene

This project can generate very large artifacts (checkpoints, optimizer states, model weights, prediction dumps).
Recommended practice:
- Keep large checkpoint artifacts under `results/checkpoints/` and out of normal Git tracking.
- Track only lightweight code/config and essential small evaluation data.
- If large binary versioning is required, use Git LFS.

## 10. Troubleshooting Notes

- If `git push` reports `src refspec main does not match any`:
  - create an initial commit first,
  - ensure current branch exists (`main` or your active branch).
- If model loading shows `UNEXPECTED` keys for MLM heads:
  - often expected when loading weights across different task heads/architectures.
- If generation JSON parse fails (`parse_ok=false`):
  - inspect `response_text` to tune prompt suffix and decoding settings.

## 11. Quick Start Checklist

1. Create and activate virtual environment.
2. Install requirements.
3. Ensure Stage 1/2/3 checkpoints exist.
4. Run one inference command or `run_eval_v2.sh`.
5. Evaluate prediction JSONL outputs.
