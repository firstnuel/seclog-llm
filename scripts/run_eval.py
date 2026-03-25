#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {ait|openssh|bgl}"
  exit 1
fi

STAGE1_ADAPTER="results/checkpoints/stage1_final/final_adapter"
STAGE3_CHECKPOINT="results/checkpoints/stage3_v2/stage3_epoch_1.pt"

case "$1" in
  ait)
    INPUT="data/processed/val_labeled_strat10k_seed42.jsonl"
    OUTPUT="results/ait_val_v2_predictions.jsonl"
    ;;
  openssh)
    INPUT="data/processed/openssh_eval.jsonl"
    OUTPUT="results/openssh_v2_predictions.jsonl"
    ;;
  bgl)
    INPUT="data/processed/ood_bgl.jsonl"
    OUTPUT="results/bgl_ood_v2_predictions.jsonl"
    ;;
  *)
    echo "Unknown target: $1"
    echo "Usage: $0 {ait|openssh|bgl}"
    exit 1
    ;;
esac

python scripts/run_combo_inference.py \
  --stage1-adapter "$STAGE1_ADAPTER" \
  --stage3-checkpoint "$STAGE3_CHECKPOINT" \
  --input "$INPUT" \
  --output "$OUTPUT"

echo "Done: $1 -> $OUTPUT"
