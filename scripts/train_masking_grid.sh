#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p exp/deep_dive logs/deep_dive results/predictions/deep_dive
PYTHON_EXE="${PYTHON_EXE:-python}"
source scripts/deep_dive_common.sh
for spec in "e22a:0.02:0.00" "e22b:0.02:0.02"; do
  IFS=: read -r id time_prob feature_prob <<<"$spec"
  experiment="${id}_mask_t${time_prob}_f${feature_prob}_1h"
  "$PYTHON_EXE" -u scripts/train_ctc.py \
    --experiment_id "${id^^}" --model_name_or_path facebook/wav2vec2-base \
    --train_manifest data/manifests/train_1h_effective_15s.jsonl \
    --eval_manifest data/manifests/dev_clean.jsonl \
    --output_dir "exp/deep_dive/$experiment" \
    --prediction_path "results/predictions/deep_dive/${experiment}_dev.jsonl" \
    --metrics_path results/deep_dive_metrics.csv --mask_time_prob "$time_prob" \
    --mask_feature_prob "$feature_prob" --learning_rate 1e-4 \
    --max_steps 1500 --eval_steps 100 --collapse_patience_evaluations 1 \
    --collapse_non_empty_threshold 0.01 --ctc_zero_infinity --fp16 \
    --gradient_checkpointing --skip_existing \
    2>&1 | tee -a "logs/deep_dive/${experiment}.log"
  evaluate_finetune_experiment "${id^^}" "$experiment" \
    "exp/deep_dive/$experiment" 0
done
