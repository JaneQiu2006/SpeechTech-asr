#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_EXE="${PYTHON_EXE:-python}"
source scripts/deep_dive_common.sh

mkdir -p exp/deep_dive logs/deep_dive results/predictions/deep_dive
RUN_POLICY=(--skip_existing)
[[ "${FOLLOWUP_OVERWRITE:-0}" == 1 ]] && RUN_POLICY=(--overwrite)
RUNTIME_ARGS=()
[[ "${DISABLE_CUDNN:-0}" == 1 ]] && RUNTIME_ARGS=(--disable_cudnn)

# E16B was selected because it is the best cost/performance midpoint and costs
# materially less than repeating both full-fine-tuning systems.
experiment=e25_top5_finetune_seed1234
echo "===== Training $experiment ====="
"$PYTHON_EXE" -u scripts/train_ctc.py \
  --experiment_id E25 \
  --model_name_or_path facebook/wav2vec2-base \
  --train_manifest data/manifests/train_10h.jsonl \
  --eval_manifest data/manifests/dev_clean.jsonl \
  --output_dir "exp/deep_dive/$experiment" \
  --prediction_path "results/predictions/deep_dive/${experiment}_dev.jsonl" \
  --metrics_path results/deep_dive_metrics.csv \
  --freeze_transformer_layers 7 \
  --learning_rate 1e-4 --weight_decay 0.005 --warmup_ratio 0.1 \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_duration_in_seconds 15 \
  --max_steps "${PARTIAL_MAX_STEPS:-2500}" \
  --eval_steps "${PARTIAL_EVAL_STEPS:-300}" \
  --load_best_model_at_end --ctc_zero_infinity --fp16 \
  --gradient_checkpointing --seed 1234 \
  "${RUNTIME_ARGS[@]}" \
  "${RUN_POLICY[@]}" \
  2>&1 | tee -a "logs/deep_dive/${experiment}.log"

evaluate_finetune_experiment E25 "$experiment" "exp/deep_dive/$experiment" 7
