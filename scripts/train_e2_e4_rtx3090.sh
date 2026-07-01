#!/usr/bin/env bash
set -euo pipefail

# Sequential RTX 3090 commands for E2, E3, and E4.
#
# Usage:
#   conda activate ssl_asr
#   bash scripts/train_e2_e4_rtx3090.sh
#
# Resume at a later experiment after inspecting earlier output:
#   bash scripts/train_e2_e4_rtx3090.sh e3
#   bash scripts/train_e2_e4_rtx3090.sh e4
#
# Override Python if needed:
#   PYTHON_EXE=/path/to/python bash scripts/train_e2_e4_rtx3090.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_EXE="${PYTHON_EXE:-python}"
START_FROM="${1:-e2}"

case "$START_FROM" in
  e2) EXPERIMENTS=(e2 e3 e4) ;;
  e3) EXPERIMENTS=(e3 e4) ;;
  e4) EXPERIMENTS=(e4) ;;
  *)
    echo "Usage: $0 [e2|e3|e4]" >&2
    exit 2
    ;;
esac

mkdir -p logs results/predictions exp

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 1
  fi
}

require_new_output() {
  local output_dir="$1"
  local prediction_path="$2"
  if [[ -e "$output_dir" || -e "$prediction_path" ]]; then
    echo "Refusing to overwrite existing E2-E4 output:" >&2
    echo "  $output_dir" >&2
    echo "  $prediction_path" >&2
    echo "Inspect or archive it, then rerun from the required experiment." >&2
    exit 1
  fi
}

run_e2() {
  local output_dir="exp/wav2vec2_finetune_10h"
  local prediction_path="results/predictions/wav2vec2_finetune_10h_dev.jsonl"
  require_new_output "$output_dir" "$prediction_path"

  "$PYTHON_EXE" -u scripts/train_ctc.py \
    --model_name_or_path facebook/wav2vec2-base \
    --train_manifest data/manifests/train_10h.jsonl \
    --eval_manifest data/manifests/dev_clean.jsonl \
    --output_dir "$output_dir" \
    --prediction_path "$prediction_path" \
    --ctc_loss_reduction mean \
    --ctc_zero_infinity \
    --max_duration_in_seconds 15 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --weight_decay 0.005 \
    --warmup_ratio 0.1 \
    --num_train_epochs 30 \
    --save_total_limit 2 \
    --eval_accumulation_steps 10 \
    --fp16 \
    --gradient_checkpointing \
    2>&1 | tee logs/wav2vec2_finetune_10h_rtx3090.log
}

run_e3() {
  local output_dir="exp/wav2vec2_finetune_1h"
  local prediction_path="results/predictions/wav2vec2_finetune_1h_dev.jsonl"
  require_new_output "$output_dir" "$prediction_path"

  "$PYTHON_EXE" -u scripts/train_ctc.py \
    --model_name_or_path facebook/wav2vec2-base \
    --train_manifest data/manifests/train_1h.jsonl \
    --eval_manifest data/manifests/dev_clean.jsonl \
    --output_dir "$output_dir" \
    --prediction_path "$prediction_path" \
    --ctc_loss_reduction mean \
    --ctc_zero_infinity \
    --max_duration_in_seconds 15 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 4 \
    --max_steps 1000 \
    --eval_steps 100 \
    --learning_rate 1e-4 \
    --weight_decay 0.005 \
    --warmup_ratio 0.1 \
    --save_total_limit 2 \
    --eval_accumulation_steps 10 \
    --fp16 \
    --gradient_checkpointing \
    2>&1 | tee logs/wav2vec2_finetune_1h_rtx3090.log
}

run_e4() {
  local output_dir="exp/wavlm_finetune_10h"
  local prediction_path="results/predictions/wavlm_finetune_10h_dev.jsonl"
  require_new_output "$output_dir" "$prediction_path"

  "$PYTHON_EXE" -u scripts/train_ctc.py \
    --model_name_or_path microsoft/wavlm-base \
    --train_manifest data/manifests/train_10h.jsonl \
    --eval_manifest data/manifests/dev_clean.jsonl \
    --output_dir "$output_dir" \
    --prediction_path "$prediction_path" \
    --ctc_loss_reduction mean \
    --ctc_zero_infinity \
    --max_duration_in_seconds 15 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --weight_decay 0.005 \
    --warmup_ratio 0.1 \
    --num_train_epochs 30 \
    --save_total_limit 2 \
    --eval_accumulation_steps 10 \
    --fp16 \
    --gradient_checkpointing \
    2>&1 | tee logs/wavlm_finetune_10h_rtx3090.log
}

require_file scripts/train_ctc.py
require_file data/manifests/train_1h.jsonl
require_file data/manifests/train_10h.jsonl
require_file data/manifests/dev_clean.jsonl
require_file data/vocab/vocab.json

"$PYTHON_EXE" -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.cuda.get_device_name(0))"

for experiment in "${EXPERIMENTS[@]}"; do
  echo "===== Starting ${experiment^^} ====="
  "run_${experiment}"
  echo "===== Completed ${experiment^^} ====="
done

echo "All requested experiments completed."
