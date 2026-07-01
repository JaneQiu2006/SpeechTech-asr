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
#
# If LibriSpeech is stored outside the project:
#   DATA_ROOT=/path/to/data bash scripts/train_e2_e4_rtx3090.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_EXE="${PYTHON_EXE:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
START_FROM="${1:-e2}"
CUDNN_ARGS=()

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
    --data_root "$DATA_ROOT" \
    --output_dir "$output_dir" \
    --prediction_path "$prediction_path" \
    --ctc_loss_reduction mean \
    --ctc_zero_infinity \
    --max_duration_in_seconds 15 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --dataloader_num_workers 0 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --weight_decay 0.005 \
    --warmup_ratio 0.1 \
    --num_train_epochs 30 \
    --save_total_limit 2 \
    --eval_accumulation_steps 10 \
    --fp16 \
    --gradient_checkpointing \
    "${CUDNN_ARGS[@]}" \
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
    --data_root "$DATA_ROOT" \
    --output_dir "$output_dir" \
    --prediction_path "$prediction_path" \
    --ctc_loss_reduction mean \
    --ctc_zero_infinity \
    --max_duration_in_seconds 15 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --dataloader_num_workers 0 \
    --gradient_accumulation_steps 8 \
    --max_steps 1000 \
    --eval_steps 100 \
    --learning_rate 1e-4 \
    --weight_decay 0.005 \
    --warmup_ratio 0.1 \
    --save_total_limit 2 \
    --eval_accumulation_steps 10 \
    --fp16 \
    --gradient_checkpointing \
    "${CUDNN_ARGS[@]}" \
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
    --data_root "$DATA_ROOT" \
    --output_dir "$output_dir" \
    --prediction_path "$prediction_path" \
    --ctc_loss_reduction mean \
    --ctc_zero_infinity \
    --max_duration_in_seconds 15 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --dataloader_num_workers 0 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --weight_decay 0.005 \
    --warmup_ratio 0.1 \
    --num_train_epochs 30 \
    --save_total_limit 2 \
    --eval_accumulation_steps 10 \
    --fp16 \
    --gradient_checkpointing \
    "${CUDNN_ARGS[@]}" \
    2>&1 | tee logs/wavlm_finetune_10h_rtx3090.log
}

require_file scripts/train_ctc.py
require_file data/manifests/train_1h.jsonl
require_file data/manifests/train_10h.jsonl
require_file data/manifests/dev_clean.jsonl
require_file data/vocab/vocab.json

"$PYTHON_EXE" -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.cuda.get_device_name(0)); print('torch=', torch.__version__, 'cudnn=', torch.backends.cudnn.version())"

if "$PYTHON_EXE" -c \
  "import torch; x=torch.zeros((2, 1, 240000), device='cuda', dtype=torch.float16); conv=torch.nn.Conv1d(1, 512, 10, 5, bias=False).cuda().half(); y=conv(x); print('cuDNN FP16 Conv1d preflight OK:', tuple(y.shape))" \
  2>logs/cudnn_preflight_error.log; then
  :
else
  echo "cuDNN Conv1d initialization failed; details: logs/cudnn_preflight_error.log" >&2
  echo "Retrying with native CUDA kernels." >&2
  CUDNN_ARGS=(--disable_cudnn)
  "$PYTHON_EXE" -c \
    "import torch; torch.backends.cudnn.enabled=False; x=torch.zeros((2, 1, 240000), device='cuda', dtype=torch.float16); conv=torch.nn.Conv1d(1, 512, 10, 5, bias=False).cuda().half(); y=conv(x); print('Native CUDA FP16 Conv1d fallback OK:', tuple(y.shape))"
fi

for experiment in "${EXPERIMENTS[@]}"; do
  echo "===== Starting ${experiment^^} ====="
  "run_${experiment}"
  echo "===== Completed ${experiment^^} ====="
done

echo "All requested experiments completed."
