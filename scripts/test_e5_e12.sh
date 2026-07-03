#!/usr/bin/env bash
set -euo pipefail

# Evaluate all retained E5-E12 experiments on full LibriSpeech test-clean.
# E7 is intentionally skipped. Existing complete metrics are reused.
#
# Usage:
#   bash scripts/test_e5_e12.sh
#   bash scripts/test_e5_e12.sh --batch-size 1
#   bash scripts/test_e5_e12.sh --overwrite

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_EXE="${PYTHON_EXE:-python}"
BATCH_SIZE=2
OVERWRITE=0
NO_FP16=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --batch-size)
      [[ "$#" -ge 2 ]] || {
        echo "--batch-size requires an integer." >&2
        exit 2
      }
      BATCH_SIZE="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --no-fp16)
      NO_FP16=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "--batch-size must be a positive integer." >&2
  exit 2
}

for required in \
  scripts/evaluate_e1_e5.py \
  scripts/evaluate_representation_ctc.py \
  data/manifests/test_clean.jsonl \
  data/vocab/vocab.json; do
  [[ -f "$required" ]] || {
    echo "Required file not found: $required" >&2
    exit 1
  }
done

"$PYTHON_EXE" -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"

CUDNN_ARGS=()
PREFLIGHT_LOG="logs/e5_e12_eval_cudnn_preflight.tmp"
mkdir -p logs results/predictions
if ! "$PYTHON_EXE" -c \
  "import torch; x=torch.zeros((1,1,240000),device='cuda',dtype=torch.float16); c=torch.nn.Conv1d(1,512,10,5,bias=False).cuda().half(); c(x); print('cuDNN preflight passed')" \
  2>"$PREFLIGHT_LOG"; then
  echo "cuDNN Conv1d preflight failed; using native CUDA kernels." >&2
  CUDNN_ARGS=(--disable_cudnn)
fi
rm -f "$PREFLIGHT_LOG"

COMMON_ARGS=(--batch_size "$BATCH_SIZE")
[[ "$OVERWRITE" -eq 1 ]] && COMMON_ARGS+=(--overwrite)
[[ "$NO_FP16" -eq 1 ]] && COMMON_ARGS+=(--no_fp16)
COMMON_ARGS+=("${CUDNN_ARGS[@]}")

require_model() {
  local directory="$1"
  [[ -d "$directory" ]] || {
    echo "Required model directory not found: $directory" >&2
    exit 1
  }
}

echo "===== Standard CTC experiments: E5, E6a, E6b, E8, E9, E12a ====="
for directory in \
  exp/wav2vec2_finetune_3h \
  exp/wav2vec2_top6_finetune_10h \
  exp/wav2vec2_top3_finetune_10h \
  exp/wav2vec2_masking_finetune_10h \
  exp/wav2vec2_frozen_bilstm_10h \
  exp/wav2vec2_finetune_1h_time_mask; do
  require_model "$directory"
done

"$PYTHON_EXE" -u scripts/evaluate_e1_e5.py \
  --experiments E5 E6A E6B E8 E9 E12A \
  "${COMMON_ARGS[@]}"

echo "===== E7 intentionally skipped ====="

echo "===== Continuous layer experiments: E10a, E10b, E10c ====="
for layer in 6 9 12; do
  experiment="wav2vec2_layer${layer}_bilstm_ctc"
  model_dir="exp/$experiment"
  require_model "$model_dir"
  representation_args=(
    --experiment "$experiment"
    --model_dir "$model_dir"
    --prediction_path "results/predictions/${experiment}_test.jsonl"
    --batch_size "$BATCH_SIZE"
  )
  [[ "$OVERWRITE" -eq 1 ]] && representation_args+=(--overwrite)
  [[ "$NO_FP16" -eq 1 ]] && representation_args+=(--no_fp16)
  representation_args+=("${CUDNN_ARGS[@]}")
  "$PYTHON_EXE" -u scripts/evaluate_representation_ctc.py \
    "${representation_args[@]}"
done

BEST_LAYER_FILE="artifacts/kmeans/best_layer.txt"
[[ -f "$BEST_LAYER_FILE" ]] || {
  echo "Missing best-layer selection: $BEST_LAYER_FILE" >&2
  exit 1
}
BEST_LAYER="$(tr -d '[:space:]' < "$BEST_LAYER_FILE")"
case "$BEST_LAYER" in
  6|9|12) ;;
  *)
    echo "Invalid best layer: '$BEST_LAYER'" >&2
    exit 1
    ;;
esac

echo "===== Discrete-unit experiments: E11a, E11b, E11c (layer $BEST_LAYER) ====="
for codebook_size in 50 100 200; do
  experiment="wav2vec2_discrete_k${codebook_size}_bilstm_ctc"
  model_dir="exp/$experiment"
  centers="artifacts/kmeans/wav2vec2_layer${BEST_LAYER}_k${codebook_size}/centers.npy"
  require_model "$model_dir"
  [[ -f "$centers" ]] || {
    echo "Codebook centers not found: $centers" >&2
    exit 1
  }
  representation_args=(
    --experiment "$experiment"
    --model_dir "$model_dir"
    --centers "$centers"
    --prediction_path "results/predictions/${experiment}_test.jsonl"
    --batch_size "$BATCH_SIZE"
  )
  [[ "$OVERWRITE" -eq 1 ]] && representation_args+=(--overwrite)
  [[ "$NO_FP16" -eq 1 ]] && representation_args+=(--no_fp16)
  representation_args+=("${CUDNN_ARGS[@]}")
  "$PYTHON_EXE" -u scripts/evaluate_representation_ctc.py \
    "${representation_args[@]}"
done

echo "===== E5-E12 test-clean evaluation complete ====="
echo "Standard metrics:       results/test_metrics.csv"
echo "Representation metrics: results/representation_test_metrics.csv"
