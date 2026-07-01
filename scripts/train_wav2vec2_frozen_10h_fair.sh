#!/usr/bin/env bash
set -euo pipefail

# Fair E1 vs E2 ablation:
#   - same data, optimizer schedule, effective batch size, and evaluation policy
#   - only the encoder-freezing strategy changes
#   - the checkpoint with the lowest dev WER is reloaded before final prediction
#
# Default: full 30-epoch controlled comparison.
# Diagnostic 10-epoch run:
#   EPOCHS=10 bash scripts/train_wav2vec2_frozen_10h_fair.sh
#
# Optional overrides:
#   PYTHON_EXE=/path/to/python
#   DATA_ROOT=/path/to/data
#   EPOCHS=30

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_EXE="${PYTHON_EXE:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
EPOCHS="${EPOCHS:-30}"

OUTPUT_DIR="exp/wav2vec2_frozen_10h_fair_${EPOCHS}ep"
PREDICTION_PATH="results/predictions/wav2vec2_frozen_10h_fair_${EPOCHS}ep_dev.jsonl"
LOG_PATH="logs/wav2vec2_frozen_10h_fair_${EPOCHS}ep.log"
CUDNN_ARGS=()

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "$EPOCHS" in
  ''|*[!0-9]*)
    echo "EPOCHS must be a positive integer, got: $EPOCHS" >&2
    exit 2
    ;;
  0)
    echo "EPOCHS must be greater than zero" >&2
    exit 2
    ;;
esac

for required_file in \
  scripts/train_ctc.py \
  data/manifests/train_10h.jsonl \
  data/manifests/dev_clean.jsonl \
  data/vocab/vocab.json; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required file not found: $required_file" >&2
    exit 1
  fi
done

if [[ -e "$OUTPUT_DIR" || -e "$PREDICTION_PATH" || -e "$LOG_PATH" ]]; then
  echo "Refusing to overwrite an existing fair-E1 artifact:" >&2
  echo "  $OUTPUT_DIR" >&2
  echo "  $PREDICTION_PATH" >&2
  echo "  $LOG_PATH" >&2
  exit 1
fi

mkdir -p exp results/predictions logs

"$PYTHON_EXE" -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.cuda.get_device_name(0)); print('torch=', torch.__version__, 'cudnn=', torch.backends.cudnn.version())"

if "$PYTHON_EXE" -c \
  "import torch; x=torch.zeros((2, 1, 240000), device='cuda', dtype=torch.float16); conv=torch.nn.Conv1d(1, 512, 10, 5, bias=False).cuda().half(); y=conv(x); print('cuDNN FP16 Conv1d preflight OK:', tuple(y.shape))" \
  2>"logs/e1_fair_${EPOCHS}ep_cudnn_preflight_error.log"; then
  :
else
  echo "cuDNN Conv1d initialization failed; using native CUDA kernels." >&2
  CUDNN_ARGS=(--disable_cudnn)
  "$PYTHON_EXE" -c \
    "import torch; torch.backends.cudnn.enabled=False; x=torch.zeros((2, 1, 240000), device='cuda', dtype=torch.float16); conv=torch.nn.Conv1d(1, 512, 10, 5, bias=False).cuda().half(); y=conv(x); print('Native CUDA FP16 Conv1d fallback OK:', tuple(y.shape))"
fi

"$PYTHON_EXE" -u scripts/train_ctc.py \
  --model_name_or_path facebook/wav2vec2-base \
  --train_manifest data/manifests/train_10h.jsonl \
  --eval_manifest data/manifests/dev_clean.jsonl \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --prediction_path "$PREDICTION_PATH" \
  --freeze_encoder \
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
  --num_train_epochs "$EPOCHS" \
  --save_total_limit 2 \
  --eval_accumulation_steps 10 \
  --seed 42 \
  --fp16 \
  --load_best_model_at_end \
  "${CUDNN_ARGS[@]}" \
  2>&1 | tee "$LOG_PATH"

echo "Fair E1 completed."
echo "  epochs: $EPOCHS"
echo "  output: $OUTPUT_DIR"
echo "  prediction: $PREDICTION_PATH"
echo "  log: $LOG_PATH"
