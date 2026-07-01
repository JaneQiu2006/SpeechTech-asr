#!/usr/bin/env bash
set -euo pipefail

# GPU experiment queue from doc/ssl_asr_remaining_gpu_training_plan.md.
# With no arguments, run fair E1-30, repaired E3, then the new experiments:
# E1-30, E3r, E5, E6a, E6b, E7, E8, E9.
# Examples:
#   bash scripts/train_new_experiments_rtx3090.sh
#   bash scripts/train_new_experiments_rtx3090.sh e5
#   bash scripts/train_new_experiments_rtx3090.sh e7 e8
#   bash scripts/train_new_experiments_rtx3090.sh e3r
#   bash scripts/train_new_experiments_rtx3090.sh e1-30

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_EXE="${PYTHON_EXE:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
CUDNN_ARGS=()
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "$#" -eq 0 ]]; then
  EXPERIMENTS=(e1-30 e3r e5 e6a e6b e7 e8 e9)
else
  EXPERIMENTS=("$@")
fi

for experiment in "${EXPERIMENTS[@]}"; do
  case "$experiment" in
    e1-30|e3r|e5|e6a|e6b|e7|e8|e9) ;;
    *)
      echo "Unknown experiment '$experiment'; use e1-30, e3r, e5, e6a, e6b, e7, e8, or e9." >&2
      exit 2
      ;;
  esac
done

case " ${EXPERIMENTS[*]} " in
  *" e5 "*)
    [[ -f data/manifests/train_3h_effective_15s.jsonl ]] || {
      echo "Missing data/manifests/train_3h_effective_15s.jsonl." >&2
      echo "Generate it with the command documented in README.md." >&2
      exit 1
    }
    ;;
esac

case " ${EXPERIMENTS[*]} " in
  *" e3r "*)
    [[ -f data/manifests/train_1h_effective_15s.jsonl ]] || {
      echo "Missing data/manifests/train_1h_effective_15s.jsonl." >&2
      echo "Generate it with the command documented in README.md." >&2
      exit 1
    }
    ;;
esac

for required_file in \
  scripts/train_ctc.py \
  data/manifests/train_10h.jsonl \
  data/manifests/dev_clean.jsonl \
  data/vocab/vocab.json; do
  [[ -f "$required_file" ]] || {
    echo "Required file not found: $required_file" >&2
    exit 1
  }
done

mkdir -p exp logs results/predictions

"$PYTHON_EXE" -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.cuda.get_device_name(0)); print('torch=', torch.__version__, 'cudnn=', torch.backends.cudnn.version())"

if ! "$PYTHON_EXE" -c \
  "import torch; x=torch.zeros((2, 1, 240000), device='cuda', dtype=torch.float16); conv=torch.nn.Conv1d(1, 512, 10, 5, bias=False).cuda().half(); print(tuple(conv(x).shape))" \
  2>logs/new_experiments_cudnn_preflight_error.log; then
  echo "cuDNN Conv1d preflight failed; using native CUDA kernels." >&2
  CUDNN_ARGS=(--disable_cudnn)
fi

run_experiment() {
  local experiment="$1"
  local output_dir
  local prediction_path
  local log_path
  local -a experiment_args

  case "$experiment" in
    e1-30)
      output_dir="exp/wav2vec2_frozen_10h_fair_30ep"
      experiment_args=(
        --model_name_or_path facebook/wav2vec2-base
        --train_manifest data/manifests/train_10h.jsonl
        --freeze_encoder
        --num_train_epochs 30
        --learning_rate 1e-4
        --warmup_ratio 0.1
        --save_total_limit 2
        --fp16
      )
      ;;
    e5)
      output_dir="exp/wav2vec2_finetune_3h"
      experiment_args=(
        --model_name_or_path facebook/wav2vec2-base
        --train_manifest data/manifests/train_3h_effective_15s.jsonl
        --max_steps 2000 --eval_steps 200
        --learning_rate 7e-5 --warmup_ratio 0.15
        --save_total_limit 5
        --collapse_patience_evaluations 5
        --fp16
        --gradient_checkpointing
      )
      ;;
    e6a)
      output_dir="exp/wav2vec2_top6_finetune_10h"
      experiment_args=(
        --model_name_or_path facebook/wav2vec2-base
        --train_manifest data/manifests/train_10h.jsonl
        --freeze_transformer_layers 6
        --max_steps 2500 --eval_steps 300
        --learning_rate 1e-4 --warmup_ratio 0.1
        --save_total_limit 3
        --collapse_patience_evaluations 4
        --fp16
        --gradient_checkpointing
      )
      ;;
    e6b)
      output_dir="exp/wav2vec2_top3_finetune_10h"
      experiment_args=(
        --model_name_or_path facebook/wav2vec2-base
        --train_manifest data/manifests/train_10h.jsonl
        --freeze_transformer_layers 9
        --max_steps 2200 --eval_steps 300
        --learning_rate 1e-4 --warmup_ratio 0.1
        --save_total_limit 3
        --collapse_patience_evaluations 4
        --fp16
        --gradient_checkpointing
      )
      ;;
    e7)
      output_dir="exp/hubert_finetune_10h"
      experiment_args=(
        --model_name_or_path facebook/hubert-base-ls960
        --train_manifest data/manifests/train_10h.jsonl
        --max_steps 3780 --eval_steps 500
        --learning_rate 1e-4 --warmup_ratio 0.1
        --save_total_limit 2
        --collapse_patience_evaluations 2
        --fp16
        --gradient_checkpointing
      )
      ;;
    e8)
      output_dir="exp/wav2vec2_masking_finetune_10h"
      experiment_args=(
        --model_name_or_path facebook/wav2vec2-base
        --train_manifest data/manifests/train_10h.jsonl
        --max_steps 3780 --eval_steps 500
        --learning_rate 1e-4 --warmup_ratio 0.1
        --mask_time_prob 0.05 --mask_feature_prob 0.05
        --save_total_limit 2
        --collapse_patience_evaluations 2
        --fp16
        --gradient_checkpointing
      )
      ;;
    e9)
      output_dir="exp/wav2vec2_frozen_bilstm_10h"
      experiment_args=(
        --model_name_or_path facebook/wav2vec2-base
        --train_manifest data/manifests/train_10h.jsonl
        --freeze_encoder
        --ctc_head_type bilstm
        --ctc_head_hidden_size 256
        --ctc_head_num_layers 2
        --ctc_head_dropout 0.1
        --per_device_train_batch_size 1
        --per_device_eval_batch_size 1
        --gradient_accumulation_steps 16
        --max_steps 3000 --eval_steps 300
        --learning_rate 1e-3 --warmup_ratio 0.1
        --save_total_limit 3
        --collapse_patience_evaluations 4
      )
      ;;
    e3r)
      output_dir="exp/wav2vec2_finetune_1h_repaired"
      experiment_args=(
        --model_name_or_path facebook/wav2vec2-base
        --train_manifest data/manifests/train_1h_effective_15s.jsonl
        --per_device_train_batch_size 2
        --per_device_eval_batch_size 2
        --gradient_accumulation_steps 4
        --max_steps 1500 --eval_steps 100
        --learning_rate 1e-4
        --encoder_learning_rate 2e-5
        --encoder_freeze_steps 100
        --warmup_ratio 0.1
        --mask_time_prob 0
        --mask_feature_prob 0
        --save_total_limit 5
        --collapse_patience_evaluations 8
        --diagnostic_train_samples 400
        --fp16
        --gradient_checkpointing
      )
      ;;
  esac

  prediction_path="results/predictions/$(basename "$output_dir")_dev.jsonl"
  if [[ "$experiment" == "e1-30" ]]; then
    # Match the standalone fair-E1 launcher exactly so either entry point
    # detects and protects the other's artifacts.
    log_path="logs/wav2vec2_frozen_10h_fair_30ep.log"
  else
    log_path="logs/$(basename "$output_dir")_rtx3090.log"
  fi
  if [[ -e "$output_dir" || -e "$prediction_path" || -e "$log_path" ]]; then
    echo "Refusing to overwrite artifacts for $experiment:" >&2
    echo "  $output_dir" >&2
    echo "  $prediction_path" >&2
    echo "  $log_path" >&2
    return 1
  fi

  echo "===== Starting ${experiment^^}: $output_dir ====="
  "$PYTHON_EXE" -u scripts/train_ctc.py \
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
    --weight_decay 0.005 \
    --load_best_model_at_end \
    --collapse_non_empty_threshold 0.01 \
    --eval_accumulation_steps 10 \
    "${CUDNN_ARGS[@]}" \
    "${experiment_args[@]}" \
    2>&1 | tee "$log_path"
  echo "===== Completed ${experiment^^} ====="
}

for experiment in "${EXPERIMENTS[@]}"; do
  run_experiment "$experiment"
done
