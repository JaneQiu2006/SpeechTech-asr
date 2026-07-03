#!/usr/bin/env bash
set -euo pipefail

# Extra-day queue from doc/ssl_representation_discrete_units_extension_plan.md.
# Default: E8 -> E9 -> E10a/b/c -> select best layer -> E11 K=50/100/200 -> E12a.
# Individual stages are also supported:
#   bash scripts/train_representation_extension.sh e10 e11

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_EXE="${PYTHON_EXE:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
CACHED_BATCH_SIZE="${CACHED_BATCH_SIZE:-16}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CUDNN_ARGS=()
STAGES=("$@")
if [[ "${#STAGES[@]}" -eq 0 ]]; then
  STAGES=(e8 e9 e10 e11 e12a)
fi

for stage in "${STAGES[@]}"; do
  case "$stage" in
    e8|e9|e10|e11|e12a) ;;
    *)
      echo "Unknown stage '$stage'; use e8, e9, e10, e11, or e12a." >&2
      exit 2
      ;;
  esac
done

mkdir -p logs exp results/predictions features artifacts/kmeans

cleanup_temporary_files() {
  # Only remove atomic-write debris owned by this extension pipeline.
  local directory
  for directory in features/wav2vec2_base artifacts/kmeans; do
    if [[ -d "$directory" ]]; then
      find "$directory" -type f -name '*.tmp' -delete
    fi
  done
}
trap cleanup_temporary_files EXIT

for required in \
  scripts/train_ctc.py \
  scripts/extract_ssl_features.py \
  scripts/train_cached_ctc_head.py \
  scripts/train_kmeans_units.py \
  scripts/select_best_representation_layer.py \
  data/manifests/train_10h.jsonl \
  data/manifests/train_1h_effective_15s.jsonl \
  data/manifests/dev_clean.jsonl \
  data/vocab/vocab.json; do
  [[ -f "$required" ]] || {
    echo "Required file not found: $required" >&2
    exit 1
  }
done

"$PYTHON_EXE" -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
"$PYTHON_EXE" -c \
  "import sklearn; print('scikit-learn=', sklearn.__version__)"
if ! "$PYTHON_EXE" -c \
  "import torch; x=torch.zeros((2,1,240000),device='cuda',dtype=torch.float16); c=torch.nn.Conv1d(1,512,10,5,bias=False).cuda().half(); print(tuple(c(x).shape))" \
  2>logs/representation_extension_cudnn_preflight_error.log; then
  echo "cuDNN Conv1d preflight failed; using native CUDA kernels." >&2
  CUDNN_ARGS=(--disable_cudnn)
fi
rm -f logs/representation_extension_cudnn_preflight_error.log

stage_requested() {
  local expected="$1"
  local item
  for item in "${STAGES[@]}"; do
    [[ "$item" == "$expected" ]] && return 0
  done
  return 1
}

run_existing_experiment() {
  local id="$1"
  local output_dir="$2"
  local prediction="$3"
  if [[ -f "$output_dir/model.safetensors" && -f "$prediction" ]]; then
    echo "===== ${id^^} already complete; skipping ====="
    return
  fi
  if [[ -e "$output_dir" || -e "$prediction" ]]; then
    echo "Partial artifacts block ${id^^}; inspect before rerunning:" >&2
    echo "  $output_dir" >&2
    echo "  $prediction" >&2
    exit 1
  fi
  bash scripts/train_new_experiments.sh "$id"
}

run_cached_head() {
  local experiment="$1"
  local train_metadata="$2"
  local eval_metadata="$3"
  local centers="${4:-}"
  local output_dir="exp/$experiment"
  local prediction="results/predictions/${experiment}_dev.jsonl"
  local log="logs/${experiment}.log"
  if [[ -f "$output_dir/completion.json" && -f "$prediction" ]]; then
    echo "===== $experiment already complete; skipping ====="
    return
  fi
  local -a resume_args=()
  [[ -d "$output_dir" ]] && resume_args=(--resume)
  local -a center_args=()
  [[ -n "$centers" ]] && center_args=(--centers "$centers")
  echo "===== Starting $experiment ====="
  "$PYTHON_EXE" -u scripts/train_cached_ctc_head.py \
    --experiment "$experiment" \
    --train_metadata "$train_metadata" \
    --eval_metadata "$eval_metadata" \
    --output_dir "$output_dir" \
    --prediction_path "$prediction" \
    --batch_size "$CACHED_BATCH_SIZE" \
    --max_steps 3000 \
    --eval_steps 200 \
    --learning_rate 1e-3 \
    --weight_decay 0.005 \
    --warmup_ratio 0.1 \
    --collapse_patience_evaluations 4 \
    "${center_args[@]}" \
    "${resume_args[@]}" \
    "${CUDNN_ARGS[@]}" \
    2>&1 | tee -a "$log"
  echo "===== Completed $experiment ====="
}

if stage_requested e8; then
  run_existing_experiment \
    e8 \
    exp/wav2vec2_masking_finetune_10h \
    results/predictions/wav2vec2_masking_finetune_10h_dev.jsonl
fi

if stage_requested e9; then
  run_existing_experiment \
    e9 \
    exp/wav2vec2_frozen_bilstm_10h \
    results/predictions/wav2vec2_frozen_bilstm_10h_dev.jsonl
fi

if stage_requested e10; then
  echo "===== Extracting frozen Wav2Vec2 train features ====="
  "$PYTHON_EXE" -u scripts/extract_ssl_features.py \
    --manifest data/manifests/train_10h.jsonl \
    --split train \
    --output_root features/wav2vec2_base \
    --layers 6 9 12 \
    --data_root "$DATA_ROOT" \
    --max_duration_in_seconds 15 \
    --batch_size "$FEATURE_BATCH_SIZE" \
    --resume \
    "${CUDNN_ARGS[@]}" \
    2>&1 | tee -a logs/wav2vec2_feature_extraction.log

  echo "===== Extracting frozen Wav2Vec2 dev features ====="
  "$PYTHON_EXE" -u scripts/extract_ssl_features.py \
    --manifest data/manifests/dev_clean.jsonl \
    --split dev \
    --output_root features/wav2vec2_base \
    --layers 6 9 12 \
    --data_root "$DATA_ROOT" \
    --max_duration_in_seconds 15 \
    --batch_size "$FEATURE_BATCH_SIZE" \
    --resume \
    "${CUDNN_ARGS[@]}" \
    2>&1 | tee -a logs/wav2vec2_feature_extraction.log

  for layer in 6 9 12; do
    run_cached_head \
      "wav2vec2_layer${layer}_bilstm_ctc" \
      "features/wav2vec2_base/train/layer${layer}/metadata.jsonl" \
      "features/wav2vec2_base/dev/layer${layer}/metadata.jsonl"
  done
  "$PYTHON_EXE" scripts/select_best_representation_layer.py \
    --experiments_root exp \
    --layers 6 9 12 \
    --output artifacts/kmeans/best_layer.txt
fi

if stage_requested e11; then
  if [[ ! -f artifacts/kmeans/best_layer.txt ]]; then
    "$PYTHON_EXE" scripts/select_best_representation_layer.py \
      --experiments_root exp \
      --layers 6 9 12 \
      --output artifacts/kmeans/best_layer.txt
  fi
  BEST_LAYER="$(tr -d '[:space:]' < artifacts/kmeans/best_layer.txt)"
  case "$BEST_LAYER" in
    6|9|12) ;;
    *)
      echo "Invalid selected layer: '$BEST_LAYER'" >&2
      exit 1
      ;;
  esac
  echo "===== E11 uses Wav2Vec2 layer $BEST_LAYER ====="
  for codebook_size in 50 100 200; do
    codebook_dir="artifacts/kmeans/wav2vec2_layer${BEST_LAYER}_k${codebook_size}"
    "$PYTHON_EXE" -u scripts/train_kmeans_units.py \
      --train_metadata "features/wav2vec2_base/train/layer${BEST_LAYER}/metadata.jsonl" \
      --eval_metadata "features/wav2vec2_base/dev/layer${BEST_LAYER}/metadata.jsonl" \
      --output_dir "$codebook_dir" \
      --codebook_size "$codebook_size" \
      --max_fit_frames 500000 \
      --seed 42 \
      --resume \
      2>&1 | tee -a "logs/wav2vec2_kmeans_k${codebook_size}.log"

    run_cached_head \
      "wav2vec2_discrete_k${codebook_size}_bilstm_ctc" \
      "$codebook_dir/train/metadata.jsonl" \
      "$codebook_dir/dev/metadata.jsonl" \
      "$codebook_dir/centers.npy"
  done
fi

if stage_requested e12a; then
  output_dir="exp/wav2vec2_finetune_1h_time_mask"
  prediction="results/predictions/wav2vec2_finetune_1h_time_mask_dev.jsonl"
  log="logs/wav2vec2_finetune_1h_time_mask.log"
  if [[ -f "$output_dir/model.safetensors" && -f "$prediction" ]]; then
    echo "===== E12a already complete; skipping ====="
  elif [[ -e "$output_dir" || -e "$prediction" || -e "$log" ]]; then
    echo "Partial E12a artifacts exist; inspect them before rerunning." >&2
    exit 1
  else
    "$PYTHON_EXE" -u scripts/train_ctc.py \
      --model_name_or_path facebook/wav2vec2-base \
      --train_manifest data/manifests/train_1h_effective_15s.jsonl \
      --eval_manifest data/manifests/dev_clean.jsonl \
      --data_root "$DATA_ROOT" \
      --output_dir "$output_dir" \
      --prediction_path "$prediction" \
      --ctc_loss_reduction mean \
      --ctc_zero_infinity \
      --max_duration_in_seconds 15 \
      --per_device_train_batch_size 2 \
      --per_device_eval_batch_size 2 \
      --gradient_accumulation_steps 4 \
      --max_steps 1500 \
      --eval_steps 100 \
      --learning_rate 1e-4 \
      --encoder_learning_rate 2e-5 \
      --encoder_freeze_steps 100 \
      --weight_decay 0.005 \
      --warmup_ratio 0.1 \
      --mask_time_prob 0.05 \
      --mask_feature_prob 0.0 \
      --load_best_model_at_end \
      --save_total_limit 5 \
      --collapse_patience_evaluations 8 \
      --collapse_non_empty_threshold 0.01 \
      --diagnostic_train_samples 400 \
      --eval_accumulation_steps 10 \
      --fp16 \
      --gradient_checkpointing \
      "${CUDNN_ARGS[@]}" \
      2>&1 | tee "$log"
  fi
fi

echo "===== Representation extension queue complete ====="
