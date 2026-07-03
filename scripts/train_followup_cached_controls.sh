#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_EXE="${PYTHON_EXE:-python}"
source scripts/deep_dive_common.sh

mkdir -p exp/deep_dive logs/deep_dive results/predictions/deep_dive \
  artifacts/kmeans features/wav2vec2_base results/figures

RUN_POLICY=(--skip_existing)
[[ "${FOLLOWUP_OVERWRITE:-0}" == 1 ]] && RUN_POLICY=(--overwrite)
RUNTIME_ARGS=()
[[ "${DISABLE_CUDNN:-0}" == 1 ]] && RUNTIME_ARGS=(--disable_cudnn)

echo "===== Ensuring layer-8 train/dev feature caches ====="
"$PYTHON_EXE" -u scripts/extract_ssl_features.py \
  --train_manifest data/manifests/train_10h.jsonl \
  --dev_manifest data/manifests/dev_clean.jsonl \
  --output_dir features/wav2vec2_base --layers 8 --skip_existing \
  "${RUNTIME_ARGS[@]}"

ensure_codebook() {
  local k="$1"
  local directory="artifacts/kmeans/deep_dive_layer8_k${k}"
  echo "===== Ensuring layer-8 K=${k} codebook ====="
  "$PYTHON_EXE" -u scripts/train_kmeans_units.py \
    --train_metadata features/wav2vec2_base/train/layer8/metadata.jsonl \
    --eval_metadata features/wav2vec2_base/dev/layer8/metadata.jsonl \
    --output_dir "$directory" --codebook_size "$k" \
    --max_fit_frames 500000 --seed 42 --resume
}

train_cached() {
  local experiment="$1"
  local representation="$2"
  local train_metadata="$3"
  local dev_metadata="$4"
  local centers="${5:-}"
  local embedding_dim="${6:-256}"
  local -a center_args=()
  [[ -n "$centers" && "$representation" == discrete_centroid ]] \
    && center_args=(--centers "$centers")
  echo "===== Training $experiment ====="
  "$PYTHON_EXE" -u scripts/train_cached_ctc_head.py \
    --experiment "$experiment" \
    --representation_type "$representation" \
    --train_metadata "$train_metadata" \
    --eval_metadata "$dev_metadata" \
    "${center_args[@]}" \
    --head_type bilstm --num_layers 2 \
    --unit_embedding_dim "$embedding_dim" \
    --tokenizer_mode vocab_only \
    --output_dir "exp/deep_dive/$experiment" \
    --prediction_path "results/predictions/deep_dive/${experiment}_dev.jsonl" \
    --metrics_path results/deep_dive_metrics.csv \
    --seed 42 --max_steps "${CACHED_MAX_STEPS:-3000}" \
    --eval_steps "${CACHED_EVAL_STEPS:-200}" \
    --batch_size "${CACHED_BATCH_SIZE:-16}" \
    "${RUNTIME_ARGS[@]}" \
    "${RUN_POLICY[@]}" \
    2>&1 | tee -a "logs/deep_dive/${experiment}.log"
  evaluate_cached_experiment \
    "$experiment" "exp/deep_dive/$experiment" "$centers" vocab_only
}

train_cached \
  e24a_layer8_continuous_vocab30 continuous \
  features/wav2vec2_base/train/layer8/metadata.jsonl \
  features/wav2vec2_base/dev/layer8/metadata.jsonl

ensure_codebook 200
K200=artifacts/kmeans/deep_dive_layer8_k200
train_cached \
  e24b_layer8_k200_centroid_vocab30 discrete_centroid \
  "$K200/train/metadata.jsonl" "$K200/dev/metadata.jsonl" "$K200/centers.npy"
train_cached \
  e24c_layer8_k200_embedding_vocab30 discrete_embedding \
  "$K200/train/metadata.jsonl" "$K200/dev/metadata.jsonl" "$K200/centers.npy" 256
train_cached \
  e24e_layer8_k200_embedding768_vocab30 discrete_embedding \
  "$K200/train/metadata.jsonl" "$K200/dev/metadata.jsonl" "$K200/centers.npy" 768

ensure_codebook 1000
K1000=artifacts/kmeans/deep_dive_layer8_k1000
train_cached \
  e24d_layer8_k1000_centroid_vocab30 discrete_centroid \
  "$K1000/train/metadata.jsonl" "$K1000/dev/metadata.jsonl" "$K1000/centers.npy"
