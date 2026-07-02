#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p exp/deep_dive logs/deep_dive results/predictions/deep_dive artifacts/kmeans results/figures
PYTHON_EXE="${PYTHON_EXE:-python}"
source scripts/deep_dive_common.sh
LAYER="$(cat artifacts/deep_dive/best_layer_e13.txt 2>/dev/null || echo 9)"
MODE="${1:-centroid}"
RUN_POLICY=(--skip_existing)
[[ "${DEEP_DIVE_OVERWRITE:-0}" == 1 ]] && RUN_POLICY=(--overwrite)
KS=(500 1000)
[[ "$MODE" == embedding ]] && KS=(200)
[[ "${ONLY_K500:-0}" == 1 ]] && KS=(500)
for k in "${KS[@]}"; do
  codebook="artifacts/kmeans/deep_dive_layer${LAYER}_k${k}"
  "$PYTHON_EXE" scripts/train_kmeans_units.py \
    --train_metadata "features/wav2vec2_base/train/layer${LAYER}/metadata.jsonl" \
    --eval_metadata "features/wav2vec2_base/dev/layer${LAYER}/metadata.jsonl" \
    --output_dir "$codebook" --codebook_size "$k" --max_fit_frames 500000 \
    --seed 42 --resume
  experiment="e17_k${k}_centroid_bilstm_ctc"
  representation="discrete_centroid"
  center_args=(--centers "$codebook/centers.npy")
  if [[ "$MODE" == embedding ]]; then
    experiment="e18_k${k}_embedding_bilstm_ctc"
    representation="discrete_embedding"
    center_args=()
  fi
  "$PYTHON_EXE" -u scripts/train_cached_ctc_head.py \
    --experiment "$experiment" --representation_type "$representation" \
    --train_metadata "$codebook/train/metadata.jsonl" \
    --eval_metadata "$codebook/dev/metadata.jsonl" "${center_args[@]}" \
    --head_type bilstm --output_dir "exp/deep_dive/$experiment" \
    --prediction_path "results/predictions/deep_dive/${experiment}_dev.jsonl" \
    --metrics_path results/deep_dive_metrics.csv "${RUN_POLICY[@]}" \
    2>&1 | tee -a "logs/deep_dive/${experiment}.log"
  evaluate_cached_experiment "$experiment" "exp/deep_dive/$experiment" \
    "$codebook/centers.npy"
done
run_deep_dive_plots
