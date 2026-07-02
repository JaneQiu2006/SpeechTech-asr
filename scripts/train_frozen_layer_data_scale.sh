#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p exp/deep_dive logs/deep_dive results/predictions/deep_dive
PYTHON_EXE="${PYTHON_EXE:-python}"
source scripts/deep_dive_common.sh
LAYER="$(cat artifacts/deep_dive/best_layer_e13.txt 2>/dev/null || echo 9)"
RUN_POLICY=(--skip_existing)
[[ "${DEEP_DIVE_OVERWRITE:-0}" == 1 ]] && RUN_POLICY=(--overwrite)
for spec in "e20a:1h" "e20b:3h"; do
  IFS=: read -r id scale <<<"$spec"
  "$PYTHON_EXE" scripts/extract_ssl_features.py \
    --manifest "data/manifests/train_${scale}_effective_15s.jsonl" \
    --split "train_${scale}" --output_root features/wav2vec2_base \
    --layers "$LAYER" --skip_existing
  experiment="${id}_${scale}_layer${LAYER}_bilstm_ctc"
  "$PYTHON_EXE" scripts/train_cached_ctc_head.py \
    --experiment "$experiment" --head_type bilstm \
    --train_metadata "features/wav2vec2_base/train_${scale}/layer${LAYER}/metadata.jsonl" \
    --eval_metadata "features/wav2vec2_base/dev/layer${LAYER}/metadata.jsonl" \
    --output_dir "exp/deep_dive/$experiment" \
    --prediction_path "results/predictions/deep_dive/${experiment}_dev.jsonl" \
    --metrics_path results/deep_dive_metrics.csv "${RUN_POLICY[@]}"
  evaluate_cached_experiment "$experiment" "exp/deep_dive/$experiment"
done
