#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p exp/deep_dive logs/deep_dive results/predictions/deep_dive artifacts/deep_dive results/figures
PYTHON_EXE="${PYTHON_EXE:-python}"
source scripts/deep_dive_common.sh
if [[ "${1:-full}" == priority ]]; then
  LAYERS=(7 8 10 11)
else
  # Nearest-to-layer-9 probes run first so an interrupted rental still yields
  # the most informative part of the curve.
  LAYERS=(7 8 10 11 1 2 3 4 5)
fi
RUN_POLICY=(--skip_existing)
[[ "${DEEP_DIVE_OVERWRITE:-0}" == 1 ]] && RUN_POLICY=(--overwrite)
COMMON=(--metrics_path results/deep_dive_metrics.csv --max_steps "${MAX_STEPS:-3000}" --eval_steps "${EVAL_STEPS:-200}" --batch_size "${CACHED_BATCH_SIZE:-16}" "${RUN_POLICY[@]}")

"$PYTHON_EXE" scripts/extract_ssl_features.py \
  --train_manifest data/manifests/train_10h.jsonl \
  --dev_manifest data/manifests/dev_clean.jsonl \
  --layers "${LAYERS[@]}" --output_dir features/wav2vec2_base \
  --batch_size "${FEATURE_BATCH_SIZE:-2}" --skip_existing

for layer in "${LAYERS[@]}"; do
  experiment="e13_layer${layer}_bilstm_ctc"
  "$PYTHON_EXE" -u scripts/train_cached_ctc_head.py \
    --experiment "$experiment" --head_type bilstm --num_lstm_layers 2 \
    --train_metadata "features/wav2vec2_base/train/layer${layer}/metadata.jsonl" \
    --eval_metadata "features/wav2vec2_base/dev/layer${layer}/metadata.jsonl" \
    --output_dir "exp/deep_dive/$experiment" \
    --prediction_path "results/predictions/deep_dive/${experiment}_dev.jsonl" \
    "${COMMON[@]}" 2>&1 | tee -a "logs/deep_dive/${experiment}.log"
  evaluate_cached_experiment "$experiment" "exp/deep_dive/$experiment"
done

"$PYTHON_EXE" scripts/select_best_representation_layer.py \
  --summary_glob "exp/deep_dive/e13_layer*_bilstm_ctc/summary.json" \
  --out artifacts/deep_dive/best_layer_e13.txt
"$PYTHON_EXE" scripts/plot_deep_dive_results.py
