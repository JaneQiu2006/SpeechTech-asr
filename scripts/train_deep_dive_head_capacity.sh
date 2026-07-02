#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p exp/deep_dive logs/deep_dive results/predictions/deep_dive results/figures
PYTHON_EXE="${PYTHON_EXE:-python}"
source scripts/deep_dive_common.sh
LAYER="${1:-$(cat artifacts/deep_dive/best_layer_e13.txt 2>/dev/null || echo 9)}"
INCLUDE_TRANSFORMER="${INCLUDE_TRANSFORMER:-0}"
RUN_POLICY=(--skip_existing)
[[ "${DEEP_DIVE_OVERWRITE:-0}" == 1 ]] && RUN_POLICY=(--overwrite)
heads=("e15a:linear:1:1e-3" "e15b:mlp:1:1e-3" "e15c:bilstm:1:8e-4" "e15d:bilstm:2:1e-3")
[[ "$INCLUDE_TRANSFORMER" == 1 ]] && heads+=("e15e:transformer:2:5e-4")
for spec in "${heads[@]}"; do
  IFS=: read -r id head layers lr <<<"$spec"
  experiment="${id}_${head}_layer${LAYER}_ctc"
  "$PYTHON_EXE" -u scripts/train_cached_ctc_head.py \
    --experiment "$experiment" --head_type "$head" --num_layers "$layers" \
    --train_metadata "features/wav2vec2_base/train/layer${LAYER}/metadata.jsonl" \
    --eval_metadata "features/wav2vec2_base/dev/layer${LAYER}/metadata.jsonl" \
    --output_dir "exp/deep_dive/$experiment" \
    --prediction_path "results/predictions/deep_dive/${experiment}_dev.jsonl" \
    --metrics_path results/deep_dive_metrics.csv --learning_rate "$lr" \
    --max_steps "${MAX_STEPS:-3000}" --eval_steps "${EVAL_STEPS:-200}" \
    --batch_size "${CACHED_BATCH_SIZE:-16}" "${RUN_POLICY[@]}" \
    2>&1 | tee -a "logs/deep_dive/${experiment}.log"
  evaluate_cached_experiment "$experiment" "exp/deep_dive/$experiment"
done
"$PYTHON_EXE" scripts/plot_deep_dive_results.py
