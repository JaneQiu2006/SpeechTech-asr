#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p exp/deep_dive logs/deep_dive results/predictions/deep_dive
PYTHON_EXE="${PYTHON_EXE:-python}"
source scripts/deep_dive_common.sh
LAYER="$(cat artifacts/deep_dive/best_layer_e13.txt 2>/dev/null || echo 9)"
experiment="e23_layer${LAYER}_bilstm_ctc_seed1234"
"$PYTHON_EXE" -u scripts/train_cached_ctc_head.py \
  --experiment "$experiment" --head_type bilstm --num_layers 2 --seed 1234 \
  --train_metadata "features/wav2vec2_base/train/layer${LAYER}/metadata.jsonl" \
  --eval_metadata "features/wav2vec2_base/dev/layer${LAYER}/metadata.jsonl" \
  --output_dir "exp/deep_dive/$experiment" \
  --prediction_path "results/predictions/deep_dive/${experiment}_dev.jsonl" \
  --metrics_path results/deep_dive_metrics.csv --skip_existing \
  2>&1 | tee -a "logs/deep_dive/${experiment}.log"
evaluate_cached_experiment "$experiment" "exp/deep_dive/$experiment"
