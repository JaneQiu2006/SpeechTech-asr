#!/usr/bin/env bash
# Shared train-then-test helpers. Source this file; do not execute it directly.

run_deep_dive_plots() {
  echo "===== Updating deep-dive figures ====="
  if ! "$PYTHON_EXE" scripts/plot_deep_dive_results.py; then
    echo "[warning] Plot generation failed; training results are preserved." >&2
    echo "[warning] Install requirements-rtx3090.txt and rerun the plotting script later." >&2
  fi
  return 0
}

evaluate_cached_experiment() {
  local experiment="$1"
  local model_dir="$2"
  local centers="${3:-}"
  local -a center_args=()
  local -a test_policy=()
  [[ -n "$centers" ]] && center_args=(--centers "$centers")
  [[ "${DEEP_DIVE_OVERWRITE:-0}" == 1 ]] && test_policy=(--overwrite)
  echo "===== Testing $experiment on test-clean ====="
  "$PYTHON_EXE" -u scripts/evaluate_representation_ctc.py \
    --experiment "$experiment" \
    --model_dir "$model_dir" \
    --manifest data/manifests/test_clean.jsonl \
    --prediction_path "results/predictions/deep_dive/${experiment}_test.jsonl" \
    --metrics_path results/deep_dive_metrics.csv \
    --batch_size "${TEST_BATCH_SIZE:-2}" \
    "${center_args[@]}" "${test_policy[@]}" \
    2>&1 | tee -a "logs/deep_dive/${experiment}.log"
}

evaluate_finetune_experiment() {
  local experiment_id="$1"
  local experiment="$2"
  local model_dir="$3"
  local frozen_layers="${4:-0}"
  local -a test_policy=()
  [[ "${DEEP_DIVE_OVERWRITE:-0}" == 1 ]] && test_policy=(--overwrite)
  echo "===== Testing $experiment on test-clean ====="
  "$PYTHON_EXE" -u scripts/evaluate_e1_e5.py \
    --custom_experiment_id "$experiment_id" \
    --custom_experiment_name "$experiment" \
    --custom_model_dir "$model_dir" \
    --custom_processor_dir "$model_dir" \
    --custom_freeze_transformer_layers "$frozen_layers" \
    --manifest data/manifests/test_clean.jsonl \
    --predictions_dir results/predictions/deep_dive \
    --metrics_path results/deep_dive_metrics.csv \
    --batch_size "${TEST_BATCH_SIZE:-2}" \
    "${test_policy[@]}" \
    2>&1 | tee -a "logs/deep_dive/${experiment}.log"
}
