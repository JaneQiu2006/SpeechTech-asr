#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_EXE="${PYTHON_EXE:-python}"
source scripts/deep_dive_common.sh
mkdir -p exp/deep_dive logs/deep_dive results/predictions/deep_dive \
  artifacts/deep_dive artifacts/kmeans results/figures
stages=("$@")
[[ ${#stages[@]} -eq 0 ]] && stages=(all_p0)
overwrite=0
filtered=()
for stage in "${stages[@]}"; do
  [[ "$stage" == --overwrite ]] && { overwrite=1; continue; }
  case "$stage" in
    all_p0) filtered+=(e13_priority e15) ;;
    all_p1) filtered+=(e13 e15 e14 e16 e17 e21) ;;
    all) filtered+=(e13 e15 e14 e16 e17 e18 e20 e21) ;;
    e13|e13_priority|e14|e15|e16|e17|e18|e20|e21|e22|e23) filtered+=("$stage") ;;
    *) echo "Unknown stage: $stage" >&2; exit 2 ;;
  esac
done
if [[ "$overwrite" == 1 ]]; then
  echo "--overwrite is forwarded to Python jobs only; existing E1-E12 artifacts are untouched."
  export DEEP_DIVE_OVERWRITE=1
fi
for stage in "${filtered[@]}"; do
  echo "===== $stage ====="
  case "$stage" in
    e13) bash scripts/train_deep_dive_layerwise.sh full ;;
    e13_priority) bash scripts/train_deep_dive_layerwise.sh priority ;;
    e14)
      "$PYTHON_EXE" scripts/extract_ssl_features.py \
        --train_manifest data/manifests/train_10h.jsonl \
        --dev_manifest data/manifests/dev_clean.jsonl \
        --layers {0..12} --output_dir features/wav2vec2_base --skip_existing
      "$PYTHON_EXE" scripts/train_layer_mixture_ctc_head.py ;;
    e15) bash scripts/train_deep_dive_head_capacity.sh ;;
    e16) bash scripts/train_partial_ft_grid.sh all ;;
    e17) bash scripts/train_deep_dive_discrete_k_grid.sh centroid ;;
    e18) bash scripts/train_deep_dive_discrete_k_grid.sh embedding ;;
    e20) bash scripts/train_frozen_layer_data_scale.sh ;;
    e21) "$PYTHON_EXE" scripts/analyze_deep_dive_errors.py --pred_glob \
      "results/predictions/**/*.jsonl" --out_csv results/deep_dive_error_stats.csv \
      --out_md results/deep_dive_error_analysis.md ;;
    e22) bash scripts/train_masking_grid.sh ;;
    e23) bash scripts/rerun_best_deep_dive_seed.sh ;;
  esac
done
run_deep_dive_plots
echo "===== status ====="
for directory in exp/deep_dive/*; do
  [[ -d "$directory" ]] || continue
  name="${directory#exp/deep_dive/}"
  [[ -f "$directory/completion.json" ]] && train_state=complete || train_state=incomplete
  [[ -f "results/predictions/deep_dive/${name}_test.jsonl" ]] \
    && test_state=complete || test_state=incomplete
  printf '%-55s train=%-10s test=%s\n' "$name" "$train_state" "$test_state"
done
