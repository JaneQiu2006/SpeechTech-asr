#!/usr/bin/env bash
# Run every experiment proposed in section 14 of the comprehensive analysis.
#
# Usage:
#   bash scripts/run_all_followup_experiments_rtx3090.sh
#   bash scripts/run_all_followup_experiments_rtx3090.sh --from seed
#   bash scripts/run_all_followup_experiments_rtx3090.sh --overwrite
#
# Environment overrides:
#   PYTHON_EXE=/path/to/python
#   DISABLE_CUDNN=1
#   CACHED_BATCH_SIZE=16 TEST_BATCH_SIZE=2
#   CACHED_MAX_STEPS=3000 PARTIAL_MAX_STEPS=2500

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_EXE="${PYTHON_EXE:-python}"

start_stage=cached
overwrite=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)
      [[ $# -ge 2 ]] || { echo "--from requires cached, seed, or analysis" >&2; exit 2; }
      start_stage="$2"
      shift 2
      ;;
    --overwrite)
      overwrite=1
      shift
      ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$start_stage" in
  cached) start_index=0 ;;
  seed) start_index=1 ;;
  analysis) start_index=2 ;;
  *) echo "--from requires cached, seed, or analysis" >&2; exit 2 ;;
esac

if [[ "$overwrite" == 1 ]]; then
  export FOLLOWUP_OVERWRITE=1
  export DEEP_DIVE_OVERWRITE=1
fi

mkdir -p exp/deep_dive logs/deep_dive results/predictions/deep_dive results/figures

required=(
  data/manifests/train_10h.jsonl
  data/manifests/train_1h_effective_15s.jsonl
  data/manifests/dev_clean.jsonl
  data/manifests/test_clean.jsonl
  data/vocab/vocab.json
  results/predictions/wav2vec2_finetune_1h_repaired_test.jsonl
  results/predictions/deep_dive/e20a_1h_layer8_bilstm_ctc_test.jsonl
  results/predictions/deep_dive/e16b_top5_finetune_test.jsonl
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "Missing required input: $path" >&2; exit 1; }
done

echo "===== Preflight ====="
"$PYTHON_EXE" -c \
  "import jiwer, numpy, sklearn, soundfile, torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); assert torch.cuda.is_available()"
"$PYTHON_EXE" -c \
  "import json; v=json.load(open('data/vocab/vocab.json')); assert max(v.values())+1 == 30, v; print('CTC vocabulary: 30 IDs')"

stages=(cached seed analysis)
for index in "${!stages[@]}"; do
  (( index < start_index )) && continue
  stage="${stages[$index]}"
  echo "===== Follow-up stage: $stage ====="
  case "$stage" in
    cached)
      bash scripts/train_followup_cached_controls.sh
      ;;
    seed)
      bash scripts/train_followup_second_seed.sh
      ;;
    analysis)
      comparisons=(
        e24_vocab30_continuous
        results/predictions/deep_dive/e13_layer8_bilstm_ctc_test.jsonl
        results/predictions/deep_dive/e24a_layer8_continuous_vocab30_test.jsonl
        e24_k200_centroid_vs_embedding
        results/predictions/deep_dive/e24b_layer8_k200_centroid_vocab30_test.jsonl
        results/predictions/deep_dive/e24c_layer8_k200_embedding_vocab30_test.jsonl
        e24_k200_centroid_vs_embedding768
        results/predictions/deep_dive/e24b_layer8_k200_centroid_vocab30_test.jsonl
        results/predictions/deep_dive/e24e_layer8_k200_embedding768_vocab30_test.jsonl
        e24_k200_vs_k1000_centroid
        results/predictions/deep_dive/e24b_layer8_k200_centroid_vocab30_test.jsonl
        results/predictions/deep_dive/e24d_layer8_k1000_centroid_vocab30_test.jsonl
        e16b_seed42_vs_seed1234
        results/predictions/deep_dive/e16b_top5_finetune_test.jsonl
        results/predictions/deep_dive/e25_top5_finetune_seed1234_test.jsonl
        one_hour_full_ft_vs_frozen
        results/predictions/wav2vec2_finetune_1h_repaired_test.jsonl
        results/predictions/deep_dive/e20a_1h_layer8_bilstm_ctc_test.jsonl
      )
      comparison_args=()
      for ((i=0; i<${#comparisons[@]}; i+=3)); do
        for path in "${comparisons[$((i+1))]}" "${comparisons[$((i+2))]}"; do
          [[ -s "$path" ]] || { echo "Missing bootstrap prediction: $path" >&2; exit 1; }
        done
        comparison_args+=(
          --comparison "${comparisons[$i]}" \
          "${comparisons[$((i+1))]}" "${comparisons[$((i+2))]}"
        )
      done
      "$PYTHON_EXE" -u scripts/paired_bootstrap_asr.py \
        "${comparison_args[@]}" --samples 10000 --seed 42 \
        --output_csv results/followup_paired_bootstrap.csv \
        --output_md results/followup_paired_bootstrap.md
      "$PYTHON_EXE" -u scripts/analyze_deep_dive_errors.py \
        --pred_glob "results/predictions/**/*.jsonl" \
        --out_csv results/deep_dive_error_stats.csv \
        --out_md results/deep_dive_error_analysis.md
      "$PYTHON_EXE" -u scripts/build_master_results.py \
        --output results/master_metrics.csv
      "$PYTHON_EXE" -u scripts/plot_deep_dive_results.py
      ;;
  esac
done

echo "===== Follow-up status ====="
expected=(
  e24a_layer8_continuous_vocab30
  e24b_layer8_k200_centroid_vocab30
  e24c_layer8_k200_embedding_vocab30
  e24d_layer8_k1000_centroid_vocab30
  e24e_layer8_k200_embedding768_vocab30
  e25_top5_finetune_seed1234
)
failed=0
for experiment in "${expected[@]}"; do
  train=missing
  test=missing
  [[ -s "exp/deep_dive/$experiment/completion.json" ]] && train=complete
  [[ -s "results/predictions/deep_dive/${experiment}_test.jsonl" ]] && test=complete
  printf '%-52s train=%-8s test=%s\n' "$experiment" "$train" "$test"
  [[ "$train" == complete && "$test" == complete ]] || failed=1
done
[[ -s results/master_metrics.csv ]] || failed=1
[[ -s results/followup_paired_bootstrap.csv ]] || failed=1
if [[ "$failed" != 0 ]]; then
  echo "One or more follow-up artifacts are incomplete." >&2
  exit 1
fi
echo "All follow-up experiments and analyses are complete."
