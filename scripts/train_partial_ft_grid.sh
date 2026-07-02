#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p exp/deep_dive logs/deep_dive results/predictions/deep_dive
PYTHON_EXE="${PYTHON_EXE:-python}"
source scripts/deep_dive_common.sh
requested="${1:-all}"
RUN_POLICY=(--skip_existing)
[[ "${DEEP_DIVE_OVERWRITE:-0}" == 1 ]] && RUN_POLICY=(--overwrite)
case "$requested" in
  top4) grid=("e16a:top4:8") ;;
  top5) grid=("e16b:top5:7") ;;
  top8) grid=("e16c:top8:4") ;;
  all) grid=("e16b:top5:7" "e16a:top4:8" "e16c:top8:4") ;;
  *) echo "Use top4, top5, top8, or all" >&2; exit 2 ;;
esac
for spec in "${grid[@]}"; do
  IFS=: read -r id name frozen <<<"$spec"
  experiment="${id}_${name}_finetune"
  "$PYTHON_EXE" -u scripts/train_ctc.py \
    --experiment_id "${id^^}" --model_name_or_path facebook/wav2vec2-base \
    --train_manifest data/manifests/train_10h.jsonl \
    --eval_manifest data/manifests/dev_clean.jsonl \
    --output_dir "exp/deep_dive/$experiment" \
    --prediction_path "results/predictions/deep_dive/${experiment}_dev.jsonl" \
    --metrics_path results/deep_dive_metrics.csv \
    --freeze_transformer_layers "$frozen" --learning_rate 1e-4 \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --weight_decay 0.005 --warmup_ratio 0.1 --max_duration_in_seconds 15 \
    --max_steps "${MAX_STEPS:-2500}" \
    --eval_steps "${EVAL_STEPS:-300}" --load_best_model_at_end \
    --ctc_zero_infinity --fp16 --gradient_checkpointing "${RUN_POLICY[@]}" \
    2>&1 | tee -a "logs/deep_dive/${experiment}.log"
  evaluate_finetune_experiment "${id^^}" "$experiment" \
    "exp/deep_dive/$experiment" "$frozen"
done
