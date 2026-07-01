param(
  [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (Test-Path "exp/wav2vec2_finetune_1h_repaired") {
  throw "exp/wav2vec2_finetune_1h_repaired already exists; refusing to overwrite it."
}

& $PythonExe -u scripts/train_ctc.py `
  --model_name_or_path facebook/wav2vec2-base `
  --train_manifest data/manifests/train_1h_effective_15s.jsonl `
  --eval_manifest data/manifests/dev_clean.jsonl `
  --output_dir exp/wav2vec2_finetune_1h_repaired `
  --prediction_path results/predictions/wav2vec2_finetune_1h_repaired_dev.jsonl `
  --ctc_loss_reduction mean `
  --ctc_zero_infinity `
  --max_duration_in_seconds 15 `
  --per_device_train_batch_size 1 `
  --per_device_eval_batch_size 1 `
  --gradient_accumulation_steps 8 `
  --max_steps 1500 `
  --eval_steps 100 `
  --learning_rate 1e-4 `
  --encoder_learning_rate 2e-5 `
  --encoder_freeze_steps 100 `
  --weight_decay 0.005 `
  --warmup_ratio 0.1 `
  --mask_time_prob 0 `
  --mask_feature_prob 0 `
  --load_best_model_at_end `
  --save_total_limit 5 `
  --eval_accumulation_steps 10 `
  --collapse_patience_evaluations 8 `
  --collapse_non_empty_threshold 0.01 `
  --diagnostic_train_samples 400 `
  --fp16 `
  --gradient_checkpointing
