param(
  [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (Test-Path "exp/wav2vec2_finetune_1h") {
  throw "exp/wav2vec2_finetune_1h already exists; refusing to overwrite it."
}

& $PythonExe -u scripts/train_ctc.py `
  --model_name_or_path facebook/wav2vec2-base `
  --train_manifest data/manifests/train_1h.jsonl `
  --eval_manifest data/manifests/dev_clean.jsonl `
  --output_dir exp/wav2vec2_finetune_1h `
  --prediction_path results/predictions/wav2vec2_finetune_1h_dev.jsonl `
  --ctc_loss_reduction mean `
  --ctc_zero_infinity `
  --max_duration_in_seconds 15 `
  --per_device_train_batch_size 1 `
  --per_device_eval_batch_size 1 `
  --gradient_accumulation_steps 16 `
  --max_steps 1000 `
  --eval_steps 100 `
  --learning_rate 1e-4 `
  --weight_decay 0.005 `
  --warmup_ratio 0.1 `
  --save_total_limit 2 `
  --eval_accumulation_steps 10 `
  --fp16 `
  --gradient_checkpointing
