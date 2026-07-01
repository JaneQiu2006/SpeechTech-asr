param(
  [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

& $PythonExe -u scripts/train_ctc.py `
  --model_name_or_path facebook/wav2vec2-base `
  --train_manifest data/manifests/train_10h.jsonl `
  --eval_manifest data/manifests/dev_clean.jsonl `
  --output_dir exp/wav2vec2_frozen_10h `
  --freeze_encoder `
  --max_duration_in_seconds 15 `
  --per_device_train_batch_size 1 `
  --per_device_eval_batch_size 1 `
  --gradient_accumulation_steps 16 `
  --learning_rate 3e-4 `
  --num_train_epochs 3 `
  --save_total_limit 2 `
  --eval_accumulation_steps 10 `
  --fp16
