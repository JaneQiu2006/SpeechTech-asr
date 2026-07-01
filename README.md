# Low-resource ASR with self-supervised speech representations

This repository contains the startup implementation described in
`doc/ssl_low_resource_asr_project_plan.md`: reproducible LibriSpeech subsets
and a minimal Wav2Vec2/WavLM CTC training pipeline.

## 1. Environment

Python 3.10 is recommended. Install a CUDA-compatible PyTorch build first,
then install the remaining dependencies:

```powershell
pip install -r requirements.txt
```

## 2. Data preparation

Put extracted LibriSpeech directories directly below `data/`:

```text
data/
├── train-clean-100/
├── dev-clean/
└── test-clean/
```

The preparation script always checks these local directories first. If a
split is absent, it downloads that split from
`openslr/librispeech_asr` on Hugging Face and materializes it below `data/`.

```powershell
python scripts/prepare_librispeech_subsets.py
```

Outputs:

```text
data/manifests/train_1h.jsonl
data/manifests/train_10h.jsonl
data/manifests/dev_clean.jsonl
data/manifests/test_clean.jsonl
data/vocab/vocab.json
```

For a local-only data check that never starts a download:

```powershell
python scripts/prepare_librispeech_subsets.py --splits dev-clean test-clean --local_only
```

## 3. Startup debug training

The first run downloads the self-supervised checkpoint
`facebook/wav2vec2-base`. This is the pretraining-only checkpoint, not the
ASR-finetuned `-960h` model.

```powershell
python scripts/train_ctc.py `
  --train_manifest data/manifests/train_1h.jsonl `
  --eval_manifest data/manifests/dev_clean.jsonl `
  --output_dir exp/debug_wav2vec2 `
  --max_train_samples 200 `
  --max_eval_samples 100 `
  --num_train_epochs 1 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 8
```

Add `--fp16` on a CUDA GPU. Use `--freeze_encoder` for the frozen SSL
baseline; without it, only the convolutional feature encoder is frozen.
Evaluation writes JSONL predictions and appends a row to
`results/metrics.csv`.

## 4. Frozen 10h baseline (E1)

After the debug run succeeds, launch the first formal experiment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_wav2vec2_frozen_10h.ps1
```

The corresponding reproducibility record is
`configs/wav2vec2_frozen_10h.yaml`. The script freezes every parameter except
the CTC projection head.

## 5. Fine-tuned 10h main experiment (E2)

E2 trains the Transformer encoder and CTC head while keeping only the
convolutional feature encoder frozen. Launch it in the `ssl_asr` environment:

```powershell
conda activate ssl_asr
powershell -ExecutionPolicy Bypass -File scripts/train_wav2vec2_finetune_10h.ps1
```

The script refuses to overwrite an existing
`exp/wav2vec2_finetune_10h` directory. Its settings are recorded in
`configs/wav2vec2_finetune_10h.yaml`. The corrected E2 schedule uses 30
epochs, a `1e-4` learning rate, 10% warmup, and weight decay 0.005; the
original 3-epoch schedule was insufficient for the randomly initialized CTC
head to leave its collapsed-output phase.

## 6. Fine-tuned 1h low-resource experiment (E3)

The original E3 collapsed because its nominal 1h manifest contained only
0.591h after the 15-second training filter. Generate a nested manifest whose
duration is accumulated after filtering:

```powershell
python scripts/prepare_librispeech_subsets.py --local_only --splits train-clean-100 --target_hours 1 --max_train_duration_in_seconds 15 --subset_suffix effective_15s
```

The repaired E3 trains on one effective hour. It uses a `1e-4` CTC-head
learning rate, a `2e-5` encoder learning rate, delays encoder updates for 100
steps, disables masking, reloads the best dev-WER checkpoint, and stops after
eight consecutive near-empty evaluations:

```powershell
conda activate ssl_asr
powershell -ExecutionPolicy Bypass -File scripts/train_wav2vec2_finetune_1h.ps1
```

The script refuses to overwrite `exp/wav2vec2_finetune_1h_repaired`, so the
collapsed run is retained for diagnosis. Its complete
settings are in `configs/wav2vec2_finetune_1h.yaml`.

## 7. WavLM 10h model comparison (E4)

E4 replaces only the self-supervised encoder with `microsoft/wavlm-base`.
It otherwise matches corrected E2: the 10h subset, 15-second duration limit,
CNN-only freeze, 30 epochs, and the same optimizer and memory settings.

```powershell
conda activate ssl_asr
powershell -ExecutionPolicy Bypass -File scripts/train_wavlm_finetune_10h.ps1
```

## 8. Remaining-GPU experiment queue (E5-E9 + E3r)

Generate the effective 3h subset used by E5:

```powershell
python scripts/prepare_librispeech_subsets.py --local_only --splits train-clean-100 --target_hours 3 --max_train_duration_in_seconds 15 --subset_suffix effective_15s
```

On the RTX 3090 server, the default command puts all new experiments first,
then runs repaired E3:

```text
E5 → E6a → E6b → E7 → E8 → E9 → E3r
```

```bash
bash scripts/train_new_experiments_rtx3090.sh
```

Experiments can also be selected explicitly:

```bash
bash scripts/train_new_experiments_rtx3090.sh e5
bash scripts/train_new_experiments_rtx3090.sh e6a e6b
bash scripts/train_new_experiments_rtx3090.sh e7
bash scripts/train_new_experiments_rtx3090.sh e8
bash scripts/train_new_experiments_rtx3090.sh e9
bash scripts/train_new_experiments_rtx3090.sh e3r
```

E6a/E6b freeze the bottom 6/9 Transformer blocks. E7 uses HuBERT-base,
E8 adds feature masking to the default time masking, and E9 trains a
two-layer BiLSTM CTC head over a frozen Wav2Vec2 encoder. E9 intentionally
uses FP32 because the CUDA FP16 LSTM startup check produced NaN loss.
All launchers refuse to overwrite an existing output, prediction, or log.

The first run downloads `microsoft/wavlm-base`. The script refuses to
overwrite `exp/wavlm_finetune_10h`; reproducibility settings are in
`configs/wavlm_finetune_10h.yaml`.

## 8. RTX 3090 Linux migration

After activating the `ssl_asr` environment on a 24GB RTX 3090 server, the
script now starts at E4 by default because E2 and E3 are complete:

```bash
bash scripts/train_e2_e4_rtx3090.sh
```

To explicitly rerun from E2 or E3:

```bash
bash scripts/train_e2_e4_rtx3090.sh e2
bash scripts/train_e2_e4_rtx3090.sh e3
```

The script uses train/eval batch size 2 with gradient accumulation 8 for
E2/E3. E4 uses batch size 1 with gradient accumulation 16 because WavLM and
the native-CUDA convolution fallback have a higher memory peak. All three
retain an effective batch size of 16 and use zero data-loader workers. The
script enables expandable CUDA allocator segments, writes one log per
experiment below `logs/`, and
refuses to overwrite existing experiment directories or prediction files.
Migrated Windows manifest paths are automatically relocated below the current
project's `data/` directory. If audio is elsewhere on the server, set
`DATA_ROOT=/path/to/data` when launching the script.
The server script also runs a real cuDNN Conv1d preflight. If the rented
host's cuDNN runtime cannot initialize, it automatically passes
`--disable_cudnn` and verifies the native CUDA Conv1d fallback before training.

For a clean RTX 3090 environment, install the official PyTorch CUDA wheel
first, then use `requirements-rtx3090.txt`. Do not install the
`datasets[audio]` extra in this environment: current major versions can pull
TorchCodec and replace the selected PyTorch CUDA build.
