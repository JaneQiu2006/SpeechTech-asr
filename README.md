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

E3 uses the nested 1h training subset with the same model and freezing policy
as corrected E2. Because only 184 utterances remain after the 15-second
filter, it trains for 1000 optimizer steps and evaluates every 100 steps:

```powershell
conda activate ssl_asr
powershell -ExecutionPolicy Bypass -File scripts/train_wav2vec2_finetune_1h.ps1
```

The script refuses to overwrite `exp/wav2vec2_finetune_1h`. Its complete
settings are in `configs/wav2vec2_finetune_1h.yaml`.

## 7. WavLM 10h model comparison (E4)

E4 replaces only the self-supervised encoder with `microsoft/wavlm-base`.
It otherwise matches corrected E2: the 10h subset, 15-second duration limit,
CNN-only freeze, 30 epochs, and the same optimizer and memory settings.

```powershell
conda activate ssl_asr
powershell -ExecutionPolicy Bypass -File scripts/train_wavlm_finetune_10h.ps1
```

The first run downloads `microsoft/wavlm-base`. The script refuses to
overwrite `exp/wavlm_finetune_10h`; reproducibility settings are in
`configs/wavlm_finetune_10h.yaml`.

## 8. RTX 3090 Linux migration

After activating the `ssl_asr` environment on a 24GB RTX 3090 server, run
E2 through E4 sequentially with an effective batch size of 16:

```bash
bash scripts/train_e2_e4_rtx3090.sh
```

To continue after a previously completed and inspected experiment, start at
E3 or E4:

```bash
bash scripts/train_e2_e4_rtx3090.sh e3
bash scripts/train_e2_e4_rtx3090.sh e4
```

The script uses train/eval batch size 4, gradient accumulation 4, and four
data-loader workers. It writes one log per experiment below `logs/` and
refuses to overwrite existing experiment directories or prediction files.
Migrated Windows manifest paths are automatically relocated below the current
project's `data/` directory. If audio is elsewhere on the server, set
`DATA_ROOT=/path/to/data` when launching the script.
