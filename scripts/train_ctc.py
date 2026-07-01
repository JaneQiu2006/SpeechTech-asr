#!/usr/bin/env python
"""Minimal manifest-based CTC fine-tuning for SSL speech encoders."""

from __future__ import annotations

import argparse
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_asr.manifest import read_jsonl
from ssl_asr.metrics import append_metrics, compute_error_rates, save_predictions
from ssl_asr.text import normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default="facebook/wav2vec2-base")
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Local data directory used to relocate manifest paths after migration.",
    )
    parser.add_argument("--vocab_path", type=Path, default=PROJECT_ROOT / "data/vocab/vocab.json")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prediction_path", type=Path)
    parser.add_argument("--metrics_path", type=Path, default=PROJECT_ROOT / "results/metrics.csv")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument(
        "--ctc_loss_reduction",
        choices=("mean", "sum"),
        default="mean",
    )
    parser.add_argument("--ctc_zero_infinity", action="store_true")
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument("--max_duration_in_seconds", type=float, default=20.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Fraction of optimizer steps used for linear learning-rate warmup.",
    )
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Positive values override num_train_epochs.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        help="Evaluate and save every N optimizer steps instead of every epoch.",
    )
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument(
        "--eval_accumulation_steps",
        type=int,
        default=10,
        help="Move evaluation predictions to CPU periodically to bound GPU memory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


class ManifestDataset:
    def __init__(self, records: list[dict], processor: Any):
        self.records = records
        self.processor = processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        try:
            import soundfile as sf
        except ImportError as error:
            raise RuntimeError("Install soundfile to load manifest audio") from error
        record = self.records[index]
        waveform, sample_rate = sf.read(record["audio_path"], dtype="float32")
        if getattr(waveform, "ndim", 1) > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != 16_000:
            raise ValueError(
                f"{record['audio_path']} is {sample_rate} Hz; startup expects 16 kHz"
            )
        inputs = self.processor(
            waveform, sampling_rate=sample_rate, return_attention_mask=True
        )
        labels = self.processor.tokenizer(normalize_text(record["text"])).input_ids
        return {
            "input_values": inputs.input_values[0],
            "attention_mask": inputs.attention_mask[0],
            "labels": labels,
            "record_index": index,
        }


@dataclass
class CTCDataCollator:
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        import torch

        input_features = [
            {
                "input_values": feature["input_values"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(
            input_features, padding=True, return_tensors="pt"
        )
        labels = self.processor.tokenizer.pad(
            label_features, padding=True, return_tensors="pt"
        )
        batch["labels"] = labels["input_ids"].masked_fill(
            labels["attention_mask"].ne(1), -100
        )
        return batch


LIBRISPEECH_SPLITS = ("train-clean-100", "dev-clean", "test-clean")


def resolve_audio_path(audio_path: str, data_root: Path) -> Path:
    """Resolve local, relative, or migrated Windows manifest audio paths."""
    raw_path = Path(audio_path).expanduser()
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.append(PROJECT_ROOT / raw_path)

    # Manifests are reproducibility artifacts and may be copied between hosts.
    # Preserve the split-relative suffix while replacing the old project root.
    normalized_parts = audio_path.replace("\\", "/").split("/")
    for split in LIBRISPEECH_SPLITS:
        if split in normalized_parts:
            split_index = len(normalized_parts) - 1 - normalized_parts[::-1].index(split)
            candidates.append(data_root / Path(*normalized_parts[split_index:]))
            break

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    expected = candidates[-1]
    raise FileNotFoundError(
        f"Audio file not found. Manifest path={audio_path!r}; "
        f"expected local path={str(expected)!r}. "
        "Copy LibriSpeech below --data_root or pass the correct --data_root."
    )


def load_records(
    path: Path,
    limit: int | None,
    max_duration: float,
    data_root: Path,
) -> list[dict]:
    records = [
        record
        for record in read_jsonl(path)
        if float(record["duration"]) <= max_duration
    ]
    if limit is not None:
        records = records[:limit]
    if not records:
        raise RuntimeError(f"No usable records in {path}")
    resolved_records = []
    for record in records:
        resolved_record = dict(record)
        resolved_record["audio_path"] = str(
            resolve_audio_path(str(record["audio_path"]), data_root)
        )
        resolved_records.append(resolved_record)
    return resolved_records


def main() -> None:
    args = parse_args()
    print("[startup] importing PyTorch and Transformers training dependencies...", flush=True)
    try:
        # On Windows, importing torch before the pyarrow dataset DLLs can make
        # Transformers.Trainer terminate with 0xC0000005 and no traceback.
        # datasets is already a project dependency, so initialize pyarrow first.
        import pyarrow.dataset  # noqa: F401
        import numpy as np
        import torch
        from transformers import (
            AutoFeatureExtractor,
            AutoModelForCTC,
            Trainer,
            TrainingArguments,
            Wav2Vec2CTCTokenizer,
            Wav2Vec2Processor,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError("Install the packages in requirements.txt first") from error

    print(
        f"[startup] dependencies ready: torch={torch.__version__}, "
        f"cuda_available={torch.cuda.is_available()}",
        flush=True,
    )
    set_seed(args.seed)
    print(f"[startup] loading tokenizer from {args.vocab_path}", flush=True)
    tokenizer = Wav2Vec2CTCTokenizer(
        str(args.vocab_path),
        bos_token=None,
        eos_token=None,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    print(
        f"[startup] loading pretrained checkpoint {args.model_name_or_path}; "
        "the first run downloads it from Hugging Face",
        flush=True,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)
    model = AutoModelForCTC.from_pretrained(
        args.model_name_or_path,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        ctc_loss_reduction=args.ctc_loss_reduction,
        ctc_zero_infinity=args.ctc_zero_infinity,
        ignore_mismatched_sizes=True,
    )

    if args.freeze_encoder:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True
    elif hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()

    named_parameters = list(model.named_parameters())
    feature_encoder_parameters = [
        parameter
        for name, parameter in named_parameters
        if ".feature_extractor." in name
    ]
    transformer_parameters = [
        parameter
        for name, parameter in named_parameters
        if ".encoder." in name
    ]
    if not feature_encoder_parameters:
        raise RuntimeError("Could not identify the CNN feature encoder parameters")
    if any(parameter.requires_grad for parameter in feature_encoder_parameters):
        raise RuntimeError("CNN feature encoder freeze check failed")
    if args.freeze_encoder:
        non_head_trainable = [
            name
            for name, parameter in named_parameters
            if parameter.requires_grad and not name.startswith("lm_head.")
        ]
        if non_head_trainable:
            raise RuntimeError(
                f"Frozen baseline has trainable non-head parameters: {non_head_trainable[:5]}"
            )
    elif not transformer_parameters or not any(
        parameter.requires_grad for parameter in transformer_parameters
    ):
        raise RuntimeError("Fine-tune mode does not have a trainable Transformer encoder")

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"parameters: total={total_params:,}, trainable={trainable_params:,} "
        f"({trainable_params / total_params:.2%}); "
        f"cnn_feature_encoder_frozen=True; "
        f"transformer_encoder_trainable={not args.freeze_encoder}",
        flush=True,
    )

    train_records = load_records(
        args.train_manifest,
        args.max_train_samples,
        args.max_duration_in_seconds,
        args.data_root,
    )
    eval_records = load_records(
        args.eval_manifest,
        args.max_eval_samples,
        args.max_duration_in_seconds,
        args.data_root,
    )
    train_dataset = ManifestDataset(train_records, processor)
    eval_dataset = ManifestDataset(eval_records, processor)
    print(
        f"[data] train={len(train_records)} utterances, "
        f"eval={len(eval_records)} utterances",
        flush=True,
    )

    def compute_metrics(prediction) -> dict[str, float]:
        prediction_ids = np.argmax(prediction.predictions, axis=-1)
        label_ids = prediction.label_ids.copy()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        predictions = processor.batch_decode(prediction_ids)
        references = processor.batch_decode(label_ids, group_tokens=False)
        return compute_error_rates(references, predictions)

    training_kwargs = dict(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=10,
        save_total_limit=args.save_total_limit,
        eval_accumulation_steps=args.eval_accumulation_steps,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )
    argument_names = inspect.signature(TrainingArguments).parameters
    eval_strategy = "steps" if args.eval_steps is not None else "epoch"
    if "eval_strategy" in argument_names:
        training_kwargs["eval_strategy"] = eval_strategy
    else:
        training_kwargs["evaluation_strategy"] = eval_strategy
    training_kwargs["save_strategy"] = eval_strategy
    if args.eval_steps is not None:
        if args.eval_steps <= 0:
            raise ValueError("--eval_steps must be positive")
        training_kwargs["eval_steps"] = args.eval_steps
        training_kwargs["save_steps"] = args.eval_steps
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CTCDataCollator(processor),
        compute_metrics=compute_metrics,
    )
    print("[train] starting Trainer.train()", flush=True)
    trainer.train()
    trainer.save_model()
    processor.save_pretrained(args.output_dir)

    output = trainer.predict(eval_dataset)
    prediction_ids = np.argmax(output.predictions, axis=-1)
    label_ids = output.label_ids.copy()
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    predictions = processor.batch_decode(prediction_ids)
    references = processor.batch_decode(label_ids, group_tokens=False)
    prediction_path = args.prediction_path or (
        PROJECT_ROOT / "results/predictions" / f"{args.output_dir.name}_dev.jsonl"
    )
    save_predictions(
        prediction_path,
        (
            {
                "id": record["id"],
                "reference": reference,
                "prediction": prediction,
                "audio_path": record["audio_path"],
                "duration": record["duration"],
            }
            for record, reference, prediction in zip(
                eval_records, references, predictions
            )
        ),
    )
    rates = compute_error_rates(references, predictions)
    append_metrics(
        args.metrics_path,
        {
            "experiment": args.output_dir.name,
            "split": args.eval_manifest.stem,
            "model": args.model_name_or_path,
            "freeze_encoder": args.freeze_encoder,
            "num_samples": len(eval_records),
            **rates,
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
    )
    print(f"evaluation: WER={rates['wer']:.4f}, CER={rates['cer']:.4f}")
    print(f"predictions: {prediction_path}")


if __name__ == "__main__":
    main()
