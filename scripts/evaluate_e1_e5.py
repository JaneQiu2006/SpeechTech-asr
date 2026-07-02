#!/usr/bin/env python
"""Evaluate the completed E1-E5 checkpoints on local LibriSpeech test-clean."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_asr.manifest import read_jsonl
from ssl_asr.metrics import append_metrics, compute_error_rates
from ssl_asr.text import normalize_text


@dataclass(frozen=True)
class Experiment:
    id: str
    name: str
    model_dir: Path
    processor_dir: Path
    freeze_encoder: bool
    model_class: str = "auto"
    freeze_transformer_layers: int = 0


EXPERIMENTS = (
    Experiment(
        "E1",
        "wav2vec2_frozen_10h_fair_30ep",
        PROJECT_ROOT / "exp/wav2vec2_frozen_10h_fair_30ep",
        PROJECT_ROOT / "exp/wav2vec2_finetune_10h",
        True,
    ),
    Experiment(
        "E2",
        "wav2vec2_finetune_10h",
        PROJECT_ROOT / "exp/wav2vec2_finetune_10h",
        PROJECT_ROOT / "exp/wav2vec2_finetune_10h",
        False,
    ),
    Experiment(
        "E3",
        "wav2vec2_finetune_1h_repaired",
        PROJECT_ROOT / "exp/wav2vec2_finetune_1h_repaired",
        PROJECT_ROOT / "exp/wav2vec2_finetune_1h_repaired",
        False,
    ),
    Experiment(
        "E4",
        "wavlm_finetune_10h",
        PROJECT_ROOT / "exp/wavlm_finetune_10h",
        PROJECT_ROOT / "exp/wavlm_finetune_10h",
        False,
    ),
    Experiment(
        "E5",
        "wav2vec2_finetune_3h",
        PROJECT_ROOT / "exp/wav2vec2_finetune_3h",
        PROJECT_ROOT / "exp/wav2vec2_finetune_10h",
        False,
    ),
    Experiment(
        "E6A",
        "wav2vec2_top6_finetune_10h",
        PROJECT_ROOT / "exp/wav2vec2_top6_finetune_10h",
        PROJECT_ROOT / "exp/wav2vec2_finetune_10h",
        False,
        freeze_transformer_layers=6,
    ),
    Experiment(
        "E6B",
        "wav2vec2_top3_finetune_10h",
        PROJECT_ROOT / "exp/wav2vec2_top3_finetune_10h",
        PROJECT_ROOT / "exp/wav2vec2_finetune_10h",
        False,
        freeze_transformer_layers=9,
    ),
    Experiment(
        "E8",
        "wav2vec2_masking_finetune_10h",
        PROJECT_ROOT / "exp/wav2vec2_masking_finetune_10h",
        PROJECT_ROOT / "exp/wav2vec2_finetune_10h",
        False,
    ),
    Experiment(
        "E9",
        "wav2vec2_frozen_bilstm_10h",
        PROJECT_ROOT / "exp/wav2vec2_frozen_bilstm_10h",
        PROJECT_ROOT / "exp/wav2vec2_finetune_10h",
        True,
        "wav2vec2_bilstm",
    ),
    Experiment(
        "E12A",
        "wav2vec2_finetune_1h_time_mask",
        PROJECT_ROOT / "exp/wav2vec2_finetune_1h_time_mask",
        PROJECT_ROOT / "exp/wav2vec2_finetune_1h_repaired",
        False,
    ),
)

METRIC_FIELDS = (
    "experiment_id",
    "experiment",
    "split",
    "checkpoint",
    "num_samples",
    "audio_hours",
    "wer",
    "cer",
    "total_params",
    "trainable_params",
    "inference_seconds",
    "rtf",
    "device",
    "batch_size",
    "ctc_vocab_size",
    "ctc_used_token_types",
    "ctc_vocab_utilization",
    "ctc_output_tokens",
    "ctc_token_rate",
    "ctc_token_throughput",
    "ctc_token_entropy_bits",
    "ctc_label_bitrate_bps",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["E1", "E2", "E3", "E4", "E5"],
        help="Experiment IDs to evaluate (default: E1 E2 E3 E4 E5).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/test_clean.jsonl",
    )
    parser.add_argument("--data_root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        default=PROJECT_ROOT / "results/predictions",
    )
    parser.add_argument(
        "--metrics_path",
        type=Path,
        default=PROJECT_ROOT / "results/test_metrics.csv",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--disable_cudnn",
        action="store_true",
        help="Use native CUDA kernels if the host cuDNN runtime is broken.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-evaluate even if a complete test result already exists.",
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="Disable CUDA FP16 inference.",
    )
    parser.add_argument("--custom_experiment_id")
    parser.add_argument("--custom_experiment_name")
    parser.add_argument("--custom_model_dir", type=Path)
    parser.add_argument("--custom_processor_dir", type=Path)
    parser.add_argument("--custom_freeze_transformer_layers", type=int, default=0)
    return parser.parse_args()


def has_model_weights(path: Path) -> bool:
    return (path / "model.safetensors").is_file() or (
        path / "pytorch_model.bin"
    ).is_file()


def resolve_model_dir(path: Path) -> Path:
    """Use a saved final model, or the numerically latest complete checkpoint."""
    if has_model_weights(path) and (path / "config.json").is_file():
        return path
    checkpoints: list[tuple[int, Path]] = []
    if path.is_dir():
        for candidate in path.glob("checkpoint-*"):
            try:
                step = int(candidate.name.removeprefix("checkpoint-"))
            except ValueError:
                continue
            if has_model_weights(candidate) and (candidate / "config.json").is_file():
                checkpoints.append((step, candidate))
    if not checkpoints:
        raise FileNotFoundError(f"No complete model or checkpoint found below {path}")
    return max(checkpoints)[1]


def relocate_audio_path(audio_path: str, data_root: Path) -> Path:
    path = Path(audio_path)
    if path.is_file():
        return path
    parts = audio_path.replace("\\", "/").split("/")
    for split in ("train-clean-100", "dev-clean", "test-clean"):
        if split in parts:
            index = len(parts) - 1 - parts[::-1].index(split)
            candidate = data_root / Path(*parts[index:])
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Audio file not found locally: {audio_path}")


def read_metrics(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def has_complete_metric(rows: Iterable[dict[str, str]], experiment_id: str) -> bool:
    return any(
        row.get("experiment_id", "").upper() == experiment_id.upper()
        and row.get("split") == "test_clean"
        and bool(row.get("wer"))
        and bool(row.get("cer"))
        and bool(row.get("ctc_token_rate"))
        and bool(row.get("ctc_label_bitrate_bps"))
        for row in rows
    )


def upsert_metric(path: Path, row: dict[str, object]) -> None:
    append_metrics(path, row)


def load_complete_predictions(path: Path, expected: int) -> list[dict] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != expected:
        print(
            f"[resume] ignoring incomplete {path}: {len(rows)}/{expected} records",
            flush=True,
        )
        return None
    required = {"id", "reference", "prediction", "audio_path", "duration"}
    if any(not required.issubset(row) for row in rows):
        print(f"[resume] ignoring malformed prediction file: {path}", flush=True)
        return None
    return rows


def write_predictions(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def parameter_counts(model, experiment: Experiment) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    if experiment.freeze_encoder:
        trained = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("lm_head.")
        )
    else:
        frozen_layer_parameter_ids: set[int] = set()
        if experiment.freeze_transformer_layers:
            base_model = getattr(model, model.base_model_prefix)
            layers = list(base_model.encoder.layers)
            if experiment.freeze_transformer_layers > len(layers):
                raise ValueError(
                    f"{experiment.id}: cannot freeze "
                    f"{experiment.freeze_transformer_layers}/{len(layers)} layers"
                )
            frozen_layer_parameter_ids = {
                id(parameter)
                for layer in layers[: experiment.freeze_transformer_layers]
                for parameter in layer.parameters()
            }
        trained = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if ".feature_extractor." not in name
            and id(parameter) not in frozen_layer_parameter_ids
        )
    return total, trained


def batches(rows: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def compute_ctc_label_statistics(
    prediction_rows: list[dict],
    tokenizer,
    vocab_size: int,
    inference_seconds: float | None,
) -> dict[str, float | int | str]:
    """Measure the decoded CTC label stream, not an acoustic-unit codebook."""
    token_counts: Counter[int] = Counter()
    for row in prediction_rows:
        token_ids = tokenizer(
            normalize_text(str(row["prediction"])),
            add_special_tokens=False,
        ).input_ids
        token_counts.update(int(token_id) for token_id in token_ids)

    output_tokens = sum(token_counts.values())
    audio_seconds = sum(float(row["duration"]) for row in prediction_rows)
    entropy = 0.0
    if output_tokens:
        entropy = -sum(
            (count / output_tokens) * math.log2(count / output_tokens)
            for count in token_counts.values()
        )
    token_rate = output_tokens / audio_seconds
    throughput: float | str = ""
    if inference_seconds is not None and inference_seconds > 0:
        throughput = output_tokens / inference_seconds
    return {
        "ctc_vocab_size": vocab_size,
        "ctc_used_token_types": len(token_counts),
        "ctc_vocab_utilization": len(token_counts) / vocab_size,
        "ctc_output_tokens": output_tokens,
        "ctc_token_rate": token_rate,
        "ctc_token_throughput": throughput,
        "ctc_token_entropy_bits": entropy,
        "ctc_label_bitrate_bps": token_rate * entropy,
    }


def evaluate(
    experiment: Experiment,
    records: list[dict],
    args: argparse.Namespace,
) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    model_dir = resolve_model_dir(experiment.model_dir)
    if not experiment.processor_dir.is_dir():
        raise FileNotFoundError(f"Processor directory not found: {experiment.processor_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("[startup] cuDNN disabled; using native CUDA kernels", flush=True)

    print(f"[{experiment.id}] checkpoint={model_dir}", flush=True)
    processor = AutoProcessor.from_pretrained(
        experiment.processor_dir, local_files_only=True
    )
    if experiment.model_class == "wav2vec2_bilstm":
        from ssl_asr.models import Wav2Vec2BiLSTMForCTC

        model = Wav2Vec2BiLSTMForCTC.from_pretrained(
            model_dir, local_files_only=True
        )
    else:
        model = AutoModelForCTC.from_pretrained(
            model_dir, local_files_only=True
        )
    model.to(args.device)
    model.eval()
    total_params, trainable_params = parameter_counts(model, experiment)

    prediction_rows: list[dict] = []
    inference_seconds = 0.0
    use_fp16 = args.device.startswith("cuda") and not args.no_fp16
    for batch_index, batch in enumerate(batches(records, args.batch_size), start=1):
        audio_arrays = []
        for record in batch:
            audio, sampling_rate = sf.read(
                relocate_audio_path(str(record["audio_path"]), args.data_root),
                dtype="float32",
            )
            if sampling_rate != 16_000:
                raise ValueError(
                    f"{record['id']}: expected 16 kHz audio, got {sampling_rate} Hz"
                )
            if audio.ndim != 1:
                audio = np.mean(audio, axis=1, dtype=np.float32)
            audio_arrays.append(audio)
        inputs = processor(
            audio_arrays,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        inputs = {
            name: value.to(args.device, non_blocking=True)
            for name, value in inputs.items()
        }
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_fp16,
        ):
            logits = model(**inputs).logits
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - start
        prediction_ids = torch.argmax(logits, dim=-1)
        output_lengths = model._get_feat_extract_output_lengths(
            inputs["attention_mask"].sum(dim=-1)
        )
        trimmed_prediction_ids = [
            token_ids[: int(output_length)].cpu().numpy()
            for token_ids, output_length in zip(prediction_ids, output_lengths)
        ]
        predictions = processor.batch_decode(trimmed_prediction_ids)
        for record, prediction in zip(batch, predictions):
            prediction_rows.append(
                {
                    "id": record["id"],
                    "reference": normalize_text(str(record["text"])),
                    "prediction": normalize_text(prediction),
                    "audio_path": str(
                        relocate_audio_path(
                            str(record["audio_path"]), args.data_root
                        ).resolve()
                    ),
                    "duration": float(record["duration"]),
                }
            )
        if batch_index % 100 == 0:
            print(
                f"[{experiment.id}] evaluated "
                f"{len(prediction_rows)}/{len(records)}",
                flush=True,
            )

    prediction_path = (
        args.predictions_dir / f"{experiment.name}_test.jsonl"
    )
    write_predictions(prediction_path, prediction_rows)
    rates = compute_error_rates(
        [row["reference"] for row in prediction_rows],
        [row["prediction"] for row in prediction_rows],
    )
    audio_seconds = sum(float(row["duration"]) for row in prediction_rows)
    ctc_statistics = compute_ctc_label_statistics(
        prediction_rows,
        processor.tokenizer,
        model.config.vocab_size,
        inference_seconds,
    )
    upsert_metric(
        args.metrics_path,
        {
            "experiment_id": experiment.id,
            "experiment": experiment.name,
            "split": "test_clean",
            "checkpoint": model_dir.relative_to(PROJECT_ROOT),
            "num_samples": len(prediction_rows),
            "audio_hours": audio_seconds / 3600,
            **rates,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "inference_seconds": inference_seconds,
            "rtf": inference_seconds / audio_seconds,
            "device": args.device,
            "batch_size": args.batch_size,
            **ctc_statistics,
        },
    )
    print(
        f"[{experiment.id}] WER={rates['wer']:.5f}, CER={rates['cer']:.5f}, "
        f"RTF={inference_seconds / audio_seconds:.5f}",
        flush=True,
    )
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()


def metric_from_predictions(
    experiment: Experiment,
    prediction_rows: list[dict],
    args: argparse.Namespace,
) -> None:
    from transformers import AutoConfig, AutoProcessor

    rates = compute_error_rates(
        [normalize_text(str(row["reference"])) for row in prediction_rows],
        [normalize_text(str(row["prediction"])) for row in prediction_rows],
    )
    audio_seconds = sum(float(row["duration"]) for row in prediction_rows)
    existing = next(
        (
            row
            for row in read_metrics(args.metrics_path)
            if row.get("experiment_id") == experiment.id
            and row.get("split") == "test_clean"
        ),
        {},
    )
    inference_seconds = (
        float(existing["inference_seconds"])
        if existing.get("inference_seconds")
        else None
    )
    processor = AutoProcessor.from_pretrained(
        experiment.processor_dir, local_files_only=True
    )
    model_dir = resolve_model_dir(experiment.model_dir)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    ctc_statistics = compute_ctc_label_statistics(
        prediction_rows,
        processor.tokenizer,
        config.vocab_size,
        inference_seconds,
    )
    updated_metric: dict[str, object] = dict(existing)
    updated_metric.update(
        {
            "experiment_id": experiment.id,
            "experiment": experiment.name,
            "split": "test_clean",
            "checkpoint": model_dir.relative_to(PROJECT_ROOT),
            "num_samples": len(prediction_rows),
            "audio_hours": audio_seconds / 3600,
            **rates,
            **ctc_statistics,
        }
    )
    upsert_metric(
        args.metrics_path,
        updated_metric,
    )
    print(
        f"[{experiment.id}] reused predictions: "
        f"WER={rates['wer']:.5f}, CER={rates['cer']:.5f}, "
        f"CTC token rate={ctc_statistics['ctc_token_rate']:.3f}/s, "
        f"label bitrate={ctc_statistics['ctc_label_bitrate_bps']:.3f} bit/s",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    custom_values = (
        args.custom_experiment_id,
        args.custom_experiment_name,
        args.custom_model_dir,
        args.custom_processor_dir,
    )
    if any(value is not None for value in custom_values):
        if not all(value is not None for value in custom_values):
            raise ValueError("All --custom_experiment_* arguments are required together")
        custom_model_dir = args.custom_model_dir
        custom_processor_dir = args.custom_processor_dir
        if not custom_model_dir.is_absolute():
            custom_model_dir = (PROJECT_ROOT / custom_model_dir).resolve()
        if not custom_processor_dir.is_absolute():
            custom_processor_dir = (PROJECT_ROOT / custom_processor_dir).resolve()
        experiments = (
            Experiment(
                str(args.custom_experiment_id).upper(),
                str(args.custom_experiment_name),
                custom_model_dir,
                custom_processor_dir,
                False,
                freeze_transformer_layers=args.custom_freeze_transformer_layers,
            ),
        )
        requested = {experiments[0].id}
    else:
        experiments = EXPERIMENTS
        requested = {item.upper() for item in args.experiments}
    known = {experiment.id for experiment in experiments}
    unknown = requested - known
    if unknown:
        raise ValueError(f"Unknown experiment IDs: {sorted(unknown)}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Test manifest not found: {args.manifest}")
    records = read_jsonl(args.manifest)
    if not records:
        raise RuntimeError(f"Test manifest is empty: {args.manifest}")
    print(
        f"[data] test-clean={len(records)} utterances, "
        f"{sum(float(row['duration']) for row in records) / 3600:.3f} hours",
        flush=True,
    )

    for experiment in experiments:
        if experiment.id not in requested:
            continue
        metric_rows = read_metrics(args.metrics_path)
        if not args.overwrite and has_complete_metric(metric_rows, experiment.id):
            print(f"[{experiment.id}] complete test metrics exist; skipping", flush=True)
            continue
        prediction_path = (
            args.predictions_dir / f"{experiment.name}_test.jsonl"
        )
        existing_predictions = load_complete_predictions(
            prediction_path, len(records)
        )
        if not args.overwrite and existing_predictions is not None:
            metric_from_predictions(experiment, existing_predictions, args)
            continue
        evaluate(experiment, records, args)


if __name__ == "__main__":
    main()
