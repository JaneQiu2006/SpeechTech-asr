#!/usr/bin/env python
"""End-to-end test evaluation for cached-head continuous/discrete CTC systems."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_asr.manifest import read_jsonl
from ssl_asr.metrics import append_metrics, compute_error_rates, save_predictions
from ssl_asr.representations import (
    CachedBiLSTMCTC,
    resolve_audio_path,
    unit_stream_statistics,
)
from ssl_asr.text import normalize_text

FIELDS = (
    "experiment",
    "split",
    "representation",
    "source_model",
    "source_layer",
    "codebook_size",
    "num_samples",
    "audio_hours",
    "wer",
    "cer",
    "encoder_params",
    "head_params",
    "trainable_params",
    "inference_seconds",
    "rtf",
    "feature_rate",
    "continuous_bitrate_bps",
    "ctc_token_rate",
    "ctc_token_entropy_bits",
    "ctc_label_bitrate_bps",
    "unit_used_types",
    "unit_utilization",
    "unit_perplexity",
    "unit_token_rate",
    "unit_entropy_bits",
    "unit_bitrate_bps",
    "unit_fixed_bitrate_bps",
    "unit_dedup_token_rate",
    "unit_dedup_bitrate_bps",
    "device",
    "batch_size",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--source_model", default="facebook/wav2vec2-base")
    parser.add_argument("--centers", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/test_clean.jsonl",
    )
    parser.add_argument(
        "--vocab_path",
        type=Path,
        default=PROJECT_ROOT / "data/vocab/vocab.json",
    )
    parser.add_argument("--data_root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--prediction_path", type=Path, required=True)
    parser.add_argument(
        "--metrics_path",
        type=Path,
        default=PROJECT_ROOT / "results/representation_test_metrics.csv",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--disable_cudnn", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_tokenizer(vocab_path: Path):
    from transformers import Wav2Vec2CTCTokenizer

    return Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )


def read_metric_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def upsert_metric(path: Path, row: dict) -> None:
    append_metrics(path, row)


def batches(rows: list[dict], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def ctc_label_statistics(
    predictions: list[str], tokenizer, audio_seconds: float
) -> dict[str, float]:
    counts: Counter[int] = Counter()
    for prediction in predictions:
        counts.update(
            tokenizer(
                normalize_text(prediction), add_special_tokens=False
            ).input_ids
        )
    total = sum(counts.values())
    entropy = (
        -sum(
            (count / total) * math.log2(count / total)
            for count in counts.values()
        )
        if total
        else 0.0
    )
    rate = total / audio_seconds
    return {
        "ctc_token_rate": rate,
        "ctc_token_entropy_bits": entropy,
        "ctc_label_bitrate_bps": rate * entropy,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    existing = read_metric_rows(args.metrics_path)
    if (
        not args.overwrite
        and any(
            row.get("experiment") == args.experiment
            and row.get("split") == "test_clean"
            and row.get("wer")
            for row in existing
        )
    ):
        print(f"[{args.experiment}] complete metrics exist; skipping", flush=True)
        return

    records = read_jsonl(args.manifest)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max_samples must be positive")
        records = records[: args.max_samples]
    if not records:
        raise RuntimeError("Evaluation manifest is empty")
    if args.prediction_path.exists() and not args.overwrite:
        prediction_rows = [
            json.loads(line)
            for line in args.prediction_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        required = {"id", "reference", "prediction", "duration", "audio_path"}
        if (
            len(prediction_rows) == len(records)
            and all(required.issubset(row) for row in prediction_rows)
        ):
            tokenizer = build_tokenizer(args.vocab_path)
            references = [normalize_text(str(row["reference"])) for row in prediction_rows]
            predictions = [normalize_text(str(row["prediction"])) for row in prediction_rows]
            rates = compute_error_rates(references, predictions)
            audio_seconds = sum(float(row["duration"]) for row in prediction_rows)
            config = json.loads(
                (args.model_dir / "config.json").read_text(encoding="utf-8")
            )
            upsert_metric(
                args.metrics_path,
                {
                    "experiment_id": args.experiment.split("_", 1)[0].upper(),
                    "experiment": args.experiment,
                    "split": "test_clean",
                    "representation": config.get("representation", ""),
                    "source_model": config.get("source_model", args.source_model),
                    "source_layer": config.get("source_layer", ""),
                    "codebook_size": config.get("codebook_size") or "",
                    "num_samples": len(prediction_rows),
                    "audio_hours": audio_seconds / 3600,
                    **rates,
                    **ctc_label_statistics(predictions, tokenizer, audio_seconds),
                    "recovered_from_predictions": True,
                },
            )
            print(
                f"[{args.experiment}] recovered test metrics from existing "
                f"predictions; WER={rates['wer']:.5f}, CER={rates['cer']:.5f}",
                flush=True,
            )
            return
        print(
            f"[{args.experiment}] incomplete prediction file; re-running test",
            flush=True,
        )
        args.overwrite = True

    import soundfile as sf
    from transformers import AutoFeatureExtractor, AutoModel

    tokenizer = build_tokenizer(args.vocab_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.source_model)
    encoder = AutoModel.from_pretrained(args.source_model).to(args.device)
    encoder.eval()
    head = CachedBiLSTMCTC.from_pretrained(args.model_dir).to(args.device)
    head.eval()
    if head.config.source_model != args.source_model:
        raise ValueError("Head source model does not match --source_model")
    centers = None
    if args.centers:
        centers = torch.from_numpy(
            np.load(args.centers, allow_pickle=False).astype(np.float32)
        ).to(args.device)
        if not head.config.representation.startswith("discrete"):
            raise ValueError("Centers were supplied for a continuous head")
        if head.config.codebook_size != len(centers):
            raise ValueError("Head and centers use different codebook sizes")
    elif head.config.representation.startswith("discrete"):
        raise ValueError("A discrete head requires --centers")

    use_fp16 = args.device.startswith("cuda") and not args.no_fp16
    prediction_rows = []
    references: list[str] = []
    predictions: list[str] = []
    unit_sequences: list[np.ndarray] = []
    total_frames = 0
    inference_seconds = 0.0
    for batch_index, batch in enumerate(batches(records, args.batch_size), start=1):
        audio_arrays = []
        local_paths = []
        for row in batch:
            path = resolve_audio_path(str(row["audio_path"]), args.data_root)
            audio, sample_rate = sf.read(path, dtype="float32")
            if sample_rate != 16_000:
                raise ValueError(f"{row['id']}: expected 16 kHz, got {sample_rate}")
            if audio.ndim != 1:
                audio = np.mean(audio, axis=1, dtype=np.float32)
            audio_arrays.append(audio)
            local_paths.append(path.resolve())
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=16_000,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        inputs = {
            name: value.to(args.device, non_blocking=True)
            for name, value in inputs.items()
        }
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_fp16
        ):
            outputs = encoder(**inputs, output_hidden_states=True)
        lengths = encoder._get_feat_extract_output_lengths(
            inputs["attention_mask"].sum(dim=-1)
        ).cpu()
        feature_sequences = []
        for item_index, length in enumerate(lengths):
            if head.config.layer_indices is not None:
                features = torch.stack(
                    [
                        outputs.hidden_states[layer][item_index, : int(length)].float()
                        for layer in head.config.layer_indices
                    ],
                    dim=1,
                )
            else:
                features = outputs.hidden_states[head.config.source_layer][
                    item_index, : int(length)
                ].float()
            total_frames += int(length)
            if centers is not None:
                distances = torch.cdist(features, centers)
                unit_ids = torch.argmin(distances, dim=-1)
                unit_sequences.append(unit_ids.detach().cpu().numpy())
                features = (
                    unit_ids
                    if head.config.unit_embedding_dim is not None
                    else centers[unit_ids]
                )
            feature_sequences.append(features)
        padded_features = torch.nn.utils.rnn.pad_sequence(
            feature_sequences, batch_first=True
        )
        with torch.inference_mode():
            logits = head(padded_features, lengths)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - start
        prediction_ids = torch.argmax(logits, dim=-1).cpu()
        decoded = tokenizer.batch_decode(
            [
                token_ids[: int(length)]
                for token_ids, length in zip(prediction_ids, lengths)
            ]
        )
        for row, local_path, prediction in zip(batch, local_paths, decoded):
            reference = normalize_text(str(row["text"]))
            prediction = normalize_text(prediction)
            references.append(reference)
            predictions.append(prediction)
            prediction_rows.append(
                {
                    "id": str(row["id"]),
                    "reference": reference,
                    "prediction": prediction,
                    "audio_path": str(local_path),
                    "duration": float(row["duration"]),
                }
            )
        if batch_index % 100 == 0:
            print(
                f"[{args.experiment}] {len(prediction_rows)}/{len(records)}",
                flush=True,
            )

    save_predictions(args.prediction_path, prediction_rows)
    rates = compute_error_rates(references, predictions)
    audio_seconds = sum(float(row["duration"]) for row in records)
    feature_rate = total_frames / audio_seconds
    encoder_params = sum(parameter.numel() for parameter in encoder.parameters())
    head_params = sum(parameter.numel() for parameter in head.parameters())
    metric = {
        "experiment_id": args.experiment.split("_", 1)[0].upper(),
        "experiment": args.experiment,
        "split": "test_clean",
        "representation": head.config.representation,
        "source_model": args.source_model,
        "source_layer": head.config.source_layer,
        "codebook_size": head.config.codebook_size or "",
        "num_samples": len(records),
        "audio_hours": audio_seconds / 3600,
        **rates,
        "encoder_params": encoder_params,
        "head_params": head_params,
        "total_params": encoder_params + head_params,
        "trainable_params": head_params,
        "trainable_ratio": head_params / (encoder_params + head_params),
        "inference_seconds": inference_seconds,
        "rtf": inference_seconds / audio_seconds,
        "feature_rate": feature_rate,
        "continuous_bitrate_bps": (
            feature_rate * head.config.input_size * 16
            if centers is None
            else ""
        ),
        **ctc_label_statistics(predictions, tokenizer, audio_seconds),
        "device": args.device,
        "batch_size": args.batch_size,
    }
    if centers is not None:
        metric.update(
            unit_stream_statistics(
                unit_sequences, audio_seconds, len(centers)
            )
        )
    upsert_metric(args.metrics_path, metric)
    print(
        f"[{args.experiment}] WER={rates['wer']:.5f}, "
        f"CER={rates['cer']:.5f}, RTF={inference_seconds / audio_seconds:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
