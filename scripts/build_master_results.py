#!/usr/bin/env python
"""Build one split-explicit, de-duplicated ASR metrics table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_asr.metrics import compute_error_rates
from ssl_asr.text import normalize_text

DEFAULT_INPUTS = (
    PROJECT_ROOT / "results/metrics.csv",
    PROJECT_ROOT / "results/test_metrics.csv",
    PROJECT_ROOT / "results/representation_test_metrics.csv",
    PROJECT_ROOT / "results/deep_dive_metrics.csv",
)

PREFERRED_FIELDS = (
    "experiment_id",
    "experiment",
    "split",
    "seed",
    "model",
    "representation",
    "source_layer",
    "head_type",
    "codebook_size",
    "tokenizer_mode",
    "ctc_vocab_size",
    "effective_hours",
    "num_samples",
    "audio_hours",
    "wer",
    "cer",
    "substitutions",
    "deletions",
    "insertions",
    "exact_match",
    "total_params",
    "trainable_params",
    "trainable_ratio",
    "rtf",
    "unit_bitrate_bps",
    "unit_dedup_bitrate_bps",
    "prediction_correction",
    "source_files",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", dest="inputs")
    parser.add_argument(
        "--prediction_glob",
        default="results/predictions/**/*.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/master_metrics.csv",
    )
    return parser.parse_args()


def clean(row: dict) -> dict[str, str]:
    return {
        str(key): str(value).strip()
        for key, value in row.items()
        if key is not None and value is not None and str(value).strip() != ""
    }


def normalize_split(value: str) -> str:
    normalized = value.lower().replace("-", "_")
    aliases = {
        "dev": "dev_clean",
        "dev_clean": "dev_clean",
        "test": "test_clean",
        "test_clean": "test_clean",
    }
    return aliases.get(normalized, normalized)


def summary_for(experiment: str) -> tuple[dict, Path | None]:
    for path in (
        PROJECT_ROOT / "exp/deep_dive" / experiment / "summary.json",
        PROJECT_ROOT / "exp" / experiment / "summary.json",
    ):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8")), path
    return {}, None


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("experiment", ""),
        normalize_split(row.get("split", "")),
        row.get("seed", ""),
    )


def merge(rows: dict, incoming: dict[str, str], source: Path) -> None:
    incoming = clean(incoming)
    if "split" in incoming:
        incoming["split"] = normalize_split(incoming["split"])
    if not incoming.get("experiment"):
        return
    key = row_key(incoming)
    current = rows.setdefault(key, {})
    current.update(incoming)
    sources = set(filter(None, current.get("source_files", "").split(";")))
    sources.add(str(source.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    current["source_files"] = ";".join(sorted(sources))


def prediction_identity(path: Path) -> tuple[str, str] | None:
    stem = path.stem
    for suffix, split in (("_dev", "dev_clean"), ("_test", "test_clean")):
        if stem.endswith(suffix):
            return stem[: -len(suffix)], split
    return None


def corrected_predictions(rows: list[dict]) -> tuple[list[str], str]:
    predictions = [normalize_text(str(row.get("prediction", ""))) for row in rows]
    references = [normalize_text(str(row.get("reference", ""))) for row in rows]
    trailing_predictions = sum(item.endswith("'") for item in predictions)
    trailing_references = sum(item.endswith("'") for item in references)
    if (
        rows
        and trailing_predictions / len(rows) >= 0.9
        and trailing_references == 0
    ):
        return [item[:-1].rstrip() if item.endswith("'") else item for item in predictions], (
            "systematic_trailing_apostrophe_removed_for_metrics"
        )
    return predictions, ""


def prediction_metric_row(path: Path) -> dict[str, str] | None:
    identity = prediction_identity(path)
    if identity is None:
        return None
    experiment, split = identity
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        return None
    required = {"id", "reference", "prediction"}
    if not all(required.issubset(record) for record in records):
        return None
    references = [normalize_text(str(record["reference"])) for record in records]
    predictions, correction = corrected_predictions(records)
    rates = compute_error_rates(references, predictions)
    summary, summary_path = summary_for(experiment)
    row = {
        "experiment_id": summary.get("experiment_id", ""),
        "experiment": experiment,
        "split": split,
        "seed": summary.get("seed", ""),
        "num_samples": len(records),
        "audio_hours": sum(float(record.get("duration", 0)) for record in records) / 3600,
        **rates,
        "prediction_correction": correction,
    }
    mappings = {
        "source_model": "model",
        "representation": "representation",
        "source_layer": "source_layer",
        "head_type": "head_type",
        "codebook_size": "codebook_size",
        "tokenizer_mode": "tokenizer_mode",
        "ctc_vocab_size": "ctc_vocab_size",
        "trainable_params": "trainable_params",
        "total_params": "total_params",
        "total_system_params": "total_params",
        "trainable_ratio": "trainable_ratio",
    }
    for source_name, target_name in mappings.items():
        if summary.get(source_name) not in (None, ""):
            row[target_name] = summary[source_name]
    if summary_path:
        row["_summary_path"] = str(summary_path)
    return clean(row)


def merge_prediction(rows: dict, metric: dict[str, str], source: Path) -> None:
    # A prediction is authoritative for recognition metrics. Enrich the closest
    # CSV row even when an older CSV omitted its seed.
    candidates = [
        key
        for key in rows
        if key[0] == metric["experiment"] and key[1] == metric["split"]
    ]
    key = row_key(metric)
    if key not in rows and len(candidates) == 1:
        key = candidates[0]
        metric["seed"] = key[2]
    current = rows.setdefault(key, {})
    current.update(metric)
    sources = set(filter(None, current.get("source_files", "").split(";")))
    sources.add(str(source.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    current["source_files"] = ";".join(sorted(sources))


def atomic_write(path: Path, rows: list[dict[str, str]]) -> None:
    all_fields = {field for row in rows for field in row if not field.startswith("_")}
    fields = [field for field in PREFERRED_FIELDS if field in all_fields]
    fields.extend(sorted(all_fields.difference(fields)))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    inputs = args.inputs or list(DEFAULT_INPUTS)
    merged: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in inputs:
        if not path.is_file() or path.stat().st_size == 0:
            print(f"[skip] missing metrics: {path}")
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                merge(merged, row, path)
    prediction_paths = sorted(PROJECT_ROOT.glob(args.prediction_glob))
    for path in prediction_paths:
        metric = prediction_metric_row(path)
        if metric is not None:
            merge_prediction(merged, metric, path)
    rows = sorted(
        merged.values(),
        key=lambda row: (
            row.get("split", ""),
            float(row.get("wer", "inf")),
            row.get("experiment", ""),
            row.get("seed", ""),
        ),
    )
    atomic_write(args.output, rows)
    print(
        f"master_rows={len(rows)} predictions_scanned={len(prediction_paths)} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
