"""ASR metrics and result persistence."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Write JSON atomically so interrupted jobs never leave a partial summary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def compute_error_rates(
    references: Sequence[str], predictions: Sequence[str]
) -> dict[str, float]:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have equal lengths")
    if not references:
        raise ValueError("cannot compute metrics for an empty input")
    try:
        from jiwer import cer, process_words, wer
    except ImportError as error:
        raise RuntimeError("Install jiwer to compute WER/CER") from error
    alignment = process_words(references, predictions)
    exact = sum(reference == prediction for reference, prediction in zip(references, predictions))
    return {
        "wer": float(wer(references, predictions)),
        "cer": float(cer(references, predictions)),
        "substitutions": int(alignment.substitutions),
        "deletions": int(alignment.deletions),
        "insertions": int(alignment.insertions),
        "exact_match": int(exact),
        "exact_match_rate": exact / len(references),
    }


def save_predictions(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def append_metrics(path: str | Path, row: dict) -> None:
    """Atomically upsert a row while allowing the CSV schema to grow."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    fields: list[str] = []
    if path.exists() and path.stat().st_size:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
    key_fields = ("experiment_id", "experiment", "split", "seed")
    active_keys = [key for key in key_fields if key in row]
    rows = [
        old
        for old in rows
        if not active_keys
        or any(str(old.get(key, "")) != str(row.get(key, "")) for key in active_keys)
    ]
    rows.append(row)
    preferred = [
        "experiment_id", "experiment", "split", "model", "representation",
        "source_layer", "head_type", "seed", "num_samples", "effective_hours",
        "wer", "cer", "substitutions", "deletions", "insertions", "exact_match",
        "total_params", "trainable_params", "trainable_ratio", "rtf",
    ]
    all_fields = set(fields)
    for item in rows:
        all_fields.update(item)
    fields = [name for name in preferred if name in all_fields]
    fields.extend(sorted(all_fields.difference(fields)))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
