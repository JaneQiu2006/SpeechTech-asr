"""ASR metrics and result persistence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence


def compute_error_rates(
    references: Sequence[str], predictions: Sequence[str]
) -> dict[str, float]:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have equal lengths")
    if not references:
        raise ValueError("cannot compute metrics for an empty input")
    try:
        from jiwer import cer, wer
    except ImportError as error:
        raise RuntimeError("Install jiwer to compute WER/CER") from error
    return {
        "wer": float(wer(references, predictions)),
        "cer": float(cer(references, predictions)),
    }


def save_predictions(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_metrics(path: str | Path, row: dict) -> None:
    """Append a row while preserving a stable, extensible CSV schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "split",
        "model",
        "freeze_encoder",
        "num_samples",
        "wer",
        "cer",
        "total_params",
        "trainable_params",
    ]
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)

