#!/usr/bin/env python
"""Paired utterance bootstrap confidence intervals for ASR prediction files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from scripts.analyze_deep_dive_errors import align_words
except ImportError:
    from analyze_deep_dive_errors import align_words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        action="append",
        nargs=3,
        metavar=("NAME", "SYSTEM_A_JSONL", "SYSTEM_B_JSONL"),
        required=True,
        help="May be repeated. Reported difference is WER(B)-WER(A).",
    )
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    return parser.parse_args()


def read_predictions(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"Prediction file is empty: {path}")
    return rows


def utterance_counts(row: dict) -> tuple[int, int]:
    reference = str(row["reference"]).split()
    prediction = str(row["prediction"]).split()
    alignment = align_words(reference, prediction)
    errors = sum(operation != "equal" for operation, _, _ in alignment)
    return errors, len(reference)


def paired_counts(path_a: Path, path_b: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows_a = read_predictions(path_a)
    rows_b = read_predictions(path_b)
    if len(rows_a) != len(rows_b):
        raise ValueError(f"Prediction lengths differ: {path_a} vs {path_b}")
    errors_a, errors_b, words = [], [], []
    for index, (row_a, row_b) in enumerate(zip(rows_a, rows_b)):
        if str(row_a.get("id")) != str(row_b.get("id")):
            raise ValueError(f"ID mismatch at row {index}: {path_a} vs {path_b}")
        if str(row_a.get("reference")) != str(row_b.get("reference")):
            raise ValueError(
                f"Reference mismatch for {row_a.get('id')}: {path_a} vs {path_b}"
            )
        error_a, word_count = utterance_counts(row_a)
        error_b, other_word_count = utterance_counts(row_b)
        if word_count != other_word_count:
            raise AssertionError("Paired references produced different word counts")
        errors_a.append(error_a)
        errors_b.append(error_b)
        words.append(word_count)
    return (
        np.asarray(errors_a, dtype=np.int64),
        np.asarray(errors_b, dtype=np.int64),
        np.asarray(words, dtype=np.int64),
    )


def bootstrap(
    name: str,
    path_a: Path,
    path_b: Path,
    samples: int,
    seed: int,
) -> dict:
    errors_a, errors_b, words = paired_counts(path_a, path_b)
    if words.sum() == 0:
        raise RuntimeError("References contain no words")
    observed_a = errors_a.sum() / words.sum()
    observed_b = errors_b.sum() / words.sum()
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    utterances = len(words)
    # Generate in bounded chunks to avoid a samples x utterances allocation.
    for start in range(0, samples, 256):
        count = min(256, samples - start)
        indices = rng.integers(0, utterances, size=(count, utterances))
        denominators = words[indices].sum(axis=1)
        differences[start : start + count] = (
            errors_b[indices].sum(axis=1) / denominators
            - errors_a[indices].sum(axis=1) / denominators
        )
    low, high = np.quantile(differences, [0.025, 0.975])
    p_two_sided = min(
        1.0,
        2 * min(
            (np.count_nonzero(differences <= 0) + 1) / (samples + 1),
            (np.count_nonzero(differences >= 0) + 1) / (samples + 1),
        ),
    )
    return {
        "comparison": name,
        "system_a": str(path_a),
        "system_b": str(path_b),
        "utterances": utterances,
        "bootstrap_samples": samples,
        "seed": seed,
        "wer_a": observed_a,
        "wer_b": observed_b,
        "difference_b_minus_a": observed_b - observed_a,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_two_sided": p_two_sided,
    }


def atomic_write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Paired bootstrap ASR comparisons",
        "",
        "Difference is WER(B) - WER(A); negative values favor B.",
        "",
        "| Comparison | WER A | WER B | Difference | 95% CI | p (two-sided) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['comparison']} | {100 * row['wer_a']:.3f}% | "
            f"{100 * row['wer_b']:.3f}% | "
            f"{100 * row['difference_b_minus_a']:+.3f}pp | "
            f"[{100 * row['ci95_low']:+.3f}, {100 * row['ci95_high']:+.3f}]pp | "
            f"{row['p_two_sided']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    rows = [
        bootstrap(name, Path(path_a), Path(path_b), args.samples, args.seed)
        for name, path_a, path_b in args.comparison
    ]
    atomic_write_csv(args.output_csv, rows)
    atomic_write_markdown(args.output_md, rows)
    for row in rows:
        print(
            f"{row['comparison']}: delta={100 * row['difference_b_minus_a']:+.3f}pp "
            f"CI=[{100 * row['ci95_low']:+.3f}, "
            f"{100 * row['ci95_high']:+.3f}]pp"
        )


if __name__ == "__main__":
    main()
