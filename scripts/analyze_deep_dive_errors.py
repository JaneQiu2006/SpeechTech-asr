#!/usr/bin/env python
"""Aggregate word-edit errors from one or more ASR prediction JSONL files."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "i", "in", "is", "it", "not",
    "of", "on", "or", "she", "that", "the", "their", "they", "this", "to",
    "was", "we", "were", "will", "with", "you",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred_glob", required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--out_md", type=Path, required=True)
    parser.add_argument("--pair", nargs=2, metavar=("SYSTEM_A", "SYSTEM_B"))
    parser.add_argument("--top_n", type=int, default=30)
    return parser.parse_args()


def align_words(reference: list[str], prediction: list[str]):
    """Return a deterministic Levenshtein alignment as (op, ref, hyp) tuples."""
    rows, cols = len(reference) + 1, len(prediction) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost[i][j] = min(
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
                cost[i - 1][j - 1] + (reference[i - 1] != prediction[j - 1]),
            )
    result = []
    i, j = len(reference), len(prediction)
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (
            reference[i - 1] != prediction[j - 1]
        ):
            op = "equal" if reference[i - 1] == prediction[j - 1] else "substitution"
            result.append((op, reference[i - 1], prediction[j - 1]))
            i -= 1
            j -= 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            result.append(("deletion", reference[i - 1], ""))
            i -= 1
        else:
            result.append(("insertion", "", prediction[j - 1]))
            j -= 1
    return list(reversed(result))


def bucket(duration: float) -> str:
    if duration < 5:
        return "<5s"
    if duration < 10:
        return "5-10s"
    if duration < 15:
        return "10-15s"
    return ">=15s"


def safe_rate(errors: int, words: int) -> float:
    return errors / words if words else 0.0


def analyze(path: Path, top_n: int) -> tuple[dict, dict]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reference_frequency = Counter(
        word
        for row in rows
        for word in str(row.get("reference", "")).split()
    )
    totals = Counter()
    confusions, deleted, inserted = Counter(), Counter(), Counter()
    bucket_counts: dict[str, Counter] = {}
    speaker_counts: dict[str, Counter] = {}
    examples = []
    pred_words = 0
    non_empty = 0
    for row in rows:
        reference = str(row.get("reference", "")).split()
        prediction = str(row.get("prediction", "")).split()
        alignment = align_words(reference, prediction)
        counts = Counter(op for op, _, _ in alignment)
        errors = counts["substitution"] + counts["deletion"] + counts["insertion"]
        totals.update(counts)
        totals["reference_words"] += len(reference)
        pred_words += len(prediction)
        non_empty += bool(prediction)
        duration_bucket = bucket(float(row.get("duration", 0)))
        bucket_counts.setdefault(duration_bucket, Counter()).update(
            errors=errors, words=len(reference)
        )
        speaker = str(row.get("speaker_id") or str(row.get("id", "")).split("-")[0])
        if speaker:
            speaker_counts.setdefault(speaker, Counter()).update(
                errors=errors, words=len(reference)
            )
        for op, ref, hyp in alignment:
            if op == "substitution":
                confusions[(ref, hyp)] += 1
            elif op == "deletion":
                deleted[ref] += 1
            elif op == "insertion":
                inserted[hyp] += 1
        totals["function_errors"] += sum(
            op != "equal" and (ref in FUNCTION_WORDS or hyp in FUNCTION_WORDS)
            for op, ref, hyp in alignment
        )
        totals["function_words"] += sum(word in FUNCTION_WORDS for word in reference)
        totals["rare_words"] += sum(reference_frequency[word] <= 2 for word in reference)
        totals["rare_word_errors"] += sum(
            op in {"substitution", "deletion"} and reference_frequency[ref] <= 2
            for op, ref, _ in alignment
            if ref
        )
        if errors:
            examples.append((safe_rate(errors, len(reference)), row))
    errors = totals["substitution"] + totals["deletion"] + totals["insertion"]
    stats = {
        "system": path.stem.removesuffix("_dev").removesuffix("_test"),
        "prediction_path": str(path),
        "utterances": len(rows),
        "wer": safe_rate(errors, totals["reference_words"]),
        "substitutions": totals["substitution"],
        "deletions": totals["deletion"],
        "insertions": totals["insertion"],
        "exact_match": totals["equal"] == totals["reference_words"] and not errors
        if len(rows) == 1 else sum(
            str(row.get("reference", "")) == str(row.get("prediction", "")) for row in rows
        ),
        "avg_prediction_words": pred_words / len(rows) if rows else 0,
        "prediction_reference_length_ratio": safe_rate(pred_words, totals["reference_words"]),
        "non_empty_ratio": non_empty / len(rows) if rows else 0,
        "function_word_error_rate": safe_rate(
            totals["function_errors"], totals["function_words"]
        ),
        "rare_word_error_rate": safe_rate(
            totals["rare_word_errors"], totals["rare_words"]
        ),
    }
    detail = {
        "confusions": confusions.most_common(top_n),
        "deleted": deleted.most_common(top_n),
        "inserted": inserted.most_common(top_n),
        "buckets": {
            name: safe_rate(counts["errors"], counts["words"])
            for name, counts in bucket_counts.items()
        },
        "speakers": {
            name: safe_rate(counts["errors"], counts["words"])
            for name, counts in speaker_counts.items()
        },
        "examples": [
            row for _, row in sorted(
                examples, key=lambda item: item[0], reverse=True
            )[:5]
        ],
    }
    for name, value in detail["buckets"].items():
        stats[f"wer_duration_{name.replace('>=', 'ge').replace('<', 'lt')}"] = value
    return stats, detail


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    paths = sorted(Path(item) for item in glob.glob(args.pred_glob, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No predictions matched: {args.pred_glob}")
    analyses = [(*analyze(path, args.top_n), path) for path in paths]
    write_csv(args.out_csv, [stats for stats, _, _ in analyses])
    lines = ["# Deep-dive ASR error analysis", "", "## System summary", ""]
    lines += [
        "| System | WER | S | D | I | Exact | Non-empty | Length ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stats, _, _ in analyses:
        lines.append(
            f"| {stats['system']} | {stats['wer']:.4f} | {stats['substitutions']} | "
            f"{stats['deletions']} | {stats['insertions']} | {stats['exact_match']} | "
            f"{stats['non_empty_ratio']:.4f} | "
            f"{stats['prediction_reference_length_ratio']:.4f} |"
        )
    for stats, detail, _ in analyses:
        lines += ["", f"## {stats['system']}", "", "Top substitutions:", ""]
        lines += [f"- `{ref}` → `{hyp}`: {count}" for (ref, hyp), count in detail["confusions"]]
        lines += ["", "Top deleted words:", ""]
        lines += [f"- `{word}`: {count}" for word, count in detail["deleted"]]
        lines += ["", "Top inserted words:", ""]
        lines += [f"- `{word}`: {count}" for word, count in detail["inserted"]]
        lines += ["", "Duration-bucket WER:", ""]
        lines += [f"- {name}: {value:.4f}" for name, value in detail["buckets"].items()]
        lines += ["", "Highest speaker-level WER:", ""]
        lines += [
            f"- {name}: {value:.4f}"
            for name, value in sorted(
                detail["speakers"].items(), key=lambda item: item[1], reverse=True
            )[:args.top_n]
        ]
        lines += ["", "Representative high-error utterances:", ""]
        lines += [
            f"- `{row.get('id', '')}` ref: {row.get('reference', '')} / "
            f"hyp: {row.get('prediction', '')}"
            for row in detail["examples"]
        ]
    if args.pair:
        selected = [
            stats for stats, _, _ in analyses
            if any(name.lower() in stats["system"].lower() for name in args.pair)
        ]
        if len(selected) == 2:
            lines += [
                "", "## Pairwise comparison", "",
                f"{selected[0]['system']} − {selected[1]['system']} WER: "
                f"{selected[0]['wer'] - selected[1]['wer']:+.4f}",
            ]
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_md.with_suffix(args.out_md.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(args.out_md)
    print(f"analyzed={len(paths)} csv={args.out_csv} markdown={args.out_md}")


if __name__ == "__main__":
    main()
