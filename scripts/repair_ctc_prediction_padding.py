#!/usr/bin/env python
"""Repair the known Trainer -100 logit-padding artifact in a prediction JSONL."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBRISPEECH_SPLITS = ("train-clean-100", "dev-clean", "test-clean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_path", type=Path)
    parser.add_argument("--backup_path", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=PROJECT_ROOT / "data")
    return parser.parse_args()


def local_audio_path(audio_path: str, data_root: Path) -> str:
    parts = audio_path.replace("\\", "/").split("/")
    for split in LIBRISPEECH_SPLITS:
        if split in parts:
            index = len(parts) - 1 - parts[::-1].index(split)
            candidate = data_root / Path(*parts[index:])
            if candidate.is_file():
                return str(candidate.resolve())
    return audio_path


def main() -> None:
    args = parse_args()
    if args.backup_path.exists():
        raise FileExistsError(f"Backup already exists: {args.backup_path}")

    with args.prediction_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise RuntimeError("Prediction file is empty")

    trailing_predictions = sum(
        str(row["prediction"]).endswith("'") for row in rows
    )
    trailing_references = sum(
        str(row["reference"]).endswith("'") for row in rows
    )
    if trailing_predictions / len(rows) < 0.9 or trailing_references:
        raise RuntimeError(
            "File does not match the expected systematic trailing-apostrophe artifact"
        )

    shutil.copy2(args.prediction_path, args.backup_path)
    relocated = 0
    for row in rows:
        prediction = str(row["prediction"])
        if prediction.endswith("'"):
            row["prediction"] = prediction[:-1].rstrip()
        old_audio_path = str(row.get("audio_path", ""))
        new_audio_path = local_audio_path(old_audio_path, args.data_root)
        row["audio_path"] = new_audio_path
        relocated += new_audio_path != old_audio_path

    temporary_path = args.prediction_path.with_suffix(
        args.prediction_path.suffix + ".tmp"
    )
    with temporary_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(args.prediction_path)

    print(f"records={len(rows)}")
    print(f"removed_trailing_apostrophes={trailing_predictions}")
    print(f"relocated_audio_paths={relocated}")
    print(f"backup={args.backup_path}")


if __name__ == "__main__":
    main()
