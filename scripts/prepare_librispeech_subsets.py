#!/usr/bin/env python
"""Prepare local-first LibriSpeech manifests and low-resource subsets."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_asr.manifest import scan_librispeech_split, write_jsonl
from ssl_asr.text import build_vocabulary

HF_SPLITS = {
    "train-clean-100": "train.100",
    "dev-clean": "validation",
    "test-clean": "test",
}
MANIFEST_NAMES = {
    "dev-clean": "dev_clean.jsonl",
    "test-clean": "test_clean.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--manifest_dir", type=Path, default=PROJECT_ROOT / "data" / "manifests"
    )
    parser.add_argument(
        "--vocab_path",
        type=Path,
        default=PROJECT_ROOT / "data" / "vocab" / "vocab.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_hours", type=float, nargs="+", default=[1, 10])
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(HF_SPLITS),
        default=list(HF_SPLITS),
    )
    parser.add_argument(
        "--local_only",
        action="store_true",
        help="Fail rather than download a missing split.",
    )
    return parser.parse_args()


def _extract_hf_audio(example: dict):
    """Handle both datasets Audio dictionaries and torchcodec decoders."""
    audio = example["audio"]
    if isinstance(audio, dict):
        return audio["array"], audio["sampling_rate"]
    samples = audio.get_all_samples()
    array = samples.data
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    if getattr(array, "ndim", 1) == 2:
        array = array[0]
    return array, samples.sample_rate


def download_hf_split(split: str, destination: Path) -> None:
    """Download one missing HF split and materialize standard FLAC files."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "A split is missing. Install datasets[audio] for the Hugging Face "
            "fallback."
        ) from error

    print(f"[download] {split} is absent; loading Hugging Face split {HF_SPLITS[split]}")
    dataset = load_dataset(
        "openslr/librispeech_asr", "clean", split=HF_SPLITS[split]
    )
    destination.mkdir(parents=True, exist_ok=True)
    transcript_handles: dict[tuple[str, str], object] = {}
    try:
        for example in dataset:
            utterance_id = str(example["id"])
            speaker, chapter, _ = utterance_id.split("-")
            chapter_dir = destination / speaker / chapter
            chapter_dir.mkdir(parents=True, exist_ok=True)
            target_path = chapter_dir / f"{utterance_id}.flac"
            cached_path = example.get("file")
            audio = example["audio"]
            if not cached_path and isinstance(audio, dict):
                cached_path = audio.get("path")
            if cached_path and Path(cached_path).is_file():
                shutil.copy2(cached_path, target_path)
            else:
                try:
                    import soundfile as sf
                except ImportError as error:
                    raise RuntimeError(
                        "Install soundfile to materialize decoded Hugging Face audio"
                    ) from error
                array, sample_rate = _extract_hf_audio(example)
                sf.write(target_path, array, sample_rate)
            key = (speaker, chapter)
            if key not in transcript_handles:
                transcript_handles[key] = (
                    chapter_dir / f"{speaker}-{chapter}.trans.txt"
                ).open("w", encoding="utf-8")
            transcript_handles[key].write(
                f"{utterance_id} {str(example['text']).upper()}\n"
            )
    finally:
        for handle in transcript_handles.values():
            handle.close()


def select_duration_prefix(
    records: list[dict], target_hours: float, seed: int
) -> list[dict]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    target_seconds = target_hours * 3600
    selected: list[dict] = []
    duration = 0.0
    for record in shuffled:
        selected.append(record)
        duration += record["duration"]
        if duration >= target_seconds:
            break
    if duration < target_seconds:
        raise RuntimeError(
            f"Requested {target_hours:g}h but only {duration / 3600:.2f}h is available"
        )
    return selected


def main() -> None:
    args = parse_args()
    all_records: dict[str, list[dict]] = {}

    for split in args.splits:
        split_dir = args.data_root / split
        if not split_dir.is_dir():
            if args.local_only:
                raise FileNotFoundError(f"Required local split is absent: {split_dir}")
            download_hf_split(split, split_dir)
        print(f"[scan] {split_dir}")
        records = scan_librispeech_split(split_dir)
        all_records[split] = records
        hours = sum(record["duration"] for record in records) / 3600
        print(f"[ok] {split}: {len(records)} utterances, {hours:.2f}h")

    train_records = all_records.get("train-clean-100")
    if train_records:
        for hours in sorted(set(args.target_hours)):
            subset = select_duration_prefix(train_records, hours, args.seed)
            label = f"{hours:g}".replace(".", "p")
            output = args.manifest_dir / f"train_{label}h.jsonl"
            write_jsonl(output, subset)
            actual = sum(record["duration"] for record in subset) / 3600
            print(f"[write] {output}: {len(subset)} utterances, {actual:.3f}h")

    for split, output_name in MANIFEST_NAMES.items():
        if split in all_records:
            output = args.manifest_dir / output_name
            write_jsonl(output, all_records[split])
            print(f"[write] {output}: {len(all_records[split])} utterances")

    vocabulary_records = train_records or [
        record for records in all_records.values() for record in records
    ]
    vocabulary = build_vocabulary(
        (record["text"] for record in vocabulary_records), args.vocab_path
    )
    print(f"[write] {args.vocab_path}: {len(vocabulary)} tokens")


if __name__ == "__main__":
    main()
