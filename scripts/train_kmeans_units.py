#!/usr/bin/env python
"""Fit a training-only MiniBatch K-means codebook and cache unit sequences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_asr.representations import (
    read_cached_metadata,
    resolve_cached_array,
    unit_stream_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_metadata", type=Path, required=True)
    parser.add_argument("--eval_metadata", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--codebook_size", type=int, required=True)
    parser.add_argument("--max_fit_frames", type=int, default=500_000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sample_training_frames(
    metadata_path: Path,
    max_frames: int,
    seed: int,
) -> np.ndarray:
    rows = read_cached_metadata(metadata_path)
    total_frames = sum(int(row["num_frames"]) for row in rows)
    target = min(total_frames, max_frames)
    rng = np.random.default_rng(seed)
    sampled: list[np.ndarray] = []
    remaining_target = target
    remaining_frames = total_frames
    for row in rows:
        path = resolve_cached_array(metadata_path, str(row["array_path"]))
        features = np.load(path, mmap_mode="r", allow_pickle=False)
        count = len(features)
        if remaining_target == remaining_frames:
            selected = np.arange(count)
        else:
            expected = remaining_target * count / remaining_frames
            take = min(count, int(np.floor(expected)))
            if rng.random() < expected - take and take < count:
                take += 1
            selected = (
                rng.choice(count, size=take, replace=False)
                if take
                else np.empty(0, dtype=np.int64)
            )
        if len(selected):
            sampled.append(np.asarray(features[selected], dtype=np.float32))
        remaining_target -= len(selected)
        remaining_frames -= count
    if remaining_target != 0:
        raise RuntimeError(
            f"Frame sampler selected {target - remaining_target}, expected {target}"
        )
    frames = np.concatenate(sampled, axis=0)
    rng.shuffle(frames, axis=0)
    return frames


def atomic_save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary_path.replace(path)


def quantize_split(
    kmeans,
    source_metadata: Path,
    output_dir: Path,
    split: str,
    codebook_size: int,
) -> tuple[Path, dict[str, float | int]]:
    rows = read_cached_metadata(source_metadata)
    output_rows = []
    unit_sequences = []
    audio_seconds = 0.0
    for index, row in enumerate(rows, start=1):
        feature_path = resolve_cached_array(
            source_metadata, str(row["array_path"])
        )
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        unit_ids = kmeans.predict(np.asarray(features, dtype=np.float32))
        dtype = np.uint8 if codebook_size <= 256 else np.uint16
        unit_ids = unit_ids.astype(dtype, copy=False)
        relative_path = Path("items") / f"{row['id']}.npy"
        array_path = output_dir / split / relative_path
        atomic_save_numpy(array_path, unit_ids)
        output_rows.append(
            {
                **row,
                "array_path": str(relative_path),
                "representation": "discrete",
                "codebook_size": codebook_size,
                "source_feature_metadata": str(source_metadata.resolve()),
            }
        )
        unit_sequences.append(unit_ids)
        audio_seconds += float(row["duration"])
        if index % 200 == 0:
            print(f"[quantize] {split}: {index}/{len(rows)}", flush=True)

    metadata_path = output_dir / split / "metadata.jsonl"
    temporary_path = metadata_path.with_suffix(".jsonl.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(metadata_path)
    statistics = unit_stream_statistics(
        unit_sequences, audio_seconds, codebook_size
    )
    return metadata_path, statistics


def main() -> None:
    args = parse_args()
    if args.codebook_size <= 1:
        raise ValueError("--codebook_size must be greater than one")
    if args.codebook_size > np.iinfo(np.uint16).max:
        raise ValueError("--codebook_size exceeds the uint16 cache format")
    if args.max_fit_frames < args.codebook_size:
        raise ValueError("--max_fit_frames must be at least codebook_size")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Output exists: {args.output_dir}; pass --resume to reuse it"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as error:
        raise RuntimeError(
            "Install scikit-learn to train discrete-unit codebooks"
        ) from error

    centers_path = args.output_dir / "centers.npy"
    model_path = args.output_dir / "kmeans.joblib"
    if centers_path.is_file() and model_path.is_file():
        if not args.resume:
            raise FileExistsError(f"Codebook already exists: {args.output_dir}")
        import joblib

        kmeans = joblib.load(model_path)
        if int(kmeans.n_clusters) != args.codebook_size:
            raise RuntimeError("Existing codebook size does not match arguments")
        print(f"[kmeans] reusing {model_path}", flush=True)
    else:
        fit_frames = sample_training_frames(
            args.train_metadata, args.max_fit_frames, args.seed
        )
        print(
            f"[kmeans] fitting K={args.codebook_size} on "
            f"{len(fit_frames):,} training frames",
            flush=True,
        )
        kmeans = MiniBatchKMeans(
            n_clusters=args.codebook_size,
            batch_size=args.batch_size,
            random_state=args.seed,
            n_init=3,
            max_iter=200,
            reassignment_ratio=0.01,
            verbose=0,
        )
        kmeans.fit(fit_frames)
        atomic_save_numpy(
            centers_path,
            np.asarray(kmeans.cluster_centers_, dtype=np.float32),
        )
        import joblib

        temporary_model = model_path.with_suffix(".joblib.tmp")
        joblib.dump(kmeans, temporary_model)
        temporary_model.replace(model_path)

    summaries = {}
    for split, metadata in (
        ("train", args.train_metadata),
        ("dev", args.eval_metadata),
    ):
        metadata_path, statistics = quantize_split(
            kmeans,
            metadata,
            args.output_dir,
            split,
            args.codebook_size,
        )
        summaries[split] = {
            "metadata": str(metadata_path.resolve()),
            **statistics,
        }
    summary = {
        "codebook_size": args.codebook_size,
        "train_metadata": str(args.train_metadata.resolve()),
        "eval_metadata": str(args.eval_metadata.resolve()),
        "centers": str(centers_path.resolve()),
        "seed": args.seed,
        "max_fit_frames": args.max_fit_frames,
        "splits": summaries,
    }
    summary_path = args.output_dir / "summary.json"
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    temporary_summary.replace(summary_path)
    print(f"[kmeans] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
