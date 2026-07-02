#!/usr/bin/env python
"""Cache selected hidden layers from a frozen SSL speech encoder."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_asr.manifest import read_jsonl
from ssl_asr.representations import resolve_audio_path
from ssl_asr.text import normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default="facebook/wav2vec2-base")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--train_manifest", type=Path)
    parser.add_argument("--dev_manifest", type=Path)
    parser.add_argument("--test_manifest", type=Path)
    parser.add_argument("--output_root", "--output_dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 9, 12])
    parser.add_argument("--data_root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--max_duration_in_seconds", type=float, default=15.0)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--disable_cudnn", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--experiment_prefix")
    return parser.parse_args()


def atomic_save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary_path.replace(path)


def batches(rows: list[dict], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def main() -> None:
    args = parse_args()
    manifests = [
        (name, path) for name, path in (
            ("train", args.train_manifest), ("dev", args.dev_manifest),
            ("test", args.test_manifest),
        ) if path is not None
    ]
    if manifests:
        if args.manifest or args.split:
            raise ValueError("Use either --manifest/--split or split-specific manifests")
        for split, manifest in manifests:
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--manifest", str(manifest), "--split", split,
                "--output_root", str(args.output_root), "--layers",
                *(str(layer) for layer in args.layers),
                "--data_root", str(args.data_root),
                "--max_duration_in_seconds", str(args.max_duration_in_seconds),
                "--batch_size", str(args.batch_size), "--device", args.device,
                "--dtype", args.dtype, "--resume",
            ]
            if args.max_samples is not None:
                command.extend(["--max_samples", str(args.max_samples)])
            if args.disable_cudnn:
                command.append("--disable_cudnn")
            subprocess.run(command, check=True)
        return
    if args.manifest is None or args.split is None:
        raise ValueError("--manifest and --split are required in single-split mode")
    if args.skip_existing:
        args.resume = True
    if args.dtype == "fp32":
        args.no_fp16 = True
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if not args.layers or min(args.layers) < 0:
        raise ValueError("--layers must contain non-negative indices")
    if len(set(args.layers)) != len(args.layers):
        raise ValueError("--layers contains duplicates")

    import soundfile as sf
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("[startup] cuDNN disabled; using native CUDA kernels", flush=True)
    records = [
        row
        for row in read_jsonl(args.manifest)
        if float(row["duration"]) <= args.max_duration_in_seconds
    ]
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max_samples must be positive")
        records = records[: args.max_samples]
    if not records:
        raise RuntimeError("No records remain after duration filtering")

    config_path = args.output_root / "cache_config.json"
    cache_config = {
        "model_name_or_path": args.model_name_or_path,
        "max_duration_in_seconds": args.max_duration_in_seconds,
        "dtype": "float32" if args.no_fp16 else "float16",
    }
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        # Older caches recorded the requested layer subset globally, which made
        # safe incremental extraction of additional layers impossible.
        existing.pop("layers", None)
        if existing != cache_config:
            raise RuntimeError(
                f"Cache configuration mismatch at {config_path}: "
                f"{existing} != {cache_config}"
            )
        if not args.resume:
            raise FileExistsError(
                f"Cache already exists: {args.output_root}; pass --resume"
            )
    else:
        args.output_root.mkdir(parents=True, exist_ok=True)
        temporary_config = config_path.with_suffix(".json.tmp")
        temporary_config.write_text(
            json.dumps(cache_config, indent=2) + "\n", encoding="utf-8"
        )
        temporary_config.replace(config_path)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_name_or_path
    )
    model = AutoModel.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()
    encoder_params = sum(parameter.numel() for parameter in model.parameters())
    max_hidden_index = int(getattr(model.config, "num_hidden_layers", 0))
    if max(args.layers) > max_hidden_index:
        raise ValueError(
            f"Requested layer {max(args.layers)}, model has layers 0–{max_hidden_index}"
        )
    use_fp16 = args.device.startswith("cuda") and not args.no_fp16

    metadata_by_layer: dict[int, list[dict]] = {
        layer: [] for layer in args.layers
    }
    processed = 0
    for batch in batches(records, args.batch_size):
        audio_arrays = []
        local_paths = []
        for row in batch:
            path = resolve_audio_path(str(row["audio_path"]), args.data_root)
            audio, sampling_rate = sf.read(path, dtype="float32")
            if sampling_rate != 16_000:
                raise ValueError(
                    f"{row['id']}: expected 16 kHz, got {sampling_rate} Hz"
                )
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
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_fp16
        ):
            outputs = model(**inputs, output_hidden_states=True)
        output_lengths = model._get_feat_extract_output_lengths(
            inputs["attention_mask"].sum(dim=-1)
        )

        for batch_index, (row, local_path, output_length) in enumerate(
            zip(batch, local_paths, output_lengths)
        ):
            frames = int(output_length)
            for layer in args.layers:
                layer_dir = args.output_root / args.split / f"layer{layer}"
                relative_array_path = Path("items") / f"{row['id']}.npy"
                array_path = layer_dir / relative_array_path
                if not array_path.is_file():
                    features = (
                        outputs.hidden_states[layer][batch_index, :frames]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32 if args.no_fp16 else np.float16, copy=False)
                    )
                    atomic_save_numpy(array_path, features)
                cached = np.load(array_path, mmap_mode="r", allow_pickle=False)
                if cached.shape != (frames, int(model.config.hidden_size)):
                    raise RuntimeError(
                        f"Invalid cached shape at {array_path}: {cached.shape}"
                    )
                metadata_by_layer[layer].append(
                    {
                        "id": str(row["id"]),
                        "text": normalize_text(str(row["text"])),
                        "audio_path": str(local_path),
                        "duration": float(row["duration"]),
                        "num_frames": frames,
                        "feature_dim": int(model.config.hidden_size),
                        "array_path": str(relative_array_path),
                        "representation": "continuous",
                        "source_model": args.model_name_or_path,
                        "source_layer": layer,
                        "encoder_params": encoder_params,
                    }
                )
        processed += len(batch)
        if processed % 100 < len(batch):
            print(f"[extract] {processed}/{len(records)} utterances", flush=True)

    for layer, metadata in metadata_by_layer.items():
        layer_dir = args.output_root / args.split / f"layer{layer}"
        metadata_path = layer_dir / "metadata.jsonl"
        temporary_path = metadata_path.with_suffix(".jsonl.tmp")
        with temporary_path.open("w", encoding="utf-8") as stream:
            for row in metadata:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary_path.replace(metadata_path)
        print(
            f"[extract] split={args.split}, layer={layer}, "
            f"records={len(metadata)}, metadata={metadata_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
