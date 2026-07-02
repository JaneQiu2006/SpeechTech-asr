#!/usr/bin/env python
"""Select the completed continuous layer experiment with the lowest dev WER."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments_root", type=Path, default=Path("exp"))
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 9, 12])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary_glob")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def select_best_layer(experiments_root: Path, layers: list[int]) -> tuple[int, float]:
    candidates = []
    for layer in layers:
        experiment_dir = experiments_root / f"wav2vec2_layer{layer}_bilstm_ctc"
        completion_path = experiment_dir / "completion.json"
        summary_path = experiment_dir / "summary.json"
        if not completion_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Incomplete E10 experiment: {experiment_dir}")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not completion.get("completed"):
            raise RuntimeError(f"Experiment is not complete: {experiment_dir}")
        if summary.get("representation") != "continuous":
            raise RuntimeError(f"Expected continuous experiment: {experiment_dir}")
        if int(summary.get("source_layer", -1)) != layer:
            raise RuntimeError(f"Layer mismatch in {summary_path}")
        candidates.append((float(summary["best_wer"]), layer))
    best_wer, best_layer = min(candidates)
    return best_layer, best_wer


def main() -> None:
    args = parse_args()
    if args.out is not None:
        args.output = args.out
    if args.output is None:
        raise ValueError("--output/--out is required")
    if len(set(args.layers)) != len(args.layers):
        raise ValueError("--layers contains duplicates")
    if args.summary_glob:
        candidates = []
        for name in glob.glob(args.summary_glob):
            summary = json.loads(Path(name).read_text(encoding="utf-8"))
            match = re.search(r"layer(\d+)", name)
            layer = int(summary.get("source_layer", match.group(1) if match else -1))
            wer = float(summary.get("dev_wer", summary.get("best_wer")))
            candidates.append((wer, layer))
        if not candidates:
            raise FileNotFoundError(f"No summaries matched: {args.summary_glob}")
        best_wer, best_layer = min(candidates)
    else:
        best_layer, best_wer = select_best_layer(args.experiments_root, args.layers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_path.write_text(f"{best_layer}\n", encoding="utf-8")
    temporary_path.replace(args.output)
    print(f"best_layer={best_layer}, dev_wer={best_wer:.6f}", flush=True)


if __name__ == "__main__":
    main()
