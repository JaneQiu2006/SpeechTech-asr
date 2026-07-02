#!/usr/bin/env python
"""Convenience entry point for E14 learned scalar layer mixture."""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
layers = list(range(13))
paths = [
    ROOT / "features/wav2vec2_base" / split / f"layer{layer}/metadata.jsonl"
    for split in ("train", "dev") for layer in layers
]
missing = [str(path) for path in paths if not path.is_file()]
if missing:
    raise FileNotFoundError("Missing layer caches:\n" + "\n".join(missing))
command = [
    sys.executable, str(ROOT / "scripts/train_cached_ctc_head.py"),
    "--experiment", "e14_layer_mixture_bilstm_ctc",
    "--representation_type", "layer_mixture",
    "--layer_metadata", *(str(path) for path in paths),
    "--train_metadata", str(paths[0]), "--eval_metadata", str(paths[13]),
    "--head_type", "bilstm", "--num_layers", "2",
    "--output_dir", str(ROOT / "exp/deep_dive/e14_layer_mixture_bilstm_ctc"),
    "--prediction_path", str(ROOT / "results/predictions/deep_dive/e14_layer_mixture_bilstm_ctc_dev.jsonl"),
    "--metrics_path", str(ROOT / "results/deep_dive_metrics.csv"),
    "--overwrite" if os.environ.get("DEEP_DIVE_OVERWRITE") == "1" else "--skip_existing",
]
subprocess.run(command, check=True)
test_command = [
    sys.executable, str(ROOT / "scripts/evaluate_representation_ctc.py"),
    "--experiment", "e14_layer_mixture_bilstm_ctc",
    "--model_dir", str(ROOT / "exp/deep_dive/e14_layer_mixture_bilstm_ctc"),
    "--manifest", str(ROOT / "data/manifests/test_clean.jsonl"),
    "--prediction_path", str(
        ROOT / "results/predictions/deep_dive/e14_layer_mixture_bilstm_ctc_test.jsonl"
    ),
    "--metrics_path", str(ROOT / "results/deep_dive_metrics.csv"),
]
if os.environ.get("DEEP_DIVE_OVERWRITE") == "1":
    test_command.append("--overwrite")
subprocess.run(test_command, check=True)
