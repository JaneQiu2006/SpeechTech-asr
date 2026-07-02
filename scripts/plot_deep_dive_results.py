#!/usr/bin/env python
"""Generate available deep-dive figures from metrics and summaries."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("results/deep_dive_metrics.csv"))
    parser.add_argument("--exp_root", type=Path, default=Path("exp/deep_dive"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--error_stats", type=Path, default=Path("results/deep_dive_error_stats.csv"))
    return parser.parse_args()


def number(row, key):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def plot_xy(plt, points, xlabel, ylabel, path):
    if not points:
        return
    points = sorted(points)
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot([p[0] for p in points], [p[1] for p in points], marker="o")
    for x, y, label in points:
        axis.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install matplotlib to generate deep-dive figures") from error
    rows = []
    if args.metrics.is_file():
        with args.metrics.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layer_points, param_points, head_points, k_points, bitrate_points, utilization_points = [], [], [], [], [], []
    for row in rows:
        wer = number(row, "wer")
        if wer is None:
            continue
        label = row.get("experiment_id") or row.get("experiment", "")
        layer = number(row, "source_layer")
        params = number(row, "trainable_params")
        codebook = number(row, "unit_codebook_size") or number(row, "codebook_size")
        bitrate = number(row, "unit_bitrate_bps")
        if layer is not None and str(row.get("representation", "")).startswith("continuous"):
            layer_points.append((layer, wer, label))
        if params is not None:
            param_points.append((params / 1e6, wer, label))
            if str(label).lower().startswith("e15"):
                head_points.append((params / 1e6, wer, label))
        if codebook is not None:
            k_points.append((codebook, wer, label))
        if bitrate is not None:
            bitrate_points.append((bitrate, wer, label))
        utilization = number(row, "unit_utilization")
        if codebook is not None and utilization is not None:
            utilization_points.append((codebook, utilization, label))
    plot_xy(plt, layer_points, "Wav2Vec2 layer", "WER", args.output_dir / "layerwise_wer_curve.png")
    plot_xy(plt, param_points, "Trainable parameters (M)", "WER", args.output_dir / "trainable_params_vs_wer.png")
    plot_xy(plt, k_points, "Codebook size K", "WER", args.output_dir / "discrete_wer_vs_k.png")
    plot_xy(plt, bitrate_points, "Empirical bitrate (bit/s)", "WER", args.output_dir / "discrete_wer_vs_bitrate.png")
    plot_xy(plt, head_points, "Trainable parameters (M)", "WER", args.output_dir / "head_capacity_wer_params.png")
    plot_xy(plt, utilization_points, "Codebook size K", "Codebook utilization", args.output_dir / "codebook_utilization.png")
    scale_rows = []
    baseline_hours = {"E3": 1.001, "E5": 3.001, "E2": 6.392}
    for metrics_path in (Path("results/test_metrics.csv"), args.metrics):
        if not metrics_path.is_file():
            continue
        with metrics_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                label = row.get("experiment_id") or row.get("experiment", "")
                wer = number(row, "wer")
                if wer is None:
                    continue
                hours = number(row, "effective_hours") or baseline_hours.get(label)
                if hours is None:
                    match = re.search(r"(1|3|10)h", row.get("experiment", ""))
                    hours = {"1": 1.001, "3": 3.001, "10": 6.392}.get(
                        match.group(1) if match else ""
                    )
                if hours is not None and (
                    label in baseline_hours or str(label).lower().startswith("e20")
                ):
                    scale_rows.append(
                        {"system": "full_finetune" if label in baseline_hours else "frozen_bilstm",
                         "effective_hours": hours, "wer": wer, "experiment": label}
                    )
    if scale_rows:
        output_csv = Path("results/data_scale_frozen_vs_finetune.csv")
        with output_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(scale_rows[0]))
            writer.writeheader()
            writer.writerows(scale_rows)
        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        for system in ("full_finetune", "frozen_bilstm"):
            selected = sorted(
                (row for row in scale_rows if row["system"] == system),
                key=lambda row: row["effective_hours"],
            )
            if selected:
                axis.plot(
                    [row["effective_hours"] for row in selected],
                    [row["wer"] for row in selected],
                    marker="o", label=system,
                )
        axis.set(xlabel="Effective labeled hours", ylabel="WER")
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(args.output_dir / "data_scale_frozen_vs_finetune.png", dpi=180)
        plt.close(figure)
    if args.error_stats.is_file():
        with args.error_stats.open(encoding="utf-8", newline="") as stream:
            error_rows = list(csv.DictReader(stream))
        if error_rows:
            figure, axis = plt.subplots(figsize=(max(6.4, len(error_rows) * 0.7), 4.2))
            systems = [row["system"] for row in error_rows]
            bottom = [0.0] * len(error_rows)
            for key, label in (
                ("substitutions", "Substitution"), ("deletions", "Deletion"),
                ("insertions", "Insertion"),
            ):
                values = [float(row[key]) for row in error_rows]
                axis.bar(systems, values, bottom=bottom, label=label)
                bottom = [left + right for left, right in zip(bottom, values)]
            axis.set(ylabel="Word error count")
            axis.tick_params(axis="x", rotation=45)
            axis.legend()
            figure.tight_layout()
            figure.savefig(args.output_dir / "error_type_breakdown.png", dpi=180)
            plt.close(figure)
            bucket_fields = [
                field for field in error_rows[0] if field.startswith("wer_duration_")
            ]
            if bucket_fields:
                figure, axis = plt.subplots(figsize=(7, 4.2))
                width = 0.8 / len(error_rows)
                positions = list(range(len(bucket_fields)))
                for index, row in enumerate(error_rows):
                    axis.bar(
                        [position + index * width for position in positions],
                        [float(row.get(field) or 0) for field in bucket_fields],
                        width=width, label=row["system"],
                    )
                axis.set_xticks(
                    [position + width * (len(error_rows) - 1) / 2 for position in positions],
                    [field.removeprefix("wer_duration_") for field in bucket_fields],
                )
                axis.set(ylabel="WER", xlabel="Duration bucket")
                axis.legend(fontsize=7)
                figure.tight_layout()
                figure.savefig(args.output_dir / "length_bucket_wer.png", dpi=180)
                plt.close(figure)
    mixture_summaries = sorted(args.exp_root.glob("e14*/summary.json"))
    if mixture_summaries:
        summary = json.loads(mixture_summaries[-1].read_text(encoding="utf-8"))
        weights = summary.get("layer_weights") or {}
        if weights:
            figure, axis = plt.subplots(figsize=(7, 4))
            layers = sorted(weights, key=int)
            axis.bar(layers, [weights[layer] for layer in layers])
            axis.set(xlabel="Layer", ylabel="Mixture weight")
            figure.tight_layout()
            figure.savefig(args.output_dir / "e14_layer_mixture_weights.png", dpi=180)
            plt.close(figure)
    print(f"figures={args.output_dir}")


if __name__ == "__main__":
    main()
