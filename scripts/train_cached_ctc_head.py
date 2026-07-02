#!/usr/bin/env python
"""Train one BiLSTM-CTC head on cached continuous or discrete SSL features."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_asr.metrics import append_metrics, compute_error_rates, save_predictions
from ssl_asr.representations import (
    CachedBiLSTMCTC,
    CachedCTCConfig,
    CachedSequenceDataset,
    collate_cached_sequences,
    read_cached_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--train_metadata", type=Path, required=True)
    parser.add_argument("--eval_metadata", type=Path, required=True)
    parser.add_argument("--centers", type=Path)
    parser.add_argument("--vocab_path", type=Path, default=PROJECT_ROOT / "data/vocab/vocab.json")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prediction_path", type=Path, required=True)
    parser.add_argument("--metrics_path", type=Path, default=PROJECT_ROOT / "results/metrics.csv")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--collapse_patience_evaluations", type=int, default=4)
    parser.add_argument("--collapse_non_empty_threshold", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable_cudnn", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_tokenizer(vocab_path: Path):
    from transformers import Wav2Vec2CTCTokenizer

    return Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )


def evaluate(
    model: CachedBiLSTMCTC,
    loader,
    tokenizer,
    device: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    references: list[str] = []
    predictions: list[str] = []
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            lengths = batch["lengths"]
            logits = model(features, lengths)
            prediction_ids = torch.argmax(logits, dim=-1).cpu()
            decoded = tokenizer.batch_decode(
                [
                    token_ids[: int(length)]
                    for token_ids, length in zip(prediction_ids, lengths)
                ]
            )
            references.extend(batch["references"])
            predictions.extend(decoded)
            records.extend(
                {
                    "id": utterance_id,
                    "reference": reference,
                    "prediction": prediction,
                    "audio_path": audio_path,
                    "duration": duration,
                }
                for utterance_id, reference, prediction, audio_path, duration in zip(
                    batch["ids"],
                    batch["references"],
                    decoded,
                    batch["audio_paths"],
                    batch["durations"],
                )
            )
    rates = compute_error_rates(references, predictions)
    rates["non_empty_ratio"] = sum(bool(item.strip()) for item in predictions) / len(
        predictions
    )
    return rates, records


def scheduler_factor(step: int, warmup_steps: int, max_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(step, 1) / warmup_steps
    remaining = max_steps - step
    decay_steps = max(max_steps - warmup_steps, 1)
    return max(remaining / decay_steps, 0.0)


def save_training_state(
    path: Path,
    model: CachedBiLSTMCTC,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    best_wer: float,
    collapse_count: int,
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "best_wer": best_wer,
            "collapse_count": collapse_count,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0 or args.eval_steps <= 0:
        raise ValueError("--max_steps and --eval_steps must be positive")
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("Batch size and accumulation steps must be positive")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("--warmup_ratio must be in [0, 1)")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1)")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("[startup] cuDNN disabled; using native CUDA kernels", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Output exists: {args.output_dir}; pass --resume to continue"
        )
    if args.prediction_path.exists() and not args.resume:
        raise FileExistsError(f"Prediction already exists: {args.prediction_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(args.vocab_path)
    centers = (
        np.load(args.centers, allow_pickle=False).astype(np.float32, copy=False)
        if args.centers
        else None
    )
    train_dataset = CachedSequenceDataset(
        args.train_metadata, tokenizer, centroids=centers
    )
    eval_dataset = CachedSequenceDataset(
        args.eval_metadata, tokenizer, centroids=centers
    )
    first = train_dataset[0]
    metadata = read_cached_metadata(args.train_metadata)
    source_models = {str(row["source_model"]) for row in metadata}
    source_layers = {int(row["source_layer"]) for row in metadata}
    encoder_params_set = {int(row["encoder_params"]) for row in metadata}
    if len(source_models) != 1 or len(source_layers) != 1 or len(encoder_params_set) != 1:
        raise RuntimeError("Training metadata mixes incompatible feature sources")
    source_model = source_models.pop()
    source_layer = source_layers.pop()
    encoder_params = encoder_params_set.pop()
    representation = "discrete" if centers is not None else "continuous"
    config = CachedCTCConfig(
        input_size=int(first["features"].shape[1]),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        vocab_size=max(tokenizer.get_vocab().values()) + 1,
        dropout=args.dropout,
        blank_id=tokenizer.pad_token_id,
        source_model=source_model,
        source_layer=source_layer,
        representation=representation,
        codebook_size=len(centers) if centers is not None else None,
    )
    model = CachedBiLSTMCTC(config).to(args.device)
    head_params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[startup] experiment={args.experiment}, representation={representation}, "
        f"layer={source_layer}, train={len(train_dataset)}, dev={len(eval_dataset)}, "
        f"encoder_params={encoder_params:,}, trainable_head_params={head_params:,}",
        flush=True,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_cached_sequences,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_cached_sequences,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    warmup_steps = int(args.max_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: scheduler_factor(step, warmup_steps, args.max_steps),
    )
    criterion = torch.nn.CTCLoss(
        blank=config.blank_id,
        reduction="mean",
        zero_infinity=True,
    )

    step = 0
    best_wer = float("inf")
    collapse_count = 0
    last_state_path = args.output_dir / "last_state.pt"
    if args.resume and last_state_path.is_file():
        state = torch.load(last_state_path, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = int(state["step"])
        best_wer = float(state["best_wer"])
        collapse_count = int(state["collapse_count"])
        print(f"[resume] step={step}, best_wer={best_wer:.5f}", flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_iterator = iter(train_loader)
    stop_for_collapse = False
    while step < args.max_steps:
        accumulated_loss = 0.0
        for _ in range(args.gradient_accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            features = batch["features"].to(args.device, non_blocking=True)
            lengths = batch["lengths"]
            labels = batch["labels"].to(args.device, non_blocking=True)
            logits = model(features, lengths)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = criterion(
                log_probs,
                labels,
                lengths.to(args.device),
                batch["label_lengths"].to(args.device),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite CTC loss at step {step + 1}")
            (loss / args.gradient_accumulation_steps).backward()
            accumulated_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % 25 == 0:
            print(
                f"[train] step={step}/{args.max_steps}, "
                f"loss={accumulated_loss / args.gradient_accumulation_steps:.5f}, "
                f"lr={scheduler.get_last_lr()[0]:.3e}",
                flush=True,
            )
        should_eval = step % args.eval_steps == 0 or step == args.max_steps
        if not should_eval:
            continue
        rates, prediction_rows = evaluate(model, eval_loader, tokenizer, args.device)
        print(
            f"[eval] step={step}, WER={rates['wer']:.5f}, "
            f"CER={rates['cer']:.5f}, "
            f"non_empty_ratio={rates['non_empty_ratio']:.4f}",
            flush=True,
        )
        if rates["wer"] < best_wer:
            best_wer = rates["wer"]
            model.save_pretrained(args.output_dir)
            save_predictions(args.prediction_path, prediction_rows)
            best_summary = {
                "experiment": args.experiment,
                "step": step,
                "best_wer": rates["wer"],
                "best_cer": rates["cer"],
                "non_empty_ratio": rates["non_empty_ratio"],
                "source_model": source_model,
                "source_layer": source_layer,
                "representation": representation,
                "codebook_size": config.codebook_size,
                "encoder_params": encoder_params,
                "trainable_head_params": head_params,
                "total_system_params": encoder_params + head_params,
            }
            summary_path = args.output_dir / "summary.json"
            temporary_summary = summary_path.with_suffix(".json.tmp")
            temporary_summary.write_text(
                json.dumps(best_summary, indent=2) + "\n", encoding="utf-8"
            )
            temporary_summary.replace(summary_path)
        collapse_count = (
            collapse_count + 1
            if rates["non_empty_ratio"] < args.collapse_non_empty_threshold
            else 0
        )
        save_training_state(
            last_state_path,
            model,
            optimizer,
            scheduler,
            step,
            best_wer,
            collapse_count,
        )
        if (
            args.collapse_patience_evaluations > 0
            and collapse_count >= args.collapse_patience_evaluations
        ):
            print("[early-stop] persistent near-empty predictions", flush=True)
            stop_for_collapse = True
            break
        model.train()

    if not (args.output_dir / "model.pt").is_file():
        raise RuntimeError("Training ended without a valid best checkpoint")
    best_model = CachedBiLSTMCTC.from_pretrained(args.output_dir).to(args.device)
    final_rates, final_rows = evaluate(
        best_model, eval_loader, tokenizer, args.device
    )
    save_predictions(args.prediction_path, final_rows)
    append_metrics(
        args.metrics_path,
        {
            "experiment": args.experiment,
            "split": "dev_clean",
            "model": f"{source_model}:layer{source_layer}:{representation}",
            "freeze_encoder": True,
            "num_samples": len(eval_dataset),
            "wer": final_rates["wer"],
            "cer": final_rates["cer"],
            "total_params": encoder_params + head_params,
            "trainable_params": head_params,
        },
    )
    completion = {
        "completed": True,
        "stopped_for_collapse": stop_for_collapse,
        "final_step": step,
        "best_wer": final_rates["wer"],
        "best_cer": final_rates["cer"],
    }
    completion_path = args.output_dir / "completion.json"
    temporary_completion = completion_path.with_suffix(".json.tmp")
    temporary_completion.write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    temporary_completion.replace(completion_path)
    print(
        f"[complete] WER={final_rates['wer']:.5f}, "
        f"CER={final_rates['cer']:.5f}, model={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
