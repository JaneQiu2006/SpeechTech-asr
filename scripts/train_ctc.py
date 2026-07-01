#!/usr/bin/env python
"""Minimal manifest-based CTC fine-tuning for SSL speech encoders."""

from __future__ import annotations

import argparse
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_asr.manifest import read_jsonl
from ssl_asr.metrics import append_metrics, compute_error_rates, save_predictions
from ssl_asr.text import normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default="facebook/wav2vec2-base")
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Local data directory used to relocate manifest paths after migration.",
    )
    parser.add_argument("--vocab_path", type=Path, default=PROJECT_ROOT / "data/vocab/vocab.json")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prediction_path", type=Path)
    parser.add_argument("--metrics_path", type=Path, default=PROJECT_ROOT / "results/metrics.csv")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument(
        "--freeze_transformer_layers",
        type=int,
        default=0,
        help="Freeze the bottom N Transformer blocks while training higher blocks.",
    )
    parser.add_argument(
        "--ctc_head_type",
        choices=("linear", "bilstm"),
        default="linear",
    )
    parser.add_argument("--ctc_head_hidden_size", type=int, default=256)
    parser.add_argument("--ctc_head_num_layers", type=int, default=2)
    parser.add_argument("--ctc_head_dropout", type=float, default=0.1)
    parser.add_argument(
        "--ctc_loss_reduction",
        choices=("mean", "sum"),
        default="mean",
    )
    parser.add_argument("--ctc_zero_infinity", action="store_true")
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument("--max_duration_in_seconds", type=float, default=20.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument(
        "--encoder_learning_rate",
        type=float,
        help="Optional lower learning rate for non-head trainable parameters.",
    )
    parser.add_argument(
        "--encoder_freeze_steps",
        type=int,
        default=0,
        help="Keep encoder optimizer groups at zero LR for the first N steps.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Fraction of optimizer steps used for linear learning-rate warmup.",
    )
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Positive values override num_train_epochs.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        help="Evaluate and save every N optimizer steps instead of every epoch.",
    )
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument(
        "--eval_accumulation_steps",
        type=int,
        default=10,
        help="Move evaluation predictions to CPU periodically to bound GPU memory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask_time_prob", type=float)
    parser.add_argument("--mask_feature_prob", type=float)
    parser.add_argument(
        "--collapse_patience_evaluations",
        type=int,
        default=0,
        help="Stop after this many consecutive near-empty evaluations; zero disables.",
    )
    parser.add_argument(
        "--collapse_non_empty_threshold",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--diagnostic_train_samples",
        type=int,
        default=0,
        help="Evaluate this many training samples after training; zero disables.",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        help="Reload the checkpoint with the lowest dev WER after training.",
    )
    parser.add_argument(
        "--disable_cudnn",
        action="store_true",
        help="Use native CUDA kernels when the host cuDNN runtime cannot initialize.",
    )
    return parser.parse_args()


def tokenizer_id_space_size(tokenizer: Any) -> int:
    """Return the output size required to represent every tokenizer ID."""
    vocabulary = tokenizer.get_vocab()
    if not vocabulary:
        raise ValueError("Tokenizer vocabulary is empty")
    token_ids = list(vocabulary.values())
    if any(not isinstance(token_id, int) or token_id < 0 for token_id in token_ids):
        raise ValueError("Tokenizer vocabulary contains invalid token IDs")
    return max(token_ids) + 1


def encode_ctc_labels(
    tokenizer: Any,
    text: str,
    vocab_size: int,
    record_id: str,
) -> list[int]:
    """Encode one transcript and fail with context on an invalid label ID."""
    labels = tokenizer(
        normalize_text(text),
        add_special_tokens=False,
    ).input_ids
    invalid_ids = sorted(
        {token_id for token_id in labels if token_id < 0 or token_id >= vocab_size}
    )
    if invalid_ids:
        raise ValueError(
            f"{record_id}: label IDs {invalid_ids} are outside the model range "
            f"[0, {vocab_size - 1}]"
        )
    return labels


def validate_record_labels(
    records: list[dict],
    tokenizer: Any,
    vocab_size: int,
    split_name: str,
) -> None:
    """Validate every transcript before spending time on model training."""
    highest_label_id = -1
    for record in records:
        labels = encode_ctc_labels(
            tokenizer,
            str(record["text"]),
            vocab_size,
            str(record["id"]),
        )
        if labels:
            highest_label_id = max(highest_label_id, max(labels))
    print(
        f"[data] {split_name} label validation passed: "
        f"highest_label_id={highest_label_id}, vocab_size={vocab_size}",
        flush=True,
    )


class ManifestDataset:
    def __init__(self, records: list[dict], processor: Any, vocab_size: int):
        self.records = records
        self.processor = processor
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        try:
            import soundfile as sf
        except ImportError as error:
            raise RuntimeError("Install soundfile to load manifest audio") from error
        record = self.records[index]
        waveform, sample_rate = sf.read(record["audio_path"], dtype="float32")
        if getattr(waveform, "ndim", 1) > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != 16_000:
            raise ValueError(
                f"{record['audio_path']} is {sample_rate} Hz; startup expects 16 kHz"
            )
        inputs = self.processor(waveform, sampling_rate=sample_rate)
        labels = encode_ctc_labels(
            self.processor.tokenizer,
            str(record["text"]),
            self.vocab_size,
            str(record["id"]),
        )
        return {
            "input_values": inputs.input_values[0],
            "labels": labels,
            "record_index": index,
        }


@dataclass
class CTCDataCollator:
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        import torch

        input_features = [
            {"input_values": feature["input_values"]}
            for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(
            input_features,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        labels = self.processor.tokenizer.pad(
            label_features, padding=True, return_tensors="pt"
        )
        batch["labels"] = labels["input_ids"].masked_fill(
            labels["attention_mask"].ne(1), -100
        )
        return batch


def prediction_ids_from_logits(logits: Any, blank_id: int) -> Any:
    """Map Trainer's synthetic -100 logit padding to the CTC blank."""
    import numpy as np

    prediction_ids = np.argmax(logits, axis=-1)
    padded_frames = np.all(logits == -100, axis=-1)
    prediction_ids[padded_frames] = blank_id
    return prediction_ids


def prediction_diagnostics(
    logits: Any,
    prediction_ids: Any,
    predictions: list[str],
    blank_id: int,
) -> dict[str, float]:
    """Measure blank dominance without counting Trainer's synthetic padding."""
    import numpy as np

    padded_frames = np.all(logits == -100, axis=-1)
    valid_frames = ~padded_frames
    valid_count = int(valid_frames.sum())
    blank_count = int(((prediction_ids == blank_id) & valid_frames).sum())
    non_empty = sum(bool(prediction.strip()) for prediction in predictions)
    return {
        "non_empty_ratio": non_empty / len(predictions) if predictions else 0.0,
        "blank_frame_ratio": blank_count / valid_count if valid_count else 1.0,
    }


def optimizer_parameter_groups(
    model: Any,
    head_learning_rate: float,
    encoder_learning_rate: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Build head/encoder and decay/no-decay AdamW parameter groups."""
    groups: dict[tuple[str, bool], list[Any]] = {
        ("head", True): [],
        ("head", False): [],
        ("encoder", True): [],
        ("encoder", False): [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        component = "head" if name.startswith("lm_head.") else "encoder"
        use_decay = not (
            name.endswith(".bias")
            or name.endswith("LayerNorm.weight")
            or name.endswith("layer_norm.weight")
        )
        groups[(component, use_decay)].append(parameter)

    result = []
    for (component, use_decay), parameters in groups.items():
        if not parameters:
            continue
        result.append(
            {
                "params": parameters,
                "lr": (
                    head_learning_rate
                    if component == "head"
                    else encoder_learning_rate
                ),
                "weight_decay": weight_decay if use_decay else 0.0,
                "group_name": component,
            }
        )
    return result


def transformer_encoder_layers(model: Any) -> list[Any]:
    """Locate the Transformer block list across supported SSL encoders."""
    base_model = getattr(model, getattr(model, "base_model_prefix", ""), None)
    encoder = getattr(base_model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        raise RuntimeError(
            f"{type(model).__name__} does not expose base_model.encoder.layers"
        )
    return list(layers)


def freeze_bottom_transformer_layers(model: Any, count: int) -> tuple[list[int], list[int]]:
    """Freeze the bottom ``count`` blocks and report frozen/trainable indices."""
    layers = transformer_encoder_layers(model)
    if count < 0 or count > len(layers):
        raise ValueError(
            f"--freeze_transformer_layers must be in [0, {len(layers)}], got {count}"
        )
    for index, layer in enumerate(layers):
        if index < count:
            for parameter in layer.parameters():
                parameter.requires_grad = False
    return list(range(count)), list(range(count, len(layers)))


LIBRISPEECH_SPLITS = ("train-clean-100", "dev-clean", "test-clean")


def resolve_audio_path(audio_path: str, data_root: Path) -> Path:
    """Resolve local, relative, or migrated Windows manifest audio paths."""
    raw_path = Path(audio_path).expanduser()
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.append(PROJECT_ROOT / raw_path)

    # Manifests are reproducibility artifacts and may be copied between hosts.
    # Preserve the split-relative suffix while replacing the old project root.
    normalized_parts = audio_path.replace("\\", "/").split("/")
    for split in LIBRISPEECH_SPLITS:
        if split in normalized_parts:
            split_index = len(normalized_parts) - 1 - normalized_parts[::-1].index(split)
            candidates.append(data_root / Path(*normalized_parts[split_index:]))
            break

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    expected = candidates[-1]
    raise FileNotFoundError(
        f"Audio file not found. Manifest path={audio_path!r}; "
        f"expected local path={str(expected)!r}. "
        "Copy LibriSpeech below --data_root or pass the correct --data_root."
    )


def load_records(
    path: Path,
    limit: int | None,
    max_duration: float,
    data_root: Path,
) -> list[dict]:
    records = [
        record
        for record in read_jsonl(path)
        if float(record["duration"]) <= max_duration
    ]
    if limit is not None:
        records = records[:limit]
    if not records:
        raise RuntimeError(f"No usable records in {path}")
    resolved_records = []
    for record in records:
        resolved_record = dict(record)
        resolved_record["audio_path"] = str(
            resolve_audio_path(str(record["audio_path"]), data_root)
        )
        resolved_records.append(resolved_record)
    return resolved_records


def main() -> None:
    args = parse_args()
    print("[startup] importing PyTorch and Transformers training dependencies...", flush=True)
    try:
        # On Windows, importing torch before the pyarrow dataset DLLs can make
        # Transformers.Trainer terminate with 0xC0000005 and no traceback.
        # datasets is already a project dependency, so initialize pyarrow first.
        import pyarrow.dataset  # noqa: F401
        import numpy as np
        import torch
        from transformers import (
            AutoFeatureExtractor,
            AutoConfig,
            AutoModelForCTC,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            Wav2Vec2CTCTokenizer,
            Wav2Vec2Processor,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError("Install the packages in requirements.txt first") from error

    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print(
            "[startup] cuDNN disabled by request; using native CUDA kernels",
            flush=True,
        )

    print(
        f"[startup] dependencies ready: torch={torch.__version__}, "
        f"cuda_available={torch.cuda.is_available()}",
        flush=True,
    )
    set_seed(args.seed)
    print(f"[startup] loading tokenizer from {args.vocab_path}", flush=True)
    tokenizer = Wav2Vec2CTCTokenizer(
        str(args.vocab_path),
        bos_token=None,
        eos_token=None,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    model_vocab_size = tokenizer_id_space_size(tokenizer)
    print(
        f"[startup] tokenizer: entries={len(tokenizer)}, "
        f"max_token_id={model_vocab_size - 1}, "
        f"model_vocab_size={model_vocab_size}, "
        f"added_vocab={tokenizer.get_added_vocab()}",
        flush=True,
    )
    print(
        f"[startup] loading pretrained checkpoint {args.model_name_or_path}; "
        "the first run downloads it from Hugging Face",
        flush=True,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)
    model_kwargs = {
        "vocab_size": model_vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "ctc_loss_reduction": args.ctc_loss_reduction,
        "ctc_zero_infinity": args.ctc_zero_infinity,
        "ignore_mismatched_sizes": True,
    }
    if args.ctc_head_type == "bilstm":
        if args.ctc_head_hidden_size <= 0 or args.ctc_head_num_layers <= 0:
            raise ValueError("BiLSTM head size and layer count must be positive")
        if not 0.0 <= args.ctc_head_dropout < 1.0:
            raise ValueError("--ctc_head_dropout must be in [0, 1)")
        config = AutoConfig.from_pretrained(args.model_name_or_path, **model_kwargs)
        if config.model_type != "wav2vec2":
            raise ValueError("The BiLSTM CTC head currently supports Wav2Vec2 only")
        config.ctc_head_hidden_size = args.ctc_head_hidden_size
        config.ctc_head_num_layers = args.ctc_head_num_layers
        config.ctc_head_dropout = args.ctc_head_dropout
        from ssl_asr.models import Wav2Vec2BiLSTMForCTC

        model = Wav2Vec2BiLSTMForCTC.from_pretrained(
            args.model_name_or_path,
            config=config,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForCTC.from_pretrained(
            args.model_name_or_path,
            **model_kwargs,
        )
    for option_name in ("mask_time_prob", "mask_feature_prob"):
        option_value = getattr(args, option_name)
        if option_value is not None:
            if not 0.0 <= option_value <= 1.0:
                raise ValueError(f"--{option_name} must be between 0 and 1")
            setattr(model.config, option_name, option_value)
    if args.mask_time_prob == 0.0 and args.mask_feature_prob == 0.0:
        model.config.apply_spec_augment = False
    print(
        "[startup] masking: "
        f"apply_spec_augment={getattr(model.config, 'apply_spec_augment', None)}, "
        f"mask_time_prob={getattr(model.config, 'mask_time_prob', None)}, "
        f"mask_feature_prob={getattr(model.config, 'mask_feature_prob', None)}",
        flush=True,
    )

    if args.freeze_encoder and args.freeze_transformer_layers:
        raise ValueError(
            "--freeze_encoder and --freeze_transformer_layers are mutually exclusive"
        )
    if args.freeze_encoder:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True
    elif hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()

    frozen_layer_indices: list[int]
    trainable_layer_indices: list[int]
    if args.freeze_encoder:
        layer_count = len(transformer_encoder_layers(model))
        frozen_layer_indices = list(range(layer_count))
        trainable_layer_indices = []
    else:
        frozen_layer_indices, trainable_layer_indices = (
            freeze_bottom_transformer_layers(
                model, args.freeze_transformer_layers
            )
        )

    named_parameters = list(model.named_parameters())
    feature_encoder_parameters = [
        parameter
        for name, parameter in named_parameters
        if ".feature_extractor." in name
    ]
    transformer_parameters = [
        parameter
        for name, parameter in named_parameters
        if ".encoder." in name
    ]
    if not feature_encoder_parameters:
        raise RuntimeError("Could not identify the CNN feature encoder parameters")
    if any(parameter.requires_grad for parameter in feature_encoder_parameters):
        raise RuntimeError("CNN feature encoder freeze check failed")
    if args.freeze_encoder:
        non_head_trainable = [
            name
            for name, parameter in named_parameters
            if parameter.requires_grad and not name.startswith("lm_head.")
        ]
        if non_head_trainable:
            raise RuntimeError(
                f"Frozen baseline has trainable non-head parameters: {non_head_trainable[:5]}"
            )
    elif not transformer_parameters or not any(
        parameter.requires_grad for parameter in transformer_parameters
    ):
        raise RuntimeError("Fine-tune mode does not have a trainable Transformer encoder")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base_model = getattr(model, getattr(model, "base_model_prefix", ""), None)
        feature_encoder = getattr(base_model, "feature_extractor", None)
        if feature_encoder is None:
            raise RuntimeError(
                "Could not identify the CNN feature encoder for checkpointing control"
            )
        for module in feature_encoder.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False
        print(
            "[startup] gradient checkpointing: non-reentrant Transformer only; "
            "frozen CNN excluded",
            flush=True,
        )

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"parameters: total={total_params:,}, trainable={trainable_params:,} "
        f"({trainable_params / total_params:.2%}); "
        f"cnn_feature_encoder_frozen=True; "
        f"ctc_head={args.ctc_head_type}; "
        f"frozen_transformer_layers={frozen_layer_indices}; "
        f"trainable_transformer_layers={trainable_layer_indices}",
        flush=True,
    )

    train_records = load_records(
        args.train_manifest,
        args.max_train_samples,
        args.max_duration_in_seconds,
        args.data_root,
    )
    eval_records = load_records(
        args.eval_manifest,
        args.max_eval_samples,
        args.max_duration_in_seconds,
        args.data_root,
    )
    validate_record_labels(
        train_records,
        tokenizer,
        model.config.vocab_size,
        args.train_manifest.stem,
    )
    validate_record_labels(
        eval_records,
        tokenizer,
        model.config.vocab_size,
        args.eval_manifest.stem,
    )
    train_dataset = ManifestDataset(train_records, processor, model.config.vocab_size)
    eval_dataset = ManifestDataset(eval_records, processor, model.config.vocab_size)
    print(
        f"[data] train={len(train_records)} utterances, "
        f"train_hours={sum(float(record['duration']) for record in train_records) / 3600:.3f}, "
        f"eval={len(eval_records)} utterances, "
        f"eval_hours={sum(float(record['duration']) for record in eval_records) / 3600:.3f}",
        flush=True,
    )

    def decode_output(prediction) -> tuple[Any, list[str], list[str]]:
        prediction_ids = prediction_ids_from_logits(
            prediction.predictions, tokenizer.pad_token_id
        )
        label_ids = prediction.label_ids.copy()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        predictions = processor.batch_decode(prediction_ids)
        references = processor.batch_decode(label_ids, group_tokens=False)
        return prediction_ids, references, predictions

    def compute_metrics(prediction) -> dict[str, float]:
        prediction_ids, references, predictions = decode_output(prediction)
        return {
            **compute_error_rates(references, predictions),
            **prediction_diagnostics(
                prediction.predictions,
                prediction_ids,
                predictions,
                tokenizer.pad_token_id,
            ),
        }

    training_kwargs = dict(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        fp16=args.fp16,
        # Checkpointing is enabled above so the frozen CNN can be excluded.
        # Letting Trainer enable it globally would checkpoint the CNN again.
        gradient_checkpointing=False,
        logging_steps=10,
        save_total_limit=args.save_total_limit,
        eval_accumulation_steps=args.eval_accumulation_steps,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        load_best_model_at_end=args.load_best_model_at_end,
    )
    argument_names = inspect.signature(TrainingArguments).parameters
    eval_strategy = "steps" if args.eval_steps is not None else "epoch"
    if "eval_strategy" in argument_names:
        training_kwargs["eval_strategy"] = eval_strategy
    else:
        training_kwargs["evaluation_strategy"] = eval_strategy
    training_kwargs["save_strategy"] = eval_strategy
    if args.load_best_model_at_end:
        training_kwargs["metric_for_best_model"] = "wer"
        training_kwargs["greater_is_better"] = False
    if args.eval_steps is not None:
        if args.eval_steps <= 0:
            raise ValueError("--eval_steps must be positive")
        training_kwargs["eval_steps"] = args.eval_steps
        training_kwargs["save_steps"] = args.eval_steps
    training_args = TrainingArguments(**training_kwargs)

    encoder_learning_rate = (
        args.encoder_learning_rate
        if args.encoder_learning_rate is not None
        else args.learning_rate
    )
    if encoder_learning_rate <= 0 or args.learning_rate <= 0:
        raise ValueError("Learning rates must be positive")
    if args.encoder_freeze_steps < 0:
        raise ValueError("--encoder_freeze_steps cannot be negative")
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(
            model,
            args.learning_rate,
            encoder_learning_rate,
            args.weight_decay,
        ),
        lr=args.learning_rate,
    )
    print(
        f"[train] learning rates: head={args.learning_rate:g}, "
        f"encoder={encoder_learning_rate:g}, "
        f"encoder_freeze_steps={args.encoder_freeze_steps}",
        flush=True,
    )

    class LowResourceStabilityCallback(TrainerCallback):
        def __init__(self) -> None:
            self.collapsed_evaluations = 0

        def on_step_begin(
            self, training_args, state, control, optimizer=None, lr_scheduler=None, **kwargs
        ):
            if optimizer is None or args.freeze_encoder:
                return control
            scheduled_lrs = (
                lr_scheduler.get_last_lr() if lr_scheduler is not None else None
            )
            for index, group in enumerate(optimizer.param_groups):
                if group.get("group_name") != "encoder":
                    continue
                if state.global_step < args.encoder_freeze_steps:
                    group["lr"] = 0.0
                elif scheduled_lrs is not None:
                    group["lr"] = scheduled_lrs[index]
            return control

        def on_evaluate(self, training_args, state, control, metrics=None, **kwargs):
            if args.collapse_patience_evaluations <= 0 or metrics is None:
                return control
            non_empty_ratio = float(metrics.get("eval_non_empty_ratio", 1.0))
            if non_empty_ratio <= args.collapse_non_empty_threshold:
                self.collapsed_evaluations += 1
            else:
                self.collapsed_evaluations = 0
            print(
                f"[diagnostic] step={state.global_step}, "
                f"non_empty_ratio={non_empty_ratio:.4f}, "
                f"collapsed_evaluations={self.collapsed_evaluations}",
                flush=True,
            )
            if self.collapsed_evaluations >= args.collapse_patience_evaluations:
                print(
                    "[diagnostic] stopping after consecutive collapsed evaluations",
                    flush=True,
                )
                control.should_training_stop = True
            return control

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CTCDataCollator(processor),
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[LowResourceStabilityCallback()],
    )
    print("[train] starting Trainer.train()", flush=True)
    trainer.train()
    trainer.save_model()
    processor.save_pretrained(args.output_dir)

    if args.diagnostic_train_samples > 0:
        diagnostic_count = min(args.diagnostic_train_samples, len(train_dataset))
        train_output = trainer.predict(
            torch.utils.data.Subset(train_dataset, range(diagnostic_count)),
            metric_key_prefix="train_diagnostic",
        )
        print(
            "[diagnostic] train subset: "
            f"samples={diagnostic_count}, "
            f"WER={train_output.metrics['train_diagnostic_wer']:.4f}, "
            f"CER={train_output.metrics['train_diagnostic_cer']:.4f}, "
            f"non_empty_ratio="
            f"{train_output.metrics['train_diagnostic_non_empty_ratio']:.4f}, "
            f"blank_frame_ratio="
            f"{train_output.metrics['train_diagnostic_blank_frame_ratio']:.4f}",
            flush=True,
        )

    output = trainer.predict(eval_dataset)
    _, references, predictions = decode_output(output)
    prediction_path = args.prediction_path or (
        PROJECT_ROOT / "results/predictions" / f"{args.output_dir.name}_dev.jsonl"
    )
    save_predictions(
        prediction_path,
        (
            {
                "id": record["id"],
                "reference": reference,
                "prediction": prediction,
                "audio_path": record["audio_path"],
                "duration": record["duration"],
            }
            for record, reference, prediction in zip(
                eval_records, references, predictions
            )
        ),
    )
    rates = compute_error_rates(references, predictions)
    append_metrics(
        args.metrics_path,
        {
            "experiment": args.output_dir.name,
            "split": args.eval_manifest.stem,
            "model": args.model_name_or_path,
            "freeze_encoder": args.freeze_encoder,
            "num_samples": len(eval_records),
            **rates,
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
    )
    print(f"evaluation: WER={rates['wer']:.4f}, CER={rates['cer']:.4f}")
    print(f"predictions: {prediction_path}")


if __name__ == "__main__":
    main()
