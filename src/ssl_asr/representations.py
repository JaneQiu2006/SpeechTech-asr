"""Shared components for cached continuous and discrete SSL representations."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .text import normalize_text


@dataclass(frozen=True)
class CachedCTCConfig:
    input_size: int
    hidden_size: int
    num_layers: int
    vocab_size: int
    dropout: float
    blank_id: int
    source_model: str
    source_layer: int
    representation: str
    codebook_size: int | None = None
    head_type: str = "bilstm"
    mlp_hidden_size: int = 512
    num_attention_heads: int = 4
    transformer_dim: int = 256
    unit_embedding_dim: int | None = None
    layer_indices: list[int] | None = None


class LayerScalarMixture(torch.nn.Module):
    """ELMo-style trainable scalar mixture over cached SSL layers."""

    def __init__(self, num_layers: int = 13) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.scalar_parameters = torch.nn.Parameter(torch.zeros(num_layers))

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.scalar_parameters, dim=0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[2] != len(self.scalar_parameters):
            raise ValueError("Mixture input must have shape [batch, time, layers, dim]")
        return torch.einsum("btld,l->btd", features, self.weights)


class CachedBiLSTMCTC(torch.nn.Module):
    """Serializable cached-feature CTC model supporting all deep-dive heads."""

    def __init__(self, config: CachedCTCConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = torch.nn.Dropout(config.dropout)
        input_size = config.input_size
        self.embedding = None
        if config.unit_embedding_dim is not None:
            if config.codebook_size is None:
                raise ValueError("Unit embedding requires codebook_size")
            self.embedding = torch.nn.Embedding(
                config.codebook_size, config.unit_embedding_dim
            )
            input_size = config.unit_embedding_dim
        self.layer_mixture = (
            LayerScalarMixture(len(config.layer_indices))
            if config.layer_indices is not None
            else None
        )
        head_type = config.head_type
        if head_type == "linear":
            self.head = torch.nn.Linear(input_size, config.vocab_size)
        elif head_type == "mlp":
            self.head = torch.nn.Sequential(
                torch.nn.Linear(input_size, config.mlp_hidden_size),
                torch.nn.GELU(),
                torch.nn.Dropout(config.dropout),
                torch.nn.Linear(config.mlp_hidden_size, config.vocab_size),
            )
        elif head_type == "bilstm":
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=True,
            )
            self.head = torch.nn.Linear(2 * config.hidden_size, config.vocab_size)
        elif head_type == "transformer":
            self.input_projection = torch.nn.Linear(input_size, config.transformer_dim)
            block = torch.nn.TransformerEncoderLayer(
                d_model=config.transformer_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=4 * config.transformer_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = torch.nn.TransformerEncoder(block, config.num_layers)
            self.head = torch.nn.Linear(config.transformer_dim, config.vocab_size)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Return padded logits, masking padding for sequence heads."""
        if self.embedding is not None:
            features = self.embedding(features.long())
        if self.layer_mixture is not None:
            features = self.layer_mixture(features)
        features = self.dropout(features)
        if self.config.head_type in {"linear", "mlp"}:
            return self.head(features)
        if self.config.head_type == "bilstm":
            packed = pack_padded_sequence(
                features, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed)
            hidden_states, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=features.shape[1]
            )
            return self.head(self.dropout(hidden_states))
        positions = torch.arange(features.shape[1], device=features.device)
        padding_mask = positions.unsqueeze(0) >= lengths.to(features.device).unsqueeze(1)
        hidden_states = self.transformer(
            self.input_projection(features), src_key_padding_mask=padding_mask
        )
        return self.head(self.dropout(hidden_states))

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary_model = output_dir / "model.pt.tmp"
        torch.save(self.state_dict(), temporary_model)
        temporary_model.replace(output_dir / "model.pt")
        temporary_config = output_dir / "config.json.tmp"
        temporary_config.write_text(
            json.dumps(asdict(self.config), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_config.replace(output_dir / "config.json")

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "CachedBiLSTMCTC":
        model_dir = Path(model_dir)
        raw_config = json.loads(
            (model_dir / "config.json").read_text(encoding="utf-8")
        )
        config = CachedCTCConfig(**raw_config)
        model = cls(config)
        state = torch.load(
            model_dir / "model.pt",
            map_location=map_location,
            weights_only=True,
        )
        # E10/E11 checkpoints predate the generic head abstraction.
        if "projection.weight" in state and "head.weight" not in state:
            state["head.weight"] = state.pop("projection.weight")
            state["head.bias"] = state.pop("projection.bias")
        model.load_state_dict(state)
        return model


def resolve_audio_path(audio_path: str, data_root: str | Path) -> Path:
    """Resolve migrated LibriSpeech paths below a local data directory."""
    path = Path(audio_path)
    if path.is_file():
        return path
    data_root = Path(data_root)
    parts = audio_path.replace("\\", "/").split("/")
    for split in ("train-clean-100", "dev-clean", "test-clean"):
        if split in parts:
            index = len(parts) - 1 - parts[::-1].index(split)
            candidate = data_root / Path(*parts[index:])
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Audio file not found locally: {audio_path}")


def read_cached_metadata(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise RuntimeError(f"Cached metadata is empty: {path}")
    return rows


def resolve_cached_array(metadata_path: Path, array_path: str) -> Path:
    path = Path(array_path)
    if path.is_absolute():
        return path
    return metadata_path.parent / path


class CachedSequenceDataset(torch.utils.data.Dataset):
    """Load variable-length cached features or unit indices."""

    def __init__(
        self,
        metadata_path: str | Path,
        tokenizer: Any,
        centroids: np.ndarray | None = None,
        discrete_embedding: bool = False,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.rows = read_cached_metadata(self.metadata_path)
        self.tokenizer = tokenizer
        self.centroids = centroids
        self.discrete_embedding = discrete_embedding
        representations = {str(row["representation"]) for row in self.rows}
        expected = "discrete" if centroids is not None or discrete_embedding else "continuous"
        if representations != {expected}:
            raise ValueError(
                f"{self.metadata_path}: expected {expected} metadata, "
                f"found {sorted(representations)}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        array_path = resolve_cached_array(
            self.metadata_path, str(row["array_path"])
        )
        array = np.load(array_path, allow_pickle=False)
        if self.centroids is not None or self.discrete_embedding:
            if array.ndim != 1:
                raise ValueError(f"Expected unit IDs in {array_path}, got {array.shape}")
            if array.size and (array.min() < 0 or array.max() >= len(self.centroids)):
                raise ValueError(f"Unit ID outside codebook range in {array_path}")
            unit_ids = array.astype(np.int64, copy=False)
            features = (
                unit_ids if self.discrete_embedding else self.centroids[unit_ids]
            )
        else:
            if array.ndim != 2:
                raise ValueError(
                    f"Expected continuous features in {array_path}, got {array.shape}"
                )
            features = array
        features = np.asarray(
            features, dtype=np.int64 if self.discrete_embedding else np.float32
        )
        labels = self.tokenizer(
            normalize_text(str(row["text"])),
            add_special_tokens=False,
        ).input_ids
        minimum_input_length = len(labels) + sum(
            left == right for left, right in zip(labels, labels[1:])
        )
        if len(features) < minimum_input_length:
            raise ValueError(
                f"{row['id']}: {len(features)} input frames cannot align "
                f"{len(labels)} CTC labels"
            )
        return {
            "id": str(row["id"]),
            "features": torch.from_numpy(features),
            "labels": torch.tensor(labels, dtype=torch.long),
            "text": normalize_text(str(row["text"])),
            "audio_path": str(row["audio_path"]),
            "duration": float(row["duration"]),
        }


class CachedLayerMixtureDataset(CachedSequenceDataset):
    """Join aligned per-layer caches without duplicating feature storage."""

    def __init__(self, metadata_paths: list[str | Path], tokenizer: Any) -> None:
        if not metadata_paths:
            raise ValueError("At least one layer metadata path is required")
        super().__init__(metadata_paths[0], tokenizer)
        self.layer_metadata_paths = [Path(path) for path in metadata_paths]
        self.layer_rows = [read_cached_metadata(path) for path in self.layer_metadata_paths]
        reference_ids = [str(row["id"]) for row in self.layer_rows[0]]
        for path, rows in zip(self.layer_metadata_paths[1:], self.layer_rows[1:]):
            if [str(row["id"]) for row in rows] != reference_ids:
                raise ValueError(f"Utterance alignment mismatch in {path}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        arrays = []
        for path, rows in zip(self.layer_metadata_paths, self.layer_rows):
            array_path = resolve_cached_array(path, str(rows[index]["array_path"]))
            arrays.append(np.load(array_path, allow_pickle=False))
        shapes = {array.shape for array in arrays}
        if len(shapes) != 1:
            raise ValueError(f"Layer frame shape mismatch for {item['id']}: {shapes}")
        # [time, layers, dimension], allowing the standard time-padding collator.
        item["features"] = torch.from_numpy(
            np.stack(arrays, axis=1).astype(np.float32, copy=False)
        )
        return item


def collate_cached_sequences(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor(
        [len(item["features"]) for item in items], dtype=torch.long
    )
    label_lengths = torch.tensor(
        [len(item["labels"]) for item in items], dtype=torch.long
    )
    return {
        "ids": [item["id"] for item in items],
        "features": torch.nn.utils.rnn.pad_sequence(
            [item["features"] for item in items], batch_first=True
        ),
        "lengths": lengths,
        "labels": torch.cat([item["labels"] for item in items]),
        "label_lengths": label_lengths,
        "references": [item["text"] for item in items],
        "audio_paths": [item["audio_path"] for item in items],
        "durations": [item["duration"] for item in items],
    }


def unit_stream_statistics(
    unit_sequences: Iterable[np.ndarray],
    audio_seconds: float,
    codebook_size: int,
) -> dict[str, float | int]:
    """Compute raw and run-length-deduplicated unit information rates."""
    if audio_seconds <= 0:
        raise ValueError("audio_seconds must be positive")
    if codebook_size <= 1:
        raise ValueError("codebook_size must be greater than one")
    raw_counts: Counter[int] = Counter()
    dedup_counts: Counter[int] = Counter()
    raw_tokens = 0
    dedup_tokens = 0
    for sequence in unit_sequences:
        values = np.asarray(sequence, dtype=np.int64)
        if values.ndim != 1:
            raise ValueError("Each unit sequence must be one-dimensional")
        if values.size and (values.min() < 0 or values.max() >= codebook_size):
            raise ValueError("Unit sequence contains an out-of-range ID")
        raw_counts.update(int(value) for value in values)
        raw_tokens += int(values.size)
        if values.size:
            keep = np.concatenate(([True], values[1:] != values[:-1]))
            deduplicated = values[keep]
            dedup_counts.update(int(value) for value in deduplicated)
            dedup_tokens += int(deduplicated.size)

    def entropy(counts: Counter[int], total: int) -> float:
        if not total:
            return 0.0
        return -sum(
            (count / total) * math.log2(count / total)
            for count in counts.values()
        )

    raw_entropy = entropy(raw_counts, raw_tokens)
    dedup_entropy = entropy(dedup_counts, dedup_tokens)
    token_rate = raw_tokens / audio_seconds
    dedup_token_rate = dedup_tokens / audio_seconds
    return {
        "unit_codebook_size": codebook_size,
        "unit_used_types": len(raw_counts),
        "unit_utilization": len(raw_counts) / codebook_size,
        "unit_perplexity": 2**raw_entropy,
        "unit_tokens": raw_tokens,
        "unit_token_rate": token_rate,
        "unit_entropy_bits": raw_entropy,
        "unit_bitrate_bps": token_rate * raw_entropy,
        "unit_fixed_bitrate_bps": token_rate
        * math.ceil(math.log2(codebook_size)),
        "unit_fixed_width_bitrate_bps": token_rate
        * math.ceil(math.log2(codebook_size)),
        "unit_dedup_tokens": dedup_tokens,
        "unit_dedup_token_rate": dedup_token_rate,
        "unit_dedup_entropy_bits": dedup_entropy,
        "unit_dedup_bitrate_bps": dedup_token_rate * dedup_entropy,
    }
