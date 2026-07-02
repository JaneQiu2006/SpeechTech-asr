import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ssl_asr.representations import (
    CachedBiLSTMCTC,
    CachedCTCConfig,
    CachedSequenceDataset,
    collate_cached_sequences,
    unit_stream_statistics,
)


class _Encoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _Tokenizer:
    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return _Encoding([1, 2])


def _write_metadata(path: Path, representation: str, array_name: str) -> None:
    row = {
        "id": "utt-1",
        "text": "ab",
        "audio_path": "audio.flac",
        "duration": 1.0,
        "num_frames": 3,
        "feature_dim": 2,
        "array_path": array_name,
        "representation": representation,
        "source_model": "test",
        "source_layer": 6,
        "encoder_params": 10,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_cached_dataset_continuous_and_discrete_share_centroid_features(tmp_path):
    continuous_dir = tmp_path / "continuous"
    continuous_dir.mkdir()
    continuous = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float16)
    np.save(continuous_dir / "features.npy", continuous)
    continuous_metadata = continuous_dir / "metadata.jsonl"
    _write_metadata(continuous_metadata, "continuous", "features.npy")

    discrete_dir = tmp_path / "discrete"
    discrete_dir.mkdir()
    units = np.array([0, 1, 0], dtype=np.uint8)
    np.save(discrete_dir / "units.npy", units)
    discrete_metadata = discrete_dir / "metadata.jsonl"
    _write_metadata(discrete_metadata, "discrete", "units.npy")
    centers = np.array([[1, 2], [3, 4]], dtype=np.float32)

    continuous_item = CachedSequenceDataset(
        continuous_metadata, _Tokenizer()
    )[0]
    discrete_item = CachedSequenceDataset(
        discrete_metadata, _Tokenizer(), centers
    )[0]
    assert continuous_item["features"].shape == (3, 2)
    assert discrete_item["features"].tolist() == [[1, 2], [3, 4], [1, 2]]
    batch = collate_cached_sequences([continuous_item, continuous_item])
    assert batch["features"].shape == (2, 3, 2)
    assert batch["lengths"].tolist() == [3, 3]


def test_packed_bilstm_ignores_right_padding():
    torch.manual_seed(42)
    config = CachedCTCConfig(
        input_size=2,
        hidden_size=3,
        num_layers=1,
        vocab_size=4,
        dropout=0.0,
        blank_id=3,
        source_model="test",
        source_layer=6,
        representation="continuous",
    )
    model = CachedBiLSTMCTC(config).eval()
    short = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    short_logits = model(short, torch.tensor([2]))
    padded = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]],
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        ]
    )
    padded_logits = model(padded, torch.tensor([2, 3]))
    torch.testing.assert_close(short_logits[0], padded_logits[0, :2])


def test_unit_stream_statistics_reports_raw_and_deduplicated_rates():
    statistics = unit_stream_statistics(
        [np.array([0, 0, 1]), np.array([1, 2])],
        audio_seconds=2.0,
        codebook_size=4,
    )
    assert statistics["unit_used_types"] == 3
    assert statistics["unit_utilization"] == 0.75
    assert statistics["unit_tokens"] == 5
    assert statistics["unit_token_rate"] == 2.5
    assert statistics["unit_dedup_tokens"] == 4
    assert statistics["unit_dedup_token_rate"] == 2.0


def test_select_best_representation_layer(tmp_path):
    from scripts.select_best_representation_layer import select_best_layer

    for layer, wer in ((6, 0.3), (9, 0.2), (12, 0.25)):
        experiment = tmp_path / f"wav2vec2_layer{layer}_bilstm_ctc"
        experiment.mkdir()
        (experiment / "completion.json").write_text(
            json.dumps({"completed": True}), encoding="utf-8"
        )
        (experiment / "summary.json").write_text(
            json.dumps(
                {
                    "best_wer": wer,
                    "representation": "continuous",
                    "source_layer": layer,
                }
            ),
            encoding="utf-8",
        )
    assert select_best_layer(tmp_path, [6, 9, 12]) == (9, 0.2)
