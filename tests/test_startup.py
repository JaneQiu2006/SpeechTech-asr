from pathlib import Path

from ssl_asr.manifest import audio_duration, scan_librispeech_split
from ssl_asr.text import normalize_text


class _Encoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _FakeTokenizer:
    def __init__(self, vocabulary, encoded):
        self._vocabulary = vocabulary
        self._encoded = encoded
        self.add_special_tokens = None

    def get_vocab(self):
        return self._vocabulary

    def __call__(self, text, add_special_tokens):
        self.add_special_tokens = add_special_tokens
        return _Encoding(self._encoded)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_normalize_text():
    assert normalize_text("THIS, IS A TEST.") == "this is a test"
    assert normalize_text("  DON'T   STOP! ") == "don't stop"


def test_scan_small_librispeech_tree(tmp_path: Path):
    source = next(Path("data/dev-clean").rglob("*.flac"))
    utterance_id = source.stem
    speaker, chapter, _ = utterance_id.split("-")
    chapter_dir = tmp_path / speaker / chapter
    chapter_dir.mkdir(parents=True)
    target = chapter_dir / source.name
    target.write_bytes(source.read_bytes())
    (chapter_dir / f"{speaker}-{chapter}.trans.txt").write_text(
        f"{utterance_id} HELLO, WORLD!\n", encoding="utf-8"
    )

    records = scan_librispeech_split(tmp_path)
    assert len(records) == 1
    assert records[0]["text"] == "hello world"
    assert records[0]["duration"] == round(audio_duration(source), 6)


def test_tokenizer_id_space_uses_maximum_id_not_length():
    from scripts.train_ctc import tokenizer_id_space_size

    tokenizer = _FakeTokenizer({"a": 0, "[UNK]": 4}, [0])
    assert len(tokenizer.get_vocab()) == 2
    assert tokenizer_id_space_size(tokenizer) == 5


def test_ctc_label_encoding_disables_special_tokens_and_checks_range():
    import pytest

    from scripts.train_ctc import encode_ctc_labels

    tokenizer = _FakeTokenizer({"a": 0}, [0, 2])
    assert encode_ctc_labels(tokenizer, "A", 3, "utt-1") == [0, 2]
    assert tokenizer.add_special_tokens is False
    with pytest.raises(ValueError, match=r"utt-1.*outside.*\[0, 1\]"):
        encode_ctc_labels(tokenizer, "A", 2, "utt-1")


def test_trainer_logit_padding_decodes_as_ctc_blank():
    import importlib.util
    import sys

    import numpy as np

    script_path = PROJECT_ROOT / "scripts" / "train_ctc.py"
    spec = importlib.util.spec_from_file_location("train_ctc_test", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    logits = np.array(
        [[[0.1, 0.9, 0.0], [-100.0, -100.0, -100.0]]],
        dtype=np.float32,
    )
    prediction_ids = module.prediction_ids_from_logits(logits, blank_id=2)
    assert prediction_ids.tolist() == [[1, 2]]


def test_prediction_diagnostics_ignore_trainer_padding():
    import numpy as np

    from scripts.train_ctc import prediction_diagnostics

    logits = np.array(
        [[[0.1, 0.9, 0.0], [0.0, 0.1, 0.9], [-100.0, -100.0, -100.0]]],
        dtype=np.float32,
    )
    prediction_ids = np.array([[1, 2, 2]])
    diagnostics = prediction_diagnostics(logits, prediction_ids, ["a"], blank_id=2)
    assert diagnostics == {"non_empty_ratio": 1.0, "blank_frame_ratio": 0.5}


def test_filtered_duration_subset_reaches_effective_target_and_is_deterministic():
    from scripts.prepare_librispeech_subsets import select_duration_prefix

    records = [{"id": str(index), "duration": 900.0} for index in range(6)]
    records.append({"id": "too-long", "duration": 1800.0})
    first = select_duration_prefix(
        records, target_hours=1.0, seed=42, max_duration_in_seconds=1000
    )
    second = select_duration_prefix(
        records, target_hours=1.0, seed=42, max_duration_in_seconds=1000
    )
    assert first == second
    assert len(first) == 4
    assert all(record["id"] != "too-long" for record in first)
    assert sum(record["duration"] for record in first) == 3600.0


def test_freeze_bottom_transformer_layers():
    import pytest

    from scripts.train_ctc import freeze_bottom_transformer_layers

    class Parameter:
        requires_grad = True

    class Layer:
        def __init__(self):
            self.parameter = Parameter()

        def parameters(self):
            return [self.parameter]

    class Encoder:
        layers = [Layer() for _ in range(4)]

    class Base:
        encoder = Encoder()

    class Model:
        base_model_prefix = "base"
        base = Base()

    model = Model()
    frozen, trainable = freeze_bottom_transformer_layers(model, 2)
    assert frozen == [0, 1]
    assert trainable == [2, 3]
    assert [layer.parameter.requires_grad for layer in model.base.encoder.layers] == [
        False,
        False,
        True,
        True,
    ]
    with pytest.raises(ValueError, match=r"must be in \[0, 4\]"):
        freeze_bottom_transformer_layers(model, 5)


def test_evaluation_resolves_latest_complete_checkpoint(tmp_path: Path):
    from scripts.evaluate_e1_e5 import resolve_model_dir

    for step in (100, 300):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"weights")
    incomplete = tmp_path / "checkpoint-500"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}", encoding="utf-8")

    assert resolve_model_dir(tmp_path) == tmp_path / "checkpoint-300"


def test_evaluation_metric_skip_is_test_split_specific():
    from scripts.evaluate_e1_e5 import has_complete_metric

    rows = [
        {
            "experiment_id": "E2",
            "split": "dev_clean",
            "wer": "0.1",
            "cer": "0.03",
        },
        {
            "experiment_id": "E3",
            "split": "test_clean",
            "wer": "0.2",
            "cer": "0.05",
            "ctc_token_rate": "12.0",
            "ctc_label_bitrate_bps": "50.0",
        },
    ]
    assert not has_complete_metric(rows, "E2")
    assert has_complete_metric(rows, "E3")


def test_ctc_label_statistics():
    from scripts.evaluate_e1_e5 import compute_ctc_label_statistics

    class Encoding:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    class Tokenizer:
        ids = {"a b": [1, 27, 2], "a": [1]}

        def __call__(self, text, add_special_tokens):
            assert not add_special_tokens
            return Encoding(self.ids[text])

    rows = [
        {"prediction": "a b", "duration": 1.0},
        {"prediction": "a", "duration": 1.0},
    ]
    statistics = compute_ctc_label_statistics(rows, Tokenizer(), 30, 1.0)
    assert statistics["ctc_output_tokens"] == 4
    assert statistics["ctc_used_token_types"] == 3
    assert statistics["ctc_vocab_utilization"] == 0.1
    assert statistics["ctc_token_rate"] == 2.0
    assert statistics["ctc_token_throughput"] == 4.0
    assert statistics["ctc_token_entropy_bits"] == 1.5
    assert statistics["ctc_label_bitrate_bps"] == 3.0
