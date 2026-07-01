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
