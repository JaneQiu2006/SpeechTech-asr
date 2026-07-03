from scripts.analyze_deep_dive_errors import align_words
from scripts.paired_bootstrap_asr import bootstrap
from ssl_asr.tokenizer import resolve_tokenizer_mode, vocabulary_id_space_size


def test_word_alignment_counts_substitution_deletion_and_insertion():
    substitution = align_words("the cat".split(), "the dog".split())
    deletion = align_words("the small cat".split(), "the cat".split())
    insertion = align_words("the cat".split(), "the small cat".split())
    assert [item[0] for item in substitution].count("substitution") == 1
    assert [item[0] for item in deletion].count("deletion") == 1
    assert [item[0] for item in insertion].count("insertion") == 1


def test_ctc_vocabulary_contract_resolves_30_and_legacy_32(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text('{"a": 0, "|": 1, "[UNK]": 2, "[PAD]": 3}', encoding="utf-8")
    assert vocabulary_id_space_size(path) == 4
    assert resolve_tokenizer_mode(path, 4) == "vocab_only"
    assert resolve_tokenizer_mode(path, 6) == "legacy_special_tokens"


def test_paired_bootstrap_uses_aligned_utterances(tmp_path):
    import json

    rows_a = [
        {"id": "a", "reference": "the cat", "prediction": "the dog"},
        {"id": "b", "reference": "a bird", "prediction": "a bird"},
    ]
    rows_b = [
        {"id": "a", "reference": "the cat", "prediction": "the cat"},
        {"id": "b", "reference": "a bird", "prediction": "bird"},
    ]
    paths = []
    for name, rows in (("a", rows_a), ("b", rows_b)):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        paths.append(path)
    result = bootstrap("test", paths[0], paths[1], samples=100, seed=42)
    assert result["wer_a"] == result["wer_b"] == 0.25
    assert result["difference_b_minus_a"] == 0.0
