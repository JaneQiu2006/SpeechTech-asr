"""CTC tokenizer construction with an explicit vocabulary contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

TokenizerMode = Literal["vocab_only", "legacy_special_tokens"]


def vocabulary_id_space_size(vocab_path: str | Path) -> int:
    vocabulary = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError(f"Vocabulary must be a non-empty JSON object: {vocab_path}")
    token_ids = list(vocabulary.values())
    if any(not isinstance(token_id, int) or token_id < 0 for token_id in token_ids):
        raise ValueError(f"Vocabulary contains invalid token IDs: {vocab_path}")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError(f"Vocabulary contains duplicate token IDs: {vocab_path}")
    return max(token_ids) + 1


def resolve_tokenizer_mode(
    vocab_path: str | Path,
    expected_vocab_size: int,
) -> TokenizerMode:
    base_size = vocabulary_id_space_size(vocab_path)
    if expected_vocab_size == base_size:
        return "vocab_only"
    if expected_vocab_size == base_size + 2:
        return "legacy_special_tokens"
    raise ValueError(
        f"Model vocab_size={expected_vocab_size} is incompatible with "
        f"vocabulary ID space {base_size}; expected {base_size} (vocab_only) "
        f"or {base_size + 2} (legacy_special_tokens)"
    )


def build_ctc_tokenizer(
    vocab_path: str | Path,
    mode: TokenizerMode = "vocab_only",
):
    from transformers import Wav2Vec2CTCTokenizer

    kwargs = {
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "word_delimiter_token": "|",
    }
    if mode == "vocab_only":
        kwargs.update(bos_token=None, eos_token=None)
    elif mode != "legacy_special_tokens":
        raise ValueError(f"Unsupported tokenizer mode: {mode}")
    tokenizer = Wav2Vec2CTCTokenizer(str(vocab_path), **kwargs)
    expected_size = vocabulary_id_space_size(vocab_path)
    if mode == "legacy_special_tokens":
        expected_size += 2
    actual_size = max(tokenizer.get_vocab().values()) + 1
    if actual_size != expected_size:
        raise RuntimeError(
            f"Tokenizer mode {mode} produced ID space {actual_size}, "
            f"expected {expected_size}"
        )
    return tokenizer
