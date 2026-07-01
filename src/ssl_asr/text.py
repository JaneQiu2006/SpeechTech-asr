"""Text normalization and character vocabulary construction."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

_NOT_LETTER = re.compile(r"[^a-z' ]+")
_NOT_LETTER_NO_APOSTROPHE = re.compile(r"[^a-z ]+")
_SPACES = re.compile(r"\s+")


def normalize_text(text: str, keep_apostrophe: bool = True) -> str:
    """Normalize an English transcript for character-level CTC."""
    text = text.lower()
    pattern = _NOT_LETTER if keep_apostrophe else _NOT_LETTER_NO_APOSTROPHE
    text = pattern.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def build_vocabulary(
    texts: Iterable[str],
    output_path: str | Path,
) -> dict[str, int]:
    """Build and save a deterministic CTC character vocabulary."""
    characters = sorted(set("".join(texts)) - {" "})
    vocabulary = {character: index for index, character in enumerate(characters)}
    vocabulary["|"] = len(vocabulary)
    vocabulary["[UNK]"] = len(vocabulary)
    vocabulary["[PAD]"] = len(vocabulary)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(vocabulary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return vocabulary

