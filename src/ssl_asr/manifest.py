"""LibriSpeech discovery, metadata, and JSONL helpers."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterable, Iterator

from .text import normalize_text


def _flac_duration(path: Path) -> float:
    """Read duration from a FLAC STREAMINFO block without audio dependencies."""
    with path.open("rb") as stream:
        if stream.read(4) != b"fLaC":
            raise ValueError(f"Not a FLAC file: {path}")
        while True:
            header = stream.read(4)
            if len(header) != 4:
                break
            is_last = bool(header[0] & 0x80)
            block_type = header[0] & 0x7F
            block_size = int.from_bytes(header[1:4], "big")
            block = stream.read(block_size)
            if block_type == 0:
                if len(block) < 18:
                    raise ValueError(f"Invalid FLAC STREAMINFO: {path}")
                packed = int.from_bytes(block[10:18], "big")
                sample_rate = (packed >> 44) & 0xFFFFF
                total_samples = packed & 0xFFFFFFFFF
                if not sample_rate:
                    raise ValueError(f"Invalid FLAC sample rate: {path}")
                return total_samples / sample_rate
            if is_last:
                break
    raise ValueError(f"FLAC STREAMINFO not found: {path}")


def audio_duration(path: str | Path) -> float:
    """Return audio duration, using a dependency-free FLAC fast path."""
    path = Path(path)
    if path.suffix.lower() == ".flac":
        return _flac_duration(path)
    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError(
            f"soundfile is required to inspect non-FLAC audio: {path}"
        ) from error
    info = sf.info(path)
    return info.frames / info.samplerate


def scan_librispeech_split(split_dir: str | Path) -> list[dict]:
    """Scan an extracted LibriSpeech split into manifest records."""
    split_dir = Path(split_dir).resolve()
    if not split_dir.is_dir():
        raise FileNotFoundError(f"LibriSpeech split directory not found: {split_dir}")

    transcripts: dict[str, str] = {}
    for transcript_path in sorted(split_dir.rglob("*.trans.txt")):
        with transcript_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    utterance_id, text = line.split(maxsplit=1)
                except ValueError as error:
                    raise ValueError(
                        f"Malformed transcript at {transcript_path}:{line_number}"
                    ) from error
                transcripts[utterance_id] = normalize_text(text)

    audio_paths = sorted(split_dir.rglob("*.flac"))
    if not audio_paths:
        raise RuntimeError(f"No FLAC files found below {split_dir}")

    records = []
    for audio_path in audio_paths:
        utterance_id = audio_path.stem
        if utterance_id not in transcripts:
            raise RuntimeError(f"Missing transcript for {audio_path}")
        parts = utterance_id.split("-")
        if len(parts) != 3:
            raise RuntimeError(f"Unexpected LibriSpeech utterance ID: {utterance_id}")
        records.append(
            {
                "id": utterance_id,
                "audio_path": str(audio_path.resolve()),
                "text": transcripts[utterance_id],
                "duration": round(audio_duration(audio_path), 6),
                "speaker_id": parts[0],
                "chapter_id": parts[1],
                "utterance_id": parts[2],
            }
        )
    return records


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]

