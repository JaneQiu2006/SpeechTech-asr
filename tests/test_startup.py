from pathlib import Path

from ssl_asr.manifest import audio_duration, scan_librispeech_split
from ssl_asr.text import normalize_text


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
