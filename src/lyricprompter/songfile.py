"""Song package on disk: songs/<id>/{song.json, lyrics.txt, audio/*.wav, cache/, prep_report.json}."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from .config import SONG_SCHEMA, SONGS_DIR
from .text.model import Song

STEM_FILES = {"mix": "audio/mix.wav", "vocals": "audio/vocals.wav",
              "instrumental": "audio/instrumental.wav", "lead": "audio/lead.wav"}


def slugify(title: str) -> str:
    s = unicodedata.normalize("NFKC", title).strip().lower()
    s = re.sub(r"[^\w\-]+", "-", s, flags=re.UNICODE).strip("-")
    return s or "song"


def song_dir(song_id: str, root: Path | None = None) -> Path:
    return Path(root or SONGS_DIR) / song_id


def new_song_id(title: str, root: Path | None = None) -> str:
    base, root = slugify(title), Path(root or SONGS_DIR)
    sid, n = base, 2
    while (root / sid).exists():
        sid, n = f"{base}-{n}", n + 1
    return sid


def stem_path(sdir: Path, name: str) -> Path:
    return Path(sdir) / STEM_FILES[name]


def save_song(song: Song, sdir: Path) -> Path:
    sdir = Path(sdir)
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / "song.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"schema": SONG_SCHEMA, **song.to_dict()}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(path)
    return path


def load_song(sdir: Path) -> Song:
    d = json.loads((Path(sdir) / "song.json").read_text(encoding="utf-8"))
    if d.get("schema") != SONG_SCHEMA:
        raise ValueError(f"unsupported song.json schema {d.get('schema')!r} in {sdir}")
    return Song.from_dict(d)


def list_songs(root: Path | None = None) -> list[Path]:
    root = Path(root or SONGS_DIR)
    return sorted(p.parent for p in root.glob("*/song.json")) if root.is_dir() else []
