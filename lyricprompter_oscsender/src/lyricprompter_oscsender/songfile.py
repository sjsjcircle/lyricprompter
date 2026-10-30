"""Stem files of a project package (the names lyricprompter's feature cache and session use)."""
from __future__ import annotations

from pathlib import Path

STEM_FILES = {"mix": "audio/mix.wav", "vocals": "audio/vocals.wav"}


def stem_path(pdir: Path, name: str) -> Path:
    return Path(pdir) / STEM_FILES[name]
