"""Project-wide paths and constants.

Importing this module pins model caches to the project's `.cache/models/` (setdefault, so an
explicit environment variable still wins). Import it before torch / transformers / audio_separator.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.environ.get("LYRICPROMPTER_CACHE", PROJECT_ROOT / ".cache"))
MODELS_DIR = CACHE_DIR / "models"
SONGS_DIR = Path(os.environ.get("LYRICPROMPTER_SONGS", PROJECT_ROOT / "songs"))
DATA_DIR = PROJECT_ROOT / "data"

os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
os.environ.setdefault("AUDIO_SEPARATOR_MODEL_DIR", str(MODELS_DIR / "audio-separator"))

# Analysis grid shared by alignment, reference features and the live tracker.
ANALYSIS_SR = 16000
HOP = 640                      # 40 ms
FPS = ANALYSIS_SR / HOP        # 25 frames / s

# Canonical storage format for song audio.
CANONICAL_SR = 44100
CANONICAL_CHANNELS = 2

SONG_SCHEMA = 1


def ref_repo_dir() -> Path | None:
    """Local clone of laurenceyoon/live-lyrics-tracking (private, unlicensed: evaluation data only)."""
    candidates = [os.environ.get("LLT_REF_REPO"), PROJECT_ROOT / "live-lyrics-tracking",
                  PROJECT_ROOT / "_reference" / "live-lyrics-tracking"]
    for c in candidates:
        if c and (Path(c) / "resources").is_dir():
            return Path(c)
    return None
