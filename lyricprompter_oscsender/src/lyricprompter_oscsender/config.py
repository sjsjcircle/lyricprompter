"""Project-wide paths and constants.

The analysis grid and the tracking styles are those of lyricprompter (the tracking modules in
`features/`, `tracking/` and `live/audio_input.py` are copied from it unchanged)."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECTS_DIR = Path(os.environ.get("OSCSENDER_PROJECTS", PROJECT_ROOT / "projects"))

# Analysis grid shared by the reference features and the live tracker (lyricprompter.config).
ANALYSIS_SR = 16000
HOP = 640                      # 40 ms
FPS = ANALYSIS_SR / HOP        # 25 frames / s

# Canonical storage format for reference audio.
CANONICAL_SR = 44100
CANONICAL_CHANNELS = 2

STYLES = ("classic", "pop")    # live tracking styles (tracking/tracker.py)
MODES = ("mix", "vocal")       # what the live input carries: the full band mix, or the vocal mic only

PROJECT_SCHEMA = 1
