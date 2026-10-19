"""Lyrics text front-end: raw (ko/ja/en, mixed) lyrics -> Line/Word/Syllable/AlignUnit trees.

`parse_lyrics` is the entry point (see `parse.py` for the rules); `detect_lang` classifies one line.
"""
from __future__ import annotations

from .normalize import detect_lang, normalize_line, normalize_lyrics, strip_lrc
from .parse import build_line, parse_lyrics

__all__ = ["build_line", "detect_lang", "normalize_line", "normalize_lyrics", "parse_lyrics", "strip_lrc"]
