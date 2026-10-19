"""Offline song preparation: separation -> lyrics parsing -> forced alignment -> song package.

Submodules are imported lazily (alignment and separation pull in torch), so
`from lyricprompter.prep import prepare_song` stays cheap until the function is called.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

Progress = Callable[[str, float, str], None]
"""(stage, percent 0..100, message). CLI form: ``PROGRESS <stage> <percent> <message>``."""

_LAZY = {
    "prepare_song": "pipeline",
    "separate_vocals": "separate",
    "DEFAULT_SEP_MODEL": "separate",
    "LEAD_SEP_MODEL": "separate",
    "align_lines": "align",
    "AlignReport": "align",
    "vocal_activity": "vad",
}

__all__ = ["DEFAULT_SEP_MODEL", "LEAD_SEP_MODEL", "AlignReport", "Progress", "align_lines",
           "prepare_song", "separate_vocals", "vocal_activity"]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
