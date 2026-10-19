"""Korean: one hangul block = one display syllable = one AlignUnit.

Romanisation is orthographic uroman applied per syllable (the romaniser MMS was trained with; per
syllable == whole word for hangul). No pronunciation rewriting (g2pk2): liaison only moves the
attribution of a coda letter to the next syllable, which an aligner can do after alignment.
"""
from __future__ import annotations

from functools import cache

from .model import AlignUnit, Syllable
from .normalize import mms_clean


def is_hangul_syllable(ch: str) -> bool:
    return "가" <= ch <= "힣"


@cache
def _uroman():
    import uroman

    return uroman.Uroman()


@cache
def syllable_roman(ch: str) -> str:
    """'꽃' -> 'ggoc', '잎' -> 'ip' (MMS vocabulary)."""
    return mms_clean(_uroman().romanize_string(ch))


def ko_units(reading: str) -> list[AlignUnit]:
    """One unit per hangul syllable of `reading` (other characters are ignored)."""
    return [AlignUnit(c, syllable_roman(c)) for c in reading if is_hangul_syllable(c)]


def ko_syllables(word: str, start: int) -> list[Syllable]:
    """Display syllables for a run of hangul syllables starting at `start` in the line."""
    return [Syllable(c, (start + i, start + i + 1), [AlignUnit(c, syllable_roman(c))])
            for i, c in enumerate(word)]
