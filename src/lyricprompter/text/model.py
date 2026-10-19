"""Lyrics data model shared by text parsing, alignment, timeline and the UI (via song.json).

Hierarchy: Song -> Line -> Word -> Syllable (display unit) -> AlignUnit (acoustic unit).
Spans are [start, end) character indices into `Line.text`. Times are seconds on the reference
(original mix) timeline and stay None until alignment fills them.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

STYLES = ("classic", "pop")      # live tracking styles (tracking/tracker.py; SongMeta.style)


@dataclass
class AlignUnit:
    reading: str                 # sung form in native script: "꽃" / "きょ" / "ev"
    roman: str                   # aligner tokens, [a-z'] only; "" => no acoustic token (time interpolated)
    start: float | None = None
    end: float | None = None
    score: float | None = None   # mean CTC posterior of its tokens (0..1)


@dataclass
class Syllable:
    text: str
    span: tuple[int, int]
    units: list[AlignUnit] = field(default_factory=list)
    start: float | None = None
    end: float | None = None

    @property
    def roman(self) -> str:
        return "".join(u.roman for u in self.units)


@dataclass
class Word:
    text: str
    span: tuple[int, int]
    lang: str                    # ko | ja | en | x
    kind: str = "lex"            # lex | num | sym | backing
    reading: str = ""
    syllables: list[Syllable] = field(default_factory=list)
    needs_review: bool = False

    @property
    def start(self) -> float | None:
        return next((s.start for s in self.syllables if s.start is not None), None)

    @property
    def end(self) -> float | None:
        return next((s.end for s in reversed(self.syllables) if s.end is not None), None)


@dataclass
class Line:
    idx: int
    text: str                    # display text (markup removed)
    raw: str = ""
    stanza: int = 0
    section: str | None = None
    backing: bool = False        # whole line is a backing vocal "(oh oh)"
    words: list[Word] = field(default_factory=list)
    start: float | None = None
    end: float | None = None
    confidence: float | None = None
    needs_review: bool = False
    flags: list[str] = field(default_factory=list)

    def syllables(self) -> list[Syllable]:
        return [s for w in self.words for s in w.syllables]


@dataclass
class SongMeta:
    id: str
    title: str = ""
    artist: str = ""
    default_lang: str = "auto"
    duration: float | None = None
    tuning: float = 0.0          # semitone fraction estimated from the mix
    stems: dict[str, str] = field(default_factory=dict)   # name -> path relative to song dir
    prep: dict[str, Any] = field(default_factory=dict)    # models / options / versions used
    style: str = "classic"       # live tracking style: "classic" (flexible tempo) | "pop" (steady tempo)


@dataclass
class Song:
    meta: SongMeta
    lines: list[Line] = field(default_factory=list)

    def iter_units(self) -> Iterator[tuple[int, int, int, int, AlignUnit]]:
        """(line_i, word_i, syl_i, unit_i, unit) in sung order."""
        for li, ln in enumerate(self.lines):
            for wi, w in enumerate(ln.words):
                for si, s in enumerate(w.syllables):
                    for ui, u in enumerate(s.units):
                        yield li, wi, si, ui, u

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Song:
        return cls(meta=SongMeta(**d["meta"]), lines=[line_from_dict(x) for x in d.get("lines", [])])


def line_from_dict(d: dict) -> Line:
    words = []
    for w in d.get("words", []):
        sylls = [Syllable(text=s["text"], span=tuple(s["span"]),
                          units=[AlignUnit(**u) for u in s.get("units", [])],
                          start=s.get("start"), end=s.get("end")) for s in w.get("syllables", [])]
        words.append(Word(text=w["text"], span=tuple(w["span"]), lang=w["lang"], kind=w.get("kind", "lex"),
                          reading=w.get("reading", ""), syllables=sylls, needs_review=w.get("needs_review", False)))
    keys = ("idx", "text", "raw", "stanza", "section", "backing", "start", "end", "confidence",
            "needs_review", "flags")
    return Line(words=words, **{k: d[k] for k in keys if k in d})
