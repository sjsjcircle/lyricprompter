"""Lyric text -> display units (syllables / morae) with romanized tokens for MMS_FA-style CTC alignment.

Each line becomes a list of Unit(display, roman, lang). `roman` is the [a-z'] string fed to the
romanized CTC model; `display` is what the prompter highlights.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from uroman import Uroman

_U = Uroman()

HANGUL = re.compile(r"[가-힣]")
KANA_KANJI = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿々ー]")
LATIN = re.compile(r"[A-Za-z']")

SMALL_KANA = set("ァィゥェォャュョヮぁぃぅぇぉゃゅょゎ")


@dataclass
class Unit:
    display: str          # text shown/highlighted (syllable, mora group, or word piece)
    roman: str            # [a-z'] tokens for the CTC model (may be '' -> skipped / interpolated)
    lang: str             # 'ko' | 'ja' | 'en' | 'other'
    word_idx: int = 0     # index of the orthographic word inside the line
    meta: dict = field(default_factory=dict)


def _clean_roman(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).lower()
    s = re.sub(r"[^a-z']", "", s)
    return s


def roman(s: str, lcode: str | None = None) -> str:
    return _clean_roman(_U.romanize_string(s, lcode=lcode))


# ---------------- Korean ----------------
_g2pk = None


def _g2p_ko(text: str) -> str:
    """Pronunciation-level Hangul (빛나는 -> 빈나는). Syllable count is preserved by g2pk2 rules."""
    global _g2pk
    try:
        if _g2pk is None:
            from g2pk2 import G2p
            _g2pk = G2p()
        out = _g2pk(text)
        if len(HANGUL.findall(out)) == len(HANGUL.findall(text)):
            return out
    except Exception:
        pass
    return text  # fall back to orthography


def korean_units(segment: str, word_idx: int) -> list[Unit]:
    pron = HANGUL.findall(_g2p_ko(segment))
    orth = HANGUL.findall(segment)
    return [Unit(o, roman(p, "kor"), "ko", word_idx, {"pron": p}) for o, p in zip(orth, pron)]


# ---------------- Japanese ----------------
def _split_morae(kata: str) -> list[str]:
    morae: list[str] = []
    for ch in kata:
        if ch in SMALL_KANA and morae:
            morae[-1] += ch
        else:
            morae.append(ch)
    return morae


def japanese_units(segment: str, word_idx: int) -> list[Unit]:
    """One Unit per morpheme surface, with mora-level romanization in meta['morae'].

    Uses pyopenjtalk(-plus) NJD 'pron' (pronunciation katakana: は->ワ, を->オ). Kanji words become a
    single display unit (e.g. 名前 -> namae) whose internal mora times can be used for furigana.
    """
    import pyopenjtalk

    units: list[Unit] = []
    for node in pyopenjtalk.run_frontend(segment):
        surf = node["string"]
        pron = node["pron"].replace("’", "").replace("ヲ", "オ")  # particle を is sung "o"
        if not KANA_KANJI.search(surf) or pron in ("、", "。", "？", "！"):
            continue
        morae = _split_morae(pron)
        romas = []
        for m in morae:
            if m == "ー" and romas:          # long vowel -> repeat previous vowel
                prev = romas[-1]
                romas.append(prev[-1] if prev and prev[-1] in "aeiou" else "")
            else:
                romas.append(roman(m, "jpn"))
        units.append(Unit(surf, "".join(romas), "ja", word_idx, {"morae": morae, "mora_roman": romas}))
    return units


# ---------------- English ----------------
def english_units(word: str, word_idx: int) -> list[Unit]:
    try:
        import pyphen
        parts = pyphen.Pyphen(lang="en_US", left=1, right=1).inserted(word).split("-")
    except Exception:
        parts = [word]
    return [Unit(p, _clean_roman(p), "en", word_idx) for p in parts if p]


# ---------------- Mixed line ----------------
def line_units(line: str) -> list[Unit]:
    """Split a (possibly mixed ko/ja/en) lyric line into units by script runs."""
    units: list[Unit] = []
    for wi, word in enumerate(line.split()):
        # split each whitespace word into script runs
        for run in re.finditer(r"[가-힣]+|[぀-ヿ㐀-䶿一-鿿豈-﫿々ー]+|[A-Za-z']+", word):
            s = run.group(0)
            if HANGUL.match(s):
                units += korean_units(s, wi)
            elif KANA_KANJI.match(s):
                units += japanese_units(s, wi)
            else:
                units += english_units(s, wi)
    return units


if __name__ == "__main__":
    for ln in ["빛나는 별처럼 너를 기다려", "君の名前を呼んだ 夜空に響く声", "Shine on me tonight", "사랑해 Baby 君が好き"]:
        print(ln)
        for u in line_units(ln):
            print("   ", u.display, "->", u.roman, u.meta)
