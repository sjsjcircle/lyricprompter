"""Ingest normalisation, LRC stripping, script classes and language detection.

Every span in the data model indexes the *normalised* line, so normalisation happens exactly once,
per line, before any markup is interpreted:

- NFC, drop zero-width / soft-hyphen / BOM / variation-selector characters, curly quotes -> straight,
  tabs -> spaces, spacing ゛/゜ -> combining (so か゛ composes to が instead of NFKC's "か ゙");
- per-character NFKC (half-width kana -> full, full-width ASCII -> ASCII, ㌔ -> キロ) EXCEPT Hangul
  compatibility / half-width jamo (NFKC would turn ㅋㅋ into conjoining ᄏᄏ), then NFC again so that
  ｶﾞ -> カ + ゙ -> ガ recomposes;
- whitespace runs collapse to one space, the line is stripped.
"""
from __future__ import annotations

import unicodedata
from itertools import pairwise

import regex as re

# --------------------------------------------------------------------------- character classes
HANGUL_SYL = r"\uAC00-\uD7A3"
HANGUL_JAMO = r"\u1100-\u11FF\u3131-\u318E\uA960-\uA97F\uD7B0-\uD7FF\uFFA0-\uFFDC"   # conjoining + compatibility
KANA = r"\p{Hiragana}\p{Katakana}"
HAN_LIKE = r"\p{Han}々〆〇ヶヵ"              # kanji + iteration / counter marks that read like kanji
JA_LETTER = rf"{KANA}{HAN_LIKE}"
LONG_MARKS = "ー〜~"                          # 〜/~ are elongation marks inside Japanese runs

_RE_KANA = re.compile(rf"[{KANA}]")
_RE_HANGUL = re.compile(r"\p{Hangul}")
_RE_HAN = re.compile(rf"[{HAN_LIKE}]")
_RE_LATIN = re.compile(r"\p{Latin}")
_RE_WORDISH = re.compile(r"[\p{L}\p{N}]")

_DROP = dict.fromkeys((0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD,          # zero-width, soft hyphen
                       *range(0xFE00, 0xFE10), *range(0xE0100, 0xE01F0)), None)  # variation selectors
_QUOTES = str.maketrans({"’": "'", "‘": "'", "‛": "'", "`": "'", "´": "'", "ʼ": "'", "＇": "'", "′": "'",
                         "“": '"', "”": '"', "„": '"', "\t": " ", "゛": "\u3099", "゜": "\u309a"})
_KEEP = re.compile(r"[\u3130-\u318F\uFFA0-\uFFDC]")       # compatibility / half-width jamo


def normalize_line(line: str) -> str:
    """Canonical display form of one lyric line (see module docstring)."""
    s = unicodedata.normalize("NFC", line).translate(_DROP).translate(_QUOTES)
    if not s.isascii():
        s = "".join(c if c.isascii() or _KEEP.match(c) else unicodedata.normalize("NFKC", c) for c in s)
        s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_lyrics(text: str) -> str:
    """`normalize_line` applied to every line (line structure is preserved)."""
    return "\n".join(normalize_line(ln) for ln in text.splitlines())


_LATIN_EXTRA = str.maketrans({"ø": "o", "æ": "ae", "œ": "oe", "ł": "l", "đ": "d", "ð": "d", "þ": "th",
                               "ı": "i", "ŋ": "ng", "ħ": "h", "ŧ": "t", "ĸ": "k", "ſ": "s"})   # no NFKD form


def mms_clean(s: str) -> str:
    """Restrict to the MMS_FA vocabulary [a-z'] (lowercase, strip diacritics, drop the rest)."""
    s = unicodedata.normalize("NFKD", s.casefold().replace("’", "'").translate(_LATIN_EXTRA))
    return re.sub(r"[^a-z']", "", s)


def kata2hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def has_sung_letters(s: str) -> bool:
    return bool(_RE_WORDISH.search(s))


# --------------------------------------------------------------------------- language detection
def script_counts(text: str) -> tuple[int, int, int, int]:
    """(kana, hangul, han, latin) character counts."""
    return (len(_RE_KANA.findall(text)), len(_RE_HANGUL.findall(text)), len(_RE_HAN.findall(text)),
            len(_RE_LATIN.findall(text)))


def detect_lang(text: str) -> str:
    """Line language: "ja" | "ko" | "en" | "x".

    Priority rule (CJK scripts win over embedded English):
    kana -> ja, hangul -> ko (if both occur, the larger count wins, ties -> ja); otherwise Han -> ja
    (a Han-only line; `parse_lyrics` re-assigns it to the song's majority CJK language); otherwise
    Latin -> en; otherwise "x" (digits / symbols only).
    """
    kana, hangul, han, latin = script_counts(text)
    if kana or hangul:
        return "ja" if kana >= hangul else "ko"
    if han:
        return "ja"
    if latin:
        return "en"
    return "x"


# --------------------------------------------------------------------------- LRC
_LRC_TIME = r"\[(\d{1,3}):(\d{1,2}(?:[.:]\d{1,3})?)\]"
_RE_LRC_LEAD = re.compile(rf"^\s*(?:{_LRC_TIME}\s*)+")
_RE_LRC_ONE = re.compile(_LRC_TIME)
_RE_LRC_META = re.compile(r"^\s*\[(?:ti|ar|al|au|by|offset|re|ve|la|length|tool|#)\s*:[^\]]*\]\s*$", re.I)
_RE_LRC_TAG = re.compile(r"^\s*\[[a-z#][a-z0-9_#-]*\s*:[^\]]*\]\s*$")     # [id:] [kana:] [hash:] [qq:] ...
_RE_LRC_WORD = re.compile(r"<\d{1,3}:\d{1,2}(?:[.:]\d{1,3})?>")


def is_lrc(raw: str) -> bool:
    return any(_RE_LRC_LEAD.match(ln) for ln in raw.replace("\ufeff", "").splitlines())


def _strip_word_tags(text: str) -> str:
    tags = list(_RE_LRC_WORD.finditer(text))
    if not tags:
        return text
    # "E  <t>del  <t>weiss,  <t> e" style (every tag after >=2 spaces): a tag glued to the next letter
    # continues the word.  Enhanced-LRC style ("<t>When <t>the"): tags are just removed.
    glued = all(m.start() >= 2 and text[m.start() - 2:m.start()].isspace() for m in tags)
    return re.sub(r"\s*" + _RE_LRC_WORD.pattern if glued else _RE_LRC_WORD.pattern, "", text)


def strip_lrc(raw: str) -> str:
    """LRC -> plain lyrics in sung (time) order: metadata dropped, line/word time tags removed.

    Lines carrying several timestamps are emitted once per timestamp. A timed line with no text
    becomes a blank line (stanza break). Untimed tag lines with a lowercase key ([id:..], [kana:..])
    are metadata too; capitalised ones ([Chorus: Jimin]) stay as section headers.
    """
    rows: list[tuple[float | None, int, str]] = []
    multi = False
    for i, line in enumerate(raw.replace("\ufeff", "").splitlines()):
        if _RE_LRC_META.match(line) or _RE_LRC_TAG.match(line):
            continue
        m = _RE_LRC_LEAD.match(line)
        if not m:
            rows.append((None, i, _strip_word_tags(line).strip()))
            continue
        times = [int(a) * 60 + float(b.replace(":", ".")) for a, b in _RE_LRC_ONE.findall(m.group(0))]
        multi |= len(times) > 1
        text = "\n".join(t.strip() for t in _strip_word_tags(line[m.end():]).split("\\n"))  # literal \n
        rows.extend((t, i, text) for t in times)
    timed = [t for t, _, _ in rows if t is not None]
    if multi or any(b < a for a, b in pairwise(timed)):
        rows = sorted((r for r in rows if r[0] is not None), key=lambda r: (r[0], r[1]))
    return "\n".join(text for _, _, text in rows)
