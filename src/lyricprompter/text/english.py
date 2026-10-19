"""English: CMUdict syllable count + pyphen orthographic split (display); letters are the aligner tokens.

- count: min over CMUdict pronunciations (sung forms tend to the short one: ev'ry), else a vowel-group
  heuristic; elongations (yeahhh, Ooooh) are looked up after collapsing repeated letters;
- split: pyphen en_US with default margins, then merge the shortest neighbours / split at vowel-group
  boundaries until the piece count matches;
- acronyms (BTS, OK, USA) are sung letter by letter: one display syllable per letter; all-caps hums
  and elongations (HMM, SHH, BRR, AAH, OOO, YEAHHH) are words.
"""
from __future__ import annotations

from functools import cache
from itertools import pairwise

import regex as re

from .model import AlignUnit, Syllable, Word
from .normalize import mms_clean

LETTER_NAMES = dict(zip("abcdefghijklmnopqrstuvwxyz", (
    "ei", "bi", "si", "di", "i", "ef", "ji", "eich", "ai", "jei", "kei", "el", "em", "en", "ou", "pi", "kyu", "ar",
    "es", "ti", "yu", "vi", "dabelyu", "eks", "wai", "zi"), strict=True))
_VOWELS = "aeiouyàáâãäåèéêëìíîïòóôõöùúûüýÿæœø"
_RE_VG = re.compile(rf"[{_VOWELS}]+", re.I)
_RE_SILENT_E = re.compile(rf"[^{_VOWELS}]e$", re.I)


@cache
def _cmu() -> dict[str, list[list[str]]]:
    import cmudict

    return cmudict.dict()


@cache
def _hyphenator():
    import pyphen

    return pyphen.Pyphen(lang="en_US")


def cmu_counts(word: str) -> list[int]:
    """Syllable counts of every CMUdict pronunciation of `word` (lowercase); [] if unknown."""
    return [sum(p[-1].isdigit() for p in pron) for pron in _cmu().get(word, [])]


def _n_vowel_groups(chunk: str) -> int:
    n = len(_RE_VG.findall(chunk))
    if n > 1 and _RE_SILENT_E.search(chunk) and not chunk.lower().endswith("le"):
        n -= 1                                              # make, some -> 1 (but lit-tle)
    return n


def _collapse(word: str, min_run: int, keep: int) -> tuple[str, list[int]]:
    """Shorten runs of >= min_run equal letters to `keep` letters -> (form, index of each kept char)."""
    out: list[str] = []
    idx: list[int] = []
    for m in re.finditer(r"(.)\1*", word):
        n = len(m.group()) if len(m.group()) < min_run else keep
        out.append(m.group()[:n])
        idx.extend(range(m.start(), m.start() + n))
    return "".join(out), idx


def _lookup(word: str) -> tuple[str, list[int], list[int]]:
    """(form, its char -> word index map, CMU counts), trying elongation collapses (yeahhh -> yeah)."""
    low = "".join(c.lower() if len(c.lower()) == 1 else c for c in word)    # keep length: İ -> i̇ is 2 chars
    cands = [(low, list(range(len(low)))), _collapse(low, 3, 2), _collapse(low, 3, 1), _collapse(low, 2, 1)]
    for form, idx in cands:
        if counts := cmu_counts(form):
            return form, idx, counts
    form, idx = cands[2]
    return form, idx, []


def _vowel_split(chunk: str) -> list[str] | None:
    groups = list(_RE_VG.finditer(chunk))
    if len(groups) < 2:
        return None
    g1, g2 = groups[0], groups[1]
    cut = g1.end() + max(0, g2.start() - g1.end() - 1)     # one consonant goes to the right
    return [chunk[:cut], chunk[cut:]]


def split_word(form: str, n: int) -> list[str]:
    """Split `form` into `n` orthographic pieces (fewer if it has too few vowel groups)."""
    parts = [p for p in _hyphenator().inserted(form, hyphen="\x00").split("\x00") if p]
    if "".join(parts) != form:
        parts = [form]
    while len(parts) > n:
        i = min(range(len(parts) - 1), key=lambda j: len(parts[j]) + len(parts[j + 1]))
        parts[i:i + 2] = [parts[i] + parts[i + 1]]
    while len(parts) < n:
        i = max(range(len(parts)), key=lambda j: _n_vowel_groups(parts[j]))
        sp = _vowel_split(parts[i]) if _n_vowel_groups(parts[i]) > 1 else None
        if sp is None:
            break
        parts[i:i + 1] = sp
    return parts


def syllabify(word: str) -> tuple[list[str], list[str]]:
    """-> (display pieces with ''.join == word, lookup-form pieces used as readings)."""
    form, idx, counts = _lookup(word)
    n = max(1, min(counts)) if counts else max(1, _n_vowel_groups(form))
    pieces = split_word(form, n)
    bounds, j = [0], 0
    for p in pieces[:-1]:
        j += len(p)
        bounds.append(idx[j])
    bounds.append(len(word))
    return [word[a:b] for a, b in pairwise(bounds)], pieces


# consonant-only interjections that are hummed / hissed, not spelled (CMUdict knows hmm, hm, mm, sh, shh)
_RE_HUM = re.compile(r"h?m+h*|s+h+|b+r+|g+r+|ps+t|z{2,}|ts+k+|p+f+t*")
KNOWN_ACRONYMS = frozenset((
    "bts", "nct", "jyp", "sm", "yg", "dj", "mc", "tv", "cd", "dvd", "mv", "pd", "ceo", "sns", "pc", "gps", "sms",
    "mtv", "vip", "nyc", "kfc", "bbc", "cnn", "fbi", "mvp", "ktx", "lp", "ep", "bgm", "pr", "rnb", "hq"))


def acronym_kind(core: str) -> tuple[bool, bool]:
    """(sung letter by letter, uncertain).  'BTS'/'OK'/'USA' yes; 'LOVE'/'US'/'MY'/'HMM'/'AAH' no."""
    if not (core.isascii() and core.isalpha() and core.isupper() and 2 <= len(core) <= 6):
        return False, False
    low = core.lower()
    counts = cmu_counts(low)
    if not re.search(r"[aeiouy]", low):
        if 0 in counts or _RE_HUM.fullmatch(low):          # HMM, SHH, MMM, BRR, PSST
            return False, False
        return True, len(low) >= 3 and not counts and low not in KNOWN_ACRONYMS
    if counts:
        return all(c == len(core) for c in counts), False
    form, _, counts = _lookup(core)
    if counts and form != low:                             # elongated word: AAH, OOO, EHH, YEAHHH
        return False, False
    return len(core) <= 3, True


def en_units(words: list[str]) -> list[AlignUnit]:
    """One unit per syllable for spoken-out words (numbers, symbols, Latin furigana)."""
    units: list[AlignUnit] = []
    for w in words:
        core = w.strip("'")
        if not core:
            continue
        if acronym_kind(core)[0]:
            units.extend(AlignUnit(c.lower(), LETTER_NAMES[c.lower()]) for c in core)
            continue
        units.extend(AlignUnit(r, mms_clean(r)) for r in syllabify(core)[1])
    return units


def en_word(token: str, start: int) -> Word:
    """Word for a Latin token at line offset `start` (may carry clitic apostrophes: 'cause, goin')."""
    lead = len(token) - len(token.lstrip("'"))
    core = token.strip("'")
    end = start + len(token)
    acronym, unsure = acronym_kind(core)
    if acronym:
        pieces, readings = list(core), [c.lower() for c in core]
        romans = [LETTER_NAMES[c] for c in readings]
    else:
        pieces, readings = syllabify(core)
        romans = [mms_clean(r) for r in readings]
        romans = [r if r.strip("'") else "" for r in romans]
    pieces[0] = token[:lead] + pieces[0]
    pieces[-1] += token[lead + len(core):]
    syls, p = [], start
    for piece, rd, rm in zip(pieces, readings, romans):
        syls.append(Syllable(piece, (p, p + len(piece)), [AlignUnit(rd, rm)]))
        p += len(piece)
    return Word(token, (start, end), "en", "lex", core.lower(), syls, needs_review=unsure)
