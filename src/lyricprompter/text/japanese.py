"""Japanese: OpenJTalk (pyopenjtalk-plus) readings, spans from `g2p_mapping()`, mora units, romanisation.

- Words are OpenJTalk morphemes. Spans come from `pyopenjtalk.g2p_mapping()` (char_span into the text
  it was given; entries it cannot place get (0, 0) and are merged into the preceding morpheme). If it
  fails, chunks are retried separately and finally read as one unit with `needs_review`.
- Display units: one per mora for kana, one per kanji run (holding all its morae: 東京 = と.う.きょ.う)
  after an okurigana split (走り出せ -> 走|り|出|せ), one per number (2人, 1000).
- AlignUnit.reading is the orthographic kana mora (furigana); AlignUnit.roman follows the
  pronunciation (`pron`: ー long vowels, は->wa, へ->e, を->o) with a Hepburn-like mora table.
- っ / ー get placeholders that `resolve_placeholders` replaces per line, across words: っ -> next
  consonant (t before ch/ts), ー -> previous vowel; nothing usable -> "" (no acoustic token).
"""
from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from functools import cache

import regex as re

from .markup import Ruby
from .model import AlignUnit, Syllable, Word
from .normalize import HAN_LIKE, LONG_MARKS, kata2hira

GEMINATE = "<Q>"
LONG = "<L>"
PLACEHOLDERS = (GEMINATE, LONG)

_BASE = {
    **dict(zip("あいうえお", ("a", "i", "u", "e", "o"))),
    **dict(zip("かきくけこ", ("ka", "ki", "ku", "ke", "ko"))), **dict(zip("がぎぐげご", ("ga", "gi", "gu", "ge", "go"))),
    **dict(zip("さしすせそ", ("sa", "shi", "su", "se", "so"))), **dict(zip("ざじずぜぞ", ("za", "ji", "zu", "ze", "zo"))),
    **dict(zip("たちつてと", ("ta", "chi", "tsu", "te", "to"))), **dict(zip("だぢづでど", ("da", "ji", "zu", "de", "do"))),
    **dict(zip("なにぬねの", ("na", "ni", "nu", "ne", "no"))), **dict(zip("はひふへほ", ("ha", "hi", "fu", "he", "ho"))),
    **dict(zip("ばびぶべぼ", ("ba", "bi", "bu", "be", "bo"))), **dict(zip("ぱぴぷぺぽ", ("pa", "pi", "pu", "pe", "po"))),
    **dict(zip("まみむめも", ("ma", "mi", "mu", "me", "mo"))), **dict(zip("やゆよ", ("ya", "yu", "yo"))),
    **dict(zip("らりるれろ", ("ra", "ri", "ru", "re", "ro"))), **dict(zip("わゐゑをんゔ", ("wa", "i", "e", "o", "n", "vu"))),
    **dict(zip("ぁぃぅぇぉゃゅょゎゕゖ", ("a", "i", "u", "e", "o", "ya", "yu", "yo", "wa", "ka", "ke"))),
}
_KATA_ONLY = {"ヷ": "va", "ヸ": "vi", "ヹ": "ve", "ヺ": "vo"}
_SMALL_Y = frozenset("ゃゅょ")
_SMALL_V = frozenset("ぁぃぅぇぉゎ")
_PLAIN_VOWELS = frozenset("あいうえお")
_RE_KANA = re.compile(r"[\p{Hiragana}\p{Katakana}]")
_RE_HIRA = re.compile(r"\p{Hiragana}")
_RE_HAN = re.compile(rf"[{HAN_LIKE}ゝゞヽヾ]")
_RE_NOT_KANA = re.compile(r"[^\p{Hiragana}\p{Katakana}ー]")
_RE_MARK = re.compile(r"\p{M}")          # combining marks left after NFC (あ゙): display only

# Words whose OpenJTalk reading is often not what is sung (heteronyms, common lyric gikun).
HETERONYMS = frozenset((
    "明日", "今日", "昨日", "一日", "何", "故郷", "宇宙", "地球", "運命", "永遠", "永久", "本気", "未来", "瞬間", "理由",
    "季節", "現在", "強敵", "上手", "一時", "真実", "本当", "時間", "都会", "他人", "生命", "宿命", "刹那", "未知"))


# --------------------------------------------------------------------------- morae / romanisation
def _attaches(prev: str, small: str) -> bool:
    p = kata2hira(prev)
    if len(p) != 1 or p in "っんー" or p in _SMALL_Y or p in _SMALL_V or p not in _BASE:
        return False
    if p in _PLAIN_VOWELS:
        return (p == "う" and small in "ぃぇぉぁ") or (p == "い" and small == "ぇ")
    return True


def morae(kana: str) -> list[str]:
    """Split kana into morae; a small kana attaches to the preceding mora (ちょ, ティ, ウィ, ヴァ)."""
    out: list[str] = []
    for ch in kana:
        h = kata2hira(ch)
        if out and (h in _SMALL_Y or h in _SMALL_V) and _attaches(out[-1], h):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def mora_roman(m: str) -> str:
    """One mora -> [a-z] tokens, or a placeholder for っ / ー."""
    h = kata2hira(m)
    if h == "っ":
        return GEMINATE
    if h in ("ー", "〜", "~"):
        return LONG
    if len(h) == 1:
        return _BASE.get(h, _KATA_ONLY.get(m, ""))
    b, s = _BASE.get(h[0], _KATA_ONLY.get(m[0], "")), _BASE.get(h[1], "")
    if not b or not s:
        return b + s
    if h[1] in _SMALL_Y:
        return b[:-1] + (s[-1] if b in ("shi", "chi", "ji") else s)         # kya, sha, cha, ja, fyu
    v = s[-1] if h[1] != "ゎ" else "a"
    if b == "u":
        return "w" + v                                                     # wi we wo
    if b == "i":
        return "y" + v                                                     # ye
    if b in ("ku", "gu"):
        return b[0] + "w" + v                                              # kwa gwa
    if h[1] == "ぇ" and len(b) == 2 and b.endswith("i") and b != "ji":
        return b[0] + "y" + v                                              # kye nye
    return b[:-1] + v                                                      # fa ti du she tsa va


def resolve_placeholders(units: list[AlignUnit], breaks: set[int] | frozenset[int] = frozenset()) -> None:
    """Replace っ/ー placeholders in place. `breaks`: i in breaks <=> pause between units i-1 and i."""
    romans = [u.roman for u in units]
    for i, r in enumerate(romans):
        if r != GEMINATE:
            continue
        nxt = ""
        for j in range(i + 1, len(romans)):
            if j in breaks:
                break
            if romans[j] and romans[j] not in PLACEHOLDERS:
                nxt = romans[j]
                break
        romans[i] = "t" if nxt.startswith(("ch", "ts")) else (nxt[0] if nxt and nxt[0] not in "aeiou'" else "")
    for i, r in enumerate(romans):
        if r != LONG:
            continue
        prv = ""
        for j in range(i - 1, -1, -1):
            if romans[j] and romans[j] not in PLACEHOLDERS:
                prv = romans[j]
                break
            if j in breaks:
                break
        if i in breaks:
            prv = ""
        romans[i] = prv[-1] if prv and prv[-1] in "aeioun" else ""
    for u, r in zip(units, romans):
        u.roman = r


def okurigana_split(surface: str, reading: str) -> list[tuple[str, str]] | None:
    """食べ物 + たべもの -> [(食, た), (べ, べ), (物, もの)]; None when the kana do not line up."""
    parts = re.findall(rf"{_RE_HAN.pattern}+|[^{HAN_LIKE}ゝゞヽヾ]+", surface)
    pat = "".join("(.+?)" if _RE_HAN.match(p) else f"({re.escape(kata2hira(p))})" for p in parts)
    m = re.fullmatch(pat, kata2hira(reading))
    return list(zip(parts, m.groups())) if m else None


# --------------------------------------------------------------------------- OpenJTalk
@cache
def _ojt():
    import pyopenjtalk

    return pyopenjtalk


@dataclass
class _Group:
    s: int               # span in the analysis string
    e: int
    read: str            # katakana
    pron: str
    unknown: bool = False


def _mapping_groups(ana: str, offset: int = 0) -> list[_Group]:
    groups: list[_Group] = []
    carry_r = carry_p = ""
    for ent in _ojt().g2p_mapping(ana):
        s, e = ent["char_span"]
        rd, pr = ent["read"], ent["pron"].replace("’", "")
        if e <= s:                       # e.g. 十 inserted by number normalisation (2024 -> 二千二十四)
            if groups:
                groups[-1].read += rd
                groups[-1].pron += pr
            else:
                carry_r, carry_p = carry_r + rd, carry_p + pr
            continue
        g = _Group(s + offset, e + offset, carry_r + rd, carry_p + pr, bool(ent["is_unknown"]))
        carry_r = carry_p = ""
        if groups and g.s < groups[-1].e:
            last = groups[-1]
            last.e, last.read, last.pron = max(last.e, g.e), last.read + g.read, last.pron + g.pron
            last.unknown |= g.unknown
        else:
            groups.append(g)
    return groups


def _fallback_groups(ana: str) -> list[_Group]:
    groups: list[_Group] = []
    for m in re.finditer(r"[^\s\p{P}\p{S}]+", ana):
        with suppress(Exception):       # analyzer failure on odd input: degrade to one unit below
            groups.extend(_mapping_groups(m.group(), m.start()))
            continue
        chunk = m.group()
        try:
            read = "".join(n["read"] for n in _ojt().run_frontend(chunk))
            pron = _ojt().g2p(chunk, kana=True)
        except Exception:  # noqa: BLE001
            read = pron = chunk if not _RE_HAN.search(chunk) else ""
        groups.append(_Group(m.start(), m.end(), read, pron, unknown=True))
    return groups


def _analyze(ana: str) -> list[_Group]:
    try:
        groups = _mapping_groups(ana)
    except Exception:  # noqa: BLE001 - g2p_mapping raises ValueError when it cannot map spans
        groups = _fallback_groups(ana)
    for m in re.finditer(r"\d+(?:\.\d+)?", ana):     # one group per number: 2024 -> 二千二十四
        hit = [g for g in groups if g.s < m.end() and g.e > m.start()]
        if len(hit) > 1:
            first = hit[0]
            first.e = max(g.e for g in hit)
            first.read, first.pron = "".join(g.read for g in hit), "".join(g.pron for g in hit)
            first.unknown = any(g.unknown for g in hit)
            groups = [g for g in groups if g is first or g not in hit]
    return groups


def unit_morae(read: str, pron: str) -> tuple[list[str], list[str], bool]:
    """(unit readings, romans, aligned): romans follow the pronunciation morae.

    Normally read and pron have the same mora count (とう/とー, は/わ) and units keep the orthographic
    read mora.  When they differ (四年: よん/よ, sung よねん) the units follow pron and `aligned` is False,
    so the caller must not split the display by the read morae.
    """
    rm = morae(kata2hira(_RE_NOT_KANA.sub("", read)))
    pm = morae(kata2hira(_RE_NOT_KANA.sub("", pron)))
    if not pm or len(pm) == len(rm):
        return rm, [mora_roman(m) for m in (pm or rm)], True
    return pm, [mora_roman(m) for m in pm], False


# --------------------------------------------------------------------------- word building
def _units(rm: list[str], roms: list[str]) -> list[AlignUnit]:
    return [AlignUnit(kata2hira(m), r) for m, r in zip(rm, roms)]


def _syllables_for(text: str, src: list[int], a: int, surface: str, rm: list[str], roms: list[str],
                   atomic: bool, limit: int) -> list[Syllable]:
    """Display syllables for analysis span [a, a+len(surface)) whose reading morae are `rm`."""
    def span(i: int, j: int) -> tuple[int, int]:
        return src[i], _end(text, src[j - 1] + 1, limit)

    whole = [Syllable(text[slice(*span(a, a + len(surface)))], span(a, a + len(surface)), _units(rm, roms))]
    if atomic:
        return whole
    split = okurigana_split(surface, "".join(rm)) or [(surface, "".join(rm))]
    out: list[Syllable] = []
    k, pos = 0, a
    for chunk, r in split:
        n = len(morae(r))
        cm = morae(chunk)
        if _RE_HAN.search(chunk) or len(cm) != n:
            sp = span(pos, pos + len(chunk))
            out.append(Syllable(text[slice(*sp)], sp, _units(rm[k:k + n], roms[k:k + n])))
        else:
            q = pos
            for i, m in enumerate(cm):
                sp = span(q, q + len(m))
                out.append(Syllable(text[slice(*sp)], sp, _units(rm[k + i:k + i + 1], roms[k + i:k + i + 1])))
                q += len(m)
        k += n
        pos += len(chunk)
    return out if k == len(rm) else whole


def _end(text: str, e: int, limit: int) -> int:
    """Extend a display span end over the combining marks `_prepare` left out (up to `limit`)."""
    while e < limit and _RE_MARK.match(text[e]):
        e += 1
    return e


def _prepare(text: str, a: int, b: int) -> tuple[str, list[int]]:
    """Analysis string for text[a:b] + index map to the line.

    Digit-group commas are dropped (1,000 -> 1000) and spaces between numbers become pauses (OpenJTalk
    reads "1 2 3" as 千二百三).  A 〜/~ between digits is a range (1〜3回: から, both analysis chars map
    to the tilde).  Otherwise 〜/~ read as ー; a long mark after kanji or hiragana (空〜, 好きーー,
    すごーい) becomes a pause '、' so the word is analysed without it (OpenJTalk would otherwise treat
    空ー as an unknown word); `_attach_long_marks` adds it back as a ー unit.  Combining marks that NFC
    could not compose (あ゙) are left out of the analysis.
    """
    seg = text[a:b]
    drop = {g.start() + m.start() for g in re.finditer(r"\d{1,3}(?:,\d{3})+", seg)
            for m in re.finditer(",", g.group())}
    chars: list[str] = []
    src: list[int] = []
    for i, c in enumerate(seg):
        if i in drop or _RE_MARK.match(c):
            continue
        if c in "〜~" and 0 < i < len(seg) - 1 and seg[i - 1].isdigit() and seg[i + 1].isdigit():
            chars.extend("から")
            src.extend((a + i, a + i))
            continue
        if c in LONG_MARKS:
            prev = next((seg[k] for k in range(i - 1, -1, -1) if not _RE_MARK.match(seg[k])), "")
            c = "、" if prev and (_RE_HAN.match(prev) or _RE_HIRA.match(prev) or prev in LONG_MARKS and
                                 chars and chars[-1] == "、") else "ー"
        elif c.isspace() and 0 < i < len(seg) - 1 and seg[i - 1].isdigit() and seg[i + 1].isdigit():
            c = "、"
        chars.append(c)
        src.append(a + i)
    return "".join(chars), src


def _plain_words(text: str, a: int, b: int) -> list[Word]:
    ana, src = _prepare(text, a, b)
    if not ana.strip():
        return []
    words: list[Word] = []
    for g in _analyze(ana):
        surface = ana[g.s:g.e]
        if not _RE_KANA.search(g.pron) and not _RE_KANA.search(g.read):
            if _RE_HAN.search(surface):                     # kanji without any reading: keep, no tokens
                span = (src[g.s], _end(text, src[g.e - 1] + 1, b))
                syl = Syllable(text[span[0]:span[1]], span, [AlignUnit(surface, "")])
                words.append(Word(syl.text, span, "ja", "lex", "", [syl], needs_review=True))
            continue                                        # punctuation, ♪, ignored tokens
        rh = kata2hira(_RE_NOT_KANA.sub("", g.read))
        rm, roms, aligned = unit_morae(g.read, g.pron)
        if not rm:
            continue
        is_num = bool(re.search(r"\d", surface))
        ranged = len(set(src[g.s:g.e])) < g.e - g.s          # から standing for the range mark 〜
        syls = _syllables_for(text, src, g.s, surface, rm, roms, is_num or not aligned or ranged, b)
        span = (src[g.s], _end(text, src[g.e - 1] + 1, b))
        review = ranged
        if is_num:
            after = ana[g.e:g.e + 1]
            review = ranged or not (_RE_HAN.search(surface) or (after and (_RE_HAN.match(after) or after in "%回")))
        elif _RE_HAN.search(surface):
            stem = re.sub(r"\p{Hiragana}+$", "", surface)          # okurigana stripped: 想い -> 想
            review = ranged or g.unknown or surface in HETERONYMS or stem in HETERONYMS
        words.append(Word(text[span[0]:span[1]], span, "ja", "num" if is_num else "lex", rh, syls, review))
    return words


def _attach_long_marks(text: str, a: int, b: int, words: list[Word]) -> None:
    """An uncovered ー/〜/~ right after a word extends it with a long-vowel unit (空〜 -> そ.ら.ー)."""
    by_end = {w.span[1]: w for w in words}
    covered = {p for w in words for p in range(*w.span)}
    for p in range(a, b):
        if text[p] not in LONG_MARKS or p in covered or (w := by_end.pop(p, None)) is None:
            continue
        w.syllables.append(Syllable(text[p], (p, p + 1), [AlignUnit("ー", LONG)]))
        w.span = (w.span[0], p + 1)
        w.text = text[w.span[0]:p + 1]
        by_end[p + 1] = w
        covered.add(p)


def ruby_word(text: str, ruby: Ruby) -> Word:
    """Word for a furigana-annotated base with a kana reading ('.' = per-character split)."""
    base = text[ruby.start:ruby.end]
    groups = [kata2hira(_RE_NOT_KANA.sub("", g)) for g in ruby.reading.split(".")]
    full = "".join(groups)
    rm = morae(full)
    roms = [mora_roman(m) for m in rm]
    with suppress(Exception):
        pron = "".join(e["pron"].replace("’", "") for e in _ojt().g2p_mapping(full))
        pm = morae(kata2hira(_RE_NOT_KANA.sub("", pron)))
        if len(pm) == len(rm):     # accept only long-vowel rewrites (とう -> とー), never は -> わ
            roms = [mora_roman(p if p == "ー" else m) for m, p in zip(rm, pm, strict=True)]
    if len(groups) > 1 and len(groups) == len(base):
        syls, k = [], 0
        for i, (ch, g) in enumerate(zip(base, groups)):
            n = len(morae(g))
            syls.append(Syllable(ch, (ruby.start + i, ruby.start + i + 1), _units(rm[k:k + n], roms[k:k + n])))
            k += n
    else:
        src = list(range(ruby.start, ruby.end))
        syls = _syllables_for(text, src, 0, base, rm, roms, not _RE_KANA.search(base), ruby.end)
    return Word(base, (ruby.start, ruby.end), "ja", "lex", full, syls)


def ja_words(text: str, a: int, b: int, ruby: list[Ruby]) -> list[Word]:
    """Words for the Japanese segment text[a:b] of a line; `ruby` are kana-reading overrides inside it."""
    words: list[Word] = []
    pos = a
    for r in sorted(ruby, key=lambda r: r.start):
        if r.start > pos:
            words.extend(_plain_words(text, pos, r.start))
        words.append(ruby_word(text, r))
        pos = r.end
    if pos < b:
        words.extend(_plain_words(text, pos, b))
    _attach_long_marks(text, a, b, words)
    return words
