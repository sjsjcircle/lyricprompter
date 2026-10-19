"""parse_lyrics: raw lyrics text -> list[Line] in sung order (contract in docs/API_CONTRACTS.md).

Per input line: normalise -> header / repeat directive? -> inline "Chorus:" label -> trailing repeat
mark (x2, ×3, *2, (2번), (2回)) -> furigana markup -> tokenise by script and build Words:
hangul runs (korean.py), Latin runs (english.py), kana/kanji segments incl. attached digits and
symbols (japanese.py via OpenJTalk), other digits / & + @ % (numbers.py, in the neighbouring
language).  Non-sung tokens (ㅋㅋ, emoji, punctuation) produce no Word.  Words inside "( )" get
kind "backing"; a line that is entirely parenthesised gets Line.backing.

Languages: each line is detected with `detect_lang`; Han-only text follows the song's majority CJK
language (default_lang if it is "ko"/"ja"; else ko only if more lines contain hangul than kana, so
the default is ja); digit/symbol-only lines use default_lang, or with "auto" the song's majority.
Han in a line with hangul but no kana is hanja without a reading: units get roman "" and the word
needs_review (give a reading with {漢字|한자} markup).

Structure: blank lines (and symbol-only lines like "* * *", or a bare verse number "2.") separate
stanzas; a header such as [Chorus] / 【サビ】 / <Hook> / *Chorus* / Verse 1: / 후렴 sets Line.section of
the following lines and starts a stanza ("[Chorus] la la" and "Chorus: la la" carry the first lyric
line after the label).  A header directly followed by a non-section header ("[Chorus]" + "[Jimin]")
is one header "Chorus: Jimin".  Leading verse numbers ("1. ", "2) ") are not sung.
Repeats are expanded in sung order, copies carry flags ["repeat_of:<idx of the first instance>"]:
line marks (x2) copy the line, a standalone "(x2)" / "Repeat x3" / "2번 반복" / "※2回繰り返し" copies
the preceding stanza, "Repeat Chorus x2" / "후렴 반복" / "サビくり返し" or an empty header that names an
earlier section copies that section, "[Chorus x2]" sings its block twice.  A count N always means
"sung N times in a row here": N-1 copies of the line / preceding stanza, N copies of a named section.

Line.needs_review (and flag "text_review", which survives alignment) is set when any word needs
review: ambiguous number, hanja without reading, heteronym / unknown kanji reading, unsure acronym.
Other flags: "no_sung_content" (e.g. only ㅋㅋ), "unresolved_repeat" (directive without a target).
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field

import regex as re

from . import english, japanese, korean, numbers
from .markup import (
    Directive,
    Header,
    Ruby,
    classify_structure,
    extract_ruby,
    is_backing_line,
    is_section_label,
    paren_ranges,
    section_key,
    split_inline_header,
    split_repeat,
    strip_verse_mark,
)
from .model import AlignUnit, Line, Syllable, Word
from .normalize import (
    HANGUL_JAMO,
    HANGUL_SYL,
    JA_LETTER,
    LONG_MARKS,
    detect_lang,
    has_sung_letters,
    is_lrc,
    mms_clean,
    normalize_line,
    script_counts,
    strip_lrc,
)

LANGS = ("ko", "ja", "en")
_RE_ROMAN = re.compile(r"[a-z']*")
_RE_TOKEN = re.compile(
    rf"(?P<num>(?:{numbers.NUMBER_PATTERN})(?:(?:st|nd|rd|th)(?!\p{{L}}))?%?)"
    r"|(?P<acr>[A-Z]{1,3}(?:&[A-Z]{1,3})+(?!\p{L}))"                         # R&B, AT&T
    r"|(?P<en>(?:'(?=(?i:cause|cos|cuz|til|till|em|bout|round|n)(?!\p{L})))?"  # 'cause
    r"\p{Latin}[\p{Latin}\p{M}]*(?:'\p{Latin}[\p{Latin}\p{M}]*)*(?:(?<=[iI][nN])'(?!\p{L}))?)"  # goin'"
    rf"|(?P<ko>[{HANGUL_SYL}]+)"
    rf"|(?P<jamo>[{HANGUL_JAMO}]+)"
    rf"|(?P<ja>[{JA_LETTER}][{JA_LETTER}{LONG_MARKS}\p{{M}}]*)"
    r"|(?P<sym>[&+@%])"
    r"|(?P<sp>\s+)"
    r"|(?P<other>.)", re.S)
_RE_KANA = re.compile(r"[\p{Hiragana}\p{Katakana}]")
_SYMBOLS = {"&": ("and", "앤"), "+": ("plus", "플러스"), "@": ("at", "앳"), "%": ("percent", "퍼센트")}
_ACRONYM_AMP = "en"


# --------------------------------------------------------------------------- tokens -> words
@dataclass
class _Tok:
    kind: str
    a: int
    b: int
    ruby: Ruby | None = None


def _tokenize(text: str, ruby: list[Ruby]) -> list[_Tok]:
    toks: list[_Tok] = []
    pos = 0
    for r in [*sorted(ruby, key=lambda r: r.start), None]:
        end = r.start if r else len(text)
        toks.extend(_Tok(m.lastgroup or "other", m.start(), m.end()) for m in _RE_TOKEN.finditer(text, pos, end))
        if r:
            toks.append(_Tok("ruby", r.start, r.end, r))
            pos = r.end
    return toks


def _ruby_lang(reading: str) -> str:
    if re.search(rf"[{HANGUL_SYL}]", reading):
        return "ko"
    if _RE_KANA.search(reading):
        return "ja"
    return "en" if re.search(r"\p{Latin}", reading) else "ja"


class _LineBuilder:
    """Words for one display line."""

    def __init__(self, text: str, ruby: list[Ruby], lang: str):
        self.text, self.lang = text, lang
        self.toks = _tokenize(text, ruby)
        self.has_kana = bool(_RE_KANA.search(text))
        self.words: list[Word] = []
        self.seg: list | None = None       # pending Japanese segment [a, b, rubies]

    def _tok_lang(self, t: _Tok) -> str | None:
        if t.kind == "ruby" and t.ruby is not None:
            return _ruby_lang(t.ruby.reading)
        if t.kind == "ja":
            return self._ja_lang(t)
        return {"ko": "ko", "en": "en", "acr": "en"}.get(t.kind)

    def _ja_lang(self, t: _Tok) -> str:
        # Han-only runs are Japanese if the line is Japanese or has any kana, else hanja inside Korean
        return "ja" if self.lang == "ja" or self.has_kana else "ko"

    def _adjacent_lang(self, i: int, *, glued: bool) -> str | None:
        """Language of the next (then previous) token; glued=True only considers touching tokens."""
        t = self.toks[i]
        nxt = next((u for u in self.toks[i + 1:] if u.kind != "sp"), None)
        prv = next((u for u in reversed(self.toks[:i]) if u.kind != "sp"), None)
        for u, touching in ((nxt, nxt is not None and nxt.a == t.b), (prv, prv is not None and prv.b == t.a)):
            if u is not None and (touching or not glued) and (lang := self._tok_lang(u)):
                return lang
        return None

    def _add_ja(self, t: _Tok) -> None:
        if self.seg is None:
            self.seg = [t.a, t.b, []]
        self.seg[1] = t.b
        if t.ruby is not None:
            self.seg[2].append(t.ruby)

    def _flush(self) -> None:
        if self.seg is not None:
            a, b, rubies = self.seg
            self.words.extend(japanese.ja_words(self.text, a, b, rubies))
            self.seg = None

    def build(self) -> list[Word]:
        text = self.text
        for i, t in enumerate(self.toks):
            if t.kind in ("sp", "other"):
                continue                              # punctuation does not end a Japanese segment
            if t.kind == "num":
                lang = "en" if re.search(r"[a-z]$", text[t.a:t.b]) else (
                    self._adjacent_lang(i, glued=True) or self.lang)
            elif t.kind == "sym":
                lang = self._adjacent_lang(i, glued=False) or self.lang
            else:
                lang = self._tok_lang(t) or ""
            if lang == "ja" and t.kind in ("ja", "num", "sym", "ruby"):
                self._add_ja(t)
                continue
            self._flush()
            word = self._word(t, lang)
            if word is not None:
                self.words.append(word)
        self._flush()
        return self.words

    def _word(self, t: _Tok, lang: str) -> Word | None:
        s, a, b = self.text[t.a:t.b], t.a, t.b
        if t.kind == "en":
            return english.en_word(s, a)
        if t.kind == "ko":
            return Word(s, (a, b), "ko", "lex", s, korean.ko_syllables(s, a))
        if t.kind == "acr":                           # letter by letter, '&' reduced: ar-en-bi
            syls = [Syllable(c, (a + k, a + k + 1),
                             [AlignUnit(c.lower(), _ACRONYM_AMP if c == "&" else english.LETTER_NAMES[c.lower()])])
                    for k, c in enumerate(s)]
            return Word(s, (a, b), "en", "lex", s.lower(), syls)
        if t.kind == "ja":                            # hanja inside a Korean line: no reading available
            syls = [Syllable(c, (a + k, a + k + 1), [AlignUnit(c, "")]) for k, c in enumerate(s)]
            return Word(s, (a, b), "ko", "lex", "", syls, needs_review=True)
        if t.kind == "ruby" and t.ruby is not None:
            return _ruby_word(self.text, t.ruby, lang)
        if t.kind == "num":
            return _number_word(self.text, t, lang)
        if t.kind == "sym":
            en, ko = _SYMBOLS[s]
            units = korean.ko_units(ko) if lang == "ko" else english.en_units([en])
            return Word(s, (a, b), "ko" if lang == "ko" else "en", "sym", ko if lang == "ko" else en,
                        [Syllable(s, (a, b), units)], needs_review=s != "&")
        return None                                   # jamo (ㅋㅋ ㅠㅠ): not sung


def _number_word(text: str, t: _Tok, lang: str) -> Word:
    s, a, b = text[t.a:t.b], t.a, t.b
    num = numbers.parse_number(s)
    if num is None:                                   # pragma: no cover - the token regex guarantees a parse
        return Word(s, (a, b), lang, "num", "", [Syllable(s, (a, b), [AlignUnit(s, "")])], needs_review=True)
    if lang == "ko":
        reading, ambiguous = numbers.ko_number(num, text[b:])
        units = korean.ko_units(reading)
    else:
        lang = "en"
        words, ambiguous = numbers.en_number_words(num)
        reading, units = " ".join(words), english.en_units(words)
    return Word(s, (a, b), lang, "num", reading, [Syllable(s, (a, b), units)], needs_review=ambiguous)


def _ruby_word(text: str, ruby: Ruby, lang: str) -> Word:
    base = text[ruby.start:ruby.end]
    groups = [g for g in ruby.reading.split(".") if g]
    if lang == "ja":
        return japanese.ruby_word(text, ruby)
    if lang == "ko":
        if 1 < len(groups) == len(base):
            syls = [Syllable(c, (ruby.start + k, ruby.start + k + 1), korean.ko_units(g))
                    for k, (c, g) in enumerate(zip(base, groups))]
        else:
            syls = [Syllable(base, (ruby.start, ruby.end), korean.ko_units("".join(groups)))]
        return Word(base, (ruby.start, ruby.end), "ko", "lex", "".join(groups), syls)
    words = re.findall(r"[\p{Latin}']+", ruby.reading.replace(".", " "))
    syls = [Syllable(base, (ruby.start, ruby.end), english.en_units(words))]
    return Word(base, (ruby.start, ruby.end), "en", "lex", " ".join(words), syls)


def _finalize_units(text: str, words: list[Word]) -> None:
    """Resolve っ/ー placeholders across the line, guarantee >= 1 unit per syllable and clean romans."""
    units: list[AlignUnit] = []
    breaks: set[int] = set()
    prev_end: int | None = None
    for w in words:
        for s in w.syllables:
            if not s.units:
                s.units = [AlignUnit(s.text, "")]
            if prev_end is not None and re.search(r"[^\p{L}\p{M}\p{N}]", text[prev_end:s.span[0]]):
                breaks.add(len(units))
            units.extend(s.units)
            prev_end = s.span[1]
    japanese.resolve_placeholders(units, breaks)
    for u in units:
        if not _RE_ROMAN.fullmatch(u.roman):
            u.roman = mms_clean(u.roman)


def build_line(text: str, ruby: list[Ruby], lang: str, raw: str = "") -> Line:
    """One display line (idx / stanza / section are filled in by the caller)."""
    words = _LineBuilder(text, ruby, lang).build()
    ranges = paren_ranges(text)
    for w in words:
        if any(a < w.span[0] < b for a, b in ranges):
            w.kind = "backing"
    _finalize_units(text, words)
    review = any(w.needs_review for w in words)
    flags = (["text_review"] if review else []) + ([] if words else ["no_sung_content"])
    return Line(idx=0, text=text, raw=raw, backing=is_backing_line(text, ranges), words=words,
                needs_review=review, flags=flags)


# --------------------------------------------------------------------------- song structure
@dataclass
class _Row:
    raw: str
    text: str
    ruby: list[Ruby]
    repeat: int


@dataclass
class _Event:
    kind: str                                 # blank | header | directive | row
    raw: str = ""
    row: _Row | None = None
    header: Header | None = None
    directive: Directive | None = None


def _scan(raw: str) -> list[_Event]:
    events: list[_Event] = []
    for raw_line in raw.splitlines():
        norm = strip_verse_mark(normalize_line(raw_line))
        if not has_sung_letters(norm):
            events.append(_Event("blank"))
            continue
        st = classify_structure(norm)
        if isinstance(st, Header):
            events.append(_Event("header", raw_line.strip(), header=st))
            continue
        if isinstance(st, Directive):
            events.append(_Event("directive", raw_line.strip(), directive=st))
            continue
        header, body = split_inline_header(norm)
        if header is not None:
            events.append(_Event("header", raw_line.strip(), header=header))
        body, rep = split_repeat(body)
        text, ruby = extract_ruby(body)
        events.append(_Event("row", raw_line.strip(), row=_Row(raw_line.strip(), text, ruby, rep)))
    return _merge_headers(events)


def _merge_headers(events: list[_Event]) -> list[_Event]:
    """'[Chorus]' directly followed by a performer tag '[Jimin]' -> one header 'Chorus: Jimin'.

    Two adjacent section headers ('[Intro]' + '[Verse 1]') stay separate: the first is empty.
    """
    out: list[_Event] = []
    for ev in events:
        prev = out[-1] if out else None
        if (ev.kind == "header" and ev.header is not None and prev is not None and prev.kind == "header"
                and prev.header is not None
                and not (is_section_label(prev.header.label) and is_section_label(ev.header.label))):
            a, b = prev.header, ev.header
            if is_section_label(b.label) and not is_section_label(a.label):
                a, b = b, a
            out[-1] = _Event("header", f"{prev.raw}\n{ev.raw}", header=Header(f"{a.label}: {b.label}",
                                                                              max(a.count, b.count)))
            continue
        out.append(ev)
    return out


def _song_langs(rows: list[_Row], default_lang: str) -> tuple[str, str]:
    """(language for Han-only text, language for digit/symbol-only lines)."""
    counts: Counter[str] = Counter()
    for r in rows:
        kana, hangul, _, latin = script_counts(r.text)
        if kana or hangul:
            counts["ja" if kana >= hangul else "ko"] += 1
        elif latin:
            counts["en"] += 1
    cjk = default_lang if default_lang in ("ko", "ja") else ("ko" if counts["ko"] > counts["ja"] else "ja")
    if default_lang != "auto":
        return cjk, default_lang
    return cjk, max(LANGS, key=lambda k: counts[k]) if counts else "en"


@dataclass
class _Block:
    label: str | None
    keys: tuple[str, str] | None
    lines: list[int] = field(default_factory=list)


class _SongBuilder:
    def __init__(self, cjk: str, fallback: str):
        self.cjk, self.fallback = cjk, fallback
        self.lines: list[Line] = []
        self.stanza = 0
        self.pending_break = False
        self.blocks: list[_Block] = [_Block(None, None)]

    # -- emission
    def _start_line(self, new_stanza: bool) -> None:
        if (self.pending_break or new_stanza) and self.lines:
            self.stanza += 1
        self.pending_break = False

    def _append(self, line: Line) -> int:
        line.idx = len(self.lines)
        line.stanza = self.stanza
        self.lines.append(line)
        self.blocks[-1].lines.append(line.idx)
        return line.idx

    def _copy(self, idx: int) -> Line:
        src = self.lines[idx]
        root = next((int(f.split(":", 1)[1]) for f in src.flags if f.startswith("repeat_of:")), idx)
        dup = copy.deepcopy(src)
        dup.flags = [f for f in dup.flags if not f.startswith("repeat_of:")] + [f"repeat_of:{root}"]
        return dup

    def _copy_block(self, indices: list[int], copies: int, label: str | None) -> None:
        for _ in range(copies):
            self._start_line(new_stanza=True)
            for i in indices:
                dup = self._copy(i)
                dup.section = label
                self._append(dup)
        self.pending_break = True

    def _line_lang(self, text: str) -> str:
        lang = detect_lang(text)
        kana, hangul, _, _ = script_counts(text)
        if lang == "ja" and not kana and not hangul:
            return self.cjk
        return self.fallback if lang == "x" else lang

    def row(self, row: _Row) -> None:
        self._start_line(new_stanza=False)
        line = build_line(row.text, row.ruby, self._line_lang(row.text), row.raw)
        line.section = self.blocks[-1].label
        first = self._append(line)
        for _ in range(row.repeat - 1):
            self._append(self._copy(first))

    def blank(self) -> None:
        self.pending_break = True
        blk = self.blocks[-1]
        if blk.lines:          # the label carries on, but "Repeat <label>" only copies the first stanza
            self.blocks.append(_Block(blk.label, None))

    def header(self, h: Header, empty: bool) -> None:
        if empty and (src := self._find(h.label)) is not None:
            self.blocks.append(_Block(h.label, section_key(h.label)))
            self._copy_block(list(src.lines), h.count, h.label)
            self.blocks.append(_Block(h.label, None))
            return
        self.blocks.append(_Block(h.label, section_key(h.label)))
        self.pending_break = True

    def close_header_block(self, count: int) -> None:
        blk = self.blocks[-1]
        if count > 1 and blk.lines:
            self._copy_block(list(blk.lines), count - 1, blk.label)

    def directive(self, d: Directive, raw: str) -> None:
        if d.kind == "section":
            src = self._find(d.name)
            if src is not None:
                self.blocks.append(_Block(src.label, src.keys))
                self._copy_block(list(src.lines), d.copies, src.label)
                self.blocks.append(_Block(None, None))    # later lines are not part of that section
                return
        else:
            stanza_lines = [ln.idx for ln in self.lines if ln.stanza == self.stanza]
            if stanza_lines:
                self._copy_block(stanza_lines, d.copies, self.lines[stanza_lines[0]].section)
                return
            if not self.lines:
                return
        self._start_line(new_stanza=False)
        self._append(Line(idx=0, text=normalize_line(raw), raw=raw, section=self.blocks[-1].label,
                          needs_review=True, flags=["unresolved_repeat"]))

    def _find(self, label: str) -> _Block | None:
        full, base = section_key(label)
        for want, k in ((full, 0), (base, 1)):
            for blk in reversed(self.blocks):
                if blk.keys is not None and blk.keys[k] == want and blk.lines:
                    return blk
        return None


def parse_lyrics(raw: str, default_lang: str = "auto") -> list[Line]:
    """Lyrics text (plain or LRC) -> Lines in sung order with words/syllables/units, all times None."""
    if default_lang not in ("auto", *LANGS):
        raise ValueError(f"default_lang must be 'auto', 'ko', 'ja' or 'en', not {default_lang!r}")
    raw = raw.replace("\ufeff", "")                    # BOM (also mid-text, from concatenated files)
    if is_lrc(raw):
        raw = strip_lrc(raw)
    events = _scan(raw)
    sb = _SongBuilder(*_song_langs([e.row for e in events if e.row is not None], default_lang))
    open_header: Header | None = None
    for i, ev in enumerate(events):
        if ev.kind == "row" and ev.row is not None:
            sb.row(ev.row)
            continue
        if open_header is not None and (ev.kind != "blank" or sb.blocks[-1].lines):
            sb.close_header_block(open_header.count)
            open_header = None
        if ev.kind == "blank":
            sb.blank()
        if ev.kind == "header" and ev.header is not None:
            nxt = next((e.kind for e in events[i + 1:] if e.kind != "blank"), "eof")
            sb.header(ev.header, empty=nxt != "row")
            open_header = ev.header if nxt == "row" else None
        elif ev.kind == "directive" and ev.directive is not None:
            sb.directive(ev.directive, ev.raw)
    if open_header is not None:
        sb.close_header_block(open_header.count)
    return sb.lines
