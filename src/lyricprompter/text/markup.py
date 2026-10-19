"""Line-level lyric markup: section headers, repeat marks / directives, furigana (ruby), backing vocals.

All functions operate on normalised lines (`normalize.normalize_line`), so full-width brackets have
already become ASCII: （）->(), ［］->[], ｜->|.  《》 and 【】 are kept as is.
"""
from __future__ import annotations

from dataclasses import dataclass

import regex as re

from .normalize import HAN_LIKE, HANGUL_SYL, KANA

MAX_REPEAT = 16


@dataclass(frozen=True)
class Ruby:
    start: int           # [start, end) of the base in the display text
    end: int
    reading: str         # as written; '.' separates per-character readings


@dataclass(frozen=True)
class Header:
    label: str
    count: int = 1       # "[Chorus x2]": the block under this header is sung `count` times in a row


@dataclass(frozen=True)
class Directive:
    kind: str            # "block" (repeat the preceding stanza) | "section" (repeat a named section)
    name: str = ""
    copies: int = 1      # extra copies appended in sung order


# --------------------------------------------------------------------------- ruby / furigana
_RUBY_READING = rf"[{KANA}ー.・]+|[{HANGUL_SYL}.]+"
_RE_RUBY = re.compile(
    r"\{(?P<b1>[^{}|]+)\|(?P<r1>[^{}|]+)\}"                         # {漢字|かんじ}
    r"|\|(?P<b2>[^|《》]+)《(?P<r2>[^《》]+)》"                        # ｜漢字《かんじ》
    rf"|(?P<b3>[{HAN_LIKE}]+)(?:\((?P<r3>{_RUBY_READING})\)"         # 漢字(かんじ) 漢字（かんじ）
    rf"|\[(?P<r4>{_RUBY_READING})\]|《(?P<r5>[^《》]+)》)")            # 漢字[かんじ] 漢字《かんじ》


def extract_ruby(text: str) -> tuple[str, list[Ruby]]:
    """Remove furigana markup -> (display text, ruby spans into it)."""
    out: list[str] = []
    ruby: list[Ruby] = []
    pos = n = 0
    for m in _RE_RUBY.finditer(text):
        base = next(g for g in (m.group("b1"), m.group("b2"), m.group("b3")) if g is not None).strip()
        reading = next(g for g in (m.group(k) for k in ("r1", "r2", "r3", "r4", "r5")) if g is not None)
        reading = re.sub(r"[\s・]+", "", reading)
        if not base or not reading.replace(".", ""):
            continue
        chunk = text[pos:m.start()]
        out.append(chunk)
        n += len(chunk)
        ruby.append(Ruby(n, n + len(base), reading))
        out.append(base)
        n += len(base)
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), ruby


# --------------------------------------------------------------------------- repeat counts
# One convention everywhere: a count ("x2", "×3", "*2", "2x", "2번", "2회", "2回", "2 times", "twice")
# is how many times its target is sung in a row at that point.  A line mark ("love you x2") or a
# block repeat ("(x2)", "Repeat x3", "2번 반복", "※2回繰り返し") targets something just sung, so it adds
# N-1 copies (a bare "Repeat" / "반복" means N=2); a section repeat ("Repeat Chorus x2", "후렴 2번 반복")
# or an empty "[Chorus x2]" header adds N copies of that section.
_MARK = r"(?:[x×✕*]\s*\d{1,3}|\d{1,3}\s*(?:[x×✕]|번|회|回(?!目)|times?))"
_COUNT = rf"(?:{_MARK}|(?<!\p{{L}})(?:once|twice|thrice)(?!\p{{L}})|(?:한|두|세|네)\s*번)"
_WORD_COUNTS = {"once": 1, "twice": 2, "thrice": 3, "한": 1, "두": 2, "세": 3, "네": 4}
_RPT_WORD = r"(?:(?<![a-z])(?:repeat|rpt\.?)(?![a-z])|반복|くり返し|繰り返し|繰返し|リピート)"
_RE_RPT = re.compile(_RPT_WORD, re.I)

_RE_REPEAT_PAREN = re.compile(                         # "(x2)" "[2번]" "(2 times)" "(repeat x2)" "(반복)"
    rf"\s*[(\[]\s*(?:(?P<w1>{_RPT_WORD})\s*)?(?P<m>{_MARK})?(?:\s*(?P<w2>{_RPT_WORD}))?\s*[)\]]\s*$", re.I)
# An ASCII "x2" / "*2" needs a space or punctuation before it ("lovex2" stays a lyric), except right
# after Hangul / kana / kanji ("사랑해x2", "好きだよx2"): a Latin x glued to a CJK run is never sung.
_RE_REPEAT_BARE = re.compile(
    rf"(?:^|(?<=[\s\p{{P}}])|(?<=[\p{{Hangul}}{KANA}{HAN_LIKE}ー])(?=[x*]))"
    r"(?:[x*]\s*\d{1,3}|\d{1,3}\s*[x×✕])\s*$|\s*[×✕]\s*\d{1,3}\s*$", re.I)
_RE_BLOCK_REPEAT = re.compile(rf"^[(\[]?\s*(?P<c>{_MARK})\s*[)\]]?$", re.I)
_RE_COUNT_TAIL = re.compile(rf"(?:[(\[]\s*(?P<p>{_COUNT})\s*[)\]]|(?<![\p{{Latin}}\d])(?P<c>{_COUNT}))\s*$", re.I)
_RE_COUNT_HEAD = re.compile(rf"^\s*(?:[(\[]\s*(?P<p>{_COUNT})\s*[)\]]|(?P<c>{_COUNT})(?![\p{{Latin}}\d]))", re.I)


def _value(count: str | None, default: int) -> int:
    if not count:
        return default
    d = re.search(r"\d+", count)
    n = int(d.group()) if d else _WORD_COUNTS[re.sub(r"\s*번$", "", count.casefold())]
    return max(1, min(n, MAX_REPEAT))


def split_repeat(text: str) -> tuple[str, int]:
    """Trailing repeat mark of a lyric line: 'love you x2' -> ('love you', 2).

    Marks: x2 ×3 *2 2x, and in brackets (x2) [x2] (2번) (2回) (2회) (2 times) (repeat x2) (반복).
    A bare x2 / *2 follows a space or punctuation, or is glued to Korean / Japanese ('사랑해x2').
    """
    m = _RE_REPEAT_PAREN.search(text)
    if m and (m.group("m") or m.group("w1") or m.group("w2")):
        start, n = m.start(), _value(m.group("m"), 2)
    elif m := _RE_REPEAT_BARE.search(text):
        start, n = m.start(), _value(m.group(), 2)
    else:
        return text, 1
    if not text[:start].strip():
        return text, 1
    return text[:start].rstrip(), n


def _take_count(s: str) -> tuple[str, int | None]:
    """Remove one count at the end (else at the start) of `s` -> (rest, N or None)."""
    for rx in (_RE_COUNT_TAIL, _RE_COUNT_HEAD):
        if m := rx.search(s):
            return (s[:m.start()] + " " + s[m.end():]).strip(), _value(m.group("p") or m.group("c"), 1)
    return s.strip(), None


# --------------------------------------------------------------------------- headers / directives
_SECTION_WORD = (
    r"(?:pre[- ]?chorus|post[- ]?chorus|chorus|verse|hook|bridge|intro|outro|interlude|refrain|rap"
    r"|breakdown|instrumental|coda|후렴구?|브릿지|브리지|벌스|코러스|훅|랩|간주|전주|후주|인트로|아웃트로|\d+\s*절"
    r"|(?:大|落ち)?サビ|[A-D]メロ|イントロ|アウトロ|間奏|ブリッジ|ラップ|コーラス|\d+\s*番)")
_NUM_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
              "eight": "8", "nine": "9", "ten": "10", "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5"}
_NUM_ALT = "|".join(sorted(_NUM_WORDS, key=len, reverse=True))
_SECTION_NUM = rf"(?:\s*\d+|\s+(?:{_NUM_ALT})(?![a-z]))?"          # Verse 2, Verse Two, Verse II
_RE_SECTION_NAME = re.compile(rf"^{_SECTION_WORD}{_SECTION_NUM}$", re.I)
_RE_SECTION_START = re.compile(rf"^{_SECTION_WORD}(?![a-z])", re.I)
_RE_HEADER_WORD = re.compile(rf"^(?P<label>{_SECTION_WORD}{_SECTION_NUM}(?:\s*\([^()]*\))?)\s*:?$", re.I)
_RE_HEADER_INLINE = re.compile(rf"^(?P<label>{_SECTION_WORD}{_SECTION_NUM})\s*:\s*(?P<rest>\S.*)$", re.I)
_RE_HEADER_PREFIX = re.compile(r"^(?:\[(?P<a>[^\[\]]+)\]|【(?P<b>[^【】]+)】|<(?P<c>[^<>]+)>)\s*(?P<rest>\S.*)$")
_RE_STRONG_WRAP = re.compile(r"^\[(?P<a>[^\[\]]+)\]$|^【(?P<b>[^【】]+)】$")      # any label is a header
_RE_WEAK_WRAP = re.compile(                                                    # only section words
    r"^(?:\((?P<a>[^()]+)\)|<(?P<b>[^<>]+)>|〈(?P<c>[^〈〉]+)〉|《(?P<d>[^《》]+)》|\*+\s*(?P<e>[^*]+?)\s*\*+"
    r"|-+\s*(?P<f>.+?)\s*-+|[~〜]+\s*(?P<g>[^~〜]+?)\s*[~〜]+|=+\s*(?P<h>[^=]+?)\s*=+|#+\s*(?P<i>[^#]+))$")
_RE_DECOR = re.compile(r"[()\[\]【】<>〈〉《》※:]|\*(?!\s*\d)")
_RE_VERSE_MARK = re.compile(r"^\d{1,2}[.)](?:\s+(?=[\p{L}(\[{|《【])|$)")


def _unwrap(s: str) -> tuple[str, bool]:
    """'[Chorus]' -> ('Chorus', strong=True), '<Chorus>' / '*Chorus*' -> ('Chorus', False)."""
    for rx, strong in ((_RE_STRONG_WRAP, True), (_RE_WEAK_WRAP, False)):
        if m := rx.match(s):
            return next(g for g in m.groups() if g is not None).strip(), strong
    return s, False


def _directive(text: str) -> Directive | None:
    """'Repeat Chorus x2', '(Repeat x2)', '[Repeat 3 times]', '후렴 2번 반복', '※2回繰り返し' -> Directive.

    The text left after removing the repeat word, counts and brackets must be empty (block repeat)
    or a section name (section repeat): 'Repeat after me' / '같은 하루의 반복' stay lyrics.
    """
    if not _RE_RPT.search(text):
        return None
    rest = _RE_DECOR.sub(" ", _RE_RPT.sub(" ", text))
    n: int | None = None
    for _ in range(2):                                 # "Repeat x2 (2번)": at most two counts
        rest, c = _take_count(rest)
        if c is None:
            break
        n = min(c * (n or 1), MAX_REPEAT)
    name = re.sub(r"^(?:the|a)\s+", "", rest.strip(" .,-~"), flags=re.I)
    if not name:
        return Directive("block", copies=max(1, (n or 2) - 1))
    if _RE_SECTION_NAME.match(name):
        return Directive("section", name, n or 1)
    return None


def classify_structure(text: str) -> Header | Directive | None:
    """A whole normalised line that is a section header or a repeat directive, else None."""
    if m := _RE_BLOCK_REPEAT.match(text):              # "(x2)", "x3", "(2번)": the preceding stanza
        return Directive("block", copies=max(1, _value(m.group("c"), 2) - 1))
    if (d := _directive(text)) is not None:
        return d
    body, n_out = _take_count(text)                     # "[Hook] (x2)", "Chorus x2", "サビ×2"
    inner, strong = _unwrap(body)
    inner, n_in = _take_count(inner)                    # "[Chorus x2]"
    n = max(1, min((n_out or 1) * (n_in or 1), MAX_REPEAT))
    if hm := _RE_HEADER_WORD.match(inner):
        return Header(hm.group("label").strip(), n)
    if strong and re.search(r"[\p{L}\p{N}]", inner):    # any [..] / 【..】 line is a header
        return Header(inner.rstrip(":").strip(), n)
    return None


def split_inline_header(text: str) -> tuple[Header | None, str]:
    """'Chorus: I love you' / '[Chorus x2] I love you' / '【サビ】君が' / '<Hook> la' -> (Header, lyrics)."""
    if m := _RE_HEADER_INLINE.match(text):
        return Header(m.group("label").strip()), m.group("rest")
    if m := _RE_HEADER_PREFIX.match(text):
        label, n = _take_count(next(g for g in (m.group("a"), m.group("b"), m.group("c")) if g is not None))
        if re.search(r"\p{L}", label) and (m.group("c") is None or _RE_HEADER_WORD.match(label)):
            return Header(label.rstrip(":").strip(), n or 1), m.group("rest")
    return None, text


def strip_verse_mark(text: str) -> str:
    """Drop a leading verse number ('1. 꽃밭에' -> '꽃밭에', '2.' -> ''): hymn numbering is not sung."""
    return _RE_VERSE_MARK.sub("", text, count=1)


def is_section_label(label: str) -> bool:
    """'Chorus', 'Verse 2: RM', '후렴', 'Bメロ' yes; performer tags such as 'Jimin' no."""
    return bool(_RE_SECTION_START.match(label.strip()))


_SYNONYMS = {"후렴": "chorus", "후렴구": "chorus", "코러스": "chorus", "サビ": "chorus", "refrain": "chorus",
             "コーラス": "chorus", "브릿지": "bridge", "브리지": "bridge", "ブリッジ": "bridge", "벌스": "verse",
             "훅": "hook", "랩": "rap", "ラップ": "rap", "イントロ": "intro", "アウトロ": "outro", "인트로": "intro",
             "아웃트로": "outro"}
_RE_NUM_WORD = re.compile(rf"(?<![a-z])(?:{_NUM_ALT})(?![a-z])")


def section_key(label: str) -> tuple[str, str]:
    """(key with numbers, key without numbers) for matching 'Repeat Chorus' against '[Chorus 2]'."""
    s = re.sub(r"\([^()]*\)", " ", label.split(":")[0]).casefold()      # "Verse 1 (Jimin)"
    s = _RE_NUM_WORD.sub(lambda m: _NUM_WORDS[m.group()], s)            # "Verse Two" == "Verse 2"
    s = re.sub(r"[\s\p{P}]+", "", s)
    base = re.sub(r"\d+", "", s)
    base = _SYNONYMS.get(base, base)
    return base + "".join(re.findall(r"\d+", s)), base


# --------------------------------------------------------------------------- backing vocals
def paren_ranges(text: str) -> list[tuple[int, int]]:
    """Top-level balanced '(...)' pairs as (open index, close index)."""
    stack: list[int] = []
    out: list[tuple[int, int]] = []
    for i, c in enumerate(text):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            a = stack.pop()
            if not stack:
                out.append((a, i))
    return out


def is_backing_line(text: str, ranges: list[tuple[int, int]]) -> bool:
    """True when every letter/digit of the line sits inside parentheses: '(oh oh)'."""
    if not ranges:
        return False
    outside = list(text)
    for a, b in ranges:
        outside[a:b + 1] = " " * (b + 1 - a)
    return not re.search(r"[\p{L}\p{N}]", "".join(outside))
