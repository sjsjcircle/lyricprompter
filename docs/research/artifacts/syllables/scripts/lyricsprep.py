"""Prototype: mixed ko/ja/en lyrics -> Song/Line/Word/Syllable(+Mora) with char spans
and an alignment-friendly romanization ([a-z'] for torchaudio MMS_FA) that maps back
to display units.  Tested on Python 3.12 / macOS arm64.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import regex as re  # needs \p{Script=...}

# --------------------------------------------------------------------------- model
@dataclass
class AlignUnit:            # smallest phonetic unit the aligner times (ko syllable / ja mora / en syllable)
    reading: str            # native-script sung form: "꼰" / "きょ" / "ev"
    roman: str              # aligner tokens, [a-z'] only; "" => no acoustic token (inherit time)
    t0: float | None = None
    t1: float | None = None

@dataclass
class Syllable:             # DISPLAY unit (what lights up on the prompter)
    text: str
    span: tuple[int, int]   # [start, end) into Line.text
    units: list[AlignUnit]  # 1 for hangul / kana mora / en syllable, N for kanji groups & numbers
    @property
    def roman(self): return "".join(u.roman for u in self.units)

@dataclass
class Word:
    text: str
    span: tuple[int, int]
    lang: str               # ko | ja | en | x
    kind: str = "lex"       # lex | num | sym | punct
    reading: str = ""       # whole-word sung form
    syllables: list[Syllable] = field(default_factory=list)
    needs_review: bool = False

@dataclass
class Line:
    idx: int
    text: str               # display text, markup (ruby / section / xN) removed
    raw: str
    stanza: int
    section: str | None = None
    backing: bool = False   # whole line is a backing vocal "(oh oh)"
    repeat: int = 1
    words: list[Word] = field(default_factory=list)

@dataclass
class Song:
    lines: list[Line]
    default_lang: str = "ko"
    def align_tokens(self, include_backing=False):
        """Flatten -> [(line_i, word_i, syl_i, unit_i, roman)], skipping empty romans."""
        out = []
        for li, ln in enumerate(self.lines):
            if ln.backing and not include_backing:
                continue
            for wi, w in enumerate(ln.words):
                if w.kind == "backing" and not include_backing:
                    continue
                for si, s in enumerate(w.syllables):
                    for ui, u in enumerate(s.units):
                        if u.roman:
                            out.append((li, wi, si, ui, u.roman))
        return out

# --------------------------------------------------------------------------- ingest normalisation
_ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u00ad"), None)
_QUOTES = str.maketrans({"’": "'", "‘": "'", "`": "'", "“": '"', "”": '"'})
def normalize_lyrics(text: str) -> str:
    """Canonical DISPLAY text; every span in the model indexes into this.
    Per-char NFKC (half-width kana -> full, full-width ASCII -> ASCII, ㌔ -> キロ) EXCEPT Hangul
    compatibility jamo (ㅋㅋ/ㅠㅠ would become conjoining jamo)."""
    text = unicodedata.normalize("NFC", text).translate(_ZW).translate(_QUOTES)
    out = "".join(c if 0x3130 <= ord(c) <= 0x318F else unicodedata.normalize("NFKC", c) for c in text)
    out = unicodedata.normalize("NFC", out)          # re-compose e.g. ｶﾞ -> カ+゙ -> ガ
    return "\n".join(re.sub(r"[ \t]+", " ", l).strip() for l in out.splitlines())

# --------------------------------------------------------------------------- markup
SECTION_RE = re.compile(r"^\s*[\[【]\s*([^\]】]+?)\s*[\]】]\s*$|^\s*(verse|chorus|hook|bridge|pre-?chorus|outro|intro|rap|후렴|間奏|サビ)\s*\d*\s*:?\s*$", re.I)
REPEAT_RE = re.compile(r"(?:^|\s)(?:\(\s*)?[x×✕*]\s*(\d+)\s*\)?\s*$|\s*\(\s*(\d+)\s*(?:x|×|번|回)\s*\)\s*$", re.I)
# ruby: {漢字|かんじ}  ｜漢字《かんじ》  漢字(かんじ) 漢字（かんじ） 漢字[かんじ]  ('.' inside reading = per-kanji split)
KANA = r"\p{Hiragana}\p{Katakana}ー・."
RUBY_RE = re.compile(
    rf"\{{(?P<b1>[^|{{}}]+)\|(?P<r1>[{KANA}]+)\}}"
    rf"|[|｜](?P<b2>[^《]+)《(?P<r2>[{KANA}]+)》"
    rf"|(?P<b3>[\p{{Han}}々〆ヶ]+)[(\[（](?P<r3>[{KANA}]+)[)\]）]")
BACKING_RE = re.compile(r"^\s*[(（](.+)[)）]\s*$")

def strip_markup(raw: str):
    """-> display text, ruby overrides [(start,end,reading)], repeat count."""
    m = REPEAT_RE.search(raw)
    repeat = 1
    if m:
        repeat = int(m.group(1) or m.group(2)); raw = raw[:m.start()]
    out, ruby, pos = [], [], 0
    for m in RUBY_RE.finditer(raw):
        out.append(raw[pos:m.start()])
        base = m.group("b1") or m.group("b2") or m.group("b3")
        rd = m.group("r1") or m.group("r2") or m.group("r3")
        s = sum(map(len, out)); ruby.append((s, s + len(base), rd)); out.append(base); pos = m.end()
    out.append(raw[pos:])
    return "".join(out), ruby, repeat

# --------------------------------------------------------------------------- script runs
TOKEN_RE = re.compile(
    r"(?P<acr>[A-Z](?:[&][A-Z])+)"                       # R&B, AT&T
    r"|(?P<en>[\p{Latin}]+(?:['’][\p{Latin}]+)*)"        # english word incl. contractions
    r"|(?P<num>[0-9０-９]+(?:[.,][0-9]+)*%?)"
    r"|(?P<ko>[\p{Hangul}]+)"
    r"|(?P<ja>[\p{Hiragana}\p{Katakana}\p{Han}ー々〆ヶ〜～]+)"
    r"|(?P<sp>\s+)"
    r"|(?P<sym>[&+@#%＆])"
    r"|(?P<punct>.)")

def line_lang(text, default):
    if re.search(r"[\p{Hiragana}\p{Katakana}]", text): return "ja"
    if re.search(r"\p{Hangul}", text): return "ko"
    if re.search(r"\p{Han}", text): return "ja" if default == "ja" else default
    if re.search(r"\p{Latin}", text): return "en"
    return default

# --------------------------------------------------------------------------- Korean
from functools import cache
@cache
def _uroman():
    import uroman as ur
    return ur.Uroman()

def mms_clean(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower()).replace("’", "'")
    return re.sub(r"[^a-z']", "", s)

@cache
def _g2pk():
    from g2pk2 import G2p   # side effects: nltk.download('cmudict'); pip-installs python-mecab-ko if missing!
    return G2p()

def ko_syllables(word: str, start: int, pron_aware=True) -> list[Syllable]:
    """1 hangul block = 1 display unit = 1 align unit. Pronunciation (연음/비음화) via g2pk2
    is used only if it preserves the block count; otherwise orthographic."""
    pron = word
    if pron_aware and re.fullmatch(r"\p{Hangul}+", word):
        try:
            p = _g2pk()(word)
            if len(p) == len(word) and re.fullmatch(r"\p{Hangul}+", p):
                pron = p
        except Exception:
            pass
    u = _uroman()
    return [Syllable(c, (start + i, start + i + 1), [AlignUnit(pc, mms_clean(u.romanize_string(pc)))])
            for i, (c, pc) in enumerate(zip(word, pron))]

def ko_jamo(ch: str):
    """Hangul syllable -> (choseong, jungseong, jongseong) compatibility jamo; arithmetic, no deps."""
    L = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"; V = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
    T = ["", *"ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"]
    n = ord(ch) - 0xAC00
    return L[n // 588], V[(n % 588) // 28], T[n % 28]

SINO = dict(zip("0123456789", ["영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]))
NATIVE_COUNTERS = ("살", "명", "개", "시", "번째", "마리", "잔", "번", "사람", "시간", "달", "가지", "곳", "송이", "걸음")
NATIVE = {1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉", 10: "열",
          20: "스물", 30: "서른", 40: "마흔", 50: "쉰", 60: "예순", 70: "일흔", 80: "여든", 90: "아흔"}

def ko_number(n: int, next_text: str) -> str:
    if 0 < n < 100 and next_text.startswith(NATIVE_COUNTERS):
        t, o = (n // 10) * 10, n % 10
        return (NATIVE.get(t, "") if t else "") + (NATIVE[o] if o else "")
        # (스물 + 한 = 스물한; 20 alone before counter = 스무 – left for review)
    from num2words import num2words
    return num2words(n, lang="ko")

# --------------------------------------------------------------------------- Japanese
SMALL = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ")
def morae(kana: str) -> list[str]:
    out: list[str] = []
    for ch in kana:
        if ch in SMALL and out and out[-1][-1] not in "っッーんン" and len(out[-1]) == 1:
            out[-1] += ch
        else:
            out.append(ch)
    return out

_B = dict(zip("あいうえおかきくけこがぎぐげごさしすせそざじずぜぞたちつてとだぢづでどなにぬねのはひふへほばびぶべぼぱぴぷぺぽまみむめもやゆよらりるれろわゐゑをんゔぁぃぅぇぉゃゅょゎゕゖ",
    "a i u e o ka ki ku ke ko ga gi gu ge go sa shi su se so za ji zu ze zo ta chi tsu te to da ji zu de do "
    "na ni nu ne no ha hi fu he ho ba bi bu be bo pa pi pu pe po ma mi mu me mo ya yu yo ra ri ru re ro "
    "wa i e o n vu a i u e o ya yu yo wa ka ke".split()))

def _kata2hira(s): return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)

def mora_roman(m: str) -> str:
    m = _kata2hira(m)
    if m in ("っ", "ー", "〜", "～"): return "?"             # resolved by context below
    b = _B.get(m[0], "")
    if len(m) == 1: return b
    s = _B.get(m[1], "")
    if m[1] in "ゃゅょ":
        return (b[:-1] if b in ("shi", "chi", "ji") else b[:-1] + "y") + s[-1]
    head = {"u": "w", "i": "y", "ku": "kw", "gu": "gw"}.get(b, b[:-1])             # ティ->ti ファ->fa ウィ->wi イェ->ye ヴァ->va
    return head + s

def romanize_morae(ms: list[str]) -> list[str]:
    return [mora_roman(m) for m in ms]       # "?" for っ/ー, resolved per LINE by resolve_context()

def resolve_context(units: list["AlignUnit"]) -> None:
    """っ -> next consonant (chotto: t), ー/〜 -> previous vowel; works across word boundaries."""
    r = [u.roman for u in units]; ms = [u.reading for u in units]
    for i, x in enumerate(r):
        if x != "?": continue
        if ms[i] in ("っ", "ッ"):
            nxt = next((y for y in r[i + 1:] if y and y != "?"), "")
            r[i] = "t" if nxt.startswith("ch") else (nxt[0] if nxt and nxt[0] not in "aiueo" else "")
        else:  # ー / 〜 : prolong previous vowel
            prv = next((y for y in reversed(r[:i]) if y and y != "?"), "")
            r[i] = prv[-1] if prv and prv[-1] in "aiueo" else ""
    for u, x in zip(units, r):
        u.roman = x

HAN = re.compile(r"[\p{Han}々〆ヶ]")
def okurigana_split(surface: str, reading_hira: str):
    """食べ物/たべもの -> [('食','た'),('べ','べ'),('物','もの')] ; None if no match."""
    parts = re.findall(r"[\p{Han}々〆ヶ]+|[^\p{Han}々〆ヶ]+", surface)
    pat = "".join("(.+?)" if HAN.match(p) else f"({re.escape(_kata2hira(p))})" for p in parts)
    m = re.fullmatch(pat, reading_hira)
    return list(zip(parts, m.groups())) if m else None

@cache
def _ojt():
    import pyopenjtalk   # pyopenjtalk-plus provides this module name
    return pyopenjtalk

def ja_nodes(text: str):
    """pyopenjtalk-plus frontend -> [(surface_span, read_kata, pron_kata)] mapped back to `text`
    even when OpenJTalk normalised the string (2人 -> 二人, ３ -> 3)."""
    nodes = [(n["string"], n["read"], n["pron"].replace("’", "")) for n in _ojt().run_frontend(text)]
    norm = unicodedata.normalize("NFKC", text)        # same length for kana/kanji/ascii lines
    out, pos, pend = [], 0, []
    for s, rd, pr in nodes:
        ns = unicodedata.normalize("NFKC", s)
        j = norm.find(ns, pos) if ns.strip() else -1
        if j == pos or (j > pos and pend):
            if pend:  # flush normalised gap (numbers etc.) as ONE group
                out.append(((pend[0][0], j), "".join(p[1] for p in pend), "".join(p[2] for p in pend)))
                pend = []
            out.append(((j, j + len(ns)), rd, pr)); pos = j + len(ns)
        elif re.fullmatch(r"[\p{P}\p{S}\s]*", ns):
            continue
        else:
            pend.append((pos, rd, pr))
    if pend:
        out.append(((pend[0][0], len(norm)), "".join(p[1] for p in pend), "".join(p[2] for p in pend)))
    return out

def ja_syllables(text: str, offset: int, ruby: list[tuple[int, int, str]]):
    """Return [Word] for a Japanese run. Kana -> one display unit per mora; kanji runs -> one display
    unit holding all of its morae (karaoke wipe interpolates inside it)."""
    words, cur = [], 0
    segs = []
    for (a, b, rd) in sorted(ruby):
        if a - offset >= cur:
            segs.append(("plain", cur, a - offset, None)); segs.append(("ruby", a - offset, b - offset, rd)); cur = b - offset
    segs.append(("plain", cur, len(text), None))
    for kind, a, b, rd in segs:
        if a >= b: continue
        if kind == "ruby":
            groups = rd.split(".")
            base = text[a:b]
            if len(groups) == len(base):     # per-kanji ruby given: 東京{とう.きょう}
                pieces = list(zip(base, groups))
            else:
                pieces = [(base, rd.replace(".", ""))]
            sylls, p = [], a
            for chunk, r in pieces:
                ms = morae(_kata2hira(r)); rs = romanize_morae(ms)
                sylls.append(Syllable(chunk, (offset + p, offset + p + len(chunk)), [AlignUnit(m, x) for m, x in zip(ms, rs)]))
                p += len(chunk)
            words.append(Word(base, (offset + a, offset + b), "ja", reading=rd.replace(".", ""), syllables=sylls))
            continue
        seg = text[a:b]
        ana = re.sub(r"[〜~]", "ー", seg)          # wave dash = lengthening; same length keeps spans valid
        for (s0, s1), read, pron in ja_nodes(ana):
            surf = seg[s0:s1]
            if not re.search(r"[\p{Han}\p{Hiragana}\p{Katakana}0-9０-９ー]", surf):
                continue
            rh, ph = _kata2hira(read), _kata2hira(pron)
            rm, pm = morae(rh), morae(ph)
            if len(pm) != len(rm): pm = rm                  # keep a 1:1 read<->pron mora map
            roms = romanize_morae(pm)
            S, W0 = [], offset + a + s0
            is_num = bool(re.search(r"[0-9０-９]", surf))
            split = None if is_num else okurigana_split(surf, rh)
            if split is None:
                split = [(surf, rh)]
            k, p = 0, W0
            for chunk, r in split:
                n = len(morae(r))
                if HAN.search(chunk) or is_num or len(morae(chunk)) != n:   # kanji / number / mismatch -> 1 unit, n morae
                    S.append(Syllable(chunk, (p, p + len(chunk)), [AlignUnit(m, x) for m, x in zip(rm[k:k + n], roms[k:k + n])]))
                else:  # kana chunk: one display unit per mora (chunk morae == reading morae)
                    cm = morae(chunk); q = p
                    for m_disp, m_r, x in zip(cm, rm[k:k + n], roms[k:k + n]):
                        S.append(Syllable(m_disp, (q, q + len(m_disp)), [AlignUnit(m_r, x)])); q += len(m_disp)
                k += n; p += len(chunk)
            words.append(Word(surf, (W0, W0 + len(surf)), "ja", "num" if is_num else "lex", rh, S, needs_review=is_num))
    return words

# --------------------------------------------------------------------------- English
@cache
def _cmu():
    import cmudict
    return cmudict.dict()

@cache
def _pyphen():
    import pyphen
    return pyphen.Pyphen(lang="en_US")

VG = re.compile(r"[aeiouy]+", re.I)
def n_vg(chunk: str) -> int:
    n = len(VG.findall(chunk))
    if n > 1 and re.search(r"[^aeiouy]e$", chunk, re.I): n -= 1     # some, wake -> 1
    return n
def en_count(word: str) -> int:
    w = word.lower().replace("’", "'")
    prons = _cmu().get(w) or _cmu().get(re.sub(r"(.)\1{2,}", r"\1\1", w))   # yeahhh -> yeahh
    if prons:
        return min(sum(p[-1].isdigit() for p in pr) for pr in prons)       # sung forms tend to the short one
    n = len(VG.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1: n -= 1   # silent e
    if re.search(r"in'$", w): pass
    return max(1, n)

def _vowel_split(chunk: str, k: int) -> list[str]:
    """split chunk into k pieces at vowel-group boundaries (maximal-onset-ish: 1 consonant to the right)."""
    out, rest = [], chunk
    while k > 1:
        groups = list(VG.finditer(rest))
        if len(groups) < 2: break
        g1, g2 = groups[0], groups[1]
        cut = g1.end() + max(0, (g2.start() - g1.end()) - 1)
        out.append(rest[:cut]); rest = rest[cut:]; k -= 1
    return out + [rest]

def en_syllables(word: str, start: int) -> list[Syllable]:
    if word.isupper() and 1 < len(word) <= 5:      # acronym: BTS -> B|T|S, sung letter by letter
        names = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "ei bi si di i ef dzi eitch ai dzei kei el em en ou pi kyu ar es ti yu vi dabelyu eks wai zi".split()))
        return [Syllable(c, (start + i, start + i + 1), [AlignUnit(c, names[c])]) for i, c in enumerate(word)]
    n = en_count(word)
    parts = _pyphen().inserted(word, hyphen="\x00").split("\x00")
    while len(parts) > n:                           # merge shortest neighbours
        i = min(range(len(parts) - 1), key=lambda j: len(parts[j]) + len(parts[j + 1]))
        parts[i:i + 2] = [parts[i] + parts[i + 1]]
    while len(parts) < n:                           # split the piece with most vowel groups
        i = max(range(len(parts)), key=lambda j: n_vg(parts[j]))
        sp = _vowel_split(parts[i], 2)
        if len(sp) < 2: break
        parts[i:i + 1] = sp
    out, p = [], start
    for part in parts:
        out.append(Syllable(part, (p, p + len(part)), [AlignUnit(part, mms_clean(part))])); p += len(part)
    return out

EN_SYM = {"&": "and", "＆": "and", "+": "plus", "%": "percent", "@": "at"}
KO_SYM = {"&": "앤", "＆": "앤", "+": "플러스", "%": "퍼센트"}
JA_SYM = {"&": "あんど", "＆": "あんど", "+": "ぷらす", "%": "ぱーせんと"}

# --------------------------------------------------------------------------- driver
def parse_song(raw_text: str, default_lang="ko", pron_aware_ko=True) -> Song:
    lines, stanza, section, prev_blank = [], 0, None, False
    for raw in normalize_lyrics(raw_text).splitlines():
        if not raw.strip():
            if not prev_blank: stanza += 1
            prev_blank = True; continue
        prev_blank = False
        m = SECTION_RE.match(raw)
        if m:
            section = (m.group(1) or m.group(2)).strip(); continue
        text, ruby, rep = strip_markup(raw.rstrip())
        backing = bool(BACKING_RE.match(text))
        ln = Line(len(lines), text, raw, stanza, section, backing, rep)
        lang = line_lang(text, default_lang)
        ja_buf = None
        def flush_ja():
            nonlocal ja_buf
            if ja_buf:
                a, b = ja_buf
                ln.words.extend(ja_syllables(text[a:b], a, [r for r in ruby if a <= r[0] < b]))
            ja_buf = None
        for t in TOKEN_RE.finditer(text):
            kind, s, a, b = t.lastgroup, t.group(), t.start(), t.end()
            if lang == "ja" and kind in ("ja", "num", "sym") or (lang == "ja" and any(r[0] <= a < r[1] for r in ruby)):
                ja_buf = (ja_buf[0] if ja_buf else a, b); continue
            flush_ja()
            if kind == "acr":
                names = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "ei bi si di i ef dzi eitch ai dzei kei el em en ou pi kyu ar es ti yu vi dabelyu eks wai zi".split()))
                syl = [Syllable(c, (a + i, a + i + 1), [AlignUnit(c, names.get(c, "en"))]) for i, c in enumerate(s)]
                ln.words.append(Word(s, (a, b), "en", "sym", s, syl, needs_review=True)); continue
            if kind == "en":
                ln.words.append(Word(s, (a, b), "en", reading=s.lower(), syllables=en_syllables(s, a)))
            elif kind == "ko" and re.fullmatch(r"[\u3130-\u318F]+", s):   # ㅋㅋ ㅠㅠ: not sung
                ln.words.append(Word(s, (a, b), "ko", "sym", s, []))
            elif kind == "ko":
                ln.words.append(Word(s, (a, b), "ko", reading=s, syllables=ko_syllables(s, a, pron_aware_ko)))
            elif kind == "num":
                n = int(re.sub(r"\D", "", unicodedata.normalize("NFKC", s)) or 0)
                if lang == "ko":
                    rd = ko_number(n, text[b:].lstrip()) + ("퍼센트" if s.endswith("%") else "")
                    u = _uroman()
                    units = [AlignUnit(c, mms_clean(u.romanize_string(c))) for c in rd]
                else:
                    from num2words import num2words
                    rd = num2words(n, lang="en")
                    units = [AlignUnit(x, mms_clean(x)) for x in re.split(r"[\s-]+", rd)]
                ln.words.append(Word(s, (a, b), lang, "num", rd, [Syllable(s, (a, b), units)], needs_review=True))
            elif kind == "sym":
                rd = (KO_SYM if lang == "ko" else EN_SYM).get(s, "")
                if rd:
                    units = [AlignUnit(c, mms_clean(_uroman().romanize_string(c))) for c in rd] if lang == "ko" else [AlignUnit(rd, rd)]
                    ln.words.append(Word(s, (a, b), lang, "sym", rd, [Syllable(s, (a, b), units)], needs_review=True))
        flush_ja()
        for pm in re.finditer(r"[(（][^)）]*[)）]", text):            # "... hold me (hold me)"
            for w in ln.words:
                if pm.start() < w.span[0] < pm.end(): w.kind = "backing"
        covered = {i for w in ln.words for sy in w.syllables for i in range(*sy.span)}
        for w in ln.words:                                   # orphan 〜/ー/~ after a unit -> prolong it
            for sy in w.syllables:
                e = sy.span[1]
                while e < len(text) and text[e] in "〜~ー" and e not in covered:
                    sy.units.append(AlignUnit("ー", "?")); sy.text += text[e]; e += 1
                sy.span = (sy.span[0], e); w.span = (w.span[0], max(w.span[1], e))
        resolve_context([u for w in ln.words for s in w.syllables for u in s.units])
        lines.append(ln)
    return Song(lines, default_lang)

if __name__ == "__main__":
    import sys, time
    sample = open(sys.argv[1], encoding="utf-8").read()
    t = time.time(); song = parse_song(sample, default_lang=sys.argv[2] if len(sys.argv) > 2 else "ko")
    print(f"# parsed in {time.time()-t:.2f}s")
    for ln in song.lines:
        disp = " ".join("|".join(f"{s.text}" + (f"<{'.'.join(u.reading for u in s.units)}>" if len(s.units) > 1 else "") for s in w.syllables) for w in ln.words)
        rom = " ".join("-".join(s.roman for s in w.syllables) for w in ln.words)
        flags = ("[backing]" if ln.backing else "") + (f"[x{ln.repeat}]" if ln.repeat > 1 else "") + (f"[{ln.section}]" if ln.section else "")
        print(f"{ln.stanza}:{ln.idx:02d} {flags} {ln.text}\n     disp: {disp}\n     roman: {rom}")
        # span round-trip check
        for w in ln.words:
            for s in w.syllables:
                assert ln.text[s.span[0]:s.span[1]] == s.text, (ln.text, s)
    print("align tokens:", len(song.align_tokens()), song.align_tokens()[:6])
    json.dump(asdict(song), open(sys.argv[1] + ".json", "w"), ensure_ascii=False, indent=1)
