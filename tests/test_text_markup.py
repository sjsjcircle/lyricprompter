"""Normalisation, language detection, LRC stripping and line markup (no analyzers needed)."""
from __future__ import annotations

import pytest

from lyricprompter.text import detect_lang, normalize_line, strip_lrc
from lyricprompter.text.markup import (
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
from lyricprompter.text.normalize import is_lrc, mms_clean


# --------------------------------------------------------------------------- normalisation
@pytest.mark.parametrize("raw,expected", [
    ("ｶﾞｯｺｳの帰り道", "ガッコウの帰り道"),               # half-width kana (dakuten recomposed)
    ("ＡＢＣ　１２３！", "ABC 123!"),                    # full-width ASCII + ideographic space
    ("ㅋㅋ ㅠㅠ", "ㅋㅋ ㅠㅠ"),                           # compatibility jamo are NOT turned into ᄏ
    ("I’m  ‘here’\t“now”", "I'm 'here' \"now\""),
    ("ze\u200bro\u00adwidth\ufeff", "zerowidth"),
    ("㌔", "キロ"),
    ("  漢字（かんじ）　", "漢字(かんじ)"),
    ("Lascia ch′io", "Lascia ch'io"),
    ("葛\U000E0100城の空 禰\uFE00豆子 ❤\uFE0F", "葛城の空 禰豆子 ❤"),   # variation selectors dropped
    ("か゛ は゜ あ゛ーー", "が ぱ あ\u3099ーー"),                     # spacing ゛/゜ compose, not "か ゙"
])
def test_normalize_line(raw, expected):
    assert normalize_line(raw) == expected


def test_normalize_composes_decomposed_hangul():
    import unicodedata
    assert normalize_line(unicodedata.normalize("NFD", "꽃밭에서")) == "꽃밭에서"


def test_mms_clean():
    assert mms_clean("Café Ñandú's") == "cafenandu's"
    assert mms_clean("ß-AB_9") == "ssab"
    assert [mms_clean(w) for w in ("Søren", "København", "Œuvre", "łódź", "Þór", "Æsir")] == [
        "soren", "kobenhavn", "oeuvre", "lodz", "thor", "aesir"]           # letters without an NFKD form


@pytest.mark.parametrize("text,lang", [
    ("사랑해", "ko"), ("愛してる", "ja"), ("Hello world", "en"), ("愛", "ja"), ("123 !!", "x"), ("", "x"),
    ("사랑해 baby", "ko"), ("君の baby", "ja"), ("ㅋㅋ", "ko"), ("ラーメン", "ja"), ("사랑 愛してる 사랑해 君に", "ko"),
    ("愛 love", "ja"),
])
def test_detect_lang(text, lang):
    assert detect_lang(text) == lang


# --------------------------------------------------------------------------- LRC
EDEL = """[ti:Edelweiss]
[la:EN]

[00:06.82]E  <00:07.87>del  <00:08.33>weiss,  <00:10.00> e  <00:10.99>del  <00:11.49>weiss
[00:12.99]E  <00:14.10>very  <00:14.75> morn  <00:15.17>ing  <00:15.59> you  <00:16.06> greet  <00:17.53> me

[01:05.37]Bless  <01:06.33> my  <01:06.92> home  <01:07.20>land
"""


def test_strip_lrc_glued_word_tags():
    assert is_lrc(EDEL)
    assert strip_lrc(EDEL).splitlines() == [
        "", "Edelweiss, edelweiss", "Every morning you greet me", "", "Bless my homeland"]


def test_strip_lrc_enhanced_and_multi_timestamps():
    enhanced = "[00:01.00]<00:01.00>When <00:01.50>the <00:02.00>truth\n[00:03.00]is <00:03.50>found"
    assert strip_lrc(enhanced).splitlines() == ["When the truth", "is found"]
    multi = "[00:10.00][00:30.00]la la\n[00:20.00]verse\n[00:40.00]end \\nnext"
    assert strip_lrc(multi).splitlines() == ["la la", "verse", "la la", "end", "next"]
    assert not is_lrc("[Chorus]\nla la")


def test_strip_lrc_bom_and_unknown_tags():
    assert is_lrc("\ufeff[00:01.00]hello world")
    assert strip_lrc("\ufeff[00:01.00]hello world\n[00:03.00]second line").splitlines() == [
        "hello world", "second line"]
    raw = "\ufeff[ti:Title]\n[id:abcd]\n[kana:1あい1こい]\n[hash:f00]\n[total:1234]\n[Chorus: Jimin]\n[00:01.00]君が好き"
    assert strip_lrc(raw).splitlines() == ["[Chorus: Jimin]", "君が好き"]     # capitalised = section header


# --------------------------------------------------------------------------- ruby
@pytest.mark.parametrize("raw,display,ruby", [
    ("明日(あした)また", "明日また", [Ruby(0, 2, "あした")]),
    ("漢字（かんじ）", "漢字（かんじ）", []),                       # only after normalisation: （ -> (
    (normalize_line("漢字（かんじ）だ"), "漢字だ", [Ruby(0, 2, "かんじ")]),
    ("{運命|さだめ}に", "運命に", [Ruby(0, 2, "さだめ")]),
    ("|宇宙《そら》へ", "宇宙へ", [Ruby(0, 2, "そら")]),
    ("君と宇宙《そら》", "君と宇宙", [Ruby(2, 4, "そら")]),
    ("走[はし]り出す", "走り出す", [Ruby(0, 1, "はし")]),
    ("{東京|とう.きょう}の", "東京の", [Ruby(0, 2, "とう.きょう")]),
    ("本気(マジ)で", "本気で", [Ruby(0, 2, "マジ")]),
    ("{愛|사랑}해", "愛해", [Ruby(0, 1, "사랑")]),
    ("{永遠|for ever}", "永遠", [Ruby(0, 2, "forever")]),
    ("a {b|び} c 明日(あした)", "a b c 明日", [Ruby(2, 3, "び"), Ruby(6, 8, "あした")]),
    ("(oh oh)", "(oh oh)", []),                                    # backing vocals, not ruby
    ("愛してる(あいしてる)", "愛してる(あいしてる)", []),            # paren after kana is not furigana
    ("愛(love)", "愛(love)", []),
])
def test_extract_ruby(raw, display, ruby):
    assert extract_ruby(raw) == (display, ruby)


# --------------------------------------------------------------------------- repeats / headers
@pytest.mark.parametrize("raw,expected", [
    ("love you x2", ("love you", 2)), ("love you X3", ("love you", 3)), ("Forever ×2", ("Forever", 2)),
    ("Forever×2", ("Forever", 2)), ("la la *2", ("la la", 2)), ("hey (x3)", ("hey", 3)),
    ("hey(x3)", ("hey", 3)), ("hey [x2]", ("hey", 2)), ("사랑해 (2번)", ("사랑해", 2)),
    ("会いたい（2回）", ("会いたい（2回）", 1)), (normalize_line("会いたい（2回）"), ("会いたい", 2)),
    ("go go 2x", ("go go", 2)), ("hey (2 times)", ("hey", 2)), ("hey x99", ("hey", 16)),
    ("x2", ("x2", 1)), ("Malcolm X", ("Malcolm X", 1)), ("lovex2", ("lovex2", 1)), ("I'm 22", ("I'm 22", 1)),
    ("I love you (Repeat x2)", ("I love you", 2)), ("hey (repeat 3 times)", ("hey", 3)), ("la la (반복)", ("la la", 2)),
    ("너를 (2번 반복)", ("너를", 2)), ("love you twice", ("love you twice", 1)), ("(Repeat x2)", ("(Repeat x2)", 1)),
    # regression RU-7: an ASCII x / * glued to Hangul / kana / kanji is a repeat mark, never a sung "x"
    ("사랑해x2", ("사랑해", 2)), ("사랑해X3", ("사랑해", 3)), ("사랑해*2", ("사랑해", 2)), ("好きだよx2", ("好きだよ", 2)),
    ("好きーx2", ("好きー", 2)), ("愛x2", ("愛", 2)), ("ㅋㅋx3", ("ㅋㅋ", 3)), ("사랑해x 2", ("사랑해", 2)),
    ("BTSx2", ("BTSx2", 1)), ("사랑해 lovex2", ("사랑해 lovex2", 1)), ("사랑해x", ("사랑해x", 1)),
])
def test_split_repeat(raw, expected):
    assert split_repeat(raw) == expected


@pytest.mark.parametrize("line,expected", [
    ("[Chorus]", Header("Chorus")), ("[Verse 1]", Header("Verse 1")), ("【サビ】", Header("サビ")),
    ("Verse 2:", Header("Verse 2")), ("후렴", Header("후렴")), ("(Chorus)", Header("Chorus")),
    ("[Chorus x2]", Header("Chorus", 2)), ("[Chorus: Artist]", Header("Chorus: Artist")),
    ("[Hook] (x2)", Header("Hook", 2)), ("Chorus (x2)", Header("Chorus", 2)), ("Bメロ", Header("Bメロ")), ("2절", Header("2절")), ("[間奏]", Header("間奏")),
    ("(x2)", Directive("block", copies=1)), ("x3", Directive("block", copies=2)), ("(2번)", Directive("block")),
    ("Repeat Chorus", Directive("section", "Chorus")), ("(Repeat chorus x2)", Directive("section", "chorus", 2)),
    ("Repeat Chorus x3", Directive("section", "Chorus", 3)), ("Repeat Chorus (x2)", Directive("section", "Chorus", 2)),
    ("후렴 반복", Directive("section", "후렴")), ("※くり返し", Directive("block")),
    ("サビくり返し", Directive("section", "サビ")), ("[Repeat]", Directive("block")),
    ("Repeat after me", None), ("같은 하루의 반복", None), ("(oh oh)", None), ("Chorus line here", None),
    ("I love you", None),
])
def test_classify_structure(line, expected):
    assert classify_structure(line) == expected


B = Directive("block")


@pytest.mark.parametrize("line,expected", [
    # block repeats with a count: N = sung N times in all -> N-1 copies (regression: phantom lyric lines)
    ("Repeat x2", B), ("(Repeat x2)", B), ("[Repeat x2]", B), ("(repeat 2x)", B), ("Repeat twice", B),
    ("Repeat x3", Directive("block", copies=2)), ("(Repeat 3 times)", Directive("block", copies=2)),
    ("Repeat 3 times", Directive("block", copies=2)), ("[Repeat ×4]", Directive("block", copies=3)),
    ("2번 반복", B), ("(2번 반복)", B), ("반복 2번", B), ("두 번 반복", B), ("3회 반복", Directive("block", copies=2)),
    ("※2回繰り返し", B), ("※繰り返し×2", B), ("(反復)", None), ("Repeat", B), ("(반복)", B),
    # section repeats with a count: N copies of the section here
    ("후렴 2번 반복", Directive("section", "후렴", 2)), ("후렴 반복 2번", Directive("section", "후렴", 2)),
    ("(후렴 반복)", Directive("section", "후렴")), ("サビ2回繰り返し", Directive("section", "サビ", 2)),
    ("Repeat the chorus twice", Directive("section", "chorus", 2)), ("[Repeat Chorus]", Directive("section", "Chorus")),
    ("Chorus (Repeat)", Directive("section", "Chorus")), ("Rpt. chorus", Directive("section", "chorus")),
    ("1番くり返し", Directive("section", "1番")),
    # lyrics that merely contain the words stay lyrics
    ("반복되는 하루", None), ("Don't repeat yourself", None), ("Repeat 'til the end", None), ("Twice", None),
    ("Once upon a time", None), ("I love you twice", None), ("(Jimin)", None), ("<3 you>", None),
])
def test_classify_repeat_directives_with_counts(line, expected):
    assert classify_structure(normalize_line(line)) == expected


@pytest.mark.parametrize("line,expected", [
    ("<Chorus>", Header("Chorus")), ("＜サビ＞", Header("サビ")), ("*Chorus*", Header("Chorus")),
    ("-Chorus-", Header("Chorus")), ("-Pre-Chorus-", Header("Pre-Chorus")), ("~サビ~", Header("サビ")),
    ("== Bridge ==", Header("Bridge")), ("Verse One:", Header("Verse One")), ("Verse 1 (Jimin)", Header("Verse 1 (Jimin)")),
    ("Chorus twice", Header("Chorus", 2)), ("サビ×2", Header("サビ", 2)), ("<Love>", None), ("*sigh*", None),
])
def test_classify_more_header_styles(line, expected):
    assert classify_structure(normalize_line(line)) == expected


@pytest.mark.parametrize("line,expected", [
    ("[Chorus] oh baby", (Header("Chorus"), "oh baby")), ("【サビ】君が好き", (Header("サビ"), "君が好き")),
    ("<Hook> la la", (Header("Hook"), "la la")), ("[Verse 2: RM] yo", (Header("Verse 2: RM"), "yo")),
    ("[Chorus x2] la", (Header("Chorus", 2), "la")), ("<3 you", (None, "<3 you")), ("<Love> me", (None, "<Love> me")),
    ("[x2] la", (None, "[x2] la")), ("Verse One: hey", (Header("Verse One"), "hey")),
])
def test_split_header_prefix(line, expected):
    assert split_inline_header(line) == expected


@pytest.mark.parametrize("line,expected", [
    ("1. 꽃밭에 앉아서", "꽃밭에 앉아서"), ("3) Every morning", "Every morning"), ("2.", ""), ("10)", ""),
    ("1. 2. 3. go", "1. 2. 3. go"), ("3.14 is pi", "3.14 is pi"), ("2 hearts", "2 hearts"), ("1.君が", "1.君が"),
])
def test_strip_verse_mark(line, expected):
    assert strip_verse_mark(line) == expected


def test_inline_header_and_section_keys():
    assert split_inline_header("Chorus: I love you") == (Header("Chorus"), "I love you")
    assert split_inline_header("Love: is all") == (None, "Love: is all")
    assert section_key("[Chorus 2]".strip("[]")) == ("chorus2", "chorus")
    assert section_key("후렴")[1] == section_key("Chorus")[1] == section_key("サビ")[1]
    assert section_key("Pre-Chorus") == ("prechorus", "prechorus")
    assert section_key("Verse Two") == section_key("Verse 2 (RM)") == section_key("Verse II") == ("verse2", "verse")
    assert section_key("인트로")[1] == section_key("Intro")[1]
    assert is_section_label("Chorus: Jimin") and is_section_label("후렴") and is_section_label("Instrumental Break")
    assert not is_section_label("Jimin") and not is_section_label("Hooked") and not is_section_label("Rapture")


def test_backing_detection():
    text = "hold me (hold me) (oh)"
    assert paren_ranges(text) == [(8, 16), (18, 21)]
    assert not is_backing_line(text, paren_ranges(text))
    for t in ("(oh oh)", "(oh) (yeah)", "(oh (ah) oh)!"):
        assert is_backing_line(t, paren_ranges(t)), t
    assert not is_backing_line("(oh oh", paren_ranges("(oh oh"))
