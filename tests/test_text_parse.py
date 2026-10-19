"""parse_lyrics end to end: invariants, structure, languages, reference data, robustness, speed."""
from __future__ import annotations

import csv
import random
import time
from itertools import pairwise
from pathlib import Path

import pytest
import regex as re

from lyricprompter.config import PROJECT_ROOT, ref_repo_dir
from lyricprompter.text import detect_lang, parse_lyrics, strip_lrc
from lyricprompter.text.model import Line, Song, SongMeta, line_from_dict

ROMAN = re.compile(r"[a-z']*")
ARTIFACTS = PROJECT_ROOT / "docs" / "research" / "artifacts" / "syllables" / "data"
TTS = PROJECT_ROOT / "data" / "research-testsets" / "tts-multilingual"
REF = ref_repo_dir()

MIXED = """[Verse 1]
꽃잎을 날려 baby 너를 원해
우린 21살, 2번째 봄 & R&B
(oh oh)

[Chorus]
明日(あした)また会えるかな　ちょっと待って
東京タワーで2人きり、ラーメン食べよう x2
{運命|さだめ}に抗う ティーンエイジャー
｜宇宙《そら》へ走り出せ Brand new day

Every morning I wake up, everything's beautiful tonight
Ooh yeahhh, somebody hold me (hold me)
"""

EDGE_CASES = [
    "ｶﾞｯｺｳの帰り道　走[はし]り出す", "｜東京《とう.きょう》の空〜 あ〜あ", "ㅋㅋ 사랑해~ 널 I’m in love (x3)",
    "24시간 100% 너만 생각해", "1000年先も Forever ×2", "すごーい! 好きーー 愛してる〜 ララ〜ラ〜 ずっとー あっ Baby",
    "今日は何を食べよう 明日も一緒に 故郷へ 1,000回 2人で 12月25日 一ヶ月", "사랑 愛してる 사랑해 Love you 君に",
    "{愛|사랑}해 {永遠|forever} 永遠(とわ)に 本気[マジ]で {一瞬|いっ.しゅん}", "I've been goin' 'round 'cause I didn't know",
    "I LOVE YOU SO MUCH, WHY? 4ever 21st century 1999 50% & you", "BTS, OK, USA, the DJ, R&B, AT&T",
    "ヴァイオリン デュエット ウィー ティーン ファンタジー クォーター イェーイ", "君の名前を呼んだ 夜空に響く声 ♪ 😊 ☆",
    "(hey) 우린 (hey!) 春の日 (ah-ah)", "１２３ ＡＢＣ ｱｲｳ", "愛", "々々 ゝ 〆切", "ーあ っ ゃ", "Oh〜 Yeah~ 사랑해〜",
    "café naïve résumé Ñandú über", "x", "2:30 3.14 1,000,000 0", "e-mail rock'n'roll twenty-one", "사랑해 (2번)",
    "Chorus: la la la", "(Repeat Chorus)", "ㅋㅋㅋ ㅠㅠ", "...", "愛してる(あいしてる)", "Hi!!!! ???", "日本語とEnglishと한국어",
    "İstanbul ǅemal ﬁne Straße",
]


def check_invariants(lines: list[Line]) -> None:
    for i, ln in enumerate(lines):
        assert ln.idx == i
        assert ln.start is None and ln.end is None and ln.confidence is None
        prev_end = 0
        for w in ln.words:
            assert w.lang in ("ko", "ja", "en"), w
            assert w.kind in ("lex", "num", "sym", "backing"), w
            assert 0 <= w.span[0] < w.span[1] <= len(ln.text), (ln.text, w)
            assert ln.text[w.span[0]:w.span[1]] == w.text, (ln.text, w)
            assert w.span[0] >= prev_end, f"overlapping words in {ln.text!r}"
            prev_end = w.span[1]
            assert w.syllables, w
            for s in w.syllables:
                assert ln.text[s.span[0]:s.span[1]] == s.text, (ln.text, s)
                assert w.span[0] <= s.span[0] < s.span[1] <= w.span[1], (ln.text, w.text, s)
                assert s.units and s.start is None and s.end is None
                for u in s.units:
                    assert ROMAN.fullmatch(u.roman), (ln.text, s.text, u)
                    assert u.start is None and u.end is None and u.score is None
            spans = [s.span for s in w.syllables]
            assert all(a[1] <= b[0] for a, b in pairwise(spans)), (ln.text, spans)


@pytest.fixture(scope="module")
def mixed() -> list[Line]:
    return parse_lyrics(MIXED)


# --------------------------------------------------------------------------- invariants
@pytest.mark.parametrize("path", sorted(ARTIFACTS.glob("*.txt")), ids=lambda p: p.name)
def test_invariants_on_research_samples(path: Path):
    lines = parse_lyrics(path.read_text(encoding="utf-8"))
    assert lines
    check_invariants(lines)


@pytest.mark.parametrize("raw", EDGE_CASES)
def test_invariants_on_edge_cases(raw: str):
    check_invariants(parse_lyrics(raw))


def test_invariants_on_random_text():
    rng = random.Random(1234)
    alphabet = ("가나다꽃밭잎을사랑해ㅋㅠ あいうえおかっちょーゃ〜ア カタナティヴァ 漢字東京明日今日々ヶ "
                "abcxyzABC'R&B 0123456789%,. ()（）[]{}|｜《》【】×*!?~♪😊　\t")
    for _ in range(300):
        line = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 30)))
        check_invariants(parse_lyrics("\n".join([line, "", line + " x2"])))


def test_empty_and_whitespace():
    assert parse_lyrics("") == []
    assert parse_lyrics("\n \n\t\n") == []
    with pytest.raises(ValueError):
        parse_lyrics("la", default_lang="fr")


def test_round_trip_through_song_json(mixed):
    song = Song(SongMeta(id="t"), mixed)
    again = Song.from_dict(song.to_dict())
    assert again.to_dict() == song.to_dict()
    assert line_from_dict(song.to_dict()["lines"][5]).flags == ["repeat_of:4"]


# --------------------------------------------------------------------------- structure
def test_mixed_song_structure(mixed):
    texts = [ln.text for ln in mixed]
    assert texts[:3] == ["꽃잎을 날려 baby 너를 원해", "우린 21살, 2번째 봄 & R&B", "(oh oh)"]
    assert texts[4] == texts[5] == "東京タワーで2人きり、ラーメン食べよう"
    assert mixed[5].flags == ["repeat_of:4"] and "repeat_of:4" not in mixed[4].flags
    assert [ln.section for ln in mixed] == ["Verse 1"] * 3 + ["Chorus"] * 7
    assert [ln.stanza for ln in mixed] == [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
    assert mixed[2].backing and all(w.kind == "backing" for w in mixed[2].words)
    hold = mixed[9]
    assert [w.kind for w in hold.words][-2:] == ["backing", "backing"] and not hold.backing
    assert mixed[3].text == "明日また会えるかな ちょっと待って"              # furigana removed, spaces normalised
    assert mixed[3].raw == "明日(あした)また会えるかな　ちょっと待って"


def test_repeat_copies_are_independent(mixed):
    a, b = mixed[4], mixed[5]
    assert a.words is not b.words and a.words[0].syllables[0].units[0] is not b.words[0].syllables[0].units[0]
    b.words[0].syllables[0].units[0].start = 1.0
    assert a.words[0].syllables[0].units[0].start is None


def test_repeat_marks_and_directives():
    lines = parse_lyrics("""[Chorus]
la la x3
oh

[Verse]
one
(x2)
Repeat Chorus
[Chorus]
[Bridge x2]
hey
""")
    assert [ln.text for ln in lines] == ["la la"] * 3 + ["oh", "one", "one"] + ["la la"] * 3 + ["oh"] + \
        ["la la"] * 3 + ["oh"] + ["hey", "hey"]
    assert [ln.idx for ln in lines] == list(range(len(lines)))
    assert lines[1].flags == lines[2].flags == ["repeat_of:0"]
    assert lines[5].flags == ["repeat_of:4"]
    assert [ln.flags for ln in lines[6:10]] == [["repeat_of:0"]] * 3 + [["repeat_of:3"]]
    assert [ln.flags for ln in lines[10:14]] == [["repeat_of:0"]] * 3 + [["repeat_of:3"]]
    assert [ln.section for ln in lines] == ["Chorus"] * 4 + ["Verse"] * 2 + ["Chorus"] * 8 + ["Bridge"] * 2
    assert lines[15].flags == ["repeat_of:14"]
    stanzas = [ln.stanza for ln in lines]
    assert stanzas == sorted(stanzas) and stanzas[6] > stanzas[5] and stanzas[10] > stanzas[9]


def _summary(raw: str) -> list[tuple[str, str | None, list[str]]]:
    lines = parse_lyrics(raw)
    check_invariants(lines)
    return [(ln.text, ln.section, ln.flags) for ln in lines]


@pytest.mark.parametrize("directive", ["(Repeat x2)", "[Repeat x2]", "Repeat x2", "(repeat 2x)", "Repeat twice",
                                       "2번 반복", "(2번 반복)", "※2回繰り返し", "※繰り返し×2", "(x2)", "Repeat"])
def test_block_repeat_with_count_copies_preceding_stanza(directive: str):
    # regression: "(Repeat x2)" was a silent backing line, "[Repeat x2]" doubled the NEXT verse
    assert _summary(f"la la\nna na\n{directive}\n\nnext verse") == [
        ("la la", None, []), ("na na", None, []), ("la la", None, ["repeat_of:0"]), ("na na", None, ["repeat_of:1"]),
        ("next verse", None, [])]


def test_repeat_counts_follow_one_convention():
    def n(raw: str, text: str = "la") -> int:
        return sum(ln.text == text for ln in parse_lyrics(raw))
    assert n("la x3") == n("la\nRepeat x3") == n("la\n(Repeat 3 times)") == n("la\n3번 반복") == 3
    assert n("[Chorus]\nla\n\n[Verse]\nyo\n\nRepeat Chorus x3") == 1 + 3      # a section: N copies here
    assert n("[후렴]\nla\n\n[2절]\nyo\n\n후렴 2번 반복") == n("[후렴]\nla\n\n후렴 반복 2번") == 1 + 2


def test_section_repeat_with_count_in_korean_and_japanese():
    got = _summary("[후렴]\n너를 사랑해\n\n[2절]\n하늘을 봐\n\n후렴 2번 반복")
    assert got == [("너를 사랑해", "후렴", []), ("하늘을 봐", "2절", []),
                   ("너를 사랑해", "후렴", ["repeat_of:0"]), ("너를 사랑해", "후렴", ["repeat_of:0"])]
    got = _summary("【サビ】\n君が好き\n\n【Bメロ】\n空\n\nサビ2回繰り返し")
    assert [t for t, _, _ in got] == ["君が好き", "空", "君が好き", "君が好き"]


def test_lines_after_section_repeat_are_not_part_of_that_section():
    # regression: "bridge line" joined the copied chorus block and was copied by the next "Repeat Chorus"
    got = _summary("[Chorus]\nla la\n\n[Verse]\nyo\nRepeat Chorus\nbridge line\n\n[Verse 2]\nhey\nRepeat Chorus")
    assert got == [("la la", "Chorus", []), ("yo", "Verse", []), ("la la", "Chorus", ["repeat_of:0"]),
                   ("bridge line", None, []), ("hey", "Verse 2", []), ("la la", "Chorus", ["repeat_of:0"])]


def test_header_followed_by_performer_tag_is_one_header():
    # regression: "[Chorus]" + "[Jimin]" made the first header "empty" and copied the earlier chorus
    got = _summary("[Chorus]\nla la la\n\n[Verse 2]\nyo yo\n\n[Chorus]\n[Jimin]\nla la la\n\n[Jungkook]\n[Bridge]\nhey")
    assert got == [("la la la", "Chorus", []), ("yo yo", "Verse 2", []), ("la la la", "Chorus: Jimin", []),
                   ("hey", "Bridge: Jungkook", [])]
    lines = parse_lyrics("[Chorus]\n[Jimin]\nla\n\n[Verse]\nyo\n\nRepeat Chorus")
    assert [ln.flags for ln in lines] == [[], [], ["repeat_of:0"]]


def test_adjacent_section_headers_keep_empty_header_semantics():
    got = _summary("[Intro]\n[Verse 1]\nhello\n\n[Chorus]\nla\n\n[Verse 2]\nyo\n\n[Chorus]\n[Bridge]\nhey")
    assert got == [("hello", "Verse 1", []), ("la", "Chorus", []), ("yo", "Verse 2", []),
                   ("la", "Chorus", ["repeat_of:1"]), ("hey", "Bridge", [])]


def test_header_prefixes_and_verse_numbers_are_not_sung():
    lines = parse_lyrics("la la\n[Chorus] oh baby\n【サビ】君が好き\n<Hook>\nyeah\n\n1. 꽃밭에 앉아서\n2.\n3) Every morning")
    check_invariants(lines)
    assert [(ln.text, ln.section, ln.stanza) for ln in lines] == [
        ("la la", None, 0), ("oh baby", "Chorus", 1), ("君が好き", "サビ", 2), ("yeah", "Hook", 3),
        ("꽃밭에 앉아서", "Hook", 4), ("Every morning", "Hook", 5)]
    assert [w.text for w in lines[1].words] == ["oh", "baby"]
    assert lines[4].raw == "1. 꽃밭에 앉아서"
    assert [(ln.text, ln.flags) for ln in parse_lyrics("[Chorus x2] la la\nna na\n\nend")] == [
        ("la la", []), ("na na", []), ("la la", ["repeat_of:0"]), ("na na", ["repeat_of:1"]), ("end", [])]


def test_bom_and_unknown_lrc_tags():
    lines = parse_lyrics("\ufeff[00:01.00]hello world\n[00:03.00]second line")
    assert [(ln.text, ln.section) for ln in lines] == [("hello world", None), ("second line", None)]
    lines = parse_lyrics("\ufeff[ti:Title]\n[id:abcd]\n[kana:1あい1こい]\n[00:01.00]君が好き")
    assert [(ln.text, ln.section) for ln in lines] == [("君が好き", None)]
    assert [ln.text for ln in parse_lyrics("\ufeff[Chorus]\nla")] == ["la"]


def test_very_long_digit_runs_do_not_crash():
    for raw, lang in (("사랑 " + "9" * 100, "auto"), ("1" * 5000, "auto"), ("1" * 300, "ja"), ("007 baby", "en")):
        lines = parse_lyrics(raw, lang)
        check_invariants(lines)
        assert lines[0].needs_review or lang == "ja"


def test_unresolved_directive_is_kept_for_review():
    lines = parse_lyrics("hello\nRepeat Chorus")
    assert lines[-1].flags == ["unresolved_repeat"] and lines[-1].needs_review and not lines[-1].words


def test_repeat_and_section_words_inside_lyrics_stay_lyrics():
    lines = parse_lyrics("Repeat after me\n같은 하루의 반복\nI'm 22\nlovex2")
    assert [ln.text for ln in lines] == ["Repeat after me", "같은 하루의 반복", "I'm 22", "lovex2"]
    assert all(ln.words for ln in lines)


@pytest.mark.parametrize("raw,text", [("사랑해x2", "사랑해"), ("好きだよx2", "好きだよ"), ("사랑해 x2", "사랑해")])
def test_repeat_mark_glued_to_cjk_doubles_the_line(raw, text):
    # regression RU-7: "사랑해x2" was one line sung as "sa rang hae x two"
    lines = parse_lyrics(raw)
    check_invariants(lines)
    assert [ln.text for ln in lines] == [text, text]
    assert all("x" not in u.roman.split() and "two" not in u.roman.split()
               for ln in lines for s in ln.syllables() for u in s.units)


def test_stanzas_and_separators():
    lines = parse_lyrics("\n\na\nb\n\n\n* * *\nc\n♪\nd\n")
    assert [(ln.text, ln.stanza) for ln in lines] == [("a", 0), ("b", 0), ("c", 1), ("d", 2)]


def test_non_sung_tokens_produce_no_syllables():
    ln = parse_lyrics("ㅋㅋ 사랑해 😊 ♪ ... !!")[0]
    assert [s.text for s in ln.syllables()] == ["사", "랑", "해"]
    only = parse_lyrics("ㅋㅋㅋ ㅠㅠ")[0]
    assert only.words == [] and only.flags == ["no_sung_content"]


# --------------------------------------------------------------------------- languages
def test_mixed_language_line():
    ln = parse_lyrics("日本語とEnglishと한국어 2人 21살")[0]
    by = {w.text: w for w in ln.words}
    assert by["日本語"].lang == "ja" and by["English"].lang == "en" and by["한국어"].lang == "ko"
    assert by["2人"].lang == "ja" and by["2人"].reading == "ふたり"
    assert by["21"].lang == "ko" and by["21"].reading == "스물한"


def test_han_only_lines_follow_song_majority():
    ja_song = parse_lyrics("愛\n君の名前\n夢")
    assert [w.lang for ln in ja_song for w in ln.words if ln.text in ("愛", "夢")] == ["ja", "ja"]
    ko_song = parse_lyrics("愛\n사랑해\n너를")
    hanja = ko_song[0].words[0]
    assert hanja.lang == "ko" and hanja.needs_review and hanja.syllables[0].roman == ""
    assert ko_song[0].needs_review and "text_review" in ko_song[0].flags
    assert parse_lyrics("愛\n사랑해", default_lang="ja")[0].words[0].lang == "ja"
    assert parse_lyrics("愛\n君の", default_lang="ko")[0].words[0].syllables[0].roman == ""


def test_digit_only_lines_follow_default_or_majority():
    assert parse_lyrics("1 2 3 4\n사랑해\n너를")[0].words[0].lang == "ko"
    assert parse_lyrics("1 2 3 4\nhello there")[0].words[0].reading == "one"
    assert parse_lyrics("1 2 3 4", default_lang="ko")[0].words[0].reading == "일"
    ja = parse_lyrics("1 2 3 4", default_lang="ja")[0]
    assert ja.words[0].lang == "ja" and ja.words[0].reading == "いち"


def test_detect_lang_agrees_with_line_words(mixed):
    for ln in mixed:
        lang = detect_lang(ln.text)
        assert lang in {w.lang for w in ln.words}, (ln.text, lang)


# --------------------------------------------------------------------------- reference data
@pytest.mark.skipif(REF is None, reason="reference repo not cloned")
def test_kkotbat_syllables_match_score_csv():
    assert REF is not None
    lines = parse_lyrics((REF / "resources/lyrics/꽃밭에서_lyrics.txt").read_text(encoding="utf-8"))
    syls = [s.text for ln in lines for s in ln.syllables()]
    with open(REF / "resources/annots/꽃밭에서_score.csv", encoding="utf-8") as f:
        score = [r["lyric"] for r in csv.DictReader(f) if r["lyric"].strip()]
    assert len(syls) == len(score) == 144
    assert syls == score
    assert all(len(s.units) == 1 and s.units[0].roman for ln in lines for s in ln.syllables())
    assert len(lines) == 15 and [ln.stanza for ln in lines][-1] == 3
    check_invariants(lines)


@pytest.mark.skipif(REF is None, reason="reference repo not cloned")
@pytest.mark.parametrize("name", ["Edelweiss.lrc", "Edelweiss_short.lrc"])
def test_edelweiss_lrc(name: str):
    assert REF is not None
    raw = (REF / "resources/lyrics" / name).read_text(encoding="utf-8")
    plain = strip_lrc(raw)
    lines = parse_lyrics(raw)
    assert [ln.text for ln in lines] == [ln.text for ln in parse_lyrics(plain)]
    assert len(lines) == (16 if name == "Edelweiss.lrc" else 4)
    assert lines[0].text == "Edelweiss, edelweiss"
    assert [s.text for s in lines[0].words[0].syllables] == ["E", "del", "weiss"]
    # LRC word onsets: every <tag> starts a syllable or a word -> our syllables are at least as fine
    lrc_lines = [ln for ln in raw.splitlines() if ln.startswith("[0")]
    for ln, src in zip(lines, lrc_lines):
        n_lrc_words = len(re.findall(r"(?:^\[[^\]]+\]|<[^>]+>)\s", src)) + 1
        assert len(ln.words) == n_lrc_words, (ln.text, src)
        assert len(ln.syllables()) >= len(ln.words)
    check_invariants(lines)


@pytest.mark.skipif(not TTS.is_dir(), reason="TTS test set missing")
@pytest.mark.parametrize("name", ["lyrics.txt", "lyrics_long.txt"])
def test_tts_lyrics(name: str):
    raw = (TTS / name).read_text(encoding="utf-8")
    lines = parse_lyrics(raw)
    assert [ln.text for ln in lines] == [x.strip() for x in raw.splitlines() if x.strip()]
    assert [detect_lang(ln.text) for ln in lines[:4]] == ["ko", "ko", "ja", "en"]
    assert {w.lang for ln in lines[:4] for w in ln.words} == {"ko", "ja", "en"}
    check_invariants(lines)


# --------------------------------------------------------------------------- speed
def test_warm_parse_of_60_line_mixed_song_is_fast():
    blocks = [MIXED, "\n".join(EDGE_CASES[:12])]
    song = "\n\n".join(blocks * 3)
    n = len([x for x in song.splitlines() if x.strip()])
    assert n >= 60
    parse_lyrics(song)                                   # warm up analyzers / dictionaries
    t = time.perf_counter()
    lines = parse_lyrics(song)
    dt = time.perf_counter() - t
    assert len(lines) >= 60
    assert dt < 2.0, f"warm parse took {dt:.2f}s"


# --------------------------------------------------------------------------- slow / exhaustive
@pytest.mark.slow
def test_every_hangul_syllable_romanises_to_mms_letters():
    from lyricprompter.text import korean
    bad = [ch for ch in map(chr, range(0xAC00, 0xD7A4)) if not re.fullmatch(r"[a-z]+", korean.syllable_roman(ch))]
    assert not bad


@pytest.mark.slow
@pytest.mark.skipif(REF is None, reason="reference repo not cloned")
def test_all_reference_lyrics_parse():
    assert REF is not None
    for path in sorted((REF / "resources/lyrics").iterdir()):
        lines = parse_lyrics(path.read_text(encoding="utf-8"))
        assert lines, path.name
        check_invariants(lines)
