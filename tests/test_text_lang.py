"""Korean / Japanese / English / number building blocks of the lyrics front-end."""
from __future__ import annotations

import pytest

from lyricprompter.text import english, japanese, korean, numbers
from lyricprompter.text.model import AlignUnit
from lyricprompter.text.parse import build_line

ROMAN_OK = set("abcdefghijklmnopqrstuvwxyz'")


# --------------------------------------------------------------------------- Korean
@pytest.mark.parametrize("ch,roman", [("꽃", "ggoc"), ("잎", "ip"), ("을", "eul"), ("의", "yi"), ("값", "gabs"),
                                      ("닭", "dalg"), ("뷁", "bwelg"), ("사", "sa"), ("랑", "rang")])
def test_korean_syllable_roman(ch, roman):
    assert korean.syllable_roman(ch) == roman


@pytest.mark.parametrize("word,n", [("꽃밭에서", 4), ("앉아서", 3), ("괜찮아", 3), ("사랑해", 3), ("값", 1)])
def test_korean_syllable_counts(word, n):
    syls = korean.ko_syllables(word, 5)
    assert len(syls) == n
    assert [s.span for s in syls] == [(5 + i, 6 + i) for i in range(n)]
    assert all(len(s.units) == 1 and s.units[0].roman and set(s.units[0].roman) <= ROMAN_OK for s in syls)
    # per-syllable romanisation concatenates to the whole-word romanisation (uroman property)
    whole = korean.mms_clean(korean._uroman().romanize_string(word))
    assert "".join(s.roman for s in syls) == whole


def test_korean_units_skip_non_hangul():
    assert [u.reading for u in korean.ko_units("스물 한!")] == ["스", "물", "한"]


# --------------------------------------------------------------------------- numbers
@pytest.mark.parametrize("digits,following,reading,ambiguous", [
    ("20", "살", "스무", False), ("21", "살", "스물한", False), ("1", "번째", "첫", False),
    ("2", "번째", "두", False), ("11", "번째", "열한", False), ("20", "번째", "스무", False),
    ("24", "시간", "스물네", False), ("3", "시", "세", False), ("2", "명의", "두", False), ("2", " 사람", "두", False),
    ("10", "년", "십", False), ("2024", "년", "이천이십사", False), ("6", "월", "유", False), ("10", "월", "시", False),
    ("12", "월", "십이", False), ("1", "개월", "일", False), ("3", "개", "세", False), ("100", "살", "백", False),
    ("101", "살", "백한", False), ("1", "번", "한", True), ("10", "대", "십", True), ("3", "대", "세", True),
    ("5", "분", "오", True), ("100%", "", "백퍼센트", False), ("3.5", "", "삼점오", True),
    ("1,000", "원", "천", False), ("0", "", "영", True), ("7", "", "칠", True), ("12345", "", "만이천삼백사십오", True),
])
def test_korean_numbers(digits, following, reading, ambiguous):
    num = numbers.parse_number(digits)
    assert num is not None
    assert numbers.ko_number(num, following) == (reading, ambiguous)


@pytest.mark.parametrize("token,words,ambiguous", [
    ("21", ["twenty", "one"], False), ("1st", ["first"], False), ("21st", ["twenty", "first"], False),
    ("1999", ["nineteen", "ninety", "nine"], True), ("50%", ["fifty", "percent"], False),
    ("3.5", ["three", "point", "five"], False), ("4", ["four"], False), ("2000", ["two", "thousand"], True),
])
def test_english_numbers(token, words, ambiguous):
    num = numbers.parse_number(token)
    assert num is not None
    assert numbers.en_number_words(num) == (words, ambiguous)


@pytest.mark.parametrize("token,en,ko", [
    ("007", ["zero", "zero", "seven"], "공공칠"), ("05", ["zero", "five"], "공오"),     # leading zero: digit by digit
    ("9" * 20, ["nine"] * 20, "구" * 20), ("1" * 5000, ["one"] * 5000, "일" * 5000),   # no OverflowError / ValueError
])
def test_spelled_numbers_are_flagged(token, en, ko):
    num = numbers.parse_number(token)
    assert num is not None and num.spelled
    assert numbers.en_number_words(num) == (en, True)
    assert numbers.ko_number(num, "") == (ko, True)


def test_decimals_are_read_digit_by_digit():
    num = numbers.parse_number("3.14159265358979323846")
    assert num is not None and not num.spelled
    words, _ = numbers.en_number_words(num)
    assert words[:2] == ["three", "point"] and len(words) == 22
    assert numbers.en_number_words(numbers.parse_number("2.05")) == (["two", "point", "zero", "five"], False)


def test_parse_number_rejects_garbage():
    assert numbers.parse_number("1.2.3") is None
    assert numbers.parse_number("abc") is None
    assert numbers.parse_number("1,000,000") == numbers.NumberToken(1000000)


# --------------------------------------------------------------------------- English
@pytest.mark.parametrize("word,pieces", [
    ("Edelweiss", ["E", "del", "weiss"]), ("morning", ["morn", "ing"]), ("happy", ["hap", "py"]),
    ("blossom", ["blos", "som"]), ("homeland", ["home", "land"]), ("forever", ["for", "ev", "er"]),
    ("beautiful", ["beau", "ti", "ful"]), ("tonight", ["to", "night"]), ("somebody", ["some", "bo", "dy"]),
    ("love", ["love"]), ("lalala", ["la", "la", "la"]), ("EVERYTHING", ["EV", "ERY", "THING"]),
])
def test_english_splits(word, pieces):
    assert english.syllabify(word)[0] == pieces


def test_english_every_is_two_sung_syllables():
    disp, _ = english.syllabify("Every")
    assert len(disp) == 2 and "".join(disp) == "Every"


@pytest.mark.parametrize("token,texts,romans", [
    ("yeahhh", ["yeahhh"], ["yeah"]), ("Ooooh", ["Ooooh"], ["ooh"]), ("I'm", ["I'm"], ["i'm"]),
    ("didn't", ["didn't"], ["didn't"]), ("goin'", ["goin'"], ["goin"]), ("'cause", ["'cause"], ["cause"]),
    ("BTS", ["B", "T", "S"], ["bi", "ti", "es"]), ("OK", ["O", "K"], ["ou", "kei"]),
    ("USA", ["U", "S", "A"], ["yu", "es", "ei"]), ("DJ", ["D", "J"], ["di", "jei"]),
    ("LOVE", ["LOVE"], ["love"]), ("US", ["US"], ["us"]), ("MY", ["MY"], ["my"]), ("hmm", ["hmm"], ["hmm"]),
    ("everything's", ["ev", "ery", "thing's"], ["ev", "ery", "thing's"]), ("Café", ["Ca", "fé"], ["ca", "fe"]),
])
def test_english_word(token, texts, romans):
    w = english.en_word(token, 3)
    assert [s.text for s in w.syllables] == texts
    assert [s.roman for s in w.syllables] == romans
    assert w.span == (3, 3 + len(token)) and w.lang == "en"
    assert w.syllables[0].span[0] == 3 and w.syllables[-1].span[1] == 3 + len(token)


@pytest.mark.parametrize("token,romans,review", [
    ("HMM", ["hmm"], False), ("MMM", ["mm"], False), ("SHH", ["shh"], False), ("BRR", ["brr"], False),
    ("GRR", ["grr"], False), ("PSST", ["psst"], False), ("AAH", ["ah"], False), ("OOO", ["o"], False),
    ("EHH", ["eh"], False), ("AYY", ["ay"], False), ("YEAHHH", ["yeah"], False),        # hums / elongations: words
    ("BTS", ["bi", "ti", "es"], False), ("NCT", ["en", "si", "ti"], False), ("TV", ["ti", "vi"], False),
    ("BBC", ["bi", "bi", "si"], False), ("SKZ", ["es", "kei", "zi"], True),          # unknown consonant run: unsure
])
def test_all_caps_hums_and_acronyms(token, romans, review):
    w = english.en_word(token, 0)
    assert [u.roman for s in w.syllables for u in s.units] == romans
    assert w.needs_review is review


def test_english_units_for_spoken_words():
    assert [u.roman for u in english.en_units(["twenty", "one"])] == ["twen", "ty", "one"]
    assert [u.roman for u in english.en_units(["BTS"])] == ["bi", "ti", "es"]


# --------------------------------------------------------------------------- Japanese morae / romanisation
@pytest.mark.parametrize("kana,morae", [
    ("ちょっと", ["ちょ", "っ", "と"]), ("ティーン", ["ティ", "ー", "ン"]), ("ヴァイオリン", ["ヴァ", "イ", "オ", "リ", "ン"]),
    ("デュエット", ["デュ", "エ", "ッ", "ト"]), ("ウィー", ["ウィ", "ー"]), ("イェーイ", ["イェ", "ー", "イ"]),
    ("ああぁ", ["あ", "あ", "ぁ"]), ("っゃ", ["っ", "ゃ"]), ("きょう", ["きょ", "う"]), ("クォーター", ["クォ", "ー", "タ", "ー"]),
])
def test_morae(kana, morae):
    assert japanese.morae(kana) == morae


@pytest.mark.parametrize("mora,roman", [
    ("ヲ", "o"), ("を", "o"), ("ん", "n"), ("し", "shi"), ("ち", "chi"), ("つ", "tsu"), ("ふ", "fu"), ("じ", "ji"),
    ("ぢ", "ji"), ("づ", "zu"), ("しゃ", "sha"), ("ちゅ", "chu"), ("じょ", "jo"), ("ぢゃ", "ja"), ("きゃ", "kya"),
    ("ティ", "ti"), ("ディ", "di"), ("トゥ", "tu"), ("ファ", "fa"), ("フィ", "fi"), ("フォ", "fo"), ("フュ", "fyu"),
    ("ウィ", "wi"), ("ウェ", "we"), ("ウォ", "wo"), ("イェ", "ye"), ("シェ", "she"), ("ジェ", "je"), ("チェ", "che"),
    ("ツァ", "tsa"), ("クァ", "kwa"), ("グァ", "gwa"), ("ヴァ", "va"), ("ヴ", "vu"), ("デュ", "dyu"), ("キェ", "kye"),
    ("ヷ", "va"), ("ぁ", "a"), ("ゃ", "ya"), ("ヶ", "ke"), ("っ", japanese.GEMINATE), ("ー", japanese.LONG),
])
def test_mora_roman(mora, roman):
    assert japanese.mora_roman(mora) == roman


def test_mora_table_is_mms_vocabulary():
    kana = [chr(c) for c in range(0x3041, 0x3097)] + [chr(c) for c in range(0x30A1, 0x30FB)]
    smalls = "ぁぃぅぇぉゃゅょゎ"
    for m in kana + [a + b for a in kana for b in smalls]:
        r = japanese.mora_roman(m)
        assert r in japanese.PLACEHOLDERS or set(r) <= ROMAN_OK, (m, r)


def _units(*romans: str) -> list[AlignUnit]:
    return [AlignUnit("", r) for r in romans]


def test_resolve_placeholders_context():
    Q, L = japanese.GEMINATE, japanese.LONG
    cases = [
        (("cho", Q, "to"), ["cho", "t", "to"]), (("ma", Q, "chi"), ["ma", "t", "chi"]),
        (("mi", Q, "tsu"), ["mi", "t", "tsu"]), (("i", Q, "ka"), ["i", "k", "ka"]), (("a", Q), ["a", ""]),
        (("a", Q, "o"), ["a", "", "o"]), (("to", L, "kyo", L), ["to", "o", "kyo", "o"]), ((L, "a"), ["", "a"]),
        (("ki", L, L), ["ki", "i", "i"]), (("n", L), ["n", "n"]), (("ma", Q, "", "te"), ["ma", "t", "", "te"]),
    ]
    for romans, expected in cases:
        units = _units(*romans)
        japanese.resolve_placeholders(units)
        assert [u.roman for u in units] == expected, romans
    units = _units("a", Q, "be")                 # a pause between units 1 and 2: あっ Baby
    japanese.resolve_placeholders(units, {2})
    assert [u.roman for u in units] == ["a", "", "be"]


def test_okurigana_split():
    assert japanese.okurigana_split("食べ物", "たべもの") == [("食", "た"), ("べ", "べ"), ("物", "もの")]
    assert japanese.okurigana_split("走り出せ", "ハシリダセ") == [("走", "はし"), ("り", "り"), ("出", "だ"), ("せ", "せ")]
    assert japanese.okurigana_split("こゝろ", "ココロ") == [("こ", "こ"), ("ゝ", "こ"), ("ろ", "ろ")]
    assert japanese.okurigana_split("東京", "とうきょう") == [("東京", "とうきょう")]
    assert japanese.okurigana_split("見る", "みた") is None


# --------------------------------------------------------------------------- Japanese readings (OpenJTalk)
def _ja(text: str):
    return [w for w in build_line(text, [], "ja").words if w.lang == "ja"]


@pytest.mark.parametrize("text,reading", [
    ("明日", "あした"), ("今日", "きょう"), ("2人", "ふたり"), ("1000年", "せんねん"), ("走り出せ", "はしりだせ"),
    ("何を", "なにを"), ("3月14日", "さんがつじゅうよっか"), ("ｶﾞｯｺｳ", "がっこう"),
])
def test_known_japanese_readings(text, reading):
    from lyricprompter.text import normalize_line
    assert "".join(w.reading for w in _ja(normalize_line(text))) == reading


def test_japanese_display_units_and_romans():
    words = _ja("東京タワーで走り出せ")
    syls = [s for w in words for s in w.syllables]
    assert [s.text for s in syls] == ["東京", "タ", "ワ", "ー", "で", "走", "り", "出", "せ"]
    assert [u.reading for u in syls[0].units] == ["と", "う", "きょ", "う"]
    assert "".join(s.roman for s in syls) == "tookyootawaadehashiridase"


def test_japanese_particles_and_long_vowels():
    line = build_line("今日は地球を回せ 未来へ", [], "ja")
    romans = [s.roman for s in line.syllables()]
    assert romans[:2] == ["kyoo", "wa"]
    assert "o" in romans and romans[-1] == "e"


def test_japanese_numbers_are_atomic_display_units():
    words = _ja("1,000回 2024年")
    nums = [w for w in words if w.kind == "num"]
    assert [w.text for w in nums] == ["1,000", "2024"]
    assert all(len(w.syllables) == 1 for w in nums)
    assert nums[1].reading == "にせんにじゅうよん"           # orthographic reading (furigana) ...
    assert nums[1].syllables[0].roman == "nisennijuuyo"       # ... but sung にせんにじゅうよねん
    assert not any(w.needs_review for w in nums)


def test_japanese_heteronyms_need_review():
    words = {w.text: w for w in _ja("明日の空 君の声")}
    assert words["明日"].needs_review
    assert not words["空"].needs_review and not words["声"].needs_review


def test_japanese_elongation_marks():
    line = build_line("空〜 好きーー すごーい ラ〜", [], "ja")
    syls = line.syllables()
    texts = [s.text for s in syls]
    assert texts == ["空", "〜", "好", "き", "ー", "ー", "す", "ご", "ー", "い", "ラ", "〜"]
    assert [s.roman for s in syls] == ["sora", "a", "su", "ki", "i", "i", "su", "go", "o", "i", "ra", "a"]
    assert not any(w.needs_review for w in line.words)


def test_japanese_range_tilde_reads_kara():
    line = build_line("1〜3回 5~10日", [], "ja")
    by = {w.text: w for w in line.words}
    assert by["〜"].reading == by["~"].reading == "から"
    assert [s.roman for s in by["〜"].syllables] == ["kara"] and by["〜"].needs_review
    assert by["3"].reading == "さん" and by["回"].reading == "かい"


def test_japanese_uncomposed_voicing_mark_stays_in_word():
    from lyricprompter.text import normalize_line
    text = normalize_line("あ゛ーー")
    line = build_line(text, [], "ja")
    assert [w.text for w in line.words] == ["あ゙ーー"]
    assert [s.text for s in line.syllables()] == ["あ゙", "ー", "ー"]
    assert [s.roman for s in line.syllables()] == ["a", "a", "a"]


def test_ruby_words():
    from lyricprompter.text.markup import extract_ruby
    text, ruby = extract_ruby("{東京|とう.きょう}へ 宇宙(そら) {一瞬|いっ.しゅん} {今日は|きょうは}")
    words = build_line(text, ruby, "ja").words
    by = {w.text: w for w in words}
    assert [s.text for s in by["東京"].syllables] == ["東", "京"]
    assert [s.roman for s in by["東京"].syllables] == ["too", "kyoo"]       # long vowels from OpenJTalk pron
    assert by["宇宙"].reading == "そら" and by["宇宙"].syllables[0].roman == "sora"
    assert [s.roman for s in by["一瞬"].syllables] == ["is", "shun"]
    assert "".join(s.roman for s in by["今日は"].syllables) == "kyooha"     # ruby keeps は as written
    assert not any(w.needs_review for w in words)


def test_g2p_mapping_failure_falls_back(monkeypatch):
    ojt = japanese._ojt()
    real = ojt.g2p_mapping

    def flaky(text, **kw):                       # whole segment fails, single chunks still work
        if " " in text:
            raise ValueError("cannot map spans")
        return real(text, **kw)

    monkeypatch.setattr(ojt, "g2p_mapping", flaky)
    line = build_line("明日また 会える", [], "ja")
    assert [w.text for w in line.words] == ["明日", "また", "会える"]
    assert "".join(w.reading for w in line.words) == "あしたまたあえる"

    def broken(text, **kw):
        raise ValueError("cannot map spans")

    monkeypatch.setattr(ojt, "g2p_mapping", broken)
    line = build_line("明日また 会える!", [], "ja")
    assert [w.text for w in line.words] == ["明日また", "会える"]
    assert all(w.needs_review for w in line.words)
    assert "".join(s.roman for s in line.syllables()) == "ashitamataaeru"
    for w in line.words:
        for s in w.syllables:
            assert line.text[s.span[0]:s.span[1]] == s.text
