"""Digits -> sung words.

Korean: native numerals (prenominal forms 한/두/세/네/스무, ordinal 첫) before native counters
(살, 명, 개, 시, 시간, 번째, ...), Sino-Korean (num2words) before Sino counters (년, 월, 일, 분, ...) and
otherwise.  English: num2words cardinal / ordinal / year.  Japanese numbers are read by OpenJTalk
(see `japanese.py`).  Every function also reports whether the reading is ambiguous (-> needs_review).
Digit strings with a leading zero (007) or more than MAX_DIGITS digits are read digit by digit and
flagged ambiguous.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import regex as re

NUMBER_PATTERN = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
MAX_DIGITS = 15
_RE_NUM = re.compile(rf"^(?P<num>{NUMBER_PATTERN})(?P<ord>st|nd|rd|th)?(?P<pct>%)?$", re.I)


@dataclass(frozen=True)
class NumberToken:
    integer: int             # 0 when `digits` is longer than MAX_DIGITS
    decimal: str = ""        # digits after the decimal point
    ordinal: bool = False    # 1st / 2nd / 3rd / 4th
    percent: bool = False
    digits: str = field(default="", compare=False)    # integer part as written (no commas)

    @property
    def spelled(self) -> bool:
        """Read digit by digit: leading zero ('007') or too long for a numeral reading."""
        return len(self.digits) > MAX_DIGITS or (len(self.digits) > 1 and self.digits[0] == "0")


def parse_number(token: str) -> NumberToken | None:
    m = _RE_NUM.match(token)
    if not m:
        return None
    whole, _, dec = m.group("num").replace(",", "").partition(".")
    return NumberToken(int(whole) if len(whole) <= MAX_DIGITS else 0, dec, bool(m.group("ord")),
                       bool(m.group("pct")), whole)


# --------------------------------------------------------------------------- Korean
_NATIVE_ONES = {1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉"}
_NATIVE_TENS = {10: "열", 20: "스물", 30: "서른", 40: "마흔", 50: "쉰", 60: "예순", 70: "일흔", 80: "여든",
                90: "아흔"}
NATIVE_COUNTERS = (
    "살", "살배기", "명", "개", "시", "시간", "마리", "잔", "번째", "사람", "달", "가지", "곳", "송이", "걸음",
    "장", "권", "병", "벌", "채", "켤레", "그루", "통", "줄", "쌍", "방울", "모금", "군데", "바퀴", "마디", "해",
    "밤", "끼", "판", "척", "알", "배", "놈", "곡", "글자", "발짝", "개비")
SINO_COUNTERS = (
    "년", "년생", "년대", "월", "일", "초", "원", "층", "퍼센트", "프로", "도", "호", "회", "주", "주년", "개월",
    "세", "킬로", "미터", "등", "학년", "번지", "점", "위", "차", "인분", "인", "억", "만", "천", "백", "십", "부",
    "편", "화", "기", "센치", "센티", "그램", "리터", "박", "일째")
# counter -> default reading kind when the counter alone cannot decide
AMBIGUOUS_COUNTERS = {"번": "native", "대": "native", "분": "sino"}
_COUNTERS = sorted({**{c: "native" for c in NATIVE_COUNTERS}, **{c: "sino" for c in SINO_COUNTERS},
                    **{c: "amb" for c in AMBIGUOUS_COUNTERS}}.items(), key=lambda kv: -len(kv[0]))


def ko_counter(following: str) -> tuple[str, str] | None:
    """Longest counter at the start of `following` (one space allowed) -> (counter, kind)."""
    s = following.removeprefix(" ")
    return next(((c, k) for c, k in _COUNTERS if s.startswith(c)), None)


def ko_sino(n: int, decimal: str = "") -> str:
    from num2words import num2words

    if decimal:
        return num2words(n, lang="ko").replace(" ", "") + "점" + "".join(
            num2words(int(d), lang="ko") for d in decimal)
    return num2words(n, lang="ko").replace(" ", "")


def ko_native(n: int, *, ordinal: bool = False) -> str:
    """Prenominal native numeral for 1 <= n (n >= 100 -> Sino hundreds + native rest): 20 -> 스무."""
    if ordinal and n == 1:
        return "첫"
    head, n = (ko_sino(n // 100 * 100), n % 100) if n >= 100 else ("", n)
    tens, ones = n // 10 * 10, n % 10
    if ones == 0:
        return head + ("스무" if tens == 20 else _NATIVE_TENS.get(tens, ""))
    return head + _NATIVE_TENS.get(tens, "") + _NATIVE_ONES[ones]


def ko_number(num: NumberToken, following: str) -> tuple[str, bool]:
    """Korean reading in hangul and an ambiguity flag. `following` is the line text after the digits."""
    n = num.integer
    if num.spelled:                                   # 007 -> 공공칠
        dec = "점" + "".join(ko_sino(int(d)) for d in num.decimal) if num.decimal else ""
        whole = "".join("공" if d == "0" else ko_sino(int(d)) for d in num.digits)
        return whole + dec + ("퍼센트" if num.percent else ""), True
    if num.percent:
        return ko_sino(n, num.decimal) + "퍼센트", False
    hit = ko_counter(following)
    if num.decimal or n == 0 or hit is None:
        return ko_sino(n, num.decimal), hit is None
    counter, kind = hit
    ambiguous = kind == "amb"
    if ambiguous:
        kind = "sino" if counter == "대" and n % 10 == 0 else AMBIGUOUS_COUNTERS[counter]
    if kind == "sino":
        if counter == "월" and n in (6, 10):
            return {6: "유", 10: "시"}[n], False
        return ko_sino(n), ambiguous
    return ko_native(n, ordinal=counter == "번째"), ambiguous


# --------------------------------------------------------------------------- English
def en_number_words(num: NumberToken) -> tuple[list[str], bool]:
    """English words for the number and an ambiguity flag (years, big numbers)."""
    from num2words import num2words

    n = num.integer
    ambiguous = False
    if num.spelled:
        digits = num.digits + ("." + num.decimal if num.decimal else "")
        words = ["point" if d == "." else num2words(int(d), lang="en") for d in digits]
        return words + (["percent"] if num.percent else []), True
    if num.decimal:                                   # digit by digit after the point (exact, unlike float)
        text = " ".join([num2words(n, lang="en"), "point", *(num2words(int(d), lang="en") for d in num.decimal)])
    elif num.ordinal:
        text = num2words(n, lang="en", to="ordinal")
    elif 1100 <= n <= 1999 or 2010 <= n <= 2099:
        text, ambiguous = num2words(n, lang="en", to="year"), True
    else:
        text, ambiguous = num2words(n, lang="en"), n >= 1000
    if num.percent:
        text += " percent"
    return [w for w in re.split(r"[\s,\-]+", text) if w], ambiguous
