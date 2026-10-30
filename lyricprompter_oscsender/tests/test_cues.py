from __future__ import annotations

import json
import math
import random
import struct

import pytest

from lyricprompter_oscsender import cues
from lyricprompter_oscsender.cues import (
    Cue,
    CueEditError,
    CueIssue,
    CueSyntaxError,
    TargetDecl,
    TimeStyle,
    add_cue,
    delete_cue,
    format_arg,
    format_args,
    format_time,
    parse_cues,
    parse_message,
    parse_time,
    set_cue_message,
    set_cue_time,
    sheet_rev,
    shift_cue,
    time_style,
)
from lyricprompter_oscsender.osc import encode_message

SPEC_EXAMPLE = """\
# lines starting with # are comments; blank lines are ignored
@target lights 192.168.0.50:8000
@target video  127.0.0.1:7000  latency=80ms

0:00.000                 /show/start
0:12.480     lights      /cue/go 12                # verse 1 wash
1:05.2       video       /clip/play "intro.mp4" 0.5
75.300                   /mixer/fader f:0.75 i:3 true
[01:20.00]               /scene "two words" nil
"""


def errors(sheet):
    return [i for i in sheet.issues if i.severity == "error"]


def warnings(sheet):
    return [i for i in sheet.issues if i.severity == "warning"]


def one_cue(line: str) -> Cue:
    sheet = parse_cues(line)
    assert not sheet.issues, sheet.issues
    assert len(sheet.cues) == 1
    return sheet.cues[0]


def line_error(line: str) -> str:
    sheet = parse_cues(line)
    assert not sheet.cues
    errs = errors(sheet)
    assert len(errs) == 1 and errs[0].line == 1, sheet.issues
    return errs[0].message


# ----------------------------------------------------------------------------------------- spec example


def test_spec_example():
    sheet = parse_cues(SPEC_EXAMPLE)
    assert sheet.issues == []
    assert sheet.text == SPEC_EXAMPLE
    assert sheet.targets == [TargetDecl("lights", "192.168.0.50", 8000, None, 2),
                             TargetDecl("video", "127.0.0.1", 7000, 80.0, 3)]
    assert [c.id for c in sheet.cues] == [5, 6, 7, 8, 9]
    assert [c.time for c in sheet.cues] == [0.0, 12.48, 65.2, 75.3, 80.0]
    c = sheet.by_id(6)
    assert (c.target, c.address, c.args, c.message, c.note) == ("lights", "/cue/go", (("i", 12),),
                                                                "/cue/go 12", "verse 1 wash")
    c = sheet.by_id(7)
    assert (c.target, c.address, c.args) == ("video", "/clip/play", (("s", "intro.mp4"), ("f", 0.5)))
    assert c.message == '/clip/play "intro.mp4" 0.5'
    c = sheet.by_id(8)
    assert c.target is None and c.args == (("f", 0.75), ("i", 3), ("T", True))
    c = sheet.by_id(9)
    assert c.args == (("s", "two words"), ("N", None)) and c.message == '/scene "two words" nil'
    assert sheet.by_id(1) is None and sheet.n_errors == 0 and sheet.n_warnings == 0


def test_sheet_to_dict_is_json():
    sheet = parse_cues(SPEC_EXAMPLE + "bad line\n")
    d = json.loads(json.dumps(sheet.to_dict()))
    assert d["rev"] == sheet_rev(SPEC_EXAMPLE + "bad line\n") and d["text"] == sheet.text
    assert d["cues"][1] == {"id": 6, "time": 12.48, "target": "lights", "address": "/cue/go",
                            "args": [["i", 12]], "message": "/cue/go 12", "note": "verse 1 wash"}
    assert d["targets"][1] == {"name": "video", "host": "127.0.0.1", "port": 7000, "latency_ms": 80.0,
                               "line": 3}
    assert d["issues"] == [{"line": 10, "severity": "error", "message": d["issues"][0]["message"]}]


def test_sheet_rev():
    assert sheet_rev("") == "da39a3ee5e6b"
    assert len(sheet_rev(SPEC_EXAMPLE)) == 12 and sheet_rev(SPEC_EXAMPLE) != sheet_rev(SPEC_EXAMPLE + " ")
    assert sheet_rev("x\ud800") != sheet_rev("x")         # lone surrogates do not raise


def test_cue_key_and_body():
    a = one_cue("1:00 lights /a 1 # n")
    b = one_cue("60.0000 lights /a 1")
    assert a.key() == b.key()
    assert a.body() == "lights /a 1 # n" and b.body() == "lights /a 1"
    assert parse_message(a.body()) == ("lights", "/a", (("i", 1),), "n")


# ----------------------------------------------------------------------------------------- times


@pytest.mark.parametrize("token, seconds", [
    ("0", 0.0), ("75", 75.0), ("75.3", 75.3), ("75.300", 75.3), ("75.", 75.0), (".5", 0.5), ("0.5", 0.5),
    ("1:05", 65.0), ("1:05.2", 65.2), ("01:05.200", 65.2), ("1:5", 65.0), ("0:59.999", 59.999),
    ("75:00", 4500.0), ("1:01:05.2", 3665.2), ("01:00:00", 3600.0), ("1:05,2", 65.2), ("75,3", 75.3),
    ("[01:20.00]", 80.0), ("[1:05,2]", 65.2), ("[75.3]", 75.3), ("[1:00:00.5]", 3600.5), (" 1:05 ", 65.0),
    ("1:59:59.999", 7199.999),
])
def test_parse_time(token, seconds):
    assert parse_time(token) == seconds


def test_parse_time_same_float_for_every_style():
    assert parse_time("1:05.2") == parse_time("65.2") == parse_time("0:01:05.200") == parse_time("[01:05,2]")
    assert parse_time("12:34.567") == float("754.567")


@pytest.mark.parametrize("token, fragment", [
    ("1:60", "seconds must be < 60"), ("1:61.5", "seconds must be < 60"),
    ("1:60:00", "minutes must be < 60"), ("1:00:60", "seconds must be < 60"),
    ("-1", "negative"), ("-0:05", "negative"), ("[-1:00]", "negative"),
    ("01:02:03:04", "frames are not supported"), ("1:2:3:4:5", "not a time"),
    ("", "not a time"), ("abc", "not a time"), ("1:", "not a time"), (":05", "not a time"),
    ("1.5:30", "not a time"), ("1:05.2.3", "not a time"), ("1,05.2", "not a time"), ("1,2,3", "not a time"),
    ("[1:05", "brackets"), ("1:05]", "brackets"), ("[]", "not a time"), ("[", "brackets"),
    ("+5", "not a time"), ("1e3", "not a time"), ("１:０５", "not a time"), ("inf", "not a time"),
    ("nan", "not a time"), (".", "not a time"),
])
def test_parse_time_errors(token, fragment):
    with pytest.raises(CueSyntaxError, match=fragment):
        parse_time(token)
    with pytest.raises(ValueError):
        parse_time(token)


@pytest.mark.parametrize("token, style", [
    ("75.3", TimeStyle("s", False, False, False)),
    ("05.3", TimeStyle("s", True, False, False)),
    ("0.5", TimeStyle("s", False, False, False)),
    ("1:05.2", TimeStyle("ms", False, False, False)),
    ("01:05.200", TimeStyle("ms", True, False, False)),
    ("[01:20.00]", TimeStyle("ms", True, True, False)),
    ("1:05,2", TimeStyle("ms", False, False, True)),
    ("1:01:05.2", TimeStyle("hms", False, False, False)),
    ("[00:01:05]", TimeStyle("hms", True, True, False)),
])
def test_time_style(token, style):
    assert time_style(token) == style
    assert time_style(token).style == style[0]                 # also reads as (style, pad, brackets, comma)


def test_time_style_rejects_invalid():
    with pytest.raises(ValueError):
        time_style("1:99")


@pytest.mark.parametrize("t, kwargs, text", [
    (75.3, {"style": "s"}, "75.300"),
    (65.2, {}, "1:05.200"),
    (65.2, {"style": "ms", "pad": True}, "01:05.200"),
    (3665.2, {"style": "hms"}, "1:01:05.200"),
    (65.2, {"style": "hms", "pad": True}, "00:01:05.200"),
    (4500.0, {"style": "ms"}, "75:00.000"),
    (80.0, {"style": "ms", "pad": True, "brackets": True}, "[01:20.000]"),
    (65.2, {"comma": True}, "1:05,200"),
    (5.0, {"style": "s", "pad": True}, "05.000"),
    (0.0, {}, "0:00.000"),
    (59.9996, {}, "1:00.000"),                       # rounded to the millisecond (carries into minutes)
    (1.0004, {"style": "s"}, "1.000"),
    (-0.0001, {"style": "s"}, "0.000"),              # rounds to 0: not negative
])
def test_format_time(t, kwargs, text):
    assert format_time(t, **kwargs) == text


def test_format_time_errors():
    for bad in (-0.001, -5.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            format_time(bad)
    with pytest.raises(ValueError, match="style"):
        format_time(1.0, "hm")


def test_format_time_round_trip_through_time_style():
    rng = random.Random(1)
    for _ in range(300):
        t = round(rng.uniform(0, 3 * 3600), 3)
        for style in ("s", "ms", "hms"):
            for pad in (False, True):
                for brackets in (False, True):
                    for comma in (False, True):
                        text = format_time(t, style, pad=pad, brackets=brackets, comma=comma)
                        assert abs(parse_time(text) - t) < 1e-6, text
                        st = time_style(text)
                        assert st.style == style and st.brackets == brackets and st.comma == comma
                        assert format_time(t, **st._asdict()) == text      # same style written back


# ----------------------------------------------------------------------------------------- targets / addresses


def test_target_versus_address():
    assert one_cue("1 /a").target is None
    assert one_cue("1 main /a").target == "main"
    assert one_cue("1 _x.y-2 /a").target == "_x.y-2"
    assert one_cue("1 /a /b").args == (("s", "/b"),)            # later tokens starting with / are strings
    assert "neither a target name" in line_error("1 2lights /a")
    assert "neither a target name" in line_error('1 "lights" /a')
    assert "neither a target name" in line_error("1 lights! /a")
    assert "missing OSC address after target" in line_error("1 lights")
    assert "missing OSC address after target" in line_error("1 lights # note")
    assert "expected an OSC address" in line_error("1 lights video /a")
    assert "expected an OSC address" in line_error('1 lights "/a"')
    assert "missing OSC address after the time" in line_error("1:00")
    assert "missing OSC address after the time" in line_error("1:00   # just a note")


@pytest.mark.parametrize("address", ["/a", "/a/b/c", "/", "/a/*", "/a/[1-3]", "/a/{x,y}/go", "/x?", "/A_b-c.d"])
def test_valid_addresses(address):
    assert one_cue(f"1 {address}").address == address


@pytest.mark.parametrize("address", ["/a,b", "/a\x01", "/é"])
def test_invalid_addresses(address):
    assert "not allowed" in line_error(f"1 {address}")


def test_address_with_hash_inside_is_invalid_but_hash_token_is_a_note():
    assert "not allowed" in line_error("1 /a#b")
    c = one_cue("1 /a #b")
    assert c.address == "/a" and c.note == "b" and c.args == ()


# ----------------------------------------------------------------------------------------- arguments


def args_of(text: str):
    return parse_message("/x " + text)[2]


@pytest.mark.parametrize("text, args", [
    ("12", (("i", 12),)), ("-7", (("i", -7),)), ("+7", (("i", 7),)), ("0", (("i", 0),)),
    ("2147483647", (("i", 2147483647),)), ("-2147483648", (("i", -2147483648),)),
    ("2147483648", (("h", 2147483648),)), ("-2147483649", (("h", -2147483649),)),
    ("9223372036854775807", (("h", 9223372036854775807),)),
    ("1.5", (("f", 1.5),)), ("-.5", (("f", -0.5),)), ("1e3", (("f", 1000.0),)), ("2.", (("f", 2.0),)),
    ("1E-3", (("f", 0.001),)), ("+.25e+1", (("f", 2.5),)), ("3.4e38", (("f", 3.4e38),)),
    ("true", (("T", True),)), ("FALSE", (("F", False),)), ("nil", (("N", None),)), ("Null", (("N", None),)),
    ("i:5", (("i", 5),)), ("i:-5", (("i", -5),)), ("h:5", (("h", 5),)), ("f:1", (("f", 1.0),)),
    ("d:0.1", (("d", 0.1),)), ("d:1e300", (("d", 1e300),)), ("s:123", (("s", "123"),)), ("s:", (("s", ""),)),
    ('s:"a b"', (("s", "a b"),)), ("s:true", (("s", "true"),)), ("s:#x", (("s", "#x"),)),
    ('"a b"', (("s", "a b"),)), ("'a b'", (("s", "a b"),)), ('""', (("s", ""),)), ("''", (("s", ""),)),
    ('"123"', (("s", "123"),)), ('"true"', (("s", "true"),)), ('"# not a note"', (("s", "# not a note"),)),
    (r'"say \"hi\""', (("s", 'say "hi"'),)), (r"'it\'s'", (("s", "it's"),)), (r'"a\\b"', (("s", "a\\b"),)),
    (r'"a\nb"', (("s", "anb"),)), ("\"it's\"", (("s", "it's"),)), ("'say \"x\"'", (("s", 'say "x"'),)),
    ("hello", (("s", "hello"),)), ("don't", (("s", "don't"),)), ('a"b', (("s", 'a"b'),)),
    ("a#b", (("s", "a#b"),)), ("intro.mp4", (("s", "intro.mp4"),)), ("x:1", (("s", "x:1"),)),
    ("1:30", (("s", "1:30"),)), ("inf", (("s", "inf"),)), ("nan", (("s", "nan"),)), ("-", (("s", "-"),)),
    ("1_000", (("s", "1_000"),)), ("０", (("s", "０"),)), ("I:5", (("s", "I:5"),)), (r"a\b", (("s", "a\\b"),)),
    ("한글", (("s", "한글"),)), ("\t1\t2 ", (("i", 1), ("i", 2))),
    ('"a b" 1 f:0.5 true nil x', (("s", "a b"), ("i", 1), ("f", 0.5), ("T", True), ("N", None), ("s", "x"))),
])
def test_arguments(text, args):
    assert args_of(text) == args


@pytest.mark.parametrize("text, fragment", [
    ("9223372036854775808", "does not fit 64 bits"), ("-9223372036854775809", "does not fit 64 bits"),
    ("i:2147483648", "write h:2147483648"), ("i:1.5", "needs an integer"), ("i:x", "needs an integer"),
    ("i:", "needs an integer"), ("h:9223372036854775808", "does not fit type 'h'"), ("h:1.0", "needs an integer"),
    ("3.5e38", "write d:3.5e38"), ("-1e39", "float32"), ("f:1e39", "write d:1e39"), ("1e400", "out of range"),
    ("d:1e400", "out of range"), ("f:abc", "needs a number"), ("d:", "needs a number"), ("f:inf", "needs a number"),
    ('"abc', "unterminated string"), ("'abc", "unterminated string"), ('"abc\\"', "unterminated string"),
    ('"ab\\', "unterminated string"), ('"abc"def', "put a space after the closing quote"),
    ('s:"abc', "unterminated"), ('"a"#note', "put a space after"), ("i:" + "9" * 5000, "out of range"),
])
def test_argument_errors(text, fragment):
    with pytest.raises(CueSyntaxError, match=fragment):
        parse_message("/x " + text)
    assert fragment in line_error("1:00 /x " + text)


def test_nul_in_string_is_an_error():
    assert "NUL" in line_error('1 /x "a\x00b"')


def test_typed_prefix_with_quoted_value():
    assert args_of('i:"5" f:"1.5" d:"2" h:"7"') == (("i", 5), ("f", 1.5), ("d", 2.0), ("h", 7))
    with pytest.raises(CueSyntaxError):
        args_of('i:"x"')


def test_float_values_are_python_floats_of_the_text():
    (tag, v), = args_of("0.1")
    assert tag == "f" and v == 0.1                       # not rounded to float32 in the cue
    assert struct.unpack(">f", encode_message("/x", [("f", v)])[8:])[0] == pytest.approx(0.1)


# ----------------------------------------------------------------------------------------- notes


def test_notes():
    c = one_cue("1:00 /a 1 #   wash   ")
    assert c.note == "wash" and c.message == "/a 1"
    c = one_cue("1:00 /a 1 # two # hashes")
    assert c.note == "two # hashes"
    c = one_cue('1:00 /a "# not a note" # real note')
    assert c.args == (("s", "# not a note"),) and c.note == "real note" and c.message == '/a "# not a note"'
    c = one_cue("1:00 /a 1 #")
    assert c.note == "" and c.message == "/a 1"
    c = one_cue("1:00 lights /a\t#\tnote 한글")
    assert c.note == "note 한글" and c.target == "lights"
    assert parse_message("lights /a 1 # hi") == ("lights", "/a", (("i", 1),), "hi")
    assert parse_message("/a") == (None, "/a", (), "")


def test_message_is_verbatim_and_trimmed():
    c = one_cue("  1:00\t\tlights   /a    1   'x  y'    # n")
    assert c.message == "/a    1   'x  y'"
    c = one_cue("1:00 /a   ")
    assert c.message == "/a"


def test_parse_message_errors():
    for bad in ["", "   ", "# note only", "lights", "lights # n", "/a\n1", "/a\r", "1:00 /a", "/a 'x"]:
        with pytest.raises(CueSyntaxError):
            parse_message(bad)


# ----------------------------------------------------------------------------------------- directives


@pytest.mark.parametrize("line, decl", [
    ("@target lights 192.168.0.50:8000", TargetDecl("lights", "192.168.0.50", 8000, None, 1)),
    ("@target v  127.0.0.1:7000  latency=80ms", TargetDecl("v", "127.0.0.1", 7000, 80.0, 1)),
    ("@target v 127.0.0.1:7000 latency=80", TargetDecl("v", "127.0.0.1", 7000, 80.0, 1)),
    ("@target v host.local:1 latency=12.5MS", TargetDecl("v", "host.local", 1, 12.5, 1)),
    ("@target v [::1]:9000", TargetDecl("v", "::1", 9000, None, 1)),
    ("@target v [fe80::1%en0]:65535 latency=0", TargetDecl("v", "fe80::1%en0", 65535, 0.0, 1)),
    ("  @TARGET v 10.0.0.255:53000   # broadcast", TargetDecl("v", "10.0.0.255", 53000, None, 1)),
])
def test_target_directive(line, decl):
    sheet = parse_cues(line)
    assert sheet.issues == [] and sheet.targets == [decl] and sheet.cues == []


@pytest.mark.parametrize("line, fragment", [
    ("@target", "incomplete @target"), ("@target v", "incomplete @target"),
    ("@target 1v 1.2.3.4:5", "not a valid target name"), ('@target "v" 1.2.3.4:5', "not a valid target name"),
    ("@target v 1.2.3.4", "missing port"), ("@target v 1.2.3.4:", "port must be"),
    ("@target v 1.2.3.4:0", "port must be"), ("@target v 1.2.3.4:65536", "port must be"),
    ("@target v 1.2.3.4:http", "port must be"), ("@target v :9000", "missing host"),
    ("@target v ::1:9000", "IPv6 host in brackets"), ("@target v [::1]9000", "[addr]:port"),
    ("@target v [::1:9000", "[addr]:port"), ("@target v []:9000", "missing host"),
    ("@target v h:1 latency=-5", "latency is a number"), ("@target v h:1 latency=", "latency is a number"),
    ("@target v h:1 latency=5s", "latency is a number"), ("@target v h:1 latency=6000", "at most 5000"),
])
def test_target_directive_errors(line, fragment):
    sheet = parse_cues(line)
    assert sheet.targets == []
    assert [i.severity for i in sheet.issues] == ["error"] and fragment in sheet.issues[0].message


def test_directive_warnings():
    sheet = parse_cues("@target v h:1 80ms\n@foo bar\n@target v other:2\n@target w h:3 colour=red latency=5\n")
    assert sheet.targets == [TargetDecl("v", "h", 1, None, 1), TargetDecl("w", "h", 3, 5.0, 4)]
    assert [(i.line, i.severity) for i in sheet.issues] == [(1, "warning"), (2, "warning"), (3, "warning"),
                                                            (4, "warning")]
    assert "'80ms'" in sheet.issues[0].message
    assert "unknown directive '@foo'" in sheet.issues[1].message
    assert "already declared on line 1" in sheet.issues[2].message


# ----------------------------------------------------------------------------------------- whole files


def test_errors_never_abort_parsing():
    text = """\
1:00 /ok1
nonsense
1:99 /bad
@target
2 /x "open
3 lights
-1 /neg
4 /a,b
5 /x 1e999
6 /ok2
1:00:00:00 /frames
7 /ok3"""
    sheet = parse_cues(text)
    assert [(c.id, c.address) for c in sheet.cues] == [(10, "/ok2"), (12, "/ok3"), (1, "/ok1")]
    assert [i.line for i in errors(sheet)] == [2, 3, 4, 5, 6, 7, 8, 9, 11]


def test_garbage_never_raises():
    rng = random.Random(7)
    alphabet = ' \t\r\n#@/"\'\\:.,[]-+eE019abiTF:sdhfnul' + "한\x00﻿"
    for _ in range(500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
        sheet = parse_cues(text)
        assert all(isinstance(i, CueIssue) for i in sheet.issues)


def test_cues_sorted_by_time_then_line():
    sheet = parse_cues("5 /e\n1 /a\n3 /c\n1 /b\n0:03 /d\n")
    assert [(c.time, c.id, c.address) for c in sheet.cues] == [(1.0, 2, "/a"), (1.0, 4, "/b"), (3.0, 3, "/c"),
                                                               (3.0, 5, "/d"), (5.0, 1, "/e")]
    assert sheet.issues == []                           # unsorted lines are fine


def test_duplicate_warnings():
    sheet = parse_cues("1:00 lights /a 1\n60 lights /a 1 # same\n1:00 /a 1\n1:00 lights /a 2\n"
                       "1:00.0004 lights /a 1\n1:00 lights /a 1.0\n0:01:00 lights /a 1\n")
    assert len(sheet.cues) == 7                          # duplicates are kept (and fire), only reported
    assert [(i.line, i.severity) for i in sheet.issues] == [(2, "warning"), (7, "warning")]
    assert all("duplicate of the cue on line 1" in i.message for i in sheet.issues)


def test_bom_and_crlf():
    text = "﻿@target v h:1\r\n1:00 v /a 1\r\n\r\n# c\r\n2:00 /b\r\n"
    sheet = parse_cues(text)
    assert sheet.issues == [] and [c.id for c in sheet.cues] == [2, 5]
    assert sheet.targets[0].name == "v" and sheet.cues[1].message == "/b"
    first_line_cue = parse_cues("﻿1:00 /a\n")
    assert first_line_cue.issues == [] and first_line_cue.cues[0].id == 1


def test_other_whitespace_are_separators():
    c = one_cue("1:00　/a 1")                       # ideographic / no-break spaces
    assert c.address == "/a" and c.args == (("i", 1),)


def test_lone_cr_line_endings():
    sheet = parse_cues("0:01.000\tlights\t/cue/go 1\r0:02.000\tlights\t/cue/go 2\r0:03.000\tlights\t/cue/go 3\r")
    assert sheet.issues == []
    assert [(c.id, c.time, c.args) for c in sheet.cues] == [(1, 1.0, (("i", 1),)), (2, 2.0, (("i", 2),)),
                                                            (3, 3.0, (("i", 3),))]
    sheet = parse_cues("# show\r0:01.000 /a\r0:02.000 /b\r")
    assert sheet.issues == [] and [(c.id, c.address) for c in sheet.cues] == [(2, "/a"), (3, "/b")]
    sheet = parse_cues("1:00 /a\r\n1:01 /b\r1:02 /c\n1:03 /d")        # mixed endings: a stray CR still splits
    assert sheet.issues == [] and [c.id for c in sheet.cues] == [1, 2, 3, 4]
    assert [i.line for i in parse_cues("1:00\r/a 1\n").issues] == [1, 2]      # not one line with a space
    text = "# a\r1:00 /a\r2:00 /b"
    assert set_cue_time(text, 2, 61) == "# a\r1:01.000 /a\r2:00 /b"
    assert add_cue(text, 90, "/c") == ("# a\r1:00 /a\r1:30.000 /c\r2:00 /b", 3)
    assert add_cue("1:00 /a\r", 90, "/c") == ("1:00 /a\r1:30.000 /c\r", 2)
    assert delete_cue(text, 2) == "# a\r2:00 /b"
    assert delete_cue(text, 3) == "# a\r1:00 /a\r"


@pytest.mark.parametrize("ch", ["\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", " ", " "])
def test_other_line_break_characters_are_reported(ch):
    sheet = parse_cues(f"# show{ch}0:01.000 /a\n1:00 /b{ch}1\n@target v h:1{ch}\n{ch}\n")
    assert [c.id for c in sheet.cues] == [2]                  # read as part of their line, as before
    assert [(i.line, i.severity) for i in sheet.issues] == [(1, "warning"), (2, "warning"), (3, "warning")]
    assert all("line break" in i.message for i in sheet.issues)
    assert parse_cues(f"1:00 /b 'x{ch}y' s:\"{ch}\" # n\n").issues == []     # text inside a quoted string


def test_hash_inside_a_bare_argument_is_reported():
    sheet = parse_cues("0:12.480 lights /cue/go 12# verse 1 wash\n")
    assert sheet.cues[0].args == (("s", "12#"), ("s", "verse"), ("i", 1), ("s", "wash"))   # spec rule unchanged
    assert [(i.line, i.severity) for i in sheet.issues] == [(1, "warning")]
    assert "'12#'" in sheet.issues[0].message and "space before #" in sheet.issues[0].message
    for ok in ['1:00 /a 12 # note', '1:00 /a "12#"', "1:00 /a s:12#", "1:00 /a s:'C#'", "1:00 /a 'C#' # n"]:
        assert parse_cues(ok).issues == [], ok
    assert format_arg("s", "C#") == '"C#"' and parse_cues(f"1:00 /a {format_arg('s', 'C#')}").issues == []


@pytest.mark.parametrize("token", ["1" * 400, "1" * 309, "1" * 307 + ".5", "360000.001", "6000:00.001",
                                   "100:00:00.001", "[99999999:00]"])
def test_time_beyond_the_limit_is_an_error(token):
    with pytest.raises(CueSyntaxError, match="beyond the latest time"):
        parse_time(token)
    sheet = parse_cues(f"{token} /a 1\n2:00 /b\n")
    assert [c.id for c in sheet.cues] == [2] and [(i.line, i.severity) for i in sheet.issues] == [(1, "error")]
    cues.format_sheet(sheet)
    json.dumps(sheet.to_dict(), allow_nan=False)
    with pytest.raises(CueEditError):
        add_cue("1:00 /a\n", token, "/b")
    with pytest.raises(CueEditError):
        set_cue_time("1:00 /a\n", 1, token)


def test_time_limit_is_inclusive_and_applies_to_numeric_edits():
    assert parse_time("100:00:00") == parse_time("360000") == cues.MAX_TIME_S
    assert set_cue_time("1:00 /a\n", 1, cues.MAX_TIME_S) == "6000:00.000 /a\n"
    for bad in (cues.MAX_TIME_S + 0.001, 1e306, 1.7e308):
        with pytest.raises(CueEditError, match="at most 360000 s"):
            set_cue_time("1:00 /a\n", 1, bad)
        with pytest.raises(CueEditError, match="at most 360000 s"):
            add_cue("1:00 /a\n", bad, "/b")
        with pytest.raises(CueEditError, match="at most 360000 s"):
            shift_cue("1:00 /a\n", 1, bad)
    with pytest.raises(CueEditError, match="before 0"):
        shift_cue("1:00 /a\n", 1, -1.7e308)


def test_comment_and_blank_lines():
    sheet = parse_cues("# only comments\n   \n\t# indented comment\n\n")
    assert sheet.cues == [] and sheet.issues == [] and sheet.targets == []
    assert parse_cues("").cues == []


# ----------------------------------------------------------------------------------------- format_arg(s)


@pytest.mark.parametrize("tag, value, text", [
    ("i", 12, "12"), ("i", -2147483648, "-2147483648"), ("h", 5, "h:5"), ("h", -(2 ** 63), f"h:{-(2 ** 63)}"),
    ("f", 0.5, "0.5"), ("f", 1, "1.0"), ("f", 1e20, "1.0e+20"), ("f", -1e-7, "-1.0e-07"), ("f", 1.5e-07, "1.5e-07"),
    ("f", -0.0, "-0.0"), ("d", 0.1, "d:0.1"), ("d", 1e300, "d:1.0e+300"), ("T", True, "true"), ("F", None, "false"),
    ("N", None, "nil"), ("s", "abc", "abc"), ("s", "", '""'), ("s", "a b", '"a b"'), ("s", "123", '"123"'),
    ("s", "1.5", '"1.5"'), ("s", "TRUE", '"TRUE"'), ("s", "null", '"null"'), ("s", "#x", '"#x"'),
    ("s", "x#", '"x#"'), ("s", '"q', '"\\"q"'), ("s", "'q", "\"'q\""), ("s", 'a"b', 'a"b'), ("s", "a\\b", "a\\b"),
    ("s", "a\\ b", '"a\\\\ b"'), ("s", "s:x", '"s:x"'), ("s", "f:1", '"f:1"'), ("s", "x:1", "x:1"),
    ("s", "/path", "/path"), ("s", "a\tb", '"a\tb"'), ("s", "한글", "한글"), ("s", "1e5", '"1e5"'),
])
def test_format_arg(tag, value, text):
    assert format_arg(tag, value) == text
    if tag not in "TFN":
        assert args_of(text) == ((tag, float(value) if tag in "fd" else value),)


@pytest.mark.parametrize("tag, value", [
    ("i", 2 ** 31), ("i", 1.0), ("i", True), ("i", "1"), ("h", 2 ** 63), ("f", 1e39), ("f", math.inf),
    ("d", math.nan), ("f", "1"), ("f", True), ("s", 5), ("s", "a\nb"), ("s", "a\rb"), ("s", "a\x00"),
    ("b", b"x"), ("I", None), ("x", 1),
])
def test_format_arg_errors(tag, value):
    with pytest.raises(ValueError):
        format_arg(tag, value)


def _random_string(rng: random.Random) -> str:
    pool = ['a', 'Z', '0', '9', ' ', '\t', '"', "'", '\\', '#', ':', '.', '-', '+', 'e', '/', '[', ']', '한',
            'é', 's', 'i', 'f', 'd', 'h', 'true', 'nil', '　', ' ', '﻿', '😀']
    return "".join(rng.choice(pool) for _ in range(rng.randint(0, 8)))


def _random_arg(rng: random.Random):
    tag = rng.choice("ihfdsTFN")
    if tag == "i":
        return tag, rng.choice([0, 1, -1, 2 ** 31 - 1, -(2 ** 31), rng.randint(-(2 ** 31), 2 ** 31 - 1)])
    if tag == "h":
        return tag, rng.choice([0, 5, 2 ** 63 - 1, -(2 ** 63), rng.randint(-(2 ** 63), 2 ** 63 - 1)])
    if tag == "f":
        exact32 = struct.unpack(">f", struct.pack(">f", rng.uniform(-1e6, 1e6)))[0]
        return tag, rng.choice([0.0, -0.0, 1.0, exact32, rng.uniform(-1e30, 1e30), 1e-40, 3.4028234663852886e38,
                                rng.random()])
    if tag == "d":
        return tag, rng.choice([0.1, 1e300, -1e-300, rng.uniform(-1e100, 1e100), 2.0 ** -1074])
    if tag == "s":
        return tag, _random_string(rng)
    return tag, {"T": True, "F": False, "N": None}[tag]


def test_format_args_round_trip_property():
    rng = random.Random(1234)
    for _ in range(2000):
        args = tuple(_random_arg(rng) for _ in range(rng.randint(0, 6)))
        text = format_args(args)
        target, address, parsed, note = parse_message(f"/x {text}")
        assert (target, address, note) == (None, "/x", "")
        assert parsed == args, text
        for (_, a), (_, b) in zip(parsed, args, strict=True):
            assert type(a) is type(b) or (a is None and b is None)
        # and inside a cue line, with a note and a target
        sheet = parse_cues(f"1:00 lights /x {text} # note\n")
        assert sheet.issues == [] and sheet.cues[0].args == args and sheet.cues[0].note == "note"


def test_format_args_empty():
    assert format_args(()) == "" and format_args([("i", 1), ("s", "a b")]) == '1 "a b"'


# ----------------------------------------------------------------------------------------- editing


EDIT_TEXT = """\
# show
@target lights 10.0.0.2:8000
0:05.000     lights      /cue/go 1   # first
0:12.480     lights      /cue/go 2
1:05.2       /clip/play "intro.mp4"
"""


def test_set_cue_time_keeps_other_lines_and_alignment():
    out = set_cue_time(EDIT_TEXT, 4, 13.0)
    assert out == EDIT_TEXT.replace("0:12.480     lights", "0:13.000     lights")
    out = set_cue_time(EDIT_TEXT, 4, 612.25)                 # longer token: the gap shrinks
    assert "10:12.250    lights      /cue/go 2\n" in out
    assert out.splitlines()[4] == '1:05.2       /clip/play "intro.mp4"'
    out = set_cue_time(EDIT_TEXT, 5, 65.0)                   # "1:05.2" -> "1:05.000", single-space columns
    assert out.splitlines()[4] == '1:05.000     /clip/play "intro.mp4"'
    assert parse_cues(out).by_id(5).time == 65.0


@pytest.mark.parametrize("line, t, new", [
    ("75.3 /a", 80, "80.000 /a"), ("05.3 /a", 7.5, "07.500 /a"), ("1:05.2 /a", 3, "0:03.000 /a"),
    ("01:05.2 /a", 3, "00:03.000 /a"), ("[01:20.00] /a", 81.5, "[01:21.500] /a"),
    ("1:05,2 /a # x", 66, "1:06,000 /a # x"), ("1:01:05.2 /a", 3600.25, "1:00:00.250 /a"),
    ("\t1:00\t/a", 2, "\t0:02.000\t/a"), ("1:00 /a", "1:30.5", "1:30.500 /a"), ("1:00 /a", "[95]", "1:35.000 /a"),
    ("1:00 /a", 0.0004, "0:00.000 /a"), ("1:00  /a", 600, "10:00.000 /a"),
])
def test_set_cue_time_styles(line, t, new):
    assert set_cue_time(line, 1, t) == new
    assert set_cue_time(line + "\r\n", 1, t) == new + "\r\n"


def test_set_cue_time_errors():
    for bad in (-1, -0.001, math.nan, math.inf, "abc", "-1:00", None, True, "1:99"):
        with pytest.raises(CueEditError):
            set_cue_time(EDIT_TEXT, 3, bad)
    for bad_id in (0, 1, 2, 6, 99, -1, "3", 3.0, None, True):
        with pytest.raises(CueEditError):
            set_cue_time(EDIT_TEXT, bad_id, 1.0)
    with pytest.raises(CueEditError, match="not a valid cue"):
        set_cue_time("1:99 /a\n", 1, 1.0)
    with pytest.raises(CueEditError):
        set_cue_time("", 1, 1.0)


def test_shift_cue():
    out = shift_cue(EDIT_TEXT, 4, 0.01)
    assert "0:12.490     lights" in out and parse_cues(out).by_id(4).time == 12.49
    out = shift_cue(EDIT_TEXT, 3, -5.0)
    assert out.splitlines()[2].startswith("0:00.000     lights")
    with pytest.raises(CueEditError, match="before 0:00"):
        shift_cue(EDIT_TEXT, 3, -5.001)
    for bad in (math.nan, "x", None):
        with pytest.raises(CueEditError):
            shift_cue(EDIT_TEXT, 3, bad)
    with pytest.raises(CueEditError):
        shift_cue(EDIT_TEXT, 2, 1.0)
    text = "0:00.100 /a\n"
    for _ in range(10):
        text = shift_cue(text, 1, 0.01)
    assert text == "0:00.200 /a\n"


def test_set_cue_message():
    out = set_cue_message(EDIT_TEXT, 4, "video /clip/stop 'a b' # new note")
    assert out.splitlines()[3] == "0:12.480     video /clip/stop 'a b' # new note"
    assert out.replace(out.splitlines()[3], "") == EDIT_TEXT.replace(EDIT_TEXT.splitlines()[3], "")
    c = parse_cues(out).by_id(4)
    assert (c.target, c.address, c.args, c.note) == ("video", "/clip/stop", (("s", "a b"),), "new note")
    out = set_cue_message(EDIT_TEXT, 3, "  /plain  ")                     # target and note dropped
    assert out.splitlines()[2] == "0:05.000     /plain"
    for bad in ("", "lights", "/a b\nc", "/a,b", "/a 'x", "2bad /a", 5, None):
        with pytest.raises(CueEditError):
            set_cue_message(EDIT_TEXT, 3, bad)
    with pytest.raises(CueEditError):
        set_cue_message(EDIT_TEXT, 1, "/a")


def test_edits_on_first_and_last_line_keep_bom_crlf_and_trailing_whitespace():
    text = "﻿1:00 /first   \r\n# keep me  \r\n2:00 /last\t"
    out = set_cue_time(text, 1, 61)
    assert out == "﻿1:01.000 /first   \r\n# keep me  \r\n2:00 /last\t"
    out = set_cue_time(text, 3, 125)
    assert out == "﻿1:00 /first   \r\n# keep me  \r\n2:05.000 /last\t"
    out = set_cue_message(text, 3, "/new 1")
    assert out == "﻿1:00 /first   \r\n# keep me  \r\n2:00 /new 1"
    out = delete_cue(text, 1)
    assert out == "﻿# keep me  \r\n2:00 /last\t"
    out = delete_cue(text, 3)
    assert out == "﻿1:00 /first   \r\n# keep me  \r\n"
    assert parse_cues(out).cues[0].id == 1


def test_delete_cue():
    out = delete_cue(EDIT_TEXT, 4)
    assert out == EDIT_TEXT.replace("0:12.480     lights      /cue/go 2\n", "")
    with pytest.raises(CueEditError):
        delete_cue(EDIT_TEXT, 1)
    with pytest.raises(CueEditError):
        delete_cue(EDIT_TEXT, 6)
    assert delete_cue("1 /a", 1) == ""


def test_add_cue_into_empty_file():
    assert add_cue("", 5, "/a 1") == ("0:05.000 /a 1\n", 1)
    assert add_cue("﻿", 5, "/a") == ("﻿0:05.000 /a\n", 1)
    assert add_cue("# header\n", 5, "/a") == ("# header\n0:05.000 /a\n", 2)
    assert add_cue("# header", 5, "/a") == ("# header\n0:05.000 /a", 2)
    assert add_cue("# a\r\n@target v h:1\r\n", 5, "v /a") == ("# a\r\n@target v h:1\r\n0:05.000 v /a\r\n", 3)


def test_add_cue_positions_and_style():
    text, n = add_cue(EDIT_TEXT, 1.0, "/before")                    # before the first cue line
    assert n == 3 and text.splitlines()[2] == "0:01.000     /before"
    assert text.splitlines()[3:] == EDIT_TEXT.splitlines()[2:]
    text, n = add_cue(EDIT_TEXT, 8.0, "lights /between # n")        # between cues
    assert n == 4 and text.splitlines()[3] == "0:08.000     lights /between # n"
    text, n = add_cue(EDIT_TEXT, 100.0, "/after")                   # after the last cue (style of 1:05.2)
    assert n == 6 and text.endswith('1:05.2       /clip/play "intro.mp4"\n1:40.000     /after\n')
    text, n = add_cue(EDIT_TEXT, 12.48, "/same_time")              # equal time: after the existing cue
    assert n == 5 and text.splitlines()[4].startswith("0:12.480     /same_time")
    sheet = parse_cues(text)
    assert sheet.issues == [] and sheet.by_id(5).address == "/same_time"
    assert [c.address for c in sheet.cues][:3] == ["/cue/go", "/cue/go", "/same_time"]


def test_add_cue_uses_neighbour_style_and_line_endings():
    assert add_cue("[01:00.00] /a\n", 90, "/b") == ("[01:00.00] /a\n[01:30.000] /b\n", 2)
    assert add_cue("60.5 /a\r\n", 90, "/b") == ("60.5 /a\r\n90.000 /b\r\n", 2)
    assert add_cue("60.5 /a", 90, "/b") == ("60.5 /a\n90.000 /b", 2)
    assert add_cue("  1:00:00\t/a\n", 1, "/b") == ("  0:00:01.000\t/b\n  1:00:00\t/a\n", 1)
    assert add_cue("1:00,5 /a\n", "2:00", "/b") == ("1:00,5 /a\n2:00,000 /b\n", 2)


def test_add_cue_unsorted_file_goes_next_to_its_time_neighbour():
    text, n = add_cue("10 /a\n50 /b\n5 /c\n", 20, "/new")
    assert n == 2 and text == "10 /a\n20.000 /new\n50 /b\n5 /c\n"
    text, n = add_cue("10 /a\n50 /b\n5 /c\n", 1, "/new")
    assert n == 3 and text == "10 /a\n50 /b\n1.000 /new\n5 /c\n"


def test_add_cue_ignores_error_lines_and_validates():
    text, n = add_cue("1:99 /bad\n0:10 /ok\n", 20, "/new")
    assert n == 3 and text == "1:99 /bad\n0:10 /ok\n0:20.000 /new\n"
    for t, msg in [(-1, "/a"), (math.nan, "/a"), ("x", "/a"), (1, ""), (1, "/a\n2 /b"), (1, "lights")]:
        with pytest.raises(CueEditError):
            add_cue(EDIT_TEXT, t, msg)


def test_edit_round_trip_ids_shift():
    text, n = add_cue(EDIT_TEXT, 8.0, "/x")
    sheet = parse_cues(text)
    assert sheet.by_id(n).address == "/x" and sheet.by_id(5).time == 12.48     # later ids shifted by one
    assert delete_cue(text, n) == EDIT_TEXT


def test_edit_errors_are_value_errors():
    assert issubclass(CueEditError, ValueError) and issubclass(CueSyntaxError, ValueError)


# ----------------------------------------------------------------------------------------- CLI


def test_cli_ok(tmp_path, capsys):
    p = tmp_path / "cues.txt"
    p.write_text(SPEC_EXAMPLE, encoding="utf-8")
    assert cues.cli([str(p)]) == 0
    out = capsys.readouterr().out
    assert "5 cues, 2 declared targets, 0 errors, 0 warnings" in out
    lines = out.splitlines()
    rows = [ln for ln in lines if "line " in ln and "/" in ln and ":" in ln.split()[0]]
    assert [r.split()[0] for r in rows] == ["0:00.000", "0:12.480", "1:05.200", "1:15.300", "1:20.000"]
    assert any("lights" in r and "/cue/go 12" in r and "# verse 1 wash" in r for r in rows)
    assert any("(default)" in r and "/show/start" in r for r in rows)
    assert "video   127.0.0.1:7000  latency 80 ms" in out


def test_cli_errors_and_json(tmp_path, capsys):
    p = tmp_path / "cues.txt"
    p.write_bytes(("﻿@target v [::1]:9000\r\n2:00 /b # late\r\n1:99 /bad\r\n1:00 v /a 'x y'\r\n"
                   "1:00 v /a 'x y'\r\n").encode())
    assert cues.cli([str(p)]) == 1
    out = capsys.readouterr().out
    assert "line 3: error:" in out and "line 5: warning: duplicate" in out and "[::1]:9000" in out
    assert out.index("1:00.000") < out.index("2:00.000")
    assert cues.cli([str(p), "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["errors"] == 1 and data["warnings"] == 1 and "text" not in data
    assert [c["id"] for c in data["cues"]] == [4, 5, 2]
    assert data["cues"][0]["args"] == [["s", "x y"]] and data["targets"][0]["host"] == "::1"


def test_cli_unreadable(tmp_path, capsys):
    assert cues.cli([str(tmp_path / "missing.txt")]) == 1
    p = tmp_path / "latin1.txt"
    p.write_bytes(b"1:00 /a \xe9\n")
    assert cues.cli([str(p)]) == 1
    assert "not UTF-8" in capsys.readouterr().err


def test_cli_dispatch(tmp_path, capsys):
    from lyricprompter_oscsender.cli import main

    p = tmp_path / "c.txt"
    p.write_text("1:00 /a\n", encoding="utf-8")
    assert main(["check", str(p)]) == 0
    assert "1 cue," in capsys.readouterr().out
