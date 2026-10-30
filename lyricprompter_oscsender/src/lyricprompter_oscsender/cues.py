"""Cue files: OSC messages at timestamps on the reference recording's time axis.

UTF-8 text (a leading BOM is ignored), one cue per line, `\\n`, `\\r\\n` or `\\r` line endings:

    # comment                                   (blank lines are ignored too)
    @target lights 192.168.0.50:8000            declares a target (IPv6: [::1]:9000)
    @target video  127.0.0.1:7000  latency=80ms
    0:12.480     lights      /cue/go 12         # verse 1 wash
    75.300                   /mixer/fader f:0.75 i:3 true
    [01:20.00]               /scene "two words" nil

A cue line is `<time> [<target>] <address> [<arg> ...] [# note]`. Times are seconds (`75.3`), `M:SS.fff`
or `H:MM:SS.fff`, optionally in LRC brackets, with `.` or `,` as the decimal separator. Each argument token
is one OSC argument: quoted strings, true / false / nil, int32 (int64 when it does not fit), float32, a
typed `i: h: f: d: s:` prefix, or any other bare word as a string. An unquoted `#` that starts a token
begins the note. The full rules are in docs/SPEC.md, section 2.

Parsing never stops at a bad line: the line is skipped and reported as a `CueIssue`. The editing
functions take and return the whole text and rewrite only the characters they change, so a
hand-formatted file keeps its layout, comments and line endings. The tokenizer records each token's
span in its line for that (shlex cannot tell a quoted token from a bare one, nor say where it was).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import operator
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from .osc import (
    FLOAT32_MAX,
    INT32_MAX,
    INT32_MIN,
    INT64_MAX,
    INT64_MIN,
    OscArg,
    OscError,
    encode_message,
    validate_address,
)

TARGET_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")       # fullmatch
TIME_STYLES = ("s", "ms", "hms")
MAX_DECL_LATENCY_MS = 5000.0      # the console's limit for a target latency (settings.validate_settings)
MAX_TIME_S = 100 * 3600.0         # the latest cue time (100:00:00.000): a later time is an error
BOM = "﻿"

_INT_RE = re.compile(r"[+-]?[0-9]+")
_FLOAT_RE = re.compile(r"[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)(?:[eE][+-]?[0-9]+)?")
_TIME_FIELD_RE = re.compile(r"[0-9]+")
_TIME_LAST_RE = re.compile(r"([0-9]*)(?:\.([0-9]*))?")
_LATENCY_RE = re.compile(r"([0-9]+(?:\.[0-9]*)?|\.[0-9]+)(ms)?", re.IGNORECASE)
_PORT_RE = re.compile(r"[0-9]{1,5}")
_PREFIX_TAGS = "ihfds"
_QUOTES = "\"'"
_KEYWORDS = {"true": ("T", True), "false": ("F", False), "nil": ("N", None), "null": ("N", None)}
_TIME_HELP = "write seconds (75.3), M:SS.fff (1:05.2) or H:MM:SS.fff"
_LINE_BREAK_RE = re.compile(r"(\r\n|\r|\n)")
# line breaks for str.splitlines (and some editors) that the cue format reads as spaces: reported
_OTHER_BREAKS_RE = re.compile("[\v\f\x1c-\x1e\x85\u2028\u2029]")


class CueSyntaxError(ValueError):
    """A time, message or line that does not follow the cue file format."""


class CueEditError(ValueError):
    """An edit that cannot be applied (the message is user-readable)."""


# ----------------------------------------------------------------------------------------- data types


@dataclass(frozen=True)
class Cue:
    id: int                          # 1-based line number in the cue file (valid for one revision of the text)
    time: float                      # reference time in seconds (>= 0)
    address: str                     # OSC address (pattern)
    args: tuple[OscArg, ...] = ()    # ((tag, value), ...)
    target: str | None = None        # target name; None = the default target
    message: str = ""                # address + arguments as written in the file
    note: str = ""                   # trailing "# comment" text

    def key(self) -> tuple:
        """Identity across re-parses of an edited file (the line number may change, the cue not)."""
        return (round(self.time, 4), self.target, self.address, self.args)

    def body(self) -> str:
        """The cue line after its time, as `set_cue_message` takes it: `[target] message [# note]`."""
        text = f"{self.target} {self.message}" if self.target else self.message
        return f"{text} # {self.note}" if self.note else text

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "time": self.time, "target": self.target, "address": self.address,
                "args": [[tag, value] for tag, value in self.args], "message": self.message, "note": self.note}


@dataclass(frozen=True)
class TargetDecl:
    name: str
    host: str
    port: int
    latency_ms: float | None = None
    line: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "host": self.host, "port": self.port, "latency_ms": self.latency_ms,
                "line": self.line}


@dataclass(frozen=True)
class CueIssue:
    line: int                        # 1-based line number (0: the whole file)
    message: str
    severity: str = "error"          # "error" (the line is ignored) | "warning"

    def to_dict(self) -> dict[str, Any]:
        return {"line": self.line, "severity": self.severity, "message": self.message}


@dataclass
class CueSheet:
    text: str
    cues: list[Cue] = field(default_factory=list)              # sorted by (time, id)
    targets: list[TargetDecl] = field(default_factory=list)
    issues: list[CueIssue] = field(default_factory=list)

    def by_id(self, cue_id: int) -> Cue | None:
        return next((c for c in self.cues if c.id == cue_id), None)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        """The parse result as JSON-able data (the server adds its own warnings to "issues")."""
        return {"rev": sheet_rev(self.text), "text": self.text, "cues": [c.to_dict() for c in self.cues],
                "targets": [t.to_dict() for t in self.targets], "issues": [i.to_dict() for i in self.issues]}


def sheet_rev(text: str) -> str:
    """Revision id of a cue text: clients send it back with edits to detect concurrent changes."""
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()[:12]


# ----------------------------------------------------------------------------------------- times


class TimeStyle(NamedTuple):
    """How a time token is written; `format_time(t, **style._asdict())` writes a time the same way."""
    style: str = "ms"                # "s" (75.300) | "ms" (1:05.200) | "hms" (1:01:05.200)
    pad: bool = False                # first field zero-padded to two digits (01:05.200)
    brackets: bool = False           # LRC style [01:05.200]
    comma: bool = False              # ',' as the decimal separator (1:05,200)


def _time_fields(token: str) -> tuple[list[str], str, str, TimeStyle]:
    """Validated fields of a time token: (leading int fields, last int part, fraction digits, style)."""
    tok = str(token).strip()
    brackets = tok.startswith("[")
    if brackets != tok.endswith("]"):
        raise CueSyntaxError(f"{tok!r} has unbalanced brackets: write [1:05.200]")
    body = tok[1:-1] if brackets else tok
    if body.startswith("-"):
        raise CueSyntaxError(f"negative time {tok!r}: times start at 0")
    comma = "," in body
    if comma:
        if "." in body or body.count(",") > 1:
            raise CueSyntaxError(f"{tok!r} is not a time: {_TIME_HELP}")
        body = body.replace(",", ".")
    parts = body.split(":")
    if len(parts) == 4:
        raise CueSyntaxError(f"{tok!r}: frames are not supported: write HH:MM:SS.fff")
    last = _TIME_LAST_RE.fullmatch(parts[-1])
    if (len(parts) > 4 or last is None or not (last.group(1) or last.group(2))
            or not all(_TIME_FIELD_RE.fullmatch(p) for p in parts[:-1])):
        raise CueSyntaxError(f"{tok!r} is not a time: {_TIME_HELP}")
    ints, frac = parts[:-1] + [last.group(1) or "0"], last.group(2) or ""
    if any(len(p.lstrip("0")) > 9 for p in ints):          # far beyond MAX_TIME_S (and int()'s digit limit)
        raise CueSyntaxError(_too_late(tok))
    names = ["seconds", "minutes"]
    for k, p in enumerate(ints[1:]):
        if int(p) >= 60:
            raise CueSyntaxError(f"{tok!r}: {names[len(ints) - 2 - k]} must be < 60")
    first = parts[0] if len(parts) > 1 else (last.group(1) or "")
    style = TimeStyle(TIME_STYLES[len(parts) - 1], len(first) >= 2 and first.startswith("0"), brackets, comma)
    return ints[:-1], ints[-1], frac, style


def parse_time(token: str) -> float:
    """Seconds of a time token (`75.3`, `1:05.2`, `01:05,200`, `1:01:05.2`, `[01:05.20]`); CueSyntaxError
    (a ValueError) when it is not a valid time or is later than MAX_TIME_S."""
    lead, sec, frac, _ = _time_fields(token)
    whole = 0
    for p in [*lead, sec]:
        whole = whole * 60 + int(p)
    # one decimal -> float conversion of the total: `1:05.2` and `65.2` give the same float
    t = float(f"{whole}.{frac or '0'}")
    if t > MAX_TIME_S:
        raise CueSyntaxError(_too_late(str(token).strip()))
    return t


def _too_late(tok: str) -> str:
    tok = tok if len(tok) <= 24 else f"{tok[:20]}..."
    return f"{tok!r} is beyond the latest time a cue can have ({format_time(MAX_TIME_S, 'hms')})"


def time_style(token: str) -> TimeStyle:
    """How a (valid) time token is written: style s / ms / hms, zero padding, brackets, decimal comma."""
    return _time_fields(token)[3]


def format_time(t: float, style: str = "ms", *, pad: bool = False, brackets: bool = False,
                comma: bool = False) -> str:
    """A time token for `t` seconds, rounded to the millisecond, always with 3 decimals:
    s -> `75.300`, ms -> `1:05.200` (minutes may exceed 59), hms -> `1:01:05.200`."""
    t = float(t)
    if not math.isfinite(t):
        raise ValueError(f"time must be finite, got {t}")
    ms = round(t * 1000)
    if ms < 0:
        raise ValueError(f"time must be >= 0, got {t}")
    sec, frac = divmod(ms, 1000)
    w = 2 if pad else 1
    if style == "s":
        body = f"{sec:0{w}d}.{frac:03d}"
    elif style == "ms":
        m, s = divmod(sec, 60)
        body = f"{m:0{w}d}:{s:02d}.{frac:03d}"
    elif style == "hms":
        h, rest = divmod(sec, 3600)
        m, s = divmod(rest, 60)
        body = f"{h:0{w}d}:{m:02d}:{s:02d}.{frac:03d}"
    else:
        raise ValueError(f"unknown time style {style!r} (one of {', '.join(TIME_STYLES)})")
    if comma:
        body = body.replace(".", ",")
    return f"[{body}]" if brackets else body


# ----------------------------------------------------------------------------------------- tokens


@dataclass(frozen=True)
class _Token:
    start: int                       # span of the token in its line
    end: int
    raw: str                         # source text
    value: str                       # quoted: the string with escapes resolved; bare: == raw
    quoted: bool = False
    prefix: str = ""                 # typed prefix in front of a quoted string (s:"a b")


def _tokenize(line: str) -> tuple[list[_Token], int | None]:
    """Tokens of one line (without its ending) and the position of the note's `#` (None: no note).

    Tokens are separated by whitespace. A token starting with a quote (or a typed prefix followed by a
    quote) runs to the matching quote; inside it a backslash takes the next character literally. A bare
    token runs to the next whitespace; quotes inside it are literal.
    """
    toks: list[_Token] = []
    i, n = 0, len(line)
    while True:
        while i < n and line[i].isspace():
            i += 1
        if i >= n:
            return toks, None
        if line[i] == "#":
            return toks, i
        start, prefix = i, ""
        if line[i] in _QUOTES:
            q = i
        elif i + 2 < n and line[i] in _PREFIX_TAGS and line[i + 1] == ":" and line[i + 2] in _QUOTES:
            prefix, q = line[i], i + 2
        else:
            while i < n and not line[i].isspace():
                i += 1
            toks.append(_Token(start, i, line[start:i], line[start:i]))
            continue
        quote, buf, i = line[q], [], q + 1
        while True:
            if i >= n:
                raise CueSyntaxError(f"unterminated string {line[start:]!r}: missing the closing {quote}")
            c = line[i]
            if c == "\\" and i + 1 < n:
                buf.append(line[i + 1])
                i += 2
            elif c == quote:
                i += 1
                break
            else:
                buf.append(c)
                i += 1
        if i < n and not line[i].isspace():
            raise CueSyntaxError(f"put a space after the closing quote of {line[start:i]!r}")
        toks.append(_Token(start, i, line[start:i], "".join(buf), True, prefix))


def _line_kind(content: str) -> str:
    s = content.lstrip()
    if not s:
        return "blank"
    return {"#": "comment", "@": "directive"}.get(s[0], "cue")


# ----------------------------------------------------------------------------------------- arguments


def _int_value(text: str, raw: str, lo: int, hi: int, tag: str) -> int:
    if not _INT_RE.fullmatch(text):
        raise CueSyntaxError(f"{raw!r}: {tag}: needs an integer")
    try:
        v = int(text)
    except ValueError:                                     # more digits than int() converts
        raise CueSyntaxError(f"integer {raw!r} is out of range") from None
    if not lo <= v <= hi:
        hint = f": write h:{text}" if tag == "i" and INT64_MIN <= v <= INT64_MAX else ""
        raise CueSyntaxError(f"{raw!r} does not fit type '{tag}' ({lo}..{hi}){hint}")
    return v


def _float_value(text: str, raw: str, tag: str) -> float:
    if not _FLOAT_RE.fullmatch(text):
        raise CueSyntaxError(f"{raw!r}: {tag}: needs a number")
    v = float(text)
    if not math.isfinite(v):
        raise CueSyntaxError(f"{raw!r} is out of range")
    if tag == "f" and abs(v) > FLOAT32_MAX:
        raise CueSyntaxError(f"{raw!r} is beyond the float32 range: write d:{text}")
    return v


def _typed(tag: str, text: str, raw: str) -> OscArg:
    if tag == "s":
        return ("s", text)
    if tag == "i":
        return ("i", _int_value(text, raw, INT32_MIN, INT32_MAX, "i"))
    if tag == "h":
        return ("h", _int_value(text, raw, INT64_MIN, INT64_MAX, "h"))
    return (tag, _float_value(text, raw, tag))


def _parse_arg(tok: _Token) -> OscArg:
    if tok.quoted:
        return _typed(tok.prefix, tok.value, tok.raw) if tok.prefix else ("s", tok.value)
    w = tok.raw
    if len(w) >= 2 and w[1] == ":" and w[0] in _PREFIX_TAGS:
        return _typed(w[0], w[2:], w)
    kw = _KEYWORDS.get(w.lower())
    if kw is not None:
        return kw
    if _INT_RE.fullmatch(w):
        try:
            v = int(w)
        except ValueError:
            raise CueSyntaxError(f"integer {w!r} is out of range") from None
        if INT32_MIN <= v <= INT32_MAX:
            return ("i", v)
        if INT64_MIN <= v <= INT64_MAX:
            return ("h", v)
        raise CueSyntaxError(f"integer {w!r} does not fit 64 bits")
    if _FLOAT_RE.fullmatch(w):
        return ("f", _float_value(w, w, "f"))
    return ("s", w)


def _bare_ok(s: str) -> bool:
    """True if the string `s` reads back as the same string when written without quotes."""
    return bool(s) and s[0] not in "\"'" and "#" not in s and not any(c.isspace() for c in s) \
        and s.lower() not in _KEYWORDS and not _FLOAT_RE.fullmatch(s) \
        and not (len(s) >= 2 and s[1] == ":" and s[0] in _PREFIX_TAGS)


def _float_text(v: float) -> str:
    r = repr(float(v))
    if "." not in r and "e" in r:                          # 1e+20 -> 1.0e+20
        mant, exp = r.split("e")
        r = f"{mant}.0e{exp}"
    return r


def format_arg(tag: str, value: Any) -> str:
    """One argument in canonical cue syntax (reads back as the same (tag, value) through `parse_message`).
    ValueError for values a cue file cannot hold (blobs, impulses, line breaks, NUL, out of range)."""
    if tag in ("T", "F", "N"):
        return {"T": "true", "F": "false", "N": "nil"}[tag]
    if tag in ("i", "h"):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"type '{tag}' needs an int, got {value!r}")
        lo, hi = (INT32_MIN, INT32_MAX) if tag == "i" else (INT64_MIN, INT64_MAX)
        if not lo <= value <= hi:
            raise ValueError(f"{value} does not fit type '{tag}'")
        return str(value) if tag == "i" else f"h:{value}"
    if tag in ("f", "d"):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"type '{tag}' needs a finite number, got {value!r}")
        if tag == "f" and abs(value) > FLOAT32_MAX:
            raise ValueError(f"{value} is beyond the float32 range")
        return _float_text(value) if tag == "f" else f"d:{_float_text(value)}"
    if tag == "s":
        if not isinstance(value, str):
            raise ValueError(f"type 's' needs a string, got {value!r}")
        if "\n" in value or "\r" in value or "\x00" in value:
            raise ValueError(f"a cue file cannot hold line breaks or NUL characters in a string: {value!r}")
        if _bare_ok(value):
            return value
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    raise ValueError(f"OSC type '{tag}' cannot be written in a cue file")


def format_args(args: tuple[OscArg, ...] | list[OscArg]) -> str:
    """Arguments in canonical cue syntax, separated by single spaces."""
    return " ".join(format_arg(tag, value) for tag, value in args)


# ----------------------------------------------------------------------------------------- lines


def _parse_body(text: str, toks: list[_Token]) -> tuple[str | None, str, tuple[OscArg, ...], str, int]:
    """(target, address, args, message, body start) of `[target] /address args` tokens."""
    if not toks:
        raise CueSyntaxError("missing OSC address (it starts with /)")
    target, i = None, 0
    first = toks[0]
    if first.quoted or not first.raw.startswith("/"):
        if first.quoted or not TARGET_NAME_RE.fullmatch(first.raw):
            raise CueSyntaxError(f"{first.raw!r} is neither a target name (letters, digits, _ . -) nor "
                                 "an OSC address (starts with /)")
        target, i = first.raw, 1
        if len(toks) == 1:
            raise CueSyntaxError(f"missing OSC address after target {target!r}")
    a = toks[i]
    if a.quoted or not a.raw.startswith("/"):
        raise CueSyntaxError(f"expected an OSC address (starting with /) after target {target!r}, "
                             f"got {a.raw!r}")
    try:
        address = validate_address(a.raw)
    except OscError as e:
        raise CueSyntaxError(str(e)) from None
    args = tuple(_parse_arg(t) for t in toks[i + 1:])
    try:
        encode_message(address, args)                      # strings with NUL, unencodable text
    except OscError as e:
        raise CueSyntaxError(f"cannot encode the message: {e}") from None
    return target, address, args, text[a.start:toks[-1].end], first.start


def parse_message(text: str) -> tuple[str | None, str, tuple[OscArg, ...], str]:
    """(target, address, args, note) of `[target] /address [args ...] [# note]` (the part of a cue line
    after its time). CueSyntaxError (a ValueError) when it is not valid."""
    if "\n" in text or "\r" in text:
        raise CueSyntaxError("a cue message is a single line")
    toks, note_at = _tokenize(text)
    target, address, args, _, _ = _parse_body(text, toks)
    return target, address, args, _note(text, note_at)


def _note(text: str, note_at: int | None) -> str:
    return "" if note_at is None else text[note_at + 1:].strip()


@dataclass(frozen=True)
class _CueLine:
    time: float
    time_tok: _Token
    style: TimeStyle
    body_start: int                  # where the target (or the address) starts in the line
    target: str | None
    address: str
    args: tuple[OscArg, ...]
    message: str
    note: str
    warnings: tuple[str, ...] = ()


def _parse_cue_line(content: str) -> _CueLine:
    toks, note_at = _tokenize(content)
    tt = toks[0]                                           # a cue line has a first token (see _line_kind)
    if tt.quoted:
        raise CueSyntaxError(f"{tt.raw!r} is not a time: {_TIME_HELP}")
    t = parse_time(tt.raw)
    if len(toks) == 1:
        raise CueSyntaxError("missing OSC address after the time")
    target, address, args, message, body_start = _parse_body(content, toks[1:])
    # a '#' inside a bare token (no target name or address can hold one): an argument where a note was meant?
    warnings = tuple(f"{k.raw!r} is sent as a string argument with a '#' in it: write a space before # to "
                     "start a note, or quote the string" for k in toks[1:]
                     if not k.quoted and "#" in k.raw and not k.raw.startswith("s:"))
    return _CueLine(t, tt, time_style(tt.raw), body_start, target, address, args, message,
                    _note(content, note_at), warnings)


def _parse_hostport(s: str) -> tuple[str, int]:
    if s.startswith("["):
        end = s.find("]")
        if end < 0 or s[end + 1:end + 2] != ":":
            raise CueSyntaxError(f"{s!r}: write an IPv6 host as [addr]:port, e.g. [::1]:9000")
        host, port = s[1:end], s[end + 2:]
    else:
        host, sep, port = s.rpartition(":")
        if not sep:
            raise CueSyntaxError(f"{s!r}: missing port: write <host>:<port>, e.g. 127.0.0.1:9000")
        if ":" in host:
            raise CueSyntaxError(f"{s!r}: write an IPv6 host in brackets, e.g. [::1]:9000")
    if not host:
        raise CueSyntaxError(f"{s!r}: missing host")
    if not _PORT_RE.fullmatch(port) or not 1 <= int(port) <= 65535:
        raise CueSyntaxError(f"{s!r}: the port must be a number 1..65535")
    return host, int(port)


def _parse_directive(content: str, lineno: int) -> tuple[TargetDecl | None, list[str]]:
    """(declaration or None, warnings) of an `@...` line; CueSyntaxError for a malformed @target."""
    toks, _ = _tokenize(content)
    word = toks[0].raw
    if word.lower() != "@target":
        return None, [f"unknown directive {word!r} (ignored); the only directive is @target"]
    usage = "write @target <name> <host>:<port> [latency=<ms>]"
    if len(toks) < 3:
        raise CueSyntaxError(f"incomplete @target: {usage}")
    name = toks[1]
    if name.quoted or not TARGET_NAME_RE.fullmatch(name.raw):
        raise CueSyntaxError(f"{name.raw!r} is not a valid target name (letters, digits, _ . -; starting "
                             "with a letter or _)")
    host, port = _parse_hostport(toks[2].raw)
    latency, warnings = None, []
    for t in toks[3:]:
        key, sep, val = t.raw.partition("=")
        if not sep or key.lower() != "latency" or t.quoted:
            warnings.append(f"unknown @target option {t.raw!r} (ignored); {usage}")
            continue
        m = _LATENCY_RE.fullmatch(val)
        if m is None:
            raise CueSyntaxError(f"{t.raw!r}: the latency is a number of milliseconds >= 0, e.g. latency=80ms")
        latency = float(m.group(1))
        if latency > MAX_DECL_LATENCY_MS:
            raise CueSyntaxError(f"{t.raw!r}: the latency must be at most {MAX_DECL_LATENCY_MS:g} ms")
    return TargetDecl(name.raw, host, port, latency, lineno), warnings


def _split_lines(text: str) -> tuple[str, list[list[str]]]:
    """(BOM, [[content, ending], ...]): splits at '\\r\\n', '\\n' and a lone '\\r' (old Mac files, stray CRs);
    `_join_lines` gives back the exact text."""
    bom = BOM if text.startswith(BOM) else ""
    parts = _LINE_BREAK_RE.split(text[len(bom):])        # content, ending, content, ..., content
    lines = [[content, ending] for content, ending in zip(parts[0::2], parts[1::2], strict=False)]
    if parts[-1]:
        lines.append([parts[-1], ""])
    return bom, lines


def _other_break(content: str) -> str | None:
    """The first character of a line that some editors show as a line break (`_OTHER_BREAKS_RE`) outside
    quoted strings (where it is just text); None when there is none."""
    if _OTHER_BREAKS_RE.search(content) is None:
        return None
    try:
        quoted = [(t.start, t.end) for t in _tokenize(content)[0] if t.quoted]
    except CueSyntaxError:
        quoted = []
    return next((m.group() for m in _OTHER_BREAKS_RE.finditer(content)
                 if not any(a <= m.start() < b for a, b in quoted)), None)


def _join_lines(bom: str, lines: list[list[str]]) -> str:
    return bom + "".join(content + ending for content, ending in lines)


def parse_cues(text: str) -> CueSheet:
    """Parse a cue file. Never raises for content: bad lines are skipped and reported in `issues`."""
    sheet = CueSheet(text)
    cues: list[Cue] = []
    seen: dict[tuple, int] = {}
    declared: dict[str, int] = {}
    for lineno, (content, _) in enumerate(_split_lines(text)[1], 1):
        kind = _line_kind(content)
        if kind == "blank":
            continue
        other = _other_break(content)
        if other is not None:
            sheet.issues.append(CueIssue(lineno, f"the line contains the character U+{ord(other):04X}, which some "
                                         "editors show as a line break: it is read as a space in this line",
                                         "warning"))
        if kind == "comment":
            continue
        try:
            if kind == "directive":
                decl, warnings = _parse_directive(content, lineno)
                sheet.issues.extend(CueIssue(lineno, w, "warning") for w in warnings)
                if decl is not None and decl.name in declared:
                    sheet.issues.append(CueIssue(lineno, f"target {decl.name!r} is already declared on line "
                                                 f"{declared[decl.name]} (the first declaration is used)",
                                                 "warning"))
                elif decl is not None:
                    declared[decl.name] = lineno
                    sheet.targets.append(decl)
                continue
            cl = _parse_cue_line(content)
        except ValueError as e:
            sheet.issues.append(CueIssue(lineno, str(e), "error"))
            continue
        sheet.issues.extend(CueIssue(lineno, w, "warning") for w in cl.warnings)
        cue = Cue(lineno, cl.time, cl.address, cl.args, cl.target, cl.message, cl.note)
        first = seen.setdefault(cue.key(), lineno)
        if first != lineno:
            sheet.issues.append(CueIssue(lineno, f"duplicate of the cue on line {first} (same time, target, "
                                         "address and arguments)", "warning"))
        cues.append(cue)
    sheet.cues = sorted(cues, key=lambda c: (c.time, c.id))
    return sheet


# ----------------------------------------------------------------------------------------- editing


def _edit_time(t: float | str) -> float:
    """A new cue time: seconds (number) or a time token (str). CueEditError when invalid, < 0 or later
    than MAX_TIME_S."""
    if isinstance(t, str):
        try:
            return parse_time(t)
        except ValueError as e:
            raise CueEditError(str(e)) from None
    if isinstance(t, bool):
        raise CueEditError(f"the time must be a number of seconds, got {t!r}")
    try:
        t = float(t)
    except (TypeError, ValueError):
        raise CueEditError(f"the time must be a number of seconds, got {t!r}") from None
    if not math.isfinite(t):
        raise CueEditError("the time must be a finite number of seconds")
    if t > MAX_TIME_S:
        raise CueEditError(f"the time must be at most {MAX_TIME_S:g} s, got {t:g}")
    if t * 1000 < -0.5:                                    # rounds to < 0 ms (no round(): no overflow)
        raise CueEditError(f"the time must be >= 0, got {t:g}")
    return max(t, 0.0)


def _edit_message(message: str) -> str:
    if not isinstance(message, str):
        raise CueEditError(f"the message must be text, got {message!r}")
    try:
        parse_message(message)
    except ValueError as e:
        raise CueEditError(str(e)) from None
    return message.strip()


def _cue_line_at(text: str, cue_id: int) -> tuple[str, list[list[str]], int, _CueLine]:
    """(BOM, lines, line index, parsed line) of the cue on line `cue_id`; CueEditError if there is none."""
    try:
        if isinstance(cue_id, bool):
            raise TypeError
        n = operator.index(cue_id)
    except TypeError:
        raise CueEditError(f"a cue id is a line number, got {cue_id!r}") from None
    bom, lines = _split_lines(text)
    if not 1 <= n <= len(lines) or _line_kind(lines[n - 1][0]) != "cue":
        raise CueEditError(f"there is no cue on line {n} (the file may have changed: reload it)")
    try:
        return bom, lines, n - 1, _parse_cue_line(lines[n - 1][0])
    except ValueError as e:
        raise CueEditError(f"line {n} is not a valid cue: {e}") from None


def _replace_time(content: str, tok: _Token, new: str) -> str:
    """Replace a time token; a run of 2+ spaces after it is resized so the next column stays aligned."""
    j = tok.end
    while j < len(content) and content[j] == " ":
        j += 1
    if j - tok.end > 1 and j < len(content) and not content[j].isspace():
        gap = max(1, (j - tok.end) - (len(new) - len(tok.raw)))
        return content[:tok.start] + new + " " * gap + content[j:]
    return content[:tok.start] + new + content[tok.end:]


def set_cue_time(text: str, cue_id: int, t: float | str) -> str:
    """Rewrite the time of the cue on line `cue_id`, written in the style of its current time token."""
    t = _edit_time(t)
    bom, lines, i, cl = _cue_line_at(text, cue_id)
    lines[i][0] = _replace_time(lines[i][0], cl.time_tok, format_time(t, **cl.style._asdict()))
    return _join_lines(bom, lines)


def shift_cue(text: str, cue_id: int, dt: float) -> str:
    """Move the cue on line `cue_id` by `dt` seconds (CueEditError if it would go before 0)."""
    _, _, _, cl = _cue_line_at(text, cue_id)
    try:
        dt = float(dt)
    except (TypeError, ValueError):
        raise CueEditError(f"the shift must be a number of seconds, got {dt!r}") from None
    if not math.isfinite(dt):
        raise CueEditError("the shift must be a finite number of seconds")
    t = cl.time + dt
    if t * 1000 < -0.5:
        raise CueEditError(f"moving the cue by {dt:+.3f} s would put it before 0:00.000")
    return set_cue_time(text, cue_id, max(t, 0.0))


def set_cue_message(text: str, cue_id: int, message: str) -> str:
    """Replace everything after the time of the cue on line `cue_id` (target, message, note) with
    `message` (`[target] /address args [# note]`, validated). The time token and the spacing after it stay."""
    message = _edit_message(message)
    bom, lines, i, cl = _cue_line_at(text, cue_id)
    lines[i][0] = lines[i][0][:cl.body_start] + message
    return _join_lines(bom, lines)


def add_cue(text: str, t: float | str, message: str) -> tuple[str, int]:
    """Insert the cue line `<time> <message>`; returns (new text, its line number).

    It goes after the cue that precedes it in time (the last cue, in (time, line) order, whose time is
    <= t), before the earliest cue when none does, or at the end of a file without cues. The time is
    written in the style of that neighbouring cue line (else `M:SS.fff`), with its indentation and, when
    its columns are aligned with spaces, the message aligned the same way.
    """
    t = _edit_time(t)
    message = _edit_message(message)
    sheet = parse_cues(text)
    bom, lines = _split_lines(text)
    t_ms = round(t * 1000) / 1000
    before = [c for c in sheet.cues if c.time <= t_ms + 1e-9]
    if before:
        anchor, idx = before[-1], before[-1].id
    elif sheet.cues:
        anchor, idx = sheet.cues[0], sheet.cues[0].id - 1
    else:
        anchor, idx = None, len(lines)
    if anchor is None:
        new = f"{format_time(t)} {message}"
    else:
        content = lines[anchor.id - 1][0]
        cl = _parse_cue_line(content)
        indent, tok = content[:cl.time_tok.start], format_time(t, **cl.style._asdict())
        gap = content[cl.time_tok.end:cl.body_start]
        if len(gap) > 1 and set(gap) == {" "}:
            gap = " " * max(1, cl.body_start - len(indent) - len(tok))
        elif "\t" not in gap:
            gap = " "
        new = f"{indent}{tok}{gap}{message}"
    nl = next((ending for _, ending in lines if ending), "\n")
    if idx < len(lines):
        lines.insert(idx, [new, nl])
    elif lines and not lines[-1][1]:                       # the file does not end with a line break
        lines[-1][1] = nl
        lines.append([new, ""])
    else:
        lines.append([new, nl])
    return _join_lines(bom, lines), idx + 1


def delete_cue(text: str, cue_id: int) -> str:
    """Remove the cue line `cue_id` (with its line ending)."""
    bom, lines, i, _ = _cue_line_at(text, cue_id)
    del lines[i]
    return _join_lines(bom, lines)


# ----------------------------------------------------------------------------------------- CLI


def _hostport(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def format_sheet(sheet: CueSheet, name: str = "cues") -> str:
    """Human-readable report: summary, declared targets, cues sorted by time, issues."""
    def plural(n: int, word: str) -> str:
        return f"{n} {word}{'' if n == 1 else 's'}"

    summary = (f"{name}: {plural(len(sheet.cues), 'cue')}, {plural(len(sheet.targets), 'declared target')}, "
               f"{plural(sheet.n_errors, 'error')}, {plural(sheet.n_warnings, 'warning')}")
    out = [summary]
    if sheet.targets:
        out += ["", "targets:"]
        w = max(len(t.name) for t in sheet.targets)
        for t in sheet.targets:
            lat = "" if t.latency_ms is None else f"  latency {t.latency_ms:g} ms"
            out.append(f"  {t.name:<{w}}  {_hostport(t.host, t.port)}{lat}   (line {t.line})")
    if sheet.cues:
        out += ["", "cues:"]
        w = max(len(c.target or "(default)") for c in sheet.cues)
        for c in sheet.cues:
            ts = format_time(c.time, "hms" if c.time >= 3600 else "ms")
            note = f"   # {c.note}" if c.note else ""
            out.append(f"  {ts:>11}  line {c.id:<4} {c.target or '(default)':<{w}}  {c.message}{note}")
    if sheet.issues:
        out += ["", "issues:"]
        out += [f"  line {i.line}: {i.severity}: {i.message}" for i in sheet.issues]
    return "\n".join(out)


def cli(argv: list[str]) -> int:
    """`oscsender check FILE [--json]`: exit status 1 when the file has errors (or cannot be read)."""
    ap = argparse.ArgumentParser(prog="oscsender check",
                                 description="Validate a cue file and print its cues sorted by time.")
    ap.add_argument("file", type=Path, help="cue file (UTF-8 text)")
    ap.add_argument("--json", action="store_true", help="print the parse result as JSON")
    a = ap.parse_args(argv)
    try:
        text = a.file.read_bytes().decode("utf-8")
    except OSError as e:
        print(f"cannot read {a.file}: {e}", file=sys.stderr)
        return 1
    except UnicodeDecodeError as e:
        print(f"{a.file} is not UTF-8 text: {e}", file=sys.stderr)
        return 1
    sheet = parse_cues(text)
    if a.json:
        data = sheet.to_dict()
        del data["text"]
        data = {"file": str(a.file), **data, "errors": sheet.n_errors, "warnings": sheet.n_warnings}
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(format_sheet(sheet, str(a.file)))
    return 1 if sheet.n_errors else 0
