"""FastAPI server: song library + prep jobs, live session control, timing review, and one 20 Hz
WebSocket broadcaster feeding the operator console (/) and the stage prompter (/prompter).

Run a single process (`lyricprompter serve`): the live session and its audio threads live in it.
There is no authentication; bind to 127.0.0.1 unless the stage network is trusted.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import json
import logging
import math
import os
import re
import shutil
import socket
import time
import unicodedata
import uuid
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AfterValidator, BaseModel, Field, ValidationError

from ..config import PROJECT_ROOT, SONGS_DIR
from ..live.session import file_signature, json_safe
from ..songfile import STEM_FILES, list_songs, load_song, new_song_id, save_song, stem_path
from ..text.model import STYLES, Line, Song, line_from_dict
from .jobs import JobManager, build_prep_argv

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
SONG_ID_RE = re.compile(r"\w[\w\-.]*")      # fullmatch (a `$` would let "x\n" through)
LANGS = ("auto", "ko", "ja", "en")
REFINE = ("auto", "none", "qwen")
UPLOAD_DIRNAME = ".uploads"
DISPLAY_FILE = ".display.json"   # prompter display settings, kept across server restarts
REHEARSAL_PREROLL = 2.0  # s of the song's own stem played before the start line in a rehearsal
WS_SEND_TIMEOUT = 2.0    # a WebSocket that accepts nothing for this long is disconnected


# --------------------------------------------------------------------------- request models


def nfc(text: str) -> str:
    """Song ids are NFC: folders copied through Finder or unzipped from macOS archives are often
    named in NFD, and APFS opens either form."""
    return unicodedata.normalize("NFC", text)


SongId = Annotated[str, AfterValidator(nfc)]
# Every float the API takes is finite: NaN / Infinity (accepted by Python's JSON parser) would
# reach the tracker or end up in the state messages, which browsers then cannot parse.
Finite = Annotated[float, Field(allow_inf_nan=False)]


class InputSpec(BaseModel):
    kind: Literal["device", "file"] = "device"
    device: int | str | None = None              # index or name substring (None: default input)
    channels: list[int] | None = None            # 1-based
    samplerate: float | None = Field(None, gt=0.0, allow_inf_nan=False)
    path: str | None = None                      # file: audio path on the server
    song_id: SongId | None = None                # file: play a stem of this song (default: the session song)
    stem: str | None = None                      # file: "mix" | "vocals" | ...
    monitor: bool = False
    speed: float = Field(1.0, gt=0.1, le=4.0, allow_inf_nan=False)
    start_s: float = Field(0.0, ge=0.0, allow_inf_nan=False)


class SessionStart(BaseModel):
    song_id: SongId
    mode: Literal["mix", "vocal"] = "mix"
    input: InputSpec = Field(default_factory=InputSpec)
    lead: bool = False
    start_line: int | None = None
    latency: float | None = Field(None, ge=-2.0, le=2.0, allow_inf_nan=False)   # as LatencyReq
    transpose: int = Field(0, ge=-12, le=12)
    auto_key: bool = True                        # automatic key detection (off when a key is set)
    style: Literal["classic", "pop"] | None = None   # tracking style (None: the song's SongMeta.style)


class StyleReq(BaseModel):
    style: Literal["classic", "pop"]


class HoldReq(BaseModel):
    on: bool = True


class JumpReq(BaseModel):
    line: int | None = None
    time: Finite | None = None
    # the song the line / time belongs to (the console sends it): refused with 409 when another
    # song is live or cued, so a line picked in one song never moves the prompter of another
    song_id: SongId | None = None


class NudgeReq(BaseModel):
    dt: float = Field(..., ge=-30.0, le=30.0, allow_inf_nan=False)


class TransposeReq(BaseModel):
    semitones: int = Field(..., ge=-12, le=12)


class LatencyReq(BaseModel):
    seconds: float = Field(..., ge=-2.0, le=2.0, allow_inf_nan=False)


class CueReq(BaseModel):
    song_id: SongId | None


class LineEdit(BaseModel):
    """Exactly one of: a new `start`, a `shift` of the start, or a new `end` (seconds)."""
    shift: Finite | None = None
    start: Finite | None = None
    end: Finite | None = None


class DisplayPatch(BaseModel):
    lead_time: float | None = Field(None, ge=0.0, le=10.0, allow_inf_nan=False)
    font_scale: float | None = Field(None, ge=0.3, le=3.0, allow_inf_nan=False)
    mirror: bool | None = None
    preview_lines: int | None = Field(None, ge=0, le=3)
    layout: Literal["line", "page"] | None = None
    page_lines: int | None = Field(None, ge=1, le=8)
    wipe: bool | None = None


@dataclass
class DisplaySettings:
    """Prompter settings set from the console and pushed to every prompter screen. A .display.json
    saved before a field existed loads with that field's default."""
    lead_time: float = 1.0
    font_scale: float = 1.0
    mirror: bool = False
    preview_lines: int = 2       # line layout: lines after the focus line; page layout: of the next page
    layout: str = "line"         # "line": one line at a time | "page": a stanza page of up to page_lines
    page_lines: int = 4          # max lines per page (a longer stanza is split into balanced pages)
    wipe: bool = True            # per-syllable karaoke wipe on the focus line


class RevalidatingStatic(StaticFiles):
    """Static files that browsers must revalidate (ETag -> 304), so an updated UI is never stale."""

    async def get_response(self, path: str, scope: Any) -> Any:
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


# --------------------------------------------------------------------------- request origin guard


SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_DEFAULT_PORTS = {"http": "80", "https": "443", "ws": "80", "wss": "443"}


def _split_host(host: str) -> tuple[str, str]:
    """'Host' header -> (lowercase name without brackets / trailing dot, port or '')."""
    host = host.strip().lower()
    if host.startswith("["):
        name, _, rest = host[1:].partition("]")
        return name, rest[1:] if rest.startswith(":") else ""
    if host.count(":") == 1:
        name, port = host.split(":")
        return name.rstrip("."), port
    return host.rstrip("."), ""


def machine_host_names() -> set[str]:
    name = socket.gethostname().lower().rstrip(".")
    base = name.removesuffix(".local")
    return {name, base, f"{base}.local"} - {""}


def _origin_netloc(url: str) -> tuple[str, str] | None:
    """(host, port) of an Origin / Referer URL with the scheme's default port filled in."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if not parts.hostname or parts.scheme not in _DEFAULT_PORTS:
        return None
    return parts.hostname.lower().rstrip("."), str(port) if port else _DEFAULT_PORTS[parts.scheme]


class OriginGuard:
    """There is no authentication, so keep other web pages away from the controls (ASGI middleware).

    * DNS rebinding: the Host header must be a loopback name, an IP literal, this machine's name
      (or `<name>.local`) or one of `allowed_hosts`; anything else gets 421.
    * Cross-site requests: a state-changing request (any method but GET/HEAD/OPTIONS) and every
      WebSocket handshake that carries an Origin (or, lacking one, a Referer) must come from this
      server's own origin; otherwise 403. Browsers always send Origin on such requests, so a page
      elsewhere cannot stop the show with a "simple" no-cors POST. Clients without either header
      (curl, scripts) are not browsers and pass.
    """

    def __init__(self, app: Any, allowed_hosts: Iterable[str] = ()):
        self.app = app
        self.allowed = {_split_host(h)[0] for h in allowed_hosts if h.strip()} | machine_host_names()

    def host_ok(self, host: str) -> bool:
        name, _ = _split_host(host)
        if not name or name == "localhost" or name.endswith(".localhost") or name in self.allowed:
            return True
        try:
            ipaddress.ip_address(name)
        except ValueError:
            return False
        return True

    @staticmethod
    def same_origin(source: str, host: str, scheme: str) -> bool:
        got = _origin_netloc(source)
        name, port = _split_host(host)
        return got is not None and got == (name, port or _DEFAULT_PORTS.get(scheme, "80"))

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        kind = scope["type"]
        if kind not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", ())}
        host = headers.get("host", "")
        reason = None
        if not self.host_ok(host):
            status, reason = 421, f"unknown host {host!r}; start the server with --allow-host {_split_host(host)[0]}"
        elif kind == "websocket" or scope.get("method", "GET") not in SAFE_METHODS:
            source = headers.get("origin") or headers.get("referer")
            if source is not None and not self.same_origin(source, host, scope.get("scheme", "http")):
                status, reason = 403, f"cross-origin request from {source!r} refused"
        if reason is None:
            await self.app(scope, receive, send)
            return
        log.warning("refused %s %s: %s", kind, scope.get("path"), reason)
        if kind == "websocket":
            await send({"type": "websocket.close", "code": 1008,
                        "reason": reason.encode()[:120].decode("utf-8", "ignore")})
        else:
            await JSONResponse({"detail": reason}, status_code=status)(scope, receive, send)


# --------------------------------------------------------------------------- song helpers


def check_song_id(song_id: str) -> str:
    """The NFC form of a valid song id (one path component: word characters, combining marks
    such as a dakuten that has no precomposed form, '-' and '.'), else HTTP 400."""
    song_id = nfc(song_id)
    bare = "".join(c for c in song_id if not unicodedata.category(c).startswith("M"))
    if (not SONG_ID_RE.fullmatch(bare) or unicodedata.category(song_id[0]).startswith("M")
            or ".." in song_id):
        raise HTTPException(400, f"invalid song id {song_id!r}")
    return song_id


def find_song_dir(root: Path, song_id: str) -> Path:
    """`root / song_id`, or the package folder whose NFC name is `song_id` (a folder named in NFD
    on a file system that, unlike APFS, does not match the two forms)."""
    sdir = root / song_id
    if (sdir / "song.json").is_file():
        return sdir
    try:
        for d in root.iterdir():
            if d.name != song_id and nfc(d.name) == song_id and (d / "song.json").is_file():
                return d
    except OSError:
        pass
    return sdir


def stems_present(sdir: Path) -> list[str]:
    return [name for name in STEM_FILES if stem_path(sdir, name).is_file()]


def song_summary(sdir: Path, rev: int = 0) -> dict:
    sid = nfc(sdir.name)                 # the id is the package folder name, never meta.id
    try:
        song = load_song(sdir)
    except (OSError, ValueError, KeyError, TypeError) as e:
        return {"id": sid, "title": sid, "error": f"{type(e).__name__}: {e}"}
    lines = song.lines
    confs = [ln.confidence for ln in lines if ln.confidence is not None]
    return {"id": sid, "title": song.meta.title or sid, "artist": song.meta.artist,
            "default_lang": song.meta.default_lang, "duration": song.meta.duration, "style": song.meta.style,
            "n_lines": len(lines),
            "n_review": sum(1 for ln in lines if ln.needs_review),
            "aligned": bool(lines) and all(ln.start is not None for ln in lines if ln.text.strip()),
            "mean_confidence": round(sum(confs) / len(confs), 3) if confs else None,
            "stems": stems_present(sdir), "rev": rev}


# --------------------------------------------------------------------------- line timing edits
# An edit always succeeds: lines keep their order and the neighbours make room (see move_line).

MIN_LINE_DUR = 0.25      # s: a pushed line starts this long after the previous line's start
FIT_MARGIN = 0.05        # s: a line whose onsets all come this far before the next line only loses its tails
MIN_END_DUR = 0.1        # s: an `end` edit keeps the line at least this long
UNTIMED_SYL_DUR = 0.25   # s per syllable for a line given its first timing ...
UNTIMED_MIN_DUR = 1.0    # ... and at least this long (never running into the next timed line)
UNDO_DEPTH = 100         # timing edits that can be undone, per song (kept in memory)
_TOL = 5e-5              # stored times are rounded to 0.1 ms


def fmt_clock(t: float) -> str:
    """Seconds as the console shows them: 141.2 -> '2:21.20'."""
    m, cs = divmod(round(max(t, 0.0) * 100), 6000)
    return f"{m}:{cs / 100:05.2f}"


def line_list(idxs: list[int]) -> str:
    """Sorted 0-based line indices as the console numbers them: [31, 32, 34] -> 'lines 32–33, 35'."""
    parts, k = [], 0
    while k < len(idxs):
        j = k
        while j + 1 < len(idxs) and idxs[j + 1] == idxs[j] + 1:
            j += 1
        parts.append(f"{idxs[k] + 1}–{idxs[j] + 1}" if j > k else f"{idxs[k] + 1}")
        k = j + 1
    return f"{'lines' if len(idxs) > 1 else 'line'} {', '.join(parts)}"


def _map_times(line: Line, f: Callable[[float], float]) -> None:
    """Apply `f` to every time of `line` (line, syllables, units), rounded to 0.1 ms; None stays None."""
    def g(t: float | None) -> float | None:
        return None if t is None else round(f(t), 4)
    line.start, line.end = g(line.start), g(line.end)
    for s in line.syllables():
        s.start, s.end = g(s.start), g(s.end)
        for u in s.units:
            u.start, u.end = g(u.start), g(u.end)


def _shift_times(line: Line, delta: float) -> None:
    _map_times(line, lambda t: t + delta)


def _scale_times(line: Line, end: float, new_end: float) -> None:
    """Map `line`'s times linearly from [start, end] onto [start, new_end]: it keeps its rhythm."""
    s0 = line.start
    assert s0 is not None and end > s0
    k = (new_end - s0) / (end - s0)
    _map_times(line, lambda t: s0 + (t - s0) * k)


def _times(line: Line) -> list[float | None]:
    out = [line.start, line.end]
    for s in line.syllables():
        out += [s.start, s.end]
        for u in s.units:
            out += [u.start, u.end]
    return out


def _span_end(line: Line) -> float | None:
    """Latest time in the line: its end, or a syllable / unit end that runs past it."""
    ts = [t for t in _times(line) if t is not None]
    return max(ts) if ts else None


def _spread(line: Line, start: float, end: float) -> None:
    """Time a line as [start, end] with its syllables spread evenly (units take their syllable's span)."""
    syls = line.syllables()
    n = max(1, len(syls))
    line.start, line.end = round(start, 4), round(end, 4)
    for k, s in enumerate(syls):
        s.start, s.end = round(start + (end - start) * k / n, 4), round(start + (end - start) * (k + 1) / n, 4)
        for u in s.units:
            u.start, u.end = s.start, s.end


def _fit(line: Line, hi: float) -> None:
    """Keep `line` from running past `hi` (the next timed line's start). When all its onsets (line,
    syllable and unit starts) come at least FIT_MARGIN before `hi`, only the tails past it are
    clipped; otherwise the whole line is squeezed linearly into [start, hi], so no syllable collapses."""
    end = _span_end(line)
    if line.start is None or end is None or end <= hi + _TOL or hi <= line.start:
        return
    if max(t for t in _times(line)[::2] if t is not None) <= hi - FIT_MARGIN + _TOL:
        _map_times(line, lambda t: min(t, hi))
    else:
        _scale_times(line, end, hi)


def _tidy(line: Line) -> None:
    """Keep an edited line consistent: every time within [start, end], each start <= its end, and
    syllable / unit starts in sung order. A no-op on consistent timing (guards odd aligner output)."""
    lo = line.start
    if lo is None:
        return
    hi = line.end if line.end is not None else _span_end(line)
    hi = lo if hi is None else max(hi, lo)
    line.end = hi

    def clamp(t: float | None, floor: float) -> float | None:
        return None if t is None else min(max(t, floor), hi)
    syl_floor = unit_floor = lo
    for s in line.syllables():
        s.start = clamp(s.start, syl_floor)
        syl_floor = syl_floor if s.start is None else s.start
        s.end = clamp(s.end, syl_floor)
        for u in s.units:
            u.start = clamp(u.start, unit_floor)
            unit_floor = unit_floor if u.start is None else u.start
            u.end = clamp(u.end, unit_floor)


def _prev_timed(lines: list[Line], i: int) -> int | None:
    return next((j for j in range(i - 1, -1, -1) if lines[j].start is not None), None)


def _next_timed(lines: list[Line], i: int) -> int | None:
    return next((j for j in range(i + 1, len(lines)) if lines[j].start is not None), None)


def _plan_push(lines: list[Line], idx: int, anchor: float, step: int, gap: float) -> list[tuple[int, float]]:
    """The timed lines after (`step` 1) or before (-1) line `idx` that must move, as [(line, new
    start)]: the first one so it starts at least `gap` from `anchor`, each further one at least
    MIN_LINE_DUR from the line it follows (or precedes). Stops at the first line that has room."""
    out: list[tuple[int, float]] = []
    ref, j = anchor, idx + step
    while 0 <= j < len(lines):
        s = lines[j].start
        if s is not None:
            want = round(ref + step * gap, 4)
            if (s - want) * step >= -_TOL:
                break
            out.append((j, want))
            ref, gap = want, MIN_LINE_DUR
        j += step
    return out


@dataclass
class LineMove:
    shift: float                # how far the edited line's start moved (0.0 for an end edit or a first timing)
    touched: list[int]          # indices of every modified line ([] when the edit changed nothing)
    pushed: list[int]           # lines whose start moved to make room (now flagged `pushed`, to review)
    before: dict[int, dict]     # the touched lines as they were (asdict), for undo


def move_line(song: Song, idx: int, *, shift: float | None = None, start: float | None = None,
              end: float | None = None) -> LineMove:
    """Edit line `idx`'s timing with exactly one of `start`, `shift` (of the start) or `end`.

    * `start` / `shift`: the line moves rigidly with its syllables / units (it keeps its rhythm),
      anywhere in [0, song duration). A line without timing gets `start` and a provisional span
      (UNTIMED_SYL_DUR per syllable, at least UNTIMED_MIN_DUR) with its syllables spread evenly.
    * `end`: the line's times are scaled linearly from [start, end] onto [start, new end].

    The neighbours make room. Following timed lines that would start less than MIN_LINE_DUR after
    the previous line's (new) start, or, for an `end` edit, before the new end, are moved rigidly
    later, preceding ones earlier, cascading only as far as needed. Such a PUSHED line gets flag
    `pushed` and needs_review: its start is now a placeholder. Then every line whose start or whose
    next timed line's start changed is fitted before that next start (see _fit; the last line before
    the song's end), so a line never ends after the next timed line starts. The edited line becomes
    `manual_timing`, reviewed and no longer `pushed`; a line only trimmed or squeezed keeps its flags.

    Raises IndexError for an unknown line, ValueError (song unchanged) for a malformed edit or one
    that would push a line before 0 s or past the song's end.
    """
    lines = song.lines
    if not 0 <= idx < len(lines):
        raise IndexError(idx)
    given = [v for v in (start, shift, end) if v is not None]
    if len(given) != 1:
        raise ValueError("give exactly one of 'start', 'shift' or 'end'")
    if not math.isfinite(float(given[0])):
        raise ValueError("times must be finite")
    ln, name = lines[idx], f"Line {idx + 1}"
    dur = song.meta.duration
    song_end = float(dur) if dur and dur > 0 else None
    if ln.start is None and start is None:
        raise ValueError(f"{name} has no timing yet: set its start first")
    if end is not None:
        new_end = round(float(end), 4)
        what = f"end at {fmt_clock(new_end)}"
        if new_end < ln.start + MIN_END_DUR - _TOL:  # type: ignore[operator]
            raise ValueError(f"{name} must end at least {MIN_END_DUR:g} s after its start "
                             f"({fmt_clock(ln.start)})")  # type: ignore[arg-type]
        if song_end is not None and new_end > song_end + _TOL:
            raise ValueError(f"{name} cannot {what}: the song ends at {fmt_clock(song_end)}")
        plan = _plan_push(lines, idx, new_end, 1, 0.0)
    else:
        new_start = round(float(start) if start is not None else ln.start + float(shift), 4)  # type: ignore[arg-type,operator]
        what = f"start at {fmt_clock(new_start)}"
        if new_start < 0:
            raise ValueError(f"{name} cannot start before the beginning of the song (0:00.00)")
        if song_end is not None and new_start >= song_end - _TOL:
            raise ValueError(f"{name} cannot {what}: the song ends at {fmt_clock(song_end)}")
        earlier = _plan_push(lines, idx, new_start, -1, MIN_LINE_DUR)
        if any(s < 0 for _, s in earlier):
            raise ValueError(f"{name} cannot {what}: {line_list(sorted(j for j, _ in earlier))} before it "
                             f"would have to start before 0:00.00 (each line keeps at least {MIN_LINE_DUR:g} s)")
        plan = _plan_push(lines, idx, new_start, 1, MIN_LINE_DUR) + earlier
    past = sorted(j for j, s in plan if j > idx and song_end is not None and s >= song_end - _TOL)
    if past:
        raise ValueError(f"{name} cannot {what}: {line_list(past)} after it would be pushed past the end "
                         f"of the song ({fmt_clock(song_end)}; each line keeps at least {MIN_LINE_DUR:g} s)")  # type: ignore[arg-type]

    before = [asdict(x) for x in lines]
    for j, s in plan:
        _shift_times(lines[j], s - lines[j].start)  # type: ignore[operator]
        lines[j].needs_review = True
        if "pushed" not in lines[j].flags:
            lines[j].flags.append("pushed")
    delta = 0.0
    if end is not None:
        old_end = ln.end if ln.end is not None else _span_end(ln)
        if old_end is None or old_end - ln.start < _TOL:  # type: ignore[operator]
            _spread(ln, ln.start, new_end)  # type: ignore[arg-type]
        else:
            _scale_times(ln, old_end, new_end)
            ln.end = new_end
    elif ln.start is None:
        nxt = _next_timed(lines, idx)       # after the pushes: at least MIN_LINE_DUR after new_start
        hi = lines[nxt].start if nxt is not None else song_end
        span = max(UNTIMED_MIN_DUR, UNTIMED_SYL_DUR * len(ln.syllables()))
        _spread(ln, new_start, new_start + span if hi is None else min(new_start + span, hi))
    else:
        delta = new_start - ln.start
        _shift_times(ln, delta)
    ln.needs_review = False
    ln.flags = [f for f in ln.flags if f != "pushed"]
    if "manual_timing" not in ln.flags:
        ln.flags.append("manual_timing")

    moved = {j for j, _ in plan} | ({idx} if end is None else set())
    fit = moved | {idx} | {p for j in moved if (p := _prev_timed(lines, j)) is not None}
    for i in sorted(fit):
        nxt = _next_timed(lines, i)
        hi = lines[nxt].start if nxt is not None else song_end
        if hi is not None:
            _fit(lines[i], hi)
    touched = [i for i in sorted(fit) if asdict(lines[i]) != before[i]]
    for i in touched:
        _tidy(lines[i])
    return LineMove(round(delta, 4), touched, sorted(j for j, _ in plan), {i: before[i] for i in touched})


@dataclass
class UndoStep:
    """One timing edit of a song, as POST /api/songs/{id}/undo reverts it."""
    label: str                  # what was edited ("line 31 start"), for the console's notice
    before: dict[int, dict]     # the lines the edit changed, as they were
    after: int                  # lines_key() of the song right after the edit


def lines_key(song: Song) -> int:
    """Fingerprint of a song's lines (kept in memory only, so hash() is stable). An undo is refused
    once the lines on disk differ from the state right after the edit (e.g. after a re-prep)."""
    return hash(json.dumps([asdict(ln) for ln in song.lines], ensure_ascii=False, sort_keys=True))


def _line_at(lines: list[Line], t: float) -> int:
    idx = -1
    for i, ln in enumerate(lines):
        if ln.start is not None and ln.start <= t:
            idx = i
    return idx


# --------------------------------------------------------------------------- live session


SessionFactory = Callable[[Path, SessionStart, Path], Any]


def default_session_factory(sdir: Path, req: SessionStart, songs_dir: Path) -> Any:
    from ..live.audio_input import DeviceInput, FileInput
    from ..live.session import LiveSession

    inp = req.input
    same_axis = False      # the input is a stem of the session's own song (file time == song time)
    if inp.kind == "device":
        source: Any = DeviceInput(inp.device, inp.channels, inp.samplerate)
    else:
        if inp.stem:
            if inp.stem not in STEM_FILES:
                raise ValueError(f"unknown stem {inp.stem!r}")
            src_dir = find_song_dir(songs_dir, check_song_id(inp.song_id)) if inp.song_id else sdir
            same_axis = src_dir.resolve() == sdir.resolve()
            path = stem_path(src_dir, inp.stem)
        elif inp.path:
            path = Path(inp.path).expanduser()
            path = path if path.is_absolute() else PROJECT_ROOT / path
        else:
            raise ValueError("file input needs 'path' or 'stem'")
        source = FileInput(path, realtime=True, speed=inp.speed, monitor=inp.monitor, start_s=inp.start_s)
    sess = LiveSession(sdir, req.mode, source, prefer_lead=req.lead, start_line=req.start_line, style=req.style)
    if same_axis and req.start_line is not None and inp.start_s == 0:
        sess.align_source_start(REHEARSAL_PREROLL)      # rehearse from line N, not from 0:00
    if req.latency is not None:
        sess.set_latency(req.latency)
    if req.transpose or not req.auto_key:
        sess.set_transpose(req.transpose)          # an operator key turns auto detection off ...
        if req.auto_key:
            sess.set_auto_key(True)                # ... unless Auto stays on: the search starts from it
    return sess


def _default_devices(refresh: bool) -> list[dict]:
    from ..live.audio_input import list_input_devices

    return list_input_devices(refresh=refresh)


_WS_RANK = {"song": 0, "display": 1, "state": 2}


class WSClient:
    """One WebSocket: a latest-wins outbox (one pending message per type) drained by the
    connection's own task, so a slow or stalled screen never delays the broadcaster or the others."""

    def __init__(self, ws: Any, send_timeout: float = WS_SEND_TIMEOUT):
        self.ws, self.send_timeout = ws, send_timeout
        self._outbox: dict[str, str] = {}
        self._ready = asyncio.Event()
        self.sent = 0

    def offer(self, msgs: Iterable[tuple[str, str]]) -> None:
        for kind, text in msgs:
            self._outbox[kind] = text
        self._ready.set()

    async def pump(self) -> None:
        """Send until a send fails or stalls for `send_timeout` (then raise)."""
        while True:
            await self._ready.wait()
            self._ready.clear()
            for kind in sorted(self._outbox, key=lambda k: _WS_RANK.get(k, 1)):
                text = self._outbox.pop(kind, None)
                if text is not None:
                    await asyncio.wait_for(self.ws.send_text(text), self.send_timeout)
                    self.sent += 1


async def _until_disconnect(ws: Any) -> None:
    """Clients only listen; reading detects the disconnect."""
    while (await ws.receive())["type"] != "websocket.disconnect":
        pass


_nan_logged = False


def _dumps(msg: dict) -> str:
    """Strict JSON (browsers reject NaN / Infinity): a non-finite number becomes null rather than
    making the whole message, and with it every screen, unreadable."""
    global _nan_logged
    try:
        return json.dumps(msg, ensure_ascii=False, allow_nan=False)
    except ValueError:
        if not _nan_logged:
            _nan_logged = True
            log.warning("non-finite number in a %r message; sent as null", msg.get("type"))
        return json.dumps(json_safe(msg), ensure_ascii=False, allow_nan=False)


def _encode(msgs: Iterable[dict]) -> list[tuple[str, str]]:
    return [(m["type"], _dumps(m)) for m in msgs]


def load_display(path: Path) -> DisplaySettings:
    """Display settings saved by `save_display` (defaults when missing or invalid)."""
    display = DisplaySettings()
    try:
        patch = DisplayPatch.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        return display
    except (OSError, ValidationError) as e:
        log.warning("ignoring the saved display settings in %s: %s", path, e)
        return display
    for k, v in patch.model_dump(exclude_none=True).items():
        setattr(display, k, v)
    return display


def save_display(path: Path, display: DisplaySettings) -> None:
    try:
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(json.dumps(asdict(display)), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        log.warning("cannot save the display settings to %s: %s", path, e)


class Hub:
    """Server-wide state: the (single) live session, display settings and WebSocket clients."""

    def __init__(self, songs_dir: Path, jobs: JobManager, session_factory: SessionFactory,
                 devices: Callable[[bool], list[dict]], hz: float, send_timeout: float = WS_SEND_TIMEOUT):
        self.songs_dir, self.jobs = songs_dir, jobs
        self.session_factory, self.devices, self.hz = session_factory, devices, hz
        self.send_timeout = send_timeout
        self.session: Any = None
        self.session_req: dict | None = None
        self.cued_song: str | None = None
        self.manual_time: float | None = None
        self.display_path = songs_dir / DISPLAY_FILE
        self.display = load_display(self.display_path)
        self.display_rev = 0
        self.song_rev: dict[str, int] = {}
        self.clients: set[WSClient] = set()
        self.op_lock = asyncio.Lock()
        self._sent_song: tuple[str | None, int] | None = None
        self._sent_display = -1
        self._song_stat: tuple[str, tuple[int, int] | None] | None = None   # watched song.json
        self.broadcasts = 0
        self.edit_lock = asyncio.Lock()      # timing edits / undos: saves and session reloads land in order
        self.undo: dict[str, list[UndoStep]] = {}   # song id -> timing edits, oldest first

    def song_dir(self, song_id: str) -> Path:
        return find_song_dir(self.songs_dir, song_id)

    def record_edit(self, song_id: str, step: UndoStep, key_before: int) -> None:
        """Remember a saved timing edit for undo. Steps recorded before the song's lines changed on
        disk (`key_before` is not what the last step left, e.g. a re-prep) no longer apply: dropped."""
        hist = self.undo.setdefault(song_id, [])
        if hist and hist[-1].after != key_before:
            hist.clear()
        hist.append(step)
        del hist[:-UNDO_DEPTH]

    # -- messages ---------------------------------------------------------------------------------
    @property
    def song_id(self) -> str | None:
        return self.session.song_id if self.session is not None else self.cued_song

    def song_message(self) -> dict:
        sid = self.song_id
        return {"type": "song", "id": sid, "rev": self.song_rev.get(sid, 0) if sid else 0}

    def display_message(self) -> dict:
        return {"type": "display", **asdict(self.display), "rev": self.display_rev}

    def state_message(self) -> dict:
        if self.session is not None:
            try:
                return self.session.snapshot()
            except Exception as e:
                log.exception("snapshot failed")
                return {"type": "state", "song_id": self.song_id, "status": "error", "error": str(e),
                        "t": time.time()}
        t = self.manual_time
        line_idx = -1
        if t is not None and self.cued_song:
            try:
                line_idx = _line_at(load_song_cached(self, self.cued_song).lines, t)
            except (OSError, ValueError, KeyError, TypeError):
                line_idx = -1
        return {"type": "state", "song_id": self.cued_song, "mode": None,
                "status": "manual" if t is not None else "idle", "ref_time": t or 0.0, "tempo": 0.0,
                "confidence": 0.0, "live_time": 0.0, "level_db": None, "line_idx": line_idx, "syl_idx": -1,
                "t": round(time.time(), 4), "age": 0.0}

    def init_messages(self) -> list[tuple[str, str]]:
        return _encode((self.song_message(), self.display_message(), self.state_message()))

    def _changes(self) -> list[dict]:
        out = []
        song_key = (self.song_id, self.song_rev.get(self.song_id, 0) if self.song_id else 0)
        if song_key != self._sent_song:
            self._sent_song = song_key
            out.append(self.song_message())
        if self.display_rev != self._sent_display:
            self._sent_display = self.display_rev
            out.append(self.display_message())
        return out

    # -- song.json changed on disk ------------------------------------------------------------------
    def song_file_stat(self, song_id: str) -> tuple[int, int] | None:
        return file_signature(self.song_dir(song_id) / "song.json")

    def watch_from(self, song_id: str, stat: tuple[int, int] | None) -> None:
        """Watch `song_id`'s song.json for changes after `stat`, taken before the song was read for a
        start or a cue. (A baseline taken later, at the watcher's next tick, would absorb a re-prep
        landing in between: the session and the screens would keep the old lines under the same rev.)
        A song already watched keeps its baseline, so a change since then is still announced."""
        if self._song_stat is None or self._song_stat[0] != song_id:
            self._song_stat = (song_id, stat)

    def note_song_saved(self, song_id: str) -> None:
        """The server itself rewrote song.json (a line edit, which bumps the rev itself)."""
        if self._song_stat is not None and self._song_stat[0] == song_id:
            self._song_stat = (song_id, self.song_file_stat(song_id))

    async def watch_song_file(self) -> bool:
        """A song.json of the announced song replaced on disk (a CLI re-prep) bumps its rev, so every
        screen re-fetches it, and gives a running session of that song the new lines. True if so."""
        sid = self.song_id
        if not sid:
            self._song_stat = None
            return False
        cur = self.song_file_stat(sid)
        if self._song_stat is None or self._song_stat[0] != sid:
            self._song_stat = (sid, cur)
            return False
        if cur is None or cur == self._song_stat[1]:        # unchanged, or briefly missing
            return False
        self._song_stat = (sid, cur)
        self.song_rev[sid] = self.song_rev.get(sid, 0) + 1
        log.info("song.json of %r changed on disk: screens reload it (rev %d)", sid, self.song_rev[sid])
        sess = self.session
        if sess is not None and sess.song_id == sid:
            try:
                song = await run_in_threadpool(load_song, self.song_dir(sid))
                if self._song_stat == (sid, cur) and self.session is sess:   # no newer save meanwhile
                    await run_in_threadpool(sess.reload_lines, song)
            except Exception:
                log.exception("the running session keeps its lines: reloading %r failed", sid)
        return True

    async def broadcaster(self) -> None:
        loop = asyncio.get_running_loop()
        period = 1.0 / self.hz
        watch_every = max(1, round(self.hz / 2))          # stat the song.json twice a second
        next_t = loop.time()
        tick = 0
        while True:
            if tick % watch_every == 0:
                try:
                    await self.watch_song_file()
                except Exception:
                    log.exception("watching song.json failed")
            tick += 1
            changes = self._changes()
            if self.clients:
                msgs = _encode([*changes, self.state_message()])
                for client in list(self.clients):
                    client.offer(msgs)
                self.broadcasts += 1
            next_t += period
            delay = next_t - loop.time()
            if delay < 0:
                next_t, delay = loop.time(), 0.0
            await asyncio.sleep(delay)

    async def serve(self, ws: Any) -> None:
        """Run one accepted WebSocket until it disconnects or stalls (then close it, so the
        browser's reconnect logic takes over)."""
        client = WSClient(ws, self.send_timeout)
        # No await between building the init messages and joining: a song change the broadcaster
        # records from now on is offered to this client too.
        client.offer(self.init_messages())
        self.clients.add(client)
        sender = asyncio.create_task(client.pump(), name="ws-send")
        listener = asyncio.create_task(_until_disconnect(ws), name="ws-recv")
        try:
            await asyncio.wait({sender, listener}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            self.clients.discard(client)
            for task in (sender, listener):
                task.cancel()
            # asyncio.wait, not gather: when Starlette/anyio tears the endpoint down by cancelling
            # its scope while we are here, the CancelledError must stay anyio's own so the scope
            # absorbs it; gather would surface a child's CancelledError that escapes the scope.
            await asyncio.wait({sender, listener})
        for task in (sender, listener):
            if not task.cancelled():
                task.exception()            # retrieved, so asyncio does not log it as unhandled
        err = None if sender.cancelled() else sender.exception()
        if err is None:
            return
        if isinstance(err, TimeoutError):
            log.warning("WebSocket client stalled for %.1f s; disconnecting it", self.send_timeout)
        else:
            log.debug("WebSocket send failed: %r", err)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ws.close(code=1001), 0.5)

    # -- session ------------------------------------------------------------------------------------
    async def stop_session(self) -> None:
        """Drop the session first, then stop it: a stop that fails (e.g. PortAudio cannot stop the
        stream of a device lost mid-session) is logged; the session is gone either way."""
        sess, self.session, self.session_req = self.session, None, None
        if sess is not None:
            self.cued_song = sess.song_id
            await run_in_threadpool(_quiet_stop, sess)


_song_cache: dict[tuple[str, float], Song] = {}


def load_song_cached(hub: Hub, song_id: str) -> Song:
    path = hub.song_dir(song_id) / "song.json"
    key = (str(path), path.stat().st_mtime)
    song = _song_cache.get(key)
    if song is None:
        song = load_song(path.parent)
        _song_cache.clear()
        _song_cache[key] = song
    return song


def _start_error(e: Exception) -> HTTPException:
    if isinstance(e, HTTPException):
        return e
    if isinstance(e, FileNotFoundError):
        return HTTPException(404, f"file not found: {e}")
    if isinstance(e, (ValueError, TypeError, IndexError)):
        return HTTPException(400, str(e))
    if isinstance(e, ImportError):
        return HTTPException(503, f"tracking modules unavailable: {e}")
    log.error("session start failed", exc_info=e)
    return HTTPException(500, f"{type(e).__name__}: {e}")


def _quiet_stop(sess: Any) -> None:
    try:
        sess.stop()
    except Exception:
        log.exception("stopping a session failed")


# --------------------------------------------------------------------------- app


def create_app(songs_dir: Path | None = None, *, session_factory: SessionFactory | None = None,
               jobs: JobManager | None = None, devices: Callable[[bool], list[dict]] | None = None,
               broadcast_hz: float = 20.0, allowed_hosts: Iterable[str] = ()) -> FastAPI:
    """`allowed_hosts`: Host names accepted besides loopback, IP literals and this machine's name
    (see OriginGuard)."""
    # absolute: prep jobs run with cwd=PROJECT_ROOT, so a relative --songs-dir (or
    # LYRICPROMPTER_SONGS) would point them somewhere else than the server
    songs_root = Path(songs_dir or SONGS_DIR).expanduser().absolute()
    songs_root.mkdir(parents=True, exist_ok=True)
    job_manager = jobs or JobManager()
    hub = Hub(songs_root, job_manager, session_factory or default_session_factory,
              devices or _default_devices, broadcast_hz)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(hub.broadcaster(), name="ws-broadcaster")
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await hub.stop_session()
            job_manager.shutdown(timeout=2.0)

    app = FastAPI(title="lyricprompter", lifespan=lifespan)
    app.add_middleware(OriginGuard, allowed_hosts=tuple(allowed_hosts))
    app.state.hub = hub

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Any, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's own 422 echoes the offending input, and a NaN / Infinity there cannot be encoded
        return JSONResponse({"detail": json_safe(jsonable_encoder(exc.errors()))}, status_code=422)

    def sdir_of(song_id: str) -> Path:
        song_id = check_song_id(song_id)
        sdir = find_song_dir(songs_root, song_id)
        if not (sdir / "song.json").is_file():
            raise HTTPException(404, f"song {song_id!r} not found")
        return sdir

    # -- pages ----------------------------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def console_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "console.html", headers={"Cache-Control": "no-cache"})

    @app.get("/prompter", include_in_schema=False)
    async def prompter_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "prompter.html", headers={"Cache-Control": "no-cache"})

    # -- songs ----------------------------------------------------------------------------------------
    @app.get("/api/songs")
    async def songs_list() -> list[dict]:
        dirs = await run_in_threadpool(list_songs, songs_root)
        return [song_summary(d, hub.song_rev.get(nfc(d.name), 0)) for d in dirs]

    @app.get("/api/songs/{song_id}")
    async def song_detail(song_id: str) -> dict:
        song_id = check_song_id(song_id)
        sdir = sdir_of(song_id)
        try:
            song = load_song(sdir)
        except Exception as e:
            raise HTTPException(500, f"cannot read song.json: {e}") from e
        return {**song.to_dict(), "id": song_id, "stems": stems_present(sdir), "style": song.meta.style,
                "rev": hub.song_rev.get(song_id, 0)}

    @app.get("/api/songs/{song_id}/audio/{stem}")
    async def song_audio(song_id: str, stem: str) -> FileResponse:
        if stem not in STEM_FILES:
            raise HTTPException(404, f"unknown stem {stem!r}")
        path = stem_path(sdir_of(song_id), stem)
        if not path.is_file():
            raise HTTPException(404, f"song {song_id!r} has no {stem} stem")
        return FileResponse(path, media_type="audio/wav")

    @app.post("/api/songs", status_code=202)
    async def song_upload(audio: Annotated[UploadFile, File()], lyrics: Annotated[str, Form()],
                          title: Annotated[str, Form()], artist: Annotated[str, Form()] = "",
                          language: Annotated[str, Form()] = "auto", options: Annotated[str, Form()] = "{}",
                          song_id: Annotated[str | None, Form()] = None) -> dict:
        title, lyrics = title.strip(), lyrics.strip("﻿")
        if not title:
            raise HTTPException(422, "title is required")
        if not lyrics.strip():
            raise HTTPException(422, "lyrics are required")
        if language not in LANGS:
            raise HTTPException(422, f"language must be one of {LANGS}")
        try:
            opts = json.loads(options or "{}")
        except json.JSONDecodeError as e:
            raise HTTPException(422, f"options is not JSON: {e}") from e
        if not isinstance(opts, dict):
            raise HTTPException(422, "options must be a JSON object")
        refine = opts.get("refine")
        if refine is not None and refine not in REFINE:
            raise HTTPException(422, f"refine must be one of {REFINE}")
        sep_model = opts.get("sep_model")
        if sep_model is not None and not re.fullmatch(r"[\w.\-]+", str(sep_model)):
            raise HTTPException(422, "invalid sep_model")
        style = opts.get("style")
        if style is not None and style not in STYLES:
            raise HTTPException(422, f"style must be one of {list(STYLES)}")
        if song_id:
            song_id = check_song_id(song_id)
            if (songs_root / song_id).exists() or song_id in job_manager.active_song_ids():
                raise HTTPException(409, f"song id {song_id!r} already exists")
        else:
            song_id = new_song_id(title, songs_root)
            base, n = song_id, 2
            while song_id in job_manager.active_song_ids() or (songs_root / song_id).exists():
                song_id, n = f"{base}-{n}", n + 1

        updir = songs_root / UPLOAD_DIRNAME / uuid.uuid4().hex[:12]
        updir.mkdir(parents=True)
        suffix = re.sub(r"[^\w.]", "", Path(audio.filename or "audio").suffix)[:10] or ".bin"
        audio_path = updir / f"audio{suffix}"

        def save_upload() -> None:
            with audio_path.open("wb") as f:
                shutil.copyfileobj(audio.file, f, length=1 << 20)
        await run_in_threadpool(save_upload)
        if audio_path.stat().st_size == 0:
            shutil.rmtree(updir, ignore_errors=True)
            raise HTTPException(422, "audio file is empty")
        lyrics_path = updir / "lyrics.txt"
        lyrics_path.write_text(lyrics, encoding="utf-8")
        argv = build_prep_argv(audio_path, lyrics_path, title=title, song_id=song_id, songs_dir=songs_root,
                               artist=artist.strip(), language=language, lead=bool(opts.get("lead")),
                               refine=refine, sep_model=sep_model, style=style)
        job = job_manager.submit(argv, song_id=song_id, title=title, cleanup=updir,
                                 expect=songs_root / song_id / "song.json", output=songs_root / song_id)
        return {"job": job.to_dict(), "song_id": song_id}

    async def announce_lines(song_id: str, song: Song) -> None:
        """After the server saved new line timings: a new rev makes every screen re-fetch the song,
        and a running session of this song picks up the lines."""
        hub.note_song_saved(song_id)
        hub.song_rev[song_id] = hub.song_rev.get(song_id, 0) + 1
        if hub.session is not None and hub.session.song_id == song_id:
            await run_in_threadpool(hub.session.reload_lines, song)

    @app.put("/api/songs/{song_id}/lines/{idx}")
    async def line_edit(song_id: str, idx: int, edit: LineEdit) -> dict:
        """Edit a line's timing: exactly one of `start`, `shift`, `end` (see move_line). Neighbours
        make room, so only a malformed edit or one that would push lines past either end of the song
        fails (422, nothing saved). `lines` holds every modified line (empty: nothing changed),
        `pushed` the lines moved to make room; POST .../undo reverts the edit."""
        song_id = check_song_id(song_id)
        sdir = sdir_of(song_id)
        async with hub.edit_lock:
            song = load_song(sdir)
            key_before = lines_key(song)
            try:
                move = move_line(song, idx, shift=edit.shift, start=edit.start, end=edit.end)
            except IndexError as e:
                raise HTTPException(404, f"line {idx} out of range") from e
            except ValueError as e:
                raise HTTPException(422, str(e)) from e
            if move.touched:
                label = f"line {idx + 1} {'end' if edit.end is not None else 'start'}"
                save_song(song, sdir)
                hub.record_edit(song_id, UndoStep(label, move.before, lines_key(song)), key_before)
                await announce_lines(song_id, song)
        return {"song_id": song_id, "shift": move.shift, "clamped": False,     # (edits are never clamped now)
                "pushed": move.pushed, "rev": hub.song_rev.get(song_id, 0),
                "undo_left": len(hub.undo.get(song_id, ())),
                "lines": {str(i): asdict(song.lines[i]) for i in move.touched}}

    @app.post("/api/songs/{song_id}/undo")
    async def line_undo(song_id: str) -> dict:
        """Revert the song's last timing edit (history in memory, UNDO_DEPTH edits per song, lost on
        restart). 409 when there is nothing to undo, or when the song's lines changed on disk since
        that edit (e.g. a re-prep; the history is then dropped). `lines` holds the restored lines."""
        song_id = check_song_id(song_id)
        sdir = sdir_of(song_id)
        async with hub.edit_lock:
            hist = hub.undo.get(song_id)
            if not hist:
                raise HTTPException(409, "Nothing to undo")
            song = load_song(sdir)
            step = hist[-1]
            if lines_key(song) != step.after:
                hist.clear()
                raise HTTPException(409, "The song's lines changed on disk since that edit (prepared again?), "
                                         "so it cannot be undone")
            for i, d in step.before.items():
                song.lines[i] = line_from_dict(d)
            save_song(song, sdir)
            hist.pop()
            await announce_lines(song_id, song)
        return {"song_id": song_id, "undone": step.label, "undo_left": len(hist),
                "rev": hub.song_rev.get(song_id, 0),
                "lines": {str(i): asdict(song.lines[i]) for i in sorted(step.before)}}

    # -- jobs -----------------------------------------------------------------------------------------
    @app.get("/api/jobs")
    async def jobs_list() -> list[dict]:
        return [j.to_dict(log_lines=0) for j in job_manager.list()]

    @app.get("/api/jobs/{job_id}")
    async def job_detail(job_id: str) -> dict:
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(404, f"job {job_id!r} not found")
        return job.to_dict(log_lines=60)

    @app.delete("/api/jobs/{job_id}")
    async def job_cancel(job_id: str) -> dict:
        if job_manager.get(job_id) is None:
            raise HTTPException(404, f"job {job_id!r} not found")
        return {"cancelled": job_manager.cancel(job_id)}

    # -- devices --------------------------------------------------------------------------------------
    @app.get("/api/devices")
    async def devices_list(refresh: bool = False) -> list[dict]:
        if refresh and hub.session is not None:
            raise HTTPException(409, "stop the session before rescanning devices")
        try:
            return await run_in_threadpool(hub.devices, refresh)
        except Exception as e:
            raise HTTPException(503, f"audio devices unavailable: {e}") from e

    # -- session --------------------------------------------------------------------------------------
    def need_session() -> Any:
        if hub.session is None:
            raise HTTPException(409, "no live session running")
        return hub.session

    @app.get("/api/session")
    async def session_get() -> dict:
        return {"session": hub.session_req, "state": hub.state_message(), "display": asdict(hub.display)}

    @app.post("/api/session/start")
    async def session_start(req: SessionStart) -> dict:
        """Start (or replace) the live session. Make before break: the running session keeps
        following until the new one has been built and started, so a rejected request (bad
        channel, missing stem, ...) leaves the stage as it was."""
        sdir = sdir_of(req.song_id)
        async with hub.op_lock:
            song_stat = hub.song_file_stat(req.song_id)      # before the session reads song.json
            try:
                sess = await run_in_threadpool(hub.session_factory, sdir, req, songs_root)
            except Exception as e:
                raise _start_error(e) from e
            old = hub.session
            try:
                try:
                    await run_in_threadpool(sess.start)
                except Exception:
                    old_kind = ((hub.session_req or {}).get("input") or {}).get("kind")
                    if old is None or req.input.kind != "device" or old_kind != "device":
                        raise
                    # the running session may hold the interface exclusively: free it and retry once
                    log.info("starting the new input failed while the old one runs; retrying after stopping it")
                    await hub.stop_session()
                    old = None
                    await run_in_threadpool(sess.start)
            except Exception as e:
                await run_in_threadpool(_quiet_stop, sess)
                raise _start_error(e) from e
            hub.session, hub.session_req = sess, req.model_dump()
            hub.cued_song, hub.manual_time = req.song_id, None
            hub.watch_from(sess.song_id, song_stat)
            if old is not None:
                await run_in_threadpool(_quiet_stop, old)
        return hub.state_message()

    @app.post("/api/session/stop")
    async def session_stop() -> dict:
        async with hub.op_lock:
            await hub.stop_session()
        return hub.state_message()

    @app.post("/api/session/cue")
    async def session_cue(req: CueReq) -> dict:
        if req.song_id is not None:
            sdir_of(req.song_id)
        if hub.session is not None and hub.session.song_id != req.song_id:
            raise HTTPException(409, "stop the running session before cueing another song")
        if req.song_id is not None:
            hub.watch_from(req.song_id, hub.song_file_stat(req.song_id))  # before the screens fetch it
        hub.cued_song, hub.manual_time = req.song_id, None
        return hub.state_message()

    @app.post("/api/session/hold")
    async def session_hold(req: HoldReq | None = None) -> dict:
        await run_in_threadpool(need_session().hold, True if req is None else req.on)
        return hub.state_message()

    @app.post("/api/session/resume")
    async def session_resume() -> dict:
        await run_in_threadpool(need_session().hold, False)
        return hub.state_message()

    @app.post("/api/session/jump")
    async def session_jump(req: JumpReq) -> dict:
        if (req.line is None) == (req.time is None):
            raise HTTPException(422, "give exactly one of 'line' or 'time'")
        if req.song_id is not None and hub.song_id is not None and req.song_id != hub.song_id:
            raise HTTPException(409, f"{req.song_id!r} is not the song on the prompter ({hub.song_id!r}); "
                                     "start or cue it first")
        if hub.session is None:
            if not hub.cued_song:
                raise HTTPException(409, "no song cued")
            try:
                song = load_song_cached(hub, hub.cued_song)
            except (OSError, ValueError, KeyError, TypeError) as e:
                raise HTTPException(409, f"cannot read the cued song {hub.cued_song!r}: {e}") from e
            if req.line is not None:
                if not 0 <= req.line < len(song.lines) or song.lines[req.line].start is None:
                    raise HTTPException(404, f"line {req.line} has no timing")
                hub.manual_time = float(song.lines[req.line].start)  # type: ignore[arg-type]
            else:
                hub.manual_time = max(0.0, float(req.time))  # type: ignore[arg-type]
            return hub.state_message()
        sess = hub.session
        try:
            if req.line is not None:
                await run_in_threadpool(sess.jump_line, req.line)
            else:
                await run_in_threadpool(sess.jump_time, float(req.time))  # type: ignore[arg-type]
        except (IndexError, ValueError) as e:
            raise HTTPException(404, f"cannot jump: {e}") from e
        return hub.state_message()

    @app.post("/api/session/nudge")
    async def session_nudge(req: NudgeReq) -> dict:
        if hub.session is None and hub.manual_time is not None:
            hub.manual_time = max(0.0, hub.manual_time + req.dt)
            return hub.state_message()
        await run_in_threadpool(need_session().nudge, req.dt)
        return hub.state_message()

    @app.post("/api/session/transpose")
    async def session_transpose(req: TransposeReq) -> dict:
        await run_in_threadpool(need_session().set_transpose, req.semitones)
        return hub.state_message()

    @app.post("/api/session/autokey")
    async def session_autokey(req: HoldReq) -> dict:
        await run_in_threadpool(need_session().set_auto_key, req.on)
        return hub.state_message()

    @app.post("/api/session/latency")
    async def session_latency(req: LatencyReq) -> dict:
        await run_in_threadpool(need_session().set_latency, req.seconds)
        return hub.state_message()

    @app.post("/api/session/style")
    async def session_style(req: StyleReq) -> dict:
        """Switch the running session's tracking style live (song.json is not changed: PUT
        /api/songs/{id}/style saves it)."""
        await run_in_threadpool(need_session().set_style, req.style)
        if hub.session_req is not None:
            hub.session_req["style"] = req.style        # GET /api/session reports the style in use
        return hub.state_message()

    @app.put("/api/songs/{song_id}/style")
    async def song_style(song_id: str, req: StyleReq) -> dict:
        """Save the song's tracking style to song.json (a new rev: screens reload the song) and switch a
        running session of this song to it. `session`: whether a live session was switched."""
        song_id = check_song_id(song_id)
        sdir = sdir_of(song_id)
        async with hub.edit_lock:                    # (song.json saves land in order with timing edits)
            song = load_song(sdir)
            if song.meta.style != req.style:
                song.meta.style = req.style
                save_song(song, sdir)
                hub.note_song_saved(song_id)
                hub.song_rev[song_id] = hub.song_rev.get(song_id, 0) + 1
            setter = getattr(hub.session, "set_style", None) if hub.session is not None else None
            live = callable(setter) and hub.session.song_id == song_id
            if live:
                await run_in_threadpool(setter, req.style)
                if hub.session_req is not None:
                    hub.session_req["style"] = req.style
        return {"song_id": song_id, "style": req.style, "rev": hub.song_rev.get(song_id, 0), "session": live}

    # -- display --------------------------------------------------------------------------------------
    @app.get("/api/display")
    async def display_get() -> dict:
        return {**asdict(hub.display), "rev": hub.display_rev}

    @app.put("/api/display")
    async def display_put(patch: DisplayPatch) -> dict:
        for k, v in patch.model_dump(exclude_none=True).items():
            setattr(hub.display, k, v)
        hub.display_rev += 1
        save_display(hub.display_path, hub.display)       # a restarted server keeps them
        return {**asdict(hub.display), "rev": hub.display_rev}

    # -- websocket ------------------------------------------------------------------------------------
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        with contextlib.suppress(WebSocketDisconnect):
            await hub.serve(ws)

    app.mount("/static", RevalidatingStatic(directory=STATIC_DIR), name="static")
    return app


# --------------------------------------------------------------------------- cli


def lan_addresses() -> list[str]:
    """This machine's IPv4 address on the local network (best effort; nothing is sent and no name is
    looked up: connecting a UDP socket only picks the outgoing interface)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.254.254.254", 1))
            ip = s.getsockname()[0]
    except OSError:
        return []
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return []
    return [] if addr.is_loopback or addr.is_unspecified else [ip]


def banner(host: str, port: int) -> str:
    """The URLs to open; for a wildcard host also the ones other devices (a tablet) can reach."""
    wildcard = host in ("0.0.0.0", "::", "")
    shown = "127.0.0.1" if wildcard else (f"[{host}]" if ":" in host else host)
    out = [f"console   http://{shown}:{port}/", f"prompter  http://{shown}:{port}/prompter"]
    if wildcard:
        names = [*lan_addresses(), *sorted(n for n in machine_host_names() if n.endswith(".local"))]
        if names:
            out.append("on a tablet or stage monitor in the same network, open one of:")
            out += [f"          http://{n}:{port}/prompter" for n in names]
    return "\n".join(out)


def cli(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lyricprompter serve", description="Run the console + prompter web server")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach prompters on other devices")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--songs-dir", type=Path, default=SONGS_DIR)
    ap.add_argument("--log-level", default="info")
    ap.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                    help="extra host name the console / prompters are opened by (repeatable); loopback, "
                         "IP addresses and this machine's name always work")
    args = ap.parse_args(argv)

    import uvicorn

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(args.songs_dir, allowed_hosts=args.allow_host)
    print(banner(args.host, args.port), flush=True)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("warning: no authentication; anyone on this network can control the session", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, workers=1)
    return 0
