"""FastAPI server: projects (reference recording + cue file), cue editing with undo, OSC targets and
latency measurement, the live session, and one 20 Hz WebSocket broadcaster feeding the console (/).

Run a single process (`oscsender serve`): the live session, its audio threads and the one `OscSender`
every OSC message goes out through (cues fired by the session, Fire / Test / preview sends, loopback
measurements) live in it. There is no authentication; bind to 127.0.0.1 unless the show network is
trusted (OriginGuard keeps other web pages away from the controls).

Cue files are the source of truth and may be edited by hand while the server runs. An edit made through
the API reads the file, applies one of the `cues` edit functions (they rewrite only what they change),
saves it atomically, records the previous text for undo, gives a running session of that project the new
cues, adds the file's new `@target` declarations to the settings and bumps the project's broadcast rev,
so every console re-fetches it. A watcher stats the cue files of the live project and of the project a
console last opened twice a second and handles a change made outside the server the same way (without
an undo record: an undo across such a change is refused and the history dropped).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import functools
import ipaddress
import json
import logging
import math
import re
import shutil
import socket
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import numpy as np
from fastapi import (
    Body,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import AfterValidator, BaseModel, Field

from .. import config
from ..audio_io import AudioError
from ..config import STYLES
from ..cues import (
    Cue,
    CueEditError,
    CueSheet,
    TargetDecl,
    add_cue,
    delete_cue,
    format_time,
    parse_cues,
    set_cue_message,
    set_cue_time,
    sheet_rev,
    shift_cue,
)
from ..latency import AudioCapture, measure_echo, measure_loopback, message_from_text, target_send
from ..osc import OscSender, Target, resolve
from ..project import (
    META_FILE,
    check_project_id,
    create_project,
    cues_path,
    list_projects,
    load_meta,
    nfc,
    project_dir,
    read_cues,
    save_meta,
    set_vocals,
    trash_project,
    waveform,
    write_cues,
)
from ..settings import (
    INPUT_LATENCY_MS,
    Settings,
    declared_conflicts,
    load_settings,
    merge_declared,
    save_settings,
    settings_from_dict,
    settings_path,
    validate_settings,
)
from ..songfile import STEM_FILES, stem_path

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_PORT = 8800
REHEARSAL_PREROLL = 2.0  # s of the project's own stem played before the start position in a rehearsal
WS_SEND_TIMEOUT = 2.0    # a WebSocket that accepts nothing for this long is disconnected
WATCH_PERIOD_S = 0.5     # how often the watched cue files are stat'ed
# A file written this recently (s) may change again without changing its signature (FAT32 keeps mtime in
# 2 s steps, HFS+ in 1 s steps): the watcher compares its text too.
WATCH_RECENT_S = 5.0
RESOLVE_RETRY_S = 5.0    # an enabled target host without an address is looked up again this often
RESOLVE_REFRESH_S = 60.0     # a resolved host name is looked up again this often (a new DHCP address)
UNDO_DEPTH = 100         # cue edits that can be undone, per project (kept in memory)
UPLOAD_PREFIX = ".upload-"   # temporary upload folders in the projects root (dot-folders are not projects)
TEST_ADDRESS = "/oscsender/test"
LOG_NAME_RE = re.compile(r"[\w\-][\w.\-]*")      # fullmatch: a show log file name


# --------------------------------------------------------------------------- JSON helpers


def json_safe(x: Any) -> Any:
    """`x` with every non-finite float replaced by None and numpy values made plain Python, recursively.
    Browsers' JSON.parse rejects NaN / Infinity, so one such value would make a whole message unreadable."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist())
    if isinstance(x, np.generic):
        return json_safe(x.item())
    return x


Signature = tuple[int, int, int, int]


def file_signature(path: Path) -> Signature | None:
    """(mtime_ns, size, inode, ctime_ns) of a file, None when it is missing: tells a file replaced on disk
    (an atomic save gets a new inode even when the time and size are those of the old file)."""
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size, st.st_ino, st.st_ctime_ns


def recently_written(sig: Signature | None) -> bool:
    """The file was written so recently that a later save may keep its signature (coarse mtime steps)."""
    return sig is not None and abs(time.time_ns() - sig[0]) < WATCH_RECENT_S * 1e9


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


_nan_logged = False


def _dumps(msg: dict) -> str:
    """Strict JSON (browsers reject NaN / Infinity): a non-finite number becomes null rather than
    making the whole message, and with it every console, unreadable."""
    global _nan_logged
    try:
        return json.dumps(msg, ensure_ascii=False, allow_nan=False, default=_json_default)
    except ValueError:
        if not _nan_logged:
            _nan_logged = True
            log.warning("non-finite number in a %r message; sent as null", msg.get("type"))
        return json.dumps(json_safe(msg), ensure_ascii=False, allow_nan=False, default=_json_default)


def _kind(msg: dict) -> str:
    """Outbox slot of a message: one per type, and one per project for project changes."""
    return f"project:{msg.get('id')}" if msg["type"] == "project" else msg["type"]


def _encode(msgs: Iterable[dict]) -> list[tuple[str, str]]:
    return [(_kind(m), _dumps(m)) for m in msgs]


# --------------------------------------------------------------------------- request models


ProjectId = Annotated[str, AfterValidator(nfc)]
# Every float the API takes is finite: NaN / Infinity (accepted by Python's JSON parser) would reach
# the tracker or the cue file, or end up in the state messages, which browsers then cannot parse.
Finite = Annotated[float, Field(allow_inf_nan=False)]
CueTime = Finite | str          # seconds, or a time token in cue file syntax ("1:05.2")


class InputSpec(BaseModel):
    kind: Literal["device", "file"] = "device"
    device: int | str | None = None              # index or name substring (None: default input)
    channels: list[int] | None = None            # 1-based
    samplerate: float | None = Field(None, gt=0.0, allow_inf_nan=False)
    stem: str | None = None                      # file: "mix" | "vocals" of the session's project
    path: str | None = None                      # file: audio path on the server
    monitor: bool = False
    speed: float = Field(1.0, gt=0.1, le=4.0, allow_inf_nan=False)
    start_s: float = Field(0.0, ge=0.0, allow_inf_nan=False)


class SessionStart(BaseModel):
    project_id: ProjectId
    mode: Literal["mix", "vocal"] = "mix"
    style: Literal["classic", "pop"] | None = None    # None: the project's style
    input: InputSpec = Field(default_factory=InputSpec)
    start_time: float | None = Field(None, ge=0.0, allow_inf_nan=False)
    start_cue: int | None = None                      # cue id (line number): start at its time
    rev: str | None = None                            # sheet_rev the start_cue was picked in (409 if stale)
    transpose: int = Field(0, ge=-12, le=12)
    auto_key: bool = True                             # automatic key detection (off when a key is set)


class OnReq(BaseModel):
    on: bool = True


class StyleReq(BaseModel):
    style: Literal["classic", "pop"]


class JumpReq(BaseModel):
    time: Finite | None = None
    cue_id: int | None = None
    # the project the cue / time belongs to (optional): refused with 409 when another project is
    # live, so a cue picked in one project never moves the session of another
    project_id: ProjectId | None = None
    rev: str | None = None          # sheet_rev the cue_id was picked in: 409 when the file changed since


class NudgeReq(BaseModel):
    dt: float = Field(..., ge=-30.0, le=30.0, allow_inf_nan=False)


class TransposeReq(BaseModel):
    semitones: int = Field(..., ge=-12, le=12)


class InputLatencyReq(BaseModel):
    ms: float = Field(..., ge=INPUT_LATENCY_MS[0], le=INPUT_LATENCY_MS[1], allow_inf_nan=False)


class CueText(BaseModel):
    text: str
    rev: str | None = None


class CuePatch(BaseModel):
    """Exactly one of a new `time`, a `shift` of the time, or a new `message` (`[target] /address args
    [# note]`)."""
    rev: str | None = None
    time: CueTime | None = None
    shift: float | None = Field(None, ge=-86400.0, le=86400.0, allow_inf_nan=False)
    message: str | None = None


class CueAdd(BaseModel):
    rev: str | None = None
    time: CueTime
    message: str


class FireReq(BaseModel):
    source: Literal["manual", "preview"] = "manual"
    rev: str | None = None


class TestReq(BaseModel):
    message: str | None = None


class EchoReq(BaseModel):
    target: str | None = None
    host: str | None = None
    port: int | None = Field(None, ge=1, le=65535)
    count: int = Field(20, ge=1, le=200)
    interval_ms: float = Field(50.0, ge=0.0, le=5000.0, allow_inf_nan=False)
    timeout_ms: float = Field(500.0, gt=0.0, le=5000.0, allow_inf_nan=False)
    message: str | None = None                   # a reply-triggering message (default: ping / pong)
    listen_port: int | None = Field(None, ge=0, le=65535)


class LoopbackReq(BaseModel):
    target: str
    device: int | str | None = None
    channels: list[int] | None = None
    count: int = Field(8, ge=1, le=50)
    interval_ms: float = Field(1500.0, ge=100.0, le=10000.0, allow_inf_nan=False)
    threshold_db: float = Field(20.0, gt=0.0, le=80.0, allow_inf_nan=False)
    message: str | None = None                   # default: the target's test message, else /oscsender/test


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


# --------------------------------------------------------------------------- project / cue helpers


def stems_present(pdir: Path) -> list[str]:
    return [name for name in STEM_FILES if stem_path(pdir, name).is_file()]


def load_text(pdir: Path) -> tuple[str, str | None]:
    """(cue file text, None), or ("", error message) when the file cannot be read (not UTF-8, ...)."""
    try:
        return read_cues(pdir), None
    except ValueError as e:
        return "", str(e)
    except OSError as e:
        return "", f"cannot read the cue file: {e}"


def _duration(pdir: Path) -> float | None:
    try:
        return load_meta(pdir).duration
    except (OSError, ValueError):
        return None


def _project_style(pdir: Path) -> str | None:
    try:
        return load_meta(pdir).style
    except (OSError, ValueError):
        return None


def _read_cues_and_duration(pdir: Path) -> tuple[str, str | None, float | None]:
    text, err = load_text(pdir)
    return text, err, _duration(pdir)


def sheet_json(sheet: CueSheet, settings: Settings, duration: float | None, *,
               read_error: str | None = None) -> dict:
    """`CueSheet.to_dict()` with the server's own warnings added to "issues" (sorted by line): cue targets
    that are neither OSC targets in the settings nor declared in the file, @target lines that differ from
    the OSC target of that name (the settings win), and cues after the end of the reference recording."""
    d = sheet.to_dict()
    issues = list(d["issues"])
    if read_error is not None:
        issues.append({"line": 0, "severity": "error", "message": read_error})
    known = {t.name for t in settings.targets} | {t.name for t in sheet.targets}
    issues += [i.to_dict() for i in declared_conflicts(settings, sheet.targets)]
    for c in sheet.cues:
        if c.target is not None and c.target not in known:
            issues.append({"line": c.id, "severity": "warning",
                           "message": f"unknown target {c.target!r} (not one of the OSC targets and not "
                                      "declared with @target): the cue cannot be sent"})
        if duration is not None and duration > 0 and c.time > duration + 1e-6:
            issues.append({"line": c.id, "severity": "warning",
                           "message": f"the cue at {format_time(c.time)} is after the end of the reference "
                                      f"recording ({format_time(duration)})"})
    d["issues"] = sorted(issues, key=lambda i: i["line"])
    return d


def project_summary(pdir: Path, rev: int = 0) -> dict:
    """One row of GET /api/projects (a project whose project.json cannot be read gets an "error")."""
    pid = nfc(pdir.name)                 # the id is the folder name
    try:
        meta = load_meta(pdir)
    except (OSError, ValueError) as e:
        return {"id": pid, "title": pid, "error": str(e), "rev": rev}
    out = {"id": pid, "title": meta.title, "duration": meta.duration, "style": meta.style,
           "stems": stems_present(pdir), "n_cues": None, "n_errors": None, "created": meta.created, "rev": rev}
    text, err = load_text(pdir)
    if err is not None:
        out["error"] = err
    else:
        sheet = parse_cues(text)
        out.update(n_cues=len(sheet.cues), n_errors=sheet.n_errors)
    return out


def _log_names(pdir: Path) -> list[str]:
    """Show log file names, newest first."""
    logs = Path(pdir) / "logs"
    try:
        return sorted((f.name for f in logs.iterdir() if f.is_file() and not f.name.startswith(".")), reverse=True)
    except OSError:
        return []


def _safe_name(filename: str | None, default: str) -> str:
    """An uploaded file's base name, usable as one path component (the original name is kept for
    project.json's "sources")."""
    name = re.sub(r"[\x00-\x1f/\\]", "_", Path(filename or "").name).strip()
    if name in ("", ".", ".."):
        return default
    while len(name.encode("utf-8")) > 200:
        name = name[1:]
    return name


def _given(f: UploadFile | None) -> bool:
    """Browsers send an empty part (no file name) for an empty file input."""
    return f is not None and bool(f.filename or f.size)


def _save_upload(f: UploadFile, folder: Path, default: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / _safe_name(f.filename, default)
    with path.open("wb") as out:
        shutil.copyfileobj(f.file, out, length=1 << 20)
    return path


def check_utf8(text: str, what: str) -> None:
    """400 for text that cannot be saved as UTF-8 (a lone surrogate, which a JSON string can carry)."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as e:
        raise HTTPException(400, f"{what} is not valid Unicode text (character {e.start + 1} is a lone "
                                 "surrogate)") from None


def _decode_cue_upload(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise HTTPException(400, f"the cue file is not UTF-8 text (byte {e.start}): save it as UTF-8") from None


@dataclass
class UndoStep:
    """One saved cue edit of a project, as POST .../cues/undo reverts it."""
    label: str                  # what was edited ("cue 12 time"), for the console's notice
    before: str                 # the cue file text before the edit
    after: str                  # ... and right after it


class SheetConflict(Exception):
    """The client edited another revision of the cue file: 409 with the current sheet."""

    def __init__(self, detail: str, cues: dict):
        super().__init__(detail)
        self.detail, self.cues = detail, cues


# --------------------------------------------------------------------------- live session


@dataclass
class SessionContext:
    """What the server hands a session factory besides the project folder and the request."""
    projects_dir: Path
    cues: list[Cue]                     # the project's cues as read for this start
    sender: OscSender                   # the Hub's sender (targets set from the settings)
    settings: Settings                  # a copy: the session owns it
    start_time: float                   # reference time to start at (start_time, the start cue's time, or 0)
    clock: Callable[[], float] = time.perf_counter


SessionFactory = Callable[[Path, SessionStart, SessionContext], Any]


def default_session_factory(pdir: Path, req: SessionStart, ctx: SessionContext) -> Any:
    """An OscSession (live.session) following a device input or a file played as one."""
    from ..live.audio_input import DeviceInput, FileInput
    from ..live.session import OscSession

    inp = req.input
    same_axis = False      # the input is a stem of the session's own project (file time == reference time)
    if req.mode == "vocal" and not stem_path(pdir, "vocals").is_file():
        raise ValueError("vocal mode needs the project's vocal stem: add one first")
    if inp.kind == "device":
        source: Any = DeviceInput(inp.device, inp.channels, inp.samplerate)
    else:
        if inp.stem:
            if inp.stem not in STEM_FILES:
                raise ValueError(f"unknown stem {inp.stem!r}; expected one of {', '.join(STEM_FILES)}")
            path = stem_path(pdir, inp.stem)
            if not path.is_file():
                raise FileNotFoundError(f"the project has no {inp.stem} stem")
            same_axis = True
        elif inp.path:
            path = Path(inp.path).expanduser()
            path = path if path.is_absolute() else config.PROJECT_ROOT / path
        else:
            raise ValueError("file input needs 'path' or 'stem'")
        source = FileInput(path, realtime=True, speed=inp.speed, monitor=inp.monitor, start_s=inp.start_s)
    sess = OscSession(pdir, req.mode, source, cues=ctx.cues, sender=ctx.sender, settings=ctx.settings,
                      style=req.style, start_time=ctx.start_time, clock=ctx.clock, log_dir=Path(pdir) / "logs")
    if same_axis and (req.start_time is not None or req.start_cue is not None) and inp.start_s == 0:
        sess.align_source_start(REHEARSAL_PREROLL)      # rehearse from cue N, not from 0:00
    sess.set_input_latency(ctx.settings.input_latency_ms)
    if req.transpose or not req.auto_key:
        sess.set_transpose(req.transpose)          # an operator key turns auto detection off ...
        if req.auto_key:
            sess.set_auto_key(True)                # ... unless Auto stays on: the search starts from it
    return sess


def _default_devices(refresh: bool) -> list[dict]:
    from ..live.audio_input import list_input_devices

    return list_input_devices(refresh=refresh)


def _start_error(e: Exception) -> HTTPException:
    if isinstance(e, HTTPException):
        return e
    if isinstance(e, FileNotFoundError):
        return HTTPException(404, f"file not found: {e}")
    if isinstance(e, (ValueError, TypeError, IndexError, KeyError)):
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


def _record_dict(rec: Any) -> dict:
    if hasattr(rec, "to_dict"):
        return rec.to_dict()
    return dict(rec) if isinstance(rec, dict) else {}


# --------------------------------------------------------------------------- OSC target hosts


HostKey = tuple[str, int]               # (host, port) as in the settings


def _is_name(host: str) -> bool:
    """A host name (its address may change), not an IP address literal."""
    try:
        ipaddress.ip_address(host.strip().strip("[]"))
    except ValueError:
        return True
    return False


class HostBook:
    """The addresses of the OSC targets' hosts, looked up apart from setting the targets.

    The Hub's sender resolves through `cached`, which only reads this book, so giving the sender new
    targets (under the settings lock, during a show) never waits on DNS: a lookup of an unknown `.local`
    name blocks for 5 s. The lookups run in worker threads outside every lock (`Hub.resolve_hosts`), once
    per host and port: a target whose host did not change is not looked up again when another one is
    edited. A failed lookup never replaces an address that worked (the target keeps sending there; the
    failure is logged)."""

    def __init__(self, lookup: Callable[[str, int], tuple[int, tuple]]):
        self.lookup = lookup
        self.good: dict[HostKey, tuple[int, tuple]] = {}
        self.error: dict[HostKey, str] = {}
        self.checked: dict[HostKey, float] = {}      # monotonic time of the last lookup

    def cached(self, host: str, port: int) -> tuple[int, tuple]:
        """The sender's resolver: the address in the book, else OSError with the lookup's error."""
        key = (host, int(port))
        hit = self.good.get(key)
        if hit is None:
            raise OSError(self.error.get(key, "looking up the address"))
        return hit

    def record(self, key: HostKey, result: tuple[int, tuple] | None, error: str | None, now: float) -> bool:
        """Store a lookup's outcome; True when what the sender sends to changes."""
        self.checked[key] = now
        if result is not None:
            changed = self.good.get(key) != result
            self.good[key] = result
            self.error.pop(key, None)
            return changed
        msg = error or "no address"
        if self.error.get(key) == msg:
            return False
        if key in self.good:
            log.warning("looking up %s:%d again failed (%s): it keeps the address it had", *key, msg)
        else:
            log.warning("cannot resolve the OSC target host %s:%d: %s", *key, msg)
        self.error[key] = msg
        return key not in self.good          # the sender reports the new error

    def retain(self, targets: Iterable[Target]) -> None:
        """Forget the hosts no target uses any more."""
        keep = {(t.host, t.port) for t in targets}
        for d in (self.good, self.error, self.checked):
            for key in [k for k in d if k not in keep]:
                del d[key]

    def new(self, targets: Iterable[Target]) -> list[HostKey]:
        """Hosts never looked up (disabled targets' too: once)."""
        return list(dict.fromkeys(k for k in ((t.host, t.port) for t in targets) if k not in self.checked))

    def missing(self, targets: Iterable[Target]) -> list[HostKey]:
        """Enabled targets' hosts without an address."""
        return list(dict.fromkeys(k for k in ((t.host, t.port) for t in targets if t.enabled)
                                  if k not in self.good))

    def due(self, targets: Iterable[Target], now: float) -> list[HostKey]:
        """Hosts to look up in the background: never looked up; enabled ones without an address every
        RESOLVE_RETRY_S; resolved host names every RESOLVE_REFRESH_S."""
        targets = list(targets)
        out = self.new(targets)
        for t in targets:
            key = (t.host, t.port)
            last = self.checked.get(key)
            if not t.enabled or last is None or key in out:
                continue
            every = RESOLVE_RETRY_S if key not in self.good else RESOLVE_REFRESH_S if _is_name(t.host) else None
            if every is not None and now - last >= every:
                out.append(key)
        return out


# --------------------------------------------------------------------------- hub


_WS_RANK = {"hello": 0, "project": 1, "settings": 1, "state": 2}


class WSClient:
    """One WebSocket: a latest-wins outbox (one pending message per type, per project for project
    changes) drained by the connection's own task, so a slow or stalled console never delays the
    broadcaster or the others."""

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
            for kind in sorted(self._outbox, key=lambda k: _WS_RANK.get(k.partition(":")[0], 1)):
                text = self._outbox.pop(kind, None)
                if text is not None:
                    await asyncio.wait_for(self.ws.send_text(text), self.send_timeout)
                    self.sent += 1


async def _until_disconnect(ws: Any) -> None:
    """Clients only listen; reading detects the disconnect."""
    while (await ws.receive())["type"] != "websocket.disconnect":
        pass


class Hub:
    """Server-wide state: the settings and the OscSender, the (single) live session, cue edit history,
    project revs, watched cue files and the WebSocket clients."""

    def __init__(self, projects_dir: Path, session_factory: SessionFactory, devices: Callable[[bool], list[dict]],
                 hz: float, sender: OscSender, *, clock: Callable[[], float] = time.perf_counter,
                 send_timeout: float = WS_SEND_TIMEOUT):
        self.projects_dir = Path(projects_dir)
        self.session_factory, self.devices, self.hz = session_factory, devices, hz
        self.sender, self.clock, self.send_timeout = sender, clock, send_timeout
        # The sender resolves through the host book (no DNS when its targets are set); the lookups the
        # sender would have made go through its own resolver, from resolve_hosts.
        self.hosts = HostBook(getattr(sender, "_resolver", resolve))
        sender._resolver = self.hosts.cached
        self._lookups: dict[HostKey, asyncio.Task] = {}
        self._resolve_wake = asyncio.Event()
        self.settings_file = settings_path(self.projects_dir)
        # what load_settings repaired in a hand-edited / damaged settings file (shown in the console until
        # the next save writes valid settings)
        self.settings_warnings: list[str] = []
        self.settings: Settings = load_settings(self.settings_file, self.settings_warnings)
        self.settings_rev = 0
        self.session: Any = None
        self.session_req: dict | None = None
        self.session_project: str | None = None
        self.focus: str | None = None              # the project a console last opened
        self.rev: dict[str, int] = {}              # project id -> broadcast rev (bumped on every change)
        self.sheet_revs: dict[str, str | None] = {}    # project id -> sheet_rev of its cue file at that rev
        self.undo: dict[str, list[UndoStep]] = {}  # project id -> cue edits, oldest first
        self.clients: set[WSClient] = set()
        self.op_lock = asyncio.Lock()              # session start / stop, project removal
        self.edit_lock = asyncio.Lock()            # cue / project.json writes land in order
        self.settings_lock = asyncio.Lock()        # settings changes (read-modify-write)
        self.loopback_lock = asyncio.Lock()        # one audio loopback measurement at a time
        self.rescanning = False                    # a device rescan (PortAudio re-initialised) runs
        self.broadcasts = 0
        # watched cue files: id -> (signature, sheet_rev of the text last announced / saved, None: unknown)
        self._watch: dict[str, tuple[Signature | None, str | None]] = {}
        self._sheets: dict[str, tuple[Signature, str, CueSheet]] = {}   # parsed cue files by signature
        self._sent_rev: dict[str, int] = {}
        self._sent_settings = 0

    def project_dir(self, pid: str) -> Path:
        return project_dir(self.projects_dir, pid)

    @property
    def live_project(self) -> str | None:
        return self.session_project if self.session is not None else None

    def device_session(self) -> bool:
        return self.session is not None and ((self.session_req or {}).get("input") or {}).get("kind") == "device"

    # -- messages ---------------------------------------------------------------------------------
    def hello_message(self) -> dict:
        return {"type": "hello", "project_id": self.live_project, "settings_rev": self.settings_rev}

    def project_message(self, pid: str) -> dict:
        return {"type": "project", "id": pid, "rev": self.rev.get(pid, 0), "sheet_rev": self.sheet_revs.get(pid)}

    def settings_message(self) -> dict:
        return {"type": "settings", "rev": self.settings_rev, "armed": self.settings.armed}

    def state_message(self) -> dict:
        if self.session is not None:
            try:
                snap = dict(self.session.snapshot())
            except Exception as e:
                log.exception("snapshot failed")
                return {"type": "state", "project_id": self.session_project, "status": "error", "error": str(e),
                        "t": round(time.time(), 4)}
            snap["type"], snap["project_id"] = "state", self.session_project
            return snap
        return {"type": "state", "status": "idle", "project_id": None, "armed": self.settings.armed,
                "t": round(time.time(), 4), "targets": self.sender.stats()}

    def init_messages(self) -> list[tuple[str, str]]:
        return _encode((self.hello_message(), self.state_message()))

    def _changes(self) -> list[dict]:
        out = []
        for pid, rev in list(self.rev.items()):
            if self._sent_rev.get(pid) != rev:
                self._sent_rev[pid] = rev
                out.append(self.project_message(pid))
        if self.settings_rev != self._sent_settings:
            self._sent_settings = self.settings_rev
            out.append(self.settings_message())
        return out

    def bump(self, pid: str, text: str | None = None) -> int:
        """A project changed (cue file, style, stems, created, removed): every console re-fetches it."""
        self.rev[pid] = self.rev.get(pid, 0) + 1
        if text is not None:
            self.sheet_revs[pid] = sheet_rev(text)
        return self.rev[pid]

    def forget(self, pid: str) -> None:
        """A removed project: drop its history and caches; the rev keeps counting (a project created
        again under the same id is still announced as a change)."""
        for d in (self.undo, self._sheets, self._watch, self.sheet_revs):
            d.pop(pid, None)
        if self.focus == pid:
            self.focus = None
        self.bump(pid)

    # -- cue files ---------------------------------------------------------------------------------
    def watch_from(self, pid: str, sig: Signature | None) -> None:
        """Watch `pid`'s cue file for changes after `sig`, taken before the file was read (a baseline
        taken later would absorb an edit landing in between). A file already watched keeps its baseline."""
        if pid not in self._watch:
            self._watch[pid] = (sig, None)

    def note_saved(self, pid: str, pdir: Path, text: str) -> None:
        """The server itself rewrote the cue file with `text` (the edit bumps the rev itself)."""
        if pid in self._watch:
            self._watch[pid] = (file_signature(cues_path(pdir)), sheet_rev(text))

    def _unchanged(self, pid: str, sig: Signature | None) -> bool:
        """The watched file's signature says it did not change (a missing file counts as unchanged: it is
        briefly missing while being replaced), and it was not written so recently that it might have."""
        return sig is None or (sig == self._watch[pid][0] and not recently_written(sig))

    async def cached_sheet(self, pid: str, pdir: Path) -> tuple[str, CueSheet]:
        """The project's cue text and sheet, re-read only when the file changed (Fire / preview sends)."""
        sig = file_signature(cues_path(pdir))
        hit = self._sheets.get(pid)
        if sig is not None and hit is not None and hit[0] == sig and not recently_written(sig):
            return hit[1], hit[2]
        text, err = await run_in_threadpool(load_text, pdir)
        if err is not None:
            raise HTTPException(400, err)
        sheet = parse_cues(text)
        if sig is not None:
            self._sheets[pid] = (sig, text, sheet)
        return text, sheet

    async def check_cue_files(self) -> list[str]:
        """Cue files of the live and the focus project replaced on disk (a text editor, a sync tool) bump
        the project's rev, reach a running session and add their declared targets. Returns the ids."""
        watched = [p for p in dict.fromkeys((self.live_project, self.focus)) if p]
        for pid in list(self._watch):
            if pid not in watched:
                del self._watch[pid]
        changed: list[str] = []
        for pid in watched:
            try:
                pdir = self.project_dir(pid)
            except ValueError:
                continue
            cur = file_signature(cues_path(pdir))
            if pid not in self._watch:
                self._watch[pid] = (cur, None)
                continue
            if self._unchanged(pid, cur):
                continue
            async with self.edit_lock:
                cur = file_signature(cues_path(pdir))
                if pid not in self._watch or self._unchanged(pid, cur):    # the server saved it meanwhile
                    continue
                known = self._watch[pid][1]
                text, err = await run_in_threadpool(load_text, pdir)
                rev = None if err else sheet_rev(text)
                if rev is not None and (rev == known or (known is None and cur == self._watch[pid][0])):
                    # the same text (touched, or a recent file read again): nothing to announce
                    self._watch[pid] = (cur, rev)
                    continue
                self._watch[pid] = (cur, rev)
                self.bump(pid, None if err else text)
                changed.append(pid)
                log.info("the cue file of %r changed on disk: consoles reload it (rev %d)", pid, self.rev[pid])
                if err is not None:
                    log.warning("%s", err)
                    continue
                sheet = parse_cues(text)
                await self.push_cues(pid, sheet.cues)
                await self.merge_targets(sheet.targets)
        return changed

    async def sync_live_cues(self) -> None:
        """Before the live session acts on a cue id (Fire, Go): a cue file changed on disk that the watcher
        has not seen yet reaches the session first, so the id means the same line as in the file."""
        pid = self.live_project
        if pid is None or pid not in self._watch:
            return
        try:
            pdir = self.project_dir(pid)
        except ValueError:
            return
        if not self._unchanged(pid, file_signature(cues_path(pdir))):
            await self.check_cue_files()

    async def watcher(self, period: float = WATCH_PERIOD_S) -> None:
        while True:
            try:
                await self.check_cue_files()
            except Exception:
                log.exception("watching the cue files failed")
            await asyncio.sleep(period)

    def record_edit(self, pid: str, step: UndoStep) -> None:
        """Remember a saved edit for undo. Steps recorded before the file changed otherwise (its text
        before this edit is not what the last step left, e.g. a hand edit) no longer apply: dropped."""
        hist = self.undo.setdefault(pid, [])
        if hist and hist[-1].after != step.before:
            hist.clear()
        hist.append(step)
        del hist[:-UNDO_DEPTH]

    async def push_cues(self, pid: str, cues: list[Cue]) -> None:
        sess = self.session
        if sess is None or self.session_project != pid:
            return
        try:
            await run_in_threadpool(sess.set_cues, list(cues))
        except Exception:
            log.exception("the running session keeps its cues: updating them failed")

    async def commit(self, pid: str, pdir: Path, old: str | None, new: str, label: str) -> list[str]:
        """Save an edited cue file (caller holds edit_lock) and announce it. `old` None: the previous
        text is unknown (unreadable file): no undo across this save. Returns the targets added."""
        if old is not None and new == old:
            return []
        await run_in_threadpool(write_cues, pdir, new)
        self.note_saved(pid, pdir, new)
        if old is not None:
            self.record_edit(pid, UndoStep(label, old, new))
        else:
            self.undo.pop(pid, None)
        sheet = parse_cues(new)
        self.bump(pid, new)
        await self.push_cues(pid, sheet.cues)
        return await self.merge_targets(sheet.targets)

    # -- settings ----------------------------------------------------------------------------------
    async def apply_settings(self, new: Settings) -> None:
        """Make `new` (validated) the settings (caller holds settings_lock): given to the sender (addresses
        from the host book: hosts not looked up yet are looked up in the background, never here) and to a
        running session, then saved; announced."""
        old = self.settings
        self.settings = new
        self.settings_rev += 1
        if new.targets != old.targets:
            self.hosts.retain(new.targets)
            self.sync_sender()
            self._resolve_wake.set()
        if self.session is not None:
            await self.settings_to_session(self.session, new, old.input_latency_ms)
        if await run_in_threadpool(save_settings, self.settings_file, new):
            self.settings_warnings = []

    async def settings_to_session(self, sess: Any, s: Settings, old_latency_ms: float | None) -> None:
        try:
            await run_in_threadpool(sess.apply_settings, s.copy())
            if old_latency_ms is None or s.input_latency_ms != old_latency_ms:
                await run_in_threadpool(sess.set_input_latency, s.input_latency_ms)
        except Exception:
            log.exception("applying the settings to the running session failed")

    async def merge_targets(self, decls: Iterable[TargetDecl]) -> list[str]:
        """Add a cue file's declared targets that the settings do not have yet; returns the names added."""
        decls = list(decls)
        if not decls:
            return []
        async with self.settings_lock:
            new, added = merge_declared(self.settings, decls)
            if added:
                log.info("OSC targets added from a cue file: %s", ", ".join(added))
                await self.apply_settings(new)
        return added

    # -- target hosts ------------------------------------------------------------------------------
    def sync_sender(self) -> None:
        """Give the sender the settings' targets with the addresses in the host book (no lookups)."""
        self.sender.set_targets(self.settings.targets)

    def start_lookups(self, keys: Iterable[HostKey]) -> list[asyncio.Task]:
        """Look up hosts in worker threads (one lookup per host at a time; a running one is shared)."""
        tasks = []
        for key in dict.fromkeys(keys):
            task = self._lookups.get(key)
            if task is None:
                task = self._lookups[key] = asyncio.create_task(self._lookup(key), name=f"lookup {key[0]}")
            tasks.append(task)
        return tasks

    async def _lookup(self, key: HostKey) -> None:
        try:
            try:
                res, err = await run_in_threadpool(self.hosts.lookup, *key), None
            except (OSError, UnicodeError, ValueError) as e:     # what OscSender.set_targets catches
                res, err = None, str(e) or type(e).__name__
            if self.hosts.record(key, res, err, time.monotonic()):
                self.sync_sender()
        finally:
            self._lookups.pop(key, None)

    async def resolve_hosts(self, keys: Iterable[HostKey]) -> None:
        """Look up `keys` and wait for them (never while holding a lock: a lookup can take seconds)."""
        tasks = self.start_lookups(keys)
        if tasks:
            await asyncio.wait(tasks)

    async def lookup_new_hosts(self, names: Iterable[str] | None = None) -> None:
        """Wait for the first lookup of the targets `names` (all when None) whose host was never looked up,
        so a reply shows whether they resolved."""
        wanted = None if names is None else set(names)
        targets = [t for t in self.settings.targets if wanted is None or t.name in wanted]
        await self.resolve_hosts(self.hosts.new(targets))

    async def retry_target(self, name: str) -> None:
        """Before a manual send (Test, Fire): an enabled target without an address is looked up again now
        (the receiver may have joined the network since)."""
        t = self.settings.target(name)
        if t is not None and t.enabled and (t.host, t.port) not in self.hosts.good:
            await self.resolve_hosts([(t.host, t.port)])

    async def resolver(self, period: float | None = None) -> None:
        """Background lookups: new hosts at once (apply_settings wakes it), enabled targets without an
        address every RESOLVE_RETRY_S, resolved host names every RESOLVE_REFRESH_S."""
        while True:
            self._resolve_wake.clear()
            try:
                self.start_lookups(self.hosts.due(self.settings.targets, time.monotonic()))
            except Exception:
                log.exception("looking up the OSC target hosts failed")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._resolve_wake.wait(), period or RESOLVE_RETRY_S)

    async def stop_lookups(self) -> None:
        tasks = list(self._lookups.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks)

    # -- broadcasting ------------------------------------------------------------------------------
    async def broadcaster(self) -> None:
        loop = asyncio.get_running_loop()
        period = 1.0 / self.hz
        next_t = loop.time()
        while True:
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
        # No await between building the init messages and joining: a change the broadcaster
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

    # -- session -----------------------------------------------------------------------------------
    async def stop_session(self) -> None:
        """Drop the session first, then stop it: a stop that fails (e.g. PortAudio cannot stop the
        stream of a device lost mid-session) is logged; the session is gone either way."""
        sess, self.session, self.session_req, self.session_project = self.session, None, None, None
        if sess is not None:
            await run_in_threadpool(_quiet_stop, sess)


# --------------------------------------------------------------------------- app


def create_app(projects_dir: Path | None = None, *, session_factory: SessionFactory | None = None,
               devices: Callable[[bool], list[dict]] | None = None, broadcast_hz: float = 20.0,
               allowed_hosts: Iterable[str] = (), sender: OscSender | None = None,
               clock: Callable[[], float] = time.perf_counter) -> FastAPI:
    """`allowed_hosts`: Host names accepted besides loopback, IP literals and this machine's name (see
    OriginGuard). `sender`: the OscSender to send through (default: a new one on `clock`); its targets
    are set from the settings at startup and it is closed at shutdown."""
    root = Path(projects_dir or config.PROJECTS_DIR).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    hub = Hub(root, session_factory or default_session_factory, devices or _default_devices, broadcast_hz,
              sender or OscSender(clock=clock), clock=clock)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await hub.lookup_new_hosts()
        hub.sync_sender()
        tasks = [asyncio.create_task(hub.broadcaster(), name="ws-broadcaster"),
                 asyncio.create_task(hub.watcher(), name="cue-watcher"),
                 asyncio.create_task(hub.resolver(), name="host-resolver")]
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await hub.stop_lookups()
            await hub.stop_session()
            hub.sender.close()

    app = FastAPI(title="oscsender", lifespan=lifespan)
    app.add_middleware(OriginGuard, allowed_hosts=tuple(allowed_hosts))
    app.state.hub = hub

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Any, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's own 422 echoes the offending input, and a NaN / Infinity there cannot be encoded
        return JSONResponse({"detail": json_safe(jsonable_encoder(exc.errors()))}, status_code=422)

    @app.exception_handler(SheetConflict)
    async def sheet_conflict(request: Any, exc: SheetConflict) -> JSONResponse:
        return JSONResponse(json_safe({"detail": exc.detail, "cues": exc.cues}), status_code=409)

    def locate(project_id: str) -> tuple[str, Path]:
        """(NFC id, folder) of an existing project; 400 for an invalid id, 404 when there is none."""
        try:
            pid = check_project_id(project_id)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        pdir = hub.project_dir(pid)
        if not (pdir / META_FILE).is_file():
            raise HTTPException(404, f"project {pid!r} not found")
        return pid, pdir

    def sheet_body(pid: str, text: str, duration: float | None, read_error: str | None = None) -> dict:
        body = sheet_json(parse_cues(text), hub.settings, duration, read_error=read_error)
        body.update(project_rev=hub.rev.get(pid, 0), undo_left=len(hub.undo.get(pid, ())))
        return json_safe(body)

    async def current_body(pid: str, pdir: Path) -> dict:
        text, err, dur = await run_in_threadpool(_read_cues_and_duration, pdir)
        return sheet_body(pid, text, dur, err)

    async def check_rev(pid: str, pdir: Path, rev: str | None, text: str) -> None:
        if rev is not None and rev != sheet_rev(text):
            raise SheetConflict("The cue file changed since you loaded it (edited elsewhere?): it was reloaded, "
                                "check it and try again", await current_body(pid, pdir))

    def need_cue(text: str, cue_id: int) -> Cue:
        cue = parse_cues(text).by_id(cue_id)
        if cue is None:
            raise HTTPException(404, f"there is no cue on line {cue_id} (the file may have changed: reload it)")
        return cue

    async def edit_cues(project_id: str, rev: str | None, change: Callable[[str], str], label: str, *,
                        replace: bool = False) -> tuple[str, dict]:
        """Read, check `rev`, apply `change` (CueEditError -> 400), save and announce; (id, response)."""
        pid, pdir = locate(project_id)
        async with hub.edit_lock:
            text, err, dur = await run_in_threadpool(_read_cues_and_duration, pdir)
            if err is not None and not replace:
                raise HTTPException(400, err)
            await check_rev(pid, pdir, rev, text)
            try:
                new = change(text)
            except CueEditError as e:
                raise HTTPException(400, str(e)) from None
            check_utf8(new, "the cue text")
            added = await hub.commit(pid, pdir, None if err else text, new, label)
            body = sheet_body(pid, new, dur)
        if added:
            await hub.lookup_new_hosts(added)
        body["targets_added"] = added
        return pid, body

    def settings_body() -> dict:
        return json_safe({**hub.settings.to_dict(), "targets_status": hub.sender.stats(), "rev": hub.settings_rev,
                          "warnings": list(hub.settings_warnings)})

    def need_session() -> Any:
        if hub.session is None:
            raise HTTPException(409, "no live session running")
        return hub.session

    def state() -> dict:
        return json_safe(hub.state_message())

    # -- pages ----------------------------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def console_page() -> FileResponse:
        page = STATIC_DIR / "console.html"
        if not page.is_file():
            raise HTTPException(404, "the console UI is not installed (server/static/console.html)")
        return FileResponse(page, headers={"Cache-Control": "no-cache"})

    # -- projects -------------------------------------------------------------------------------------
    @app.get("/api/projects")
    async def projects_list() -> list[dict]:
        def rows() -> list[dict]:
            return [project_summary(d, hub.rev.get(nfc(d.name), 0)) for d in list_projects(root)]
        return json_safe(await run_in_threadpool(rows))

    @app.post("/api/projects", status_code=201)
    async def project_create(audio: Annotated[UploadFile, File()], title: Annotated[str, Form()],
                             cues: Annotated[UploadFile | None, File()] = None,
                             cues_text: Annotated[str, Form()] = "",
                             vocals: Annotated[UploadFile | None, File()] = None,
                             style: Annotated[str, Form()] = "classic") -> dict:
        """A new project from a reference recording (any format ffmpeg reads), a cue file (or pasted cue
        text) and an optional vocal stem. The conversion runs in a worker thread."""
        title = title.strip()
        if not title:
            raise HTTPException(400, "the project title is empty")
        if style not in STYLES:
            raise HTTPException(400, f"style must be one of {', '.join(STYLES)}")
        cues_name = None
        if _given(cues):
            text = _decode_cue_upload(await cues.read())
            cues_name = _safe_name(cues.filename, "cues.txt")
        else:
            text = cues_text.removeprefix("﻿")
        updir = Path(await run_in_threadpool(tempfile.mkdtemp, prefix=UPLOAD_PREFIX, dir=root))
        try:
            audio_path = await run_in_threadpool(_save_upload, audio, updir / "audio", "audio")
            if audio_path.stat().st_size == 0:
                raise HTTPException(400, "the audio file is empty")
            vocals_path = None
            if _given(vocals):
                vocals_path = await run_in_threadpool(_save_upload, vocals, updir / "vocals", "vocals")
            try:
                pdir, warnings = await run_in_threadpool(functools.partial(
                    create_project, root, title=title, audio=audio_path, cues_text=text, vocals=vocals_path,
                    style=style, cues_name=cues_name))
            except (ValueError, FileNotFoundError, AudioError) as e:
                raise HTTPException(400, str(e)) from e
        finally:
            await run_in_threadpool(shutil.rmtree, updir, True)
        pid = nfc(pdir.name)
        added = await hub.merge_targets(parse_cues(text).targets)
        if added:
            await hub.lookup_new_hosts(added)
        hub.bump(pid, text)
        summary = await run_in_threadpool(project_summary, pdir, hub.rev.get(pid, 0))
        return json_safe({"project": summary, "warnings": warnings, "targets_added": added})

    @app.get("/api/projects/{project_id}")
    async def project_detail(project_id: str) -> dict:
        """The project, its cue sheet and its show logs. The project becomes the console's focus: its
        cue file is watched for changes made on disk."""
        pid, pdir = locate(project_id)
        hub.watch_from(pid, file_signature(cues_path(pdir)))      # before the file is read
        hub.focus = pid
        try:
            meta = await run_in_threadpool(load_meta, pdir)
        except (OSError, ValueError) as e:
            raise HTTPException(500, f"cannot read project.json: {e}") from e
        text, err, _ = await run_in_threadpool(_read_cues_and_duration, pdir)
        if err is None:
            # a cue file that did not come through this server (oscsender add, copied in) declares targets
            # the settings may not have yet
            added = await hub.merge_targets(parse_cues(text).targets)
            if added:
                await hub.lookup_new_hosts(added)
        logs = await run_in_threadpool(_log_names, pdir)
        return json_safe({"project": {**meta.to_dict(), "stems": stems_present(pdir)},
                          "cues": sheet_body(pid, text, meta.duration, err), "logs": logs,
                          "rev": hub.rev.get(pid, 0)})

    @app.delete("/api/projects/{project_id}")
    async def project_delete(project_id: str) -> dict:
        """Move the project to the trash (projects/.trash/); 409 while it is the live session's."""
        pid, _ = locate(project_id)
        async with hub.op_lock:
            if hub.live_project == pid:
                raise HTTPException(409, "stop the live session of this project before deleting it")
            async with hub.edit_lock:
                try:
                    dst = await run_in_threadpool(trash_project, root, pid)
                except FileNotFoundError as e:
                    raise HTTPException(404, str(e)) from e
                hub.forget(pid)
        return {"id": pid, "trashed": dst.name}

    @app.get("/api/projects/{project_id}/audio/{stem}")
    async def project_audio(project_id: str, stem: str) -> FileResponse:
        if stem not in STEM_FILES:
            raise HTTPException(404, f"unknown stem {stem!r}")
        pid, pdir = locate(project_id)
        path = stem_path(pdir, stem)
        if not path.is_file():
            raise HTTPException(404, f"project {pid!r} has no {stem} stem")
        return FileResponse(path, media_type="audio/wav")

    @app.get("/api/projects/{project_id}/waveform")
    async def project_waveform(project_id: str, points: Annotated[int, Query(ge=100, le=20000)] = 2000,
                               stem: str = "mix") -> dict:
        _, pdir = locate(project_id)
        try:
            return json_safe(await run_in_threadpool(waveform, pdir, stem, points))
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/projects/{project_id}/cues.txt")
    async def project_cues_file(project_id: str) -> Response:
        pid, pdir = locate(project_id)
        path = cues_path(pdir)
        headers = {"Cache-Control": "no-cache"}
        if not path.is_file():
            return Response(b"", media_type="text/plain; charset=utf-8", headers={
                **headers, "Content-Disposition": 'attachment; filename="cues.txt"'})
        return FileResponse(path, media_type="text/plain; charset=utf-8", filename=f"{pid}-cues.txt",
                            headers=headers)

    @app.get("/api/projects/{project_id}/logs/{name}")
    async def project_log(project_id: str, name: str) -> FileResponse:
        """A show log (CSV) named in the project's "logs"."""
        _, pdir = locate(project_id)
        path = pdir / "logs" / name
        if not LOG_NAME_RE.fullmatch(name) or ".." in name or not path.is_file():
            raise HTTPException(404, f"no show log {name!r}")
        return FileResponse(path, media_type="text/csv; charset=utf-8", filename=name)

    @app.put("/api/projects/{project_id}/style")
    async def project_style(project_id: str, req: StyleReq) -> dict:
        """Save the project's tracking style and switch a running session of this project to it.
        `session`: whether a live session was switched."""
        pid, pdir = locate(project_id)
        async with hub.edit_lock:
            meta = await run_in_threadpool(load_meta, pdir)
            if meta.style != req.style:
                meta.style = req.style
                await run_in_threadpool(save_meta, pdir, meta)
                hub.bump(pid)
            sess = hub.session if hub.live_project == pid else None
            if sess is not None:
                await run_in_threadpool(sess.set_style, req.style)
                if hub.session_req is not None:
                    hub.session_req["style"] = req.style
        return {"id": pid, "style": req.style, "rev": hub.rev.get(pid, 0), "session": sess is not None}

    @app.post("/api/projects/{project_id}/vocals")
    async def project_vocals(project_id: str, vocals: Annotated[UploadFile, File()]) -> dict:
        """Add or replace the vocal stem (converted and fitted to the mix length; warnings say how)."""
        pid, pdir = locate(project_id)
        if not _given(vocals):
            raise HTTPException(400, "no vocals file")
        updir = Path(await run_in_threadpool(tempfile.mkdtemp, prefix=UPLOAD_PREFIX, dir=root))
        try:
            path = await run_in_threadpool(_save_upload, vocals, updir, "vocals")
            if path.stat().st_size == 0:
                raise HTTPException(400, "the vocals file is empty")
            async with hub.edit_lock:
                try:
                    warnings = await run_in_threadpool(set_vocals, pdir, path)
                except (ValueError, FileNotFoundError, AudioError) as e:
                    raise HTTPException(400, str(e)) from e
                hub.bump(pid)
        finally:
            await run_in_threadpool(shutil.rmtree, updir, True)
        summary = await run_in_threadpool(project_summary, pdir, hub.rev.get(pid, 0))
        return json_safe({"project": summary, "warnings": warnings})

    # -- cues -----------------------------------------------------------------------------------------
    @app.get("/api/projects/{project_id}/cues")
    async def cues_get(project_id: str) -> dict:
        pid, pdir = locate(project_id)
        return await current_body(pid, pdir)

    @app.put("/api/projects/{project_id}/cues")
    async def cues_put(project_id: str, req: CueText) -> dict:
        """Replace the whole cue file (saved even when it has issues: they come back in the sheet)."""
        new = req.text.removeprefix("﻿")
        return (await edit_cues(project_id, req.rev, lambda _old: new, "cue file", replace=True))[1]

    @app.post("/api/projects/{project_id}/cues/upload")
    async def cues_upload(project_id: str, cues: Annotated[UploadFile, File()],
                          rev: Annotated[str | None, Form()] = None) -> dict:
        new = _decode_cue_upload(await cues.read())
        return (await edit_cues(project_id, rev, lambda _old: new, "cue file", replace=True))[1]

    @app.post("/api/projects/{project_id}/cues/add")
    async def cues_add(project_id: str, req: CueAdd) -> dict:
        added: list[int] = []

        def change(text: str) -> str:
            new, line = add_cue(text, req.time, req.message)
            added.append(line)
            return new
        _, body = await edit_cues(project_id, req.rev, change, "added cue")
        body["added"] = added[0]
        return body

    @app.post("/api/projects/{project_id}/cues/undo")
    async def cues_undo(project_id: str) -> dict:
        """Revert the project's last cue edit (history in memory, UNDO_DEPTH edits per project, lost on
        restart). 409 when there is nothing to undo, or when the cue file changed on disk since that edit
        (the history is then dropped)."""
        pid, pdir = locate(project_id)
        async with hub.edit_lock:
            hist = hub.undo.get(pid)
            if not hist:
                raise HTTPException(409, "Nothing to undo")
            text, err, dur = await run_in_threadpool(_read_cues_and_duration, pdir)
            step = hist[-1]
            if err is not None or text != step.after:
                hist.clear()
                raise HTTPException(409, "The cue file changed on disk since that edit, so it cannot be undone")
            await run_in_threadpool(write_cues, pdir, step.before)
            hub.note_saved(pid, pdir, step.before)
            hist.pop()
            sheet = parse_cues(step.before)
            hub.bump(pid, step.before)
            await hub.push_cues(pid, sheet.cues)
            added = await hub.merge_targets(sheet.targets)
            body = sheet_body(pid, step.before, dur)
        if added:
            await hub.lookup_new_hosts(added)
        body.update(undone=step.label, targets_added=added)
        return body

    @app.patch("/api/projects/{project_id}/cues/{cue_id}")
    async def cue_patch(project_id: str, cue_id: int, req: CuePatch) -> dict:
        given = [k for k in ("time", "shift", "message") if getattr(req, k) is not None]
        if len(given) != 1:
            raise HTTPException(422, "give exactly one of 'time', 'shift' or 'message'")

        def change(text: str) -> str:
            need_cue(text, cue_id)
            if req.time is not None:
                return set_cue_time(text, cue_id, req.time)
            if req.shift is not None:
                return shift_cue(text, cue_id, req.shift)
            return set_cue_message(text, cue_id, req.message or "")
        return (await edit_cues(project_id, req.rev, change, f"cue {cue_id} {given[0]}"))[1]

    @app.delete("/api/projects/{project_id}/cues/{cue_id}")
    async def cue_delete(project_id: str, cue_id: int, rev: str | None = None) -> dict:
        def change(text: str) -> str:
            need_cue(text, cue_id)
            return delete_cue(text, cue_id)
        return (await edit_cues(project_id, rev, change, f"deleted cue {cue_id}"))[1]

    @app.post("/api/projects/{project_id}/cues/{cue_id}/fire")
    async def cue_fire(project_id: str, cue_id: int, req: FireReq | None = None) -> dict:
        """Send one cue now, armed or not. A manual fire of the live project's cue goes through the
        session (it is in the fired log and the show log); a preview send, or any fire without a live
        session of this project, goes straight out through the sender."""
        req = req or FireReq()
        pid, pdir = locate(project_id)
        text, sheet = await hub.cached_sheet(pid, pdir)
        await check_rev(pid, pdir, req.rev, text)
        cue = sheet.by_id(cue_id)
        if cue is None:
            raise HTTPException(404, f"there is no cue on line {cue_id} (the file may have changed: reload it)")
        if req.source == "manual":
            await hub.retry_target(cue.target or hub.settings.default_target)
        sess = hub.session if hub.live_project == pid else None
        if req.source == "manual" and sess is not None:
            await hub.sync_live_cues()
            try:
                rec = _record_dict(await run_in_threadpool(sess.fire_cue, cue_id))
            except (LookupError, ValueError) as e:
                raise HTTPException(404, f"the live session has no cue {cue_id}: {e}") from e
            return json_safe({"ok": bool(rec.get("sent")), "error": rec.get("error"), "target": rec.get("target"),
                              "address": rec.get("address", cue.address), "logged": True, "seq": rec.get("seq")})
        target = cue.target or hub.settings.default_target
        res = hub.sender.send(target, cue.address, cue.args)
        return {"ok": res.ok, "error": res.error, "target": target, "address": cue.address, "logged": False}

    # -- settings and targets -------------------------------------------------------------------------
    @app.get("/api/settings")
    async def settings_get() -> dict:
        return settings_body()

    @app.put("/api/settings")
    async def settings_put(data: Annotated[Any, Body()]) -> dict:
        """Partial update: any of targets (the whole list), default_target, armed, skip_policy, catchup_s,
        input_latency_ms. 400 with the reason when the result is not valid (nothing changes then). The
        hosts of new or changed targets are looked up (outside the locks) before the reply; enabled targets
        without an address are looked up again in the background."""
        async with hub.settings_lock:
            try:
                new = settings_from_dict(data, hub.settings)
            except (ValueError, TypeError) as e:
                raise HTTPException(400, str(e)) from e
            check_utf8(json.dumps(new.to_dict(), ensure_ascii=False), "a setting")
            old = hub.settings
            if new != old:
                await hub.apply_settings(new)
        hub.start_lookups(hub.hosts.missing(hub.settings.targets))
        before = {(t.host, t.port) for t in old.targets}
        await hub.lookup_new_hosts(t.name for t in new.targets if (t.host, t.port) not in before)
        return settings_body()

    @app.post("/api/session/arm")
    async def session_arm(req: OnReq) -> dict:
        """OSC output on (armed) or off (dry run: cues are logged, not sent); same as PUT armed."""
        async with hub.settings_lock:
            if hub.settings.armed != req.on:
                new = hub.settings.copy()
                new.armed = req.on
                await hub.apply_settings(new)
        return settings_body()

    @app.post("/api/targets/{name}/test")
    async def target_test(name: str, req: TestReq | None = None) -> dict:
        """Send `message`, the target's test message, or /oscsender/test to one target now."""
        t = hub.settings.target(name)
        if t is None:
            raise HTTPException(404, f"unknown target {name!r}")
        text = ((req.message if req else None) or "").strip() or t.test_message.strip() or TEST_ADDRESS
        try:
            address, args = message_from_text(text)
        except ValueError as e:
            raise HTTPException(400, f"test message: {e}") from e
        await hub.retry_target(name)
        res = hub.sender.send(name, address, args)
        return {"ok": res.ok, "error": res.error, "target": name, "address": address}

    # -- latency --------------------------------------------------------------------------------------
    @app.post("/api/latency/echo")
    async def latency_echo(req: EchoReq) -> dict:
        """Network round trips to a responder (`oscsender echo` on the receiving machine, or a receiver
        that answers `message`); `estimate_ms` is half the median round trip."""
        name = req.target or None
        if name is not None:
            t = hub.settings.target(name)
            if t is None:
                raise HTTPException(404, f"unknown target {name!r}")
            host, port = t.host, t.port
        elif req.host and req.host.strip() and req.port:
            host, port = req.host.strip(), req.port
        else:
            raise HTTPException(422, "give a target, or a host and a port")
        message = None
        if req.message and req.message.strip():
            try:
                message = message_from_text(req.message)
            except ValueError as e:
                raise HTTPException(400, f"message: {e}") from e
        try:
            rep = await run_in_threadpool(functools.partial(
                measure_echo, host, port, count=req.count, interval_s=req.interval_ms / 1000.0,
                timeout_s=req.timeout_ms / 1000.0, message=message, listen_port=req.listen_port, target=name))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return rep.to_dict()

    @app.post("/api/latency/loopback")
    async def latency_loopback(req: LoopbackReq) -> dict:
        """Delay from sending the target's test message to its sound at an audio input. 409 while a
        session listens to an audio device (the input may be held exclusively, and the show comes first)."""
        if hub.device_session():
            raise HTTPException(409, "stop the live session before measuring with the audio input")
        t = hub.settings.target(req.target)
        if t is None:
            raise HTTPException(404, f"unknown target {req.target!r}")
        text = (req.message or "").strip() or t.test_message.strip() or TEST_ADDRESS
        try:
            address, args = message_from_text(text)
        except ValueError as e:
            raise HTTPException(400, f"test message: {e}") from e
        if hub.loopback_lock.locked():
            raise HTTPException(409, "a loopback measurement is already running")
        if hub.rescanning:
            raise HTTPException(409, "the audio devices are being rescanned: try again")
        async with hub.loopback_lock:
            try:
                capture = AudioCapture(req.device, req.channels, clock=hub.sender.clock)
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            send = target_send(hub.sender, req.target, address, args)
            rep = await run_in_threadpool(functools.partial(
                measure_loopback, send, capture, count=req.count, interval_s=req.interval_ms / 1000.0,
                threshold_db=req.threshold_db, clock=hub.sender.clock, target=req.target))
        return rep.to_dict()

    # -- devices --------------------------------------------------------------------------------------
    @app.get("/api/devices")
    async def devices_list(refresh: bool = False) -> list[dict]:
        """`refresh` re-initialises PortAudio, which closes every open stream: refused (409) while a session
        exists or is being started / stopped, or a loopback measurement runs; a session start waits for it."""
        if not refresh:
            return await list_devices(False)
        if hub.session is not None or hub.op_lock.locked() or hub.loopback_lock.locked():
            raise HTTPException(409, "stop the session (and any measurement) before rescanning devices")
        async with hub.op_lock:
            hub.rescanning = True
            try:
                return await list_devices(True)
            finally:
                hub.rescanning = False

    async def list_devices(refresh: bool) -> list[dict]:
        try:
            return json_safe(await run_in_threadpool(hub.devices, refresh))
        except Exception as e:
            raise HTTPException(503, f"audio devices unavailable: {e}") from e

    # -- session --------------------------------------------------------------------------------------
    @app.get("/api/session")
    async def session_get() -> dict:
        return json_safe({"session": hub.session_req, "project_id": hub.live_project, "state": hub.state_message()})

    @app.post("/api/session/start")
    async def session_start(req: SessionStart) -> dict:
        """Start (or replace) the live session. Make before break: the running session keeps going
        until the new one has been built and started, so a rejected request (bad channel, missing
        stem, ...) leaves the show as it was."""
        pid, pdir = locate(req.project_id)
        if req.start_time is not None and req.start_cue is not None:
            raise HTTPException(422, "give at most one of 'start_time' or 'start_cue'")
        if req.mode == "vocal" and not stem_path(pdir, "vocals").is_file():
            raise HTTPException(400, "vocal mode needs the project's vocal stem: add one first")
        if req.input.kind == "device" and hub.loopback_lock.locked():
            raise HTTPException(409, "a loopback measurement is using the audio input")
        async with hub.op_lock:
            sig = file_signature(cues_path(pdir))          # before the cue file is read
            rev0, settings_rev0 = hub.rev.get(pid, 0), hub.settings_rev
            text, err = await run_in_threadpool(load_text, pdir)
            if err is not None:
                raise HTTPException(400, err)
            style0 = await run_in_threadpool(_project_style, pdir)
            sheet = parse_cues(text)
            start_time = 0.0
            if req.start_cue is not None:
                await check_rev(pid, pdir, req.rev, text)        # the id means another line now: 409
                cue = sheet.by_id(req.start_cue)
                if cue is None:
                    raise HTTPException(404, f"there is no cue on line {req.start_cue}")
                start_time = cue.time
            elif req.start_time is not None:
                start_time = float(req.start_time)
            added = await hub.merge_targets(sheet.targets)     # declared targets the settings lack
            if added:
                await hub.lookup_new_hosts(added)
            ctx = SessionContext(root, list(sheet.cues), hub.sender, hub.settings.copy(), start_time, hub.clock)
            try:
                sess = await run_in_threadpool(hub.session_factory, pdir, req, ctx)
            except Exception as e:
                raise _start_error(e) from e
            old = hub.session
            try:
                try:
                    await run_in_threadpool(sess.start)
                except Exception:
                    if old is None or req.input.kind != "device" or not hub.device_session():
                        raise
                    # the running session may hold the interface exclusively: free it and retry once, with
                    # a new session (the failed one has closed its show log)
                    log.info("starting the new input failed while the old one runs; retrying after stopping it")
                    await run_in_threadpool(_quiet_stop, sess)
                    await hub.stop_session()
                    old = None
                    sess = await run_in_threadpool(hub.session_factory, pdir, req,
                                                   dataclasses.replace(ctx, settings=hub.settings.copy()))
                    await run_in_threadpool(sess.start)
            except Exception as e:
                await run_in_threadpool(_quiet_stop, sess)
                raise _start_error(e) from e
            hub.session, hub.session_req, hub.session_project = sess, req.model_dump(), pid
            hub.watch_from(pid, sig)
            if old is not None:
                await run_in_threadpool(_quiet_stop, old)
            # changes saved while the session was being built never reached it
            if hub.settings_rev != settings_rev0:
                await hub.settings_to_session(sess, hub.settings, None)
            if hub.rev.get(pid, 0) != rev0:
                text, err = await run_in_threadpool(load_text, pdir)
                if err is None:
                    await hub.push_cues(pid, parse_cues(text).cues)
                style = await run_in_threadpool(_project_style, pdir)
                if style is not None and style != style0 and style != getattr(sess, "style", req.style or style0):
                    try:                                  # PUT .../style landed during the build
                        await run_in_threadpool(sess.set_style, style)
                        hub.session_req["style"] = style
                    except Exception:
                        log.exception("switching the new session to the project's style failed")
        return state()

    @app.post("/api/session/stop")
    async def session_stop() -> dict:
        async with hub.op_lock:
            await hub.stop_session()
        return state()

    @app.post("/api/session/hold")
    async def session_hold(req: OnReq | None = None) -> dict:
        await run_in_threadpool(need_session().hold, True if req is None else req.on)
        return state()

    @app.post("/api/session/jump")
    async def session_jump(req: JumpReq) -> dict:
        if (req.time is None) == (req.cue_id is None):
            raise HTTPException(422, "give exactly one of 'time' or 'cue_id'")
        sess = need_session()
        if req.project_id is not None and req.project_id != hub.live_project:
            raise HTTPException(409, f"{req.project_id!r} is not the live project ({hub.live_project!r}); "
                                     "start a session of it first")
        if req.cue_id is not None and req.rev is not None and hub.live_project is not None:
            pid = hub.live_project
            pdir = hub.project_dir(pid)
            text, _ = await hub.cached_sheet(pid, pdir)
            await check_rev(pid, pdir, req.rev, text)        # the id means another line now: 409
        try:
            if req.cue_id is not None:
                await hub.sync_live_cues()
                await run_in_threadpool(sess.jump_cue, req.cue_id)
            else:
                await run_in_threadpool(sess.jump_time, max(0.0, float(req.time)))  # type: ignore[arg-type]
        except (LookupError, ValueError) as e:
            raise HTTPException(404, f"cannot jump: {e}") from e
        return state()

    @app.post("/api/session/nudge")
    async def session_nudge(req: NudgeReq) -> dict:
        await run_in_threadpool(need_session().nudge, req.dt)
        return state()

    @app.post("/api/session/transpose")
    async def session_transpose(req: TransposeReq) -> dict:
        await run_in_threadpool(need_session().set_transpose, req.semitones)
        return state()

    @app.post("/api/session/autokey")
    async def session_autokey(req: OnReq) -> dict:
        await run_in_threadpool(need_session().set_auto_key, req.on)
        return state()

    @app.post("/api/session/latency")
    async def session_latency(req: InputLatencyReq) -> dict:
        """The tracker's input latency compensation (ms): saved as the `input_latency_ms` setting and
        applied to a running session."""
        async with hub.settings_lock:
            if hub.settings.input_latency_ms != req.ms:
                new = hub.settings.copy()
                new.input_latency_ms = req.ms
                await hub.apply_settings(validate_settings(new))
        return state()

    @app.post("/api/session/style")
    async def session_style(req: StyleReq) -> dict:
        """Switch the running session's tracking style live (project.json is not changed: PUT
        /api/projects/{id}/style saves it)."""
        await run_in_threadpool(need_session().set_style, req.style)
        if hub.session_req is not None:
            hub.session_req["style"] = req.style
        return state()

    # -- websocket ------------------------------------------------------------------------------------
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        with contextlib.suppress(WebSocketDisconnect):
            await hub.serve(ws)

    app.mount("/static", RevalidatingStatic(directory=STATIC_DIR, check_dir=False), name="static")
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
    """The URL to open; for a wildcard host also the ones other devices (a tablet) can reach."""
    wildcard = host in ("0.0.0.0", "::", "")
    shown = "127.0.0.1" if wildcard else (f"[{host}]" if ":" in host else host)
    out = [f"console   http://{shown}:{port}/"]
    if wildcard:
        names = [*lan_addresses(), *sorted(n for n in machine_host_names() if n.endswith(".local"))]
        if names:
            out.append("on another device in the same network (a tablet), open one of:")
            out += [f"          http://{n}:{port}/" for n in names]
    return "\n".join(out)


def cli(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="oscsender serve", description="Run the console web server")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open the console from other devices")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--projects-dir", type=Path, default=None,
                    help=f"projects root (default: {config.PROJECTS_DIR})")
    ap.add_argument("--log-level", default="info")
    ap.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                    help="extra host name the console is opened by (repeatable); loopback, IP addresses and "
                         "this machine's name always work")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:              # argparse: --help -> 0, usage error -> 2
        return int(e.code or 0)

    import uvicorn

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(args.projects_dir, allowed_hosts=args.allow_host)
    print(banner(args.host, args.port), flush=True)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("warning: no authentication; anyone on this network can control the session and send OSC",
              flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, workers=1)
    return 0
