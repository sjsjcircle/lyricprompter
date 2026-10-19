"""LiveSession: live audio source -> FeatureExtractor -> LiveTracker, with thread-safe operator
controls and a JSON-able `snapshot()` for the WebSocket broadcaster.

Audio is processed on the source's worker thread; controls arrive from the server's event loop.
One re-entrant lock serialises tracker access between the two.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import FPS, SONGS_DIR
from ..songfile import load_song, stem_path
from ..songfile import song_dir as song_dir_of
from ..text.model import STYLES, Line, Song

log = logging.getLogger(__name__)

MODES = ("mix", "vocal")


# --------------------------------------------------------------------------- dependency factories


@dataclass
class SessionDeps:
    """Constructors for the feature / tracking / timeline modules (overridable in tests)."""
    feature_config: Callable[[str, float], Any]          # (mode, tuning) -> FeatureConfig
    extractor: Callable[[Any], Any]                      # cfg -> FeatureExtractor
    load_ref: Callable[[Path, str, Any], Any]            # (song_dir, stem, cfg) -> FeatureFrames
    tracker_params: Callable[[str], Any]                 # mode -> TrackerParams
    tracker: Callable[..., Any]                          # (ref, fps, params, *, ref_active, line_starts)
    timeline: Callable[[list[Line]], Any]                # lines -> Timeline
    ref_activity: Callable[[np.ndarray, float, Any], np.ndarray] | None = None   # (rms_db, fps, params)


def _feature_config(mode: str, tuning: float) -> Any:
    from ..features.extractor import FeatureConfig

    tuning = float(tuning or 0.0)
    for_mode = getattr(FeatureConfig, "for_mode", None)
    if callable(for_mode):
        try:
            cfg = for_mode(mode, tuning=tuning)
        except TypeError:
            cfg = for_mode(mode)
    else:
        cfg = FeatureConfig(kind=mode)
    return dataclasses.replace(cfg, tuning=tuning)


def _extractor(cfg: Any) -> Any:
    from ..features.extractor import FeatureExtractor

    return FeatureExtractor(cfg)


def _load_ref(sdir: Path, stem: str, cfg: Any) -> Any:
    from ..features.extractor import load_ref_features

    return load_ref_features(sdir, stem, cfg)


def _tracker_params(mode: str) -> Any:
    from ..tracking.tracker import TrackerParams

    for_mode = getattr(TrackerParams, "for_mode", None)
    return for_mode(mode) if callable(for_mode) else TrackerParams(mode=mode)


def with_style(params: Any, mode: str, style: str) -> Any:
    """`params` (a TrackerParams) with the tracking style's preset fields; other params are left alone."""
    if not (dataclasses.is_dataclass(params) and hasattr(params, "style")):
        return params
    try:
        from ..tracking.tracker import style_fields

        fields = style_fields(mode, style)
    except (ImportError, ValueError, KeyError):
        fields = {"style": style}
    return dataclasses.replace(params, **fields)


def _tracker(ref: Any, fps: float, params: Any, **kw: Any) -> Any:
    from ..tracking.tracker import LiveTracker

    return LiveTracker(ref, fps, params, **kw)


def _timeline(lines: list[Line]) -> Any:
    from ..timeline import Timeline

    return Timeline(lines)


def _ref_activity(rms_db: np.ndarray, fps: float, params: Any) -> np.ndarray:
    try:
        from ..tracking.tracker import ref_activity
    except ImportError:
        return activity_from_rms(rms_db, fps)
    kw = {k: getattr(params, f"ref_{k}") for k in ("rel_db", "min_gap_s", "min_on_s") if hasattr(params, f"ref_{k}")}
    return ref_activity(rms_db, fps, **kw)


def default_deps() -> SessionDeps:
    return SessionDeps(_feature_config, _extractor, _load_ref, _tracker_params, _tracker, _timeline, _ref_activity)


# --------------------------------------------------------------------------- helpers


def reference_stem(sdir: Path, mode: str, prefer_lead: bool = False) -> str:
    """mix -> "mix"; vocal -> "lead" when requested and present, else "vocals"."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "mix":
        return "mix"
    return "lead" if prefer_lead and stem_path(sdir, "lead").is_file() else "vocals"


def activity_from_rms(rms_db: np.ndarray, fps: float = FPS, rel_db: float = -35.0,
                      dilate_s: float = 0.3) -> np.ndarray:
    """Fallback vocal-activity mask per reference frame (used when the tracker module provides no
    `ref_activity`): rms above (p99 + rel_db), dilated by `dilate_s` like prep's stem gating."""
    rms = np.asarray(rms_db, dtype=np.float64)
    finite = rms[np.isfinite(rms)]
    if finite.size == 0:
        return np.zeros(len(rms), dtype=bool)
    act = rms > np.percentile(finite, 99) + rel_db
    k = round(dilate_s * fps)
    if k > 0 and act.any():
        act = np.convolve(act.astype(np.int32), np.ones(2 * k + 1, dtype=np.int32), mode="same") > 0
    return act


def line_starts(lines: list[Line]) -> list[float]:
    return [float(ln.start) for ln in lines if ln.start is not None]


def _num(x: Any, nd: int = 3) -> float | None:
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def json_safe(x: Any) -> Any:
    """`x` with every non-finite float (NaN / inf) replaced by None, recursively (dicts, lists,
    tuples). Browsers' JSON.parse rejects NaN / Infinity, so one such value would make a whole
    state message unreadable."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.floating):
        return json_safe(float(x))
    return x


def package_id(song_dir: Path) -> str:
    """A song's id is its package folder name (NFC; what the server and the UIs key everything
    by). song.json's meta.id is informational: a copied or renamed package keeps the old one."""
    return unicodedata.normalize("NFC", Path(song_dir).absolute().name)


def file_signature(path: Path) -> tuple[int, int] | None:
    """(mtime_ns, size) of a file, None when it is missing: tells a file replaced on disk."""
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


REPREPPED_TEXT = ("This song was prepared again from a different recording while it was live: the prompters "
                  "show its new lines, but tracking still follows the previous recording. Start the session "
                  "again to follow the new one.")


# --------------------------------------------------------------------------- session


class LiveSession:
    """Track one live source against one prepared song.

    `start_line` / `start_time` choose where the tracker anchors (default: the song start).
    `style` is the tracking style ("classic" | "pop"; None: the song's SongMeta.style), applied to `params`.
    `on_update(session)` is called on the audio worker thread after every tracker step (used by the
    `simulate` CLI to print line changes without polling).
    """

    def __init__(self, song_dir: Path, mode: str, source: Any, params: Any | None = None, *,
                 prefer_lead: bool = False, start_line: int | None = None, start_time: float | None = None,
                 style: str | None = None, deps: SessionDeps | None = None,
                 on_update: Callable[[LiveSession], None] | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if style is not None and style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}, got {style!r}")
        self.song_dir = Path(song_dir)
        self.song: Song = load_song(self.song_dir)
        self.style = style or (self.song.meta.style if self.song.meta.style in STYLES else "classic")
        self.song_id = package_id(self.song_dir) or self.song.meta.id
        self.mode, self.source, self.on_update = mode, source, on_update
        self.deps = deps or default_deps()
        self.stem = reference_stem(self.song_dir, mode, prefer_lead)
        if not stem_path(self.song_dir, self.stem).is_file():
            raise FileNotFoundError(f"reference stem missing: {stem_path(self.song_dir, self.stem)}")

        # the mix defines the song's time axis (prep keeps a byte-identical mix untouched on a re-prep)
        self._axis_sig = file_signature(stem_path(self.song_dir, "mix"))
        self.warning: str | None = None

        self.feature_cfg = self.deps.feature_config(mode, self.song.meta.tuning)
        self.fps = float(getattr(self.feature_cfg, "sr", 16000)) / float(getattr(self.feature_cfg, "hop", 640))
        if params is None:
            params = self.deps.tracker_params(mode)
        elif getattr(params, "mode", mode) != mode and dataclasses.is_dataclass(params):
            params = dataclasses.replace(params, mode=mode)
        self.params = with_style(params, mode, self.style)
        self.ref = self.deps.load_ref(self.song_dir, self.stem, self.feature_cfg)
        self.ref_active = None
        if mode == "vocal":
            act = self.deps.ref_activity or (lambda r, f, _p: activity_from_rms(r, f))
            self.ref_active = np.asarray(act(self.ref.rms_db, self.fps, self.params), dtype=bool)
        self.timeline = self.deps.timeline(self.song.lines)
        self.tracker = self.deps.tracker(self.ref, self.fps, self.params, ref_active=self.ref_active,
                                         line_starts=line_starts(self.song.lines))
        self.extractor = self.deps.extractor(self.feature_cfg)

        if start_line is not None:
            start_time = self.timeline.line_start(self._check_line(start_line))
        self.start_time = max(0.0, float(start_time or 0.0))
        self.transpose = 0
        self.latency = float(getattr(self.params, "latency_s", 0.0) or 0.0)
        self.started_at: float | None = None
        self.error: str | None = None
        self._errors = 0
        self._lock = threading.RLock()
        self._state = self.tracker.state
        self._state_t = time.time()
        self._running = False
        self._stopped = False

    # -- audio path (source worker thread) --------------------------------------------------------
    def _on_audio(self, y: np.ndarray) -> None:
        with self._lock:
            if not self._running:
                return
            try:
                frames = self.extractor.push(y)
                if len(frames.feats):
                    self._state = self.tracker.push(frames)
                    self._state_t = time.time()
                else:
                    return
            except Exception as e:
                self._errors += 1
                self.error = f"{type(e).__name__}: {e}"
                if self._errors == 1 or self._errors % 250 == 0:
                    log.exception("tracking step failed (%d errors)", self._errors)
                return
        if self.on_update is not None:
            self.on_update(self)

    # -- lifecycle ---------------------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._running or self._stopped:
                raise RuntimeError("session already started")
            self.extractor.reset()
            self.tracker.start(self.start_time)
            self._state, self._state_t = self.tracker.state, time.time()
            self._running = True
            self.started_at = time.time()
        try:
            self.source.start(self._on_audio)
        except Exception:
            with self._lock:
                self._running = False
            raise

    def stop(self) -> None:
        # Stop the source first and without the lock: its worker may be waiting for it.
        with self._lock:
            if self._stopped:
                return
            self._running = False
            self._stopped = True
        try:
            self.source.stop()
        finally:
            with self._lock:
                self.tracker.stop()
                self._state, self._state_t = self.tracker.state, time.time()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def source_finished(self) -> bool:
        return bool(getattr(self.source, "finished", False))

    def align_source_start(self, preroll: float = 0.0) -> float:
        """Start a file source `preroll` seconds before the tracker's anchor and anchor there too.

        Only for a file on the reference's own time axis (a stem of this song), e.g. a rehearsal
        from line N. Call before `start()`. Returns the new start time.
        """
        if self._running or self._stopped:
            raise RuntimeError("align_source_start must be called before start()")
        if not hasattr(self.source, "start_s"):
            raise TypeError(f"{type(self.source).__name__} has no start offset")
        t = max(0.0, self.start_time - max(0.0, float(preroll)))
        self.source.start_s = t
        self.start_time = t
        return t

    # -- operator controls (any thread) -----------------------------------------------------------
    def _refresh(self) -> None:
        self._state, self._state_t = self.tracker.state, time.time()

    def _check_line(self, idx: int) -> int:
        idx, n = int(idx), len(self.song.lines)
        if not 0 <= idx < n:
            raise IndexError(f"line {idx} out of range (the song has {n} lines)")
        return idx

    def jump_line(self, idx: int) -> None:
        self.jump_time(self.timeline.line_start(self._check_line(idx)))

    def jump_time(self, ref_time: float) -> None:
        ref_time = float(ref_time)
        if not math.isfinite(ref_time):
            raise ValueError(f"jump time must be finite, got {ref_time}")
        with self._lock:
            self.tracker.jump(max(0.0, ref_time))
            self._refresh()

    def hold(self, on: bool) -> None:
        with self._lock:
            self.tracker.hold(bool(on))
            self._refresh()

    def nudge(self, dt: float) -> None:
        with self._lock:
            self.tracker.nudge(float(dt))
            self._refresh()

    def set_transpose(self, k: int) -> None:
        """Operator key: also turns automatic key detection off."""
        with self._lock:
            self.tracker.set_transpose(int(k))
            self.transpose = int(k)
            self._refresh()

    def set_auto_key(self, on: bool) -> None:
        with self._lock:
            setter = getattr(self.tracker, "set_auto_key", None)
            if callable(setter):
                setter(bool(on))
            self._refresh()

    def set_style(self, style: str) -> None:
        """Switch the tracking style of the running session (the tracker goes on from where it is)."""
        if style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}, got {style!r}")
        with self._lock:
            setter = getattr(self.tracker, "set_style", None)
            if callable(setter):
                setter(style)
            elif hasattr(self.params, "style"):
                self.params = with_style(self.params, self.mode, style)
            self.style = style
            self._refresh()

    def set_latency(self, s: float) -> None:
        """Output latency compensation (seconds; the tracker adds tempo * s to ref_time)."""
        s = float(s)
        if not math.isfinite(s):
            raise ValueError(f"latency must be finite, got {s}")
        with self._lock:
            setter = getattr(self.tracker, "set_latency", None)
            if callable(setter):
                setter(s)
            else:
                tparams = getattr(self.tracker, "params", self.params)
                tparams.latency_s = s
                if tparams is not self.params and hasattr(self.params, "latency_s"):
                    self.params.latency_s = s
            self.latency = s

    def reload_lines(self, song: Song | None = None) -> None:
        """Pick up edited line timings (display cursor only; the tracker keeps its line starts).

        The reference features stay those loaded at start: when the package was prepared again from
        another recording (its mix changed), `warning` asks the operator to restart the session."""
        song = song or load_song(self.song_dir)
        timeline = self.deps.timeline(song.lines)
        axis = file_signature(stem_path(self.song_dir, "mix"))
        with self._lock:
            self.song, self.timeline = song, timeline
            setter = getattr(self.tracker, "set_line_starts", None)
            if callable(setter):
                setter(line_starts(song.lines))
            if axis is not None and self._axis_sig is not None and axis != self._axis_sig:
                self.warning = REPREPPED_TEXT

    # -- state ---------------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            st, t_state, timeline = self._state, self._state_t, self.timeline
        cur = timeline.locate(float(st.ref_time))
        now = time.time()
        status = st.status
        ended = self.source_finished and self._running
        if ended or self._stopped:
            status = "stopped"
        src = self.source
        level = getattr(src, "level_db", None)
        stats = json_safe(src.stats()) if callable(getattr(src, "stats", None)) else {}
        suggest = getattr(st, "key_suggest", None)
        return {"type": "state", "song_id": self.song_id, "mode": self.mode, "style": self.style, "status": status,
                "ref_time": _num(st.ref_time), "tempo": _num(st.tempo), "confidence": _num(st.confidence),
                "live_time": _num(st.live_time), "level_db": _num(level, 1),
                "line_idx": int(cur.line_idx), "syl_idx": int(cur.syl_idx), "t": round(t_state, 4),
                "age": round(max(0.0, now - t_state), 4), "syl_progress": _num(cur.syl_progress),
                "next_line_idx": int(cur.next_line_idx), "in_gap": bool(cur.in_gap),
                "frame": int(getattr(st, "frame", 0)), "active": bool(getattr(st, "active", False)),
                "stem": self.stem, "transpose": int(getattr(st, "transpose", self.transpose)),
                "key_auto": bool(getattr(st, "key_auto", False)),
                "key_suggest": None if suggest is None else int(suggest),
                "latency": _num(self.latency), "ended": bool(ended), "error": self.error,
                "warning": self.warning, "source": stats}


# --------------------------------------------------------------------------- cli


def _fmt_t(t: float | None) -> str:
    if t is None:
        return "   --.--"
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):3d}:{s:05.2f}"


def _fmt_key(k: int) -> str:
    return f"{k:+d}" if k else "0"


class _LinePrinter:
    """Print a line every time the displayed line (or the tracker status) changes, and one every
    time the key changes (transpose, automatic detection on / off, a suggested key)."""

    def __init__(self) -> None:
        self.last: tuple[int, str] | None = None
        self.key: tuple[int, bool, int | None] | None = None
        self.key_changes = 0                 # transpose changes seen after the first state
        self.t0 = time.perf_counter()

    def __call__(self, sess: LiveSession) -> None:
        snap = sess.snapshot()
        self._print_key(snap)
        key = (snap["line_idx"], snap["status"])
        if key == self.last:
            return
        self.last = key
        idx = snap["line_idx"]
        text = sess.song.lines[idx].text if 0 <= idx < len(sess.song.lines) else ""
        gap = " (gap)" if snap["in_gap"] and idx >= 0 else ""
        print(f"live {_fmt_t(snap['live_time'])}  ref {_fmt_t(snap['ref_time'])}  "
              f"line {idx:3d}{gap:6}  {snap['status']:<8} conf {snap['confidence'] or 0:.2f}  "
              f"tempo {snap['tempo'] or 0:.2f}  wall {time.perf_counter() - self.t0:7.2f}s  | {text}",
              flush=True)

    def _print_key(self, snap: dict) -> None:
        k, auto, sug = int(snap.get("transpose") or 0), bool(snap.get("key_auto")), snap.get("key_suggest")
        if sug == k:
            sug = None
        key = (k, auto, sug)
        if key == self.key:
            return
        if self.key is not None and k != self.key[0]:
            self.key_changes += 1
        self.key = key
        how = "automatic detection" if auto else "fixed"
        lean = f"; leaning to {_fmt_key(sug)}" if sug is not None else ""
        print(f"live {_fmt_t(snap['live_time'])}  key {_fmt_key(k)} ({how}{lean})", flush=True)


def cli(argv: list[str]) -> int:
    from .audio_input import FileInput

    ap = argparse.ArgumentParser(prog="lyricprompter simulate",
                                 description="Track an audio file as if it were live and print line changes")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--song", help="song id under --songs-dir")
    g.add_argument("--song-dir", type=Path, help="song package directory")
    ap.add_argument("--songs-dir", type=Path, default=SONGS_DIR)
    ap.add_argument("--file", type=Path, required=True, help="live recording to play as input")
    ap.add_argument("--mode", choices=MODES, default="mix")
    ap.add_argument("--style", choices=STYLES, default=None,
                    help="tracking style: classic (flexible tempo, follows every syllable) or pop (steady band "
                         "tempo, keeps moving through lines it cannot hear); default: the song's style")
    ap.add_argument("--lead", action="store_true", help="vocal mode: use the lead stem if present")
    ap.add_argument("--realtime", action="store_true", help="pace input at real time (default: as fast as possible)")
    ap.add_argument("--monitor", action="store_true", help="also play the input (implies --realtime)")
    ap.add_argument("--speed", type=float, default=1.0, help="pacing factor for --realtime")
    ap.add_argument("--start-s", type=float, default=0.0, help="start offset into --file (s)")
    ap.add_argument("--start-line", type=int, default=None, help="anchor the tracker at this line")
    ap.add_argument("--latency", type=float, default=None, help="latency compensation (s)")
    ap.add_argument("--transpose", type=int, default=None,
                    help="key preset: semitones the live take is above the reference (e.g. 2 when it is sung "
                         "a whole tone higher; 0 = the original key); turns automatic key detection off "
                         "unless --auto-key is also given (then it only seeds the search). Default: detect "
                         "the key automatically (reliable with accompaniment; a lone voice may need the preset).")
    ap.add_argument("--auto-key", action=argparse.BooleanOptionalAction, default=None,
                    help="automatic key detection (default: on unless --transpose is given); "
                         "--no-auto-key alone rehearses at the original key")
    args = ap.parse_args(argv)

    sdir = args.song_dir or song_dir_of(args.song, args.songs_dir)
    if not (sdir / "song.json").is_file():
        ap.error(f"no song.json in {sdir}")
    realtime = args.realtime or args.monitor
    source = FileInput(args.file, realtime=realtime, speed=args.speed, monitor=args.monitor, start_s=args.start_s)
    printer = _LinePrinter()
    t_build = time.perf_counter()
    sess = LiveSession(sdir, args.mode, source, prefer_lead=args.lead, start_line=args.start_line,
                       style=args.style, on_update=printer)
    print(f"song {sess.song_id!r}: {len(sess.song.lines)} lines, reference stem {sess.stem!r} "
          f"({len(sess.ref.rms_db)} frames), {sess.style} style, ready in {time.perf_counter() - t_build:.1f}s",
          flush=True)
    if args.latency is not None:
        sess.set_latency(args.latency)
    auto = args.auto_key if args.auto_key is not None else args.transpose is None
    if args.transpose is not None or not auto:        # same rules as POST /api/session/start
        sess.set_transpose(args.transpose or 0)       # an operator key turns auto detection off ...
        if auto:
            sess.set_auto_key(True)                   # ... unless it only seeds the search
    t0 = time.perf_counter()
    sess.start()
    try:
        while not source.wait(0.25):
            pass
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    finally:
        sess.stop()
    wall = time.perf_counter() - t0
    st = source.stats()
    audio_s = st["position_s"]
    snap = sess.snapshot()
    switches = getattr(sess.tracker, "key_switches", None)
    switches = printer.key_changes if not isinstance(switches, int) else switches
    key = (f"final key {_fmt_key(int(snap['transpose']))} "
           f"({'automatic detection' if snap['key_auto'] else 'fixed'}, "
           f"{switches} key switch{'' if switches == 1 else 'es'})")
    print(f"done: {audio_s:.1f}s of audio in {wall:.1f}s ({audio_s / max(wall, 1e-9):.1f}x realtime), {key}, "
          f"overflows {st['overflows']}, errors {st['errors']}" + (f", last error {sess.error}" if sess.error else ""))
    return 0 if sess.error is None and st["errors"] == 0 else 1
