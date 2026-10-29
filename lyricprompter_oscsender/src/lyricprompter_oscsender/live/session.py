"""OscSession: live audio source -> FeatureExtractor -> LiveTracker -> CueRunner -> OSC, with thread-safe
operator controls, a show log and a JSON-able `snapshot()` for the console's WebSocket broadcaster.

Adapted from lyricprompter's LiveSession. Audio is processed on the source's worker thread: every block
that completes a feature frame is pushed through the tracker, and the tracker state becomes a
`TrackPoint` fed to the cue runner, which fires the cues the reference position reached (sub-frame
timing comes from the runner's timer thread extrapolating between updates). Operator controls arrive
from the server's threads.

Locks: one re-entrant session lock serialises tracker access, and the runner has its own lock. The
order is always session -> runner, never the reverse: the audio thread feeds the runner while it holds
the session lock (so an operator jump can never be overtaken by a stale tracker point), and every
callback the runner makes, from any thread (`send`, `on_fire`, `live_time`), is lock-free on the session
side: `send` only reads the current settings object, `live_time` a plain attribute the audio thread
replaces, and `on_fire` only queues the record for the show log writer (it never blocks a fire).

Vocal-mode guard: while the singer is silent (the tracker coasts, or its live input is inactive) the
cues from just before the next reference phrase onset on wait (`TrackPoint.hold_from`), so a cue placed
at a vocal entry fires when the voice actually comes back, not when the coasting position reaches it.
Pop style once locked runs on at the band's tempo and is not guarded. While the tracker's relock after a
pause is provisional or settling, the TrackPoint says so (`settling`): the scheduler then reads a move back
over the coasted positions as a settle, not as a new pass (no cue fires twice).

A start or an operator jump hands the scheduler the tracker's published position (the requested time plus
the input latency look-ahead; frame 0 as 0.04 s) with the requested time as the boundary: the cues from
the requested time to the published position fire at once, a jump to a cue's time, or a start at 0:00,
makes that cue due at once, and the next tracker update continues from the same position (no false jump).
"""
from __future__ import annotations

import csv
import dataclasses
import logging
import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from .. import project
from ..config import MODES, STYLES
from ..cues import Cue
from ..osc import OscSender, SendResult
from ..scheduler import CueRunner, CueScheduler, FireRecord, SchedulerParams, TrackPoint
from ..settings import Settings, latency_s
from ..songfile import stem_path

log = logging.getLogger(__name__)

GUARD_LEAD_S = 0.1          # vocal guard: cues this close before the next phrase onset wait for the voice too
LIVE_EXTRAP_S = 0.1         # FireRecord.live_time extrapolates at most this far past the last tracker update
FIRED_IN_SNAPSHOT = 30
UPCOMING_IN_SNAPSHOT = 3
LOG_COLUMNS = ("wall_iso", "live_time", "ref_time", "cue_id", "cue_time", "target", "address", "message",
               "reason", "sent", "dry_run", "error", "ref_late_ms")


class StemMissingError(FileNotFoundError, ValueError):
    """The reference stem a mode needs is not in the project (vocal mode without a vocal stem)."""


class UnknownCueError(KeyError, ValueError):
    """No cue with that id in the session's cue list."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "unknown cue"


# --------------------------------------------------------------------------- dependency factories


@dataclass
class SessionDeps:
    """Constructors for the feature / tracking modules (overridable in tests)."""
    feature_config: Callable[[str, float], Any]          # (mode, tuning) -> FeatureConfig
    extractor: Callable[[Any], Any]                      # cfg -> FeatureExtractor
    load_ref: Callable[[Path, str, Any], Any]            # (project_dir, stem, cfg) -> FeatureFrames
    tracker_params: Callable[[str], Any]                 # mode -> TrackerParams
    tracker: Callable[..., Any]                          # (ref, fps, params, *, ref_active, line_starts)
    ref_activity: Callable[[np.ndarray, float, Any], np.ndarray] | None = None   # (rms_db, fps, params)


def _feature_config(mode: str, tuning: float) -> Any:
    from ..features.extractor import FeatureConfig

    return FeatureConfig.for_mode(mode, tuning=float(tuning or 0.0))


def _extractor(cfg: Any) -> Any:
    from ..features.extractor import FeatureExtractor

    return FeatureExtractor(cfg)


def _load_ref(pdir: Path, stem: str, cfg: Any) -> Any:
    from ..features.extractor import load_ref_features

    return load_ref_features(pdir, stem, cfg)


def _tracker_params(mode: str) -> Any:
    from ..tracking.tracker import TrackerParams

    return TrackerParams.for_mode(mode)


def _tracker(ref: Any, fps: float, params: Any, **kw: Any) -> Any:
    from ..tracking.tracker import LiveTracker

    return LiveTracker(ref, fps, params, **kw)


def _ref_activity(rms_db: np.ndarray, fps: float, params: Any) -> np.ndarray:
    from ..tracking.tracker import ref_activity

    kw = {k: getattr(params, f"ref_{k}") for k in ("rel_db", "min_gap_s", "min_on_s") if hasattr(params, f"ref_{k}")}
    return ref_activity(rms_db, fps, **kw)


def default_deps() -> SessionDeps:
    return SessionDeps(_feature_config, _extractor, _load_ref, _tracker_params, _tracker, _ref_activity)


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


# --------------------------------------------------------------------------- helpers


def reference_stem(pdir: Path, mode: str) -> str:
    """The stem a mode tracks against: "mix" for mix, "vocals" for vocal (StemMissingError without one)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    stem = "mix" if mode == "mix" else "vocals"
    path = stem_path(pdir, stem)
    if not path.is_file():
        if stem == "vocals":
            raise StemMissingError(f"project {Path(pdir).name!r} has no vocal stem: vocal mode follows the "
                                   f"vocal mic against one (add it to the project; {path} is missing)")
        raise StemMissingError(f"project {Path(pdir).name!r} has no reference recording: {path} is missing")
    return stem


def _num(x: Any, nd: int = 3) -> float | None:
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
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
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


@dataclass
class Tracking:
    """Everything that follows one live input against one project (built by `build_tracking`)."""
    project_dir: Path
    meta: project.ProjectMeta
    mode: str
    style: str
    stem: str
    feature_cfg: Any
    fps: float
    params: Any
    ref: Any
    ref_active: np.ndarray | None
    tracker: Any
    extractor: Any


def build_tracking(pdir: Path, mode: str, style: str | None = None, params: Any | None = None,
                   deps: SessionDeps | None = None) -> Tracking:
    """Reference features (cached), tracker and extractor for a project, built like lyricprompter's
    LiveSession: FeatureConfig.for_mode(mode, tuning), the mode's reference stem, TrackerParams.for_mode
    with the style preset (`style` None: the project's style), reference vocal activity in vocal mode.
    The tracker gets no line starts (vocal-mode coasting is bounded by the reference activity onsets)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if style is not None and style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, got {style!r}")
    pdir = Path(pdir)
    deps = deps or default_deps()
    meta = project.load_meta(pdir)
    style = style or (meta.style if meta.style in STYLES else "classic")
    stem = reference_stem(pdir, mode)
    cfg = deps.feature_config(mode, meta.tuning)
    fps = float(getattr(cfg, "sr", 16000)) / float(getattr(cfg, "hop", 640))
    if params is None:
        params = deps.tracker_params(mode)
    elif getattr(params, "mode", mode) != mode and dataclasses.is_dataclass(params):
        params = dataclasses.replace(params, mode=mode)
    params = with_style(params, mode, style)
    ref = deps.load_ref(pdir, stem, cfg)
    ref_active = None
    if mode == "vocal" and deps.ref_activity is not None:
        ref_active = np.asarray(deps.ref_activity(ref.rms_db, fps, params), dtype=bool)
    tracker = deps.tracker(ref, fps, params, ref_active=ref_active, line_starts=None)
    return Tracking(pdir, meta, mode, style, stem, cfg, fps, params, ref, ref_active, tracker,
                    deps.extractor(cfg))


def hold_from(tracker: Any, mode: str, fps: float, state: Any) -> float | None:
    """Vocal-mode guard for a tracker state: while the singer is silent (coasting, or the live input is
    inactive) and the position is not free-running (pop style, locked), cues from GUARD_LEAD_S before the
    next phrase onset at / after the position on wait; else None."""
    if mode != "vocal" or bool(getattr(tracker, "_free_run", False)):
        return None
    if not (state.status == "coasting" or not bool(getattr(state, "active", True))):
        return None
    onsets = getattr(tracker, "onsets", None)
    ref_time = float(state.ref_time)
    if onsets is None or len(onsets) == 0 or not math.isfinite(ref_time):
        return None
    frame = ref_time * fps - 1.0                      # published position as a reference frame index
    k = int(np.searchsorted(onsets, frame - 0.5, side="left"))
    if k >= len(onsets):
        return None
    return (float(onsets[k]) + 1.0) / fps - GUARD_LEAD_S


def relock_settling(tracker: Any, mode: str) -> bool:
    """Vocal mode: the tracker's relock after a pause is still provisional or settling (its OLTW may still
    step back), so the published position may yet move back over what it coasted through."""
    if mode != "vocal":
        return False
    free = getattr(getattr(tracker, "oltw", None), "free", 0)
    return getattr(tracker, "_prov", None) is not None or (isinstance(free, int) and free > 0)


def track_point(tracker: Any, mode: str, fps: float, state: Any, t: float, *, guard: bool = True) -> TrackPoint:
    """The scheduler's view of a tracker state at clock time `t`. `guard` adds the vocal-mode guard."""
    return TrackPoint(ref_time=float(state.ref_time), rate=float(state.tempo), status=str(state.status), t=float(t),
                      hold_from=hold_from(tracker, mode, fps, state) if guard else None,
                      settling=relock_settling(tracker, mode))


# --------------------------------------------------------------------------- show log


class ShowLog:
    """Appends every FireRecord to a CSV file (LOG_COLUMNS) from its own writer thread: `put` only queues,
    so writing never delays a fire. Each row is flushed (a crash mid-show keeps what was fired)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows = 0
        self.error: str | None = None
        self._closed = False
        self._q: queue.SimpleQueue[FireRecord | None] = queue.SimpleQueue()
        self._f = open(self.path, "x", newline="", encoding="utf-8")   # noqa: SIM115 - closed by the writer
        try:
            self._w = csv.writer(self._f)
            self._w.writerow(LOG_COLUMNS)
            self._f.flush()
        except BaseException:
            self._f.close()
            raise
        self._thread = threading.Thread(target=self._run, name="show-log", daemon=True)
        self._thread.start()

    @classmethod
    def create(cls, log_dir: Path, when: float | None = None) -> ShowLog:
        """A new log `log_dir/show-YYYYmmdd-HHMMSS.csv` (-2, -3 ... when that name is taken)."""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time() if when is None else when))
        for n in range(1, 1000):
            path = log_dir / (f"show-{stamp}.csv" if n == 1 else f"show-{stamp}-{n}.csv")
            try:
                return cls(path)
            except FileExistsError:
                continue
        raise FileExistsError(f"no free show log name in {log_dir}")

    def put(self, rec: FireRecord) -> None:
        if not self._closed:
            self._q.put(rec)

    def close(self, timeout: float = 2.0) -> None:
        """Write what is queued, close the file and stop the writer (waits at most `timeout`)."""
        if self._closed:
            return
        self._closed = True
        self._q.put(None)
        if self._thread is not threading.current_thread():
            self._thread.join(timeout)

    @staticmethod
    def row(rec: FireRecord) -> list[str]:
        wall = datetime.fromtimestamp(rec.wall).astimezone().isoformat(timespec="milliseconds")

        def num(v: float | None, nd: int) -> str:
            return "" if v is None or not math.isfinite(v) else f"{v:.{nd}f}"

        return [wall, num(rec.live_time, 3), num(rec.ref_time, 3), str(rec.cue_id), num(rec.time, 3), rec.target,
                rec.address, rec.message, rec.reason, "true" if rec.sent else "false",
                "true" if rec.dry_run else "false", rec.error or "", num(rec.ref_late_ms, 1)]

    def _run(self) -> None:
        try:
            while True:
                rec = self._q.get()
                if rec is None:
                    return
                if self.error is not None:
                    continue                          # keep draining: put() must never block
                try:
                    self._w.writerow(self.row(rec))
                    self._f.flush()
                    self.rows += 1
                except (OSError, ValueError) as e:
                    self.error = f"show log {self.path.name}: {e}"
                    log.warning("cannot write the show log %s: %s", self.path, e)
        finally:
            try:
                self._f.close()
            except OSError as e:
                log.warning("cannot close the show log %s: %s", self.path, e)


# --------------------------------------------------------------------------- session


class OscSession:
    """Track one live source against one project and fire its cues.

    `start_time` anchors the tracker (default: the beginning). `style` overrides the project's tracking
    style. `settings` gives the target latencies, default target, skip policy, catch-up window, arming
    and the input latency (the sender's targets are the server's job). `log_dir` writes a show log
    (CSV) per started session. `on_update(session)` is called on the audio thread after every tracker
    step; `on_fire(record)` for every FireRecord, in order, from whichever thread fired: it must not
    block nor take the session lock. `timer=False` runs the cue runner without its timer thread (cues
    fire at tracker updates and `runner.poll()`; tests and offline tools).
    """

    def __init__(self, project_dir: Path, mode: str, source: Any, params: Any | None = None, *,
                 cues: list[Cue], sender: OscSender, settings: Settings | None = None, style: str | None = None,
                 start_time: float = 0.0, deps: SessionDeps | None = None,
                 clock: Callable[[], float] = time.perf_counter, log_dir: Path | None = None,
                 on_update: Callable[[OscSession], None] | None = None,
                 on_fire: Callable[[FireRecord], None] | None = None, timer: bool = True):
        trk = build_tracking(project_dir, mode, style, params, deps)
        self.project_dir = trk.project_dir
        self.meta = trk.meta
        self.project_id = trk.meta.id
        self.mode, self.style, self.stem = trk.mode, trk.style, trk.stem
        self.feature_cfg, self.fps, self.params = trk.feature_cfg, trk.fps, trk.params
        self.ref, self.ref_active = trk.ref, trk.ref_active
        self.tracker, self.extractor = trk.tracker, trk.extractor
        self.source, self.sender, self.on_update = source, sender, on_update
        self.log_dir = None if log_dir is None else Path(log_dir)
        self.show_log: ShowLog | None = None
        self._user_on_fire = on_fire
        self._clock = clock

        start_time = float(start_time or 0.0)
        if not math.isfinite(start_time):
            raise ValueError(f"start time must be finite, got {start_time}")
        self.start_time = max(0.0, start_time)
        self._settings = (settings or Settings()).copy()
        self.transpose = 0
        self.input_latency_ms = 0.0
        self._set_input_latency(self._settings.input_latency_ms)

        self.started_at: float | None = None
        self.error: str | None = None
        self.warning: str | None = None
        self._errors = 0
        self._n_samples = self._n_frames = 0     # audio samples / feature frames since start()
        self._lock = threading.RLock()
        self._running = False
        self._stopped = False
        self._cues = list(cues)
        self._sched_params = SchedulerParams(catchup_s=self._settings.catchup_s,
                                             skip_policy=self._settings.skip_policy)
        self.scheduler = CueScheduler(self._cues, partial(latency_s, self._settings), self._sched_params)
        self.runner = CueRunner(self.scheduler, self._send, clock=clock, armed=self._settings.armed,
                                default_target=self._settings.default_target, on_fire=self._on_fire,
                                live_time=self._live_time, timer=timer)
        self._set_state(self.tracker.state, clock())

    # -- callbacks the runner makes (never take the session lock) ---------------------------------
    def _send(self, cue: Cue) -> SendResult:
        return self.sender.send(cue.target or self._settings.default_target, cue.address, cue.args)

    def _live_time(self) -> float | None:
        live, t = self._live
        if live is None:
            return None
        return live + min(max(self._clock() - t, 0.0), LIVE_EXTRAP_S)

    def _on_fire(self, rec: FireRecord) -> None:
        show_log = self.show_log
        if show_log is not None:
            show_log.put(rec)
        if self._user_on_fire is not None:
            self._user_on_fire(rec)

    # -- state ---------------------------------------------------------------------------------------
    def _set_state(self, st: Any, t: float) -> None:
        self._state, self._state_t, self._state_wall = st, t, time.time()
        live = getattr(st, "live_time", None)
        self._live = (float(live) if live is not None and math.isfinite(live) else None, t)

    def _point(self, st: Any, t: float, *, guard: bool = True) -> TrackPoint:
        return track_point(self.tracker, self.mode, self.fps, st, t, guard=guard)

    def _refresh_feed(self) -> None:
        """After an operator control: publish the tracker's new state and feed it to the runner as a normal
        update (a small forward move fires the crossed cues late, a small backward one re-fires nothing)."""
        t = self._clock()
        st = self.tracker.state
        self._set_state(st, t)
        if self._running:
            self.runner.feed(self._point(st, t))

    # -- audio path (source worker thread) --------------------------------------------------------
    def _on_audio(self, y: np.ndarray) -> None:
        with self._lock:
            if not self._running:
                return
            try:
                self._n_samples += len(y)
                frames = self.extractor.push(y)
                if not len(frames):
                    return
                self._n_frames += len(frames)
                st = self.tracker.push(frames)
                # the newest frame ended `excess` samples before the end of this block: stamp the update at
                # that moment, so the block size adds no lateness (up to a block, 20 ms for a rehearsal file)
                hop, sr = self.feature_cfg.hop, self.feature_cfg.sr
                excess = min(max(self._n_samples - self._n_frames * hop, 0), len(y))
                t = self._clock() - excess / sr
                self._set_state(st, t)
                pt = self._point(st, t)
            except Exception as e:
                self._errors += 1
                self.error = f"{type(e).__name__}: {e}"
                if self._errors == 1 or self._errors % 250 == 0:
                    log.exception("tracking step failed (%d errors)", self._errors)
                return
            try:
                self.runner.feed(pt)                  # under the session lock: see the module docstring
            except Exception as e:
                self.error = f"cue runner: {type(e).__name__}: {e}"
                log.exception("cue runner update failed")
        if self.on_update is not None:
            self.on_update(self)

    # -- lifecycle ---------------------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._running or self._stopped:
                raise RuntimeError("session already started")
            if self.log_dir is not None and self.show_log is None:
                try:
                    self.show_log = ShowLog.create(self.log_dir)
                except OSError as e:
                    self.warning = f"no show log: {e}"
                    log.warning("cannot create a show log in %s: %s", self.log_dir, e)
            self.extractor.reset()
            self._n_samples = self._n_frames = 0    # since the extractor reset (see _on_audio)
            self.tracker.start(self.start_time)
            t = self._clock()
            st = self.tracker.state
            self._set_state(st, t)
            self.runner.start(self._point(st, t), boundary=self.start_time)
            self._running = True
            self.started_at = time.time()
        try:
            self.source.start(self._on_audio)
        except Exception:
            with self._lock:
                self._running = False
            self.runner.stop()
            self._close_log()
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
            self.runner.stop()
            with self._lock:
                self.tracker.stop()
                self._set_state(self.tracker.state, self._clock())
            self._close_log()

    def _close_log(self) -> None:
        if self.show_log is not None:
            self.show_log.close()
            if self.show_log.error and not self.warning:
                self.warning = self.show_log.error

    @property
    def running(self) -> bool:
        return self._running

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def source_finished(self) -> bool:
        return bool(getattr(self.source, "finished", False))

    def align_source_start(self, preroll: float = 0.0) -> float:
        """Start a file source `preroll` seconds before the tracker's anchor and anchor there too.

        Only for a file on the reference's own time axis (a stem of this project), e.g. a rehearsal
        from cue N. Call before `start()`. Returns the new start time.
        """
        if self._running or self._stopped:
            raise RuntimeError("align_source_start must be called before start()")
        if not hasattr(self.source, "start_s"):
            raise TypeError(f"{type(self.source).__name__} has no start offset")
        t = max(0.0, self.start_time - max(0.0, float(preroll)))
        self.source.start_s = t
        self.start_time = t
        return t

    # -- cues and settings -------------------------------------------------------------------------
    @property
    def cues(self) -> list[Cue]:
        with self._lock:
            return list(self._cues)

    @property
    def settings(self) -> Settings:
        return self._settings.copy()

    def cue(self, cue_id: int) -> Cue:
        """The cue with this id (UnknownCueError otherwise)."""
        with self._lock:
            c = next((c for c in self._cues if c.id == cue_id), None)
        if c is None:
            raise UnknownCueError(f"no cue {cue_id} in the cue file")
        return c

    def set_cues(self, cues: list[Cue]) -> None:
        """New cue list (the file was edited / reloaded): fired state carries over per cue; a new cue
        behind the position is never fired late."""
        with self._lock:
            self._cues = list(cues)
            self.runner.set_cues(self._cues)

    def apply_settings(self, settings: Settings) -> None:
        """Target latencies, default target, skip policy, catch-up window, arming and (when it changed)
        the input latency. The sender's targets are not touched (the server sets them)."""
        s = settings.copy()
        with self._lock:
            old, self._settings = self._settings, s
            self._sched_params = dataclasses.replace(self._sched_params, catchup_s=s.catchup_s,
                                                     skip_policy=s.skip_policy)
            self.runner.set_default_target(s.default_target)
            self.runner.set_latency_of(partial(latency_s, s))
            self.runner.set_params(self._sched_params)
            self.runner.armed = s.armed
            if s.input_latency_ms != old.input_latency_ms:
                self._set_input_latency(s.input_latency_ms)
                self._refresh_feed()

    def set_armed(self, on: bool) -> None:
        with self._lock:
            self._settings = dataclasses.replace(self._settings.copy(), armed=bool(on))
            self.runner.armed = bool(on)

    @property
    def armed(self) -> bool:
        return self.runner.armed

    def fire_cue(self, cue_id: int) -> FireRecord:
        """Send a cue now, regardless of arming (reason "manual"; logged; the pass is not changed)."""
        return self.runner.fire_manual(self.cue(cue_id))

    # -- operator controls (any thread) -----------------------------------------------------------
    def jump_time(self, ref_time: float) -> None:
        """Operator jump: the tracker re-anchors at `ref_time`; cues at / after it are armed again (one
        exactly at it fires now, also in vocal mode), cues jumped over go through the skip policy."""
        ref_time = float(ref_time)
        if not math.isfinite(ref_time):
            raise ValueError(f"jump time must be finite, got {ref_time}")
        ref_time = max(0.0, ref_time)
        with self._lock:
            self.tracker.jump(ref_time)
            t = self._clock()
            st = self.tracker.state
            self._set_state(st, t)
            if self._running:
                self.runner.jump(self._point(st, t, guard=False), boundary=ref_time)

    def jump_cue(self, cue_id: int) -> None:
        """Go to cue N: jump to its time (it fires now)."""
        self.jump_time(self.cue(cue_id).time)

    def hold(self, on: bool) -> None:
        with self._lock:
            self.tracker.hold(bool(on))
            self._refresh_feed()

    def nudge(self, dt: float) -> None:
        dt = float(dt)
        if not math.isfinite(dt):
            raise ValueError(f"nudge must be finite, got {dt}")
        with self._lock:
            self.tracker.nudge(dt)
            self._refresh_feed()

    def set_transpose(self, k: int) -> None:
        """Operator key: also turns automatic key detection off."""
        with self._lock:
            self.tracker.set_transpose(int(k))
            self.transpose = int(k)
            self._refresh_feed()

    def set_auto_key(self, on: bool) -> None:
        with self._lock:
            self.tracker.set_auto_key(bool(on))
            self._refresh_feed()

    def set_style(self, style: str) -> None:
        """Switch the tracking style of the running session (the tracker goes on from where it is)."""
        if style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}, got {style!r}")
        with self._lock:
            self.tracker.set_style(style)
            self.style = style
            self._refresh_feed()

    def _set_input_latency(self, ms: float) -> None:
        ms = float(ms)
        if not math.isfinite(ms):
            raise ValueError(f"input latency must be finite, got {ms}")
        self.tracker.set_latency(ms / 1000.0)
        self.input_latency_ms = ms

    def set_input_latency(self, ms: float) -> None:
        """Tracker latency compensation (ms): the published position runs this much ahead (x tempo)."""
        with self._lock:
            self._set_input_latency(ms)
            self._refresh_feed()

    # -- state ---------------------------------------------------------------------------------------
    def input_latency_reported_ms(self) -> float | None:
        """The audio input's latency as PortAudio reports it (device inputs), else None."""
        lat = getattr(getattr(self.source, "_stream", None), "latency", None)
        if isinstance(lat, (tuple, list)):
            lat = lat[0] if lat else None
        v = _num(lat, 6)
        return None if v is None else round(v * 1000.0, 1)

    def snapshot(self) -> dict:
        with self._lock:
            st, t_state, wall = self._state, self._state_t, self._state_wall
            settings = self._settings
        runner = self.runner
        position = runner.position()
        upcoming = runner.upcoming(UPCOMING_IN_SNAPSHOT)
        records = runner.records(limit=FIRED_IN_SNAPSHOT)
        seq = runner.last_seq
        counts = runner.counts()
        armed = runner.armed
        age = max(0.0, self._clock() - t_state)
        status = st.status
        ended = self.source_finished and self._running
        if ended or self._stopped:
            status = "stopped"
        src = self.source
        stats = src.stats() if callable(getattr(src, "stats", None)) else {}
        targets = self.sender.stats() if callable(getattr(self.sender, "stats", None)) else {}
        suggest = getattr(st, "key_suggest", None)
        snap = {
            "type": "state", "project_id": self.project_id, "mode": self.mode, "style": self.style,
            "status": status, "ref_time": _num(st.ref_time), "position": _num(position),
            "tempo": _num(st.tempo), "confidence": _num(st.confidence), "live_time": _num(st.live_time),
            "level_db": _num(getattr(src, "level_db", None), 1), "t": round(wall, 4), "age": round(age, 4),
            "frame": int(getattr(st, "frame", 0)), "active": bool(getattr(st, "active", False)),
            "transpose": int(getattr(st, "transpose", self.transpose)),
            "key_auto": bool(getattr(st, "key_auto", False)),
            "key_suggest": None if suggest is None else int(suggest),
            "input_latency_ms": _num(self.input_latency_ms, 1),
            "input_latency_reported_ms": self.input_latency_reported_ms(),
            "armed": bool(armed), "ended": bool(ended), "error": self.error,
            "warning": self.warning or (self.show_log.error if self.show_log is not None else None),
            "source": stats,
            "upcoming": [{"id": c.id, "time": c.time, "target": c.target or settings.default_target,
                          "message": c.message, "eta": _num(eta)} for c, eta in upcoming],
            "fired_seq": int(seq), "fired": [r.to_dict() for r in records], "counts": counts,
            "targets": targets,
        }
        return json_safe(snap)
