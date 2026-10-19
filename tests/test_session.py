"""LiveSession with fake feature/tracker/timeline deps and a real FileInput(realtime=False)."""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import types
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lyricprompter.live import session as sess_mod
from lyricprompter.live.audio_input import FileInput
from lyricprompter.live.session import LiveSession, SessionDeps, activity_from_rms, reference_stem
from lyricprompter.songfile import load_song, save_song
from lyricprompter.text.model import AlignUnit, Line, Song, SongMeta, Syllable, Word

HOP = 640
FPS = 25.0
CONTRACT_KEYS = {"type", "song_id", "mode", "status", "ref_time", "tempo", "confidence", "live_time",
                 "level_db", "line_idx", "syl_idx", "t"}

# ----------------------------------------------------------------------------- fakes


@dataclass
class FakeFrames:
    feats: np.ndarray
    rms_db: np.ndarray

    def __len__(self) -> int:
        return len(self.feats)


@dataclass(frozen=True)
class FakeCfg:
    kind: str
    tuning: float = 0.0
    sr: int = 16000
    hop: int = HOP


class FakeExtractor:
    def __init__(self, cfg: FakeCfg):
        self.cfg, self.buf, self.resets, self.samples = cfg, np.zeros(0, np.float32), 0, 0

    def reset(self) -> None:
        self.buf = np.zeros(0, np.float32)
        self.resets += 1

    def push(self, y: np.ndarray) -> FakeFrames:
        assert y.dtype == np.float32 and y.ndim == 1
        self.samples += len(y)
        self.buf = np.concatenate([self.buf, y])
        n = len(self.buf) // HOP
        blocks, self.buf = self.buf[:n * HOP].reshape(n, HOP), self.buf[n * HOP:]
        rms = 10 * np.log10(np.mean(blocks.astype(np.float64) ** 2, axis=1) + 1e-12) if n else np.zeros(0)
        return FakeFrames(np.zeros((n, 3), np.float32), rms.astype(np.float32))


@dataclass
class FakeState:
    status: str
    ref_time: float
    tempo: float
    confidence: float
    live_time: float
    frame: int
    active: bool = True


@dataclass
class FakeParams:
    mode: str = "mix"
    latency_s: float = 0.0


class FakeTracker:
    """Perfect follower: ref position advances one frame per live frame."""

    def __init__(self, ref, fps, params, *, ref_active=None, line_starts=None):
        self.ref, self.fps, self.params = ref, fps, params
        self.ref_active, self.line_starts = ref_active, line_starts
        self.pos, self.frame, self.status = 0.0, 0, "waiting"
        self.calls: list[tuple] = []
        self.fail_after: int | None = None

    def start(self, ref_time: float = 0.0) -> None:
        self.calls.append(("start", ref_time))
        self.pos, self.status = ref_time, "tracking"

    def push(self, frames: FakeFrames) -> FakeState:
        if self.fail_after is not None and self.frame >= self.fail_after:
            raise RuntimeError("diverged")
        for _ in range(len(frames)):
            self.frame += 1
            if self.status == "tracking":
                self.pos += 1.0 / self.fps
        return self.state

    def jump(self, t: float) -> None:
        self.calls.append(("jump", t))
        self.pos = t

    def hold(self, on: bool) -> None:
        self.calls.append(("hold", on))
        self.status = "hold" if on else "tracking"

    def nudge(self, dt: float) -> None:
        self.calls.append(("nudge", dt))
        self.pos += dt

    def set_transpose(self, k: int) -> None:
        self.calls.append(("transpose", k))

    def set_auto_key(self, on: bool) -> None:
        self.calls.append(("autokey", on))

    def stop(self) -> None:
        self.calls.append(("stop",))
        self.status = "stopped"

    @property
    def state(self) -> FakeState:
        return FakeState(self.status, self.pos + self.params.latency_s, 1.0, 0.9, self.frame / self.fps, self.frame)


@dataclass
class FakeCursor:
    line_idx: int
    syl_idx: int
    syl_progress: float
    next_line_idx: int
    in_gap: bool


class FakeTimeline:
    def __init__(self, lines: list[Line]):
        self.lines = lines

    def line_start(self, idx: int) -> float:
        return float(self.lines[idx].start)

    def locate(self, t: float) -> FakeCursor:
        li = max((i for i, ln in enumerate(self.lines) if ln.start <= t), default=-1)
        if li < 0:
            return FakeCursor(-1, -1, 0.0, 0, True)
        ln = self.lines[li]
        nxt = li + 1 if li + 1 < len(self.lines) else -1
        if t >= ln.end:
            return FakeCursor(li, -1, 0.0, nxt, True)
        syl = ln.syllables()
        si = max((k for k, s in enumerate(syl) if s.start <= t), default=-1)
        prog = (t - syl[si].start) / (syl[si].end - syl[si].start) if si >= 0 else 0.0
        return FakeCursor(li, si, min(1.0, prog), nxt, False)


@dataclass
class Recorder:
    trackers: list[FakeTracker] = field(default_factory=list)
    loads: list[tuple] = field(default_factory=list)
    activity_calls: int = 0


def make_deps(rec: Recorder, n_ref: int = 200) -> SessionDeps:
    def load_ref(sdir: Path, stem: str, cfg: FakeCfg) -> FakeFrames:
        rec.loads.append((sdir, stem, cfg))
        rms = np.full(n_ref, -80.0, np.float32)
        rms[50:100] = -10.0
        return FakeFrames(np.zeros((n_ref, 3), np.float32), rms)

    def tracker(*a, **kw) -> FakeTracker:
        t = FakeTracker(*a, **kw)
        rec.trackers.append(t)
        return t

    def ref_activity(rms: np.ndarray, fps: float, params: FakeParams) -> np.ndarray:
        rec.activity_calls += 1
        return rms > -40

    return SessionDeps(feature_config=lambda mode, tuning: FakeCfg(kind=mode, tuning=tuning),
                       extractor=FakeExtractor, load_ref=load_ref, tracker_params=lambda mode: FakeParams(mode),
                       tracker=tracker, timeline=FakeTimeline, ref_activity=ref_activity)


# ----------------------------------------------------------------------------- song fixture


def make_line(idx: int, text: str, start: float, dur: float = 0.8) -> Line:
    syls = []
    step = dur / len(text)
    for k, ch in enumerate(text):
        t0 = start + k * step
        syls.append(Syllable(text=ch, span=(k, k + 1), units=[AlignUnit(ch, "a", t0, t0 + step, 0.9)],
                             start=t0, end=t0 + step))
    word = Word(text=text, span=(0, len(text)), lang="ko", syllables=syls)
    return Line(idx=idx, text=text, words=[word], start=start, end=start + dur, confidence=0.8)


@pytest.fixture
def song_dir(tmp_path: Path) -> Path:
    sdir = tmp_path / "songs" / "demo"
    song = Song(meta=SongMeta(id="demo", title="Demo", tuning=0.12, duration=4.0),
                lines=[make_line(0, "꽃밭에", 0.5), make_line(1, "앉아서", 1.5), make_line(2, "보니", 2.5)])
    save_song(song, sdir)
    (sdir / "audio").mkdir()
    y = np.zeros(44100 * 4, np.float32)
    for name in ("mix", "vocals"):
        sf.write(sdir / "audio" / f"{name}.wav", y, 44100)
    return sdir


@pytest.fixture
def live_wav(tmp_path: Path) -> Path:
    sr = 44100
    t = np.arange(int(sr * 3.0)) / sr
    y = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    path = tmp_path / "live.wav"
    sf.write(path, np.stack([y, y], 1), sr)
    return path


# ----------------------------------------------------------------------------- helpers


def test_reference_stem_choice(song_dir: Path) -> None:
    assert reference_stem(song_dir, "mix") == "mix"
    assert reference_stem(song_dir, "mix", prefer_lead=True) == "mix"
    assert reference_stem(song_dir, "vocal") == "vocals"
    assert reference_stem(song_dir, "vocal", prefer_lead=True) == "vocals"      # no lead.wav yet
    (song_dir / "audio" / "lead.wav").write_bytes(b"")
    assert reference_stem(song_dir, "vocal", prefer_lead=True) == "lead"
    assert reference_stem(song_dir, "vocal") == "vocals"
    with pytest.raises(ValueError):
        reference_stem(song_dir, "karaoke")


def test_activity_from_rms_thresholds_and_dilates() -> None:
    rms = np.full(100, -80.0)
    rms[40:50] = -6.0
    act = activity_from_rms(rms, fps=25.0, rel_db=-35.0, dilate_s=0.2)
    assert act[40:50].all()
    assert act[35:40].all() and act[50:55].all()          # 0.2 s = 5 frames of dilation
    assert not act[:34].any() and not act[56:].any()
    assert not activity_from_rms(np.full(5, -np.inf)).any()


# ----------------------------------------------------------------------------- session


def run_session(sess: LiveSession, timeout: float = 10.0) -> None:
    sess.start()
    assert sess.source.wait(timeout)


def test_session_tracks_file_and_snapshots(song_dir: Path, live_wav: Path) -> None:
    rec = Recorder()
    seen: list[tuple[int, str]] = []

    def on_update(s: LiveSession) -> None:
        snap = s.snapshot()
        key = (snap["line_idx"], snap["in_gap"])
        if not seen or seen[-1] != key:
            seen.append(key)

    sess = LiveSession(song_dir, "mix", FileInput(live_wav, realtime=False), deps=make_deps(rec), on_update=on_update)
    assert sess.stem == "mix" and sess.ref_active is None and rec.activity_calls == 0
    (sdir, stem, cfg) = rec.loads[0]
    assert (sdir, stem, cfg.kind, cfg.tuning) == (song_dir, "mix", "mix", 0.12)
    trk = rec.trackers[0]
    assert trk.fps == FPS and trk.line_starts == [0.5, 1.5, 2.5]
    run_session(sess)
    snap = sess.snapshot()
    assert CONTRACT_KEYS <= snap.keys()
    assert snap["type"] == "state" and snap["song_id"] == "demo" and snap["mode"] == "mix"
    assert snap["status"] == "stopped" and snap["ended"] is True          # the file has finished
    assert snap["live_time"] == pytest.approx(3.0, abs=0.05)
    assert snap["ref_time"] == pytest.approx(3.0, abs=0.05)
    assert snap["line_idx"] == 2 and snap["in_gap"] is False and snap["syl_idx"] == 1   # 2.5-3.3 s
    assert snap["level_db"] == pytest.approx(-13.5, abs=0.3)               # 0.3-amplitude sine RMS
    assert seen[:4] == [(-1, True), (0, False), (0, True), (1, False)]
    assert sess.extractor.samples == pytest.approx(48000, abs=2)
    sess.stop()
    sess.stop()                                                             # idempotent
    assert trk.calls[0] == ("start", 0.0) and trk.calls[-1] == ("stop",)


def test_session_vocal_mode_uses_vocals_and_ref_activity(song_dir: Path, live_wav: Path) -> None:
    rec = Recorder()
    sess = LiveSession(song_dir, "vocal", FileInput(live_wav, realtime=False), deps=make_deps(rec))
    assert sess.stem == "vocals" and rec.loads[0][1] == "vocals" and rec.loads[0][2].kind == "vocal"
    assert rec.activity_calls == 1
    act = rec.trackers[0].ref_active
    assert act.dtype == bool and act[50:100].all() and not act[:50].any()
    assert rec.trackers[0].params.mode == "vocal"


def test_session_default_ref_activity_fallback(song_dir: Path, live_wav: Path) -> None:
    rec = Recorder()
    deps = make_deps(rec)
    deps.ref_activity = None
    sess = LiveSession(song_dir, "vocal", FileInput(live_wav, realtime=False), deps=deps)
    assert sess.ref_active is not None and sess.ref_active[60:90].all()


def test_session_controls_delegate(song_dir: Path, live_wav: Path) -> None:
    rec = Recorder()
    sess = LiveSession(song_dir, "mix", FileInput(live_wav, realtime=True, speed=0.5), deps=make_deps(rec),
                       start_line=1)
    trk = rec.trackers[0]
    sess.start()
    try:
        assert trk.calls[0] == ("start", 1.5)
        sess.jump_line(2)
        assert sess.snapshot()["ref_time"] == pytest.approx(2.5, abs=0.1)
        sess.hold(True)
        assert sess.snapshot()["status"] == "hold"
        sess.hold(False)
        sess.nudge(-0.5)
        sess.set_transpose(-2)
        sess.set_auto_key(True)
        sess.set_latency(0.25)
        snap = sess.snapshot()
        assert snap["transpose"] == -2 and snap["latency"] == 0.25
        assert snap["key_auto"] is False and snap["key_suggest"] is None   # fake state has no key fields
        assert trk.params.latency_s == 0.25                    # no set_latency(): params updated in place
        sess.jump_time(-3.0)
    finally:
        sess.stop()
    names = [c[0] for c in trk.calls]
    assert names == ["start", "jump", "hold", "hold", "nudge", "transpose", "autokey", "jump", "stop"]
    assert trk.calls[-2] == ("jump", 0.0)


def test_session_prefers_tracker_set_latency(song_dir: Path, live_wav: Path) -> None:
    rec = Recorder()
    deps = make_deps(rec)
    got = []
    sess = LiveSession(song_dir, "mix", FileInput(live_wav, realtime=False), deps=deps)
    rec.trackers[0].set_latency = got.append                   # type: ignore[attr-defined]
    sess.set_latency(0.1)
    assert got == [0.1] and rec.trackers[0].params.latency_s == 0.0


def test_session_controls_are_thread_safe(song_dir: Path, live_wav: Path) -> None:
    rec = Recorder()
    sess = LiveSession(song_dir, "mix", FileInput(live_wav, realtime=False), deps=make_deps(rec))
    errors: list[BaseException] = []

    def hammer() -> None:
        try:
            for k in range(300):
                sess.nudge(0.0)
                sess.snapshot()
                if k % 50 == 0:
                    sess.jump_line(k // 50 % 3)
        except BaseException as e:   # noqa: BLE001 - surfaced by the assertion below
            errors.append(e)

    threads = [threading.Thread(target=hammer) for _ in range(3)]
    sess.start()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert sess.source.wait(10)
    sess.stop()
    assert not errors and sess.error is None


def test_session_survives_tracker_errors(song_dir: Path, live_wav: Path) -> None:
    rec = Recorder()
    sess = LiveSession(song_dir, "mix", FileInput(live_wav, realtime=False), deps=make_deps(rec))
    rec.trackers[0].fail_after = 10
    run_session(sess)
    snap = sess.snapshot()
    assert "diverged" in snap["error"] and snap["source"]["errors"] == 0
    sess.stop()


def test_session_validation(song_dir: Path, live_wav: Path, tmp_path: Path) -> None:
    deps = make_deps(Recorder())
    with pytest.raises(ValueError):
        LiveSession(song_dir, "karaoke", FileInput(live_wav, realtime=False), deps=deps)
    (song_dir / "audio" / "vocals.wav").unlink()
    with pytest.raises(FileNotFoundError):
        LiveSession(song_dir, "vocal", FileInput(live_wav, realtime=False), deps=deps)
    with pytest.raises(FileNotFoundError):
        LiveSession(tmp_path / "nope", "mix", FileInput(live_wav, realtime=False), deps=deps)


def test_reload_lines_updates_timeline(song_dir: Path, live_wav: Path) -> None:
    rec = Recorder()
    sess = LiveSession(song_dir, "mix", FileInput(live_wav, realtime=False), deps=make_deps(rec))
    song = sess.song
    song.lines[0].start = 0.1
    sess.reload_lines(song)
    assert sess.timeline.line_start(0) == 0.1


def test_default_deps_use_for_mode_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    @dataclass(frozen=True)
    class Cfg:
        kind: str = "mix"
        tuning: float = 0.0
        hop: int = 640

        @classmethod
        def for_mode(cls, mode: str, tuning: float = 0.0) -> Cfg:
            return cls(kind=mode, tuning=tuning, hop=320 if mode == "vocal" else 640)

    @dataclass
    class Params:
        mode: str = "mix"
        ref_rel_db: float = 20.0

        @classmethod
        def for_mode(cls, mode: str) -> Params:
            return cls(mode=mode, ref_rel_db=30.0)

    calls = []
    fake_ext = types.ModuleType("lyricprompter.features.extractor")
    fake_ext.FeatureConfig = Cfg
    fake_trk = types.ModuleType("lyricprompter.tracking.tracker")
    fake_trk.TrackerParams = Params
    fake_trk.ref_activity = lambda rms, fps, **kw: calls.append(kw) or np.ones(len(rms), bool)
    monkeypatch.setitem(sys.modules, "lyricprompter.features.extractor", fake_ext)
    monkeypatch.setitem(sys.modules, "lyricprompter.tracking.tracker", fake_trk)
    deps = sess_mod.default_deps()
    cfg = deps.feature_config("vocal", 0.25)
    assert (cfg.kind, cfg.tuning, cfg.hop) == ("vocal", 0.25, 320)
    params = deps.tracker_params("vocal")
    assert params == Params("vocal", 30.0)
    assert deps.ref_activity(np.zeros(4), 25.0, params).all() and calls == [{"rel_db": 30.0}]


def test_align_source_start_and_rehearsal_factory(song_dir: Path, live_wav: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    from lyricprompter.server import app as app_mod

    rec = Recorder()
    monkeypatch.setattr(sess_mod, "default_deps", lambda: make_deps(rec))
    # rehearsal from line 2 on the song's own mix: the file and the tracker both start 2 s before it
    req = app_mod.SessionStart(song_id="demo", start_line=2, input=app_mod.InputSpec(kind="file", stem="mix"))
    sess = app_mod.default_session_factory(song_dir, req, song_dir.parent)
    want = 2.5 - app_mod.REHEARSAL_PREROLL
    assert sess.source.start_s == pytest.approx(want) and sess.start_time == pytest.approx(want)
    assert sess.source.realtime
    sess.source.realtime = False                      # the server paces in real time; the test need not
    run_session(sess)
    sess.stop()
    assert rec.trackers[-1].calls[0] == ("start", pytest.approx(want))
    assert sess.source.file_position_s == pytest.approx(4.0, abs=0.05)          # read from 0.5 s to the end
    # line 0 (0.5 s) clamps the pre-roll at the file start
    req = app_mod.SessionStart(song_id="demo", start_line=0, input=app_mod.InputSpec(kind="file", stem="mix"))
    assert app_mod.default_session_factory(song_dir, req, song_dir.parent).start_time == 0.0
    # only before start(), and only for sources with a start offset
    with pytest.raises(RuntimeError):
        sess.align_source_start(1.0)
    other = LiveSession(song_dir, "mix", types.SimpleNamespace(start=lambda cb: None, stop=lambda: None),
                        start_line=1, deps=make_deps(rec))
    with pytest.raises(TypeError):
        other.align_source_start(1.0)
    fresh = LiveSession(song_dir, "mix", FileInput(live_wav, realtime=False), start_line=1, deps=make_deps(rec))
    assert fresh.align_source_start(0.4) == pytest.approx(1.1) and fresh.source.start_s == pytest.approx(1.1)


def test_simulate_cli_prints_line_changes(song_dir: Path, live_wav: Path, monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sess_mod, "default_deps", lambda: make_deps(Recorder()))
    rc = sess_mod.cli(["--song-dir", str(song_dir), "--file", str(live_wav), "--mode", "mix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "reference stem 'mix'" in out
    assert "| 꽃밭에" in out and "| 앉아서" in out and "| 보니" in out
    assert "3.0s of audio" in out
    rc = sess_mod.cli(["--song", "demo", "--songs-dir", str(song_dir.parent), "--file", str(live_wav),
                       "--mode", "vocal", "--start-line", "1"])
    assert rc == 0 and "line   1" in capsys.readouterr().out


# ----------------------------------------------------------------------------- review regressions


def test_session_id_is_the_package_folder(song_dir: Path, live_wav: Path) -> None:
    """A copied / renamed package keeps song.json's old meta.id; the server, the screens and the
    edits all key by the folder name (NFC), so the session must too."""
    deps = make_deps(Recorder())
    copy = song_dir.parent / "demo-live"
    shutil.copytree(song_dir, copy)
    assert LiveSession(copy, "mix", FileInput(live_wav, realtime=False), deps=deps).song_id == "demo-live"
    nfd = song_dir.parent / unicodedata.normalize("NFD", "꽃밭에서")
    shutil.copytree(song_dir, nfd)
    sess = LiveSession(nfd, "mix", FileInput(live_wav, realtime=False), deps=deps)
    assert sess.song_id == "꽃밭에서" and sess.snapshot()["song_id"] == "꽃밭에서"


def test_renamed_package_runs_under_its_folder_id_in_the_server(song_dir: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the session announced meta.id ('demo') for the package 'demo-live', so prompters
    fetched another (or a missing) song, edits never reached the session and cueing it was a 409."""
    from fastapi.testclient import TestClient

    from lyricprompter.server.app import create_app
    from lyricprompter.server.jobs import JobManager

    rec = Recorder()
    monkeypatch.setattr(sess_mod, "default_deps", lambda: make_deps(rec))
    root = song_dir.parent
    song_dir.rename(root / "demo-live")                      # meta.id stays "demo"; no "demo" package
    app = create_app(root, jobs=JobManager(), devices=lambda refresh: [], allowed_hosts=("testserver",))
    hub = app.state.hub
    with TestClient(app) as c:
        r = c.post("/api/session/start", json={"song_id": "demo-live", "input": {"kind": "file", "stem": "mix"}})
        assert r.status_code == 200 and r.json()["song_id"] == "demo-live"
        with c.websocket_connect("/ws") as ws:
            assert ws.receive_json() == {"type": "song", "id": "demo-live", "rev": 0}
        assert c.put("/api/songs/demo-live/lines/1", json={"shift": 0.2}).json()["rev"] == 1
        assert hub.session.timeline.line_start(1) == pytest.approx(1.7)          # the edit reached it
        assert hub.song_message() == {"type": "song", "id": "demo-live", "rev": 1}
        assert c.post("/api/session/cue", json={"song_id": "demo-live"}).status_code == 200
        c.post("/api/session/stop")
        assert hub.cued_song == "demo-live"
        st = c.post("/api/session/jump", json={"line": 2}).json()
        assert st["status"] == "manual" and st["line_idx"] == 2


def test_snapshot_is_strict_json_after_an_inf_sample(song_dir: Path, tmp_path: Path) -> None:
    """Regression: one +inf input sample kept the source's level_db at inf, and the snapshot passed
    the source stats through raw: every WebSocket state message carried `Infinity`."""
    sr = 44100
    y = (0.3 * np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr)).astype(np.float32)
    y[sr // 2] = np.inf
    path = tmp_path / "glitch.wav"
    sf.write(path, y, sr, subtype="FLOAT")
    sess = LiveSession(song_dir, "mix", FileInput(path, realtime=False), deps=make_deps(Recorder()))
    run_session(sess)
    snap = sess.snapshot()
    sess.stop()
    json.dumps(snap, allow_nan=False)
    assert snap["source"]["level_db"] == pytest.approx(-13.5, abs=0.5) and snap["level_db"] is not None

    class BrokenStats:                       # any other source's non-finite stats are nulled too
        level_db = float("inf")

        def stats(self) -> dict:
            return {"level_db": float("inf"), "peak_db": float("-inf"), "xs": [float("nan"), 1.0]}

    sess = LiveSession(song_dir, "mix", BrokenStats(), deps=make_deps(Recorder()))
    snap = sess.snapshot()
    json.dumps(snap, allow_nan=False)
    assert snap["source"] == {"level_db": None, "peak_db": None, "xs": [None, 1.0]} and snap["level_db"] is None
    with pytest.raises(ValueError):
        sess.set_latency(float("nan"))
    with pytest.raises(ValueError):
        sess.jump_time(float("inf"))
    assert snap["latency"] == 0.0


def reprep(sdir: Path, new_recording: bool) -> None:
    """What `prep` does to a live package: new timings (one line dropped), maybe a new mix."""
    song = load_song(sdir)
    song.lines[0].text = "REPREPPED"
    song.lines.pop()
    if new_recording:
        tmp = sdir / "audio" / ".mix.new.wav"
        sf.write(tmp, np.zeros(44100 * 5, np.float32), 44100)
        os.replace(tmp, sdir / "audio" / "mix.wav")
    save_song(song, sdir)


def test_reload_after_a_reprep_from_another_recording_asks_for_a_restart(song_dir: Path, live_wav: Path) -> None:
    """reload_lines swaps the lines but keeps the reference features: say so when the package's
    recording changed under the session (never for a line edit or a re-prep of the same recording)."""
    sess = LiveSession(song_dir, "mix", FileInput(live_wav, realtime=False), deps=make_deps(Recorder()))
    sess.reload_lines(sess.song)                                   # a line edit
    reprep(song_dir, new_recording=False)
    sess.reload_lines()
    assert sess.warning is None and sess.snapshot()["warning"] is None and len(sess.song.lines) == 2
    reprep(song_dir, new_recording=True)
    sess.reload_lines()
    assert sess.snapshot()["warning"] == sess_mod.REPREPPED_TEXT


def test_reprep_while_the_session_is_built_reaches_it_and_the_screens(song_dir: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the server took the song.json baseline at the watcher's first tick after a start
    (up to 0.5 s later), so a re-prep landing in between was absorbed: the screens kept rev 0, the
    session its old lines, and the next unrelated line edit swapped them mid-show."""
    from fastapi.testclient import TestClient

    from lyricprompter.server.app import create_app
    from lyricprompter.server.jobs import JobManager

    rec = Recorder()
    deps = make_deps(rec)
    load_ref = deps.load_ref

    def load_ref_then_reprep(sdir: Path, stem: str, cfg: FakeCfg) -> FakeFrames:
        ref = load_ref(sdir, stem, cfg)                # song.json was read before this
        reprep(sdir, new_recording=True)
        return ref
    deps.load_ref = load_ref_then_reprep
    monkeypatch.setattr(sess_mod, "default_deps", lambda: deps)
    app = create_app(song_dir.parent, jobs=JobManager(), devices=lambda refresh: [],
                     allowed_hosts=("testserver",))
    hub = app.state.hub
    with TestClient(app) as c:
        r = c.post("/api/session/start", json={"song_id": "demo", "input": {"kind": "file", "stem": "mix"}})
        assert r.status_code == 200 and len(hub.session.song.lines) == 3        # read before the re-prep
        with c.websocket_connect("/ws") as ws:
            got, deadline = None, time.monotonic() + 3
            while got is None and time.monotonic() < deadline:
                m = ws.receive_json()
                got = m if m["type"] == "song" and m["rev"] == 1 else None
        state = c.get("/api/session").json()["state"]
        assert got == {"type": "song", "id": "demo", "rev": 1}
        assert [ln.text for ln in hub.session.song.lines] == ["REPREPPED", "앉아서"]
        assert state["warning"] == sess_mod.REPREPPED_TEXT          # tracking needs a restart
        c.post("/api/session/stop")


@dataclass
class KeyState(FakeState):
    transpose: int = 0
    key_auto: bool = False
    key_suggest: int | None = None


class KeyTracker(FakeTracker):
    """Automatic key detection: leans to +2 after 0.5 s of audio, switches to it after 1 s."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.key, self.auto, self.key_switches = 0, True, 0

    def set_transpose(self, k: int) -> None:
        super().set_transpose(k)
        self.key, self.auto = k, False

    def set_auto_key(self, on: bool) -> None:
        self.calls.append(("auto", on))
        self.auto = on

    def push(self, frames: FakeFrames) -> FakeState:
        super().push(frames)
        if self.auto and self.frame >= self.fps and self.key != 2:
            self.key, self.key_switches = 2, self.key_switches + 1
        return self.state

    @property
    def state(self) -> KeyState:
        base = vars(FakeTracker.state.fget(self))
        sug = 2 if self.auto and self.key != 2 and self.frame >= self.fps / 2 else None
        return KeyState(**base, transpose=self.key, key_auto=self.auto, key_suggest=sug)


def test_simulate_cli_shows_the_key_and_honours_transpose_0(song_dir: Path, live_wav: Path,
                                                            monkeypatch: pytest.MonkeyPatch,
                                                            capsys: pytest.CaptureFixture[str]) -> None:
    """Regression: simulate never showed what automatic key detection did, and `--transpose 0` was
    a no-op although its help says it turns automatic detection off."""
    rec = Recorder()
    deps = make_deps(rec)
    deps.tracker = lambda *a, **kw: rec.trackers.append(KeyTracker(*a, **kw)) or rec.trackers[-1]
    monkeypatch.setattr(sess_mod, "default_deps", lambda: deps)
    base = ["--song-dir", str(song_dir), "--file", str(live_wav)]

    assert sess_mod.cli(base) == 0
    out = capsys.readouterr().out
    assert "key 0 (automatic detection)" in out
    assert "key 0 (automatic detection; leaning to +2)" in out
    assert "key +2 (automatic detection)" in out
    assert "final key +2 (automatic detection, 1 key switch)" in out
    assert rec.trackers[-1].calls[0][0] == "start"                       # nothing preset

    for extra in (["--transpose", "0"], ["--no-auto-key"]):               # the original key, auto off
        assert sess_mod.cli(base + extra) == 0
        out = capsys.readouterr().out
        assert ("transpose", 0) in rec.trackers[-1].calls and not rec.trackers[-1].auto, extra
        assert "final key 0 (fixed, 0 key switches)" in out and "leaning" not in out, extra

    assert sess_mod.cli(base + ["--transpose", "2", "--auto-key"]) == 0   # a preset that only seeds
    trk = rec.trackers[-1]
    assert trk.calls[:2] == [("transpose", 2), ("auto", True)] and trk.auto
    assert "final key +2 (automatic detection, 0 key switches)" in capsys.readouterr().out


# ----------------------------------------------------------------------------- slow: real modules + audio

STEMS = Path(__file__).resolve().parents[1] / "data" / "stems" / "edelweiss_short"


def real_song(tmp_path: Path) -> Path:
    """Song package over the pre-separated Edelweiss (second take) stems with synthetic line times."""
    if not (STEMS / "vocals.wav").is_file():
        pytest.skip("data/stems/edelweiss_short not available")
    sdir = tmp_path / "songs" / "edelweiss-short"
    lines = [make_line(i, text, start, dur=4.0) for i, (text, start) in
             enumerate([("에델바이스", 7.0), ("아침마다", 14.5), ("작고하얀", 21.0), ("반가워요", 28.0)])]
    save_song(Song(meta=SongMeta(id="edelweiss-short", title="Edelweiss"), lines=lines), sdir)
    (sdir / "audio").mkdir()
    for name in ("mix", "vocals"):
        (sdir / "audio" / f"{name}.wav").symlink_to(STEMS / f"{name}.wav")
    return sdir


@pytest.mark.slow
@pytest.mark.parametrize(("mode", "stem", "p95_ms"), [("mix", "mix", 80.0), ("vocal", "vocals", 250.0)])
def test_real_tracker_follows_its_own_reference(tmp_path: Path, mode: str, stem: str, p95_ms: float) -> None:
    pytest.importorskip("lyricprompter.tracking.tracker")
    pytest.importorskip("lyricprompter.features.extractor")
    sdir = real_song(tmp_path)
    live, ref = [], []

    def on_update(s: LiveSession) -> None:
        live.append(s._state.live_time)
        ref.append(s._state.ref_time)

    src = FileInput(STEMS / f"{stem}.wav", realtime=False)
    sess = LiveSession(sdir, mode, src, on_update=on_update)
    assert (sess.ref_active is not None) == (mode == "vocal")
    sess.start()
    assert src.wait(60)
    snap = sess.snapshot()
    sess.stop()
    assert sess.error is None and src.errors == 0 and src.overflows == 0
    assert snap["live_time"] == pytest.approx(39.1, abs=0.2)
    err = np.abs(np.array(ref) - np.array(live))[np.array(live) > 3.0]
    assert np.median(err) < 0.05 and np.percentile(err, 95) * 1000 < p95_ms
    assert snap["line_idx"] == 3
