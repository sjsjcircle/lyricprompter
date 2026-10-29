"""OscSession: the real feature extractor + tracker follow synthetic live takes (synth.render_song with a known
tempo map) pushed by a hand-driven fake source, and the cue runner fires the project's cues.

Most tests run the runner without its timer thread on an injected clock (`Driver`: push a 20 ms block at
its end time, then poll every 5 ms like the timer thread would), so firing is deterministic. One test uses
the timer thread on a scaled real clock and a real OscSender to a local UDP socket; one hammers the
operator controls from another thread while audio runs (lock order).
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soxr

from lyricprompter_oscsender import project as P
from lyricprompter_oscsender import synth as S
from lyricprompter_oscsender.audio_io import save_wav
from lyricprompter_oscsender.cues import parse_cues
from lyricprompter_oscsender.live.session import (
    LOG_COLUMNS,
    OscSession,
    StemMissingError,
    UnknownCueError,
    build_tracking,
    default_deps,
    hold_from,
    json_safe,
    reference_stem,
)
from lyricprompter_oscsender.osc import OscSender, SendResult, Target, decode_message
from lyricprompter_oscsender.settings import Settings

SR = 16000
BLOCK = 320                    # 20 ms, like a sound card block
POLL_S = 0.005                 # the Driver's timer resolution

SNAPSHOT_KEYS = {"type", "project_id", "mode", "style", "status", "ref_time", "position", "tempo", "confidence",
                 "live_time", "level_db", "t", "age", "frame", "active", "transpose", "key_auto", "key_suggest",
                 "input_latency_ms", "input_latency_reported_ms", "armed", "ended", "error", "warning", "source",
                 "upcoming", "fired_seq", "fired", "counts", "targets"}

# mix song: 20 s reference, a live take that is first 6% slower then 3% faster; a cue every 2 s
MIX_TM = ([0.0, 10.0, 20.0], [0.0, 10.6, 20.3])
MIX_CUES = "".join(f"{1.5 + 2 * k:.3f}  /cue {k + 1}\n" for k in range(9))      # 1.5, 3.5, ..., 17.5

# vocal song: three sung phrases; the live singer waits 3 s longer before the second one
PHRASES = ((1.0, 5.0), (7.0, 11.0), (13.0, 17.0))
VOCAL_TM = ([0.0, 5.2, 6.9, 19.0], [0.0, 5.2, 9.9, 22.0])
VOCAL_CUES = "3.000 /a\n7.000 /b\n9.000 /c\n15.000 /d\n"


# ----------------------------------------------------------------------------- fixtures / fakes


def write_project(root: Path, pid: str, mix: np.ndarray, vocals: np.ndarray | None = None, *,
                  cues: str = "", style: str = "classic") -> Path:
    """A project folder written directly (no ffmpeg): canonical 44.1 kHz stereo PCM24 stems + project.json."""
    pdir = root / pid
    stems = {"mix": mix} if vocals is None else {"mix": mix, "vocals": vocals}
    for name, y in stems.items():
        y44 = soxr.resample(np.asarray(y, np.float32), SR, 44100)
        save_wav(pdir / "audio" / f"{name}.wav", np.stack([y44, y44], axis=1), 44100)
    (pdir / "cache").mkdir()
    (pdir / "logs").mkdir()
    P.save_meta(pdir, P.ProjectMeta(id=pid, title=pid, created="2026-01-01T00:00:00+00:00",
                                    duration=len(mix) / SR, tuning=0.0, style=style, stems=list(stems)))
    P.write_cues(pdir, cues)
    return pdir


@pytest.fixture(scope="module")
def mix_song(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, np.ndarray]:
    score = S.make_score(20, 3)
    pdir = write_project(tmp_path_factory.mktemp("projects"), "mixsong", S.render_song(score), cues=MIX_CUES)
    return pdir, S.render_song(score, tempo_map=MIX_TM)


def vocal_score(seconds: float = 19.0, seed: int = 5) -> S.Score:
    """A lead voice singing PHRASES (legato eighth notes on a scale) and nothing else."""
    rng = np.random.default_rng(seed)
    rows, m = [], 67.0
    for a, b in PHRASES:
        t = a
        while t < b - 1e-9:
            m = float(np.clip(m + rng.choice([-2, -1, 1, 2]), 60, 76))
            rows.append((t, 0.24, m, 0.25, "lead"))
            t += 0.25
    return S.Score(notes=S.make_notes(rows), duration=seconds)


@pytest.fixture(scope="module")
def vocal_song(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, np.ndarray]:
    vs = vocal_score()
    vocals = S.render_song(vs)
    band = S.render_song(S.make_score(vs.duration, 8))
    pdir = write_project(tmp_path_factory.mktemp("projects"), "vocalsong", 0.6 * band + vocals, vocals,
                         cues=VOCAL_CUES)
    live = S.render_song(vs, tempo_map=VOCAL_TM)
    live = live + 1e-3 * np.random.default_rng(0).standard_normal(len(live)).astype(np.float32)   # mic noise
    return pdir, live.astype(np.float32)


class FakeClock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeSource:
    """A live input the test drives: `push(block)` calls the session's on_audio on the calling thread."""

    def __init__(self) -> None:
        self.on_audio = None
        self.finished = False
        self.stopped = False
        self.start_s = 0.0
        self.level_db = -20.0

    def start(self, on_audio) -> None:
        self.on_audio = on_audio

    def stop(self) -> None:
        self.stopped = True

    def push(self, y: np.ndarray) -> None:
        if self.on_audio is not None and not self.stopped:
            self.on_audio(np.ascontiguousarray(y, dtype=np.float32))

    def stats(self) -> dict:
        return {"level_db": self.level_db, "peak_db": float("-inf"), "finished": self.finished}


class RecSender:
    """Records what would be sent (target, address, args, clock time)."""

    def __init__(self, clock=time.perf_counter) -> None:
        self.clock = clock
        self.sent: list[tuple[str, str, tuple, float]] = []
        self.lock = threading.Lock()

    def send(self, target: str, address: str, args=()) -> SendResult:
        with self.lock:
            self.sent.append((target, address, tuple(args), self.clock()))
        return SendResult(True, target, None, self.clock(), 0)

    def stats(self) -> dict:
        return {"main": {"sent": len(self.sent), "errors": 0}}


class Driver:
    """Feeds a live take block by block on a fake clock, polling the runner between blocks."""

    def __init__(self, sess: OscSession, src: FakeSource, live: np.ndarray, clock: FakeClock):
        self.sess, self.src, self.live, self.clock = sess, src, live, clock
        self.i = 0

    @property
    def t(self) -> float:
        return self.i / SR

    def run_until(self, live_s: float) -> None:
        end = min(len(self.live), round(live_s * SR))
        while self.i < end:
            y = self.live[self.i:self.i + BLOCK]
            self.i += len(y)
            self.clock.now = self.i / SR
            self.src.push(y)
            self.sess.runner.poll()
            for k in range(1, round(BLOCK / SR / POLL_S)):
                self.clock.now = self.i / SR + k * POLL_S
                self.sess.runner.poll()
        self.clock.now = self.i / SR


def make_session(pdir: Path, mode: str = "mix", *, settings: Settings | None = None, sender=None,
                 timer: bool = False, clock=None, **kw) -> tuple[OscSession, FakeSource, FakeClock]:
    clock = clock or FakeClock()
    src = FakeSource()
    cues = parse_cues(P.read_cues(pdir)).cues
    sess = OscSession(pdir, mode, src, cues=cues, sender=sender or RecSender(clock), settings=settings or Settings(),
                      clock=clock, timer=timer, **kw)
    return sess, src, clock


def started(pdir: Path, live: np.ndarray, mode: str = "mix", **kw) -> tuple[OscSession, Driver]:
    sess, src, clock = make_session(pdir, mode, **kw)
    sess.start()
    return sess, Driver(sess, src, live, clock)


def recs(sess: OscSession) -> list:
    return sess.runner.records(limit=1000)


def by_cue(sess: OscSession) -> dict[int, list]:
    out: dict[int, list] = {}
    for r in recs(sess):
        out.setdefault(r.cue_id, []).append(r)
    return out


def mix_gt(cue_time: float) -> float:
    return float(S.score_to_perf(cue_time, MIX_TM))


# ----------------------------------------------------------------------------- construction


def test_reference_stem_and_build(mix_song, vocal_song) -> None:
    mix_dir, _ = mix_song
    voc_dir, _ = vocal_song
    assert reference_stem(mix_dir, "mix") == "mix"
    assert reference_stem(voc_dir, "vocal") == "vocals"
    with pytest.raises(StemMissingError) as ei:
        reference_stem(mix_dir, "vocal")
    assert isinstance(ei.value, FileNotFoundError) and isinstance(ei.value, ValueError)
    assert "vocal stem" in str(ei.value)
    with pytest.raises(ValueError):
        reference_stem(mix_dir, "drums")

    trk = build_tracking(voc_dir, "vocal")
    assert trk.stem == "vocals" and trk.style == "classic" and trk.params.mode == "vocal"
    assert trk.ref_active is not None and trk.ref_active.dtype == bool
    # no line starts: the phrase onsets are the rising edges of the reference vocal activity
    onsets_s = (trk.tracker.onsets + 1) / trk.fps
    for a, _ in PHRASES:
        assert np.min(np.abs(onsets_s - a)) < 0.1
    assert build_tracking(voc_dir, "vocal", "pop").params.style == "pop"
    assert build_tracking(mix_dir, "mix").ref_active is None


def test_project_style_and_override(tmp_path: Path, mix_song) -> None:
    pdir, _ = mix_song
    pop_dir = tmp_path / "popsong"
    pop_dir.mkdir()
    for name in ("audio", "cache"):
        (pop_dir / name).symlink_to(pdir / name)
    P.save_meta(pop_dir, dataclasses.replace(P.load_meta(pdir), id="popsong", style="pop"))
    sess, _, _ = make_session(pop_dir)
    assert sess.style == "pop" and sess.tracker.params.style == "pop" and sess.project_id == "popsong"
    sess2, _, _ = make_session(pop_dir, style="classic")
    assert sess2.style == "classic" and sess2.tracker.params.style == "classic"
    with pytest.raises(ValueError):
        make_session(pop_dir, style="disco")


def test_vocal_mode_without_vocals_is_an_error(mix_song) -> None:
    with pytest.raises(StemMissingError):
        make_session(mix_song[0], "vocal")


# ----------------------------------------------------------------------------- firing


def test_cues_fire_once_near_ground_truth(mix_song) -> None:
    pdir, live = mix_song
    sess, drv = started(pdir, live)
    drv.run_until(len(live) / SR)
    cues = sess.cues
    fired = by_cue(sess)
    assert sorted(fired) == [c.id for c in cues]
    assert all(len(v) == 1 and v[0].sent and not v[0].dry_run for v in fired.values())
    err = [abs(fired[c.id][0].live_time - mix_gt(c.time)) for c in cues]
    assert np.median(err) < 0.06 and max(err) < 0.2
    assert [s[1:3] for s in sess.sender.sent] == [("/cue", ((("i", k + 1),))) for k in range(len(cues))]
    assert all(s[0] == "main" for s in sess.sender.sent)
    assert {r.reason for r in recs(sess)} <= {"track", "catchup"}


def test_realtime_timer_thread_and_udp(mix_song) -> None:
    """The timer thread fires between tracker updates on a (4x scaled) real clock; a real OscSender sends
    to a local UDP socket."""
    pdir, live = mix_song
    take = live[:round(10.5 * SR)]
    speed = 4.0
    t0 = time.perf_counter()

    def clock() -> float:
        return (time.perf_counter() - t0) * speed

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    sends: list[tuple[str, float]] = []

    class TimedSender(OscSender):
        def send(self, target, address, args=(), *, packet=None):
            res = super().send(target, address, args, packet=packet)
            sends.append((address, res.t))
            return res

    sender = TimedSender(clock=clock)
    settings = Settings(targets=[Target("main", "127.0.0.1", rx.getsockname()[1])])
    assert sender.set_targets(settings.targets) == {"main": None}
    sess, src, _ = make_session(pdir, settings=settings, sender=sender, timer=True, clock=clock)
    try:
        t0 = time.perf_counter()
        sess.start()
        for i in range(0, len(take), BLOCK):
            end = (i + BLOCK) / SR
            while clock() < end:
                time.sleep(0.0005)
            src.push(take[i:i + BLOCK])
        time.sleep(0.05)
    finally:
        sess.stop()
    expected = [c for c in sess.cues if mix_gt(c.time) < len(take) / SR - 0.15]
    fired = by_cue(sess)
    assert [cid for cid in fired if fired[cid][0].sent] == [c.id for c in expected]
    assert all(len(fired[c.id]) == 1 for c in expected)
    err = [t - mix_gt(c.time) for c, (_, t) in zip(expected, sends)]
    assert np.median(np.abs(err)) < 0.06 and max(np.abs(err)) < 0.2
    packets = [decode_message(rx.recvfrom(4096)[0]) for _ in expected]
    rx.close()
    sender.close()
    assert packets == [("/cue", [("i", k + 1)]) for k in range(len(expected))]
    assert not sess.runner.running and sess.snapshot()["status"] == "stopped"


def test_start_position_cue_fires_at_once_and_earlier_ones_are_chased(mix_song) -> None:
    pdir, live = mix_song
    shifted = live[round(mix_gt(5.5) * SR):]               # the live take from the cue at 5.5
    sess, drv = started(pdir, shifted, start_time=5.5)
    first = recs(sess)
    # "latest" policy: of the cues before the start (/cue 1, 2: one address) only the latest is sent; the
    # cue at the start position fires at once
    assert [(r.cue_id, r.reason, r.sent) for r in first] == [(2, "start", True), (1, "skipped", False),
                                                            (3, "track", True)]
    drv.run_until(3.0)
    assert [r.cue_id for r in recs(sess)][3:] == [4]


def test_operator_jump_to_cue_fires_it(mix_song) -> None:
    pdir, live = mix_song
    sess, drv = started(pdir, live)
    drv.run_until(mix_gt(8.0))
    n = len(recs(sess))
    assert [r.cue_id for r in recs(sess)] == [1, 2, 3, 4]
    sess.jump_cue(2)                                        # back: cue 2 (3.5 s) fires again now
    new = recs(sess)[n:]
    assert [(r.cue_id, r.sent) for r in new] == [(2, True)]
    assert sess.snapshot()["position"] == pytest.approx(3.5, abs=0.01)
    n = len(recs(sess))
    sess.jump_cue(8)                                        # forward to 15.5 s over cues 3..7
    new = recs(sess)[n:]
    assert [(r.cue_id, r.reason, r.sent) for r in new] == [
        (7, "chase", True), (3, "skipped", False), (4, "skipped", False), (5, "skipped", False),
        (6, "skipped", False), (8, "track", True)]
    with pytest.raises(UnknownCueError) as ei:
        sess.jump_cue(99)
    assert isinstance(ei.value, KeyError) and isinstance(ei.value, ValueError)
    assert str(ei.value) == "no cue 99 in the cue file"
    with pytest.raises(ValueError):
        sess.jump_time(float("nan"))


def test_nudge_forward_fires_crossed_cue_backward_refires_nothing(mix_song) -> None:
    pdir, live = mix_song
    sess, drv = started(pdir, live)
    drv.run_until(mix_gt(5.2))                              # cue 3 (5.5 s) is 0.3 s ahead
    assert [r.cue_id for r in recs(sess)] == [1, 2]
    sess.nudge(0.5)
    last = recs(sess)[-1]
    assert (last.cue_id, last.reason, last.sent) == (3, "catchup", True)
    sess.nudge(-0.5)                                        # back before cue 3: it does not fire again
    drv.run_until(mix_gt(8.0))
    assert [r.cue_id for r in recs(sess)] == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        sess.nudge(float("inf"))


def test_disarmed_dry_run_then_armed(mix_song) -> None:
    pdir, live = mix_song
    sess, drv = started(pdir, live, settings=Settings(armed=False))
    drv.run_until(mix_gt(4.0))
    assert [(r.cue_id, r.sent, r.dry_run) for r in recs(sess)] == [(1, False, True), (2, False, True)]
    assert sess.sender.sent == [] and sess.snapshot()["armed"] is False
    assert sess.runner.counts()["dry"] == 2
    sess.set_armed(True)
    drv.run_until(mix_gt(6.0))
    assert [(r.cue_id, r.sent, r.dry_run) for r in recs(sess)][2:] == [(3, True, False)]
    assert len(sess.sender.sent) == 1 and sess.snapshot()["armed"] is True and sess.settings.armed


def test_manual_fire_ignores_arming(mix_song) -> None:
    pdir, _ = mix_song
    sess, _, _ = make_session(pdir, settings=Settings(armed=False))
    rec = sess.fire_cue(5)
    assert (rec.cue_id, rec.reason, rec.sent, rec.dry_run) == (5, "manual", True, False)
    assert sess.sender.sent[0][:3] == ("main", "/cue", (("i", 5),))
    with pytest.raises(UnknownCueError):
        sess.fire_cue(1000)


def test_hold_freezes_the_cues(mix_song) -> None:
    pdir, live = mix_song
    sess, drv = started(pdir, live)
    drv.run_until(mix_gt(6.0))
    assert [r.cue_id for r in recs(sess)] == [1, 2, 3]
    sess.hold(True)
    held_at = sess.snapshot()["position"]
    drv.run_until(mix_gt(6.0) + 2.5)                        # cue 4 (7.5 s) passes in the music meanwhile
    snap = sess.snapshot()
    assert snap["status"] == "hold" and snap["position"] == pytest.approx(held_at, abs=0.05)
    assert [r.cue_id for r in recs(sess)] == [1, 2, 3]
    sess.hold(False)
    drv.run_until(mix_gt(12.0))
    assert [r.cue_id for r in recs(sess)][:5] == [1, 2, 3, 4, 5]
    assert all(len(v) == 1 for v in by_cue(sess).values())


def test_stop_ends_firing(mix_song) -> None:
    pdir, live = mix_song
    sess, drv = started(pdir, live)
    drv.run_until(mix_gt(4.0))
    sess.stop()
    assert drv.src.stopped
    n = len(recs(sess))
    drv.src.stopped = False                                 # a straggling block after stop is ignored
    drv.run_until(mix_gt(10.0))
    assert len(recs(sess)) == n
    assert sess.tracker.state.status == "stopped" and not sess.running and sess.stopped
    snap = sess.snapshot()
    assert snap["status"] == "stopped" and not sess.runner.running
    sess.stop()                                             # idempotent
    with pytest.raises(RuntimeError):
        sess.start()


def test_snapshot_contract_is_strict_json(mix_song) -> None:
    pdir, live = mix_song
    sess, drv = started(pdir, live)
    drv.src.level_db = float("-inf")
    drv.run_until(mix_gt(6.2))
    snap = sess.snapshot()
    assert set(snap) == SNAPSHOT_KEYS
    json.dumps(snap, allow_nan=False)
    assert snap["type"] == "state" and snap["project_id"] == "mixsong" and snap["mode"] == "mix"
    assert snap["status"] == "tracking" and snap["level_db"] is None and snap["source"]["peak_db"] is None
    assert snap["position"] == pytest.approx(6.2, abs=0.15) and snap["tempo"] > 0
    assert [f["cue_id"] for f in snap["fired"]] == [1, 2, 3]            # newest last
    assert snap["fired_seq"] == 3 and snap["counts"] == {"sent": 3, "skipped": 0, "errors": 0, "dry": 0}
    assert set(snap["fired"][0]) >= {"seq", "cue_id", "time", "target", "message", "reason", "sent", "live_time"}
    up = snap["upcoming"]
    assert [u["id"] for u in up] == [4, 5, 6] and set(up[0]) == {"id", "time", "target", "message", "eta"}
    assert up[0]["target"] == "main" and up[0]["message"] == "/cue 4" and 0 < up[0]["eta"] < 1.6
    assert snap["input_latency_ms"] == 0.0 and snap["input_latency_reported_ms"] is None
    assert snap["targets"] == {"main": {"sent": 3, "errors": 0}} and snap["ended"] is False
    drv.src.finished = True
    snap = sess.snapshot()
    assert snap["status"] == "stopped" and snap["ended"] is True


def test_snapshot_before_start_and_reported_latency(mix_song) -> None:
    pdir, _ = mix_song
    sess, src, _ = make_session(pdir)
    src._stream = type("Stream", (), {"latency": 0.0123})()
    snap = sess.snapshot()
    assert set(snap) == SNAPSHOT_KEYS and snap["input_latency_reported_ms"] == 12.3
    assert [u["id"] for u in snap["upcoming"]] == [1, 2, 3] and snap["fired"] == []
    json.dumps(snap, allow_nan=False)
    assert json_safe({"a": [np.float32("nan"), (1.0, math.inf)], "b": np.int64(3)}) == {"a": [None, [1.0, None]],
                                                                                      "b": 3}


def test_settings_latency_default_target_and_input_latency(mix_song) -> None:
    pdir, live = mix_song
    s = Settings(targets=[Target("main", "127.0.0.1", 9000), Target("video", "127.0.0.1", 9001, latency_ms=300)])
    sess, drv = started(pdir, live, settings=s)
    drv.run_until(mix_gt(4.0))
    s2 = dataclasses.replace(s.copy(), default_target="video", skip_policy="all", catchup_s=2.0,
                             input_latency_ms=40.0)
    sess.apply_settings(s2)
    assert sess.input_latency_ms == 40.0 and sess.tracker.params.latency_s == pytest.approx(0.04)
    assert sess.runner._sched.params.skip_policy == "all" and sess.runner._sched.params.catchup_s == 2.0
    drv.run_until(mix_gt(8.0))
    late = {r.cue_id: r for r in recs(sess)}
    assert late[3].target == "video" and sess.sender.sent[2][0] == "video"
    # cue 3 and 4 go to "video" now: sent ~300 ms (+ the 40 ms input look-ahead) early
    for cid, t in ((3, 5.5), (4, 7.5)):
        assert late[cid].live_time - mix_gt(t) == pytest.approx(-0.34, abs=0.1)
    sess.set_input_latency(0)
    assert sess.snapshot()["input_latency_ms"] == 0.0 and sess.tracker.params.latency_s == 0.0


def test_set_cues_while_running(mix_song) -> None:
    pdir, live = mix_song
    sess, drv = started(pdir, live)
    drv.run_until(mix_gt(6.0))
    text = MIX_CUES + "4.000 /new behind\n9.000 /new ahead\n"
    sess.set_cues(parse_cues(text).cues)
    drv.run_until(mix_gt(10.0))
    got = [(r.cue_id, r.address) for r in recs(sess)]
    assert got == [(1, "/cue"), (2, "/cue"), (3, "/cue"), (4, "/cue"), (11, "/new"), (5, "/cue")]


def test_vocal_guard_waits_for_the_voice(vocal_song) -> None:
    """A cue at a vocal entry fires when the singer comes back, not when the coasting position reaches it."""
    pdir, live = vocal_song
    sess, drv = started(pdir, live, "vocal")
    entry = 7.0                                             # /b; the singer resumes at live 10.0
    positions = []
    while drv.t < 9.8:
        drv.run_until(drv.t + 0.1)
        positions.append(sess.snapshot()["position"])
    snap = sess.snapshot()
    assert snap["status"] == "coasting" and max(positions) > entry - 0.1   # the position waits at the entry
    fired = by_cue(sess)
    assert list(fired) == [1]                               # /a only: /b is held
    up = snap["upcoming"]
    assert up[0]["id"] == 2 and up[0]["eta"] is None
    drv.run_until(12.5)
    fired = by_cue(sess)
    assert list(fired)[:3] == [1, 2, 3]
    b = fired[2][0]
    assert 10.0 <= b.live_time <= 10.4 and b.sent
    assert fired[3][0].live_time == pytest.approx(float(S.score_to_perf(9.0, VOCAL_TM)), abs=0.25)


def test_vocal_jump_to_cue_fires_while_the_singer_is_silent(vocal_song) -> None:
    pdir, live = vocal_song
    sess, drv = started(pdir, live, "vocal")
    drv.run_until(8.0)
    assert [r.cue_id for r in recs(sess)] == [1]
    sess.jump_cue(4)                                        # /d at 15 s: the operator's go fires it now
    # /d fires as the jump target, /c (another address) is chased by the "latest" policy, and /b, held at
    # the entry, is no longer held (chased too when the coasting position had not quite reached it)
    new = {r.cue_id: r.reason for r in recs(sess)[1:]}
    assert set(new) == {2, 3, 4} and all(r.sent for r in recs(sess))
    assert new[2] in ("chase", "catchup") and new[3] == "chase" and new[4] == "track"
    pt = sess.runner._sched.point
    assert pt.hold_from is None and pt.ref_time == pytest.approx(15.0, abs=1e-6)


@pytest.mark.parametrize("stop_at,pause", [(8.5, 2.0), (8.5, 1.5), (8.0, 2.0), (8.0, 1.0)])
def test_vocal_pause_mid_phrase_sends_no_cue_twice(vocal_song, stop_at: float, pause: float) -> None:
    """The singer stops mid-phrase and carries on where they stopped: the tracker coasts on meanwhile (the
    cues on the coasted stretch fire), then the relock settles back over the coasted positions (at 8.0 s only
    after its provisional run along the coasting path: the points say `settling`). That undoes an overshoot,
    it is no new pass: no cue is sent twice, none is missed."""
    pdir, _ = vocal_song
    vocals = S.render_song(vocal_score())
    a = round(stop_at * SR)
    live = np.concatenate([vocals[:a], np.zeros(round(pause * SR), np.float32), vocals[a:]])
    live = (live + 1e-3 * np.random.default_rng(0).standard_normal(len(live))).astype(np.float32)
    sess, src, clock = make_session(pdir, "vocal")
    times = [3.0, 7.0, 8.0, 9.0, 9.5, 10.0, 10.5, 13.0, 15.0]
    sess.set_cues(parse_cues("".join(f"{t:.3f} /c {k}\n" for k, t in enumerate(times, 1))).cues)
    sess.start()
    backs = []
    feed = sess.runner.feed

    def spy(pt) -> None:
        p = sess.runner._sched.point
        if p is not None and p.ref_time - pt.ref_time >= 1.0:
            backs.append((p.ref_time, pt.ref_time))
        feed(pt)

    sess.runner.feed = spy
    Driver(sess, src, live, clock).run_until(len(live) / SR)
    assert backs, "the relock no longer moves back over the coasted stretch: the test lost its point"
    sent = [r.cue_id for r in recs(sess) if r.sent]
    assert sorted(sent) == list(range(1, len(times) + 1))


@pytest.mark.parametrize("latency_ms", [400.0, -300.0])
def test_input_latency_at_start_and_operator_jump(mix_song, latency_ms: float) -> None:
    """The tracker publishes a start / jump target plus the input-latency look-ahead. The cues from the requested
    time to there fire at once, none is jumped over (here the look-ahead is longer than the catch-up window);
    with a negative input latency the jump target still fires at once."""
    pdir, live = mix_song
    st = Settings(skip_policy="skip", catchup_s=0.3, input_latency_ms=latency_ms)
    sess, src, clock = make_session(pdir, settings=st, start_time=5.0)
    sess.set_cues(parse_cues("5.000 /a\n5.150 /b\n5.300 /c\n5.600 /d\n8.000 /e\n"
                             "10.000 /f\n10.150 /g\n10.300 /h\n10.600 /i\n").cues)
    drv = Driver(sess, src, live, clock)
    drv.i = round(mix_gt(5.0) * SR)                         # the band plays from 5.0 s
    clock.now = drv.t
    sess.start()
    if latency_ms > 0:
        assert [(r.cue_id, r.reason, r.sent) for r in recs(sess)] == [(1, "track", True), (2, "track", True),
                                                                     (3, "track", True)]
    else:
        assert [(r.cue_id, r.reason, r.sent) for r in recs(sess)] == [(1, "track", True)]
    drv.run_until(mix_gt(7.0))
    assert [r.cue_id for r in recs(sess)] == [1, 2, 3, 4]
    n = len(recs(sess))
    sess.jump_cue(6)                                        # Go to cue 6 (10.0 s); the band goes there too
    new = [(r.cue_id, r.reason, r.sent) for r in recs(sess)[n:]]
    if latency_ms > 0:
        assert new == [(5, "skipped", False), (6, "track", True), (7, "track", True), (8, "track", True)]
    else:
        assert new == [(5, "skipped", False), (6, "track", True)]
    drv.i = round(mix_gt(10.0) * SR)
    drv.run_until(mix_gt(11.5))
    assert [r.cue_id for r in recs(sess) if r.sent] == [1, 2, 3, 4, 6, 7, 8, 9]


def test_hold_from_rules() -> None:
    @dataclasses.dataclass
    class St:
        status: str
        ref_time: float
        active: bool

    class Trk:
        onsets = np.array([24, 99, 174])
        _free_run = False

    fps = 25.0
    assert hold_from(Trk, "vocal", fps, St("coasting", 2.0, False)) == pytest.approx(4.0 - 0.1)
    assert hold_from(Trk, "vocal", fps, St("tracking", 4.0, False)) == pytest.approx(4.0 - 0.1)  # at the onset
    assert hold_from(Trk, "vocal", fps, St("tracking", 4.0, True)) is None
    assert hold_from(Trk, "mix", fps, St("coasting", 2.0, False)) is None
    assert hold_from(Trk, "vocal", fps, St("coasting", 7.5, False)) is None                      # no onset left
    Trk._free_run = True
    assert hold_from(Trk, "vocal", fps, St("coasting", 2.0, False)) is None


def test_show_log_csv(mix_song, tmp_path: Path) -> None:
    pdir, live = mix_song
    seen = []
    sess, drv = started(pdir, live, settings=Settings(armed=False), log_dir=tmp_path / "logs", on_fire=seen.append)
    drv.run_until(mix_gt(6.0))
    sess.fire_cue(9)
    sess.stop()
    files = list((tmp_path / "logs").glob("show-*.csv"))
    assert len(files) == 1 and files[0] == sess.show_log.path
    rows = list(csv.reader(files[0].open(encoding="utf-8", newline="")))
    assert tuple(rows[0]) == LOG_COLUMNS
    body = [dict(zip(LOG_COLUMNS, r)) for r in rows[1:]]
    assert [(r["cue_id"], r["sent"], r["dry_run"]) for r in body] == [
        ("1", "false", "true"), ("2", "false", "true"), ("3", "false", "true"), ("9", "true", "false")]
    assert [r["reason"] in ("track", "catchup") for r in body[:3]] == [True] * 3 and body[3]["reason"] == "manual"
    assert body[0]["cue_time"] == "1.500" and body[0]["target"] == "main" and body[0]["message"] == "/cue 1"
    assert body[0]["address"] == "/cue" and body[0]["error"] == "" and float(body[0]["ref_late_ms"]) >= 0
    assert abs(float(body[0]["live_time"]) - mix_gt(1.5)) < 0.1 and "T" in body[0]["wall_iso"]
    assert [r.cue_id for r in seen] == [1, 2, 3, 9]
    sess2, _, _ = make_session(pdir, log_dir=tmp_path / "logs")
    sess2.start()
    sess2.stop()
    assert len(list((tmp_path / "logs").glob("show-*.csv"))) == 2   # a second session in the same second


def test_tracking_errors_are_reported_not_raised(mix_song) -> None:
    pdir, live = mix_song
    deps = default_deps()
    real = deps.tracker

    def tracker(*a, **kw):
        t = real(*a, **kw)
        push = t.push

        def flaky(frames):
            if t.state.live_time > 2.0:
                raise RuntimeError("diverged")
            return push(frames)

        t.push = flaky
        return t

    sess, drv = started(pdir, live, deps=dataclasses.replace(deps, tracker=tracker))
    drv.run_until(4.0)
    assert sess.error == "RuntimeError: diverged" and sess.snapshot()["error"] == "RuntimeError: diverged"


def test_align_source_start(mix_song) -> None:
    pdir, _ = mix_song
    sess, src, _ = make_session(pdir, start_time=9.0)
    assert sess.align_source_start(2.0) == 7.0 and src.start_s == 7.0 and sess.start_time == 7.0
    sess.start()
    with pytest.raises(RuntimeError):
        sess.align_source_start(1.0)
    sess.stop()


def test_operator_controls_from_another_thread_never_deadlock(mix_song) -> None:
    """Lock order: the audio thread feeds the runner under the session lock while the timer thread fires,
    the operator jumps / nudges / holds / edits and the broadcaster snapshots."""
    pdir, live = mix_song
    sess, src, _ = make_session(pdir, timer=True, clock=time.perf_counter, log_dir=None)
    sess.start()
    done = threading.Event()
    errors: list[BaseException] = []

    def audio() -> None:
        try:
            for i in range(0, len(live), BLOCK):
                src.push(live[i:i + BLOCK])
                time.sleep(0.0002)
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)
        finally:
            done.set()

    def operator() -> None:
        rng = np.random.default_rng(1)
        cues = sess.cues
        try:
            while not done.is_set():
                k = int(rng.integers(0, 9))
                if k == 0:
                    sess.jump_time(float(rng.uniform(0, 18)))
                elif k == 1:
                    sess.nudge(float(rng.choice([-0.5, 0.5])))
                elif k == 2:
                    sess.hold(bool(rng.integers(0, 2)))
                elif k == 3:
                    sess.set_cues(cues)
                elif k == 4:
                    sess.apply_settings(Settings(skip_policy=str(rng.choice(["skip", "latest", "all"]))))
                elif k == 5:
                    sess.fire_cue(int(rng.integers(1, 10)))
                elif k == 6:
                    sess.set_style(str(rng.choice(["classic", "pop"])))
                elif k == 7:
                    sess.jump_cue(int(rng.integers(1, 10)))
                json.dumps(sess.snapshot(), allow_nan=False)
                time.sleep(0.001)
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)

    threads = [threading.Thread(target=audio), threading.Thread(target=operator),
               threading.Thread(target=operator)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=60)
    alive = [th for th in threads if th.is_alive()]
    stopper = threading.Thread(target=sess.stop, daemon=True)
    stopper.start()
    stopper.join(timeout=10)
    assert not alive and not stopper.is_alive(), "deadlock"
    assert errors == [] and sess.error is None
    assert len(recs(sess)) > 0
