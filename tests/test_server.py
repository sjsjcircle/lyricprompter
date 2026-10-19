"""FastAPI server: REST + WebSocket with a fake live session and a fake `prep` subprocess."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from lyricprompter.live.session import package_id
from lyricprompter.server import app as app_mod
from lyricprompter.server import jobs as jobs_mod
from lyricprompter.server.app import Hub, OriginGuard, create_app, move_line
from lyricprompter.server.jobs import JobManager, build_prep_argv, parse_progress
from lyricprompter.songfile import load_song, save_song
from lyricprompter.text.model import AlignUnit, Line, Song, SongMeta, Syllable, Word

STATIC = Path(app_mod.__file__).parent / "static"
TEST_HOSTS = ("testserver",)          # Starlette's TestClient sends Host: testserver

# ----------------------------------------------------------------------------- fixtures


def make_line(idx: int, text: str, start: float | None, dur: float = 0.8, review: bool = False) -> Line:
    syls, step = [], dur / len(text)
    for k, ch in enumerate(text):
        t0 = None if start is None else round(start + k * step, 4)
        t1 = None if start is None else round(start + (k + 1) * step, 4)
        syls.append(Syllable(text=ch, span=(k, k + 1), units=[AlignUnit(ch, "a", t0, t1, 0.5)], start=t0, end=t1))
    word = Word(text=text, span=(0, len(text)), lang="ko", syllables=syls)
    end = None if start is None else round(start + dur, 4)
    return Line(idx=idx, text=text, words=[word], start=start, end=end, confidence=0.05 if review else 0.8,
                needs_review=review)


@pytest.fixture
def song_root(tmp_path: Path) -> Path:
    root = tmp_path / "songs"
    sdir = root / "demo"
    song = Song(meta=SongMeta(id="demo", title="Demo song", artist="Tester", default_lang="ko", duration=4.0),
                lines=[make_line(0, "꽃밭에", 0.5), make_line(1, "앉아서", 1.5, review=True),
                       make_line(2, "보니", 2.5)])
    save_song(song, sdir)
    (sdir / "audio").mkdir()
    sr = 16000
    sf.write(sdir / "audio" / "mix.wav", (0.1 * np.sin(np.arange(sr * 2) * 0.05)).astype(np.float32), sr,
             subtype="PCM_16")
    return root


FAKE_PREP = textwrap.dedent('''
    import argparse, json, os, pathlib, sys, time
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    for flag in ("--audio", "--lyrics", "--title", "--id", "--songs-dir", "--lang", "--artist", "--refine",
                 "--sep-model"):
        ap.add_argument(flag)
    ap.add_argument("--lead", action="store_true")
    a = ap.parse_args()
    sdir = pathlib.Path(a.songs_dir) / a.id
    sdir.mkdir(parents=True, exist_ok=True)               # B1 works in place: partial package first
    (sdir / "prep_report.json").write_text('{"status": "running"}')
    print(f"PID {os.getpid()}", flush=True)
    for pct, stage in ((0, "convert"), (40, "separate"), (90, "align")):
        print(f"PROGRESS {stage} {pct} working on {stage}", flush=True)
        time.sleep(0.02)
    if "FAIL" in a.title:
        print("error: RuntimeError: separation exploded", file=sys.stderr, flush=True)
        sys.exit(1)
    if "NOJSON" in a.title:
        sys.exit(0)
    if "BOOM" in a.title:
        print("BOOM", flush=True)
    if "SLOW" in a.title or "BOOM" in a.title:
        time.sleep(30)
    lyrics = pathlib.Path(a.lyrics).read_text(encoding="utf-8")
    (sdir / "argv.json").write_text(json.dumps(sys.argv[1:]))
    (sdir / "song.json").write_text(json.dumps({"schema": 1, "meta": {"id": a.id, "title": a.title},
                                                "lines": [{"idx": 0, "text": lyrics.splitlines()[0]}]}))
    print("PROGRESS done 100 finished", flush=True)
    print(sdir, flush=True)
''')


class FakeSession:
    def __init__(self, sdir: Path, req: app_mod.SessionStart):
        self.song_id, self.req = package_id(sdir), req
        self.calls: list[tuple] = []
        self.reloaded: Song | None = None
        self.status, self.ref_time = "waiting", 0.0
        self.started = self.stopped = False

    def start(self) -> None:
        self.started, self.status = True, "tracking"

    def stop(self) -> None:
        self.stopped, self.status = True, "stopped"

    def hold(self, on: bool) -> None:
        self.calls.append(("hold", on))
        self.status = "hold" if on else "tracking"

    def jump_line(self, idx: int) -> None:
        self.calls.append(("jump_line", idx))
        self.ref_time = [0.5, 1.5, 2.5][idx]

    def jump_time(self, t: float) -> None:
        self.calls.append(("jump_time", t))
        self.ref_time = t

    def nudge(self, dt: float) -> None:
        self.calls.append(("nudge", dt))
        self.ref_time += dt

    def set_transpose(self, k: int) -> None:
        self.calls.append(("transpose", k))

    def set_auto_key(self, on: bool) -> None:
        self.calls.append(("autokey", on))

    def set_latency(self, s: float) -> None:
        self.calls.append(("latency", s))

    def reload_lines(self, song: Song) -> None:
        self.calls.append(("reload", song.lines[0].start))
        self.reloaded = song

    def snapshot(self) -> dict:
        return {"type": "state", "song_id": self.song_id, "mode": self.req.mode, "status": self.status,
                "ref_time": self.ref_time, "tempo": 1.0, "confidence": 0.8, "live_time": 1.0, "level_db": -20.0,
                "line_idx": 0, "syl_idx": 0, "t": time.time(), "age": 0.0}


class Factory:
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []
        self.fail: Exception | None = None

    def __call__(self, sdir: Path, req: app_mod.SessionStart, songs_root: Path) -> FakeSession:
        if self.fail is not None:
            raise self.fail
        s = FakeSession(sdir, req)
        self.sessions.append(s)
        return s


DEVICES = [{"index": 2, "name": "Stage Interface", "channels": 4, "default_samplerate": 48000.0,
            "hostapi": "Core Audio", "is_default": True}]


@pytest.fixture
def server(song_root: Path, tmp_path: Path):
    script = tmp_path / "fake_prep.py"
    script.write_text(FAKE_PREP)
    factory = Factory()
    device_calls: list[bool] = []

    def devices(refresh: bool) -> list[dict]:
        device_calls.append(refresh)
        if refresh == "boom":
            raise RuntimeError
        return DEVICES

    jobs = JobManager(command_prefix=[sys.executable, str(script)])
    app = create_app(song_root, session_factory=factory, jobs=jobs, devices=devices, allowed_hosts=TEST_HOSTS)
    with TestClient(app) as client:
        client.factory, client.hub, client.root, client.device_calls = factory, app.state.hub, song_root, device_calls
        yield client


def wait_job(client: TestClient, job_id: str, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled") and job["finished"]:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {job}")


def upload(client: TestClient, title: str = "New Song", **extra) -> dict:
    data = {"title": title, "lyrics": "첫 줄 가사\n둘째 줄", "artist": "Band", "language": "ko",
            "options": json.dumps({"lead": True, "refine": "none"}), **extra}
    return client.post("/api/songs", data=data, files={"audio": ("take 1.m4a", b"\x00" * 2048, "audio/mp4")})


# ----------------------------------------------------------------------------- pages / songs


def test_pages_and_static_assets(server: TestClient) -> None:
    for path, marker in (("/", "lyricprompter"), ("/prompter", 'id="reel"')):
        r = server.get(path)
        assert r.status_code == 200 and "text/html" in r.headers["content-type"] and marker in r.text
    for asset in ("common.js", "console.js", "prompter.js", "console.css", "prompter.css"):
        r = server.get(f"/static/{asset}")
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache", asset
        assert server.get(f"/static/{asset}", headers={"If-None-Match": r.headers["etag"]}).status_code == 304
    # offline on stage: no external resources anywhere in the UI
    for f in STATIC.iterdir():
        text = f.read_text(encoding="utf-8")
        assert "http://" not in text and "https://" not in text, f.name


def test_song_list_and_detail(server: TestClient) -> None:
    (server.root / "broken").mkdir()
    (server.root / "broken" / "song.json").write_text("{not json")
    songs = {s["id"]: s for s in server.get("/api/songs").json()}
    demo = songs["demo"]
    assert demo["title"] == "Demo song" and demo["n_lines"] == 3 and demo["n_review"] == 1
    assert demo["stems"] == ["mix"] and demo["aligned"] is True and demo["mean_confidence"] == pytest.approx(0.55)
    assert "error" in songs["broken"]
    d = server.get("/api/songs/demo").json()
    assert d["meta"]["title"] == "Demo song" and len(d["lines"]) == 3 and d["rev"] == 0
    assert d["lines"][1]["words"][0]["syllables"][0]["start"] == 1.5
    assert server.get("/api/songs/nope").status_code == 404
    assert server.get("/api/songs/.hidden").status_code == 400


def test_audio_supports_range_requests(server: TestClient) -> None:
    full = server.get("/api/songs/demo/audio/mix")
    assert full.status_code == 200 and full.headers["content-type"] == "audio/wav"
    assert full.headers.get("accept-ranges") == "bytes"
    size = len(full.content)
    part = server.get("/api/songs/demo/audio/mix", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206
    assert part.headers["content-range"] == f"bytes 100-199/{size}"
    assert part.content == full.content[100:200]
    tail = server.get("/api/songs/demo/audio/mix", headers={"Range": "bytes=-44"})
    assert tail.status_code == 206 and tail.content == full.content[-44:]
    assert server.get("/api/songs/demo/audio/vocals").status_code == 404
    assert server.get("/api/songs/demo/audio/passwd").status_code == 404


# ----------------------------------------------------------------------------- upload + jobs


def test_upload_runs_prep_job(server: TestClient) -> None:
    r = upload(server)
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["song_id"] == "new-song"
    job = wait_job(server, body["job"]["id"])
    assert job["status"] == "done" and job["percent"] == 100 and job["returncode"] == 0
    assert job["stage"] == "done"
    argv = json.loads((server.root / "new-song" / "argv.json").read_text())
    assert argv[0] == "prep"
    opts = dict(a.split("=", 1) for a in argv[1:] if "=" in a)
    assert opts["--title"] == "New Song" and opts["--id"] == "new-song" and opts["--lang"] == "ko"
    assert opts["--songs-dir"] == str(server.root) and opts["--artist"] == "Band" and "--lead" in argv
    assert Path(opts["--audio"]).suffix == ".m4a"
    assert not (server.root / ".uploads").exists() or not any((server.root / ".uploads").iterdir())
    ids = [s["id"] for s in server.get("/api/songs").json()]
    assert "new-song" in ids and ".uploads" not in ids
    assert any(j["id"] == job["id"] for j in server.get("/api/jobs").json())
    # same title again gets a fresh id
    r2 = upload(server)
    assert r2.json()["song_id"] == "new-song-2"
    wait_job(server, r2.json()["job"]["id"])


def test_upload_validation(server: TestClient) -> None:
    assert upload(server, title="  ").status_code == 422
    assert upload(server, language="fr").status_code == 422
    assert upload(server, options="[1]").status_code == 422
    assert upload(server, options="{bad").status_code == 422
    assert upload(server, options=json.dumps({"refine": "maybe"})).status_code == 422
    assert upload(server, song_id="demo").status_code == 409
    assert upload(server, song_id="../x").status_code == 400
    r = server.post("/api/songs", data={"title": "x", "lyrics": "a"}, files={"audio": ("a.wav", b"", "audio/wav")})
    assert r.status_code == 422


def test_failed_and_cancelled_jobs(server: TestClient) -> None:
    job = wait_job(server, upload(server, title="FAIL song").json()["job"]["id"])
    assert job["status"] == "failed" and "separation exploded" in job["error"]
    assert any("separation exploded" in line for line in job["log"])
    jid = upload(server, title="SLOW song").json()["job"]["id"]
    deadline = time.monotonic() + 5
    while server.get(f"/api/jobs/{jid}").json()["stage"] != "align" and time.monotonic() < deadline:
        time.sleep(0.05)
    queued = upload(server, title="Queued song").json()["job"]["id"]
    assert server.get(f"/api/jobs/{queued}").json()["status"] == "queued"
    assert server.delete(f"/api/jobs/{queued}").json() == {"cancelled": True}
    assert server.delete(f"/api/jobs/{jid}").json() == {"cancelled": True}
    assert wait_job(server, jid)["status"] == "cancelled"
    assert server.get(f"/api/jobs/{queued}").json()["status"] == "cancelled"
    assert server.delete("/api/jobs/nope").status_code == 404


def test_failed_or_cancelled_job_removes_its_partial_package(server: TestClient) -> None:
    root = server.root
    job = wait_job(server, upload(server, title="FAIL song").json()["job"]["id"])
    assert job["status"] == "failed" and not (root / "fail-song").exists()
    job = wait_job(server, upload(server, title="NOJSON song").json()["job"]["id"])
    assert job["status"] == "failed" and "song.json" in job["error"] and not (root / "nojson-song").exists()
    jid = upload(server, title="SLOW song").json()["job"]["id"]
    deadline = time.monotonic() + 5
    while not (root / "slow-song" / "prep_report.json").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert (root / "slow-song").is_dir()                                   # partial package while running
    server.delete(f"/api/jobs/{jid}")
    assert wait_job(server, jid)["status"] == "cancelled" and not (root / "slow-song").exists()
    # the ids are free again: a retry neither gets "-2" nor a 409
    r = upload(server, title="Slow song")
    assert r.json()["song_id"] == "slow-song"
    assert wait_job(server, r.json()["job"]["id"])["status"] == "done"
    r = upload(server, title="Retry", song_id="fail-song")
    assert r.status_code == 202 and wait_job(server, r.json()["job"]["id"])["status"] == "done"
    assert {"slow-song", "fail-song"} <= {s["id"] for s in server.get("/api/songs").json()}


def test_failed_job_keeps_a_package_that_existed_before(tmp_path: Path) -> None:
    out = tmp_path / "songs" / "kept"
    out.mkdir(parents=True)
    (out / "audio.wav").write_bytes(b"stems from an earlier run")
    jm = JobManager(command_prefix=[sys.executable, "-c", "import sys; sys.exit(3)"])
    try:
        job = jm.wait(jm.submit([], output=out).id, timeout=10)
        assert job is not None and job.status == "failed" and job.returncode == 3
        assert (out / "audio.wav").is_file()
    finally:
        jm.shutdown()


def test_job_output_error_kills_the_process_before_the_next_job(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "fake_prep.py"
    script.write_text(FAKE_PREP)
    real_parse = jobs_mod.parse_progress

    def parse(line: str):
        if line == "BOOM":
            raise RuntimeError("output handling exploded")
        return real_parse(line)
    monkeypatch.setattr(jobs_mod, "parse_progress", parse)
    root = tmp_path / "songs"
    jm = JobManager(command_prefix=[sys.executable, str(script)])
    try:
        def argv(title: str, sid: str) -> list[str]:
            return build_prep_argv(tmp_path / "a.wav", tmp_path / "l.txt", title=title, song_id=sid, songs_dir=root)
        (tmp_path / "l.txt").write_text("line one", encoding="utf-8")
        boom = jm.submit(argv("BOOM", "boom"), output=root / "boom")
        nxt = jm.submit(argv("Next", "next"), output=root / "next")
        b = jm.wait(boom.id, timeout=15)
        assert b.status == "failed" and "output handling exploded" in b.error
        pid = int(next(ln for ln in b.log if ln.startswith("PID "))[4:])
        with pytest.raises(ProcessLookupError):           # killed and reaped, not left sleeping for 30 s
            os.kill(pid, 0)
        assert not (root / "boom").exists()
        n = jm.wait(nxt.id, timeout=15)
        assert n.status == "done" and n.started >= b.finished
    finally:
        jm.shutdown()


def test_parse_progress_and_argv() -> None:
    assert parse_progress("PROGRESS align 42.5 aligning line 3/10") == ("align", 42.5, "aligning line 3/10")
    assert parse_progress("PROGRESS separate 140") == ("separate", 100.0, "")
    assert parse_progress("loading model...") is None
    assert parse_progress("PROGRESS align") is None                      # malformed: kept as a log line
    argv = build_prep_argv(Path("/a.wav"), Path("/l.txt"), title="T", song_id="t", songs_dir=Path("/s"))
    assert argv == ["prep", "--audio=/a.wav", "--lyrics=/l.txt", "--title=T", "--id=t", "--songs-dir=/s",
                    "--lang=auto"]
    argv = build_prep_argv(Path("/a.wav"), Path("/l.txt"), title="-Intro-", song_id="t", songs_dir=Path("/s"),
                           artist="-M-", lead=True, refine="none", sep_model="m.ckpt")
    assert argv[3] == "--title=-Intro-" and "--artist=-M-" in argv and argv[-1] == "--lead"
    assert "--refine=none" in argv and "--sep-model=m.ckpt" in argv


def test_prep_argv_matches_pipeline_cli() -> None:
    pipeline = pytest.importorskip("lyricprompter.prep.pipeline")
    argv = build_prep_argv(Path("/a.wav"), Path("/l.txt"), title="-Intro-", song_id="t", songs_dir=Path("/s"),
                           artist="-M-", language="ja", lead=True, refine="qwen", sep_model="m.ckpt")
    args = pipeline.build_parser().parse_args(argv[1:])
    assert (args.audio, args.lyrics, args.title, args.song_id, args.songs_dir) == (
        Path("/a.wav"), "/l.txt", "-Intro-", "t", Path("/s"))
    assert (args.artist, args.lang, args.lead, args.refine, args.sep_model) == ("-M-", "ja", True, "qwen", "m.ckpt")


# ----------------------------------------------------------------------------- devices / session


def test_devices(server: TestClient) -> None:
    assert server.get("/api/devices").json() == DEVICES
    assert server.get("/api/devices?refresh=true").status_code == 200
    assert server.device_calls == [False, True]


def test_devices_error_is_503(song_root: Path) -> None:
    def broken(refresh: bool) -> list[dict]:
        raise OSError("PortAudio not initialized")
    app = create_app(song_root, session_factory=Factory(), jobs=JobManager(), devices=broken,
                     allowed_hosts=TEST_HOSTS)
    with TestClient(app) as client:
        r = client.get("/api/devices")
    assert r.status_code == 503 and "PortAudio" in r.json()["detail"]


def test_session_lifecycle_and_controls(server: TestClient) -> None:
    assert server.post("/api/session/hold", json={"on": True}).status_code == 409
    assert server.get("/api/session").json()["state"]["status"] == "idle"
    body = {"song_id": "demo", "mode": "vocal", "lead": True, "start_line": 1, "latency": 0.1,
            "input": {"kind": "device", "device": 2, "channels": [1, 2]}}
    st = server.post("/api/session/start", json=body).json()
    assert st["status"] == "tracking" and st["song_id"] == "demo"
    sess = server.factory.sessions[0]
    assert sess.started and sess.req.mode == "vocal" and sess.req.input.channels == [1, 2]
    assert server.get("/api/session").json()["session"]["start_line"] == 1
    assert server.post("/api/session/hold", json={"on": True}).json()["status"] == "hold"
    assert server.post("/api/session/resume").json()["status"] == "tracking"
    assert server.post("/api/session/jump", json={"line": 2}).json()["ref_time"] == 2.5
    assert server.post("/api/session/jump", json={"time": 1.25}).json()["ref_time"] == 1.25
    assert server.post("/api/session/jump", json={}).status_code == 422
    assert server.post("/api/session/jump", json={"line": 9}).status_code == 404
    server.post("/api/session/nudge", json={"dt": -0.5})
    server.post("/api/session/transpose", json={"semitones": 2})
    server.post("/api/session/latency", json={"seconds": 0.15})
    assert server.post("/api/session/transpose", json={"semitones": 40}).status_code == 422
    assert server.post("/api/session/autokey", json={"on": True}).status_code == 200
    assert ("nudge", -0.5) in sess.calls and ("transpose", 2) in sess.calls and ("latency", 0.15) in sess.calls
    assert sess.calls[-1] == ("autokey", True)
    assert server.get("/api/devices?refresh=true").status_code == 409
    # restarting replaces the running session
    server.post("/api/session/start", json={"song_id": "demo", "input": {"kind": "file", "stem": "mix"}})
    assert sess.stopped and len(server.factory.sessions) == 2
    st = server.post("/api/session/stop").json()
    assert st["status"] == "idle" and st["song_id"] == "demo" and server.factory.sessions[1].stopped


def test_session_start_errors(server: TestClient) -> None:
    assert server.post("/api/session/start", json={"song_id": "nope"}).status_code == 404
    server.factory.fail = ValueError("channels [9] out of range")
    r = server.post("/api/session/start", json={"song_id": "demo"})
    assert r.status_code == 400 and "out of range" in r.json()["detail"]
    server.factory.fail = FileNotFoundError("vocals.wav")
    assert server.post("/api/session/start", json={"song_id": "demo"}).status_code == 404
    server.factory.fail = ImportError("no module named lyricprompter.tracking.tracker")
    assert server.post("/api/session/start", json={"song_id": "demo"}).status_code == 503
    assert server.hub.session is None


def test_manual_cue_and_jump_without_session(server: TestClient) -> None:
    assert server.post("/api/session/jump", json={"line": 1}).status_code == 409
    assert server.post("/api/session/cue", json={"song_id": "demo"}).json()["status"] == "idle"
    st = server.post("/api/session/jump", json={"line": 1}).json()
    assert st["status"] == "manual" and st["ref_time"] == 1.5 and st["line_idx"] == 1
    st = server.post("/api/session/nudge", json={"dt": 1.0}).json()
    assert st["ref_time"] == 2.5 and st["line_idx"] == 2
    assert server.post("/api/session/cue", json={"song_id": "nope"}).status_code == 404


def test_default_session_factory_builds_sources(song_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built: dict = {}

    class FakeLive:
        def __init__(self, sdir, mode, source, **kw):
            built.update(sdir=sdir, mode=mode, source=source, **kw)
            self.latency = self.transpose = None
            self.auto_key = True
            self.aligned: list[float] = []

        def set_latency(self, s):
            self.latency = s

        def set_transpose(self, k):
            self.transpose, self.auto_key = k, False

        def set_auto_key(self, on):
            self.auto_key = on

        def align_source_start(self, preroll):
            self.aligned.append(preroll)

    import lyricprompter.live.session as live_session
    monkeypatch.setattr(live_session, "LiveSession", FakeLive)
    shutil.copytree(song_root / "demo", song_root / "other")
    demo = song_root / "demo"

    def build(transpose=-1, auto_key=True, **inp) -> FakeLive:
        start_line = inp.pop("start_line", None)
        req = app_mod.SessionStart(song_id="demo", mode="mix", latency=0.2, transpose=transpose, auto_key=auto_key,
                                   start_line=start_line, input=app_mod.InputSpec(kind="file", **inp))
        return app_mod.default_session_factory(demo, req, song_root)

    # key: with Auto on a given key only seeds the search; Auto off makes it the operator's key (even 0)
    s = build(stem="mix")
    assert (s.transpose, s.auto_key) == (-1, True)
    s = build(stem="mix", auto_key=False)
    assert (s.transpose, s.auto_key) == (-1, False)
    s = build(stem="mix", transpose=0, auto_key=False)
    assert (s.transpose, s.auto_key) == (0, False)
    s = build(stem="mix", transpose=0)
    assert (s.transpose, s.auto_key) == (None, True)

    sess = build(stem="mix", speed=1.5)
    assert built["source"].path == demo / "audio" / "mix.wav" and built["source"].speed == 1.5
    assert sess.latency == 0.2 and sess.transpose == -1 and built["start_line"] is None and sess.aligned == []
    # rehearsal from line N on the song's own stem: start the file 2 s before that line
    assert build(stem="mix", start_line=2).aligned == [app_mod.REHEARSAL_PREROLL]
    assert built["start_line"] == 2
    # ... but not when the file is another song's, an explicit offset, or an arbitrary path
    assert build(stem="mix", start_line=2, song_id="other").aligned == []
    assert built["source"].path == song_root / "other" / "audio" / "mix.wav"
    assert build(stem="mix", start_line=2, start_s=1.0).aligned == []
    assert build(path=str(demo / "audio" / "mix.wav"), start_line=2).aligned == []
    with pytest.raises(ValueError):
        build()
    with pytest.raises(ValueError):
        build(stem="karaoke")


# ----------------------------------------------------------------------------- line edits
# (push / squeeze semantics in depth, undo and the console's Lines panel: tests/test_line_edit.py)


def test_put_line_shift_and_start(server: TestClient) -> None:
    server.post("/api/session/start", json={"song_id": "demo"})
    r = server.put("/api/songs/demo/lines/1", json={"shift": 0.2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["shift"] == 0.2 and body["rev"] == 1 and body["clamped"] is False and set(body["lines"]) == {"1"}
    assert body["pushed"] == [] and body["undo_left"] == 1
    song = load_song(server.root / "demo")
    ln = song.lines[1]
    assert ln.start == pytest.approx(1.7) and ln.end == pytest.approx(2.5)        # ends where the next starts
    assert ln.words[0].syllables[0].start == pytest.approx(1.7)
    assert ln.words[0].syllables[0].units[0].start == pytest.approx(1.7)
    assert not ln.needs_review and "manual_timing" in ln.flags
    assert song.lines[0].start == 0.5 and song.lines[2].start == 2.5
    assert ("reload", 0.5) in server.factory.sessions[0].calls
    # a start inside the previous line (refused before): that line is squeezed, never collapsed
    r = server.put("/api/songs/demo/lines/1", json={"start": 1.0})
    assert r.status_code == 200 and r.json()["shift"] == pytest.approx(-0.7) and r.json()["pushed"] == []
    song = load_song(server.root / "demo")
    assert song.lines[1].start == 1.0 and song.lines[0].end == 1.0
    assert [(s.start, s.end) for s in song.lines[0].syllables()] == [
        (0.5, 0.6667), (0.6667, 0.8333), (0.8333, 1.0)]
    assert set(r.json()["lines"]) == {"0", "1"} and r.json()["rev"] == 2
    # closer than MIN_LINE_DUR after the previous line's start: that line is pushed and flagged
    r = server.put("/api/songs/demo/lines/1", json={"start": 0.6}).json()
    assert r["pushed"] == [0] and r["rev"] == 3
    l0 = load_song(server.root / "demo").lines[0]
    assert l0.start == pytest.approx(0.35) and l0.needs_review and l0.flags == ["pushed"]
    r = server.post("/api/songs/demo/undo").json()
    assert r["undone"] == "line 2 start" and r["rev"] == 4 and set(r["lines"]) == {"0", "1"}
    assert load_song(server.root / "demo").lines[0].start == 0.5
    # refused, nothing saved
    for body in ({}, {"shift": 1, "start": 2}, {"start": -0.5}, {"start": 4.0}):
        assert server.put("/api/songs/demo/lines/1", json=body).status_code == 422, body
    assert server.put("/api/songs/demo/lines/7", json={"shift": 1}).status_code == 404
    assert server.get("/api/songs/demo").json()["rev"] == 4


def durations(line: Line) -> list[float]:
    return [s.end - s.start for s in line.syllables()] + [u.end - u.start for s in line.syllables() for u in s.units]


def test_move_line_makes_room_and_never_collapses_a_neighbour() -> None:
    song = Song(meta=SongMeta(id="x"), lines=[make_line(0, "가나", 1.0), make_line(1, "다라마바", 4.0, dur=2.0),
                                              make_line(2, "사아", 6.0)])
    move = move_line(song, 2, start=3.0)                  # "Set start" with the player parked early (was refused)
    assert move.shift == -3.0 and move.pushed == [1] and move.touched == [1, 2]
    assert song.lines[1].start == pytest.approx(2.75) and song.lines[1].needs_review
    assert [(s.start, s.end) for s in song.lines[1].syllables()] == [
        (2.75, 2.8125), (2.8125, 2.875), (2.875, 2.9375), (2.9375, 3.0)]        # squeezed, keeps its rhythm
    assert all(d > 0 for ln in song.lines for d in durations(ln))
    move = move_line(song, 0, shift=10.0)                 # far later: the lines in the way move along
    assert move.shift == 10.0 and move.pushed == [1, 2]
    assert [ln.start for ln in song.lines] == pytest.approx([11.0, 11.25, 11.5])
    assert all(d > 0 for ln in song.lines for d in durations(ln))


def test_move_line_skips_untimed_lines_and_validates() -> None:
    song = Song(meta=SongMeta(id="x", duration=8.0), lines=[make_line(0, "가나", 1.0), make_line(1, "다라", 2.0),
                                                            make_line(2, "마", None), make_line(3, "바사", 4.0)])
    move = move_line(song, 1, start=3.5)                  # skips the untimed line 2
    assert move.shift == pytest.approx(1.5) and move.touched == [1] and song.lines[1].start == pytest.approx(3.5)
    assert all(s.end <= 4.0 for s in song.lines[1].syllables())
    move = move_line(song, 1, start=3.9)                  # too close to line 4: pushed, line 3 stays untimed
    assert move.pushed == [3] and song.lines[3].start == pytest.approx(4.15) and song.lines[2].start is None
    with pytest.raises(ValueError, match="no timing yet"):
        move_line(song, 2, shift=0.1)
    with pytest.raises(ValueError):
        move_line(song, 1, shift=float("nan"))
    with pytest.raises(IndexError):
        move_line(song, 9, shift=0.1)
    with pytest.raises(ValueError, match="before the beginning"):
        move_line(song, 0, start=-3.0)
    with pytest.raises(ValueError, match="before the beginning"):
        move_line(song, 0, shift=-3.0)                    # a nudge is not clamped any more either
    with pytest.raises(ValueError, match="the song ends at 0:08.00"):
        move_line(song, 3, start=8.0)


def test_move_line_handles_aligner_overlaps_and_noop() -> None:
    lines = [make_line(0, "가나", 1.0), make_line(1, "다라", 1.45, review=True), make_line(2, "마", 3.0)]
    song = Song(meta=SongMeta(id="x"), lines=lines)
    # line 1 starts before line 0's last onset (aligner output): a nudge earlier used to be blocked;
    # now line 0 is squeezed in front of it
    move = move_line(song, 1, shift=-0.1)
    assert move.shift == pytest.approx(-0.1) and move.touched == [0, 1] and move.pushed == []
    assert song.lines[0].end == pytest.approx(1.35) and song.lines[1].start == pytest.approx(1.35)
    assert not song.lines[1].needs_review and "manual_timing" in song.lines[1].flags
    assert all(d > 0 for ln in song.lines for d in durations(ln))
    move_line(song, 2, start=3.0)                         # a start set where it is confirms the line ...
    assert "manual_timing" in song.lines[2].flags
    assert move_line(song, 2, start=3.0).touched == []    # ... and then changes nothing


# ----------------------------------------------------------------------------- websocket


def test_websocket_init_and_20hz_broadcast(server: TestClient) -> None:
    server.post("/api/session/start", json={"song_id": "demo"})
    with server.websocket_connect("/ws") as ws, server.websocket_connect("/ws") as ws2:
        first = [ws.receive_json() for _ in range(3)]
        assert [m["type"] for m in first] == ["song", "display", "state"]
        assert first[0] == {"type": "song", "id": "demo", "rev": 0}
        assert first[1]["lead_time"] == 1.0 and first[1]["mirror"] is False
        b0 = server.hub.broadcasts
        t0 = time.monotonic()
        states = 0
        while time.monotonic() - t0 < 0.6:
            m = ws.receive_json()
            states += m["type"] == "state"
        rate = (server.hub.broadcasts - b0) / (time.monotonic() - t0)
        assert 12 <= rate <= 28, rate               # one broadcaster at ~20 Hz, shared by both clients
        assert states >= 8
        assert ws2.receive_json()["type"] == "song"
        # display changes and song edits are pushed to every client
        server.put("/api/display", json={"mirror": True, "lead_time": 2.5})
        server.put("/api/songs/demo/lines/0", json={"shift": 0.1})
        got = {}
        for _ in range(40):
            m = ws.receive_json()
            got.setdefault(m["type"], m)
            if "display" in got and "song" in got:
                break
        assert got["display"]["mirror"] is True and got["display"]["lead_time"] == 2.5
        assert got["song"] == {"type": "song", "id": "demo", "rev": 1}
    assert not server.hub.clients or len(server.hub.clients) <= 2


def test_websocket_exit_without_close_is_clean(server: TestClient) -> None:
    """Regression: leaving the context without ws.close() made the endpoint's teardown escape
    anyio's cancel scope (concurrent.futures.CancelledError on every exit)."""
    server.post("/api/session/start", json={"song_id": "demo"})
    for _ in range(5):
        with server.websocket_connect("/ws") as ws, server.websocket_connect("/ws") as ws2:
            assert ws.receive_json()["type"] == "song" and ws2.receive_json()["type"] == "song"
    deadline = time.monotonic() + 2
    while server.hub.clients and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not server.hub.clients


class FakeWS:
    def __init__(self, stall: bool = False):
        self.stall, self.sent, self.closed = stall, [], None
        self.gone = asyncio.Event()

    async def send_text(self, text: str) -> None:
        if self.stall:
            await asyncio.sleep(3600)
        self.sent.append(json.loads(text))

    async def receive(self) -> dict:
        await self.gone.wait()
        return {"type": "websocket.disconnect"}

    async def close(self, code: int = 1000) -> None:
        self.closed = code


def test_hub_drops_a_stalled_client_and_keeps_serving_the_others(song_root: Path) -> None:
    async def scenario() -> tuple[int, FakeWS, FakeWS, int]:
        hub = Hub(song_root, JobManager(), Factory(), lambda refresh: [], hz=20.0, send_timeout=0.2)
        hub.cued_song = "demo"
        good, stuck = FakeWS(), FakeWS(stall=True)
        broadcaster = asyncio.create_task(hub.broadcaster())
        serving = [asyncio.create_task(hub.serve(w)) for w in (good, stuck)]
        await asyncio.sleep(0.5)
        assert serving[1].done() and serving[1].exception() is None       # dropped after 0.2 s
        n0 = len(good.sent)
        await asyncio.sleep(1.0)
        n_states = sum(m["type"] == "state" for m in good.sent[n0:])
        good.gone.set()
        await asyncio.wait_for(serving[0], 1.0)
        broadcaster.cancel()
        with pytest.raises(asyncio.CancelledError):
            await broadcaster
        hub.jobs.shutdown()
        return n_states, good, stuck, len(hub.clients)

    n_states, good, stuck, n_clients = asyncio.run(scenario())
    assert stuck.closed == 1001 and good.closed is None and n_clients == 0
    assert [m["type"] for m in good.sent[:3]] == ["song", "display", "state"]
    assert 12 <= n_states <= 28, n_states                  # the healthy screen kept its ~20 Hz


def test_cross_origin_control_is_refused(server: TestClient) -> None:
    server.post("/api/session/start", json={"song_id": "demo"})
    sess = server.factory.sessions[0]
    evil = {"Origin": "https://evil.example"}
    for path in ("stop", "hold", "resume"):                # "simple" no-cors POSTs from another page
        r = server.post(f"/api/session/{path}", headers={**evil, "Content-Type": "text/plain"}, content=b"")
        assert r.status_code == 403, path
    assert server.post("/api/session/stop", headers={"Origin": "null"}).status_code == 403
    assert server.post("/api/session/stop", headers={"Referer": "http://evil.example/x"}).status_code == 403
    assert server.put("/api/display", json={"mirror": True}, headers=evil).status_code == 403
    assert server.delete("/api/jobs/x", headers=evil).status_code == 403
    r = server.post("/api/songs", headers=evil, data={"title": "x", "lyrics": "a"},
                    files={"audio": ("a.wav", b"x", "audio/wav")})
    assert r.status_code == 403 and not server.hub.jobs.list()
    assert server.hub.session is sess and sess.calls == [] and not sess.stopped
    # reads stay open (the browser keeps cross-origin responses from the page anyway)
    assert server.get("/api/songs", headers=evil).status_code == 200
    # the console itself (same origin) and non-browser clients (no Origin / Referer) pass
    same = {"Origin": "http://testserver"}
    assert server.post("/api/session/hold", json={"on": True}, headers=same).json()["status"] == "hold"
    assert server.post("/api/session/resume", headers={"Referer": "http://testserver/"}).json()["status"] == "tracking"
    assert server.post("/api/session/hold", json={"on": True}).status_code == 200
    with pytest.raises(WebSocketDisconnect) as exc, server.websocket_connect("/ws", headers=evil):
        pass
    assert exc.value.code == 1008
    with server.websocket_connect("/ws", headers=same) as ws:
        assert ws.receive_json()["type"] == "song"


def test_unknown_host_is_refused(server: TestClient) -> None:
    """DNS rebinding: a page on attacker.example that resolves to 127.0.0.1 sends Host: attacker.example."""
    for host in ("attacker.example", "attacker.example:8000", "localhost.attacker.example"):
        assert server.get("/api/songs", headers={"Host": host}).status_code == 421, host
        assert server.post("/api/session/stop", headers={"Host": host}).status_code == 421, host
    with pytest.raises(WebSocketDisconnect), server.websocket_connect("/ws", headers={"Host": "attacker.example"}):
        pass
    for host in ("127.0.0.1:8000", "localhost:8000", "[::1]:8000", "192.168.1.20:8000", "stage.localhost"):
        assert server.get("/api/songs", headers={"Host": host}).status_code == 200, host


def test_origin_guard_rules() -> None:
    g = OriginGuard(None, allowed_hosts=["Stage-Mac.local", " "])
    assert g.host_ok("stage-mac.local:8000") and g.host_ok("STAGE-MAC.LOCAL.") and g.host_ok("10.0.0.7")
    assert g.host_ok("[fe80::1]:80") and g.host_ok("") and not g.host_ok("evil.example")
    assert app_mod.machine_host_names() <= g.allowed
    same = OriginGuard.same_origin
    assert same("http://127.0.0.1:8000", "127.0.0.1:8000", "http")
    assert same("http://localhost", "localhost", "http") and same("http://LOCALHOST:80", "localhost", "ws")
    assert same("https://stage.local", "stage.local:443", "https")
    assert not same("http://127.0.0.1:8001", "127.0.0.1:8000", "http")
    assert not same("null", "127.0.0.1:8000", "http") and not same("http://[bad", "x", "http")


def test_display_settings_validation(server: TestClient) -> None:
    assert server.put("/api/display", json={"font_scale": 9}).status_code == 422
    d = server.put("/api/display", json={"font_scale": 1.4, "preview_lines": 1}).json()
    assert d["font_scale"] == 1.4 and d["preview_lines"] == 1 and d["rev"] == 1
    assert server.get("/api/display").json()["font_scale"] == 1.4


# ----------------------------------------------------------------------------- JS timeline parity


NODE_SCRIPT = """
import { prepareSong, locate } from %s;
const song = JSON.parse(process.argv[2]);
const ts = JSON.parse(process.argv[3]);
const lines = prepareSong(song);
console.log(JSON.stringify(ts.map((t) => locate(lines, t))));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_prompter_js_locate_matches_python_timeline(tmp_path: Path) -> None:
    timeline = pytest.importorskip("lyricprompter.timeline")
    lines = [make_line(0, "꽃밭에", 0.5), make_line(1, "앉아서", 1.5), make_line(2, "보니", None),
             make_line(3, "고운", 3.0, dur=1.2), make_line(4, "나비", 3.9)]
    lines[3].words[0].syllables[1].start = None          # missing syllable times are filled the same way
    lines[3].words[0].syllables[0].end = None
    lines[4].end = None
    song = Song(meta=SongMeta(id="p"), lines=lines)
    ts = [float(t) for t in np.round(np.arange(-0.5, 6.0, 0.05), 3)] + [0.5, 1.3, 3.0, 3.9]
    tl = timeline.Timeline(song.lines)
    want = [tl.locate(t) for t in ts]
    script = tmp_path / "parity.mjs"
    script.write_text(NODE_SCRIPT % json.dumps((STATIC / "common.js").as_uri()))
    out = subprocess.run(["node", str(script), json.dumps(song.to_dict()), json.dumps(ts)], capture_output=True,
                         text=True, check=True).stdout
    got = json.loads(out)
    for t, w, g in zip(ts, want, got):
        assert (g["line_idx"], g["syl_idx"], g["next_line_idx"], g["in_gap"]) == (
            w.line_idx, w.syl_idx, w.next_line_idx, w.in_gap), t
        assert g["syl_progress"] == pytest.approx(w.syl_progress, abs=1e-9), t


DISPLAY_SCRIPT = """
import { mergeDisplay, resetDisplay, restoreDisplayState, setLocalDisplay } from %s;
const D = { lead_time: 1.0, font_scale: 1.0, mirror: false, preview_lines: 2 };
const srv = (patch, rev) => ({ type: "display", ...D, ...patch, rev });
const out = {};
const reload = (st) => restoreDisplayState(D, JSON.parse(JSON.stringify(st)));
let st = restoreDisplayState(D, null);
mergeDisplay(st, srv({}, 0));
setLocalDisplay(st, { mirror: true });                        // M pressed on a teleprompter glass
out.lead_changed = mergeDisplay(st, srv({ lead_time: 2.0 }, 1));   // the console nudges the lead time
out.after_lead = { ...st.settings };
st = reload(st);                                              // page reload: init message again
mergeDisplay(st, srv({ lead_time: 2.0 }, 1));
out.after_reload = { ...st.settings };
mergeDisplay(st, srv({}, 0));                                 // server restart: defaults, rev 0
out.after_restart = { ...st.settings };
mergeDisplay(st, srv({ mirror: true }, 1));                   // the console sets mirror itself ...
mergeDisplay(st, srv({ mirror: false }, 2));                  // ... and clears it: it wins
out.console_mirror = st.settings.mirror;
let fresh = reload({ settings: { ...D, font_scale: 1.6 }, overrides: ["font_scale"], server: {} });
mergeDisplay(fresh, srv({ font_scale: 1.2 }, 3));             // first contact keeps this screen's value
out.first_contact = fresh.settings.font_scale;
out.reset_changed = resetDisplay(fresh);
out.after_reset = { font: fresh.settings.font_scale, overrides: fresh.overrides };
out.bad_saved = restoreDisplayState(D, { settings: { mirror: "yes", lead_time: 3 }, overrides: ["x", "mirror"] });
console.log(JSON.stringify(out));
"""


def run_node(tmp_path: Path, script: str, *args: str) -> dict:
    path = tmp_path / "check.mjs"
    path.write_text(script % json.dumps((STATIC / "common.js").as_uri()))
    out = subprocess.run(["node", str(path), *args], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_prompter_keeps_its_own_display_settings(tmp_path: Path) -> None:
    """A screen's own mirror survives console edits of other settings, reloads and server restarts."""
    out = run_node(tmp_path, DISPLAY_SCRIPT)
    assert out["lead_changed"] is True
    assert out["after_lead"] == {"lead_time": 2.0, "font_scale": 1.0, "mirror": True, "preview_lines": 2}
    assert out["after_reload"]["mirror"] is True and out["after_reload"]["lead_time"] == 2.0
    assert out["after_restart"]["mirror"] is True and out["after_restart"]["lead_time"] == 1.0
    assert out["console_mirror"] is False
    assert out["first_contact"] == 1.6
    assert out["reset_changed"] is True and out["after_reset"] == {"font": 1.2, "overrides": []}
    assert out["bad_saved"]["settings"] == {"lead_time": 3, "font_scale": 1.0, "mirror": False, "preview_lines": 2}
    assert out["bad_saved"]["overrides"] == ["mirror"]


STALE_SCRIPT = """
import { RefClock, shownStatus } from %s;
const out = {};
out.status = [shownStatus({ status: "tracking", age: 0.1 }), shownStatus({ status: "coasting", age: 1.5 }),
              shownStatus({ status: "hold", age: 9 }), shownStatus({ status: "manual" })];
const c = new RefClock();
c.update({ status: "tracking", ref_time: 10, tempo: 1, age: 0 }, 0);
out.t_live = c.tick(300);
c.update({ status: "tracking", ref_time: 10, tempo: 1, age: 1.2 }, 1200);   // no tracker step for 1.2 s
out.moving = c.moving;
out.t_stale = [c.tick(1216), c.tick(5000)];
c.update({ status: "tracking", ref_time: 11.5, tempo: 1, age: 0.02 }, 5016); // audio is back
out.t_back = c.tick(5032);
const h = new RefClock();                                                       // Hold after extrapolating
h.update({ status: "tracking", ref_time: 20, tempo: 1, age: 0 }, 0);
h.tick(0); h.tick(100); h.tick(200);
const before = h.tick(300);
h.update({ status: "hold", ref_time: 20.1, tempo: 0, age: 0 }, 310);
out.hold = [before, h.tick(320), h.tick(2000)];
h.update({ status: "hold", ref_time: 19.6, tempo: 0, age: 0 }, 2010);         // operator nudge -0.5 s
out.nudge = h.tick(2020);
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_stalled_input_shows_no_audio_and_freezes_the_clock(tmp_path: Path) -> None:
    out = run_node(tmp_path, STALE_SCRIPT)
    assert out["status"] == ["tracking", "stale", "hold", "manual"]
    assert out["t_live"] == pytest.approx(10.3)
    assert out["moving"] is False
    assert out["t_stale"] == [pytest.approx(10.35), pytest.approx(10.35)]      # held, no jump back
    assert out["t_back"] == pytest.approx(11.5 + 0.036)                         # 20 ms age + 16 ms
    before, *held = out["hold"]
    assert before > 20.2 and held == [before, before]          # Hold freezes, no step back to 20.1
    assert out["nudge"] == pytest.approx(19.6)                 # a real nudge back still shows


def test_ui_uses_the_shared_status_and_display_rules() -> None:
    prompter = (STATIC / "prompter.js").read_text(encoding="utf-8")
    console = (STATIC / "console.js").read_text(encoding="utf-8")
    for name in ("mergeDisplay(display, msg)", "setLocalDisplay(display,", "restoreDisplayState(DEFAULTS,",
                 "resetDisplay(display)", "setStatus(shownStatus(msg)"):
        assert name in prompter, name
    assert "shownStatus(msg)" in console
    assert 'id="b-reset"' in (STATIC / "prompter.html").read_text(encoding="utf-8")
    # the console sends only the edited field, not all four
    assert "displayPatch[el.name]" in console


# ----------------------------------------------------------------------------- review regressions


def strict_json(text: str):
    """json.loads that, like a browser's JSON.parse, refuses NaN / Infinity."""
    def refuse(const: str):
        raise ValueError(f"not valid JSON: {const}")
    return json.loads(text, parse_constant=refuse)


def raw_post(client: TestClient, path: str, body: str, method: str = "POST"):
    return client.request(method, path, content=body, headers={"content-type": "application/json"})


def test_messages_and_api_never_carry_nan_or_infinity(server: TestClient) -> None:
    """Regression: one inf input sample stuck the level meter at inf, and the broadcaster sent
    `"level_db": Infinity`, which every browser refuses: all screens froze silently."""
    enc = app_mod._encode([{"type": "state", "a": float("nan"), "b": [1.5, float("-inf")], "c": {"d": float("inf")}}])
    assert strict_json(enc[0][1]) == {"type": "state", "a": None, "b": [1.5, None], "c": {"d": None}}
    server.post("/api/session/start", json={"song_id": "demo"})
    sess = server.factory.sessions[0]
    snapshot = sess.snapshot
    sess.snapshot = lambda: {**snapshot(), "latency": float("nan"),
                             "source": {"level_db": float("inf"), "peak_db": -6.0}}
    with server.websocket_connect("/ws") as ws:
        msgs = [strict_json(ws.receive_text()) for _ in range(4)]
    state = next(m for m in msgs if m["type"] == "state")
    assert state["source"] == {"level_db": None, "peak_db": -6.0} and state["latency"] is None
    assert state["ref_time"] == 0.0
    # non-finite numbers never get in (Python's JSON parser accepts NaN / Infinity literals)
    assert raw_post(server, "/api/session/jump", '{"time": Infinity}').status_code == 422
    assert raw_post(server, "/api/session/nudge", '{"dt": NaN}').status_code == 422
    assert raw_post(server, "/api/session/latency", '{"seconds": NaN}').status_code == 422
    assert raw_post(server, "/api/songs/demo/lines/1", '{"shift": Infinity}', "PUT").status_code == 422
    assert raw_post(server, "/api/display", '{"font_scale": NaN}', "PUT").status_code == 422
    for body in ('{"song_id": "demo", "latency": NaN}', '{"song_id": "demo", "latency": 1e9}',
                 '{"song_id": "demo", "transpose": 40}',
                 '{"song_id": "demo", "input": {"kind": "file", "stem": "mix", "start_s": Infinity}}'):
        assert raw_post(server, "/api/session/start", body).status_code == 422, body
    server.post("/api/session/stop")
    server.post("/api/session/cue", json={"song_id": "demo"})
    assert raw_post(server, "/api/session/jump", '{"time": Infinity}').status_code == 422      # manual mode
    assert sess.calls == []


def test_go_on_a_line_of_another_song_never_moves_the_live_one(server: TestClient) -> None:
    """Regression: with song A live, Go on line N of song B (selected to look at it) jumped A."""
    shutil.copytree(server.root / "demo", server.root / "other")
    server.post("/api/session/start", json={"song_id": "demo"})
    sess = server.factory.sessions[0]
    r = server.post("/api/session/jump", json={"line": 2, "song_id": "other"})
    assert r.status_code == 409 and "not the song on the prompter" in r.json()["detail"]
    assert sess.calls == []
    assert server.post("/api/session/jump", json={"line": 2, "song_id": "demo"}).json()["ref_time"] == 2.5
    server.post("/api/session/stop")
    server.post("/api/session/cue", json={"song_id": "other"})
    assert server.post("/api/session/jump", json={"line": 1, "song_id": "demo"}).status_code == 409
    st = server.post("/api/session/jump", json={"line": 1, "song_id": "other"}).json()
    assert st["status"] == "manual" and st["song_id"] == "other" and st["ref_time"] == 1.5
    # a cued song whose package disappeared is a 409, not an unhandled error
    shutil.rmtree(server.root / "other")
    assert server.post("/api/session/jump", json={"line": 1}).status_code == 409


def test_nfd_package_folders_open_by_their_nfc_id(server: TestClient) -> None:
    """Regression: folders copied through Finder / unzipped from macOS archives are named in NFD;
    combining marks (dakuten, accents) failed the id check, so listed songs could not be opened."""
    names = ("ガラスの靴", "café", "꽃밭에서")
    for name in names:
        shutil.copytree(server.root / "demo", server.root / unicodedata.normalize("NFD", name))
    listed = {s["id"] for s in server.get("/api/songs").json()}
    assert set(names) <= listed
    for name in names:
        for form in ("NFC", "NFD"):
            r = server.get(f"/api/songs/{quote(unicodedata.normalize(form, name))}")
            assert r.status_code == 200 and r.json()["id"] == name, (name, form)
    nfd = unicodedata.normalize("NFD", "ガラスの靴")
    st = server.post("/api/session/start", json={"song_id": nfd, "input": {"kind": "file", "stem": "mix"}}).json()
    assert st["song_id"] == "ガラスの靴"
    assert server.post("/api/session/cue", json={"song_id": nfd}).status_code == 200      # the running song
    assert server.put(f"/api/songs/{quote(nfd)}/lines/0", json={"shift": 0.1}).json()["rev"] == 1
    assert server.get(f"/api/songs/{quote('ガラスの靴')}").json()["rev"] == 1
    assert server.factory.sessions[0].calls[-1][0] == "reload"                         # the edit reached it
    check = app_mod.check_song_id
    assert check(unicodedata.normalize("NFD", "ㇷ゚x")) == "ㇷ゚x"           # a mark with no precomposed form
    for bad in ("゙abc", ".hidden", "a/b", "../x", "", "a..b", "new\n"):
        with pytest.raises(app_mod.HTTPException):
            check(bad)


def test_song_json_replaced_on_disk_reaches_every_screen(server: TestClient) -> None:
    """Regression: a CLI re-prep rewrote song.json under the same rev, so prompters never re-fetched
    it, and the next unrelated line edit swapped the running session's lines mid-show."""
    server.post("/api/session/start", json={"song_id": "demo"})
    sess, hub = server.factory.sessions[0], server.hub
    deadline = time.monotonic() + 3
    while hub._sent_song != ("demo", 0) and time.monotonic() < deadline:   # the broadcaster announced it
        time.sleep(0.02)
    with server.websocket_connect("/ws") as ws:
        assert ws.receive_json() == {"type": "song", "id": "demo", "rev": 0}
        song = load_song(server.root / "demo")
        song.lines[0].text = "REPREPPED"
        song.lines.pop()
        save_song(song, server.root / "demo")
        got, deadline = None, time.monotonic() + 3
        while got is None and time.monotonic() < deadline:
            m = ws.receive_json()
            got = m if m["type"] == "song" else None
    assert got == {"type": "song", "id": "demo", "rev": 1}
    assert sess.reloaded is not None and sess.reloaded.lines[0].text == "REPREPPED" and len(sess.reloaded.lines) == 2
    assert server.get("/api/songs/demo").json()["rev"] == 1
    # the server's own edit counts once
    assert server.put("/api/songs/demo/lines/1", json={"shift": 0.1}).json()["rev"] == 2
    time.sleep(0.7)
    assert hub.song_rev["demo"] == 2


def test_song_json_replaced_right_after_a_cue_is_announced(server: TestClient) -> None:
    """Regression: the watcher took its baseline at its first tick after a cue (up to 0.5 s later),
    so a re-prep landing in between kept rev 0 and the screens never re-fetched it."""
    hub = server.hub
    with server.websocket_connect("/ws") as ws:
        assert ws.receive_json() == {"type": "song", "id": None, "rev": 0}
        assert server.post("/api/session/cue", json={"song_id": "demo"}).status_code == 200
        song = load_song(server.root / "demo")
        song.lines[0].text = "REPREPPED"
        save_song(song, server.root / "demo")                 # before the watcher's next tick
        revs, deadline = [], time.monotonic() + 3
        while 1 not in revs and time.monotonic() < deadline:
            m = ws.receive_json()
            revs += [m["rev"]] if m["type"] == "song" and m["id"] == "demo" else []
    assert revs[-1] == 1 and hub.song_rev["demo"] == 1
    assert server.get("/api/songs/demo").json()["lines"][0]["text"] == "REPREPPED"
    # cueing the song again (or starting it) keeps watching from the same baseline: no new rev
    server.post("/api/session/cue", json={"song_id": "demo"})
    server.post("/api/session/start", json={"song_id": "demo"})
    time.sleep(0.7)
    assert hub.song_rev["demo"] == 1


def test_stop_succeeds_when_the_lost_input_device_cannot_be_stopped(server: TestClient) -> None:
    """Regression: PortAudio fails to stop / close the stream of a device lost mid-session; Stop
    then answered HTTP 500 although the session was gone."""
    def lost() -> None:
        raise OSError("Internal PortAudio error [PaErrorCode -9986]")
    server.post("/api/session/start", json={"song_id": "demo", "input": {"kind": "device", "device": 2}})
    server.factory.sessions[0].stop = lost
    r = server.post("/api/session/stop")
    assert r.status_code == 200 and r.json()["status"] == "idle" and r.json()["song_id"] == "demo"
    assert server.hub.session is None
    server.post("/api/session/start", json={"song_id": "demo", "input": {"kind": "device", "device": 2}})
    server.factory.sessions[1].stop = lost                    # also when the server shuts down


def test_display_settings_survive_a_server_restart(song_root: Path) -> None:
    """Regression: a restarted server sent default display settings, and every prompter dropped the
    console's mirror / text size / lead time."""
    def client() -> TestClient:
        return TestClient(create_app(song_root, session_factory=Factory(), jobs=JobManager(),
                                     devices=lambda refresh: [], allowed_hosts=TEST_HOSTS))
    with client() as c:
        c.put("/api/display", json={"mirror": True, "font_scale": 1.6})
    with client() as c:
        d = c.get("/api/display").json()
        with c.websocket_connect("/ws") as ws:
            ws.receive_json()
            pushed = ws.receive_json()
        ids = {s["id"] for s in c.get("/api/songs").json()}
    assert (d["mirror"], d["font_scale"], d["lead_time"]) == (True, 1.6, 1.0)
    assert pushed["type"] == "display" and pushed["mirror"] is True and pushed["font_scale"] == 1.6
    assert ids == {"demo"}
    (song_root / app_mod.DISPLAY_FILE).write_text('{"font_scale": 99}')        # invalid: defaults
    with client() as c:
        assert c.get("/api/display").json()["font_scale"] == 1.0


def test_relative_songs_dir_works_for_web_uploads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: prep jobs run with cwd=PROJECT_ROOT, so with a relative --songs-dir every upload
    failed (paths relative to the server's cwd)."""
    script = tmp_path / "fake_prep.py"
    script.write_text(FAKE_PREP)
    show, project = tmp_path / "show", tmp_path / "project"
    show.mkdir()
    project.mkdir()
    monkeypatch.chdir(show)
    jobs = JobManager(command_prefix=[sys.executable, str(script)], cwd=project)
    app = create_app(Path("mysongs"), session_factory=Factory(), jobs=jobs, devices=lambda refresh: [],
                     allowed_hosts=TEST_HOSTS)
    with TestClient(app) as client:
        assert app.state.hub.songs_dir == show / "mysongs"
        job = wait_job(client, upload(client).json()["job"]["id"])
    assert job["status"] == "done", job
    assert (show / "mysongs" / "new-song" / "song.json").is_file() and not (project / "mysongs").exists()


def test_serve_banner_names_addresses_a_tablet_can_open(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    """Regression: `serve --host 0.0.0.0` printed only 127.0.0.1 URLs."""
    import ipaddress

    import uvicorn
    real = app_mod.lan_addresses()
    assert all(ipaddress.ip_address(ip).version == 4 and not ipaddress.ip_address(ip).is_loopback for ip in real)
    monkeypatch.setattr(app_mod, "lan_addresses", lambda: ["192.168.1.20"])
    monkeypatch.setattr(app_mod, "machine_host_names", lambda: {"stage-mac", "stage-mac.local"})
    out = app_mod.banner("0.0.0.0", 8000)
    assert "console   http://127.0.0.1:8000/" in out
    assert "http://192.168.1.20:8000/prompter" in out and "http://stage-mac.local:8000/prompter" in out
    assert "tablet" not in app_mod.banner("127.0.0.1", 8000)
    assert "http://[::1]:8000/prompter" in app_mod.banner("::1", 8000)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    assert app_mod.cli(["--host", "0.0.0.0", "--port", "8799", "--songs-dir", str(tmp_path / "songs")]) == 0
    out = capsys.readouterr().out
    assert "http://192.168.1.20:8799/prompter" in out and "no authentication" in out


def test_refine_options_say_what_they_do() -> None:
    html = (STATIC / "console.html").read_text(encoding="utf-8")
    assert "Refine every line" not in html                  # both modes refine the same uncertain lines
    assert "if the Qwen model is already downloaded" in html and "download the model if needed" in html


CONSOLE_HARNESS = r"""
// Stub DOM + fetch + WebSocket around the real static/console.js. Prints one JSON object.
const CONSOLE = process.argv[2], scenario = process.argv[3];
const ELS = new Map();
class El {
  constructor(sel) {
    this.sel = sel; this.dataset = {}; this.style = { setProperty() {} }; this.hidden = false; this.textContent = "";
    this._html = ""; this.value = ""; this.checked = false; this.disabled = false; this.title = ""; this.files = [];
    this.handlers = {}; this.attrs = {}; this.readyState = 0; this.src = ""; this.scrollTop = 0; this.open = false;
    this.classList = { toggle() {}, add() {}, remove() {}, contains() { return false; } };
    this.elements = new Proxy({}, { get: (t, k) => (typeof k === "string" ? (t[k] ||= new El(`${sel}.${k}`)) : undefined) });
  }
  set innerHTML(h) { this._html = h; this._go = null; }
  get innerHTML() { return this._html; }
  addEventListener(t, f) { (this.handlers[t] ||= []).push(f); }
  async fire(t, ev = {}) { for (const f of this.handlers[t] || []) await f({ preventDefault() {}, target: this, currentTarget: this, ...ev }); }
  setAttribute(k, v) { this.attrs[k] = v; } getAttribute(k) { return this.attrs[k]; } removeAttribute(k) { delete this.attrs[k]; }
  querySelector(s) { return el(`${this.sel} ${s}`); }
  querySelectorAll(s) {
    if (s !== "button[data-go]") return [];
    this._go ||= [...this._html.matchAll(/data-go="(\d+)" type="button" (disabled)?/g)].map((m) => {
      const b = new El(`go${m[1]}`); b.dataset.go = m[1]; b.disabled = !!m[2]; return b;
    });
    return this._go;
  }
  closest() { return null; } append() {} focus() {} reset() {} scrollIntoView() {} play() { return Promise.resolve(); }
}
const el = (sel) => { if (!ELS.has(sel)) ELS.set(sel, new El(sel)); return ELS.get(sel); };
const KEY_BTNS = [-1, 1].map((d) => { const b = new El(`key${d}`); b.dataset.transpose = String(d); return b; });
globalThis.document = {
  querySelector: el, querySelectorAll: (s) => (s === "[data-transpose]" ? KEY_BTNS : []),
  createElement: (t) => new El(t), addEventListener() {}, activeElement: null,
};
globalThis.requestAnimationFrame = () => 0;
globalThis.matchMedia = () => ({ matches: true });
globalThis.location = { protocol: "http:", host: "test:8000" };
let sock = null;
globalThis.WebSocket = class { constructor(url) { this.url = url; sock = this; } close() {} };

const SONGS = { A: [0.5, 1.5, 2.5], B: [3.0, 4.0, null] };
const song = (id) => ({ id, meta: { title: `Song ${id}` }, stems: ["mix"], rev: 0,
  lines: SONGS[id].map((t, i) => ({ idx: i, text: `${id} line ${i}`, start: t, end: t == null ? null : t + 0.8, words: [] })) });
const CALLS = [];
let failSongFetches = 0;
globalThis.fetch = async (path, opts = {}) => {
  const body = opts.body ? JSON.parse(opts.body) : undefined;
  CALLS.push({ path, method: opts.method || "GET", body });
  let data = {};
  if (path === "/api/songs") data = Object.keys(SONGS).map((id) => ({ id, title: `Song ${id}` }));
  else if (path.startsWith("/api/songs/")) {
    if (failSongFetches > 0) { failSongFetches--; throw new TypeError("Load failed"); }
    data = song(decodeURIComponent(path.split("/")[3]));
  } else if (path === "/api/jobs" || path.startsWith("/api/devices")) data = [];
  return { ok: true, status: 200, headers: { get: () => "application/json" }, json: async () => data };
};

await import(CONSOLE);
const tick = (ms = 5) => new Promise((r) => setTimeout(r, ms));
await tick();
sock.onopen();
const send = (m) => sock.onmessage({ data: JSON.stringify(m) });
const state = (m) => send({ type: "state", status: m.mode ? "tracking" : "idle", age: 0, ref_time: 0, ...m });
const idle = (id) => state({ song_id: id, mode: null });
const live = (id, transpose, extra = {}) => state({ song_id: id, mode: "mix", transpose, key_auto: true, key_suggest: null, ...extra });
const fs = el("#f-session");
Object.assign(fs.elements.kind, { value: "file" }); Object.assign(fs.elements.file_src, { value: "stem:mix" });
fs.elements.mode.value = "mix"; fs.elements.start_line.value = ""; fs.elements.latency.value = "0";
const select = async (id) => { await el("#songs").fire("click", { target: { closest: () => ({ dataset: { id } }) } }); await tick(); };
const click = async (sel) => { await el(sel).fire("click"); await tick(); };
const start = async () => { await fs.fire("submit"); await tick(); return CALLS.filter((c) => c.path === "/api/session/start").at(-1).body; };
const posts = (p) => CALLS.filter((c) => c.path === `/api/session/${p}`).map((c) => c.body);
const key = () => ({ shown: el("#o-transpose").value, auto: el("#b-autokey").attrs["aria-pressed"] });
const goClick = async (idx) => {
  const btn = { dataset: { go: String(idx) }, closest: () => ({ dataset: { idx: String(idx) } }) };
  await el("#t-lines tbody").fire("click", { target: { closest: (s) => (s === "button" ? btn : null) } });
  await tick();
};
const goDisabled = () => el("#t-lines tbody").querySelectorAll("button[data-go]").map((b) => b.disabled);
const out = {};
idle(null);

if (scenario === "key") {
  // (a) a key detected for song A never seeds song B
  await select("A");
  out.startA = await start();
  for (let i = 0; i < 3; i++) live("A", 2);
  await select("B");                                    // looking at the next song while A runs
  live("A", 2);
  out.whileALive = key();
  await click("#b-stop"); idle("A");
  out.afterStop = key();
  await click("#b-cue"); idle("B");
  out.startB = await start();
  // ... but restarting the same song keeps it as the seed
  for (let i = 0; i < 3; i++) live("B", -1);
  await click("#b-stop"); idle("B");
  out.restartB = await start();
  // (b) an operator key set before Start, then Auto back on: seeds the search
  live("B", 0); await click("#b-stop"); idle("B");
  await select("A"); idle("A");
  await KEY_BTNS[1].fire("click"); await KEY_BTNS[1].fire("click"); await tick();
  out.afterPlus = key();
  await click("#b-autokey");
  out.afterAuto = key();
  out.seeded = await start();
  // (c) a hand-set key (Auto off) belongs to its song too
  live("A", 2, { key_auto: false }); await click("#b-stop"); idle("A");
  await KEY_BTNS[1].fire("click"); await tick();
  out.manualA = key();
  await select("B"); idle("B");
  out.manualThenB = key();
  out.startManualB = await start();
} else if (scenario === "suggest") {
  await select("A"); await start();
  live("A", 0, { key_suggest: 3 });
  const sug = el("#b-keysuggest");
  out.running = { hidden: sug.hidden, label: sug.textContent };
  await click("#b-stop"); idle("A");
  out.afterStop = sug.hidden;
  await select("B"); idle("A");
  out.afterSelect = sug.hidden;
  await select("A"); await start();
  live("A", 0, { key_suggest: 3 });
  await click("#b-autokey");                         // Auto off: no suggestion to apply
  out.afterAutoOff = sug.hidden;
} else if (scenario === "go") {
  await select("A"); await start();
  live("A", 0);
  await select("B");                                 // the next song, while A is live
  live("A", 0);
  out.goWhileOtherLive = goDisabled();
  await goClick(1);
  out.jumpsWhileOtherLive = posts("jump");
  out.notice = el("#notice").textContent;
  await select("A"); live("A", 0);
  out.goOnLiveSong = goDisabled();
  await goClick(1);
  out.jumpsOnLiveSong = posts("jump");
  await click("#b-stop"); idle("A");
  await select("B"); idle("A");
  out.goWhenStopped = goDisabled();
  await goClick(0);                                  // no session: cue B, then jump in B
  out.manual = { cues: posts("cue"), jumps: posts("jump") };
} else if (scenario === "song-retry") {
  await select("A");
  const n = () => CALLS.filter((c) => c.path === "/api/songs/B").length;
  failSongFetches = 1;
  send({ type: "song", id: "B", rev: 0 });           // the live song, not the selected one
  await tick(20);
  out.afterFail = n();
  out.notice = el("#notice").textContent;
  await tick(1200);                                  // retried after 1 s
  out.afterRetry = n();
  await tick(1200);
  out.settled = n();
  sock.onopen(); send({ type: "song", id: "B", rev: 0 }); await tick(20);   // reconnect: fetched again
  out.afterReconnect = n();
} else if (scenario === "input-warn") {
  await select("A"); await start();
  const warn = () => el("#live-warn").textContent;
  live("A", 0, { source: { running: true, silent_s: 4, stalled_s: 0, last_error: null } });
  out.silence = warn();                              // zeros arrive: permission / channel hint
  live("A", 0, { source: { running: true, silent_s: 4, stalled_s: 4, last_error: "the input device stopped" } });
  out.stalled = warn();                              // nothing arrives: the device stopped
  live("A", 0, { warning: "Prepared again: start the session again." });
  out.reprepped = warn();
}
console.log(JSON.stringify(out));
process.exit(0);
"""

PROMPTER_HARNESS = r"""
// The real static/prompter.js with a stub DOM: song fetch failures are retried; a reconnect
// fetches the announced song again; a failed fetch of a newer rev keeps the shown lines up.
const PROMPTER = process.argv[2];
class Stub {
  constructor() { this.dataset = {}; this.style = { setProperty() {} }; this.hidden = false; this.textContent = "";
    this.value = ""; this.checked = false; this.classList = { toggle() {}, add() {}, remove() {} }; }
  addEventListener() {} removeAttribute() {} append() {} querySelector() { return new Stub(); } closest() { return null; }
}
const els = new Map();
globalThis.document = {
  getElementById: (id) => { if (!els.has(id)) els.set(id, new Stub()); return els.get(id); },
  querySelector: () => new Stub(), documentElement: new Stub(), body: new Stub(),
  addEventListener() {}, visibilityState: "hidden",
};
globalThis.location = { protocol: "http:", host: "stage:8000" };
globalThis.localStorage = { getItem: () => null, setItem() {} };
globalThis.matchMedia = () => ({ matches: true });
globalThis.requestAnimationFrame = () => 0;
let sock = null;
globalThis.WebSocket = class { constructor(url) { this.url = url; sock = this; } close() {} };
let fetches = 0, fail = 0, title = "Edelweiss";
globalThis.fetch = async () => {
  fetches++;
  if (fail > 0) { fail--; throw new TypeError("Load failed"); }
  const song = { meta: { title }, lines: [{ text: "Edelweiss", start: 6.5, end: 12.8, words: [] }] };
  return { ok: true, headers: { get: () => "application/json" }, json: async () => song };
};
await import(PROMPTER);
const tick = (ms) => new Promise((r) => setTimeout(r, ms));
sock.onopen();
const send = (m) => sock.onmessage({ data: JSON.stringify(m) });
const shown = () => ({ empty: !els.get("empty").hidden, text: els.get("empty").textContent, title: els.get("title").textContent, fetches });
const out = {};
fail = 1;
send({ type: "song", id: "edelweiss", rev: 0 });
await tick(20);
out.failed = shown();
for (let i = 0; i < 20; i++) { send({ type: "state", status: "tracking", ref_time: 7, tempo: 1, age: 0 }); await tick(60); }
out.retried = shown();                                  // retried after 1 s while the socket stayed healthy
fail = 1; title = "Edelweiss (edited)";
send({ type: "song", id: "edelweiss", rev: 1 });        // a newer rev fails to load: keep the lines up
await tick(20);
out.newRevFailed = shown();
await tick(1100);
out.newRevRetried = shown();
title = "Edelweiss (re-prepped)";
sock.onopen();                                          // reconnect (restarted server: rev 0 again)
send({ type: "song", id: "edelweiss", rev: 0 });
await tick(20);
out.reconnected = shown();
console.log(JSON.stringify(out));
process.exit(0);
"""


def run_harness(tmp_path: Path, script: str, module: str, *args: str) -> dict:
    path = tmp_path / "harness.mjs"
    path.write_text(script, encoding="utf-8")
    out = subprocess.run(["node", str(path), (STATIC / module).as_uri(), *args], capture_output=True, text=True,
                         check=True, timeout=60).stdout
    return json.loads(out.strip().splitlines()[-1])


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_console_key_belongs_to_its_song(tmp_path: Path) -> None:
    """Regression: a key detected for song A seeded song B's search, and an operator key set before
    Start (Auto back on) was dropped."""
    out = run_harness(tmp_path, CONSOLE_HARNESS, "console.js", "key")
    assert out["startA"]["transpose"] == 0 and out["startA"]["auto_key"] is True
    assert out["whileALive"] == {"shown": "+2", "auto": "true"}         # the panel shows the live song
    assert out["afterStop"] == {"shown": "0", "auto": "true"}           # ... then the selected one's key
    assert (out["startB"]["song_id"], out["startB"]["transpose"], out["startB"]["auto_key"]) == ("B", 0, True)
    assert out["restartB"]["transpose"] == -1                           # its own detected key seeds a restart
    assert out["afterPlus"] == {"shown": "+2", "auto": "false"} and out["afterAuto"] == {"shown": "+2", "auto": "true"}
    assert (out["seeded"]["song_id"], out["seeded"]["transpose"], out["seeded"]["auto_key"]) == ("A", 2, True)
    assert out["manualA"] == {"shown": "+3", "auto": "false"}
    assert out["manualThenB"] == {"shown": "0", "auto": "false"}         # a hand-set key stays with its song
    assert (out["startManualB"]["transpose"], out["startManualB"]["auto_key"]) == (0, False)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_console_key_suggestion_only_while_a_session_runs(tmp_path: Path) -> None:
    out = run_harness(tmp_path, CONSOLE_HARNESS, "console.js", "suggest")
    assert out == {"running": {"hidden": False, "label": "+3?"}, "afterStop": True, "afterSelect": True,
                   "afterAutoOff": True}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_console_go_never_jumps_another_live_song(tmp_path: Path) -> None:
    out = run_harness(tmp_path, CONSOLE_HARNESS, "console.js", "go")
    assert out["goWhileOtherLive"] == [True, True, True] and out["jumpsWhileOtherLive"] == []
    assert "Song A is live" in out["notice"]
    assert out["goOnLiveSong"] == [False, False, False] and out["jumpsOnLiveSong"] == [{"line": 1, "song_id": "A"}]
    assert out["goWhenStopped"] == [False, False, True]                  # B's line 2 has no timing
    assert out["manual"] == {"cues": [{"song_id": "B"}], "jumps": [{"line": 1, "song_id": "A"},
                                                                     {"line": 0, "song_id": "B"}]}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_console_retries_the_live_song_and_reloads_it_on_reconnect(tmp_path: Path) -> None:
    out = run_harness(tmp_path, CONSOLE_HARNESS, "console.js", "song-retry")
    assert (out["afterFail"], out["afterRetry"], out["settled"], out["afterReconnect"]) == (1, 2, 2, 3)
    assert "Retrying" in out["notice"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_console_tells_a_stopped_input_device_from_digital_silence(tmp_path: Path) -> None:
    """Regression: an input device that stopped delivering audio (unplugged) was reported as a
    microphone permission / channel problem."""
    out = run_harness(tmp_path, CONSOLE_HARNESS, "console.js", "input-warn")
    assert "microphone" in out["silence"]
    assert "Input error: the input device stopped" in out["stalled"] and "microphone" not in out["stalled"]
    assert "Prepared again: start the session again." in out["reprepped"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_prompter_retries_a_failed_song_fetch(tmp_path: Path) -> None:
    """Regression: one failed GET /api/songs/<id> left the stage screen on 'Could not load the song'
    for as long as the WebSocket stayed up (the song is announced only when it changes)."""
    out = run_harness(tmp_path, PROMPTER_HARNESS, "prompter.js")
    assert out["failed"]["empty"] and "Retrying" in out["failed"]["text"] and out["failed"]["fetches"] == 1
    assert out["retried"] == {"empty": False, "text": "", "title": "Edelweiss", "fetches": 2}
    assert out["newRevFailed"]["title"] == "Edelweiss" and not out["newRevFailed"]["empty"]   # lines stay up
    assert out["newRevRetried"]["title"] == "Edelweiss (edited)"
    assert out["reconnected"]["title"] == "Edelweiss (re-prepped)"     # re-fetched after a reconnect


# ----------------------------------------------------------------------------- slow: real session end to end

STEMS = Path(__file__).resolve().parents[1] / "data" / "stems" / "edelweiss_short"


@pytest.mark.slow
def test_real_session_over_http_and_websocket(tmp_path: Path) -> None:
    pytest.importorskip("lyricprompter.tracking.tracker")
    if not (STEMS / "mix.wav").is_file():
        pytest.skip("data/stems/edelweiss_short not available")
    root = tmp_path / "songs"
    lines = [make_line(i, t, s, dur=4.0) for i, (t, s) in enumerate([("에델", 7.0), ("바이스", 14.5), ("아침", 21.0)])]
    save_song(Song(meta=SongMeta(id="ew", title="Edelweiss"), lines=lines), root / "ew")
    (root / "ew" / "audio").mkdir()
    (root / "ew" / "audio" / "mix.wav").symlink_to(STEMS / "mix.wav")
    app = create_app(root, jobs=JobManager(), devices=lambda refresh: [], allowed_hosts=TEST_HOSTS)
    with TestClient(app) as client:
        r = client.post("/api/session/start", json={"song_id": "ew", "mode": "mix",
                                                    "input": {"kind": "file", "stem": "mix", "speed": 4.0}})
        assert r.status_code == 200, r.text
        with client.websocket_connect("/ws") as ws:
            states, t0 = [], time.monotonic()
            while time.monotonic() - t0 < 2.0:
                m = ws.receive_json()
                if m["type"] == "state":
                    states.append(m)
        last = states[-1]
        assert last["status"] == "tracking" and last["mode"] == "mix" and last["level_db"] > -60
        assert 4.0 < last["live_time"] < 9.5                       # ~2 s of wall time at 4x
        assert abs(last["ref_time"] - last["live_time"]) < 0.2      # self-simulation
        assert client.post("/api/session/hold", json={"on": True}).json()["status"] == "hold"
        assert client.post("/api/session/jump", json={"line": 2}).json()["ref_time"] == pytest.approx(21.0, abs=0.05)
        assert client.post("/api/session/resume").json()["status"] in ("tracking", "coasting")
        assert client.post("/api/session/stop").json()["status"] == "idle"
