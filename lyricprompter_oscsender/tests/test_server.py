"""FastAPI server: REST + WebSocket with a fake live session, a real OscSender sending to a UDP socket
bound in the test, and projects built from a short synthetic recording (ffmpeg converts it)."""
from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import socket
import sys
import threading
import time
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from lyricprompter_oscsender.cues import parse_cues, sheet_rev
from lyricprompter_oscsender.latency import LatencyReport, serve_echo
from lyricprompter_oscsender.osc import OscSender, Target, decode_packet
from lyricprompter_oscsender.project import create_project, load_meta, read_cues
from lyricprompter_oscsender.scheduler import FireRecord
from lyricprompter_oscsender.server import app as app_mod
from lyricprompter_oscsender.server.app import Hub, OriginGuard, SessionContext, create_app
from lyricprompter_oscsender.settings import Settings, load_settings, save_settings, settings_path
from lyricprompter_oscsender.synth import make_score, render_song

TEST_HOSTS = ("testserver",)          # Starlette's TestClient sends Host: testserver
DURATION = 4.0
CUES = ("# demo cues\n"                                           # 1
        "\n"                                                       # 2
        "0:00.500          /show/start\n"                          # 3
        "0:01.000  lights  /cue/go 12   # verse 1\n"               # 4
        "1.5               /mixer/fader f:0.75 i:3 true\n"         # 5
        "0:02.000  ghost   /x\n"                                   # 6
        "0:09.000          /late\n")                               # 7


# ----------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """projects/demo built once from a 4 s synthetic song; tests work on copies."""
    base = tmp_path_factory.mktemp("template")
    wav = base / "take.wav"
    sf.write(wav, render_song(make_score(DURATION, seed=3), sr=16000), 16000, subtype="PCM_16")
    root = base / "projects"
    create_project(root, title="Demo song", audio=wav, cues_text=CUES, project_id="demo", cues_name="cues.txt")
    return base


@pytest.fixture
def wav_bytes(template: Path) -> bytes:
    return (template / "take.wav").read_bytes()


@pytest.fixture
def udp() -> Iterator[socket.socket]:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.settimeout(2.0)
    yield s
    s.close()


def recv(sock: socket.socket) -> tuple[str, list]:
    data, _ = sock.recvfrom(65536)
    return decode_packet(data)[0]


def nothing_received(sock: socket.socket, wait: float = 0.2) -> bool:
    sock.settimeout(wait)
    try:
        sock.recvfrom(65536)
    except TimeoutError:
        return True
    finally:
        sock.settimeout(2.0)
    return False


class FakeSession:
    def __init__(self, pdir: Path, req: app_mod.SessionStart, ctx: SessionContext):
        self.pdir, self.req, self.ctx = Path(pdir), req, ctx
        self.cues = list(ctx.cues)
        self.settings = ctx.settings
        self.calls: list[tuple] = []
        self.status, self.ref_time = "waiting", ctx.start_time
        self.started = self.stopped = False
        self.seq = 0

    def start(self) -> None:
        self.started, self.status = True, "tracking"

    def stop(self) -> None:
        self.stopped, self.status = True, "stopped"

    def hold(self, on: bool) -> None:
        self.calls.append(("hold", on))
        self.status = "hold" if on else "tracking"

    def jump_time(self, t: float) -> None:
        self.calls.append(("jump_time", t))
        self.ref_time = t

    def jump_cue(self, cue_id: int) -> None:
        cue = next((c for c in self.cues if c.id == cue_id), None)
        if cue is None:
            raise KeyError(f"no cue {cue_id}")
        self.calls.append(("jump_cue", cue_id))
        self.ref_time = cue.time

    def nudge(self, dt: float) -> None:
        self.calls.append(("nudge", dt))
        self.ref_time += dt

    def set_transpose(self, k: int) -> None:
        self.calls.append(("transpose", k))

    def set_auto_key(self, on: bool) -> None:
        self.calls.append(("autokey", on))

    def set_style(self, style: str) -> None:
        self.calls.append(("style", style))

    def set_input_latency(self, ms: float) -> None:
        self.calls.append(("input_latency", ms))

    def set_cues(self, cues: list) -> None:
        self.cues = list(cues)
        self.calls.append(("set_cues", [c.id for c in cues]))

    def apply_settings(self, s: Settings) -> None:
        self.settings = s
        self.calls.append(("apply_settings", s.armed))

    def fire_cue(self, cue_id: int) -> FireRecord:
        cue = next((c for c in self.cues if c.id == cue_id), None)
        if cue is None:
            raise KeyError(f"no cue {cue_id}")
        target = cue.target or self.settings.default_target
        res = self.ctx.sender.send(target, cue.address, cue.args)
        self.seq += 1
        self.calls.append(("fire_cue", cue_id))
        return FireRecord(self.seq, cue.id, cue.time, target, cue.address, cue.message, "manual", res.ok, res.error,
                          time.time(), self.ref_time, 0.0, None, False)

    def snapshot(self) -> dict:
        return {"type": "state", "project_id": self.pdir.name, "mode": self.req.mode, "status": self.status,
                "ref_time": self.ref_time, "position": self.ref_time, "tempo": 1.0, "confidence": 0.8,
                "live_time": 1.0, "level_db": -20.0, "t": time.time(), "age": 0.0, "armed": self.settings.armed}


class Factory:
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []
        self.fail: Exception | None = None

    def __call__(self, pdir: Path, req: app_mod.SessionStart, ctx: SessionContext) -> FakeSession:
        if self.fail is not None:
            raise self.fail
        s = FakeSession(pdir, req, ctx)
        self.sessions.append(s)
        return s


DEVICES = [{"index": 2, "name": "Stage Interface", "channels": 4, "default_samplerate": 48000.0,
            "hostapi": "Core Audio", "is_default": True}]


def write_settings(root: Path, port: int) -> None:
    save_settings(settings_path(root), Settings(targets=[Target("main", "127.0.0.1", port),
                                                         Target("lights", "127.0.0.1", port, latency_ms=20.0)]))


@pytest.fixture
def project_root(template: Path, tmp_path: Path, udp: socket.socket) -> Path:
    root = tmp_path / "projects"
    shutil.copytree(template / "projects" / "demo", root / "demo")
    write_settings(root, udp.getsockname()[1])
    return root


@pytest.fixture
def server(project_root: Path, udp: socket.socket) -> Iterator[TestClient]:
    factory = Factory()
    device_calls: list[bool] = []

    def devices(refresh: bool) -> list[dict]:
        device_calls.append(refresh)
        return DEVICES

    app = create_app(project_root, session_factory=factory, devices=devices, allowed_hosts=TEST_HOSTS)
    with TestClient(app) as client:
        client.factory, client.hub, client.root = factory, app.state.hub, project_root
        client.device_calls, client.udp = device_calls, udp
        yield client


def strict_json(text: str) -> dict:
    def reject(token: str) -> None:
        raise ValueError(f"non-standard JSON token {token}")
    return json.loads(text, parse_constant=reject)


def raw(client: TestClient, path: str, body: str, method: str = "POST"):
    return client.request(method, path, content=body, headers={"content-type": "application/json"})


def wait_for(cond, timeout: float = 3.0, step: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(step)
    return bool(cond())


def receive_until(ws, pred, n: int = 200) -> dict:
    for _ in range(n):
        m = ws.receive_json()
        if pred(m):
            return m
    raise AssertionError("message not received")


def start_file_session(client: TestClient, **extra) -> FakeSession:
    body = {"project_id": "demo", "input": {"kind": "file", "stem": "mix"}, **extra}
    r = client.post("/api/session/start", json=body)
    assert r.status_code == 200, r.text
    return client.factory.sessions[-1]


# ----------------------------------------------------------------------------- pages


def test_console_page_and_static_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = app_mod.STATIC_DIR / "console.html"
    with TestClient(create_app(tmp_path / "p1", session_factory=Factory(), allowed_hosts=TEST_HOSTS)) as c:
        assert c.get("/").status_code == (200 if real.is_file() else 404)
    static = tmp_path / "static"
    static.mkdir()
    (static / "console.html").write_text("<!doctype html><title>oscsender</title>", encoding="utf-8")
    (static / "console.js").write_text("export const x = 1;\n", encoding="utf-8")
    monkeypatch.setattr(app_mod, "STATIC_DIR", static)
    with TestClient(create_app(tmp_path / "p2", session_factory=Factory(), allowed_hosts=TEST_HOSTS)) as c:
        r = c.get("/")
        assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "oscsender" in r.text
        assert r.headers["cache-control"] == "no-cache"
        r = c.get("/static/console.js")
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
        assert c.get("/static/console.js", headers={"If-None-Match": r.headers["etag"]}).status_code == 304
        assert c.get("/static/missing.js").status_code == 404
        (static / "console.html").unlink()
        r = c.get("/")
        assert r.status_code == 404 and "console" in r.json()["detail"]


# ----------------------------------------------------------------------------- projects


def test_project_list_and_detail(server: TestClient) -> None:
    (server.root / "broken").mkdir()
    (server.root / "broken" / "project.json").write_text("{not json")
    rows = {r["id"]: r for r in server.get("/api/projects").json()}
    demo = rows["demo"]
    assert demo["title"] == "Demo song" and demo["style"] == "classic" and demo["stems"] == ["mix"]
    assert demo["n_cues"] == 5 and demo["n_errors"] == 0 and demo["duration"] == pytest.approx(DURATION, abs=0.01)
    assert demo["created"].endswith("+00:00")
    assert "error" in rows["broken"]
    d = server.get("/api/projects/demo").json()
    assert d["project"]["id"] == "demo" and d["project"]["stems"] == ["mix"] and d["project"]["schema"] == 1
    assert d["logs"] == [] and d["rev"] == 0
    cues = d["cues"]
    assert cues["rev"] == sheet_rev(CUES) and cues["text"] == CUES and [c["id"] for c in cues["cues"]] == [3, 4, 5, 6, 7]
    c5 = cues["cues"][2]
    assert c5["args"] == [["f", 0.75], ["i", 3], ["T", True]] and c5["message"] == "/mixer/fader f:0.75 i:3 true"
    assert cues["cues"][1]["target"] == "lights" and cues["cues"][1]["note"] == "verse 1"
    issues = {(i["line"], i["severity"]): i["message"] for i in cues["issues"]}
    assert "unknown target 'ghost'" in issues[(6, "warning")]
    assert "after the end" in issues[(7, "warning")]
    assert set(issues) == {(6, "warning"), (7, "warning")}
    assert cues["undo_left"] == 0 and cues["project_rev"] == 0
    assert server.hub.focus == "demo"
    assert server.get("/api/projects/nope").status_code == 404
    assert server.get("/api/projects/.hidden").status_code == 400
    assert server.get("/api/projects/a..b").status_code == 400
    # show logs are listed newest first and can be downloaded
    logs = server.root / "demo" / "logs"
    (logs / "show-20260101-100000.csv").write_text("wall_iso\n", encoding="utf-8")
    (logs / "show-20260102-100000.csv").write_text("wall_iso\nx\n", encoding="utf-8")
    d = server.get("/api/projects/demo").json()
    assert d["logs"] == ["show-20260102-100000.csv", "show-20260101-100000.csv"]
    r = server.get("/api/projects/demo/logs/show-20260102-100000.csv")
    assert r.status_code == 200 and r.text == "wall_iso\nx\n"
    assert server.get("/api/projects/demo/logs/..%2Fproject.json").status_code == 404
    assert server.get("/api/projects/demo/logs/nope.csv").status_code == 404


def test_audio_waveform_and_cue_download(server: TestClient) -> None:
    full = server.get("/api/projects/demo/audio/mix")
    assert full.status_code == 200 and full.headers["content-type"] == "audio/wav"
    assert full.headers.get("accept-ranges") == "bytes"
    part = server.get("/api/projects/demo/audio/mix", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206 and part.content == full.content[100:200]
    assert part.headers["content-range"] == f"bytes 100-199/{len(full.content)}"
    assert server.get("/api/projects/demo/audio/vocals").status_code == 404
    assert server.get("/api/projects/demo/audio/passwd").status_code == 404
    w = server.get("/api/projects/demo/waveform?points=100").json()
    assert w["points"] == 100 and len(w["min"]) == len(w["max"]) == 100 and w["duration"] == pytest.approx(DURATION, abs=0.01)
    assert max(w["max"]) > 0.01 and min(w["min"]) < -0.01
    assert server.get("/api/projects/demo/waveform?points=50").status_code == 422
    assert server.get("/api/projects/demo/waveform?points=20001").status_code == 422
    assert server.get("/api/projects/demo/waveform?stem=vocals").status_code == 404
    r = server.get("/api/projects/demo/cues.txt")
    assert r.status_code == 200 and r.content == CUES.encode() and r.headers["content-type"].startswith("text/plain")
    assert "attachment" in r.headers["content-disposition"] and "demo-cues.txt" in r.headers["content-disposition"]


def test_create_project_by_upload(server: TestClient, wav_bytes: bytes, udp: socket.socket) -> None:
    port = udp.getsockname()[1]
    text = f"﻿@target fx 127.0.0.1:{port} latency=15\n0:00.250 fx /fx/on 1\n1.0 /two\n"
    r = server.post("/api/projects", data={"title": "  New Show ", "style": "pop"},
                    files={"audio": ("Take 1 (final).wav", wav_bytes, "audio/wav"),
                           "cues": ("show cues.txt", text.encode(), "text/plain"),
                           "vocals": ("vox.wav", wav_bytes, "audio/wav")})
    assert r.status_code == 201, r.text
    body = r.json()
    p = body["project"]
    assert p["id"] == "new-show" and p["title"] == "New Show" and p["style"] == "pop" and p["stems"] == ["mix", "vocals"]
    assert p["n_cues"] == 2 and p["n_errors"] == 0 and body["warnings"] == [] and body["targets_added"] == ["fx"]
    meta = load_meta(server.root / "new-show")
    assert meta.sources == {"audio": "Take 1 (final).wav", "vocals": "vox.wav", "cues": "show cues.txt"}
    assert read_cues(server.root / "new-show") == text.removeprefix("﻿")
    fx = load_settings(settings_path(server.root)).target("fx")
    assert fx is not None and (fx.port, fx.latency_ms) == (port, 15.0)
    assert server.hub.sender.stats()["fx"]["resolved"] == f"127.0.0.1:{port}"
    assert server.hub.rev["new-show"] == 1
    # pasted cue text when no file is given; the same title gets a fresh id
    r = server.post("/api/projects", data={"title": "New Show", "cues_text": "0:01.000 /a\n"},
                    files={"audio": ("a.wav", wav_bytes, "audio/wav")})
    assert r.status_code == 201 and r.json()["project"]["id"] == "new-show-2" and r.json()["project"]["n_cues"] == 1
    assert r.json()["project"]["style"] == "classic" and r.json()["targets_added"] == []
    ids = {row["id"] for row in server.get("/api/projects").json()}
    assert {"demo", "new-show", "new-show-2"} <= ids and not any(i.startswith(".") for i in ids)
    # validation: nothing is created, no upload folder is left behind
    bad = [({"title": " "}, {"audio": ("a.wav", wav_bytes, "audio/wav")}),
           ({"title": "x", "style": "jazz"}, {"audio": ("a.wav", wav_bytes, "audio/wav")}),
           ({"title": "x"}, {"audio": ("a.wav", b"", "audio/wav")}),
           ({"title": "x"}, {"audio": ("a.wav", b"this is not audio" * 64, "audio/wav")}),
           ({"title": "x"}, {"audio": ("a.wav", wav_bytes, "audio/wav"), "cues": ("c.txt", b"\xff\xfe1 /a", "text/plain")})]
    for data, files in bad:
        r = server.post("/api/projects", data=data, files=files)
        assert r.status_code == 400, (data, r.text)
    assert server.post("/api/projects", data={"title": "x"}).status_code == 422          # no audio
    assert not list(server.root.glob(".upload-*")) and not (server.root / "x").exists()


def test_delete_style_and_vocals(server: TestClient, wav_bytes: bytes) -> None:
    sess = start_file_session(server)
    assert server.delete("/api/projects/demo").status_code == 409
    r = server.put("/api/projects/demo/style", json={"style": "pop"})
    assert r.json() == {"id": "demo", "style": "pop", "rev": 1, "session": True}
    assert load_meta(server.root / "demo").style == "pop" and sess.calls[-1] == ("style", "pop")
    assert server.get("/api/session").json()["session"]["style"] == "pop"
    assert server.put("/api/projects/demo/style", json={"style": "rock"}).status_code == 422
    # vocal mode needs the vocal stem
    r = server.post("/api/session/start", json={"project_id": "demo", "mode": "vocal"})
    assert r.status_code == 400 and "vocal stem" in r.json()["detail"]
    r = server.post("/api/projects/demo/vocals", files={"vocals": ("vox.flac.wav", wav_bytes, "audio/wav")})
    assert r.status_code == 200, r.text
    assert r.json()["project"]["stems"] == ["mix", "vocals"] and r.json()["warnings"] == []
    assert load_meta(server.root / "demo").sources["vocals"] == "vox.flac.wav"
    assert server.post("/api/projects/demo/vocals", files={"vocals": ("v.wav", b"junk" * 50, "audio/wav")}).status_code == 400
    assert server.post("/api/session/start", json={"project_id": "demo", "mode": "vocal",
                                                   "input": {"kind": "file", "stem": "vocals"}}).status_code == 200
    server.post("/api/session/stop")
    r = server.delete("/api/projects/demo")
    assert r.status_code == 200 and r.json()["trashed"].startswith("demo-")
    assert (server.root / ".trash" / r.json()["trashed"] / "project.json").is_file()
    assert server.get("/api/projects").json() == [] and server.get("/api/projects/demo").status_code == 404
    assert server.delete("/api/projects/demo").status_code == 404


# ----------------------------------------------------------------------------- cue edits


def test_cue_edits_rewrite_only_what_they_change(server: TestClient) -> None:
    rev = server.get("/api/projects/demo/cues").json()["rev"]
    r = server.patch("/api/projects/demo/cues/5", json={"time": 1.25, "rev": rev})
    assert r.status_code == 200, r.text
    body = r.json()
    lines = body["text"].splitlines()
    assert lines[4] == "1.250             /mixer/fader f:0.75 i:3 true" and lines[:4] == CUES.splitlines()[:4]
    assert body["project_rev"] == 1 and body["undo_left"] == 1 and body["targets_added"] == []
    assert body["rev"] == sheet_rev(body["text"]) and read_cues(server.root / "demo") == body["text"]
    body = server.patch("/api/projects/demo/cues/4", json={"time": "0:01.250"}).json()
    assert body["text"].splitlines()[3] == "0:01.250  lights  /cue/go 12   # verse 1"
    body = server.patch("/api/projects/demo/cues/3", json={"shift": 0.25}).json()
    assert body["text"].splitlines()[2] == "0:00.750          /show/start"
    body = server.patch("/api/projects/demo/cues/6", json={"message": "main /y 1 # fixed"}).json()
    assert body["text"].splitlines()[5] == "0:02.000  main /y 1 # fixed"
    assert not any("ghost" in i["message"] for i in body["issues"])
    body = server.post("/api/projects/demo/cues/add", json={"time": 3.0, "message": "/added 1", "rev": body["rev"]}).json()
    assert body["added"] == 7 and body["text"].splitlines()[6] == "0:03.000  /added 1"
    assert [c["id"] for c in body["cues"]] == [3, 4, 5, 6, 7, 8]
    r = server.delete(f"/api/projects/demo/cues/3?rev={body['rev']}")
    assert r.status_code == 200 and "/show/start" not in r.json()["text"] and r.json()["undo_left"] == 6
    assert r.json()["project_rev"] == 6 and server.hub.rev["demo"] == 6
    # errors: nothing is written
    before = read_cues(server.root / "demo")
    assert server.patch("/api/projects/demo/cues/5", json={"time": 1.0, "shift": 1.0}).status_code == 422
    assert server.patch("/api/projects/demo/cues/5", json={}).status_code == 422
    assert server.patch("/api/projects/demo/cues/1", json={"time": 1.0}).status_code == 404     # the comment line
    assert server.patch("/api/projects/demo/cues/99", json={"time": 1.0}).status_code == 404
    assert server.delete("/api/projects/demo/cues/99").status_code == 404
    for bad in ({"time": -1.0}, {"time": "1:75.0"}, {"shift": -10.0}, {"message": "no-slash"},
                {"message": "/a 99999999999999999999999"}):
        r = server.patch("/api/projects/demo/cues/4", json=bad)
        assert r.status_code == 400 and r.json()["detail"], bad
    assert server.post("/api/projects/demo/cues/add", json={"time": 1.0, "message": "bad address"}).status_code == 400
    assert server.patch("/api/projects/nope/cues/4", json={"time": 1.0}).status_code == 404
    assert read_cues(server.root / "demo") == before and server.hub.rev["demo"] == 6


def test_stale_rev_is_a_conflict_with_the_current_sheet(server: TestClient) -> None:
    old = server.get("/api/projects/demo/cues").json()["rev"]
    server.patch("/api/projects/demo/cues/5", json={"shift": 0.1})
    cur = read_cues(server.root / "demo")
    for r in (server.patch("/api/projects/demo/cues/4", json={"shift": 0.1, "rev": old}),
              server.post("/api/projects/demo/cues/add", json={"time": 1, "message": "/a", "rev": old}),
              server.delete(f"/api/projects/demo/cues/4?rev={old}"),
              server.put("/api/projects/demo/cues", json={"text": "", "rev": old}),
              server.post("/api/projects/demo/cues/4/fire", json={"source": "preview", "rev": old})):
        assert r.status_code == 409, r.text
        assert "changed" in r.json()["detail"] and r.json()["cues"]["text"] == cur
        assert r.json()["cues"]["rev"] == sheet_rev(cur)
    assert read_cues(server.root / "demo") == cur and server.hub.rev["demo"] == 1
    # an unchanged edit (same time) saves nothing and records no undo
    body = server.patch("/api/projects/demo/cues/4", json={"time": 1.0}).json()
    assert body["text"] == cur and body["undo_left"] == 1 and server.hub.rev["demo"] == 1


def test_replace_upload_and_undo(server: TestClient) -> None:
    crlf = "0:01.000 /one\r\n0:00.500 /zero\r\n2 bogus\r\n"
    body = server.put("/api/projects/demo/cues", json={"text": crlf}).json()
    assert body["text"] == crlf and [c["id"] for c in body["cues"]] == [2, 1]
    assert any(i["line"] == 3 and i["severity"] == "error" for i in body["issues"])       # saved with issues
    assert (server.root / "demo" / "cues.txt").read_bytes() == crlf.encode()
    body = server.patch("/api/projects/demo/cues/1", json={"time": 1.5}).json()
    assert body["text"] == "0:01.500 /one\r\n0:00.500 /zero\r\n2 bogus\r\n"
    r = server.post("/api/projects/demo/cues/upload",
                    files={"cues": ("c.txt", "﻿0:03.000 /up\n".encode(), "text/plain")})
    assert r.status_code == 200 and r.json()["text"] == "0:03.000 /up\n" and r.json()["undo_left"] == 3
    bad = server.post("/api/projects/demo/cues/upload", files={"cues": ("c.txt", b"\xff\xfe", "text/plain")})
    assert bad.status_code == 400 and "UTF-8" in bad.json()["detail"]
    # undo walks back through the edits
    texts = [server.post("/api/projects/demo/cues/undo").json() for _ in range(3)]
    assert [t["text"] for t in texts] == ["0:01.500 /one\r\n0:00.500 /zero\r\n2 bogus\r\n", crlf, CUES]
    assert [t["undo_left"] for t in texts] == [2, 1, 0] and texts[0]["undone"] == "cue file"
    assert read_cues(server.root / "demo") == CUES and server.hub.rev["demo"] == 6
    r = server.post("/api/projects/demo/cues/undo")
    assert r.status_code == 409 and r.json()["detail"] == "Nothing to undo"
    # a hand edit since the last edit: the history no longer applies
    server.patch("/api/projects/demo/cues/5", json={"shift": 0.5})
    server.patch("/api/projects/demo/cues/5", json={"shift": 0.5})
    (server.root / "demo" / "cues.txt").write_text(CUES + "0:03.500 /hand\n", encoding="utf-8")
    r = server.post("/api/projects/demo/cues/undo")
    assert r.status_code == 409 and "changed on disk" in r.json()["detail"]
    assert server.post("/api/projects/demo/cues/undo").json()["detail"] == "Nothing to undo"
    assert read_cues(server.root / "demo").endswith("/hand\n")
    # an edit after a hand edit starts a fresh history
    server.patch("/api/projects/demo/cues/3", json={"shift": 0.1})
    assert server.post("/api/projects/demo/cues/undo").json()["text"].endswith("/hand\n")


def test_undo_depth_is_limited(server: TestClient) -> None:
    for _ in range(app_mod.UNDO_DEPTH + 5):
        server.patch("/api/projects/demo/cues/3", json={"shift": 0.001})
    assert len(server.hub.undo["demo"]) == app_mod.UNDO_DEPTH


def test_edits_reach_the_live_session(server: TestClient) -> None:
    sess = start_file_session(server)
    assert [c.id for c in sess.cues] == [3, 4, 5, 6, 7]
    server.post("/api/projects/demo/cues/add", json={"time": 0.0, "message": "/first"})
    assert sess.calls[-1] == ("set_cues", [3, 4, 5, 6, 7, 8]) and sess.cues[0].address == "/first"
    server.delete("/api/projects/demo/cues/3")
    assert [c.address for c in sess.cues][:2] == ["/show/start", "/cue/go"]
    server.post("/api/projects/demo/cues/undo")
    assert sess.cues[0].address == "/first"
    # another project's edits never reach it
    shutil.copytree(server.root / "demo", server.root / "other")
    n = len(sess.calls)
    server.patch("/api/projects/other/cues/4", json={"shift": 0.1})
    assert len(sess.calls) == n


def test_declared_targets_are_added_to_the_settings(server: TestClient, udp: socket.socket) -> None:
    port = udp.getsockname()[1]
    sess = start_file_session(server)
    text = f"@target fx 127.0.0.1:{port} latency=15ms\n@target main 10.0.0.1:1\n0:01 fx /fx/go 1\n0:02 main /m\n"
    body = server.put("/api/projects/demo/cues", json={"text": text}).json()
    assert body["targets_added"] == ["fx"] and not any("unknown target" in i["message"] for i in body["issues"])
    # main is declared with another address than the settings target of that name: the settings win, said so
    conflicts = [i for i in body["issues"] if "the OSC targets setting wins" in i["message"]]
    assert [(i["line"], i["severity"]) for i in conflicts] == [(2, "warning")]
    assert "10.0.0.1:1" in conflicts[0]["message"] and f"127.0.0.1:{port}" in conflicts[0]["message"]
    s = server.get("/api/settings").json()
    assert [t["name"] for t in s["targets"]] == ["main", "lights", "fx"]
    assert s["targets"][0]["host"] == "127.0.0.1"               # existing targets are not changed
    assert s["targets"][2]["latency_ms"] == 15.0 and s["targets_status"]["fx"]["resolved"] == f"127.0.0.1:{port}"
    assert load_settings(settings_path(server.root)).target("fx") is not None
    assert ("apply_settings", True) in sess.calls and sess.settings.target("fx") is not None
    rev = s["rev"]
    body = server.patch("/api/projects/demo/cues/3", json={"shift": 0.5}).json()
    assert body["targets_added"] == [] and server.get("/api/settings").json()["rev"] == rev
    r = server.post("/api/projects/demo/cues/3/fire", json={"source": "preview"})
    assert r.json() == {"ok": True, "error": None, "target": "fx", "address": "/fx/go", "logged": False}
    assert recv(udp) == ("/fx/go", [("i", 1)])



def test_declared_targets_of_a_cue_file_written_elsewhere_are_added_when_the_project_is_opened(
        server: TestClient, udp: socket.socket) -> None:
    port = udp.getsockname()[1]
    cues = server.root / "demo" / "cues.txt"
    # cue files that never came through the server (oscsender add, copied in, an editor): a session started
    # without opening the project first gets the declared target ...
    cues.write_text(f"@target fx 127.0.0.1:{port}\n0:01 fx /fx/go 1\n", encoding="utf-8")
    start_file_session(server)
    assert server.factory.sessions[-1].settings.target("fx") is not None
    # ... and so does opening the project in the console
    cues.write_text(f"@target fx2 127.0.0.1:{port}\n0:01 fx2 /fx/go 2\n", encoding="utf-8")
    body = server.get("/api/projects/demo").json()
    assert not any("unknown target" in i["message"] for i in body["cues"]["issues"])
    s = server.get("/api/settings").json()
    assert {"fx", "fx2"} <= {t["name"] for t in s["targets"]}
    assert s["targets_status"]["fx2"]["resolved"] == f"127.0.0.1:{port}"

# ----------------------------------------------------------------------------- fire / test


def test_fire_sends_now_even_when_disarmed(server: TestClient, udp: socket.socket) -> None:
    assert server.post("/api/session/arm", json={"on": False}).json()["armed"] is False
    r = server.post("/api/projects/demo/cues/5/fire", json={"source": "preview"})
    assert r.json() == {"ok": True, "error": None, "target": "main", "address": "/mixer/fader", "logged": False}
    assert recv(udp) == ("/mixer/fader", [("f", 0.75), ("i", 3), ("T", True)])
    rev = server.get("/api/projects/demo/cues").json()["rev"]
    r = server.post("/api/projects/demo/cues/4/fire", json={"source": "manual", "rev": rev})
    assert r.json()["ok"] is True and r.json()["target"] == "lights"
    assert recv(udp) == ("/cue/go", [("i", 12)])
    assert server.post("/api/projects/demo/cues/3/fire").json()["ok"] is True          # no body: manual
    assert recv(udp) == ("/show/start", [])
    r = server.post("/api/projects/demo/cues/6/fire", json={"source": "preview"}).json()
    assert r["ok"] is False and "unknown target 'ghost'" in r["error"]
    assert server.post("/api/projects/demo/cues/1/fire").status_code == 404
    assert server.post("/api/projects/demo/cues/4/fire", json={"source": "other"}).status_code == 422
    # an edit is seen by the next fire (the parsed sheet is cached by file signature)
    server.patch("/api/projects/demo/cues/5", json={"message": "/mixer/fader f:0.5"})
    server.post("/api/projects/demo/cues/5/fire", json={"source": "preview"})
    assert recv(udp) == ("/mixer/fader", [("f", 0.5)])
    assert server.hub.sender.stats()["main"]["sent"] == 3


def test_manual_fire_goes_through_the_live_session(server: TestClient, udp: socket.socket) -> None:
    sess = start_file_session(server)
    r = server.post("/api/projects/demo/cues/4/fire", json={"source": "manual"}).json()
    assert r == {"ok": True, "error": None, "target": "lights", "address": "/cue/go", "logged": True, "seq": 1}
    assert sess.calls[-1] == ("fire_cue", 4) and recv(udp)[0] == "/cue/go"
    n = len(sess.calls)
    r = server.post("/api/projects/demo/cues/3/fire", json={"source": "preview"}).json()
    assert r["logged"] is False and len(sess.calls) == n and recv(udp)[0] == "/show/start"
    # a cue of another project is sent directly
    shutil.copytree(server.root / "demo", server.root / "other")
    assert server.post("/api/projects/other/cues/3/fire").json()["logged"] is False
    assert len(sess.calls) == n and recv(udp)[0] == "/show/start"


def test_target_test_messages(server: TestClient, udp: socket.socket) -> None:
    assert server.post("/api/targets/main/test").json() == {"ok": True, "error": None, "target": "main",
                                                            "address": "/oscsender/test"}
    assert recv(udp) == ("/oscsender/test", [])
    r = server.post("/api/targets/lights/test", json={"message": '/hello 1 "two words" f:0.5'})
    assert r.json()["ok"] is True and recv(udp) == ("/hello", [("i", 1), ("s", "two words"), ("f", 0.5)])
    targets = server.get("/api/settings").json()["targets"]
    targets[1]["test_message"] = "/lights/flash 3"
    assert server.put("/api/settings", json={"targets": targets}).status_code == 200
    server.post("/api/targets/lights/test", json={})
    assert recv(udp) == ("/lights/flash", [("i", 3)])
    assert server.post("/api/targets/nope/test").status_code == 404
    assert server.post("/api/targets/main/test", json={"message": "no-slash"}).status_code == 400
    targets[0]["enabled"] = False
    server.put("/api/settings", json={"targets": targets})
    r = server.post("/api/targets/main/test").json()
    assert r["ok"] is False and "disabled" in r["error"] and nothing_received(udp)


# ----------------------------------------------------------------------------- settings


def test_settings_validation_and_target_resolution(server: TestClient, udp: socket.socket) -> None:
    port = udp.getsockname()[1]
    s = server.get("/api/settings").json()
    assert s["default_target"] == "main" and s["armed"] is True and s["rev"] == 0
    assert s["targets_status"]["main"] == {"sent": 0, "errors": 0, "last_error": None, "resolve_error": None,
                                           "resolved": f"127.0.0.1:{port}", "enabled": True}
    sess = start_file_session(server)
    r = server.put("/api/settings", json={"skip_policy": "all", "catchup_s": 2.5, "unknown": 1})
    assert r.status_code == 200 and r.json()["skip_policy"] == "all" and r.json()["rev"] == 1
    saved = load_settings(settings_path(server.root))
    assert (saved.skip_policy, saved.catchup_s, len(saved.targets)) == ("all", 2.5, 2)
    assert sess.calls[-1] == ("apply_settings", True) and sess.settings.skip_policy == "all"
    # targets are re-resolved: a host name, a new target
    targets = [{"name": "main", "host": "localhost", "port": port},
               {"name": "far", "host": "no-such-host.invalid", "port": 9000}]
    r = server.put("/api/settings", json={"targets": targets})
    assert r.status_code == 200, r.text
    st = r.json()["targets_status"]
    assert st["main"]["resolved"] == f"127.0.0.1:{port}" and st["far"]["resolved"] is None
    assert "cannot resolve" in st["far"]["resolve_error"] and set(st) == {"main", "far"}
    server.post("/api/targets/main/test")
    assert recv(udp)[0] == "/oscsender/test"
    # invalid settings change nothing
    before = server.get("/api/settings").json()
    for bad in ({"catchup_s": 99}, {"default_target": "nope"}, {"skip_policy": "maybe"},
                {"targets": [{"name": "main", "host": "x", "port": 70000}]},
                {"targets": [{"name": "a b", "host": "x", "port": 1}]},
                {"targets": [{"name": "main", "host": "x", "port": 1}, {"name": "main", "host": "y", "port": 2}]},
                {"targets": []}, {"armed": "yes"}, {"input_latency_ms": 5000}, [1, 2]):
        r = server.put("/api/settings", json=bad)
        assert r.status_code == 400 and r.json()["detail"], bad
    assert raw(server, "/api/settings", '{"catchup_s": NaN}', "PUT").status_code == 400
    assert server.get("/api/settings").json() == before
    # input latency: setting and session control, saved either way
    n = len(sess.calls)
    server.put("/api/settings", json={"input_latency_ms": 40})
    assert ("input_latency", 40.0) in sess.calls[n:]
    st = server.post("/api/session/latency", json={"ms": 25.5}).json()
    assert st["status"] == "tracking" and sess.calls[-1] == ("input_latency", 25.5)
    assert load_settings(settings_path(server.root)).input_latency_ms == 25.5
    assert server.post("/api/session/latency", json={"ms": 3000}).status_code == 422
    server.post("/api/session/stop")
    assert server.post("/api/session/latency", json={"ms": 10}).json()["status"] == "idle"
    assert server.get("/api/settings").json()["input_latency_ms"] == 10.0


class FakeDNS:
    """An OscSender resolver: names in `slow` block until `release` (an unknown `.local` name takes 5 s)
    and then fail, names in `down` fail, any other name is 127.0.0.1."""

    def __init__(self) -> None:
        self.slow: set[str] = set()
        self.down: set[str] = set()
        self.asked, self.release = threading.Event(), threading.Event()
        self.calls: list[str] = []

    def __call__(self, host: str, port: int) -> tuple[int, tuple]:
        self.calls.append(host)
        if host in self.slow:
            self.asked.set()
            self.release.wait(10.0)
        if host in self.slow or host in self.down:
            raise OSError(f"nodename nor servname provided, or not known ({host})")
        return socket.AF_INET, ("127.0.0.1", int(port))


@contextmanager
def dns_server(root: Path, dns: FakeDNS, targets: list[Target]) -> Iterator[TestClient]:
    save_settings(settings_path(root), Settings(targets=targets))
    factory = Factory()
    app = create_app(root, session_factory=factory, sender=OscSender(resolver=dns), allowed_hosts=TEST_HOSTS)
    with TestClient(app) as client:
        client.factory, client.hub, client.root = factory, app.state.hub, root
        yield client


def test_host_lookups_never_hold_up_arming_settings_or_cue_edits(project_root: Path, udp: socket.socket) -> None:
    """A new target's host lookup runs outside the locks: DISARM, a latency tweak and a cue edit made
    while it runs take effect at once, and the hosts of unchanged targets are not looked up again."""
    port = udp.getsockname()[1]
    dns = FakeDNS()
    with dns_server(project_root, dns, [Target("main", "127.0.0.1", port)]) as c:
        sess = start_file_session(c)
        targets = [*c.get("/api/settings").json()["targets"], {"name": "desk", "host": "desk-mac.local", "port": 9000}]
        dns.slow.add("desk-mac.local")
        replies: dict = {}
        th = threading.Thread(target=lambda: replies.update(put=c.put("/api/settings", json={"targets": targets})))
        th.start()
        try:
            assert dns.asked.wait(3.0)                     # the lookup of desk-mac.local runs
            t0 = time.monotonic()
            r = c.post("/api/session/arm", json={"on": False})
            assert r.status_code == 200 and r.json()["armed"] is False and sess.calls[-1] == ("apply_settings", False)
            tweak = [dict(t) for t in targets]
            tweak[0]["latency_ms"] = 12.0
            assert c.put("/api/settings", json={"targets": tweak}).status_code == 200
            assert sess.settings.target("main").latency_ms == 12.0 and sess.settings.target("desk") is not None
            assert c.patch("/api/projects/demo/cues/3", json={"shift": 0.1}).status_code == 200
            assert time.monotonic() - t0 < 1.0 and "put" not in replies
        finally:
            dns.release.set()
            th.join(5.0)
        st = replies["put"].json()["targets_status"]
        assert st["desk"]["resolved"] is None and "desk-mac.local" in st["desk"]["resolve_error"]
        assert st["main"]["resolved"] == f"127.0.0.1:{port}" and dns.calls.count("127.0.0.1") == 1


def test_failed_lookups_are_retried_and_a_good_address_is_kept(project_root: Path, udp: socket.socket,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    port = udp.getsockname()[1]
    dns = FakeDNS()
    dns.down.add("lx.local")
    with dns_server(project_root, dns, [Target("main", "127.0.0.1", port), Target("lights", "lx.local", port)]) as c:
        st = c.get("/api/settings").json()["targets_status"]["lights"]
        assert st["resolved"] is None and "lx.local" in st["resolve_error"]
        # the lighting desk joins the network after the server started: Test looks it up again
        dns.down.clear()
        assert c.post("/api/targets/lights/test").json()["ok"] is True and recv(udp)[0] == "/oscsender/test"
        # editing another target does not look lx.local up again
        n = dns.calls.count("lx.local")
        targets = c.get("/api/settings").json()["targets"]
        targets[0]["latency_ms"] = 15.0
        assert c.put("/api/settings", json={"targets": targets}).json()["targets_status"]["lights"]["resolved"]
        assert dns.calls.count("lx.local") == n
        # a manual Fire looks an unresolved target up again too
        dns.down.add("lx2.local")
        targets[1]["host"] = "lx2.local"
        assert c.put("/api/settings", json={"targets": targets}).json()["targets_status"]["lights"]["resolved"] is None
        dns.down.clear()
        r = c.post("/api/projects/demo/cues/4/fire", json={"source": "manual"}).json()
        assert r["ok"] is True and recv(udp) == ("/cue/go", [("i", 12)])
    # in the background: an unresolved host is retried, a resolved name refreshed; a failed refresh keeps
    # the address that worked
    monkeypatch.setattr(app_mod, "RESOLVE_RETRY_S", 0.05)
    monkeypatch.setattr(app_mod, "RESOLVE_REFRESH_S", 0.05)
    dns.down.add("lx.local")
    with dns_server(project_root, dns, [Target("main", "127.0.0.1", port), Target("lights", "lx.local", port)]) as c:
        assert c.get("/api/settings").json()["targets_status"]["lights"]["resolved"] is None
        dns.down.clear()
        assert wait_for(lambda: c.get("/api/settings").json()["targets_status"]["lights"]["resolved"] is not None)
        dns.down.add("lx.local")
        n = dns.calls.count("lx.local")
        assert wait_for(lambda: dns.calls.count("lx.local") >= n + 2)
        r = c.post("/api/projects/demo/cues/4/fire", json={"source": "preview"}).json()
        assert r["ok"] is True and recv(udp) == ("/cue/go", [("i", 12)])
        assert c.get("/api/settings").json()["targets_status"]["lights"]["resolve_error"] is None


def test_arm_and_settings_survive_a_restart(project_root: Path) -> None:
    def client() -> TestClient:
        return TestClient(create_app(project_root, session_factory=Factory(), allowed_hosts=TEST_HOSTS))
    with client() as c:
        assert c.post("/api/session/arm", json={"on": False}).json()["armed"] is False
        assert c.get("/api/settings").json()["rev"] == 1
        assert c.post("/api/session/arm", json={"on": False}).json()["rev"] == 1       # unchanged: no new rev
    with client() as c:
        s = c.get("/api/settings").json()
        assert s["armed"] is False and [t["name"] for t in s["targets"]] == ["main", "lights"]
        assert c.post("/api/session/arm", json={}).json()["armed"] is True


def test_settings_repaired_at_startup_are_reported_until_saved(project_root: Path) -> None:
    path = settings_path(project_root)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["targets"][0]["port"] = "nine thousand"            # a hand edit gone wrong
    path.write_text(json.dumps(data), encoding="utf-8")
    with TestClient(create_app(project_root, session_factory=Factory(), allowed_hosts=TEST_HOSTS)) as c:
        warnings = c.get("/api/settings").json()["warnings"]
        assert warnings and any("main" in w for w in warnings)
        assert c.post("/api/session/arm", json={"on": False}).status_code == 200     # a save writes valid settings
        assert c.get("/api/settings").json()["warnings"] == []


# ----------------------------------------------------------------------------- latency


@contextmanager
def echo_responder() -> Iterator[int]:
    stop, ready = threading.Event(), threading.Event()
    box: dict = {}

    def on_ready(port: int) -> None:
        box["port"] = port
        ready.set()

    th = threading.Thread(target=lambda: serve_echo(0, "127.0.0.1", stop=stop, on_ready=on_ready, poll_s=0.02),
                          daemon=True)
    th.start()
    assert ready.wait(2.0)
    try:
        yield box["port"]
    finally:
        stop.set()
        th.join(2.0)


def test_latency_echo(server: TestClient) -> None:
    with echo_responder() as port:
        r = server.post("/api/latency/echo", json={"host": "127.0.0.1", "port": port, "count": 4, "interval_ms": 5})
        assert r.status_code == 200, r.text
        rep = r.json()
        assert rep["method"] == "echo" and rep["sent"] == rep["received"] == 4 and len(rep["samples_ms"]) == 4
        assert rep["estimate_ms"] is not None and rep["error"] is None and rep["target"] is None
        targets = [*server.get("/api/settings").json()["targets"], {"name": "echo", "host": "127.0.0.1", "port": port}]
        server.put("/api/settings", json={"targets": targets})
        rep = server.post("/api/latency/echo", json={"target": "echo", "count": 2, "interval_ms": 0}).json()
        assert rep["target"] == "echo" and rep["port"] == port and rep["received"] == 2
        rep = server.post("/api/latency/echo", json={"target": "echo", "count": 2,
                                                     "message": "/oscsender/ping 1 d:0.5"}).json()
        assert rep["received"] == 2
    rep = server.post("/api/latency/echo", json={"host": "127.0.0.1", "port": port, "count": 1, "timeout_ms": 50}).json()
    assert rep["received"] == 0 and rep["estimate_ms"] is None
    assert server.post("/api/latency/echo", json={"target": "nope"}).status_code == 404
    assert server.post("/api/latency/echo", json={"host": "127.0.0.1"}).status_code == 422
    assert server.post("/api/latency/echo", json={"target": "main", "count": 0}).status_code == 422
    assert server.post("/api/latency/echo", json={"target": "main", "count": 201}).status_code == 422
    assert server.post("/api/latency/echo", json={"target": "main", "message": "bad"}).status_code == 400
    assert raw(server, "/api/latency/echo", '{"target": "main", "timeout_ms": NaN}').status_code == 422


def test_latency_loopback(server: TestClient, udp: socket.socket, monkeypatch: pytest.MonkeyPatch) -> None:
    r = server.post("/api/latency/loopback", json={"target": "lights", "channels": [0]})
    assert r.status_code == 400 and "1-based" in r.json()["detail"]
    made: list[tuple] = []
    measured: list[dict] = []

    class FakeCapture:
        def __init__(self, device=None, channels=None, **kw):
            made.append((device, channels, kw))

    def fake_measure(send, capture, **kw) -> LatencyReport:
        t = send()                       # the real send through the hub's sender
        measured.append({"capture": capture, "t": t, **kw})
        return LatencyReport.from_samples("loopback", [12.0, 14.0], target=kw.get("target"), sent=2, received=2)

    monkeypatch.setattr(app_mod, "AudioCapture", FakeCapture)
    monkeypatch.setattr(app_mod, "measure_loopback", fake_measure)
    r = server.post("/api/latency/loopback", json={"target": "lights", "device": 2, "channels": [1, 2], "count": 2,
                                                   "interval_ms": 800, "threshold_db": 25})
    assert r.status_code == 200, r.text
    assert r.json()["estimate_ms"] == 13.0 and r.json()["target"] == "lights"
    assert made[0][:2] == (2, [1, 2]) and made[0][2]["clock"] is server.hub.sender.clock
    assert measured[0]["count"] == 2 and measured[0]["interval_s"] == 0.8 and measured[0]["threshold_db"] == 25
    assert recv(udp) == ("/oscsender/test", [])
    targets = server.get("/api/settings").json()["targets"]
    targets[1]["test_message"] = "/click 1"
    server.put("/api/settings", json={"targets": targets})
    server.post("/api/latency/loopback", json={"target": "lights"})
    assert recv(udp) == ("/click", [("i", 1)])
    server.post("/api/latency/loopback", json={"target": "lights", "message": "/other"})
    assert recv(udp) == ("/other", [])
    assert server.post("/api/latency/loopback", json={"target": "nope"}).status_code == 404
    assert server.post("/api/latency/loopback", json={"target": "lights", "message": "x"}).status_code == 400
    # the audio input belongs to a running device session
    server.post("/api/session/start", json={"project_id": "demo", "input": {"kind": "device", "device": 2}})
    r = server.post("/api/latency/loopback", json={"target": "lights"})
    assert r.status_code == 409 and "session" in r.json()["detail"]
    start_file_session(server)                  # a rehearsal from a file does not use the input
    assert server.post("/api/latency/loopback", json={"target": "lights"}).status_code == 200
    assert len(measured) == 4


# ----------------------------------------------------------------------------- devices / session


def test_devices(server: TestClient) -> None:
    assert server.get("/api/devices").json() == DEVICES
    assert server.get("/api/devices?refresh=true").status_code == 200
    assert server.device_calls == [False, True]
    start_file_session(server)
    assert server.get("/api/devices?refresh=true").status_code == 409


def test_devices_error_is_503(project_root: Path) -> None:
    def broken(refresh: bool) -> list[dict]:
        raise OSError("PortAudio not initialized")
    with TestClient(create_app(project_root, session_factory=Factory(), devices=broken,
                               allowed_hosts=TEST_HOSTS)) as c:
        r = c.get("/api/devices")
    assert r.status_code == 503 and "PortAudio" in r.json()["detail"]


def test_session_lifecycle_and_controls(server: TestClient) -> None:
    assert server.post("/api/session/hold", json={"on": True}).status_code == 409
    st = server.get("/api/session").json()
    assert st["state"]["status"] == "idle" and st["state"]["project_id"] is None and st["session"] is None
    body = {"project_id": "demo", "mode": "mix", "style": "pop", "start_cue": 4, "transpose": 2,
            "input": {"kind": "device", "device": 2, "channels": [1, 2]}}
    st = server.post("/api/session/start", json=body).json()
    assert st["status"] == "tracking" and st["project_id"] == "demo" and st["ref_time"] == 1.0
    sess = server.factory.sessions[0]
    assert sess.started and sess.req.input.channels == [1, 2] and sess.req.style == "pop"
    assert sess.ctx.start_time == 1.0 and [c.id for c in sess.ctx.cues] == [3, 4, 5, 6, 7]
    assert sess.ctx.sender is server.hub.sender and sess.ctx.settings == server.hub.settings
    assert sess.ctx.settings is not server.hub.settings
    got = server.get("/api/session").json()
    assert got["session"]["start_cue"] == 4 and got["project_id"] == "demo"
    assert server.post("/api/session/hold", json={"on": True}).json()["status"] == "hold"
    assert server.post("/api/session/hold", json={"on": False}).json()["status"] == "tracking"
    assert server.post("/api/session/jump", json={"cue_id": 5}).json()["ref_time"] == 1.5
    assert server.post("/api/session/jump", json={"time": 2.25}).json()["ref_time"] == 2.25
    assert server.post("/api/session/jump", json={"time": -3}).json()["ref_time"] == 0.0
    assert server.post("/api/session/jump", json={}).status_code == 422
    assert server.post("/api/session/jump", json={"time": 1, "cue_id": 3}).status_code == 422
    assert server.post("/api/session/jump", json={"cue_id": 99}).status_code == 404
    r = server.post("/api/session/jump", json={"cue_id": 5, "project_id": "other"})
    assert r.status_code == 409 and "not the live project" in r.json()["detail"]
    assert server.post("/api/session/jump", json={"cue_id": 5, "project_id": "demo"}).status_code == 200
    server.post("/api/session/nudge", json={"dt": -0.5})
    server.post("/api/session/transpose", json={"semitones": -3})
    assert server.post("/api/session/transpose", json={"semitones": 40}).status_code == 422
    assert server.post("/api/session/autokey", json={"on": True}).status_code == 200
    assert server.post("/api/session/style", json={"style": "classic"}).status_code == 200
    assert server.get("/api/session").json()["session"]["style"] == "classic"
    for call in (("nudge", -0.5), ("transpose", -3), ("autokey", True), ("style", "classic")):
        assert call in sess.calls, call
    # starting again replaces the running session (make before break)
    st = server.post("/api/session/start", json={"project_id": "demo", "start_time": 2.0,
                                                 "input": {"kind": "file", "stem": "mix"}}).json()
    assert sess.stopped and len(server.factory.sessions) == 2 and st["ref_time"] == 2.0
    st = server.post("/api/session/stop").json()
    assert st["status"] == "idle" and st["project_id"] is None and server.factory.sessions[1].stopped
    assert server.post("/api/session/nudge", json={"dt": 0.5}).status_code == 409


def test_session_start_errors(server: TestClient) -> None:
    assert server.post("/api/session/start", json={"project_id": "nope"}).status_code == 404
    assert server.post("/api/session/start", json={"project_id": "demo", "start_cue": 1}).status_code == 404
    assert server.post("/api/session/start", json={"project_id": "demo", "start_cue": 3,
                                                   "start_time": 1.0}).status_code == 422
    assert server.post("/api/session/start", json={"project_id": "demo", "mode": "voice"}).status_code == 422
    server.factory.fail = ValueError("channels [9] out of range")
    r = server.post("/api/session/start", json={"project_id": "demo"})
    assert r.status_code == 400 and "out of range" in r.json()["detail"]
    server.factory.fail = FileNotFoundError("vocals.wav")
    assert server.post("/api/session/start", json={"project_id": "demo"}).status_code == 404
    server.factory.fail = ImportError("no module named lyricprompter_oscsender.live.session")
    assert server.post("/api/session/start", json={"project_id": "demo"}).status_code == 503
    assert server.hub.session is None
    # a session that fails to start leaves the running one alone
    server.factory.fail = None
    sess = start_file_session(server)
    server.factory.fail = None

    def boom() -> None:
        raise RuntimeError("cannot open the device")
    orig = Factory.__call__

    def failing(self, pdir, req, ctx):
        s = orig(self, pdir, req, ctx)
        s.start = boom
        return s
    server.factory.__class__ = type("FailingFactory", (Factory,), {"__call__": failing})
    r = server.post("/api/session/start", json={"project_id": "demo", "input": {"kind": "file", "stem": "mix"}})
    assert r.status_code == 500 and "cannot open the device" in r.json()["detail"]
    assert server.hub.session is sess and not sess.stopped and server.factory.sessions[-1].stopped


def test_device_retry_starts_a_new_session_with_its_own_show_log(project_root: Path) -> None:
    """Restarting on the same audio interface: the new input cannot open while the old session holds it;
    the server stops the old one and retries with a newly built session (the one that failed has closed
    its show log), so what fires is logged."""
    from lyricprompter_oscsender.live.session import OscSession

    class Interface:                     # stands in for DeviceInput; one stream at a time
        busy = False

        def __init__(self) -> None:
            self.open = False

        def start(self, callback) -> None:
            if Interface.busy:
                raise OSError("Error opening InputStream: Device unavailable [PaErrorCode -9985]")
            Interface.busy = self.open = True

        def stop(self) -> None:
            if self.open:
                Interface.busy = self.open = False

    built: list[OscSession] = []

    def factory(pdir: Path, req: app_mod.SessionStart, ctx: SessionContext) -> OscSession:
        built.append(OscSession(pdir, req.mode, Interface(), cues=ctx.cues, sender=ctx.sender, settings=ctx.settings,
                                start_time=ctx.start_time, clock=ctx.clock, log_dir=Path(pdir) / "logs"))
        return built[-1]

    body = {"project_id": "demo", "start_time": 2.5, "input": {"kind": "device", "device": 2}}
    app = create_app(project_root, session_factory=factory, allowed_hosts=TEST_HOSTS)
    with TestClient(app) as c:
        assert c.post("/api/session/start", json=body).status_code == 200
        r = c.post("/api/session/start", json=body)
        assert r.status_code == 200, r.text
        assert len(built) == 3 and app.state.hub.session is built[2] and built[0].stopped and built[1].stopped
        assert c.post("/api/projects/demo/cues/3/fire", json={"source": "manual"}).json()["logged"] is True
        log_path = built[2].show_log.path
        c.post("/api/session/stop")
    rows = log_path.read_text(encoding="utf-8").splitlines()
    assert sum(",manual," in row for row in rows) == 1


def test_go_and_start_at_a_cue_picked_in_a_stale_sheet_are_refused(server: TestClient) -> None:
    """Cue ids are line numbers: after another console inserts a cue, a Go or a start at "cue 5" from a view
    of the old file would act on another line. With the sheet's rev they are refused (409 + the sheet)."""
    old = server.get("/api/projects/demo/cues").json()["rev"]
    sess = start_file_session(server, start_cue=5, rev=old)
    assert sess.ctx.start_time == 1.5
    added = server.post("/api/projects/demo/cues/add", json={"time": 0.75, "message": "/inserted"}).json()
    assert added["added"] == 4                     # the fader cue moves from line 5 to line 6
    cur = read_cues(server.root / "demo")
    n = len(sess.calls)
    r = server.post("/api/session/jump", json={"cue_id": 5, "project_id": "demo", "rev": old})
    assert r.status_code == 409 and r.json()["cues"]["text"] == cur and len(sess.calls) == n
    r = server.post("/api/session/start", json={"project_id": "demo", "start_cue": 5, "rev": old,
                                                "input": {"kind": "file", "stem": "mix"}})
    assert r.status_code == 409 and r.json()["cues"]["rev"] == sheet_rev(cur)
    assert server.hub.session is sess and len(server.factory.sessions) == 1
    r = server.post("/api/session/jump", json={"cue_id": 6, "project_id": "demo", "rev": sheet_rev(cur)})
    assert r.status_code == 200 and r.json()["ref_time"] == 1.5
    assert server.post("/api/session/jump", json={"cue_id": 3}).status_code == 200        # rev is optional


def test_device_rescan_and_session_start_never_overlap(project_root: Path) -> None:
    """A rescan re-initialises PortAudio (closing every open stream): refused while a start is under way;
    a start (and a loopback measurement) asked for during a rescan waits for it (is refused)."""
    events: list[str] = []

    class SlowFactory(Factory):
        def __call__(self, pdir: Path, req: app_mod.SessionStart, ctx: SessionContext) -> FakeSession:
            events.append("build")
            time.sleep(0.4)
            s = super().__call__(pdir, req, ctx)
            events.append("built")
            return s

    def devices(refresh: bool) -> list[dict]:
        if refresh:
            events.append("rescan")
            time.sleep(0.4)
            events.append("rescanned")
        return DEVICES

    body = {"project_id": "demo", "input": {"kind": "device", "device": 2}}
    app = create_app(project_root, session_factory=SlowFactory(), devices=devices, allowed_hosts=TEST_HOSTS)
    with TestClient(app) as c:
        th = threading.Thread(target=lambda: c.post("/api/session/start", json=body))
        th.start()
        assert wait_for(lambda: "build" in events)
        assert c.get("/api/devices?refresh=true").status_code == 409
        th.join(5.0)
        assert c.post("/api/session/stop").json()["status"] == "idle"
        events.clear()
        th = threading.Thread(target=lambda: c.get("/api/devices?refresh=true"))
        th.start()
        assert wait_for(lambda: "rescan" in events)
        assert c.post("/api/latency/loopback", json={"target": "main"}).status_code == 409
        assert c.post("/api/session/start", json=body).status_code == 200
        th.join(5.0)
    assert events == ["rescan", "rescanned", "build", "built"]


def test_a_style_saved_while_the_session_is_built_reaches_it(server: TestClient) -> None:
    building = threading.Event()
    orig = Factory.__call__

    def slow(self, pdir, req, ctx):
        building.set()
        time.sleep(0.4)
        return orig(self, pdir, req, ctx)
    server.factory.__class__ = type("SlowFactory", (Factory,), {"__call__": slow})
    th = threading.Thread(target=start_file_session, args=(server,), kwargs={"style": "classic"})
    th.start()
    assert building.wait(3.0)
    r = server.put("/api/projects/demo/style", json={"style": "pop"})
    assert r.status_code == 200 and r.json()["session"] is False
    th.join(5.0)
    sess = server.factory.sessions[-1]
    assert sess.calls[-1] == ("style", "pop") and server.get("/api/session").json()["session"]["style"] == "pop"


def test_stop_succeeds_when_the_session_cannot_be_stopped(server: TestClient) -> None:
    sess = start_file_session(server)

    def lost() -> None:
        raise OSError("Internal PortAudio error [PaErrorCode -9986]")
    sess.stop = lost
    r = server.post("/api/session/stop")
    assert r.status_code == 200 and r.json()["status"] == "idle" and server.hub.session is None


def test_default_session_factory_builds_sources(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built: dict = {}

    class FakeOsc:
        def __init__(self, pdir, mode, source, params=None, **kw):
            built.clear()
            built.update(pdir=pdir, mode=mode, source=source, **kw)
            self.calls: list[tuple] = []

        def align_source_start(self, preroll):
            self.calls.append(("align", preroll))

        def set_input_latency(self, ms):
            self.calls.append(("latency", ms))

        def set_transpose(self, k):
            self.calls.append(("transpose", k))

        def set_auto_key(self, on):
            self.calls.append(("autokey", on))

    fake = types.ModuleType("lyricprompter_oscsender.live.session")
    fake.OscSession = FakeOsc
    monkeypatch.setitem(sys.modules, "lyricprompter_oscsender.live.session", fake)
    pdir = project_root / "demo"
    sender = OscSender()
    cues = parse_cues(CUES).cues

    def build(mode: str = "mix", transpose: int = 0, auto_key: bool = True, start_cue=None, start_time=None,
              **inp) -> FakeOsc:
        req = app_mod.SessionStart(project_id="demo", mode=mode, transpose=transpose, auto_key=auto_key,
                                   start_cue=start_cue, start_time=start_time,
                                   input=app_mod.InputSpec(kind="file", **inp))
        ctx = SessionContext(project_root, cues, sender, Settings(input_latency_ms=12.0), start_time or 0.0)
        return app_mod.default_session_factory(pdir, req, ctx)

    s = build(stem="mix", speed=1.5)
    assert built["source"].path == pdir / "audio" / "mix.wav" and built["source"].speed == 1.5
    assert built["cues"] == cues and built["sender"] is sender and built["log_dir"] == pdir / "logs"
    assert built["settings"].input_latency_ms == 12.0 and built["start_time"] == 0.0 and built["style"] is None
    assert s.calls == [("latency", 12.0)]                      # no transpose: automatic key detection
    # rehearsal from a position on the project's own stem: 2 s of pre-roll
    s = build(stem="mix", start_time=3.0)
    assert ("align", app_mod.REHEARSAL_PREROLL) in s.calls and built["start_time"] == 3.0
    assert ("align", 2.0) not in build(stem="mix", start_time=3.0, start_s=1.0).calls
    assert ("align", 2.0) not in build(path=str(pdir / "audio" / "mix.wav"), start_time=3.0).calls
    assert ("align", 2.0) not in build(stem="mix").calls
    # key: with Auto on a given key only seeds the search; Auto off makes it the operator's key (even 0)
    assert build(stem="mix", transpose=-1).calls[1:] == [("transpose", -1), ("autokey", True)]
    assert build(stem="mix", transpose=-1, auto_key=False).calls[1:] == [("transpose", -1)]
    assert build(stem="mix", transpose=0, auto_key=False).calls[1:] == [("transpose", 0)]
    for bad in ({}, {"stem": "karaoke"}, {"stem": "mix", "mode": "vocal"}):
        with pytest.raises(ValueError):
            build(**bad)
    with pytest.raises(FileNotFoundError):
        build(stem="vocals")
    shutil.copy(pdir / "audio" / "mix.wav", pdir / "audio" / "vocals.wav")
    build(stem="vocals", mode="vocal")
    assert built["mode"] == "vocal" and built["source"].path == pdir / "audio" / "vocals.wav"
    sender.close()


# ----------------------------------------------------------------------------- websocket


def test_websocket_hello_state_and_changes(server: TestClient) -> None:
    with server.websocket_connect("/ws") as ws:
        assert ws.receive_json() == {"type": "hello", "project_id": None, "settings_rev": 0}
        st = ws.receive_json()
        assert st["type"] == "state" and st["status"] == "idle" and st["project_id"] is None and st["armed"] is True
        assert "main" in st["targets"]
        b0, t0, states = server.hub.broadcasts, time.monotonic(), 0
        while time.monotonic() - t0 < 0.6:
            states += ws.receive_json()["type"] == "state"
        rate = (server.hub.broadcasts - b0) / (time.monotonic() - t0)
        assert 12 <= rate <= 28, rate
        assert states >= 8
        server.patch("/api/projects/demo/cues/3", json={"shift": 0.1})
        m = receive_until(ws, lambda m: m["type"] == "project")
        assert m == {"type": "project", "id": "demo", "rev": 1, "sheet_rev": sheet_rev(read_cues(server.root / "demo"))}
        server.put("/api/settings", json={"catchup_s": 2.0})
        assert receive_until(ws, lambda m: m["type"] == "settings") == {"type": "settings", "rev": 1, "armed": True}
        start_file_session(server)
        st = receive_until(ws, lambda m: m["type"] == "state" and m["status"] != "idle")
        assert st["project_id"] == "demo" and st["status"] == "tracking" and st["mode"] == "mix"
    with server.websocket_connect("/ws") as ws:
        assert ws.receive_json() == {"type": "hello", "project_id": "demo", "settings_rev": 1}


def test_websocket_exit_without_close_is_clean(server: TestClient) -> None:
    start_file_session(server)
    for _ in range(5):
        with server.websocket_connect("/ws") as ws, server.websocket_connect("/ws") as ws2:
            assert ws.receive_json()["type"] == "hello" and ws2.receive_json()["type"] == "hello"
    assert wait_for(lambda: not server.hub.clients, 2.0)


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


def test_hub_drops_a_stalled_client_and_keeps_serving_the_others(project_root: Path) -> None:
    async def scenario() -> tuple[int, FakeWS, FakeWS, int]:
        hub = Hub(project_root, Factory(), lambda refresh: [], 20.0, OscSender(), send_timeout=0.2)
        good, stuck = FakeWS(), FakeWS(stall=True)
        broadcaster = asyncio.create_task(hub.broadcaster())
        serving = [asyncio.create_task(hub.serve(w)) for w in (good, stuck)]
        await asyncio.sleep(0.5)
        assert serving[1].done() and serving[1].exception() is None       # dropped after 0.2 s
        n0 = len(good.sent)
        hub.bump("demo", "x")
        hub.bump("other")
        await asyncio.sleep(1.0)
        n_states = sum(m["type"] == "state" for m in good.sent[n0:])
        good.gone.set()
        await asyncio.wait_for(serving[0], 1.0)
        broadcaster.cancel()
        with pytest.raises(asyncio.CancelledError):
            await broadcaster
        hub.sender.close()
        return n_states, good, stuck, len(hub.clients)

    n_states, good, stuck, n_clients = asyncio.run(scenario())
    assert stuck.closed == 1001 and good.closed is None and n_clients == 0
    assert [m["type"] for m in good.sent[:2]] == ["hello", "state"]
    assert 12 <= n_states <= 28, n_states                  # the healthy console kept its ~20 Hz
    projects = [m for m in good.sent if m["type"] == "project"]          # one slot per project
    assert sorted((m["id"], m["rev"]) for m in projects) == [("demo", 1), ("other", 1)]


def test_cue_file_edited_on_disk_reaches_consoles_and_the_session(server: TestClient, udp: socket.socket) -> None:
    port = udp.getsockname()[1]
    sess = start_file_session(server)
    path = server.root / "demo" / "cues.txt"
    with server.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"
        text = f"@target disk 127.0.0.1:{port}\n0:00.250 disk /from/disk\n" + CUES
        path.write_text(text, encoding="utf-8")
        m = receive_until(ws, lambda m: m["type"] == "project" and m.get("sheet_rev") == sheet_rev(text))
        assert m["id"] == "demo" and m["rev"] >= 1
        assert wait_for(lambda: sess.cues and sess.cues[0].address == "/from/disk")
        assert receive_until(ws, lambda m: m["type"] == "settings")["rev"] >= 1
    assert server.hub.settings.target("disk") is not None
    body = server.get("/api/projects/demo/cues").json()
    assert body["text"] == text and body["project_rev"] == server.hub.rev["demo"]
    # the server's own saves are not announced twice
    rev = server.hub.rev["demo"]
    server.patch("/api/projects/demo/cues/2", json={"shift": 0.25})
    time.sleep(1.2)
    assert server.hub.rev["demo"] == rev + 1


def test_fire_and_go_by_id_see_a_disk_edit_the_watcher_has_not_seen_yet(server: TestClient,
                                                                         udp: socket.socket) -> None:
    sess = start_file_session(server)
    path = server.root / "demo" / "cues.txt"
    path.write_text("0:00.100 /inserted\n" + CUES, encoding="utf-8")      # every id moves down one line
    r = server.post("/api/projects/demo/cues/8/fire", json={"source": "manual"})
    assert r.status_code == 200 and r.json()["address"] == "/late" and r.json()["logged"] is True
    assert recv(udp) == ("/late", [])
    assert [c[0] for c in sess.calls].index("set_cues") < [c[0] for c in sess.calls].index("fire_cue")
    path.write_text("0:00.100 /a\n0:00.200 /b\n" + CUES, encoding="utf-8")
    assert server.post("/api/session/jump", json={"cue_id": 9}).json()["ref_time"] == 9.0


def test_an_edit_that_keeps_the_file_signature_is_seen(server: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """FAT32 keeps mtime in 2 s steps (HFS+ 1 s): two same-size saves within one step look alike to the
    file signature. The watcher compares the text of a recently written file."""
    path = server.root / "demo" / "cues.txt"
    frozen = time.time_ns()
    monkeypatch.setattr(app_mod, "WATCH_RECENT_S", 60.0, raising=False)
    monkeypatch.setattr(app_mod, "file_signature",
                        lambda p: (frozen, Path(p).stat().st_size, 0, 0) if Path(p).is_file() else None)
    sess = start_file_session(server)
    for t in ("0:00.6", "0:00.7"):             # the first edit changes the size, the second one does not
        text = CUES.replace("0:00.500", t)
        path.write_text(text, encoding="utf-8")
        assert wait_for(lambda text=text: server.hub.sheet_revs.get("demo") == sheet_rev(text)), t
        assert wait_for(lambda t=t: abs(sess.cues[0].time - float(t[2:])) < 1e-9), t
    rev = server.hub.rev["demo"]
    time.sleep(1.2)
    assert server.hub.rev["demo"] == rev                 # read again while recent, but not announced again


def test_text_that_cannot_be_saved_as_utf8_is_refused(server: TestClient) -> None:
    for path, body, method in (("/api/projects/demo/cues", '{"text": "0:01 /a\\n# \\ud800\\n"}', "PUT"),
                               ("/api/projects/demo/cues/4", '{"message": "/x 1 # \\ud800"}', "PATCH"),
                               ("/api/projects/demo/cues/add", '{"time": 1, "message": "/x # \\udfff"}', "POST")):
        r = raw(server, path, body, method)
        assert r.status_code == 400 and "Unicode" in r.json()["detail"], (path, r.text)
    assert read_cues(server.root / "demo") == CUES and server.hub.rev.get("demo", 0) == 0
    before = server.get("/api/settings").json()
    r = raw(server, "/api/settings", '{"targets": [{"name": "main", "host": "h\\ud800", "port": 9000}]}', "PUT")
    assert r.status_code == 400 and r.json()["detail"] and server.get("/api/settings").json() == before


def test_the_focus_project_is_watched_too(server: TestClient) -> None:
    for name in ("other", "third"):
        shutil.copytree(server.root / "demo", server.root / name)
    assert server.get("/api/projects/other").status_code == 200
    (server.root / "other" / "cues.txt").write_text(CUES + "0:03.000 /new\n", encoding="utf-8")
    (server.root / "third" / "cues.txt").write_text(CUES + "0:03.000 /new\n", encoding="utf-8")
    assert wait_for(lambda: server.hub.rev.get("other") == 1)
    time.sleep(1.1)
    assert "third" not in server.hub.rev and server.hub.rev["other"] == 1
    # a change between opening a project and the watcher's next look is still announced
    server.get("/api/projects/third")
    (server.root / "third" / "cues.txt").write_text(CUES, encoding="utf-8")
    assert wait_for(lambda: server.hub.rev.get("third") == 1)


def test_messages_and_api_never_carry_nan_or_infinity(server: TestClient) -> None:
    enc = app_mod._encode([{"type": "state", "a": float("nan"), "b": [1.5, float("-inf")],
                            "c": {"d": np.float32("inf"), "e": np.int64(3)}}])
    assert strict_json(enc[0][1]) == {"type": "state", "a": None, "b": [1.5, None], "c": {"d": None, "e": 3}}
    assert app_mod.json_safe({"x": (np.float64(1.5), np.array([1.0, np.nan]))}) == {"x": [1.5, [1.0, None]]}
    sess = start_file_session(server)
    snapshot = sess.snapshot
    sess.snapshot = lambda: {**snapshot(), "tempo": float("nan"), "source": {"level_db": float("inf")}}
    with server.websocket_connect("/ws") as ws:
        msgs = [strict_json(ws.receive_text()) for _ in range(3)]
    state = next(m for m in msgs if m["type"] == "state")
    assert state["tempo"] is None and state["source"] == {"level_db": None}
    r = server.post("/api/session/hold", json={"on": True})
    assert r.status_code == 200 and strict_json(r.text)["tempo"] is None
    for path, body, method in (("/api/session/jump", '{"time": Infinity}', "POST"),
                               ("/api/session/nudge", '{"dt": NaN}', "POST"),
                               ("/api/session/latency", '{"ms": NaN}', "POST"),
                               ("/api/projects/demo/cues/4", '{"time": NaN}', "PATCH"),
                               ("/api/projects/demo/cues/4", '{"shift": Infinity}', "PATCH"),
                               ("/api/projects/demo/cues/add", '{"time": NaN, "message": "/a"}', "POST"),
                               ("/api/session/start", '{"project_id": "demo", "start_time": Infinity}', "POST")):
        assert raw(server, path, body, method).status_code == 422, (path, body)
    assert read_cues(server.root / "demo") == CUES and ("nudge", float("nan")) not in sess.calls


def test_shutdown_stops_the_session_and_closes_the_sender(project_root: Path, udp: socket.socket) -> None:
    sender, factory = OscSender(), Factory()
    with TestClient(create_app(project_root, session_factory=factory, sender=sender, allowed_hosts=TEST_HOSTS)) as c:
        assert set(sender.targets) == {"main", "lights"}         # set from the settings at startup
        c.post("/api/session/start", json={"project_id": "demo", "input": {"kind": "file", "stem": "mix"}})
        c.post("/api/projects/demo/cues/3/fire", json={"source": "preview"})
        assert recv(udp)[0] == "/show/start" and sender._socks
    assert factory.sessions[0].stopped and not sender._socks


def test_real_session_fires_cues_over_osc(project_root: Path, udp: socket.socket) -> None:
    """End to end: the default session factory, lyricprompter's tracker following the project's own mix
    (played at 2x), the scheduler firing through the server's sender, the show log."""
    app = create_app(project_root, devices=lambda refresh: [], allowed_hosts=TEST_HOSTS)
    with TestClient(app) as client:
        r = client.post("/api/session/start", json={"project_id": "demo", "input": {"kind": "file", "stem": "mix",
                                                                                     "speed": 2.0}})
        assert r.status_code == 200, r.text
        got: list[tuple[str, list]] = []
        deadline = time.monotonic() + 4.0
        while len(got) < 3 and time.monotonic() < deadline:
            udp.settimeout(max(0.01, deadline - time.monotonic()))
            with contextlib.suppress(TimeoutError):
                got.append(recv(udp))
        udp.settimeout(2.0)
        assert got == [("/show/start", []), ("/cue/go", [("i", 12)]),
                       ("/mixer/fader", [("f", 0.75), ("i", 3), ("T", True)])]
        assert wait_for(lambda: client.get("/api/session").json()["state"]["status"] == "stopped", 3.0)
        st = client.get("/api/session").json()["state"]
        assert st["project_id"] == "demo" and st["ended"] is True
        fired = {f["cue_id"]: f for f in st["fired"]}
        assert [fired[i]["sent"] for i in (3, 4, 5)] == [True, True, True] and 7 not in fired
        assert fired[6]["sent"] is False and "ghost" in fired[6]["error"]
        assert st["counts"]["sent"] == 3 and st["counts"]["errors"] == 1
        r = client.post("/api/projects/demo/cues/4/fire", json={"source": "manual"}).json()
        assert r["ok"] is True and r["logged"] is True and r["seq"] == st["fired_seq"] + 1
        assert recv(udp) == ("/cue/go", [("i", 12)])
        assert client.post("/api/session/stop").json()["status"] == "idle"
        logs = client.get("/api/projects/demo").json()["logs"]
    assert len(logs) == 1
    rows = (project_root / "demo" / "logs" / logs[0]).read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("wall_iso,") and len(rows) == 1 + 5 and ",manual," in rows[-1]


# ----------------------------------------------------------------------------- origin guard / cli


def test_cross_origin_control_is_refused(server: TestClient, udp: socket.socket) -> None:
    sess = start_file_session(server)
    evil = {"Origin": "https://evil.example"}
    for path in ("stop", "hold", "arm"):                # "simple" no-cors POSTs from another page
        r = server.post(f"/api/session/{path}", headers={**evil, "Content-Type": "text/plain"}, content=b"")
        assert r.status_code == 403, path
    assert server.post("/api/session/stop", headers={"Origin": "null"}).status_code == 403
    assert server.post("/api/session/stop", headers={"Referer": "http://evil.example/x"}).status_code == 403
    assert server.put("/api/settings", json={"armed": False}, headers=evil).status_code == 403
    assert server.post("/api/projects/demo/cues/3/fire", headers=evil).status_code == 403
    assert server.post("/api/targets/main/test", headers=evil).status_code == 403
    assert server.delete("/api/projects/demo", headers=evil).status_code == 403
    assert server.hub.session is sess and sess.calls == [] and not sess.stopped and nothing_received(udp)
    assert server.hub.settings.armed is True
    assert server.get("/api/projects", headers=evil).status_code == 200      # reads stay open
    same = {"Origin": "http://testserver"}
    assert server.post("/api/session/hold", json={"on": True}, headers=same).json()["status"] == "hold"
    assert server.post("/api/session/hold", json={"on": False}, headers={"Referer": "http://testserver/"}).status_code == 200
    with pytest.raises(WebSocketDisconnect) as exc, server.websocket_connect("/ws", headers=evil):
        pass
    assert exc.value.code == 1008
    with server.websocket_connect("/ws", headers=same) as ws:
        assert ws.receive_json()["type"] == "hello"


def test_unknown_host_is_refused(server: TestClient) -> None:
    """DNS rebinding: a page on attacker.example that resolves to 127.0.0.1 sends Host: attacker.example."""
    for host in ("attacker.example", "attacker.example:8800", "localhost.attacker.example"):
        assert server.get("/api/projects", headers={"Host": host}).status_code == 421, host
        assert server.post("/api/session/stop", headers={"Host": host}).status_code == 421, host
    with pytest.raises(WebSocketDisconnect), server.websocket_connect("/ws", headers={"Host": "attacker.example"}):
        pass
    for host in ("127.0.0.1:8800", "localhost:8800", "[::1]:8800", "192.168.1.20:8800", "stage.localhost"):
        assert server.get("/api/projects", headers={"Host": host}).status_code == 200, host


def test_origin_guard_rules() -> None:
    g = OriginGuard(None, allowed_hosts=["Stage-Mac.local", " "])
    assert g.host_ok("stage-mac.local:8800") and g.host_ok("STAGE-MAC.LOCAL.") and g.host_ok("10.0.0.7")
    assert g.host_ok("[fe80::1]:80") and g.host_ok("") and not g.host_ok("evil.example")
    assert app_mod.machine_host_names() <= g.allowed
    same = OriginGuard.same_origin
    assert same("http://127.0.0.1:8800", "127.0.0.1:8800", "http")
    assert same("http://localhost", "localhost", "http") and same("http://LOCALHOST:80", "localhost", "ws")
    assert not same("http://127.0.0.1:8801", "127.0.0.1:8800", "http")
    assert not same("null", "127.0.0.1:8800", "http") and not same("http://[bad", "x", "http")


def test_serve_banner_and_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                              capsys: pytest.CaptureFixture[str]) -> None:
    import uvicorn

    monkeypatch.setattr(app_mod, "lan_addresses", lambda: ["192.168.1.20"])
    monkeypatch.setattr(app_mod, "machine_host_names", lambda: {"stage-mac", "stage-mac.local"})
    out = app_mod.banner("0.0.0.0", 8800)
    assert "console   http://127.0.0.1:8800/" in out
    assert "http://192.168.1.20:8800/" in out and "http://stage-mac.local:8800/" in out
    assert "tablet" not in app_mod.banner("127.0.0.1", 8800)
    assert "http://[::1]:8800/" in app_mod.banner("::1", 8800)
    runs: list[dict] = []
    monkeypatch.setattr(app_mod.config, "PROJECTS_DIR", tmp_path / "default")
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: runs.append(kw))
    assert app_mod.cli(["--host", "0.0.0.0", "--port", "8799", "--projects-dir", str(tmp_path / "p"),
                        "--allow-host", "show.lan"]) == 0
    out = capsys.readouterr().out
    assert "http://192.168.1.20:8799/" in out and "no authentication" in out
    assert runs[-1]["host"] == "0.0.0.0" and runs[-1]["port"] == 8799 and (tmp_path / "p").is_dir()
    assert app_mod.cli([]) == 0 and runs[-1]["port"] == app_mod.DEFAULT_PORT == 8800
    assert runs[-1]["host"] == "127.0.0.1" and (tmp_path / "default").is_dir()
    assert "no authentication" not in capsys.readouterr().out
    assert app_mod.cli(["--port", "x"]) == 2
