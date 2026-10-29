"""End to end through the HTTP API with nothing faked: the settings are set and a project is created over
HTTP (a synthetic song, ffmpeg converts it), a session is started with the server's default factory (a
rehearsal FileInput playing the project's own mix in real time, lyricprompter's tracker, the cue runner's
timer thread) and the cues arrive as OSC at a UDP socket bound here, in order and once each, while the
WebSocket state lists them as fired. A second session rehearses from a cue (2 s pre-roll, the cues before
it chased by the skip policy). Each test runs in real time: about 15 and 6 s.
"""
from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from lyricprompter_oscsender.osc import decode_packet
from lyricprompter_oscsender.server.app import create_app
from lyricprompter_oscsender.synth import make_score, render_song

TEST_HOSTS = ("testserver",)          # Starlette's TestClient sends Host: testserver
SONG_S = 16.0
FX_LATENCY_S = 0.06
# (time, target, address, args as decoded) -- main is the default target, fx is declared in the file
CUES = [
    (0.8, None, "/show/cue", [("i", 1)]),
    (1.9, "fx", "/fx/hit", [("s", "one")]),
    (3.2, None, "/show/cue", [("i", 2)]),
    (4.5, None, "/show/cue", [("i", 3)]),
    (6.0, None, "/mixer/fader", [("f", 0.5), ("i", 3), ("T", True)]),
    (7.4, "fx", "/fx/hit", [("s", "two words")]),
    (8.8, None, "/show/cue", [("i", 4)]),
    (10.2, None, "/show/cue", [("i", 5)]),
    (11.6, "fx", "/fx/hit", [("s", "three")]),
    (13.0, None, "/show/cue", [("i", 6)]),
]
TIMING_SPREAD_S = 0.15                # arrival - cue time may vary this much across the cues of one run

pytestmark = pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="ffmpeg not on PATH")


def osc_arg(tag: str, value: object) -> str:
    return {"i": str(value), "s": f'"{value}"', "f": f"f:{value}", "T": "true"}[tag]


def cue_text(port: int) -> str:
    lines = ["# integration cues", f"@target fx 127.0.0.1:{port}  latency={FX_LATENCY_S * 1000:.0f}ms", ""]
    for t, target, address, args in CUES:
        m, s = divmod(t, 60)
        lines.append(f"{int(m)}:{s:06.3f}  {target or '':<4}  {address} {' '.join(osc_arg(*a) for a in args)}")
    return "\n".join(lines) + "\n"


def cue_ids(text: str) -> list[int]:
    """Line numbers (= cue ids) of the cue lines, in file order."""
    return [i for i, line in enumerate(text.splitlines(), 1) if line[:1].isdigit()]


@dataclass
class Collector:
    """Receives OSC datagrams on its own thread, with the perf_counter time of arrival."""
    sock: socket.socket
    got: list[tuple[float, str, list]] = field(default_factory=list)
    stop: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self.sock.settimeout(0.05)
        self.thread = threading.Thread(target=self._run, name="udp-collector", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                data, _ = self.sock.recvfrom(65536)
            except TimeoutError:
                continue
            except OSError:
                return
            t = time.perf_counter()
            for address, args in decode_packet(data):
                self.got.append((t, address, [tuple(a) for a in args]))

    def wait_for(self, n: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while len(self.got) < n and time.monotonic() < deadline:
            time.sleep(0.02)
        return len(self.got) >= n

    def close(self) -> None:
        self.stop.set()
        self.thread.join(1.0)


@dataclass
class Env:
    client: TestClient
    udp: Collector
    pid: str
    ids: list[int]                    # cue ids in CUES order
    root: Path


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Env]:
    base = tmp_path_factory.mktemp("integration")
    wav = base / "song.wav"
    sf.write(wav, render_song(make_score(SONG_S, seed=11), sr=16000), 16000, subtype="PCM_16")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    udp = Collector(sock)
    root = base / "projects"
    app = create_app(root, devices=lambda refresh: [], allowed_hosts=TEST_HOSTS)
    try:
        with TestClient(app) as client:
            r = client.put("/api/settings", json={"targets": [{"name": "main", "host": "127.0.0.1", "port": port}],
                                                  "default_target": "main", "armed": True, "skip_policy": "latest"})
            assert r.status_code == 200, r.text
            text = cue_text(port)
            r = client.post("/api/projects", data={"title": "Integration song", "cues_text": text, "style": "classic"},
                            files={"audio": ("song.wav", wav.read_bytes(), "audio/wav")})
            assert r.status_code == 201, r.text
            body = r.json()
            assert body["targets_added"] == ["fx"]
            proj = body["project"]
            assert proj["n_cues"] == len(CUES) and proj["n_errors"] == 0
            assert proj["duration"] == pytest.approx(SONG_S, abs=0.05)
            yield Env(client, udp, proj["id"], cue_ids(text), root)
            client.post("/api/session/stop")
    finally:
        udp.close()
        sock.close()


def expected(i: int) -> tuple[str, list]:
    return CUES[i][2], [tuple(a) for a in CUES[i][3]]


def latency(i: int) -> float:
    return FX_LATENCY_S if CUES[i][1] == "fx" else 0.0


def test_settings_and_project_made_over_http(env: Env) -> None:
    s = env.client.get("/api/settings").json()
    fx = next(t for t in s["targets"] if t["name"] == "fx")
    assert fx["port"] == env.udp.sock.getsockname()[1] and fx["latency_ms"] == pytest.approx(FX_LATENCY_S * 1000)
    assert s["targets_status"]["fx"]["resolved"] is not None
    sheet = env.client.get(f"/api/projects/{env.pid}/cues").json()
    assert [c["id"] for c in sheet["cues"]] == env.ids and sheet["issues"] == []
    assert [c["target"] for c in sheet["cues"]] == [c[1] for c in CUES]


def test_real_session_fires_every_cue_once_in_order(env: Env) -> None:
    client, udp = env.client, env.udp
    udp.got.clear()
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        t0 = time.perf_counter()
        r = client.post("/api/session/start", json={"project_id": env.pid, "mode": "mix",
                                                    "input": {"kind": "file", "stem": "mix", "monitor": False}})
        assert r.status_code == 200, r.text
        st = r.json()
        assert st["type"] == "state" and st["project_id"] == env.pid and st["mode"] == "mix"
        assert st["style"] == "classic" and st["armed"] is True and st["status"] in ("waiting", "tracking")
        assert [u["id"] for u in st["upcoming"]] == env.ids[:3]

        # the WebSocket state shows every cue as fired (and sent), newest records included
        want = set(env.ids)
        last = None
        deadline = time.monotonic() + CUES[-1][0] + 8.0
        while time.monotonic() < deadline:
            m = ws.receive_json()
            if m["type"] != "state" or m["status"] == "idle":     # the idle state queued at connect
                continue
            last = m
            if want <= {f["cue_id"] for f in m["fired"]}:
                break
        assert last is not None and want <= {f["cue_id"] for f in last["fired"]}, last
    assert udp.wait_for(len(CUES), 2.0)
    time.sleep(0.3)                                    # nothing more arrives (each cue once)

    st = client.get("/api/session").json()["state"]
    assert client.post("/api/session/stop").json()["status"] == "idle"
    got = list(udp.got)
    assert [(a, args) for _, a, args in got] == [expected(i) for i in range(len(CUES))]

    # timing: live == reference, so arrival - (cue time - target latency) is the same for every cue
    offsets = np.array([t - t0 - (CUES[i][0] - latency(i)) for i, (t, _, _) in enumerate(got)])
    assert offsets.max() - offsets.min() < TIMING_SPREAD_S, offsets.round(3)

    fired = sorted(last["fired"], key=lambda f: f["seq"])
    assert [f["cue_id"] for f in fired] == env.ids
    assert all(f["sent"] and not f["dry_run"] and f["error"] is None for f in fired), fired
    assert all(f["reason"] in ("track", "catchup") for f in fired), [f["reason"] for f in fired]
    assert [f["target"] for f in fired] == [c[1] or "main" for c in CUES]
    assert st["counts"]["sent"] == len(CUES) and st["counts"]["errors"] == 0
    assert st["targets"]["main"]["sent"] >= 7 and st["targets"]["fx"]["sent"] >= 3

    detail = client.get(f"/api/projects/{env.pid}").json()
    assert len(detail["logs"]) >= 1
    csv = client.get(f"/api/projects/{env.pid}/logs/{detail['logs'][0]}")
    assert csv.status_code == 200
    rows = csv.text.splitlines()
    assert rows[0].startswith("wall_iso,") and len(rows) == 1 + len(CUES)


def test_rehearsal_from_a_cue_prerolls_and_chases_the_cues_before(env: Env) -> None:
    """Start at the cue at 11.6 s: the file plays from 9.6 s (2 s pre-roll) and the tracker starts there;
    the cues before 9.6 s are jumped over (skip policy "latest": the last one per target + address is
    sent at once, reason "start"), the ones after it fire in time."""
    client, udp = env.client, env.udp
    udp.got.clear()
    k = 8                                              # CUES[8] = 11.6 s
    r = client.post("/api/session/start", json={"project_id": env.pid, "start_cue": env.ids[k],
                                                "input": {"kind": "file", "stem": "mix"}})
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["source"] is not None
    starts = sorted(st["fired"], key=lambda f: f["seq"])
    chased = {(f["target"], f["address"]): f["cue_id"] for f in starts if f["reason"] == "start"}
    # last cue before 9.6 s per (target, address): /show/cue 4 (8.8), /fx/hit "two words" (7.4), the fader
    assert chased == {("main", "/show/cue"): env.ids[6], ("fx", "/fx/hit"): env.ids[5],
                      ("main", "/mixer/fader"): env.ids[4]}
    assert {f["cue_id"] for f in starts if f["reason"] == "skipped"} == {env.ids[i] for i in (0, 1, 2, 3)}
    assert all(f["ref_time"] == pytest.approx(9.6, abs=0.05) for f in starts)

    assert udp.wait_for(3 + 3, 8.0), udp.got
    time.sleep(0.3)
    got = [(a, args) for _, a, args in udp.got]
    assert sorted(got[:3]) == sorted([expected(4), expected(5), expected(6)])     # the chase, at once
    assert got[3:] == [expected(7), expected(8), expected(9)]
    state = client.get("/api/session").json()["state"]
    assert client.post("/api/session/stop").json()["status"] == "idle"
    late = {f["cue_id"]: f for f in state["fired"] if f["reason"] in ("track", "catchup")}
    assert sorted(late) == [env.ids[i] for i in (7, 8, 9)]
