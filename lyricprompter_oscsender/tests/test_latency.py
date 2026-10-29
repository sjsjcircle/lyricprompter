"""Latency measurement: echo round trips against local UDP responders, and audio loopback through a fake
input stream driven by a virtual clock (no real audio device, no real waiting for the audio tests)."""
from __future__ import annotations

import json
import math
import re
import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from lyricprompter_oscsender import cues, latency
from lyricprompter_oscsender.latency import (
    PING_ADDRESS,
    PONG_ADDRESS,
    AudioCapture,
    LatencyReport,
    detect_onset,
    measure_echo,
    measure_loopback,
    serve_echo,
)
from lyricprompter_oscsender.live import audio_input as ai
from lyricprompter_oscsender.osc import OscSender, Target, decode_message, encode_bundle, encode_message

# ----------------------------------------------------------------------------------------- helpers


def free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def echo_server(**kw):
    """serve_echo on 127.0.0.1 (ephemeral port) in a thread; stops it on exit."""
    stop, ready = threading.Event(), threading.Event()
    box: dict = {"seen": []}

    def on_ready(port: int) -> None:
        box["port"] = port
        ready.set()

    def run() -> None:
        box["replies"] = serve_echo(0, "127.0.0.1", stop=stop, on_ready=on_ready, poll_s=0.02,
                                    on_message=lambda *a: box["seen"].append(a), **kw)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    assert ready.wait(2.0)
    try:
        yield SimpleNamespace(port=box["port"], seen=box["seen"], box=box)
    finally:
        stop.set()
        th.join(2.0)
        assert not th.is_alive()


class Responder:
    """A scripted UDP responder: `handler(address, args, src)` returns [(delay_s, packet, dest | None)]."""

    def __init__(self, handler):
        self.handler = handler
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.02)
        self.port = self.sock.getsockname()[1]
        self.received: list[tuple[str, list]] = []
        self._stop = threading.Event()
        self._timers: list[threading.Timer] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _send(self, packet: bytes, dest) -> None:
        try:
            self.sock.sendto(packet, dest)
        except OSError:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, src = self.sock.recvfrom(65536)
            except TimeoutError:
                continue
            except OSError:
                return
            address, args = decode_message(data)
            self.received.append((address, args))
            for delay, packet, dest in self.handler(address, args, src):
                if delay <= 0:
                    self._send(packet, dest or src)
                else:
                    t = threading.Timer(delay, self._send, (packet, dest or src))
                    t.daemon = True
                    self._timers.append(t)
                    t.start()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(1.0)
        for t in self._timers:
            t.cancel()
        self.sock.close()


def pong(args) -> bytes:
    return encode_message(PONG_ADDRESS, args)


def seq_of(args) -> int:
    return int(args[0][1])


def json_roundtrip(d: dict) -> dict:
    return json.loads(json.dumps(d, allow_nan=False))


# ----------------------------------------------------------------------------------------- report


def test_report_statistics_on_known_samples():
    xs = [7.0, 1.0, 10.0, 3.0, 5.0, 2.0, 9.0, 4.0, 6.0, 8.0]
    rep = LatencyReport.from_samples("echo", xs, estimate_factor=0.5, host="h", port=1, sent=12, received=10)
    assert rep.samples_ms == xs                       # kept in trial order
    assert rep.min_ms == 1.0 and rep.max_ms == 10.0
    assert rep.median_ms == pytest.approx(5.5)
    assert rep.p95_ms == pytest.approx(9.55)          # linear interpolation
    assert rep.jitter_ms == pytest.approx(7.75 - 3.25)
    assert rep.estimate_ms == pytest.approx(2.75)
    assert rep.lost == 2
    one = LatencyReport.from_samples("loopback", [37.25])
    assert (one.min_ms, one.median_ms, one.p95_ms, one.max_ms, one.jitter_ms, one.estimate_ms) == \
        (37.25, 37.25, 37.25, 37.25, 0.0, 37.25)


def test_report_is_json_safe_without_samples():
    rep = LatencyReport.from_samples("echo", [], sent=5, received=0, note="no reply")
    d = json_roundtrip(rep.to_dict())
    assert d["samples_ms"] == [] and d["sent"] == 5 and d["received"] == 0
    for k in ("min_ms", "median_ms", "p95_ms", "max_ms", "jitter_ms", "estimate_ms"):
        assert d[k] is None
    assert set(d) == {"method", "target", "host", "port", "sent", "received", "samples_ms", "min_ms", "median_ms",
                      "p95_ms", "max_ms", "jitter_ms", "estimate_ms", "note", "error"}
    # non-finite values never leak: samples are dropped, stray fields become null
    rep = LatencyReport.from_samples("echo", [1.0, float("nan"), float("inf"), 3.0])
    assert rep.samples_ms == [1.0, 3.0] and rep.median_ms == 2.0
    rep.max_ms = float("nan")
    rep.samples_ms.append(np.float64("inf"))
    d = json_roundtrip(rep.to_dict())
    assert d["max_ms"] is None and d["samples_ms"] == [1.0, 3.0, None]


# ----------------------------------------------------------------------------------------- echo


def test_echo_round_trip_against_serve_echo():
    with echo_server() as srv:
        rep = measure_echo("127.0.0.1", srv.port, count=10, interval_s=0.005, timeout_s=0.5, target="main")
    assert rep.error is None
    assert rep.method == "echo" and rep.target == "main" and rep.host == "127.0.0.1" and rep.port == srv.port
    assert rep.sent == rep.received == 10 and len(rep.samples_ms) == 10
    assert 0 < rep.median_ms < 50
    assert rep.min_ms <= rep.median_ms <= rep.p95_ms <= rep.max_ms and rep.jitter_ms >= 0
    assert rep.estimate_ms == pytest.approx(rep.median_ms / 2, abs=0.002)
    assert "10/10 replies" in rep.note
    assert srv.box["replies"] == 10
    pings = [s for s in srv.seen if s[1] == PING_ADDRESS]
    assert [seq_of(s[2]) for s in pings] == list(range(1, 11))
    assert all(s[3] for s in pings) and all(s[2][1][0] == "d" for s in pings)
    json_roundtrip(rep.to_dict())


def test_serve_echo_returns_when_stopped():
    stop = threading.Event()
    out: dict = {}
    th = threading.Thread(target=lambda: out.setdefault("n", serve_echo(0, "127.0.0.1", stop=stop, poll_s=0.02)),
                          daemon=True)
    th.start()
    time.sleep(0.05)
    t0 = time.monotonic()
    stop.set()
    th.join(1.0)
    assert not th.is_alive() and time.monotonic() - t0 < 0.5
    assert out["n"] == 0


def test_serve_echo_answers_pings_only_and_survives_garbage():
    with echo_server() as srv, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as c:
        c.settimeout(0.2)
        c.sendto(b"\x00garbage", ("127.0.0.1", srv.port))
        c.sendto(encode_message("/cue/go", [("i", 3)]), ("127.0.0.1", srv.port))
        with pytest.raises(TimeoutError):
            c.recvfrom(65536)
        c.sendto(encode_bundle([encode_message(PING_ADDRESS, [("i", 7), ("d", 1.5)])]), ("127.0.0.1", srv.port))
        data, _ = c.recvfrom(65536)
        assert decode_message(data) == (PONG_ADDRESS, [("i", 7), ("d", 1.5)])
    assert [(s[1], s[3]) for s in srv.seen] == [("/cue/go", False), (PING_ADDRESS, True)]


def test_serve_echo_all_mirrors_packets_unchanged():
    msg = encode_message("/clip/play", [("s", "intro.mp4"), ("f", 0.5), ("T", True)])
    bundle = encode_bundle([encode_message("/a", [("i", 1)]), encode_message("/b")])
    with echo_server(echo_all=True) as srv, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as c:
        c.settimeout(1.0)
        for packet in (msg, bundle):
            c.sendto(packet, ("127.0.0.1", srv.port))
            assert c.recvfrom(65536)[0] == packet
    assert [s[1] for s in srv.seen] == ["/clip/play", "/a", "/b"] and all(s[3] for s in srv.seen)


def test_serve_echo_port_in_use_raises():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as busy:
        busy.bind(("127.0.0.1", 0))
        with pytest.raises(OSError):
            serve_echo(busy.getsockname()[1], "127.0.0.1", stop=threading.Event())
    with pytest.raises(ValueError):
        serve_echo(70000, "127.0.0.1", stop=threading.Event())


def test_echo_packet_loss_respects_timeouts():
    def drop_even(address, args, src):
        return [] if seq_of(args) % 2 == 0 else [(0, pong(args), None)]

    with Responder(drop_even) as r:
        t0 = time.monotonic()
        rep = measure_echo("127.0.0.1", r.port, count=10, interval_s=0.01, timeout_s=0.1)
        elapsed = time.monotonic() - t0
    assert rep.error is None
    assert rep.sent == 10 and rep.received == 5 and rep.lost == 5 and len(rep.samples_ms) == 5
    assert elapsed >= 5 * 0.1 - 0.01                  # every lost ping waited its full timeout
    assert elapsed < 1.5                              # ... and no longer
    assert "5 lost" in rep.note and "100 ms" in rep.note
    assert len(r.received) == 10


def test_echo_ignores_stale_replies():
    # ping 1 is answered after its timeout, while ping 2 is waiting; ping 2's own pong comes 50 ms later.
    def handler(address, args, src):
        return {1: [(0.3, pong(args), None)], 2: [(0.15, pong(args), None)]}.get(seq_of(args), [(0, pong(args), None)])

    with Responder(handler) as r:
        rep = measure_echo("127.0.0.1", r.port, count=3, interval_s=0.01, timeout_s=0.2)
    assert rep.sent == 3 and rep.received == 2
    assert rep.samples_ms[0] >= 140                   # ping 2's own pong, not ping 1's late one (~100 ms)
    assert rep.samples_ms[1] < 50


def test_echo_ignores_foreign_packets():
    def handler(address, args, src):
        seq = seq_of(args)
        return [(0, b"not OSC at all", None), (0, encode_message("/other", args), None),
                (0, pong([("i", seq + 100)]), None), (0, pong([("s", str(seq))]), None),
                (0, encode_message(PONG_ADDRESS), None), (0.02, pong(args), None)]

    with Responder(handler) as r:
        rep = measure_echo("127.0.0.1", r.port, count=3, interval_s=0.0, timeout_s=0.5)
    assert rep.sent == rep.received == 3
    assert min(rep.samples_ms) >= 15                  # only the real pong, 20 ms after each ping, counted


def test_echo_generic_message_takes_the_first_reply_of_any_address():
    seen: list[str] = []

    def qlab_like(address, args, src):
        seen.append(address)
        if address != "/version":
            return []
        reply = encode_message("/reply/version", [("s", "5.0")])
        return [(0.01, reply, None), (0.015, reply, None)]     # a duplicate: drained before the next send

    with Responder(qlab_like) as r:
        rep = measure_echo("127.0.0.1", r.port, count=5, interval_s=0.05, timeout_s=0.3, message=("/version", ()))
    assert seen == ["/version"] * 5
    assert rep.sent == rep.received == 5
    assert rep.min_ms >= 9                            # never the duplicate of the previous reply
    assert rep.estimate_ms == pytest.approx(rep.median_ms / 2, abs=0.002)
    assert "reply processing" in rep.note


def test_echo_generic_without_reply_and_echo_all_server():
    with Responder(lambda *a: []) as r:
        rep = measure_echo("127.0.0.1", r.port, count=2, interval_s=0.0, timeout_s=0.05,
                           message=("/cue/go", [("i", 1)]))
    assert rep.sent == 2 and rep.received == 0 and rep.estimate_ms is None and rep.error is None
    assert "no reply" in rep.note
    assert r.received == [("/cue/go", [("i", 1)])] * 2
    json_roundtrip(rep.to_dict())
    with echo_server(echo_all=True) as srv:
        rep = measure_echo("127.0.0.1", srv.port, count=3, interval_s=0.0, message=("/cue/go", [("i", 1)]))
    assert rep.received == 3


def test_echo_listen_port_for_fixed_reply_port():
    lp = free_udp_port()

    def to_fixed_port(address, args, src):
        return [(0, pong(args), ("127.0.0.1", lp))]

    with Responder(to_fixed_port) as r:
        rep = measure_echo("127.0.0.1", r.port, count=3, interval_s=0.0, timeout_s=0.5, listen_port=lp)
    assert rep.error is None and rep.received == 3


def test_echo_errors_are_reported_not_raised():
    rep = measure_echo("127.0.0.1", 70000, count=3)
    assert rep.error and "cannot resolve" in rep.error and rep.sent == 0 and rep.estimate_ms is None
    json_roundtrip(rep.to_dict())
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as busy:
        busy.bind(("0.0.0.0", 0))
        rep = measure_echo("127.0.0.1", 9, count=1, listen_port=busy.getsockname()[1])
    assert rep.error and "cannot listen" in rep.error and rep.sent == 0
    rep = measure_echo("127.0.0.1", 9, message=("no-slash", ()))
    assert rep.error and "cannot encode" in rep.error
    rep = measure_echo("127.0.0.1", 9, listen_port=70000)
    assert rep.error and "cannot listen" in rep.error
    with pytest.raises(ValueError):
        measure_echo("127.0.0.1", 9, count=0)


def test_echo_without_responder():
    port = free_udp_port()
    rep = measure_echo("127.0.0.1", port, count=2, interval_s=0.0, timeout_s=0.05)
    assert rep.sent == 2 and rep.received == 0 and rep.error is None and rep.median_ms is None
    assert "oscsender echo" in rep.note


# ----------------------------------------------------------------------------------------- fake audio


class VClock:
    """Virtual clock; `sleep` advances it and runs the fake streams' callbacks that fall due."""

    def __init__(self, t: float = 1000.0):
        self.t = t
        self.streams: list[FakeStream] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        end = self.t + max(0.0, dt)
        while True:
            due = [(s.next_callback(), i) for i, s in enumerate(self.streams) if s.active]
            due = [d for d in due if d[0] <= end]
            if not due:
                break
            when, i = min(due)
            self.t = max(self.t, when)
            self.streams[i].fire()
        self.t = end


class FakeStream:
    """A PortAudio input stream: block k (first sample ADC time adc0 + k*B/sr) is delivered `delivery_s`
    after its last sample, with time_info in a PortAudio clock offset by `pa_offset`."""

    def __init__(self, vc: VClock, scene, *, delivery_s=0.006, reported_latency=None, time_info=True,
                 pa_offset=5000.0, stall_after=None, adc_age=None, **kw):
        self.vc, self.scene, self.kw = vc, scene, kw
        self.samplerate = float(kw["samplerate"] or 48000)
        self.block = int(kw["blocksize"] or 256)
        self.channels = int(kw["channels"])
        self.callback = kw["callback"]
        self.delivery_s = delivery_s
        self.latency = delivery_s if reported_latency is None else reported_latency
        self.time_info, self.pa_offset, self.stall_after, self.adc_age = time_info, pa_offset, stall_after, adc_age
        self.active = self.closed = False
        self.block_starts: list[float] = []

    def start(self):
        self.active, self.k, self.adc0 = True, 0, self.vc.t
        self.vc.streams.append(self)

    def next_callback(self) -> float:
        if self.stall_after is not None and self.k >= self.stall_after:
            return math.inf
        return self.adc0 + (self.k + 1) * self.block / self.samplerate + self.delivery_s

    def fire(self):
        t = self.adc0 + (self.k * self.block + np.arange(self.block)) / self.samplerate
        x = np.asarray(self.scene(t, self.channels), np.float32)
        if self.time_info:
            age = (self.vc.t - t[0]) if self.adc_age is None else self.adc_age
            ti = SimpleNamespace(currentTime=self.vc.t + self.pa_offset,
                                 inputBufferAdcTime=self.vc.t - age + self.pa_offset)
        else:
            ti = SimpleNamespace(currentTime=0.0, inputBufferAdcTime=0.0)
        self.block_starts.append(float(t[0]))
        self.k += 1
        self.callback(x, self.block, ti, SimpleNamespace(input_overflow=False))

    def stop(self):
        self.active = False

    def close(self):
        self.active, self.closed = False, True


def stream_factory(vc: VClock, scene, made: list | None = None, **opts):
    def make(**kw):
        s = FakeStream(vc, scene, **opts, **kw)
        if made is not None:
            made.append(s)
        return s
    return make


class ClickScene:
    """Noise floor plus a decaying 1 kHz click `delay_s` after every send (on `click_channel`)."""

    def __init__(self, vc: VClock, delay_s: float = 0.037, noise_db: float | None = -60.0, respond: bool = True,
                 click_channel: int = 0, click_s: float = 0.03, seed: int = 1):
        self.vc, self.delay_s, self.respond, self.click_channel, self.click_s = vc, delay_s, respond, click_channel, click_s
        self.noise = 0.0 if noise_db is None else 10.0 ** (noise_db / 20.0)
        self.rng = np.random.default_rng(seed)
        self.sends: list[float] = []
        self.clicks: list[float] = []

    def send(self) -> float:
        t = self.vc()
        self.sends.append(t)
        if self.respond:
            self.clicks.append(t + self.delay_s)
        return t

    def __call__(self, t: np.ndarray, channels: int) -> np.ndarray:
        x = self.rng.standard_normal((len(t), channels)) * self.noise
        for c in self.clicks:
            dt = t - c
            m = (dt >= 0) & (dt < self.click_s)
            x[m, self.click_channel] += 0.3 * np.sin(2 * np.pi * 1000.0 * dt[m]) * np.exp(-dt[m] / 0.008)
        return x


def time_code_scene(origin: float):
    """Every channel carries its sample's own ADC time (relative to `origin`) plus 10 * channel number."""
    def scene(t: np.ndarray, channels: int) -> np.ndarray:
        return (t - origin)[:, None] + 10.0 * np.arange(1, channels + 1)[None, :]
    return scene


# ----------------------------------------------------------------------------------------- audio capture


@pytest.mark.parametrize("delivery_s,reported", [(0.008, 0.002), (0.003, 0.003)])
def test_capture_adc_timestamps_from_time_info(delivery_s, reported):
    vc = VClock()
    made: list = []
    cap = AudioCapture(channels=[1], samplerate=16000, clock=vc,
                       stream_factory=stream_factory(vc, time_code_scene(vc.t), made, delivery_s=delivery_s,
                                                     reported_latency=reported))
    cap.start()
    vc.sleep(0.5)
    times, x = cap.segment(0.0, 1e9)
    starts = np.asarray(made[0].block_starts)
    assert len(times) == len(starts) * 256
    # stamped with the true ADC time of every sample, whatever latency the stream reports
    assert np.allclose(times[::256], starts, atol=1e-9)
    assert np.allclose(np.diff(times), 1 / 16000, atol=1e-9)
    assert np.allclose(x, times - 1000.0 + 10.0, atol=1e-4)
    assert cap.stats()["timestamps"] == "adc" and cap.stats()["input_latency_s"] == reported
    cap.stop()
    assert made[0].closed and not cap.running
    assert len(cap.segment(0.0, 1e9)[0]) == len(times)          # still readable after stop


def test_capture_without_time_info_uses_reported_latency():
    vc = VClock()
    made: list = []
    cap = AudioCapture(samplerate=16000, clock=vc,
                       stream_factory=stream_factory(vc, time_code_scene(vc.t), made, delivery_s=0.008,
                                                     reported_latency=0.002, time_info=False))
    cap.start()
    vc.sleep(0.3)
    times, _ = cap.segment(0.0, 1e9)
    starts = np.asarray(made[0].block_starts)
    # clock_at_callback - stream.latency - frames / sr: off by (true delivery delay - reported latency)
    assert np.allclose(times[::256], starts + (0.008 - 0.002), atol=1e-9)
    assert cap.stats()["timestamps"] == "estimated"
    cap.stop()


def test_capture_ignores_implausible_adc_times():
    vc = VClock()
    made: list = []
    cap = AudioCapture(samplerate=16000, clock=vc,
                       stream_factory=stream_factory(vc, time_code_scene(vc.t), made, delivery_s=0.004, adc_age=5.0))
    cap.start()
    vc.sleep(0.1)
    times, _ = cap.segment(0.0, 1e9)
    assert np.allclose(times[::256], made[0].block_starts, atol=1e-9)    # fallback (reported == delivery)
    assert cap.stats()["timestamps"] == "estimated"
    cap.stop()


def test_capture_ring_buffer_and_segments():
    vc = VClock()
    cap = AudioCapture(samplerate=16000, clock=vc, buffer_s=0.05, blocksize=100,
                       stream_factory=stream_factory(vc, time_code_scene(vc.t)))
    assert cap.latest_time() is None and cap.earliest_time() is None
    assert len(cap.segment(0, 1e9)[0]) == 0
    cap.start()
    vc.sleep(1.0)
    latest, earliest = cap.latest_time(), cap.earliest_time()
    assert latest - earliest == pytest.approx(0.05)
    times, x = cap.segment(0.0, 1e9)
    assert len(times) == 800 and times[0] == pytest.approx(earliest) and times[-1] == pytest.approx(latest - 1 / 16000)
    assert np.allclose(x, times - 1000.0 + 10.0, atol=1e-4)        # samples stay aligned across the wrap
    t0, t1 = earliest + 0.0101, earliest + 0.0302
    times, x = cap.segment(t0, t1)
    assert times[0] >= t0 and times[-1] < t1 and len(times) == pytest.approx((t1 - t0) * 16000, abs=1)
    assert np.allclose(np.diff(times), 1 / 16000) and np.allclose(x, times - 1000.0 + 10.0, atol=1e-4)
    assert len(cap.segment(t1, t0)[0]) == 0 and len(cap.segment(latest + 1, latest + 2)[0]) == 0
    cap.stop()


def test_capture_huge_block_keeps_the_newest_samples():
    vc = VClock()
    cap = AudioCapture(samplerate=1000, clock=vc, buffer_s=0.1, blocksize=250,
                       stream_factory=stream_factory(vc, time_code_scene(vc.t)))
    cap.start()
    vc.sleep(0.3)
    times, x = cap.segment(0.0, 1e9)
    assert len(times) == 100 and np.allclose(np.diff(times), 0.001)
    assert np.allclose(x, times - 1000.0 + 10.0, atol=1e-4)
    cap.stop()


def test_capture_downmixes_selected_channels():
    vc = VClock()
    made: list = []
    cap = AudioCapture(channels=[3, 2], samplerate=16000, clock=vc,
                       stream_factory=stream_factory(vc, time_code_scene(vc.t), made))
    cap.start()
    vc.sleep(0.05)
    times, x = cap.segment(0, 1e9)
    assert made[0].kw["channels"] == 3 and cap.channels == [2, 3]
    assert np.allclose(x, times - 1000.0 + 25.0, atol=1e-4)        # mean of channels 2 (20) and 3 (30)
    cap.stop()
    with pytest.raises(ValueError):
        AudioCapture(channels=[0])


class FakeSD:
    PortAudioError = RuntimeError

    def __init__(self):
        self.opened: list[dict] = []
        sd = self

        class InputStream:
            def __init__(self, **kw):
                sd.opened.append(kw)
                self.kw, self.samplerate, self.latency = kw, kw["samplerate"], 0.005
                self.started = self.closed = False

            def start(self):
                self.started = True

            def stop(self):
                raise RuntimeError("device lost")             # stop() must not turn this into an error

            def close(self):
                self.closed = True

        self.InputStream = InputStream

    def query_devices(self, device=None, kind=None):
        assert kind == "input"
        return {"name": "Stage Interface", "index": 2, "max_input_channels": 4, "default_samplerate": 44100.0}


def test_capture_opens_a_sounddevice_stream(monkeypatch):
    sd = FakeSD()
    monkeypatch.setattr(ai, "_sounddevice", lambda: sd)
    cap = AudioCapture(device="Stage", channels=[1, 3])
    cap.start()
    kw = sd.opened[0]
    assert kw["device"] == 2 and kw["channels"] == 3 and kw["samplerate"] == 44100.0 and kw["dtype"] == "float32"
    assert cap.samplerate == 44100.0 and cap.running
    block = np.tile(np.array([[0.2, 9.0, 0.4]], np.float32), (441, 1))
    kw["callback"](block, 441, None, SimpleNamespace(input_overflow=True))
    _times, x = cap.segment(0, 1e12)
    assert len(x) == 441 and np.allclose(x, 0.3) and cap.overflows == 1
    assert cap.stats()["timestamps"] == "estimated"
    cap.stop()
    assert not cap.running
    with pytest.raises(ValueError):
        AudioCapture(device=2, channels=[5]).start()


# ----------------------------------------------------------------------------------------- loopback


def test_detect_onset_on_known_signal():
    sr = 48000
    t_send = 50.0
    times = t_send - 0.3 + np.arange(int(1.3 * sr)) / sr
    x = np.zeros_like(times, dtype=np.float32)                    # digital silence: threshold -70 dBFS
    m = times >= t_send + 0.1234
    x[m] = 0.05 * np.sin(2 * np.pi * 500 * (times[m] - t_send - 0.1234) + 0.5)
    onset, floor = detect_onset(times, x, t_send)
    assert floor < -150 and onset == pytest.approx(t_send + 0.1234, abs=1e-4)
    # a sound starting late in a 1 ms frame that only the next frame's RMS reveals is not reported a frame late
    rng = np.random.default_rng(3)
    noise = (rng.standard_normal(len(times)) * 0.001).astype(np.float32)        # -60 dBFS: threshold -40
    start = t_send + 0.2009
    m = times >= start
    y = noise.copy()
    y[m] += 0.02 * np.cos(2 * np.pi * 200 * (times[m] - start))
    onset, floor = detect_onset(times, y, t_send)
    assert floor == pytest.approx(-60, abs=1.0) and onset == pytest.approx(start, abs=1e-4)
    onset, floor = detect_onset(times, x, t_send, window_s=0.1)   # after the window: no onset
    assert onset is None and floor is not None
    keep = times >= t_send                                       # no history before the send: no floor
    assert detect_onset(times[keep], x[keep], t_send) == (None, None)


@pytest.mark.parametrize("sr,time_info", [(48000, True), (16000, False)])
def test_loopback_measures_a_known_delay(sr, time_info):
    vc = VClock()
    scene = ClickScene(vc, delay_s=0.037)
    made: list = []
    cap = AudioCapture(samplerate=sr, clock=vc,
                       stream_factory=stream_factory(vc, scene, made, delivery_s=0.006, time_info=time_info))
    rep = measure_loopback(scene.send, cap, count=8, clock=vc, sleep=vc.sleep, target="sampler")
    assert rep.error is None and rep.method == "loopback" and rep.target == "sampler"
    assert rep.sent == rep.received == 8 and len(rep.samples_ms) == 8
    assert rep.estimate_ms == pytest.approx(37.0, abs=2.0)
    assert all(abs(s - 37.0) <= 2.0 for s in rep.samples_ms)
    assert "8/8 sounds detected" in rep.note
    floor = float(re.search(r"floor (-?[\d.]+) dBFS", rep.note).group(1))
    assert floor == pytest.approx(-60.0, abs=1.0)
    assert ("estimated" in rep.note) == (not time_info)
    assert not cap.running and made[0].closed                     # started here, so stopped here
    gaps = np.diff(scene.sends)
    assert np.all(gaps < 0.5) and np.all(gaps > 0.037 + 0.03)     # next trial once the click has died away
    json_roundtrip(rep.to_dict())


def test_loopback_selected_channel_and_running_capture():
    vc = VClock()
    scene = ClickScene(vc, delay_s=0.012, click_channel=1)
    cap = AudioCapture(channels=[2], samplerate=16000, clock=vc, stream_factory=stream_factory(vc, scene))
    cap.start()
    rep = measure_loopback(scene.send, cap, count=3, clock=vc, sleep=vc.sleep)
    assert rep.received == 3 and rep.estimate_ms == pytest.approx(12.0, abs=2.0)
    assert cap.running                                            # the caller's capture stays open
    cap.stop()


def test_loopback_without_a_sound_reports_misses():
    vc = VClock()
    scene = ClickScene(vc, respond=False)
    cap = AudioCapture(samplerate=16000, clock=vc, stream_factory=stream_factory(vc, scene))
    rep = measure_loopback(scene.send, cap, count=4, interval_s=0.6, window_s=0.4, clock=vc, sleep=vc.sleep)
    assert rep.error is None and rep.sent == 4 and rep.received == 0
    assert rep.estimate_ms is None and rep.median_ms is None and rep.samples_ms == []
    assert "no sound detected in 4 trials" in rep.note
    assert np.allclose(np.diff(scene.sends), 0.6, atol=0.011)     # a miss waits the whole interval
    d = json_roundtrip(rep.to_dict())
    assert d["estimate_ms"] is None


def test_loopback_sound_later_than_the_window_never_leaks_into_the_next_trial():
    vc = VClock()
    scene = ClickScene(vc, delay_s=0.45)
    cap = AudioCapture(samplerate=16000, clock=vc, stream_factory=stream_factory(vc, scene))
    rep = measure_loopback(scene.send, cap, count=3, interval_s=0.6, window_s=0.3, pre_s=0.1, clock=vc,
                           sleep=vc.sleep)
    assert rep.sent == 3 and rep.received == 0


def test_loopback_digital_silence_floor_and_threshold():
    vc = VClock()
    scene = ClickScene(vc, delay_s=0.0205, noise_db=None)
    cap = AudioCapture(samplerate=16000, clock=vc, stream_factory=stream_factory(vc, scene))
    rep = measure_loopback(scene.send, cap, count=3, clock=vc, sleep=vc.sleep)
    assert rep.received == 3 and rep.estimate_ms == pytest.approx(20.5, abs=1.0)
    assert "threshold -70.0 dBFS" in rep.note


def test_loopback_errors():
    vc = VClock()
    scene = ClickScene(vc)
    cap = AudioCapture(samplerate=16000, clock=vc, stream_factory=stream_factory(vc, scene))

    def broken_send() -> float:
        raise RuntimeError("target 'x' is disabled")

    rep = measure_loopback(broken_send, cap, count=3, clock=vc, sleep=vc.sleep)
    assert rep.sent == 0 and "target 'x' is disabled" in rep.error and not cap.running

    cap = AudioCapture(samplerate=16000, clock=vc, stream_factory=stream_factory(vc, scene, stall_after=0))
    rep = measure_loopback(scene.send, cap, count=3, clock=vc, sleep=vc.sleep)
    assert rep.sent == 0 and "no audio" in rep.error

    vc = VClock()
    scene = ClickScene(vc, respond=False)
    cap = AudioCapture(samplerate=16000, clock=vc, stream_factory=stream_factory(vc, scene, stall_after=30))
    rep = measure_loopback(scene.send, cap, count=3, clock=vc, sleep=vc.sleep)
    assert rep.sent == 1 and "stopped delivering" in rep.error

    def no_device(**kw):
        raise RuntimeError("Error opening InputStream: Invalid device")

    rep = measure_loopback(scene.send, AudioCapture(clock=vc, stream_factory=no_device), clock=vc, sleep=vc.sleep)
    assert "cannot open the audio input" in rep.error and rep.sent == 0
    json_roundtrip(rep.to_dict())
    with pytest.raises(ValueError):
        measure_loopback(scene.send, cap, count=0)


def test_target_send_uses_the_sender_clock():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as rx:
        rx.bind(("127.0.0.1", 0))
        rx.settimeout(1.0)
        sender = OscSender(clock=lambda: 123.5)
        sender.set_targets([Target("sampler", "127.0.0.1", rx.getsockname()[1]),
                            Target("off", "127.0.0.1", 9, enabled=False)])
        assert latency.target_send(sender, "sampler", "/play", [("i", 1)])() == 123.5
        assert decode_message(rx.recvfrom(65536)[0]) == ("/play", [("i", 1)])
        with pytest.raises(RuntimeError, match="disabled"):
            latency.target_send(sender, "off", "/play")()
        sender.close()


# ----------------------------------------------------------------------------------------- cli


def test_message_from_text_with_and_without_the_cue_parser(monkeypatch):
    def not_yet(text):
        raise NotImplementedError

    monkeypatch.setattr(cues, "parse_message", not_yet, raising=False)
    assert latency.message_from_text("/version") == ("/version", ())
    assert latency.message_from_text("  /cue/go 3 ") == ("/cue/go", ())       # bare address fallback
    with pytest.raises(ValueError):
        latency.message_from_text("go")
    with pytest.raises(ValueError):
        latency.message_from_text("   ")
    monkeypatch.setattr(cues, "parse_message", lambda text: ("lights", "/cue/go", (("i", 3),), "note"),
                        raising=False)
    assert latency.message_from_text("lights /cue/go 3 # note") == ("/cue/go", (("i", 3),))


def test_cli_latency_against_echo(capsys):
    with echo_server() as srv:
        code = latency.cli_latency(["--host", "127.0.0.1", "--port", str(srv.port), "--count", "3",
                                    "--interval-ms", "5"])
        out = capsys.readouterr().out
        assert code == 0
        assert "3 sent, 3 received" in out and "suggested target latency" in out
        code = latency.cli_latency(["--host", "127.0.0.1", "--port", str(srv.port), "--count", "2", "--json"])
        d = json.loads(capsys.readouterr().out)
        assert code == 0 and d["method"] == "echo" and d["received"] == 2 and d["estimate_ms"] is not None


def test_cli_latency_message_mode_and_failures(capsys):
    with echo_server(echo_all=True) as srv:
        assert latency.cli_latency(["--host", "127.0.0.1", "--port", str(srv.port), "--count", "2",
                                    "--message", "/version"]) == 0
    assert "2 sent, 2 received" in capsys.readouterr().out
    port = free_udp_port()
    assert latency.cli_latency(["--host", "127.0.0.1", "--port", str(port), "--count", "1",
                                "--timeout-ms", "30"]) == 1
    assert "no reply" in capsys.readouterr().out
    assert latency.cli_latency(["--host", "127.0.0.1", "--port", str(port), "--message", "no-slash"]) == 2
    assert "--message" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        latency.cli_latency(["--host", "127.0.0.1", "--count", "0"])


def test_cli_dispatches_latency(capsys):
    from lyricprompter_oscsender.cli import main

    with echo_server() as srv:
        assert main(["latency", "--host", "127.0.0.1", "--port", str(srv.port), "--count", "1"]) == 0
    assert "1 sent, 1 received" in capsys.readouterr().out


def test_cli_echo_prints_answered_pings(monkeypatch, capsys):
    real = latency.serve_echo

    def serve_then_stop(port, host, **kw):
        stop = threading.Event()
        on_ready = kw.pop("on_ready")

        def ready(p: int) -> None:
            on_ready(p)

            def client() -> None:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as c:
                    c.settimeout(1.0)
                    c.sendto(encode_message("/cue/go", [("i", 1)]), ("127.0.0.1", p))
                    c.sendto(encode_message(PING_ADDRESS, [("i", 42), ("d", 0.5)]), ("127.0.0.1", p))
                    c.recvfrom(65536)
                stop.set()
            threading.Thread(target=client, daemon=True).start()

        return real(port, host, stop=stop, on_ready=ready, **kw)

    monkeypatch.setattr(latency, "serve_echo", serve_then_stop)
    assert latency.cli_echo(["--host", "127.0.0.1", "--port", "0"]) == 0
    out = capsys.readouterr().out
    assert "answering /oscsender/ping on UDP 127.0.0.1:" in out
    assert "ping #42 from 127.0.0.1:" in out and "ignored /cue/go" in out

    def interrupted(port, host, **kw):
        kw["on_ready"](port)
        kw["on_message"](("10.0.0.2", 5000), "/a/b", [("f", 0.5), ("s", "x y"), ("T", True)], True)
        raise KeyboardInterrupt

    monkeypatch.setattr(latency, "serve_echo", interrupted)
    assert latency.cli_echo(["--echo-all", "--port", "9001"]) == 0
    out = capsys.readouterr().out
    assert "every OSC message" in out and '/a/b 0.5 "x y" true  from 10.0.0.2:5000' in out
    assert "stopped (1 answered)" in out
    assert latency.cli_echo(["--quiet"]) == 0
    assert "/a/b" not in capsys.readouterr().out


def test_cli_echo_port_in_use(capsys):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as busy:
        busy.bind(("127.0.0.1", 0))
        assert latency.cli_echo(["--host", "127.0.0.1", "--port", str(busy.getsockname()[1])]) == 1
    assert "cannot listen" in capsys.readouterr().err
