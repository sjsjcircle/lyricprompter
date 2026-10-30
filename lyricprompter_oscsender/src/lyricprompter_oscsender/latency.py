"""OSC output latency: network echo round trips, an echo responder, and audio loopback.

A target's `latency_ms` (osc.Target) is how long the receiver takes from our send to its effect; the
scheduler sends that target's cues this much earlier. Two automatic measurements fill it in:

* **Network echo** — `measure_echo` sends sequential pings from one UDP socket and times the replies.
  With `message=None` it sends `/oscsender/ping i:seq d:send_time` and accepts only an
  `/oscsender/pong` whose first argument is that `seq` (run `serve_echo` / `oscsender echo` on the
  receiving machine): late answers to earlier pings, other addresses and packets that are not OSC are
  ignored. With `message=(address, args)` it sends that message instead and takes the first OSC packet
  that comes back as the reply (receivers that answer queries, e.g. QLab, or are patched to echo).
  Anything already waiting on the socket is drained before every send, so a late reply can only be
  taken for the next one if it arrives after that next send. The next ping goes out `interval_s` after
  the previous one, but never before its reply or its `timeout_s`. The estimate is half the median
  round trip: the one-way network delay, without the receiver's own reaction time.
* **Audio loopback** — `measure_loopback` makes a receiver that produces sound play its test message and
  listens for it on an audio input (`AudioCapture`). Every captured block is stamped with the clock
  time its first sample was captured at the ADC, so the measured delay excludes our own input latency
  and covers everything from the send to the sound reaching the microphone (network, receiver, its
  audio output). Detection: 1 ms RMS envelope; the floor is the median envelope over the `pre_s`
  before the send; the onset is the first envelope frame within `window_s` after the send above
  `max(floor + threshold_db, -70 dBFS)`, refined to the first sample of that frame (or the frame
  before) whose magnitude reaches the threshold. A trial without an onset is a miss. After a hit the next trial starts once
  the input is back near the floor (the sound has died away; at the latest `interval_s` after the
  send); after a miss it waits the whole `interval_s`, so a sound later than the window cannot land in
  the next trial.

`LatencyReport` statistics cover the successful trials only; lost pings and missed sounds show as
`sent - received`. Times on the measurement side come from one injectable clock (`time.perf_counter`
by default, the clock `osc.OscSender` stamps its sends with).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import socket
import struct
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Self

import numpy as np

from .osc import OscArg, OscError, OscSender, decode_packet, encode_message, resolve, validate_address

log = logging.getLogger(__name__)

PING_ADDRESS = "/oscsender/ping"
PONG_ADDRESS = "/oscsender/pong"
DEFAULT_ECHO_PORT = 9001
MAX_DATAGRAM = 65536

FRAME_S = 0.001              # loopback envelope resolution
MIN_THRESHOLD_DB = -70.0     # an onset is never searched below this level (digitally silent inputs)
RECOVER_DB = 6.0             # "back near the floor": within this many dB (at most half the threshold) ...
RECENT_S = 0.05              # ... for 90 % of the last 50 ms after the onset (the sound has died away), and
                             # the median of the last `pre_s` (the next trial's floor is not inflated)
MAX_ADC_AGE_S = 1.0          # PortAudio ADC times claiming an older first sample are treated as missing


def _json_safe(x: Any) -> Any:
    """`x` with non-finite floats replaced by None, recursively (browsers reject NaN / Infinity)."""
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    return x


# ----------------------------------------------------------------------------------------- report


@dataclass
class LatencyReport:
    """One measurement. All times in ms; statistics over the successful trials only."""
    method: str                         # "echo" | "loopback"
    target: str | None = None
    host: str | None = None
    port: int | None = None
    sent: int = 0
    received: int = 0
    samples_ms: list[float] = field(default_factory=list)   # echo: round trips; loopback: send -> sound
    min_ms: float | None = None
    median_ms: float | None = None
    p95_ms: float | None = None
    max_ms: float | None = None
    jitter_ms: float | None = None      # interquartile range
    estimate_ms: float | None = None    # suggested target latency
    note: str = ""
    error: str | None = None

    @classmethod
    def from_samples(cls, method: str, samples_ms: Sequence[float], *, estimate_factor: float = 1.0,
                     **fields: Any) -> LatencyReport:
        """A report with the statistics of `samples_ms` (non-finite samples are dropped);
        `estimate_ms = median * estimate_factor`. `fields` are the other dataclass fields."""
        xs = [round(float(v), 3) for v in samples_ms if math.isfinite(float(v))]
        rep = cls(method, samples_ms=xs, **fields)
        if xs:
            q25, q50, q75, q95 = (float(q) for q in np.percentile(np.asarray(xs, dtype=np.float64),
                                                                  [25, 50, 75, 95]))
            rep.min_ms, rep.max_ms = min(xs), max(xs)
            rep.median_ms, rep.p95_ms = round(q50, 3), round(q95, 3)
            rep.jitter_ms = round(q75 - q25, 3)
            rep.estimate_ms = round(q50 * estimate_factor, 3)
        return rep

    @property
    def lost(self) -> int:
        return max(0, self.sent - self.received)

    def to_dict(self) -> dict:
        """JSON-safe dict (no NaN / Infinity)."""
        return _json_safe(asdict(self))


# ----------------------------------------------------------------------------------------- echo


def _drain(sock: socket.socket) -> None:
    """Discard every datagram already waiting on `sock`."""
    sock.setblocking(False)
    try:
        for _ in range(10000):
            try:
                sock.recvfrom(MAX_DATAGRAM)
            except BlockingIOError:
                return
            except OSError:          # e.g. an ICMP error reported on the socket: nothing to read
                continue
    finally:
        sock.setblocking(True)


def _same_seq(arg: OscArg, seq: int) -> bool:
    tag, value = arg
    return tag in ("i", "h", "f", "d") and not isinstance(value, bool) and value == seq


def _await_reply(sock: socket.socket, clock: Callable[[], float], t0: float, deadline: float,
                 seq: int | None) -> float | None:
    """Seconds from `t0` (clock) to the matching reply, None when none arrives before `deadline`
    (time.monotonic). `seq=None` takes the first OSC packet; otherwise only the pong for `seq`."""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        sock.settimeout(remaining)
        try:
            data, _src = sock.recvfrom(MAX_DATAGRAM)
        except TimeoutError:
            return None
        except OSError:              # an ICMP error on some platforms; keep waiting until the deadline
            time.sleep(0.001)
            continue
        t1 = clock()
        try:
            messages = decode_packet(data)
        except (ValueError, RecursionError, struct.error):     # not OSC (or a pathological bundle)
            continue
        if seq is None:
            return t1 - t0
        if any(addr == PONG_ADDRESS and args and _same_seq(args[0], seq) for addr, args in messages):
            return t1 - t0


def _echo_note(sent: int, received: int, timeout_s: float, generic: bool, host: str, port: int,
               listen_port: int | None) -> str:
    timeout_ms = f"{timeout_s * 1000:g} ms"
    if sent and not received:
        if generic:
            where = f"to port {listen_port}" if listen_port else "(set a listen port if it replies to a fixed port)"
            return (f"no reply within {timeout_ms}: does the receiver answer this message and send the reply back "
                    f"to this machine {where}? A firewall may also block the UDP reply.")
        return (f"no reply within {timeout_ms}: is `oscsender echo --port {port}` running on {host}? "
                "A firewall may also block UDP between the machines.")
    parts = [f"{received}/{sent} replies"]
    if sent > received:
        parts.append(f"{sent - received} lost (no reply within {timeout_ms})")
    if received:
        parts.append("estimate = half the median time to the reply (includes the receiver's reply processing)"
                     if generic else
                     "estimate = half the median round trip (one-way network delay; the receiver's own "
                     "reaction time is not included)")
    return "; ".join(parts)


def measure_echo(host: str, port: int, *, count: int = 20, interval_s: float = 0.05, timeout_s: float = 0.5,
                 message: tuple[str, Sequence[OscArg]] | None = None, listen_port: int | None = None,
                 clock: Callable[[], float] = time.perf_counter, target: str | None = None) -> LatencyReport:
    """Round trips to an OSC responder (see the module docstring). `samples_ms` are round trips,
    `estimate_ms` is half their median. Network problems never raise: they end the measurement and
    come back as `error` (with the samples gathered so far). Waiting uses time.monotonic; `clock`
    only stamps the send / receive times."""
    if count < 1:
        raise ValueError("count must be >= 1")
    if timeout_s <= 0 or interval_s < 0:
        raise ValueError("timeout_s must be > 0 and interval_s >= 0")
    generic = message is not None
    rtts: list[float] = []
    sent = 0

    def report(error: str | None = None) -> LatencyReport:
        note = "" if error and not sent else _echo_note(sent, len(rtts), timeout_s, generic, host, port, listen_port)
        return LatencyReport.from_samples("echo", [r * 1000.0 for r in rtts], estimate_factor=0.5, target=target,
                                          host=host, port=port, sent=sent, received=len(rtts), note=note,
                                          error=error)

    fixed: bytes | None = None
    if message is not None:
        try:
            address, args = message
            fixed = encode_message(address, list(args))
        except (OscError, TypeError, ValueError) as e:
            return report(f"cannot encode the message: {e}")
    try:
        family, dest = resolve(host, port)
    except (OSError, UnicodeError, ValueError) as e:
        return report(f"cannot resolve {host}:{port}: {e}")

    sock = socket.socket(family, socket.SOCK_DGRAM)
    try:
        try:
            if family == socket.AF_INET:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)    # x.x.x.255 receivers
            sock.bind(("::" if family == socket.AF_INET6 else "0.0.0.0", int(listen_port or 0)))
        except (OSError, OverflowError) as e:
            return report(f"cannot listen on UDP port {listen_port or 0}: {e}")
        next_send = time.monotonic()
        for seq in range(1, count + 1):
            wait = next_send - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _drain(sock)
            t0 = clock()
            packet = fixed if fixed is not None else encode_message(PING_ADDRESS, [("i", seq), ("d", float(t0))])
            try:
                sock.sendto(packet, dest)
            except OSError as e:
                return report(f"send to {host}:{port} failed: {e}")
            sent_at = time.monotonic()
            sent += 1
            next_send = sent_at + interval_s
            rtt = _await_reply(sock, clock, t0, sent_at + timeout_s, None if generic else seq)
            if rtt is not None:
                rtts.append(rtt)
    finally:
        sock.close()
    return report()


def _send_reply(sock: socket.socket, data: bytes, dest: Any) -> bool:
    try:
        sock.sendto(data, dest)
        return True
    except OSError as e:
        log.warning("echo reply to %s failed: %s", dest, e)
        return False


def serve_echo(port: int = DEFAULT_ECHO_PORT, host: str = "0.0.0.0", *, echo_all: bool = False,
               stop: threading.Event | None = None,
               on_message: Callable[[Any, str, list[OscArg], bool], None] | None = None,
               on_ready: Callable[[int], None] | None = None, poll_s: float = 0.1) -> int:
    """Answer `/oscsender/ping` with `/oscsender/pong` + the same arguments, sent back to the ping's
    source address and port (`echo_all`: mirror every OSC packet back unchanged instead). Runs until
    `stop` is set (checked every `poll_s`; None = until KeyboardInterrupt) and returns the number of
    replies sent. `on_ready(bound_port)` is called once listening (port 0 picks a free one);
    `on_message(source, address, args, replied)` for every message received. Raises OSError when it
    cannot listen (e.g. the port is in use), ValueError for a port outside 0..65535."""
    if not 0 <= int(port) <= 65535:
        raise ValueError(f"port {port} out of range 0..65535")
    infos = socket.getaddrinfo(host or None, int(port), type=socket.SOCK_DGRAM, flags=socket.AI_PASSIVE)
    infos.sort(key=lambda i: i[0] != socket.AF_INET)
    family, _, _, _, addr = infos[0]
    sock = socket.socket(family, socket.SOCK_DGRAM)
    replies = 0

    def notify(src: Any, address: str, args: list[OscArg], replied: bool) -> None:
        if on_message is None:
            return
        try:
            on_message(src, address, args, replied)
        except Exception:
            log.exception("echo on_message callback failed")

    try:
        sock.bind(addr)
        sock.settimeout(poll_s)
        if on_ready is not None:
            on_ready(int(sock.getsockname()[1]))
        while stop is None or not stop.is_set():
            try:
                data, src = sock.recvfrom(MAX_DATAGRAM)
            except TimeoutError:
                continue
            except OSError as e:     # an ICMP error reported on the socket; keep serving
                log.debug("echo receive: %s", e)
                time.sleep(min(poll_s, 0.01))
                continue
            try:
                messages = decode_packet(data)
            except (ValueError, RecursionError, struct.error):
                log.debug("echo: ignored a packet that is not OSC from %s", src)
                continue
            if echo_all:
                ok = _send_reply(sock, data, src)
                replies += ok
                for address, args in messages:
                    notify(src, address, args, ok)
                continue
            for address, args in messages:
                ok = False
                if address == PING_ADDRESS:
                    try:
                        ok = _send_reply(sock, encode_message(PONG_ADDRESS, args), src)
                    except OscError as e:
                        log.warning("echo: cannot answer a ping from %s: %s", src, e)
                    replies += ok
                notify(src, address, args, ok)
        return replies
    finally:
        sock.close()


# ----------------------------------------------------------------------------------------- audio capture


def _sounddevice() -> Any:
    """sounddevice, imported lazily through live.audio_input (tests substitute a fake module there)."""
    from .live import audio_input

    return audio_input._sounddevice()


class AudioCapture:
    """A sound-card input kept in a ring buffer with the clock time of every sample.

    `channels` are 1-based device channels, downmixed by averaging (default: channel 1). The stream
    callback stamps each block with the clock time of its first sample's ADC capture:
    `clock_at_callback - (time_info.currentTime - time_info.inputBufferAdcTime)` when PortAudio provides
    both (non-zero, and an age below 1 s), else `clock_at_callback - stream.latency - frames / sr`.
    `stream_factory(**kwargs)` replaces `sounddevice.InputStream` (tests); it gets `device`,
    `channels` (the highest selected channel), `samplerate` (None = the stream's default), `blocksize`,
    `dtype`, `latency` and `callback`, and must return an object with `start()`, `stop()`, `close()`
    and optionally `samplerate` / `latency`.
    """

    def __init__(self, device: int | str | None = None, channels: list[int] | None = None,
                 samplerate: float | None = None, *, clock: Callable[[], float] = time.perf_counter,
                 stream_factory: Callable[..., Any] | None = None, buffer_s: float = 15.0, blocksize: int = 0,
                 latency: str | float = "low"):
        chans = sorted({int(c) for c in (channels or [1])})
        if not chans or chans[0] < 1:
            raise ValueError(f"channels are 1-based: {channels}")
        if buffer_s <= 0:
            raise ValueError("buffer_s must be > 0")
        self.device, self.channels = device, chans
        self.requested_samplerate = samplerate
        self.clock, self.stream_factory = clock, stream_factory
        self.buffer_s, self.blocksize, self.latency = float(buffer_s), int(blocksize), latency
        self.samplerate = 0.0
        self.stream_latency_s = 0.0          # PortAudio's reported input latency
        self.overflows = 0                   # PortAudio input overflows
        self.adc_blocks = 0                  # blocks stamped from PortAudio's ADC time
        self.estimated_blocks = 0            # blocks stamped from the reported stream latency
        self._lock = threading.Lock()
        self._ring = np.zeros(0, np.float32)
        self._blocks: deque[tuple[int, int, float]] = deque()   # (first sample index, frames, first sample time)
        self._total = 0                      # samples written since start
        self._pick: list[int] = []
        self._stream: Any = None

    # -- stream -----------------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            raise RuntimeError("AudioCapture already started")
        kw: dict[str, Any] = {"device": self.device, "channels": max(self.channels),
                              "samplerate": self.requested_samplerate, "blocksize": self.blocksize,
                              "dtype": "float32", "latency": self.latency, "callback": self._callback}
        factory = self.stream_factory
        if factory is None:
            sd = _sounddevice()
            info = sd.query_devices(self.device, "input")
            n_in = int(info["max_input_channels"])
            if max(self.channels) > n_in:
                raise ValueError(f"channels {self.channels} out of range 1..{n_in} for {info['name']!r}")
            kw["device"] = int(info.get("index", self.device if isinstance(self.device, int) else 0))
            kw["samplerate"] = float(self.requested_samplerate or info["default_samplerate"])
            factory = sd.InputStream
        self._pick = [c - 1 for c in self.channels]
        # The callback may run as soon as the stream starts: allocate the ring buffer before that, for the
        # rate the stream actually runs at.
        stream = factory(**kw)
        sr = float(getattr(stream, "samplerate", None) or kw["samplerate"] or 0.0)
        if sr <= 0:
            stream.close()
            raise ValueError("the input stream did not report a sample rate")
        lat = getattr(stream, "latency", 0.0)
        if isinstance(lat, (tuple, list)):
            lat = lat[0]
        with self._lock:
            self.samplerate = sr
            self.stream_latency_s = float(lat or 0.0)
            self._ring = np.zeros(max(1, round(self.buffer_s * sr)), np.float32)
            self._blocks.clear()
            self._total = 0
        self.overflows = self.adc_blocks = self.estimated_blocks = 0
        self._stream = stream
        try:
            stream.start()
        except Exception:
            self._stream = None
            stream.close()
            raise

    def stop(self) -> None:
        """Stop and close the stream (the captured audio stays readable)."""
        stream, self._stream = self._stream, None
        if stream is None:
            return
        for step in (stream.stop, stream.close):
            try:
                step()
            except Exception as e:  # noqa: BLE001  (a device lost mid-measurement: the stream is gone anyway)
                log.warning("audio capture: %s failed: %s: %s", step.__name__, type(e).__name__, e)

    close = stop

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- PortAudio side ---------------------------------------------------------------------------
    def _adc_time(self, now: float, frames: int, time_info: Any) -> float:
        cur = float(getattr(time_info, "currentTime", 0.0) or 0.0)
        adc = float(getattr(time_info, "inputBufferAdcTime", 0.0) or 0.0)
        if cur and adc and 0.0 <= cur - adc < MAX_ADC_AGE_S:
            self.adc_blocks += 1
            return now - (cur - adc)
        self.estimated_blocks += 1
        return now - self.stream_latency_s - frames / self.samplerate

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        now = self.clock()
        if getattr(status, "input_overflow", False):
            self.overflows += 1
        x = np.asarray(indata)
        if x.ndim == 1:
            x = x[:, None]
        cols = [c for c in self._pick if c < x.shape[1]] or [0]
        mono = x[:, cols[0]].astype(np.float32) if len(cols) == 1 else x[:, cols].mean(axis=1, dtype=np.float32)
        if mono.size:
            self._write(mono, self._adc_time(now, len(mono), time_info))

    def _write(self, mono: np.ndarray, t_first: float) -> None:
        with self._lock:
            ring, sr = self._ring, self.samplerate
            cap, n, start = ring.size, mono.size, self._total
            if n > cap:                                   # keep the newest `cap` samples of a huge block
                skip = n - cap
                mono, n, start, t_first = mono[skip:], cap, start + skip, t_first + skip / sr
            pos = start % cap
            first = min(n, cap - pos)
            ring[pos:pos + first] = mono[:first]
            ring[:n - first] = mono[first:]
            self._blocks.append((start, n, t_first))
            self._total = start + n
            oldest = self._total - cap
            while self._blocks and self._blocks[0][0] + self._blocks[0][1] <= oldest:
                self._blocks.popleft()

    # -- reading ----------------------------------------------------------------------------------
    def latest_time(self) -> float | None:
        """Clock time just after the newest captured sample (None before the first block)."""
        with self._lock:
            if not self._blocks:
                return None
            _start, n, tf = self._blocks[-1]
            return tf + n / self.samplerate

    def earliest_time(self) -> float | None:
        """Clock time of the oldest sample still in the ring buffer."""
        with self._lock:
            if not self._blocks:
                return None
            start, _n, tf = self._blocks[0]
            return tf + (max(start, self._total - self._ring.size) - start) / self.samplerate

    def segment(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        """(times, samples) of the buffered samples captured in [t0, t1): float64 clock times and the
        float32 mono samples, in capture order."""
        empty = (np.zeros(0, np.float64), np.zeros(0, np.float32))
        if not t1 > t0:
            return empty
        with self._lock:
            sr, ring, cap = self.samplerate, self._ring, self._ring.size
            oldest = self._total - cap
            picked: list[tuple[int, int, float]] = []
            newest_found = False
            for blk in reversed(self._blocks):
                _start, n, tf = blk
                if not newest_found and tf < t1:
                    newest_found = True
                if newest_found:
                    picked.append(blk)
                    if tf + n / sr <= t0 - 0.1:           # (slack for jittery estimated block times)
                        break
            if not picked:
                return empty
            picked.reverse()
            a = max(picked[0][0], oldest)
            b = picked[-1][0] + picked[-1][1]
            idx = np.arange(a, b) % cap
            data = ring[idx]
        times = np.concatenate([tf + np.arange(max(start, a) - start, n) / sr for start, n, tf in picked])
        keep = (times >= t0) & (times < t1)
        return times[keep], data[keep]

    def stats(self) -> dict:
        if self.adc_blocks and self.estimated_blocks:
            stamps = "mixed"
        elif self.adc_blocks or self.estimated_blocks:
            stamps = "adc" if self.adc_blocks else "estimated"
        else:
            stamps = None
        return {"device": self.device, "channels": self.channels, "samplerate": self.samplerate,
                "running": self.running, "overflows": self.overflows, "timestamps": stamps,
                "input_latency_s": self.stream_latency_s, "buffered_s": round(self._buffered_s(), 3)}

    def _buffered_s(self) -> float:
        with self._lock:
            return min(self._total, self._ring.size) / self.samplerate if self.samplerate else 0.0


# ----------------------------------------------------------------------------------------- loopback


def _envelope(times: np.ndarray, x: np.ndarray, t0: float, n: int,
              frame_s: float = FRAME_S) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RMS envelope in dBFS of the samples in n frames from t0: (env, frame has samples, frame index of
    every sample (-1 outside))."""
    k = np.floor((np.asarray(times, np.float64) - t0) / frame_s).astype(np.int64)
    inside = (k >= 0) & (k < n)
    k = np.where(inside, k, -1)
    v = np.asarray(x, np.float64)[inside]
    ss = np.bincount(k[inside], weights=v * v, minlength=n)[:n]
    cnt = np.bincount(k[inside], minlength=n)[:n]
    has = cnt > 0
    env = np.full(n, np.nan)
    env[has] = 10.0 * np.log10(ss[has] / cnt[has] + 1e-20)
    return env, has, k


def detect_onset(times: np.ndarray, samples: np.ndarray, t_send: float, *, window_s: float = 1.0,
                 threshold_db: float = 20.0, pre_s: float = 0.3,
                 frame_s: float = FRAME_S) -> tuple[float | None, float | None]:
    """(onset clock time | None, floor dBFS | None) of a sound after `t_send` (see the module docstring).
    The floor is None when less than a quarter of the `pre_s` before the send was captured."""
    n_pre, n_win = max(1, round(pre_s / frame_s)), max(1, round(window_s / frame_s))
    env, has, k = _envelope(times, samples, t_send - n_pre * frame_s, n_pre + n_win, frame_s)
    pre = env[:n_pre][has[:n_pre]]
    if pre.size < max(3, n_pre // 4):
        return None, None
    floor = float(np.median(pre))
    thr = max(floor + threshold_db, MIN_THRESHOLD_DB)
    hits = np.flatnonzero(has[n_pre:] & (np.nan_to_num(env[n_pre:], nan=-np.inf) > thr))
    if not hits.size:
        return None, floor
    j = n_pre + int(hits[0])
    # the first sample reaching the threshold, in this frame or (a sound starting late in it) the one before
    loud = np.flatnonzero((k >= max(j - 1, n_pre)) & (k <= j) & (np.abs(samples) >= 10.0 ** (thr / 20.0)))
    onset = float(times[loud[0]]) if loud.size else t_send + (j - n_pre) * frame_s
    return onset, floor


def _recovered(capture: AudioCapture, level_db: float, pre_s: float, onset: float) -> bool:
    """Whether the input is back below `level_db` after the sound that started at `onset`: 90 % of the
    last RECENT_S (all of it after the onset) and the median of the last `pre_s`."""
    latest = capture.latest_time()
    if latest is None or latest - RECENT_S < onset:
        return False
    n_recent = round(RECENT_S / FRAME_S)
    n = max(n_recent, round(pre_s / FRAME_S))
    t0 = latest - n * FRAME_S
    times, x = capture.segment(t0, latest + 1.0)
    env, has, _ = _envelope(times, x, t0, n)
    recent = env[n - n_recent:][has[n - n_recent:]]
    return (recent.size > 0 and float(np.percentile(recent, 90)) <= level_db
            and float(np.median(env[has])) <= level_db)


def _loopback_note(sent: int, received: int, floors: list[float], threshold_db: float, window_s: float,
                   stamps: str | None) -> str:
    misses = sent - received
    if not sent:
        parts = []
    elif not received:
        parts = [(f"no sound detected in {sent} trial{'s' if sent != 1 else ''}: check that the test message "
                  "makes the receiver play a sound, that the selected input device / channels hear it, and the "
                  "threshold and window")]
    else:
        parts = [f"{received}/{sent} sounds detected"]
        if misses:
            parts.append(f"{misses} without a sound above the threshold within {window_s * 1000:.0f} ms")
        if received:
            parts.append("estimate = median delay from the send to the sound at the input (network, receiver "
                         "and its audio output)")
    if floors:
        f = float(np.median(floors))
        parts.append(f"input floor {f:.1f} dBFS, threshold {max(f + threshold_db, MIN_THRESHOLD_DB):.1f} dBFS")
    if stamps in ("estimated", "mixed"):
        parts.append("input timestamps estimated from the reported stream latency (no ADC times from the audio "
                     "driver): expect a few ms of error")
    return "; ".join(parts)


def measure_loopback(send: Callable[[], float], capture: AudioCapture, *, count: int = 8,
                     interval_s: float = 1.5, window_s: float = 1.0, threshold_db: float = 20.0,
                     pre_s: float = 0.3, clock: Callable[[], float] = time.perf_counter,
                     sleep: Callable[[float], None] = time.sleep, target: str | None = None,
                     poll_s: float = 0.01, stall_s: float = 1.0) -> LatencyReport:
    """Delay from `send()` to the receiver's sound at `capture`'s input (see the module docstring).

    `send()` sends the test message and returns its clock time (the clock `capture` stamps with; it may
    raise to abort the measurement). A capture that is not running is started here (and stopped at the
    end). `sleep` waits between polls of the capture (every `poll_s`); with a fake clock it must advance
    that clock. An input that delivers no audio for `stall_s` past the moment it should have ends the
    measurement with an error. Never raises for device or send problems: they come back as `error`."""
    if count < 1:
        raise ValueError("count must be >= 1")
    if window_s <= 0 or pre_s <= 0 or threshold_db <= 0 or interval_s < 0 or poll_s <= 0:
        raise ValueError("window_s, pre_s, threshold_db and poll_s must be > 0, interval_s >= 0")
    lat_ms: list[float] = []
    floors: list[float] = []
    sent = 0
    error: str | None = None
    margin = min(RECOVER_DB, threshold_db / 2.0)
    started = False
    if not capture.running:
        try:
            capture.start()
            started = True
        except Exception as e:  # noqa: BLE001  (PortAudio / device errors: reported, not raised)
            return LatencyReport("loopback", target=target, error=f"cannot open the audio input: {e}")
    try:
        deadline = clock() + pre_s + stall_s
        while True:                       # enough input history for the first floor
            first, last = capture.earliest_time(), capture.latest_time()
            if first is not None and last is not None and last - first >= pre_s:
                break
            if clock() > deadline:
                error = "the audio input delivers no audio (check the device and its channels)"
                break
            sleep(poll_s)
        for i in range(count if error is None else 0):
            try:
                t_send = float(send())
            except Exception as e:  # noqa: BLE001  (the caller's send: reported, not raised)
                error = f"sending the test message failed: {e}"
                break
            sent += 1
            end, onset, floor = t_send + window_s, None, None
            while True:
                latest = capture.latest_time()
                if latest is not None and latest > t_send:
                    times, x = capture.segment(t_send - pre_s - FRAME_S, end + FRAME_S)
                    onset, floor = detect_onset(times, x, t_send, window_s=window_s, threshold_db=threshold_db,
                                                pre_s=pre_s)
                    if onset is not None or latest >= end:
                        break
                if clock() > end + stall_s:
                    error = "the audio input stopped delivering audio during the measurement"
                    break
                sleep(poll_s)
            if error is not None:
                break
            if floor is not None:
                floors.append(floor)
            if onset is not None:
                lat_ms.append((onset - t_send) * 1000.0)
            if i == count - 1:
                break
            until = t_send + interval_s
            if onset is None or floor is None:
                while clock() < until:
                    sleep(poll_s)
                continue
            level = max(floor + margin, MIN_THRESHOLD_DB - margin)
            while clock() < until and not _recovered(capture, level, pre_s, onset):
                sleep(poll_s)
    finally:
        if started:
            capture.stop()
    stamps = capture.stats().get("timestamps")
    return LatencyReport.from_samples("loopback", lat_ms, target=target, sent=sent, received=len(lat_ms),
                                      note=_loopback_note(sent, len(lat_ms), floors, threshold_db, window_s, stamps),
                                      error=error)


def target_send(sender: OscSender, target: str, address: str,
                args: Sequence[OscArg] = ()) -> Callable[[], float]:
    """A `send` for measure_loopback: one message to a named target of `sender`; returns the send time on
    the sender's clock (use the same clock for the AudioCapture). Raises RuntimeError when it fails."""
    def send() -> float:
        res = sender.send(target, address, args)
        if not res.ok:
            raise RuntimeError(res.error or f"cannot send to {target!r}")
        return res.t
    return send


# ----------------------------------------------------------------------------------------- cli


def message_from_text(text: str) -> tuple[str, tuple[OscArg, ...]]:
    """`"/address arg ..."` in cue-file syntax -> (address, args); raises ValueError. A leading target
    name is ignored. Without the cue parser (cues.parse_message), only the address is used."""
    try:
        from .cues import parse_message
        _target, address, args, _note = parse_message(text)
        return address, tuple(args)
    except (ImportError, NotImplementedError):
        tokens = text.split()
        if not tokens:
            raise ValueError("empty message") from None
        if len(tokens) > 1:
            log.warning("the cue-file parser is not available: arguments of %r are ignored", text)
        return validate_address(tokens[0]), ()


def _fmt_arg(tag: str, value: Any) -> str:
    if tag in "TFNI":
        return {"T": "true", "F": "false", "N": "nil", "I": "impulse"}[tag]
    if tag == "s":
        return json.dumps(value, ensure_ascii=False)
    if tag == "b":
        return f"<{len(value)} bytes>"
    if tag == "f":
        text = f"{value:.7g}"
        return text if any(c in text for c in ".eni") else text + ".0"
    return f"{tag}:{value}" if tag in "hd" else str(value)


def _clock_text() -> str:
    t = time.time()
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t % 1 * 1000):03d}"


def cli_echo(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="oscsender echo",
        description="Answer OSC pings so the sending machine can measure the network round trip "
                    "(`oscsender latency`, or Targets > Measure in the console). Run it on the receiving machine.")
    ap.add_argument("--host", default="0.0.0.0", help="address to listen on (default: every interface)")
    ap.add_argument("--port", type=int, default=DEFAULT_ECHO_PORT, help="UDP port (default %(default)s)")
    ap.add_argument("--echo-all", action="store_true",
                    help=f"mirror every OSC message back unchanged, not only {PING_ADDRESS}")
    ap.add_argument("--quiet", action="store_true", help="do not print the messages it answers")
    args = ap.parse_args(argv)
    answered = 0
    hinted = False

    def on_ready(port: int) -> None:
        what = "every OSC message" if args.echo_all else PING_ADDRESS
        print(f"answering {what} on UDP {args.host}:{port} (Ctrl+C to stop)", flush=True)
        print(f"measure from the sending machine: oscsender latency --host <this machine's IP> --port {port}",
              flush=True)

    def on_message(src: Any, address: str, osc_args: list[OscArg], replied: bool) -> None:
        nonlocal answered, hinted
        answered += replied
        if args.quiet:
            return
        if address == PING_ADDRESS and replied and osc_args and osc_args[0][0] == "i":
            print(f"{_clock_text()}  ping #{osc_args[0][1]} from {src[0]}:{src[1]}", flush=True)
        elif replied:
            text = " ".join([address] + [_fmt_arg(t, v) for t, v in osc_args])
            print(f"{_clock_text()}  {text}  from {src[0]}:{src[1]}", flush=True)
        elif not hinted:
            hinted = True
            print(f"{_clock_text()}  ignored {address} from {src[0]}:{src[1]} (not a ping; --echo-all mirrors "
                  "every message)", flush=True)

    try:
        serve_echo(args.port, args.host, echo_all=args.echo_all, on_message=on_message, on_ready=on_ready)
    except KeyboardInterrupt:
        print(f"\nstopped ({answered} answered)", flush=True)
        return 0
    except (OSError, ValueError) as e:
        print(f"error: cannot listen on UDP {args.host}:{args.port}: {e}", file=sys.stderr)
        return 1
    return 0


def report_text(rep: LatencyReport) -> str:
    """Human-readable lines for a report (CLI output)."""
    where = " ".join(w for w in (rep.target, f"{rep.host}:{rep.port}" if rep.host else None) if w)
    lines = [f"{rep.method}{' ' + where if where else ''}: {rep.sent} sent, {rep.received} received"]
    if rep.median_ms is not None:
        lines.append(f"  min {rep.min_ms:.2f} ms  median {rep.median_ms:.2f} ms  p95 {rep.p95_ms:.2f} ms  "
                     f"max {rep.max_ms:.2f} ms  jitter (IQR) {rep.jitter_ms:.2f} ms")
    if rep.estimate_ms is not None:
        lines.append(f"  suggested target latency: {rep.estimate_ms:.1f} ms")
    if rep.note:
        lines.append(f"  {rep.note}")
    if rep.error:
        lines.append(f"error: {rep.error}")
    return "\n".join(lines)


def cli_latency(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="oscsender latency",
        description="Measure the OSC round trip to a receiver: pings answered by `oscsender echo` on that machine, "
                    "or (--message) any message the receiver replies to. The suggested target latency is half the "
                    "median round trip. Exit status 1 when nothing came back.")
    ap.add_argument("--host", required=True, help="receiver host name or IP")
    ap.add_argument("--port", type=int, default=DEFAULT_ECHO_PORT, help="receiver UDP port (default %(default)s)")
    ap.add_argument("--count", type=int, default=20, help="number of pings (default %(default)s)")
    ap.add_argument("--interval-ms", type=float, default=50.0, help="time between pings (default %(default)s)")
    ap.add_argument("--timeout-ms", type=float, default=500.0,
                    help="a ping without a reply after this long is lost (default %(default)s)")
    ap.add_argument("--message", default=None,
                    help='send this message (cue-file syntax, e.g. "/version" or "/cue/go 3") instead of '
                         f"{PING_ADDRESS} and time the first reply of any address")
    ap.add_argument("--listen-port", type=int, default=None,
                    help="receive replies on this UDP port (for receivers that reply to a fixed port)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)
    if args.count < 1:
        ap.error("--count must be >= 1")
    if args.timeout_ms <= 0 or args.interval_ms < 0:
        ap.error("--timeout-ms must be > 0 and --interval-ms >= 0")
    if args.listen_port is not None and not 0 <= args.listen_port <= 65535:
        ap.error("--listen-port must be 0..65535")
    message = None
    if args.message is not None:
        try:
            message = message_from_text(args.message)
        except ValueError as e:
            print(f"error: --message: {e}", file=sys.stderr)
            return 2
    if not args.json:
        what = f"{args.count} x {message[0] if message else 'ping'}"
        print(f"measuring the OSC round trip to {args.host}:{args.port} ({what}) ...", flush=True)
    rep = measure_echo(args.host, args.port, count=args.count, interval_s=args.interval_ms / 1000.0,
                       timeout_s=args.timeout_ms / 1000.0, message=message, listen_port=args.listen_port)
    if args.json:
        print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=1, allow_nan=False))
    else:
        print(report_text(rep))
    return 0 if rep.error is None and rep.received > 0 else 1
