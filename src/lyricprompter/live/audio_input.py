"""Live audio sources: a sound-card input (sounddevice) and a file played as if it were live.

Both deliver 16 kHz mono float32 blocks to `on_audio` from their own worker thread. The PortAudio
callback of `DeviceInput` only copies into a bounded queue; downmix, level metering and resampling
(soxr.ResampleStream) happen on the worker so the audio thread can never be stalled by DSP.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import soxr

from ..config import ANALYSIS_SR

log = logging.getLogger(__name__)

AudioCallback = Callable[[np.ndarray], None]
SILENCE_DB = -120.0
STALL_S = 1.0       # a running input stream that delivers nothing for this long has stopped
STALLED_TEXT = ("the input device stopped delivering audio (unplugged, switched off or taken over by another "
                "app?); check it, then start the session again")


def _sounddevice() -> Any:
    """Import sounddevice lazily (PortAudio init) so tests can substitute a fake module."""
    import sounddevice

    return sounddevice


def _db(power: float) -> float:
    return 10.0 * math.log10(power) if power > 1e-12 else SILENCE_DB


class LevelMeter:
    """RMS / peak meter in dBFS with fast attack and exponential release (thread-safe enough:
    written by one worker, read as plain floats)."""

    def __init__(self, release_s: float = 0.3, peak_hold_s: float = 1.0):
        self.release_s, self.peak_hold_s = release_s, peak_hold_s
        self._ms = 0.0
        self._peak = 0.0
        self._peak_t = 0.0
        self._last_sound = time.monotonic()
        self.samples = 0

    def update(self, mono: np.ndarray, sr: float) -> None:
        if mono.size == 0:
            return
        now = time.monotonic()
        ms = float(np.mean(np.square(mono, dtype=np.float64)))
        peak = float(np.max(np.abs(mono)))
        if not (math.isfinite(ms) and math.isfinite(peak)):
            # A driver glitch (NaN / inf sample): meter the finite samples only. An inf would
            # otherwise stick in the release (inf * decay == inf) for the rest of the session.
            clean = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
            ms = float(np.mean(np.square(clean, dtype=np.float64)))
            peak = float(np.max(np.abs(clean)))
        if peak > 0.0:
            self._last_sound = now               # any signal at all (DC too): not a muted input
        # meter the AC part: a DC offset (bad cable / interface bias) is not loudness
        mean = float(np.mean(np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64))
        if mean:
            ms = max(0.0, ms - mean * mean)
            peak = float(np.max(np.abs(np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0) - mean)))
        decay = math.exp(-(mono.size / sr) / self.release_s)
        self._ms = max(ms, self._ms * decay)
        if peak >= self._peak or now - self._peak_t > self.peak_hold_s:
            self._peak, self._peak_t = peak, now
        self.samples += mono.size

    @property
    def level_db(self) -> float:
        return round(_db(self._ms), 1)

    @property
    def peak_db(self) -> float:
        return round(_db(self._peak * self._peak), 1)

    @property
    def silent_s(self) -> float:
        """Seconds since the last non-zero sample (exact digital silence usually means the
        microphone permission was denied, or the wrong channel is selected)."""
        return 0.0 if self.samples == 0 else time.monotonic() - self._last_sound


class _Source:
    """Shared worker-thread bookkeeping for the audio sources."""

    def __init__(self, quality: str):
        self.quality = quality
        self.meter = LevelMeter()
        self.overflows = 0          # blocks dropped by our bounded queue (worker too slow)
        self.xruns = 0              # PortAudio input overflows / output underflows
        self.errors = 0             # exceptions raised by on_audio
        self.last_error: str | None = None
        self.delivered = 0          # 16 kHz samples delivered to on_audio
        self._on_audio: AudioCallback | None = None
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def level_db(self) -> float:
        return self.meter.level_db

    @property
    def peak_db(self) -> float:
        return self.meter.peak_db

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def finished(self) -> bool:
        """True once the worker has exited (end of file, stop() or a fatal error)."""
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the worker has exited; returns False on timeout."""
        return self._done.wait(timeout)

    def _deliver(self, y: np.ndarray) -> None:
        if y.size == 0 or self._on_audio is None:
            return
        self.delivered += y.size
        try:
            self._on_audio(y)
        except Exception as e:  # keep the audio path alive; surface the error in stats()
            self.errors += 1
            if self.errors == 1 or self.errors % 100 == 0:
                log.exception("on_audio failed (%d errors so far)", self.errors)
            self.last_error = f"{type(e).__name__}: {e}"

    def _spawn(self, target: Callable[[], None], name: str) -> None:
        if self.running:
            raise RuntimeError(f"{type(self).__name__} already started")
        self._stop.clear()
        self._done.clear()
        self._thread = threading.Thread(target=self._guarded(target), name=name, daemon=True)
        self._thread.start()

    def _guarded(self, target: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                target()
            except Exception as e:
                log.exception("%s worker crashed", type(self).__name__)
                self.last_error = f"{type(e).__name__}: {e}"
            finally:
                self._done.set()
        return run

    def _join(self, timeout: float = 2.0) -> None:
        self._stop.set()
        th = self._thread
        if th is not None and th is not threading.current_thread():
            th.join(timeout)

    @property
    def stalled_s(self) -> float:
        """Seconds the running input has delivered nothing (0 for sources that cannot stall)."""
        return 0.0

    def stats(self) -> dict:
        stalled = self.stalled_s
        error = STALLED_TEXT if stalled > STALL_S else self.last_error
        return {"level_db": self.level_db, "peak_db": self.peak_db, "silent_s": round(self.meter.silent_s, 2),
                "stalled_s": round(stalled, 2), "overflows": self.overflows, "xruns": self.xruns,
                "errors": self.errors, "last_error": error, "position_s": round(self.position_s, 3),
                "running": self.running, "finished": self.finished}

    @property
    def position_s(self) -> float:
        return self.delivered / ANALYSIS_SR


# --------------------------------------------------------------------------------------------- devices


def list_input_devices(refresh: bool = False) -> list[dict]:
    """Input-capable devices: {index, name, channels, default_samplerate, hostapi, is_default}.

    `refresh=True` re-initialises PortAudio so hot-plugged interfaces show up; only do that while
    no stream is open.
    """
    sd = _sounddevice()
    if refresh:
        sd._terminate()
        sd._initialize()
    apis = sd.query_hostapis()
    try:
        default_in = int(sd.default.device[0])
    except (AttributeError, IndexError, TypeError, ValueError):
        default_in = -1
    if default_in < 0 and apis:
        default_in = int(apis[0].get("default_input_device", -1))
    out = []
    for i, d in enumerate(sd.query_devices()):
        if int(d.get("max_input_channels", 0)) <= 0:
            continue
        idx = int(d.get("index", i))
        out.append({"index": idx, "name": d["name"], "channels": int(d["max_input_channels"]),
                    "default_samplerate": float(d["default_samplerate"]),
                    "hostapi": apis[d["hostapi"]]["name"] if d.get("hostapi") is not None else "",
                    "is_default": idx == default_in})
    return out


class DeviceInput(_Source):
    """Capture selected channels of a sound-card input and deliver 16 kHz mono float32.

    `channels` are 1-based device channels, downmixed by averaging (default: channel 1, or 1+2 on
    a stereo device). The stream runs at the device's native rate unless `samplerate` is given.
    """

    def __init__(self, device: int | str | None = None, channels: list[int] | None = None,
                 samplerate: float | None = None, blocksize: int = 0, *, queue_blocks: int = 256,
                 quality: str = "MQ", latency: str | float = "low"):
        super().__init__(quality)
        sd = _sounddevice()
        info = sd.query_devices(device, "input")
        self.device_index = int(info["index"])
        self.device_name = str(info["name"])
        self.max_channels = int(info["max_input_channels"])
        if self.max_channels <= 0:
            raise ValueError(f"device {self.device_name!r} has no input channels")
        if channels is None:
            channels = [1, 2] if self.max_channels == 2 else [1]
        chans = sorted({int(c) for c in channels})
        bad = [c for c in chans if not 1 <= c <= self.max_channels]
        if not chans or bad:
            raise ValueError(f"channels {channels} out of range 1..{self.max_channels} "
                             f"for {self.device_name!r}")
        self.channels = chans
        self.samplerate = float(samplerate or info["default_samplerate"])
        self.blocksize, self.latency = int(blocksize), latency
        try:
            self.hostapi = str(sd.query_hostapis()[int(info["hostapi"])]["name"])
        except (KeyError, IndexError, TypeError, ValueError):
            self.hostapi = ""
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=max(2, int(queue_blocks)))
        self._pick: list[int] | None = None     # column indices to downmix (None: all columns)
        self._stream: Any = None
        self._rs: soxr.ResampleStream | None = None
        self._last_cb: float | None = None       # monotonic time of the last callback (None: no stream)

    # -- PortAudio side (realtime thread): copy + enqueue only -------------------------------
    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        self._last_cb = time.monotonic()
        if getattr(status, "input_overflow", False):
            self.xruns += 1
        self._enqueue(indata.copy())

    @property
    def stalled_s(self) -> float:
        """Seconds since the open stream's last callback. A device that is unplugged or fails
        mid-session simply stops calling back (the stream may still report itself active), and the
        level meter alone would read that as digital silence, i.e. a microphone permission problem."""
        last = self._last_cb
        if last is None or self._stream is None or not self.running:
            return 0.0
        return max(0.0, time.monotonic() - last)

    def _enqueue(self, block: np.ndarray) -> None:
        try:
            self._q.put_nowait(block)
        except queue.Full:
            try:
                self._q.get_nowait()        # drop the oldest block
            except queue.Empty:
                pass
            self.overflows += 1
            try:
                self._q.put_nowait(block)
            except queue.Full:
                pass

    # -- worker side ---------------------------------------------------------------------------
    def _downmix(self, block: np.ndarray) -> np.ndarray:
        x = block if self._pick is None else block[:, self._pick]
        return x[:, 0].astype(np.float32, copy=True) if x.shape[1] == 1 else x.mean(axis=1, dtype=np.float32)

    def _process(self, block: np.ndarray) -> None:
        mono = self._downmix(block)
        self.meter.update(mono, self.samplerate)
        assert self._rs is not None
        self._deliver(self._rs.resample_chunk(mono))

    def _work(self) -> None:
        while not self._stop.is_set():
            try:
                block = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            self._process(block)

    def _open_stream(self) -> Any:
        sd = _sounddevice()
        kw = {"device": self.device_index, "samplerate": self.samplerate, "blocksize": self.blocksize,
              "dtype": "float32", "latency": self.latency, "callback": self._callback}
        if self.hostapi == "Core Audio" and hasattr(sd, "CoreAudioSettings"):
            try:
                cmap = [c - 1 for c in self.channels]
                stream = sd.InputStream(channels=len(cmap),
                                        extra_settings=sd.CoreAudioSettings(channel_map=cmap), **kw)
                self._pick = None
                return stream
            except (getattr(sd, "PortAudioError", OSError), ValueError, TypeError) as e:
                log.warning("channel_map stream failed on %r (%s); opening all channels", self.device_name, e)
        stream = sd.InputStream(channels=max(self.channels), **kw)
        self._pick = [c - 1 for c in self.channels]
        return stream

    def start(self, on_audio: AudioCallback) -> None:
        if self.running:
            raise RuntimeError("DeviceInput already started")
        self._on_audio = on_audio
        self._rs = soxr.ResampleStream(self.samplerate, ANALYSIS_SR, 1, dtype="float32", quality=self.quality)
        while not self._q.empty():
            self._q.get_nowait()
        self._stream = self._open_stream()
        self._spawn(self._work, f"DeviceInput[{self.device_name}]")
        try:
            self._stream.start()
        except Exception:
            self._join()
            self._close_stream()
            raise
        self._last_cb = time.monotonic()        # (some devices take a while to start)

    def _close_stream(self) -> None:
        """Stop and close the stream. A device that was lost mid-session can make PortAudio fail
        both: the stream is gone either way, so that is logged and never turns a stop into an error."""
        stream, self._stream, self._last_cb = self._stream, None, None
        if stream is None:
            return
        lost = (getattr(_sounddevice(), "PortAudioError", OSError), OSError, RuntimeError)
        for step in (stream.stop, stream.close):
            try:
                step()
            except lost as e:
                log.warning("input stream of %r: %s failed: %s: %s", self.device_name, step.__name__,
                            type(e).__name__, e)

    def stop(self) -> None:
        try:
            self._close_stream()
        finally:
            self._join()

    def describe(self) -> dict:
        return {"kind": "device", "device": self.device_index, "name": self.device_name,
                "channels": self.channels, "samplerate": self.samplerate, "hostapi": self.hostapi}


# --------------------------------------------------------------------------------------------- files


def _ffmpeg_decode(path: Path, start_s: float, sr: int = 44100, channels: int = 2) -> np.ndarray:
    """Decode anything ffmpeg understands (m4a/AAC, video, ...) to float32 [n, channels]."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(f"cannot decode {path}: soundfile failed and ffmpeg is not installed")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(0.0, start_s):.3f}", "-i", str(path),
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(channels), "-ar", str(sr), "pipe:1"]
    res = subprocess.run(cmd, capture_output=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {res.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(res.stdout, dtype=np.float32).reshape(-1, channels)


def open_audio_blocks(path: Path, start_s: float, block_s: float) -> tuple[int, int, int, Iterator[np.ndarray]]:
    """(samplerate, channels, total_frames_after_start, iterator of float32 [n, ch] blocks)."""
    import soundfile as sf

    try:
        f = sf.SoundFile(str(path))
    except (sf.LibsndfileError, RuntimeError):
        data = _ffmpeg_decode(path, start_s)
        sr, blk = 44100, max(1, round(44100 * block_s))

        def chunks() -> Iterator[np.ndarray]:
            for i in range(0, len(data), blk):
                yield data[i:i + blk]
        return sr, data.shape[1], len(data), chunks()

    sr, nch = int(f.samplerate), int(f.channels)
    start = min(round(max(0.0, start_s) * sr), f.frames)
    f.seek(start)
    blk = max(1, round(sr * block_s))

    def blocks() -> Iterator[np.ndarray]:
        with f:
            yield from f.blocks(blocksize=blk, dtype="float32", always_2d=True)
    return sr, nch, f.frames - start, blocks()


class FileInput(_Source):
    """Play an audio file as a live input: 16 kHz mono blocks paced at real time x `speed`.

    `realtime=False` delivers as fast as the consumer accepts (offline simulation). `monitor=True`
    also plays the file on an output device; the output's blocking writes then pace delivery, and
    analysis blocks are held back by the output latency so tracking and what you hear stay aligned.
    With `speed != 1` the monitor plays tape-style (pitch follows speed).
    """

    def __init__(self, path: Path, realtime: bool = True, speed: float = 1.0, monitor: bool = False,
                 start_s: float = 0.0, *, block_s: float = 0.02, quality: str = "MQ",
                 output_device: int | str | None = None, channels: list[int] | None = None):
        super().__init__(quality)
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if speed <= 0:
            raise ValueError("speed must be > 0")
        if monitor and not realtime:
            raise ValueError("monitor playback needs realtime=True")
        self.realtime, self.speed, self.monitor = realtime, float(speed), monitor
        self.start_s, self.block_s = max(0.0, float(start_s)), float(block_s)
        self.output_device = output_device
        self.channels = channels            # 1-based file channels to downmix (None: all)
        self.samplerate = 0
        self.duration_s: float | None = None
        self._file_frames = 0

    @property
    def file_position_s(self) -> float:
        """Position in the source file (seconds) of the last block read."""
        return self.start_s + (self._file_frames / self.samplerate if self.samplerate else 0.0)

    def _downmix(self, x: np.ndarray) -> np.ndarray:
        if self.channels:
            x = x[:, [c - 1 for c in self.channels if 0 < c <= x.shape[1]] or [0]]
        return x[:, 0].astype(np.float32, copy=True) if x.shape[1] == 1 else x.mean(axis=1, dtype=np.float32)

    def start(self, on_audio: AudioCallback) -> None:
        self._on_audio = on_audio
        self._file_frames = 0
        self.delivered = 0
        self._spawn(self._work, f"FileInput[{self.path.name}]")

    def stop(self) -> None:
        self._join()

    def _open_monitor(self, sr: int, nch: int) -> tuple[Any, soxr.ResampleStream, int, int]:
        sd = _sounddevice()
        info = sd.query_devices(self.output_device, "output")
        out_sr = int(info["default_samplerate"]) or sr
        out_ch = max(1, min(2, int(info["max_output_channels"])))
        stream = sd.OutputStream(device=int(info["index"]), samplerate=out_sr, channels=out_ch,
                                 dtype="float32", latency="low")
        stream.start()
        mon_rs = soxr.ResampleStream(sr, out_sr / self.speed, out_ch, dtype="float32", quality=self.quality)
        delay_blocks = max(0, round(float(getattr(stream, "latency", 0.0) or 0.0) / self.block_s))
        return stream, mon_rs, out_ch, delay_blocks

    @staticmethod
    def _to_output(x: np.ndarray, out_ch: int) -> np.ndarray:
        if x.shape[1] == out_ch:
            return np.ascontiguousarray(x, dtype=np.float32)
        if x.shape[1] == 1:
            return np.ascontiguousarray(np.repeat(x, out_ch, axis=1), dtype=np.float32)
        if out_ch == 1:
            return x.mean(axis=1, keepdims=True, dtype=np.float32)
        return np.ascontiguousarray(x[:, :out_ch], dtype=np.float32)

    def _work(self) -> None:
        sr, nch, total, blocks = open_audio_blocks(self.path, self.start_s, self.block_s)
        self.samplerate, self.duration_s = sr, self.start_s + total / sr
        rs = soxr.ResampleStream(sr, ANALYSIS_SR, 1, dtype="float32", quality=self.quality)
        out = mon_rs = None
        out_ch, delay_blocks = 2, 0
        pending: deque[np.ndarray] = deque()
        if self.monitor:
            out, mon_rs, out_ch, delay_blocks = self._open_monitor(sr, nch)
        try:
            t0 = time.perf_counter()
            for x in blocks:
                if self._stop.is_set():
                    break
                mono = self._downmix(x)
                self.meter.update(mono, sr)
                y = rs.resample_chunk(mono)
                self._file_frames += len(x)
                if out is not None and mon_rs is not None:
                    if out.write(mon_rs.resample_chunk(self._to_output(x, out_ch))):
                        self.xruns += 1
                    pending.append(y)
                    while len(pending) > delay_blocks:
                        self._deliver(pending.popleft())
                    continue
                if self.realtime:
                    wait = t0 + (self._file_frames / sr) / self.speed - time.perf_counter()
                    if wait > 0 and self._stop.wait(wait):
                        break
                self._deliver(y)
            if not self._stop.is_set():
                if out is not None and mon_rs is not None:
                    out.write(mon_rs.resample_chunk(np.zeros((0, out_ch), np.float32), last=True))
                pending.append(rs.resample_chunk(np.zeros(0, np.float32), last=True))
                while pending:
                    self._deliver(pending.popleft())
        finally:
            if out is not None:
                try:
                    out.stop()
                finally:
                    out.close()

    def describe(self) -> dict:
        return {"kind": "file", "path": str(self.path), "realtime": self.realtime, "speed": self.speed,
                "monitor": self.monitor, "start_s": self.start_s}

    def stats(self) -> dict:
        return {**super().stats(), "file_position_s": round(self.file_position_s, 3),
                "duration_s": self.duration_s}


# --------------------------------------------------------------------------------------------- cli


def cli(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lyricprompter devices", description="List audio input devices")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = ap.parse_args(argv)
    devs = list_input_devices()
    if args.json:
        print(json.dumps(devs, ensure_ascii=False, indent=1))
        return 0
    if not devs:
        print("no input devices found")
        return 1
    print(f"{'':2}{'idx':>3}  {'ch':>3}  {'rate':>6}  {'host api':<12} name")
    for d in devs:
        mark = "*" if d["is_default"] else " "
        print(f"{mark:2}{d['index']:>3}  {d['channels']:>3}  {d['default_samplerate']:>6.0f}  "
              f"{d['hostapi']:<12} {d['name']}")
    print("\n* = default input. Choose the device and its channels (1-based) in the console under "
          "Input > Audio interface (`lyricprompter serve`).")
    return 0
