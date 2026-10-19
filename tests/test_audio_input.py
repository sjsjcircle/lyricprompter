"""Audio sources with a fake sounddevice module (no real devices are ever opened)."""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from lyricprompter.live import audio_input as ai

# ----------------------------------------------------------------------------- fake sounddevice

DEVICES = [
    {"name": "Speakers", "index": 0, "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2,
     "default_samplerate": 48000.0},
    {"name": "Built-in Mic", "index": 1, "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0,
     "default_samplerate": 48000.0},
    {"name": "Stage Interface 4x4", "index": 2, "hostapi": 0, "max_input_channels": 4, "max_output_channels": 4,
     "default_samplerate": 44100.0},
    {"name": "Stereo Loopback", "index": 3, "hostapi": 1, "max_input_channels": 2, "max_output_channels": 2,
     "default_samplerate": 48000.0},
]
HOSTAPIS = [{"name": "Core Audio", "default_input_device": 1, "default_output_device": 0},
            {"name": "JACK", "default_input_device": -1, "default_output_device": -1}]


class FakePortAudioError(Exception):
    pass


class FakeSD:
    PortAudioError = FakePortAudioError

    def __init__(self, fail_channel_map: bool = False):
        self.inputs: list[FakeInputStream] = []
        self.outputs: list[FakeOutputStream] = []
        self.default = SimpleNamespace(device=[1, 0])
        self.fail_channel_map = fail_channel_map
        sd = self

        class CoreAudioSettings:
            def __init__(self, channel_map=None, **kw):
                self.channel_map = channel_map

        class InputStream(FakeInputStream):
            def __init__(self, **kw):
                if sd.fail_channel_map and kw.get("extra_settings") is not None:
                    raise FakePortAudioError("channel map unsupported")
                super().__init__(**kw)
                sd.inputs.append(self)

        class OutputStream(FakeOutputStream):
            def __init__(self, **kw):
                super().__init__(**kw)
                sd.outputs.append(self)

        self.CoreAudioSettings, self.InputStream, self.OutputStream = CoreAudioSettings, InputStream, OutputStream

    def query_hostapis(self, index=None):
        return HOSTAPIS if index is None else HOSTAPIS[index]

    def query_devices(self, device=None, kind=None):
        if kind is None and device is None:
            return [dict(d) for d in DEVICES]
        key = "max_input_channels" if kind == "input" else "max_output_channels"
        if device is None:
            device = self.default.device[0 if kind == "input" else 1]
        if isinstance(device, str):
            hits = [d for d in DEVICES if device.lower() in d["name"].lower() and d[key] > 0]
            if len(hits) != 1:
                raise ValueError(f"no unique device matching {device!r}")
            return dict(hits[0])
        d = DEVICES[int(device)]
        if d[key] <= 0:
            raise ValueError("not a device of that kind")
        return dict(d)


class FakeInputStream:
    def __init__(self, **kw):
        self.kw = kw
        self.started = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def feed(self, block: np.ndarray, overflow: bool = False) -> None:
        """Simulate one PortAudio callback."""
        self.kw["callback"](block.astype(np.float32), len(block), None, SimpleNamespace(input_overflow=overflow))


class FakeOutputStream:
    def __init__(self, **kw):
        self.kw = kw
        self.latency = 0.04
        self.frames_written = 0
        self.started = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def write(self, data: np.ndarray) -> bool:
        assert data.dtype == np.float32 and data.ndim == 2 and data.shape[1] == self.kw["channels"]
        self.frames_written += len(data)
        time.sleep(len(data) / self.kw["samplerate"] * 0.2)   # a (fast) device clock
        return False


@pytest.fixture
def fake_sd(monkeypatch: pytest.MonkeyPatch) -> FakeSD:
    sd = FakeSD()
    monkeypatch.setattr(ai, "_sounddevice", lambda: sd)
    return sd


class Collector:
    def __init__(self):
        self.blocks: list[np.ndarray] = []
        self.lock = threading.Lock()

    def __call__(self, y: np.ndarray) -> None:
        assert y.dtype == np.float32 and y.ndim == 1
        with self.lock:
            self.blocks.append(y.copy())

    @property
    def audio(self) -> np.ndarray:
        with self.lock:
            return np.concatenate(self.blocks) if self.blocks else np.zeros(0, np.float32)

    def wait_for(self, n: int, timeout: float = 3.0) -> np.ndarray:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and len(self.audio) < n:
            time.sleep(0.01)
        return self.audio


def sine(freq: float, sr: float, seconds: float, amp: float = 1.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_freq(y: np.ndarray, sr: float) -> float:
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return float(np.argmax(spec) * sr / len(y))


# ----------------------------------------------------------------------------- devices


def test_list_input_devices_filters_and_labels(fake_sd: FakeSD) -> None:
    devs = ai.list_input_devices()
    assert [d["index"] for d in devs] == [1, 2, 3]
    by = {d["index"]: d for d in devs}
    assert by[1]["is_default"] and not by[2]["is_default"]
    assert by[2] == {"index": 2, "name": "Stage Interface 4x4", "channels": 4, "default_samplerate": 44100.0,
                     "hostapi": "Core Audio", "is_default": False}
    assert by[3]["hostapi"] == "JACK"


def test_devices_cli_prints_table(fake_sd: FakeSD, capsys: pytest.CaptureFixture[str]) -> None:
    assert ai.cli([]) == 0
    out = capsys.readouterr().out
    assert "Stage Interface 4x4" in out and "Speakers" not in out
    assert "--device" not in out and "in the console" in out      # no command takes --device / --channels
    assert ai.cli(["--json"]) == 0
    assert '"channels": 4' in capsys.readouterr().out


def test_device_input_resolves_device_and_channels(fake_sd: FakeSD) -> None:
    mic = ai.DeviceInput()
    assert (mic.device_index, mic.channels, mic.samplerate, mic.hostapi) == (1, [1], 48000.0, "Core Audio")
    loop = ai.DeviceInput("loopback")
    assert loop.channels == [1, 2] and loop.hostapi == "JACK"
    iface = ai.DeviceInput(2, channels=[4, 3, 3], samplerate=48000)
    assert iface.channels == [3, 4] and iface.samplerate == 48000.0
    with pytest.raises(ValueError):
        ai.DeviceInput(2, channels=[5])
    with pytest.raises(ValueError):
        ai.DeviceInput(2, channels=[0])


def test_core_audio_uses_zero_based_channel_map(fake_sd: FakeSD) -> None:
    inp = ai.DeviceInput(2, channels=[2, 4])
    inp.start(Collector())
    try:
        (stream,) = fake_sd.inputs
        assert stream.started
        assert stream.kw["channels"] == 2
        assert stream.kw["extra_settings"].channel_map == [1, 3]
        assert stream.kw["samplerate"] == 44100.0 and stream.kw["dtype"] == "float32"
    finally:
        inp.stop()
    assert stream.closed and not inp.running


def test_channel_map_failure_falls_back_to_opening_all(fake_sd: FakeSD) -> None:
    fake_sd.fail_channel_map = True
    inp = ai.DeviceInput(2, channels=[3])
    got = Collector()
    inp.start(got)
    try:
        (stream,) = fake_sd.inputs
        assert stream.kw["channels"] == 3 and "extra_settings" not in stream.kw
        block = np.zeros((441, 3), np.float32)
        block[:, 2] = 0.5
        for _ in range(50):
            stream.feed(block)
        y = got.wait_for(7000)
    finally:
        inp.stop()
    assert len(y) > 7000
    assert np.allclose(y[2000:7000], 0.5, atol=1e-3)     # only channel 3 (DC 0.5) is used


def test_worker_downmixes_and_resamples_to_16k(fake_sd: FakeSD) -> None:
    inp = ai.DeviceInput("loopback")               # JACK: opens max(channels), picks on the worker
    got = Collector()
    inp.start(got)
    try:
        (stream,) = fake_sd.inputs
        left = sine(440.0, 48000, 1.0)
        x = np.stack([left, np.zeros_like(left)], axis=1)
        for i in range(0, len(x), 480):
            stream.feed(x[i:i + 480])
        y = got.wait_for(15500)
    finally:
        inp.stop()
    assert 15500 <= len(y) <= 16000                # 1 s in, minus the resampler's group delay
    assert abs(dominant_freq(y[1000:15000], 16000) - 440.0) < 3.0
    assert np.max(np.abs(y[2000:15000])) == pytest.approx(0.5, abs=0.02)   # (L + 0) / 2
    assert inp.level_db == pytest.approx(-9.0, abs=0.5)                     # 0.5-amplitude sine RMS


def test_callback_queue_drops_oldest_and_counts(fake_sd: FakeSD) -> None:
    inp = ai.DeviceInput(1, queue_blocks=2)
    for k in range(5):                               # no worker running: queue fills
        inp._callback(np.full((4, 1), k, np.float32), 4, None, SimpleNamespace(input_overflow=k == 0))
    assert inp.overflows == 3 and inp.xruns == 1
    kept = [inp._q.get_nowait()[0, 0] for _ in range(2)]
    assert kept == [3.0, 4.0]


def test_callback_copies_the_buffer(fake_sd: FakeSD) -> None:
    inp = ai.DeviceInput(1)
    buf = np.ones((8, 1), np.float32)
    inp._callback(buf, 8, None, None)
    buf[:] = 0                                       # PortAudio reuses its buffer
    assert inp._q.get_nowait().sum() == 8


def test_on_audio_errors_do_not_kill_the_worker(fake_sd: FakeSD) -> None:
    calls = []

    def boom(y: np.ndarray) -> None:
        calls.append(len(y))
        raise RuntimeError("tracker exploded")

    inp = ai.DeviceInput(1)
    inp.start(boom)
    try:
        (stream,) = fake_sd.inputs
        for _ in range(20):
            stream.feed(np.zeros((480, 1)))
        deadline = time.monotonic() + 2
        while len(calls) < 5 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert inp.running
    finally:
        inp.stop()
    assert inp.errors >= 5 and "tracker exploded" in (inp.last_error or "")


def test_a_lost_device_is_reported_and_stops_cleanly(fake_sd: FakeSD) -> None:
    """Regression: a device that stopped calling back (unplugged) looked like digital silence (the
    console blamed the microphone permission), and its PortAudio stop()/close() errors made the
    session's Stop an HTTP 500."""
    inp = ai.DeviceInput(2, channels=[1])
    got = Collector()
    inp.start(got)
    (stream,) = fake_sd.inputs
    for _ in range(20):
        stream.feed(np.full((441, 1), 0.1))
    got.wait_for(2000)
    st = inp.stats()
    assert st["stalled_s"] < ai.STALL_S and st["last_error"] is None
    inp._last_cb -= 2.5                                    # no callback for 2.5 s: the device is gone
    st = inp.stats()
    assert st["stalled_s"] >= 2.5 and st["last_error"] == ai.STALLED_TEXT and st["running"]
    stream.feed(np.full((441, 1), 0.1))                    # it came back
    assert inp.stats()["last_error"] is None

    def lost() -> None:
        raise FakePortAudioError("Internal PortAudio error [PaErrorCode -9986]")
    stream.stop = stream.close = lost
    inp.stop()                                             # logged, never raised
    assert not inp.running and inp._stream is None
    assert inp.stats()["stalled_s"] == 0.0 and inp.stats()["last_error"] is None


def test_level_meter_db_and_silence() -> None:
    m = ai.LevelMeter()
    m.update(sine(1000, 48000, 0.1), 48000)
    assert m.level_db == pytest.approx(-3.0, abs=0.1)
    assert m.peak_db == pytest.approx(0.0, abs=0.1)
    m2 = ai.LevelMeter()
    assert m2.level_db == ai.SILENCE_DB and m2.silent_s == 0.0
    m2.update(np.zeros(4800, np.float32), 48000)
    time.sleep(0.02)
    assert m2.level_db == ai.SILENCE_DB and m2.silent_s > 0.01


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_level_meter_ignores_non_finite_samples(bad: float) -> None:
    """Regression: one inf sample (a driver glitch) kept level_db at inf for the whole session
    (inf * decay == inf), which made every WebSocket state message invalid JSON."""
    m = ai.LevelMeter()
    y = sine(1000, 48000, 0.1)
    y[100] = bad
    m.update(y, 48000)
    assert m.level_db == pytest.approx(-3.0, abs=0.1) and m.peak_db == pytest.approx(0.0, abs=0.1)
    for _ in range(50):                                    # 5 s of a quiet signal: the release works
        m.update(sine(1000, 48000, 0.1) * 0.01, 48000)
    assert m.level_db == pytest.approx(-43.0, abs=0.5)
    m.update(np.full(480, bad, np.float32), 48000)         # a block of nothing but garbage
    stats = {"level": m.level_db, "peak": m.peak_db}
    assert all(np.isfinite(v) for v in stats.values()), stats


# ----------------------------------------------------------------------------- file input


@pytest.fixture
def wav(tmp_path: Path):
    def make(seconds: float = 1.0, sr: int = 44100, channels: int = 2, name: str = "in.wav") -> Path:
        y = sine(440.0, sr, seconds, 0.5)
        data = np.stack([y] * channels, axis=1) if channels > 1 else y
        path = tmp_path / name
        sf.write(path, data, sr, subtype="PCM_24")
        return path
    return make


def test_file_input_fast_mode_delivers_everything(wav) -> None:
    src = ai.FileInput(wav(1.0), realtime=False)
    got = Collector()
    t0 = time.perf_counter()
    src.start(got)
    assert src.wait(5.0)
    assert time.perf_counter() - t0 < 1.0
    y = got.audio
    assert abs(len(y) - 16000) <= 2
    assert abs(dominant_freq(y, 16000) - 440.0) < 2.0
    assert src.finished and not src.running
    assert src.position_s == pytest.approx(1.0, abs=1e-3)
    assert src.duration_s == pytest.approx(1.0, abs=1e-3)
    assert -10 < src.level_db < -5


def test_file_input_realtime_pacing_respects_speed(wav) -> None:
    src = ai.FileInput(wav(0.5), realtime=True, speed=5.0)
    got = Collector()
    t0 = time.perf_counter()
    src.start(got)
    assert src.wait(5.0)
    wall = time.perf_counter() - t0
    assert 0.08 <= wall <= 0.4                       # 0.5 s of audio at 5x
    assert abs(len(got.audio) - 8000) <= 2


def test_file_input_start_offset_and_stop(wav) -> None:
    src = ai.FileInput(wav(2.0), realtime=False, start_s=1.5)
    got = Collector()
    src.start(got)
    assert src.wait(5.0)
    assert abs(len(got.audio) - 8000) <= 2
    slow = ai.FileInput(wav(5.0, name="long.wav"), realtime=True)
    slow.start(Collector())
    time.sleep(0.1)
    t0 = time.perf_counter()
    slow.stop()
    assert time.perf_counter() - t0 < 0.5 and slow.finished
    assert 0.0 < slow.file_position_s < 1.0


def test_file_input_channel_selection(tmp_path: Path) -> None:
    sr = 16000
    data = np.zeros((sr, 2), np.float32)
    data[:, 1] = 0.25
    path = tmp_path / "lr.wav"
    sf.write(path, data, sr, subtype="FLOAT")
    left, right = Collector(), Collector()
    a = ai.FileInput(path, realtime=False, channels=[1])
    a.start(left)
    b = ai.FileInput(path, realtime=False, channels=[2])
    b.start(right)
    assert a.wait(5) and b.wait(5)
    assert np.allclose(left.audio, 0.0) and np.allclose(right.audio[100:-100], 0.25, atol=1e-4)


def test_file_input_monitor_plays_in_sync(fake_sd: FakeSD, wav) -> None:
    src = ai.FileInput(wav(1.0), realtime=True, speed=2.0, monitor=True)
    got = Collector()
    src.start(got)
    assert src.wait(5.0)
    (out,) = fake_sd.outputs
    assert out.kw["samplerate"] == 48000 and out.kw["channels"] == 2 and out.closed   # device native rate
    # tape-style 2x: 1 s of 44.1 kHz audio becomes 0.5 s of 48 kHz output frames
    assert out.frames_written == pytest.approx(24000, abs=60)
    assert abs(len(got.audio) - 16000) <= 2


def test_file_input_monitor_mono_file_to_stereo_output(fake_sd: FakeSD, wav) -> None:
    src = ai.FileInput(wav(0.5, sr=48000, channels=1, name="mono.wav"), realtime=True, monitor=True)
    got = Collector()
    src.start(got)
    assert src.wait(5.0) and src.last_error is None
    (out,) = fake_sd.outputs
    assert out.frames_written == pytest.approx(24000, abs=60)
    assert abs(len(got.audio) - 8000) <= 2
    assert src.stats()["stalled_s"] == 0.0 and src.stats()["last_error"] is None   # a file never stalls


def test_file_input_validation(tmp_path: Path, wav) -> None:
    with pytest.raises(FileNotFoundError):
        ai.FileInput(tmp_path / "missing.wav")
    with pytest.raises(ValueError):
        ai.FileInput(wav(0.1), realtime=False, monitor=True)
    with pytest.raises(ValueError):
        ai.FileInput(wav(0.1), speed=0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_file_input_decodes_compressed_audio_with_ffmpeg(wav, tmp_path: Path) -> None:
    m4a = tmp_path / "in.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(wav(1.0)), "-c:a", "aac", str(m4a)], check=True)
    src = ai.FileInput(m4a, realtime=False)
    got = Collector()
    src.start(got)
    assert src.wait(10.0)
    assert src.last_error is None
    assert abs(len(got.audio) - 16000) < 400        # AAC priming / padding
    assert abs(dominant_freq(got.audio, 16000) - 440.0) < 3.0


@pytest.mark.slow
def test_file_input_real_stem_fast_mode() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "stems" / "edelweiss_short" / "mix.wav"
    if not path.is_file():
        pytest.skip("data/stems/edelweiss_short not available")
    info = sf.info(str(path))
    src = ai.FileInput(path, realtime=False)
    got = Collector()
    levels: list[float] = []

    def on_audio(y: np.ndarray) -> None:
        got(y)
        levels.append(src.level_db)

    t0 = time.perf_counter()
    src.start(on_audio)
    assert src.wait(30)
    assert time.perf_counter() - t0 < 5.0
    assert abs(len(got.audio) - round(info.duration * 16000)) <= 2
    assert max(levels) > -30 and src.meter.samples == info.frames and src.stats()["errors"] == 0


def test_level_meter_ignores_dc_offset() -> None:
    from lyricprompter.live.audio_input import LevelMeter
    rng = np.random.default_rng(0)
    noise = (0.001 * rng.standard_normal(4800)).astype(np.float32)
    clean, biased = LevelMeter(), LevelMeter()
    for _ in range(5):
        clean.update(noise, 48000.0)
        biased.update(noise + 0.05, 48000.0)                 # a 0.05 DC bias would read as about -26 dBFS
    assert abs(biased.level_db - clean.level_db) < 1.0
    assert biased.peak_db < -40.0
