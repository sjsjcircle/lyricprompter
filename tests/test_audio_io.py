from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lyricprompter import audio_io
from lyricprompter.audio_io import (
    AudioError,
    duration,
    fit_length,
    load_mono,
    read_audio,
    save_wav,
    to_canonical_wav,
)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg/ffprobe not installed")


def sine(freq: float, sr: int, seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(round(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def peak_freq(y: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return float(np.argmax(spec) * sr / len(y))


@needs_ffmpeg
def test_mono_is_duplicated_at_unity_gain(tmp_path: Path) -> None:
    src = tmp_path / "mono.wav"
    sf.write(src, sine(440, 22050, 1.0), 22050)
    out = to_canonical_wav(src, tmp_path / "out" / "mix.wav")
    info = sf.info(out)
    assert (info.samplerate, info.channels, info.subtype) == (44100, 2, "PCM_24")
    assert abs(info.frames - 44100) <= 2
    y, _ = sf.read(out, dtype="float32")
    assert np.array_equal(y[:, 0], y[:, 1])
    assert np.abs(y).max() == pytest.approx(0.5, abs=0.01)     # plain `-ac 2` would give 0.354
    assert not list((tmp_path / "out").glob(".*part*"))


@needs_ffmpeg
def test_stereo_48k_flac_keeps_channels(tmp_path: Path) -> None:
    left = sine(1000, 48000, 0.5)
    src = tmp_path / "in.flac"
    sf.write(src, np.stack([left, np.zeros_like(left)], axis=1), 48000)
    out = to_canonical_wav(src, tmp_path / "mix.wav")
    y, sr = sf.read(out, dtype="float32")
    assert sr == 44100 and y.shape[1] == 2
    assert abs(y.shape[0] - 22050) <= 2
    assert np.abs(y[:, 1]).max() < 1e-3
    assert peak_freq(y[:, 0], sr) == pytest.approx(1000, abs=3)


@needs_ffmpeg
def test_m4a_input_conversion_and_fallback_decoding(tmp_path: Path) -> None:
    wav = tmp_path / "tone.wav"
    sf.write(wav, sine(660, 44100, 1.0), 44100)
    m4a = tmp_path / "tone.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(wav), "-c:a", "aac",
                    "-b:a", "128k", str(m4a)], check=True)
    with pytest.raises(sf.LibsndfileError):
        sf.info(str(m4a))                               # libsndfile can't read it ...
    out = to_canonical_wav(m4a, tmp_path / "mix.wav")   # ... ffmpeg can
    assert sf.info(out).channels == 2
    assert duration(m4a) == pytest.approx(1.0, abs=0.05)   # ffprobe fallback
    y = load_mono(m4a, 16000)                               # ffmpeg decode fallback
    assert y.dtype == np.float32 and abs(len(y) - 16000) < 800
    assert peak_freq(y, 16000) == pytest.approx(660, abs=3)


@needs_ffmpeg
def test_in_place_conversion(tmp_path: Path) -> None:
    p = tmp_path / "x.wav"
    sf.write(p, sine(220, 32000, 0.25), 32000, subtype="PCM_16")
    assert to_canonical_wav(p, p) == p
    assert sf.info(p).samplerate == 44100 and sf.info(p).channels == 2


@needs_ffmpeg
def test_conversion_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        to_canonical_wav(tmp_path / "missing.wav", tmp_path / "o.wav")
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(b"not audio at all" * 10)
    with pytest.raises(AudioError):
        to_canonical_wav(junk, tmp_path / "o.wav")
    assert not (tmp_path / "o.wav").exists()


def test_load_mono_downmix_and_resample(tmp_path: Path) -> None:
    sr = 44100
    left, right = sine(500, sr, 1.0, 0.4), sine(500, sr, 1.0, 0.2)
    p = tmp_path / "st.wav"
    sf.write(p, np.stack([left, right], axis=1), sr, subtype="FLOAT")
    y = load_mono(p, sr)
    np.testing.assert_allclose(y, (left + right) / 2, atol=1e-6)
    y16 = load_mono(p)
    assert y16.dtype == np.float32 and y16.flags.c_contiguous
    assert len(y16) == 16000
    assert peak_freq(y16, 16000) == pytest.approx(500, abs=2)
    assert np.abs(y16[1000:-1000]).max() == pytest.approx(0.3, abs=0.01)


def test_save_wav_layouts_and_subtypes(tmp_path: Path) -> None:
    st = np.stack([sine(300, 8000, 0.1), sine(600, 8000, 0.1)])          # [ch, n] -> transposed
    p = save_wav(tmp_path / "a.wav", st, 8000)
    y, sr = read_audio(p)
    assert sr == 8000 and y.shape == (800, 2) and sf.info(p).subtype == "PCM_24"
    np.testing.assert_allclose(y, st.T, atol=1e-6)
    m = save_wav(tmp_path / "b.wav", np.full(10, 1.5, np.float32), 8000, subtype="FLOAT")
    assert read_audio(m)[0].max() == pytest.approx(1.5)                    # FLOAT keeps headroom
    flac = save_wav(tmp_path / "c.flac", sine(300, 8000, 0.1), 8000)
    assert sf.info(flac).format == "FLAC"
    with pytest.raises(ValueError):
        save_wav(tmp_path / "bad.wav", np.array([0.0, np.nan]), 8000)
    assert sorted(x.name for x in tmp_path.iterdir()) == ["a.wav", "b.wav", "c.flac"]


def test_duration_and_fit_length(tmp_path: Path) -> None:
    p = save_wav(tmp_path / "d.wav", np.zeros((12345, 2), np.float32), 44100)
    assert duration(p) == pytest.approx(12345 / 44100)
    assert audio_io.num_frames(p) == 12345
    with pytest.raises(FileNotFoundError):
        duration(tmp_path / "nope.wav")
    y = np.ones((5, 2), np.float32)
    assert fit_length(y, 3).shape == (3, 2)
    padded = fit_length(y, 8)
    assert padded.shape == (8, 2) and padded[5:].sum() == 0
    assert fit_length(np.ones(4), 6).tolist() == [1, 1, 1, 1, 0, 0]
