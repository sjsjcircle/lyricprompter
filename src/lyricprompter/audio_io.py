"""Audio file I/O: ffmpeg conversion to the canonical song format, mono analysis loading, WAV writing.

Canonical song audio is 44.1 kHz stereo PCM24 WAV (`config.CANONICAL_SR` / `CANONICAL_CHANNELS`). Every
input goes through ffmpeg first, because libsndfile (and librosa >= 1.0) cannot decode m4a/AAC.
Analysis audio is float32 mono, downmixed by channel mean and resampled with soxr HQ.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .config import CANONICAL_CHANNELS, CANONICAL_SR

CANONICAL_SUBTYPE = "PCM_24"
# swresample defaults (filter_size=32) are fine for preview but we only convert once per song.
_RESAMPLE_FILTER = "filter_size=64:phase_shift=12:cutoff=0.97"


class AudioError(RuntimeError):
    """Audio could not be decoded, converted or probed."""


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise AudioError(f"{name} not found on PATH (install it, e.g. `brew install ffmpeg`)")
    return path


def _run(cmd: list[str], what: str, *, binary_out: bool = False) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        tail = "\n".join(err[-8:]) or f"exit code {proc.returncode}"
        raise AudioError(f"{what} failed:\n{tail}")
    return proc


def probe(path: Path) -> dict:
    """ffprobe the first audio stream: {"channels", "sample_rate", "duration"} (duration may be None)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    proc = _run([_tool("ffprobe"), "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=channels,sample_rate,duration:format=duration",
                 "-of", "json", str(path)], f"ffprobe {path.name}")
    info = json.loads(proc.stdout or b"{}")
    streams = info.get("streams") or []
    if not streams:
        raise AudioError(f"no audio stream in {path}")
    st = streams[0]
    dur = st.get("duration") or (info.get("format") or {}).get("duration")
    return {"channels": int(st.get("channels") or 0), "sample_rate": int(st.get("sample_rate") or 0),
            "duration": float(dur) if dur not in (None, "N/A") else None}


def to_canonical_wav(src: Path, dst: Path) -> Path:
    """Convert any ffmpeg-readable input (wav/flac/mp3/m4a/...) to 44.1 kHz stereo PCM24 WAV.

    Uses the first audio stream only. Mono is duplicated to both channels at unity gain (plain
    `-ac 2` would attenuate by 3 dB); more than two channels are downmixed by ffmpeg's default matrix.
    The output is written to a temp file and renamed, so `dst` may equal `src`.
    """
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"audio file not found: {src}")
    channels = probe(src)["channels"]
    filters = []
    if channels == 1:
        filters.append("pan=stereo|c0=c0|c1=c0")
    filters.append(f"aresample={CANONICAL_SR}:{_RESAMPLE_FILTER}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.stem}.part.wav")
    cmd = [_tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
           "-map", "0:a:0", "-vn", "-sn", "-dn", "-map_metadata", "-1",
           "-af", ",".join(filters), "-ac", str(CANONICAL_CHANNELS), "-ar", str(CANONICAL_SR),
           "-c:a", "pcm_s24le", "-bitexact", "-f", "wav", str(tmp)]
    try:
        _run(cmd, f"ffmpeg conversion of {src.name}")
        if not tmp.is_file() or sf.info(str(tmp)).frames == 0:
            raise AudioError(f"ffmpeg produced no audio for {src}")
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst


def _ffmpeg_decode(path: Path) -> tuple[np.ndarray, int]:
    """Decode with ffmpeg to float32 [n, ch] at the native rate (for formats libsndfile can't read)."""
    info = probe(path)
    ch, sr = info["channels"], info["sample_rate"]
    if ch < 1 or sr < 1:
        raise AudioError(f"cannot determine channels / sample rate of {path}")
    proc = _run([_tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
                 "-map", "0:a:0", "-vn", "-f", "f32le", "-c:a", "pcm_f32le", "pipe:1"],
                f"ffmpeg decode of {Path(path).name}")
    y = np.frombuffer(proc.stdout, dtype="<f4")   # read-only view of immutable bytes -> copy below
    return y[: y.size - y.size % ch].reshape(-1, ch).copy(), sr


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """float32 [n, ch] at the file's rate; soundfile first, ffmpeg decode as fallback."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        y, sr = _ffmpeg_decode(path)
    return y, int(sr)


def resample(y: np.ndarray, sr_in: int, sr_out: int, quality: str = "HQ") -> np.ndarray:
    """soxr resampling along axis 0; float32 out. No-op when the rates match."""
    y = np.asarray(y, dtype=np.float32)
    if int(sr_in) == int(sr_out) or y.shape[0] == 0:
        return y
    return np.asarray(soxr.resample(y, sr_in, sr_out, quality=quality), dtype=np.float32)


def load_mono(path: Path, sr: int = 16000) -> np.ndarray:
    """float32 mono at `sr`: mean of all channels, soxr HQ resampling."""
    y, file_sr = read_audio(path)
    mono = y.mean(axis=1, dtype=np.float32) if y.shape[1] > 1 else y[:, 0]
    return np.ascontiguousarray(resample(mono, file_sr, sr))


def _frames_first(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        return y
    if y.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {y.shape}")
    # [ch, n] (librosa / torch convention) -> [n, ch]; ambiguous tiny arrays are left as [n, ch].
    if y.shape[0] <= 2 < y.shape[1]:
        y = y.T
    return y


def save_wav(path: Path, y: np.ndarray, sr: int, subtype: str = CANONICAL_SUBTYPE) -> Path:
    """Write [n] or [n, ch] (a [ch, n] array with ch <= 2 is transposed) atomically.

    PCM subtypes clip to [-1, 1]; pass subtype="FLOAT" to keep headroom.
    """
    path = Path(path)
    y = _frames_first(y)
    if not np.all(np.isfinite(y)):
        raise ValueError(f"refusing to write non-finite samples to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.part{path.suffix or '.wav'}")
    fmt = (path.suffix[1:] or "wav").upper()
    try:
        sf.write(str(tmp), y, int(sr), subtype=subtype, format=fmt)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def duration(path: Path) -> float:
    """Seconds; soundfile header first, ffprobe as fallback."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")
    try:
        info = sf.info(str(path))
        if info.samplerate > 0 and info.frames > 0:
            return info.frames / info.samplerate
    except sf.LibsndfileError:
        pass
    dur = probe(path)["duration"]
    if dur is None:
        raise AudioError(f"cannot determine duration of {path}")
    return dur


def num_frames(path: Path) -> int:
    """Sample frames in a libsndfile-readable file."""
    return int(sf.info(str(path)).frames)


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    """Zero-pad or trim [n, ...] audio along axis 0 to exactly `n` frames."""
    if y.shape[0] == n:
        return y
    if y.shape[0] > n:
        return y[:n]
    pad = [(0, n - y.shape[0])] + [(0, 0)] * (y.ndim - 1)
    return np.pad(y, pad)
