"""Streaming audio features shared by the reference (offline) and the live path.

Causal framing: frame k covers stream samples [(k+1)*hop - n_fft, (k+1)*hop) (zeros before the
stream start), so it becomes available exactly when sample (k+1)*hop arrives; its timestamp is
(k+1)*hop/sr. `compute_offline` streams a whole signal through the very same `push` code path.

The feature vector concatenates per-group unit vectors scaled by sqrt(w_g / sum(w)), so every frame
has unit norm and `1 - a @ b` equals the weight-averaged per-group cosine distance. Groups:

- chroma (12): power STFT -> chroma filterbank built once with the song tuning -> c / max(c) ->
  log1p(gamma * .) -> + eps (a silent frame becomes the uniform vector); rolled by `transpose`.
- mfcc: c1..c(n_mfcc-1) of the dB mel spectrum (optional causal mean normalisation) plus a constant
  "anchor" component, so silent frames point at the anchor axis and match each other.
- flux: half-wave rectified frame-to-frame increase of the dB mel spectrum (onsets) plus an anchor.

`rms_db` is the level (dBFS) of the newest hop of samples, used for activity detection.

Every analysis frame has its mean removed first (spectrum and `rms_db`): a DC offset of the input (cheap
interfaces) would otherwise read as signal, raise the activity gate's noise floor to the DC level and hide a
quiet singer, and leak into the lowest spectral bins. The reference goes through the same path.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.fft
import scipy.signal

from ..config import ANALYSIS_SR, HOP

log = logging.getLogger(__name__)

FEATURE_VERSION = 3              # 3: DC removed per frame (spectrum and rms_db)
_OFFLINE_BLOCK = 16384          # samples per push() in compute_offline
_MAX_BATCH = 256                # frames processed per vectorised batch (bounds memory)
_MAX_ABS = np.float32(1e4)      # input samples are clipped to +-this (keeps the power spectrum finite)


@dataclass(frozen=True)
class FeatureConfig:
    kind: str = "mix"            # "mix" | "vocal" (informational; behaviour is set by the fields below)
    sr: int = ANALYSIS_SR
    hop: int = HOP
    n_fft: int = 2048
    tuning: float = 0.0          # semitone fraction of the song (estimated offline from the reference)
    transpose: int = 0           # roll this stream's chroma up by k semitones
    # group weights (0 disables a group)
    w_chroma: float = 1.0
    w_mfcc: float = 1.0
    w_flux: float = 0.0
    # chroma
    chroma_gamma: float = 100.0      # log1p(gamma * c/max c) compression; 0 = linear
    chroma_eps: float = 0.02
    # mel / mfcc / flux
    n_mels: int = 40
    fmax: float = 8000.0
    n_mfcc: int = 13             # c0..c12 computed, c0 (level) dropped
    mfcc_anchor: float = 10.0    # dB-scale constant component
    mfcc_cmn_s: float = 0.0      # causal cepstral mean normalisation time constant (0 = off)
    flux_anchor: float = 6.0     # dB-scale constant component
    floor_db: float = -80.0      # absolute power floor (dBFS) of the log compressions

    def key(self) -> str:
        """Stable short hash of every field (plus the feature code version) for cache file names."""
        blob = json.dumps({**asdict(self), "_v": FEATURE_VERSION}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    @property
    def fps(self) -> float:
        return self.sr / self.hop

    @classmethod
    def for_mode(cls, mode: str, tuning: float = 0.0) -> FeatureConfig:
        """Tuned defaults per tracking mode (see docs/eval/TRACKING.md)."""
        if mode == "mix":
            return cls(kind="mix", tuning=tuning, n_fft=2048, chroma_gamma=1.0, w_chroma=1.0, w_mfcc=0.5,
                       w_flux=0.0, n_mfcc=20, mfcc_cmn_s=5.0, mfcc_anchor=2.0)
        if mode == "vocal":
            return cls(kind="vocal", tuning=tuning, n_fft=2048, chroma_gamma=1.0, w_chroma=1.0, w_mfcc=1.0,
                       w_flux=0.0, n_mfcc=20, mfcc_cmn_s=5.0, mfcc_anchor=2.0)
        raise ValueError(f"unknown mode {mode!r} (expected 'mix' or 'vocal')")


@dataclass
class FeatureFrames:
    feats: np.ndarray            # [n, D] float32
    rms_db: np.ndarray           # [n] float32
    groups: tuple[tuple[str, int, int], ...] = ()   # (name, start col, stop col)

    def __len__(self) -> int:
        return int(self.rms_db.shape[0])

    def group(self, name: str) -> slice | None:
        for g, a, b in self.groups:
            if g == name:
                return slice(a, b)
        return None

    def slice(self, start: int, stop: int | None = None) -> FeatureFrames:
        return FeatureFrames(self.feats[start:stop], self.rms_db[start:stop], self.groups)

    @classmethod
    def empty(cls, dim: int = 0, groups: tuple[tuple[str, int, int], ...] = ()) -> FeatureFrames:
        return cls(np.zeros((0, dim), np.float32), np.zeros(0, np.float32), groups)

    @classmethod
    def concat(cls, parts: list[FeatureFrames]) -> FeatureFrames:
        if not parts:
            return cls.empty()
        return cls(np.concatenate([p.feats for p in parts]).astype(np.float32, copy=False),
                   np.concatenate([p.rms_db for p in parts]).astype(np.float32, copy=False),
                   parts[0].groups)


def _unit_rows(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


class FeatureExtractor:
    """Stateful causal feature extractor; feed 16 kHz mono float32 blocks of any length."""

    def __init__(self, cfg: FeatureConfig):
        if cfg.n_fft < cfg.hop:
            raise ValueError("n_fft must be >= hop")
        self.cfg = cfg
        import librosa  # heavy import, only needed to design the filterbanks

        self._window = scipy.signal.get_window("hann", cfg.n_fft, fftbins=True).astype(np.float32)
        # power normalised so that a full-scale sinusoid peaks at ~1.0 in its bin
        self._pscale = np.float32(1.0 / (cfg.n_fft / 4.0) ** 2)
        self._floor = np.float32(10.0 ** (cfg.floor_db / 10.0))
        w = {"chroma": cfg.w_chroma, "mfcc": cfg.w_mfcc, "flux": cfg.w_flux}
        total = sum(v for v in w.values() if v > 0)
        if total <= 0:
            raise ValueError("at least one feature group weight must be > 0")
        self._scale = {k: np.float32(np.sqrt(v / total)) for k, v in w.items() if v > 0}
        self._fb_chroma = None
        self._fb_mel = None
        self._dct = None
        groups, col = [], 0
        if "chroma" in self._scale:
            self._fb_chroma = np.ascontiguousarray(librosa.filters.chroma(
                sr=cfg.sr, n_fft=cfg.n_fft, tuning=cfg.tuning, dtype=np.float32).T)
            groups.append(("chroma", col, col + 12))
            col += 12
        if "mfcc" in self._scale or "flux" in self._scale:
            self._fb_mel = np.ascontiguousarray(librosa.filters.mel(
                sr=cfg.sr, n_fft=cfg.n_fft, n_mels=cfg.n_mels, fmin=0.0, fmax=min(cfg.fmax, cfg.sr / 2),
                dtype=np.float32).T)
        if "mfcc" in self._scale:
            dct = scipy.fft.dct(np.eye(cfg.n_mels), type=2, norm="ortho", axis=0)[1:cfg.n_mfcc]
            self._dct = np.ascontiguousarray(dct.T.astype(np.float32))
            groups.append(("mfcc", col, col + cfg.n_mfcc))   # n_mfcc - 1 coefficients + anchor
            col += cfg.n_mfcc
        if "flux" in self._scale:
            groups.append(("flux", col, col + cfg.n_mels + 1))
            col += cfg.n_mels + 1
        self.groups: tuple[tuple[str, int, int], ...] = tuple(groups)
        self.dim = col
        self._cmn_a = (1.0 / (cfg.mfcc_cmn_s * cfg.fps)) if cfg.mfcc_cmn_s > 0 else 0.0
        self.reset()

    def reset(self) -> None:
        cfg = self.cfg
        self._buf = np.zeros(cfg.n_fft - cfg.hop, np.float32)   # left zero padding / history
        self._n_frames = 0
        self._prev_mel = np.full(cfg.n_mels, 10.0 * np.log10(self._floor), np.float32)
        self._cmn_zi = None if self._dct is None else np.zeros((1, self._dct.shape[1]))

    def push(self, samples: np.ndarray) -> FeatureFrames:
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("push() expects mono 1-D samples")
        if x.size and not np.abs(x).max() <= _MAX_ABS:                  # also catches NaN / inf
            # one NaN/inf sample (driver glitch) would otherwise poison the causal CMN state for good
            x = np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -_MAX_ABS, _MAX_ABS)
        buf = np.concatenate([self._buf, x]) if x.size else self._buf
        cfg = self.cfg
        n = (len(buf) - (cfg.n_fft - cfg.hop)) // cfg.hop
        if n <= 0:
            self._buf = buf
            return FeatureFrames.empty(self.dim, self.groups)
        view = np.lib.stride_tricks.sliding_window_view(buf, cfg.n_fft)[::cfg.hop]
        parts = [self._process(view[a:min(a + _MAX_BATCH, n)]) for a in range(0, n, _MAX_BATCH)]
        self._buf = buf[n * cfg.hop:].copy()
        return parts[0] if len(parts) == 1 else FeatureFrames.concat(parts)

    def _process(self, frames: np.ndarray) -> FeatureFrames:
        cfg = self.cfg
        n = frames.shape[0]
        frames = frames - frames.mean(axis=1, keepdims=True)              # DC is neither level nor spectrum
        newest = frames[:, -cfg.hop:]
        rms_db = (10.0 * np.log10(np.einsum("ij,ij->i", newest, newest) / cfg.hop + 1e-12)).astype(np.float32)
        spec = np.fft.rfft(frames * self._window, axis=1)
        power = (spec.real * spec.real + spec.imag * spec.imag) * self._pscale
        out = []
        if self._fb_chroma is not None:
            c = power @ self._fb_chroma
            c = c / (c.max(axis=1, keepdims=True) + self._floor)
            if cfg.chroma_gamma > 0:
                c = np.log1p(cfg.chroma_gamma * c) / np.float32(np.log1p(cfg.chroma_gamma))
            c = c + np.float32(cfg.chroma_eps)
            if cfg.transpose % 12:
                c = np.roll(c, cfg.transpose, axis=1)
            out.append(_unit_rows(c) * self._scale["chroma"])
        if self._fb_mel is not None:
            mel_db = 10.0 * np.log10(power @ self._fb_mel + self._floor)
            if self._dct is not None:
                cep = (mel_db @ self._dct).astype(np.float64)
                if self._cmn_a > 0:
                    a = self._cmn_a
                    ema, self._cmn_zi = scipy.signal.lfilter([a], [1.0, a - 1.0], cep, axis=0, zi=self._cmn_zi)
                    t = np.arange(self._n_frames + 1, self._n_frames + n + 1, dtype=np.float64)
                    cep = cep - ema / (1.0 - (1.0 - a) ** t)[:, None]     # bias-corrected EMA mean
                v = np.concatenate([cep, np.full((n, 1), cfg.mfcc_anchor)], axis=1)
                out.append((_unit_rows(v) * self._scale["mfcc"]).astype(np.float32))
            if "flux" in self._scale:
                prev = np.concatenate([self._prev_mel[None, :], mel_db[:-1]], axis=0)
                self._prev_mel = mel_db[-1].copy()
                o = np.maximum(mel_db - prev, 0.0)
                v = np.concatenate([o, np.full((n, 1), cfg.flux_anchor, np.float32)], axis=1)
                out.append(_unit_rows(v) * self._scale["flux"])
        self._n_frames += n
        feats = np.concatenate(out, axis=1) if len(out) > 1 else out[0]
        return FeatureFrames(np.ascontiguousarray(feats, dtype=np.float32), rms_db, self.groups)


def compute_offline(y: np.ndarray, cfg: FeatureConfig) -> FeatureFrames:
    """Whole-signal features through the streaming path (frames = len(y) // hop)."""
    ext = FeatureExtractor(cfg)
    y = np.asarray(y, dtype=np.float32)
    parts = [ext.push(y[i:i + _OFFLINE_BLOCK]) for i in range(0, len(y), _OFFLINE_BLOCK)]
    parts = [p for p in parts if len(p)]
    return FeatureFrames.concat(parts) if parts else FeatureFrames.empty(ext.dim, ext.groups)


def load_mono(path: Path, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Read any soundfile-readable file as float32 mono at `sr` (soxr HQ)."""
    import soundfile as sf
    import soxr

    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if file_sr != sr:
        y = soxr.resample(y, file_sr, sr, quality="HQ")
    return np.ascontiguousarray(y, dtype=np.float32)


def _stem_file(song_dir: Path, stem: str) -> Path:
    from ..songfile import STEM_FILES, stem_path

    if stem in STEM_FILES:
        return stem_path(song_dir, stem)
    p = song_dir / stem
    return p if p.is_file() else song_dir / "audio" / f"{stem}.wav"


def load_ref_features(song_dir: Path, stem: str, cfg: FeatureConfig) -> FeatureFrames:
    """Reference features for `stem` ("mix", "vocals", ...), cached in <song_dir>/cache/.

    The cache file is feat_<cfg.key()>_<stem>.npz and is recomputed when the stem's size or mtime
    changes."""
    song_dir = Path(song_dir)
    src = _stem_file(song_dir, stem)
    st = src.stat()
    tag = np.array([st.st_size, st.st_mtime_ns], dtype=np.int64)
    safe = stem.replace("/", "_").replace("\\", "_")
    cache = song_dir / "cache" / f"feat_{cfg.key()}_{safe}.npz"
    if cache.is_file():
        try:
            with np.load(cache, allow_pickle=False) as z:
                if np.array_equal(z["src_tag"], tag):
                    groups = tuple((str(n), int(a), int(b)) for n, a, b in
                                   zip(z["group_names"], z["group_start"], z["group_stop"]))
                    ff = FeatureFrames(z["feats"].astype(np.float32), z["rms_db"].astype(np.float32), groups)
                    if ff.feats.ndim == 2 and len(ff.feats) == len(ff):
                        return ff
        except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile) as e:
            log.info("ignoring unreadable feature cache %s (%s): recomputing", cache, e)
    ff = compute_offline(load_mono(src, cfg.sr), cfg)
    _write_cache(cache, ff, tag, cfg)
    return ff


def _write_cache(cache: Path, ff: FeatureFrames, tag: np.ndarray, cfg: FeatureConfig) -> None:
    """Atomic write through a unique temp file; concurrent writers of the same key both succeed
    (they produce identical content, the last replace wins). A failed write only loses the cache."""
    tmp = None
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache.parent, prefix=cache.name + ".", suffix=".tmp",
                                         delete=False) as fh:
            tmp = Path(fh.name)
            np.savez(fh, feats=ff.feats, rms_db=ff.rms_db, src_tag=tag,
                     group_names=np.array([g[0] for g in ff.groups], dtype="U16"),
                     group_start=np.array([g[1] for g in ff.groups], dtype=np.int64),
                     group_stop=np.array([g[2] for g in ff.groups], dtype=np.int64),
                     cfg=json.dumps(asdict(cfg)))
        os.replace(tmp, cache)
        tmp = None
    except OSError as e:
        log.warning("could not write feature cache %s: %s", cache, e)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
