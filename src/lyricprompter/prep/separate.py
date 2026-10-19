"""Vocal separation with python-audio-separator (MPS), written as sample-aligned song stems.

`separate_vocals(mix, out_dir)` writes out_dir/{vocals,instrumental[,lead]}.wav, each exactly as long as
the mix, plus out_dir/stems.json recording the model, a mix fingerprint and timings. A later call with the
same model and the same mix reuses those files instead of separating again. stems.json is removed right
before a stem it lists is overwritten (after the new model has loaded and run, so a model that fails to
load costs nothing) and rewritten once the stems are consistent again, so it never vouches for files that
a different model, a user-supplied stem or a failed run produced.

`import_vocals` takes a pre-separated vocal stem instead. It is cross-correlated with the mix: a time
offset is corrected, and a level mismatch (a separator that peak-normalized its input, a mono source
upmixed at -3 dB) is rescaled so that instrumental = mix - vocals holds; both are recorded in stems.json
and reported as warnings.

Model notes (docs/research/*separation.json):
  * audio-separator peak-normalizes every stem on its own (normalization_threshold <= 1), which rescales
    each stem of a loud master by a different factor. We switch that off on the loaded model instance and
    feed the model a FLOAT copy of the mix (it writes stems in the input's subtype), so stem gains are kept
    and, for 2-stem models (secondary = mix - primary), vocals + instrumental == mix;
  * stems are stored as PCM24, or as FLOAT when they exceed full scale (never clipped);
  * the karaoke model's "Vocals" output is the lead singer(s); its "Instrumental" (accompaniment plus
    backing vocals) is not written;
  * output stem names differ per model ("vocals"/"other", "Vocals"/"Instrumental", 4-stem Demucs ...) and
    are mapped by name: one vocal stem, everything else summed into the instrumental;
  * model downloads go to a .part file that is renamed when complete (with byte progress, and without
    the library's stderr bar); a cut-off zip checkpoint left by an older interrupted download is moved
    aside so it is fetched again.
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
import warnings
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Self

import numpy as np
import soundfile as sf

from .. import config
from ..audio_io import fit_length, read_audio, resample, save_wav, to_canonical_wav
from . import Progress

DownloadProgress = Callable[[str, int, int], None]
"""(file name, bytes done, total bytes or 0 when unknown)."""

DEFAULT_SEP_MODEL = "bs_roformer_vocals_resurrection_unwa.ckpt"
LEAD_SEP_MODEL = "bs_roformer_karaoke_frazer_becruily.ckpt"
LICENSE_CLEAN_SEP_MODEL = "vocals_mel_band_roformer.ckpt"   # Kim Mel-Band RoFormer, MIT weights

MANIFEST = "stems.json"
MANIFEST_VERSION = 2          # 2: stems without per-stem peak normalization (v1 stems are not reused)
STEM_SUBTYPE = "PCM_24"       # FLOAT for a stem that exceeds full scale, see save_stem()
EXTERNAL_MODEL = "external"
NORMALIZATION_THRESHOLD = 1.0   # constructor argument only (it rejects > 1); disabled after load_model
# MPS autocast: ~40% faster warm chunks than fp32 on the M5 Pro (4.1 vs 6.7 s per chunk, Resurrection),
# vocal stem within 61 dB SNR of fp32 (scratch/b1-prep-io/bench_autocast.py).
USE_AUTOCAST = True

_VOCAL_NAMES = {"vocals", "vocal", "voice", "lead", "lead vocals", "lead_vocals", "singing"}
_INSTRUMENTAL_NAMES = {"instrumental", "inst", "other", "no vocals", "no_vocals", "accompaniment", "music"}
_ZIP_MAGIC = b"PK\x03\x04"

# Supplied vocal stems (import_vocals / match_vocals). On data/stems (edelweiss, edelweiss_short,
# kkotbat_ref) the true lag wins with corr 0.55-0.97 and a 6-20x margin over any other lag; an unrelated
# stem scores corr < 0.01 and a margin of ~1 (scratch/b1-prep-io/probe_vocal_match.py).
MATCH_SR = 8000                 # coarse cross-correlation rate; the lag is refined at the mix rate
MATCH_MAX_LAG_S = 3.0
MATCH_EXCLUDE_S = 0.02          # pitch-period side lobes around the peak do not count as rivals
MATCH_MIN_CORR = 0.05
MATCH_MIN_DOMINANCE = 3.0
LAG_WARN_S = 0.02               # smaller offsets are corrected silently (they only affect the instrumental)
# Level of the stem in the mix: median projection over the strongest 0.1% of vocal STFT bins (where
# leakage barely matters): 1.001-1.003 for unity-gain stems, 1.25 for a stem scaled by 0.8.
GAIN_TOL = 0.05
GAIN_MIN_BINS = 256
SEPARATOR_DEFAULT_PEAK = 0.9    # audio-separator's default normalization_threshold: input scaled to this peak

# Loaded Separators keyed by model filename. One heavy model per process: loading another evicts it.
_MAX_LOADED = 1
_loaded: OrderedDict[str, Any] = OrderedDict()
# Receiver of model download progress while a separation step loads its model (see _load).
_on_download: ContextVar[DownloadProgress | None] = ContextVar("lyricprompter_sep_download", default=None)


class SeparationError(RuntimeError):
    """The separator failed or produced unusable stems."""


# --------------------------------------------------------------------------------------------- models

def model_dir() -> Path:
    """Persistent model cache (AUDIO_SEPARATOR_MODEL_DIR; audio-separator raises if it is missing)."""
    d = Path(os.environ.get("AUDIO_SEPARATOR_MODEL_DIR") or config.MODELS_DIR / "audio-separator")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _truncated_checkpoint(path: Path) -> bool:
    """A zip-format torch checkpoint without its central directory, i.e. a cut-off download."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != _ZIP_MAGIC:
                return False
        return not zipfile.is_zipfile(path)
    except OSError:
        return False


def _quarantine_truncated(path: Path) -> Path | None:
    """Move a truncated checkpoint to '<name>.corrupt' so audio-separator downloads it again."""
    if not path.is_file() or not _truncated_checkpoint(path):
        return None
    bad = path.with_name(path.name + ".corrupt")
    os.replace(path, bad)
    return bad


def is_model_cached(model: str) -> bool:
    p = model_dir() / model
    return p.is_file() and not _truncated_checkpoint(p)


class _SilentBar:
    """Stand-in for tqdm that forwards (done, total) to `on_update` instead of drawing on stderr."""

    on_update: Callable[[int, int], None] = staticmethod(lambda done, total: None)

    def __init__(self, iterable: Any = None, *args: Any, total: float | None = None, **kwargs: Any):
        self._it = iterable
        if total is None and iterable is not None and hasattr(iterable, "__len__"):
            total = len(iterable)
        self.total = int(total or 0)
        self.n = 0

    def __iter__(self) -> Iterator[Any]:
        self.on_update(self.n, self.total)
        for x in self._it if self._it is not None else ():
            yield x
            self.update(1)

    def update(self, n: float = 1) -> None:
        self.n += int(n)
        self.on_update(self.n, self.total)

    def set_description(self, *args: Any, **kwargs: Any) -> None: ...
    def set_postfix(self, *args: Any, **kwargs: Any) -> None: ...
    def refresh(self, *args: Any, **kwargs: Any) -> None: ...
    def close(self) -> None: ...

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None: ...


@contextmanager
def _patched_tqdm(module: Any, on_update: Callable[[int, int], None]) -> Iterator[None]:
    """Replace `module.tqdm` with a _SilentBar reporting to `on_update` (no-op if the module has none)."""
    original = getattr(module, "tqdm", None) if module is not None else None
    if original is None:
        yield
        return
    module.tqdm = type("_Bar", (_SilentBar,), {"on_update": staticmethod(on_update)})
    try:
        yield
    finally:
        module.tqdm = original


def _atomic_downloads(sep: Any) -> None:
    """Make the Separator download into '<file>.part' and rename when complete.

    Its own downloader writes straight to the final path, so an interrupted first download (Ctrl-C, a
    server cancel, a dropped connection) would leave a truncated file that later runs treat as cached.
    Its tqdm bar (which would flood a captured stderr) is replaced by byte counts sent to the current
    `_on_download` receiver.
    """
    fetch = getattr(sep, "download_file_if_not_exists", None)
    if fetch is None:
        return
    module = sys.modules.get(type(sep).__module__)

    def download(url: str, output_path: str) -> None:
        if os.path.isfile(output_path):
            return
        name = Path(output_path).name
        part = Path(f"{output_path}.part")
        part.unlink(missing_ok=True)          # left by a killed run
        receiver = _on_download.get()
        try:
            with _patched_tqdm(module, lambda done, total: receiver and receiver(name, done, total)):
                fetch(url, str(part))
            os.replace(part, output_path)
        finally:
            part.unlink(missing_ok=True)

    sep.download_file_if_not_exists = download


def get_separator(model: str) -> Any:
    """Loaded `audio_separator.separator.Separator` for `model`, cached in-process (downloads on first use)."""
    sep = _loaded.get(model)
    if sep is not None:
        _loaded.move_to_end(model)
        return sep
    while len(_loaded) >= _MAX_LOADED:
        _free(_loaded.popitem(last=False)[1])
    mdir = model_dir()
    _quarantine_truncated(mdir / model)
    from audio_separator.separator import Separator  # heavy: torch + onnxruntime

    # Separator() calls warnings.filterwarnings("ignore") process-wide; keep that scoped.
    with warnings.catch_warnings():
        sep = Separator(log_level=logging.ERROR, model_file_dir=str(mdir), output_dir=tempfile.gettempdir(),
                        output_format="WAV", use_soundfile=True,
                        normalization_threshold=NORMALIZATION_THRESHOLD, use_autocast=USE_AUTOCAST)
        _atomic_downloads(sep)
        try:
            sep.load_model(model_filename=model)
        except ValueError as e:
            raise SeparationError(f"unknown separation model {model!r}: {e}") from e
        except (SystemExit, RuntimeError, OSError) as e:
            # An unreadable checkpoint makes the MDXC loader log and call sys.exit(1).
            cause = e.__context__ if isinstance(e, SystemExit) and e.__context__ is not None else e
            _free(sep)
            moved = _quarantine_truncated(mdir / model)
            hint = (f"; the truncated file was moved to {moved.name} and is downloaded again on the next run"
                    if moved else f" (if {mdir / model} is corrupt, delete it so it is downloaded again)")
            raise SeparationError(f"could not load separation model {model!r}: {cause}{hint}") from e
    _loaded[model] = sep
    return sep


def release_separators() -> None:
    """Drop every cached Separator and return GPU memory."""
    while _loaded:
        _free(_loaded.popitem(last=False)[1])


def _free(sep: Any) -> None:
    sep.model_instance = None
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _tqdm_progress(sep: Any, report: Callable[[float], None]) -> AbstractContextManager[None]:
    """Route the architecture module's tqdm loop into `report(fraction)` (and silence the bar)."""
    module = sys.modules.get(type(sep.model_instance).__module__)
    return _patched_tqdm(module, lambda done, total: report(done / total) if total else None)


def _vocal_stem_name(inst: Any) -> str | None:
    """The model's own name for its vocal output, if it has one (for output_single_stem)."""
    for name in (getattr(inst, "primary_stem_name", None), getattr(inst, "secondary_stem_name", None)):
        if isinstance(name, str) and name.strip().lower() in _VOCAL_NAMES:
            return name
    return None


def _run_model(model: str, mix_wav: Path, work: Path, report: Callable[[float], None], *,
               vocal_only: bool = False) -> dict[str, Path]:
    """Run one model on `mix_wav`; {lowercased stem name: written file}."""
    sep = get_separator(model)
    inst = sep.model_instance
    sep.output_dir = inst.output_dir = str(work)
    inst.normalization_threshold = math.inf          # keep stem gains (module docstring)
    inst.output_single_stem = _vocal_stem_name(inst) if vocal_only else None
    y, sr = read_audio(mix_wav)
    src = save_wav(work / mix_wav.name, y, sr, subtype="FLOAT")   # FLOAT in -> FLOAT stems out
    del y
    with warnings.catch_warnings(), _tqdm_progress(sep, report):
        warnings.simplefilter("ignore")
        try:
            outs = sep.separate(str(src))
        except Exception as e:  # library raises many unrelated types; keep the cause chained
            raise SeparationError(f"{model} failed on {mix_wav.name}: {e}") from e
    stems: dict[str, Path] = {}
    for o in outs or []:
        p = Path(o) if Path(o).is_absolute() else work / o
        if not p.is_file() or p.stat().st_size == 0:   # cf. audio-separator issue #297
            raise SeparationError(f"{model} did not write {p.name}")
        stems[_stem_name(p.name)] = p
    if not stems:
        raise SeparationError(f"{model} produced no output for {mix_wav.name}")
    return stems


def _stem_name(filename: str) -> str:
    """Default output names are '<audio>_(<Stem>)_<model>.wav'."""
    found = re.findall(r"\(([^()]+)\)", filename)
    return (found[-1] if found else Path(filename).stem).strip().lower()


# ---------------------------------------------------------------------------------------- stem maths

def save_stem(path: Path, y: np.ndarray, sr: int) -> Path:
    """PCM24 when the stem fits in [-1, 1], FLOAT otherwise (stems of a loud master can exceed it)."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return save_wav(path, y, sr, subtype=STEM_SUBTYPE if peak <= 1.0 else "FLOAT")


def _conform(y: np.ndarray, sr: int, channels: int, mix_sr: int) -> np.ndarray:
    """Resample to the mix rate and match its channel count (length unchanged)."""
    y = resample(y, sr, mix_sr)
    if y.shape[1] != channels:
        y = y.mean(axis=1, keepdims=True) if channels == 1 else np.repeat(y[:, :1], channels, axis=1)
    return y


def _match(y: np.ndarray, sr: int, n: int, channels: int, mix_sr: int) -> np.ndarray:
    return fit_length(_conform(y, sr, channels, mix_sr), n)


# ----------------------------------------------------------------------------------- supplied vocals

@dataclass
class VocalMatch:
    """How a supplied vocal stem lines up with the mix (see `match_vocals`)."""

    lag: int              # frames at the mix rate: supplied[t + lag] lines up with mix[t]
    corr: float           # normalized cross-correlation at the best lag (mono, MATCH_SR)
    dominance: float      # that peak over the largest value at any other lag (beyond MATCH_EXCLUDE_S)
    gain: float           # factor applied to the stem (1.0 unless its level is off by more than GAIN_TOL)
    gain_estimate: float | None
    gain_how: str         # "unity" | "estimate" | "separator-normalization" | "unverified"
    matched: bool         # corr and dominance high enough to trust lag and gain

    def to_dict(self, sr: int) -> dict[str, Any]:
        d = asdict(self)
        d.update(lag_s=round(self.lag / sr, 6), corr=round(self.corr, 4), dominance=round(self.dominance, 2),
                 gain=round(self.gain, 6),
                 gain_estimate=None if self.gain_estimate is None else round(self.gain_estimate, 4))
        return d


def _mono(y: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(y.mean(axis=1, dtype=np.float32) if y.ndim == 2 else y, dtype=np.float32)


def _dot_at(m: np.ndarray, v: np.ndarray, k: int, chunk: int = 1 << 20) -> float:
    """sum_t m[t] * v[t + k] over the overlap, accumulated in float64."""
    a, b = (m, v[k:]) if k >= 0 else (m[-k:], v)
    n = min(len(a), len(b))
    if n <= 0:
        return -math.inf
    a, b = a[:n], b[:n]
    return math.fsum(float(np.dot(a[i:i + chunk], b[i:i + chunk])) for i in range(0, n, chunk))


def _stft_projection(m: np.ndarray, v: np.ndarray, n_fft: int = 1024,
                     hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """(Re(M conj V), |V|^2) per STFT bin of two equally long mono signals (Hann window)."""
    win = np.hanning(n_fft).astype(np.float32)

    def spec(x: np.ndarray) -> np.ndarray:
        x = np.pad(x, (0, max(0, n_fft - len(x))))
        frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
        return np.fft.rfft(frames * win, axis=1)

    mm, vv = spec(m), spec(v)
    return np.real(mm * np.conj(vv)).ravel(), (np.abs(vv) ** 2).ravel()


def _gain_estimate(m: np.ndarray, v: np.ndarray, sr: int) -> float | None:
    """Level of the (aligned, mono) stem in the mix: median projection over its strongest STFT bins."""
    a, b = resample(m, sr, MATCH_SR), resample(v, sr, MATCH_SR)
    n = min(len(a), len(b))
    proj, power = _stft_projection(a[:n], b[:n])
    k = max(GAIN_MIN_BINS, power.size // 1000)
    if power.size < k:
        return None
    top = np.argpartition(power, -k)[-k:]
    top = top[power[top] > 0]
    if top.size < GAIN_MIN_BINS // 2:
        return None
    g = float(np.median(proj[top] / power[top]))
    return g if math.isfinite(g) and g > 0 else None


def match_vocals(vocals: np.ndarray, mix: np.ndarray, sr: int) -> VocalMatch:
    """Offset and level of a supplied vocal stem against the mix (both [n, ch] at `sr`).

    The lag comes from a cross-correlation at MATCH_SR within +-MATCH_MAX_LAG_S, refined to a frame at
    `sr`; it is trusted only when the correlation peak is clearly the best lag. The gain is the factor that
    brings the stem to its level in the mix; a gain within GAIN_TOL of audio-separator's default input
    normalization (mix peak / 0.9) is taken as exactly that.
    """
    vm, mm = _mono(vocals), _mono(mix)
    a = resample(mm, sr, MATCH_SR).astype(np.float64)
    b = resample(vm, sr, MATCH_SR).astype(np.float64)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    unmatched = VocalMatch(0, 0.0, 0.0, 1.0, None, "unverified", False)
    if norm == 0.0 or min(len(a), len(b)) < MATCH_SR // 2:
        return unmatched
    nfft = 1 << (len(a) + len(b) - 1).bit_length()
    r = np.fft.irfft(np.conj(np.fft.rfft(a, nfft)) * np.fft.rfft(b, nfft), nfft)
    span = min(int(MATCH_MAX_LAG_S * MATCH_SR), nfft // 2 - 1)
    lags = np.r_[0:span + 1, -span:0]
    vals = np.r_[r[:span + 1], r[nfft - span:]]
    p = int(np.argmax(vals))
    peak = float(vals[p])
    rivals = np.abs(lags - lags[p]) > MATCH_EXCLUDE_S * MATCH_SR
    runner_up = float(np.max(np.abs(vals[rivals]))) if rivals.any() else 0.0
    corr = peak / norm
    dominance = min(peak / runner_up, 1e6) if runner_up > 0 else 1e6
    if peak <= 0 or corr < MATCH_MIN_CORR or dominance < MATCH_MIN_DOMINANCE:
        return VocalMatch(0, corr, max(dominance, 0.0), 1.0, None, "unverified", False)

    centre = round(int(lags[p]) * sr / MATCH_SR)
    radius = math.ceil(sr / MATCH_SR) + 2
    lag = max(range(centre - radius, centre + radius + 1), key=lambda k: _dot_at(mm, vm, k))
    aligned = fit_length(_shift(vm, lag), len(mm))
    est = _gain_estimate(mm, aligned, sr)
    gain, how = 1.0, "unity"
    if est is not None and abs(est - 1.0) > GAIN_TOL:
        mix_peak = float(np.max(np.abs(mix)))
        sep_gain = mix_peak / SEPARATOR_DEFAULT_PEAK
        if mix_peak > SEPARATOR_DEFAULT_PEAK and abs(est - sep_gain) <= GAIN_TOL * sep_gain:
            gain, how = sep_gain, "separator-normalization"
        else:
            gain, how = est, "estimate"
    elif est is None:
        how = "unverified"
    return VocalMatch(lag, corr, dominance, gain, est, how, True)


def _shift(y: np.ndarray, lag: int) -> np.ndarray:
    """y advanced by `lag` frames (drop the first `lag`, or prepend -lag zeros)."""
    if lag > 0:
        return y[lag:]
    if lag < 0:
        return np.pad(y, [(-lag, 0)] + [(0, 0)] * (y.ndim - 1))
    return y


def _vocal_match_warnings(m: VocalMatch, sr: int, source_frames: int, mix_frames: int) -> list[str]:
    out = []
    if not m.matched:
        out.append(f"supplied vocals could not be lined up with the mix (correlation {m.corr:.3f}, peak "
                   f"margin {m.dominance:.1f}x); check that they were separated from this recording, else "
                   "timings and instrumental.wav are wrong")
    if abs(m.lag) > LAG_WARN_S * sr:
        out.append(f"supplied vocals were offset by {m.lag / sr:+.3f}s against the mix; shifted to line up")
    if m.gain != 1.0:
        peak = m.gain * SEPARATOR_DEFAULT_PEAK
        why = (f"audio-separator's default normalization of a mix peaking at {peak:.3f}"
               if m.gain_how == "separator-normalization" else "estimated")
        out.append(f"supplied vocals are at {1 / m.gain:.3f}x their level in the mix ({why}); rescaled by "
                   f"{m.gain:.4f} so that instrumental = mix - vocals")
    if abs(source_frames - m.lag - mix_frames) > sr:
        out.append(f"supplied vocals are {source_frames / sr:.2f}s but the mix is {mix_frames / sr:.2f}s; "
                   "padded/trimmed to the mix")
    return out


def split_roles(names: list[str]) -> tuple[str | None, list[str]]:
    """(vocal stem, non-vocal stems) from a model's output stem names."""
    vocal = [n for n in names if n in _VOCAL_NAMES]
    if len(vocal) > 1:
        raise SeparationError(f"ambiguous vocal stems {vocal}")
    rest = [n for n in names if n not in vocal]
    if not vocal and not any(n in _INSTRUMENTAL_NAMES for n in rest):
        raise SeparationError(f"model outputs {names} contain neither a vocal nor an instrumental stem")
    return (vocal[0] if vocal else None), rest


def _vocals_instrumental(stems: dict[str, Path], mix_wav: Path) -> tuple[np.ndarray, np.ndarray, int, dict]:
    mix, mix_sr = read_audio(mix_wav)
    n, ch = mix.shape
    vocal_name, rest = split_roles(list(stems))

    def load(name: str) -> np.ndarray:
        y, sr = read_audio(stems[name])
        return _match(y, sr, n, ch, mix_sr)

    if vocal_name is None:
        if len(rest) != 1:
            raise SeparationError(f"cannot derive vocals from stems {rest}")
        instrumental = load(rest[0])
        vocals = mix - instrumental
        how = {"vocals": "mix-" + rest[0], "instrumental": rest[0]}
    else:
        vocals = load(vocal_name)
        if rest:
            instrumental = load(rest[0])
            for name in rest[1:]:
                instrumental += load(name)
            how = {"vocals": vocal_name, "instrumental": "+".join(rest)}
        else:
            instrumental = mix - vocals
            how = {"vocals": vocal_name, "instrumental": "mix-" + vocal_name}
    return vocals, instrumental, mix_sr, how


# ------------------------------------------------------------------------------------------ manifest

def file_sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_manifest(out_dir: Path) -> dict | None:
    p = Path(out_dir) / MANIFEST
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get("version") == MANIFEST_VERSION else None


def invalidate_manifest(out_dir: Path) -> None:
    """Forget what out_dir's stems are (call before overwriting any of them)."""
    (Path(out_dir) / MANIFEST).unlink(missing_ok=True)


def write_manifest(out_dir: Path, *, model: str, mix_wav: Path, stems: dict[str, Path],
                   lead_model: str | None = None, mix_sha256: str | None = None, **extra: Any) -> Path:
    """Record which model produced which stems for which mix (out_dir/stems.json)."""
    out_dir = Path(out_dir)
    frames, _, sr = _frames(mix_wav)
    d = {"version": MANIFEST_VERSION, "model": model, "lead_model": lead_model,
         "mix": Path(mix_wav).name, "mix_sha256": mix_sha256 or file_sha256(Path(mix_wav)),
         "mix_frames": frames, "sr": sr,
         "stems": {k: os.path.relpath(Path(v), out_dir) for k, v in stems.items()},
         "subtypes": {k: sf.info(str(v)).subtype for k, v in stems.items()},
         "audio_separator": _version("audio-separator"), **extra}
    p = out_dir / MANIFEST
    tmp = p.with_name(f".{MANIFEST}.part")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return p


def _frames(path: Path) -> tuple[int, int, int]:
    """(frames, channels, samplerate) from the file header."""
    info = sf.info(str(path))
    return int(info.frames), int(info.channels), int(info.samplerate)


def cached_stems(out_dir: Path, mix_wav: Path, model: str, lead: bool = False,
                 mix_sha256: str | None = None) -> dict[str, Path] | None:
    """All valid stems recorded in out_dir/stems.json for `model` and this exact mix, or None.

    Requires vocals (+ instrumental for separator models, + lead when `lead`); every listed file must
    exist with exactly the mix's frame count.
    """
    out_dir = Path(out_dir)
    m = read_manifest(out_dir)
    if not m or m.get("model") != model:
        return None
    try:
        frames = _frames(mix_wav)[0]
        if m.get("mix_frames") != frames or m.get("mix_sha256") != (mix_sha256 or file_sha256(Path(mix_wav))):
            return None
        out = {name: out_dir / rel for name, rel in (m.get("stems") or {}).items()
               if (out_dir / rel).is_file() and _frames(out_dir / rel)[0] == frames}
    except (OSError, RuntimeError):
        return None
    required = {"vocals"} | ({"lead"} if lead else set())
    if not model.startswith(EXTERNAL_MODEL):
        required.add("instrumental")
    return out if required <= out.keys() else None


def _valid_lead(out_dir: Path, manifest: dict, sha: str, frames: int) -> str | None:
    """Model of the lead.wav that `manifest` records for this exact mix, else None."""
    p = out_dir / "lead.wav"
    model = manifest.get("lead_model")
    try:
        ok = (bool(model) and manifest.get("mix_sha256") == sha and "lead" in (manifest.get("stems") or {})
              and p.is_file() and _frames(p)[0] == frames)
    except (OSError, RuntimeError):
        ok = False
    return model if ok else None


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


# ------------------------------------------------------------------------------------------ public

def _reporter(progress: Progress | None, lo: float, hi: float, msg: str) -> Callable[[float], None]:
    """fraction 0..1 -> progress("separate", lo..hi, msg), deduplicated on whole percents."""
    last = [-1]

    def report(frac: float) -> None:
        pct = round(lo + (hi - lo) * min(max(frac, 0.0), 1.0))
        if progress is not None and pct != last[0]:
            last[0] = pct
            progress("separate", float(pct), msg)
    return report


def _download_reporter(progress: Progress | None, lo: float, hi: float) -> DownloadProgress | None:
    """Model download bytes -> progress("separate", lo..hi, "downloading <file> a/b MB"), ~every 2%."""
    if progress is None:
        return None
    last: dict[str, int] = {}

    def report(name: str, done: int, total: int) -> None:
        step = max(total // 50, 1 << 20) if total > 0 else 8 << 20
        mark = done // step
        if last.get(name) == mark and not (total and done >= total):
            return
        last[name] = mark
        frac = min(done / total, 1.0) if total > 0 else 0.0
        size = f"{done / 1e6:.0f}/{total / 1e6:.0f} MB" if total > 0 else f"{done / 1e6:.0f} MB"
        progress("separate", round(lo + (hi - lo) * frac, 1), f"downloading {name} {size}")
    return report


def _load(model: str, progress: Progress | None, lo: float, hi: float) -> Any:
    """get_separator with a "loading" message and download progress in lo..hi."""
    _reporter(progress, lo, lo, f"loading {model}")(0.0)
    token = _on_download.set(_download_reporter(progress, lo, hi))
    try:
        return get_separator(model)
    finally:
        _on_download.reset(token)


def _clean_work_dirs(out_dir: Path) -> None:
    """Remove '.sep-*' folders a killed run left behind (live runs remove their own)."""
    for d in out_dir.glob(".sep-*"):
        shutil.rmtree(d, ignore_errors=True)


def separate_lead(mix_wav: Path, out_dir: Path, *, model: str = LEAD_SEP_MODEL,
                  progress: Progress | None = None, pct_range: tuple[float, float] = (0.0, 100.0),
                  before_write: Callable[[], None] | None = None) -> Path:
    """Lead vocals via the karaoke model (its "Vocals" stem) -> out_dir/lead.wav, mix-length.

    Does not write stems.json; `separate_vocals(lead=True)` / `import_vocals(lead=True)` record it.
    `before_write` runs once the model has produced its output, right before lead.wav is replaced.
    """
    mix_wav, out_dir = Path(mix_wav), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = pct_range
    mid = lo + 0.05 * (hi - lo)                       # lo..mid: loading / downloading the model
    _load(model, progress, lo, mid)
    work = Path(tempfile.mkdtemp(prefix=".sep-", dir=out_dir))
    try:
        stems = _run_model(model, mix_wav, work, _reporter(progress, mid, hi, f"lead vocals ({model})"),
                           vocal_only=True)
        vocal_name, _ = split_roles(list(stems))
        if vocal_name is None:
            raise SeparationError(f"{model} has no vocal (lead) stem: {sorted(stems)}")
        n, ch, mix_sr = _frames(mix_wav)
        y, sr = read_audio(stems[vocal_name])
        y = _match(y, sr, n, ch, mix_sr)
        if before_write is not None:
            before_write()
        return save_stem(out_dir / "lead.wav", y, mix_sr)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _finish(out_dir: Path, mix_wav: Path, sha: str, result: dict[str, Path], *, model: str, run_lead: bool,
            prev_lead: str | None, progress: Progress | None, lead_pct: tuple[float, float],
            timings: dict[str, float], **extra: Any) -> dict[str, Path]:
    """stems.json for `result` (vocals and instrumental already written), then the optional lead pass.

    The manifest is written before the lead pass so freshly written vocals stay reusable when the lead
    model fails; a still valid earlier lead stays listed until the new lead.wav is about to replace it.
    """
    def record(lead_model: str | None) -> None:
        write_manifest(out_dir, model=model, mix_wav=mix_wav, stems=result, lead_model=lead_model,
                       mix_sha256=sha, timings={k: round(v, 3) for k, v in timings.items()},
                       peak_normalization=False, **extra)

    def lead_changes() -> None:
        result.pop("lead", None)
        record(None)

    lead_model = prev_lead
    if prev_lead is not None:
        result["lead"] = out_dir / "lead.wav"
    record(lead_model)
    if run_lead:
        t0 = time.perf_counter()
        result["lead"] = separate_lead(mix_wav, out_dir, progress=progress, pct_range=lead_pct,
                                       before_write=lead_changes)
        timings["lead"] = time.perf_counter() - t0
        lead_model = LEAD_SEP_MODEL
        record(lead_model)
    _reporter(progress, 100, 100, "stems written")(1.0)
    return result


def separate_vocals(mix_wav: Path, out_dir: Path, *, model: str = DEFAULT_SEP_MODEL, lead: bool = False,
                    progress: Progress | None = None, reuse: bool = True) -> dict[str, Path]:
    """Separate `mix_wav` into out_dir/{vocals,instrumental[,lead]}.wav, each exactly mix-length.

    progress gets ("separate", 0..100 within this step, message). With `reuse`, stems that stems.json
    records for the same model and an identical mix are returned without running the model (a lead stem
    from an earlier run on the same mix is kept and returned too).
    """
    mix_wav, out_dir = Path(mix_wav), Path(out_dir)
    if not mix_wav.is_file():
        raise FileNotFoundError(f"mix not found: {mix_wav}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _clean_work_dirs(out_dir)
    sha = file_sha256(mix_wav)
    frames = _frames(mix_wav)[0]
    if reuse and (hit := cached_stems(out_dir, mix_wav, model, lead, mix_sha256=sha)) is not None:
        _reporter(progress, 100, 100, "reusing cached stems")(1.0)
        return hit

    prev = read_manifest(out_dir) or {}
    prev_lead = _valid_lead(out_dir, prev, sha, frames)
    if prev_lead is None:
        (out_dir / "lead.wav").unlink(missing_ok=True)   # unrecorded or made from a different mix
    have = cached_stems(out_dir, mix_wav, model, False, mix_sha256=sha) if reuse else None
    run_lead = lead and not (reuse and prev_lead == LEAD_SEP_MODEL)
    split = 60.0 if run_lead and have is None else 100.0     # percent band: vocals pass | lead pass
    timings: dict[str, float] = {}
    derived: dict = prev.get("derived") or {}
    precision = prev.get("precision")

    if have is not None:
        result = {k: have[k] for k in ("vocals", "instrumental")}
        timings.update(prev.get("timings") or {})
    else:
        t0 = time.perf_counter()
        precision = str(getattr(_load(model, progress, 0, 5), "effective_precision", None))
        timings["load"] = time.perf_counter() - t0
        work = Path(tempfile.mkdtemp(prefix=".sep-", dir=out_dir))
        try:
            t0 = time.perf_counter()
            stems = _run_model(model, mix_wav, work, _reporter(progress, 5, split - 5, f"vocals ({model})"))
            timings["separate"] = time.perf_counter() - t0
            _reporter(progress, split - 5, split - 5, "writing stems")(1.0)
            vocals, instrumental, sr, derived = _vocals_instrumental(stems, mix_wav)
            invalidate_manifest(out_dir)          # the stems it lists are about to be replaced
            result = {"vocals": save_stem(out_dir / "vocals.wav", vocals, sr),
                      "instrumental": save_stem(out_dir / "instrumental.wav", instrumental, sr)}
        finally:
            shutil.rmtree(work, ignore_errors=True)

    return _finish(out_dir, mix_wav, sha, result, model=model, run_lead=run_lead, prev_lead=prev_lead,
                   progress=progress, lead_pct=(split if have is None else 0.0, 99.0), timings=timings,
                   derived=derived, precision=precision)


def import_vocals(vocals_src: Path, mix_wav: Path, out_dir: Path, *, lead: bool = False,
                  progress: Progress | None = None) -> dict[str, Path]:
    """Use a pre-separated vocal stem of this mix instead of running a separator.

    Writes out_dir/vocals.wav (canonical, lined up with the mix by `match_vocals`, rescaled to its level
    in the mix, padded/trimmed to the mix) and instrumental = mix - vocals, which is how 2-stem
    separators define it. stems.json records model "external", the source, the match ("check") and
    "warnings" for the prep report. A lead from an earlier run on the same mix is kept; with `lead` and
    none available, the karaoke model runs on the mix.
    """
    vocals_src, mix_wav, out_dir = Path(vocals_src), Path(mix_wav), Path(out_dir)
    if not vocals_src.is_file():
        raise FileNotFoundError(f"vocals file not found: {vocals_src}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _clean_work_dirs(out_dir)
    sha = file_sha256(mix_wav)
    n, ch, _ = _frames(mix_wav)
    prev_lead = _valid_lead(out_dir, read_manifest(out_dir) or {}, sha, n)
    _reporter(progress, 0, 0, f"importing vocals {vocals_src.name}")(0.0)
    with tempfile.TemporaryDirectory(prefix=".sep-", dir=out_dir) as tmp:
        v, sr = read_audio(to_canonical_wav(vocals_src, Path(tmp) / "vocals.wav"))
    m, mix_sr = read_audio(mix_wav)
    source_frames = v.shape[0]
    v = _conform(v, sr, ch, mix_sr)
    _reporter(progress, 3, 3, "lining up vocals with the mix")(0.0)
    match = match_vocals(v, m, mix_sr)
    v = fit_length(_shift(v, match.lag), n)
    if match.gain != 1.0:
        v *= np.float32(match.gain)
    warnings_ = _vocal_match_warnings(match, mix_sr, source_frames, n)
    invalidate_manifest(out_dir)              # the stems it lists are about to be replaced
    if prev_lead is None:
        (out_dir / "lead.wav").unlink(missing_ok=True)
    result = {"vocals": save_stem(out_dir / "vocals.wav", v, mix_sr),
              "instrumental": save_stem(out_dir / "instrumental.wav", m - v, mix_sr)}
    del m, v
    return _finish(out_dir, mix_wav, sha, result, model=EXTERNAL_MODEL,
                   run_lead=lead and prev_lead != LEAD_SEP_MODEL, prev_lead=prev_lead, progress=progress,
                   lead_pct=(10.0, 99.0), timings={},
                   derived={"vocals": EXTERNAL_MODEL, "instrumental": "mix-vocals"},
                   source=str(vocals_src), source_sha256=file_sha256(vocals_src), source_frames=source_frames,
                   check=match.to_dict(mix_sr), warnings=warnings_)
