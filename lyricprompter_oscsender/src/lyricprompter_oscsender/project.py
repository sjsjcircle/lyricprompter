"""Project store: one folder per performance piece under the projects root, and `oscsender add` / `list`.

    projects/<id>/project.json     {"schema", "id", "title", "created", "duration", "tuning", "style",
                                    "sources": {"audio", "vocals", "cues"}, "stems": ["mix"(, "vocals")]}
    projects/<id>/audio/mix.wav     the reference recording, 44.1 kHz stereo PCM24 (needs ffmpeg)
    projects/<id>/audio/vocals.wav  optional vocal stem (vocal-mode tracking), cut / padded to the mix length
    projects/<id>/cues.txt          the cue file, UTF-8 (the source of truth; may be edited by hand)
    projects/<id>/cache/            feature cache (features/extractor.py) and waveform peaks
    projects/<id>/logs/             show logs
    projects/.trash/                deleted projects (<id>-<YYYYmmdd-HHMMSS>)

A project is built in a private `.tmp-<id>-<uuid>` folder next to the others and moved into place in one
rename once everything (conversion, tuning, project.json) succeeded: a failed or interrupted `create_project`
never leaves a half project, and `list_projects` skips dot-folders. Every file the store rewrites
(project.json, cues.txt, vocals.wav, waveform caches) is written to a temp file and renamed over the old one.

The folder name is the project id (as for lyricprompter's song packages); `load_meta` reports the folder
name even when project.json says otherwise (a copied folder keeps working under its new name).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from . import config
from .audio_io import AudioError, duration, to_canonical_wav
from .config import PROJECT_SCHEMA, STYLES
from .songfile import STEM_FILES, stem_path

log = logging.getLogger(__name__)

META_FILE = "project.json"
CUES_FILE = "cues.txt"
TRASH_DIR = ".trash"
TMP_PREFIX = ".tmp-"
STALE_TMP_S = 6 * 3600           # build folders older than this were left by a killed run
LENGTH_WARN_S = 0.5              # a vocal stem whose length differs more than this from the mix is reported
WAVEFORM_POINTS = (100, 20000)   # allowed range of waveform() points
PEAKS_VERSION = 1                # bump when the peak computation changes (invalidates waveform caches)
_ID_RE = re.compile(r"\w[\w\-.]*")      # fullmatch (a `$` would let "x\n" through)
_ID_MAX_BYTES = 200                     # leaves room for the .tmp-/trash suffixes within NAME_MAX (255)
_SLUG_MAX_BYTES = 96
_BLOCK = 1 << 16
# librosa.estimate_tuning on the 16 kHz mono mix: 4096-point FFT (3.9 Hz bins, piptrack refines by
# parabolic interpolation), hop 1024, 0.01-semitone histogram resolution, default fmin/fmax 150-4000 Hz.
TUNING_PARAMS = {"sr": config.ANALYSIS_SR, "n_fft": 4096, "hop_length": 1024, "resolution": 0.01,
                 "bins_per_octave": 12}
_TUNING_CHUNK_FRAMES = 256       # STFT frames per piptrack call in estimate_tuning (16 s of audio, ~20 MB)


@dataclass
class ProjectMeta:
    id: str
    title: str
    created: str                     # ISO 8601 UTC, seconds
    duration: float                  # s, of audio/mix.wav
    tuning: float                    # deviation of the mix from A440 in semitones (estimate_tuning)
    style: str = "classic"           # live tracking style (config.STYLES)
    sources: dict = field(default_factory=lambda: {"audio": None, "vocals": None, "cues": None})
    stems: list[str] = field(default_factory=lambda: ["mix"])

    def to_dict(self) -> dict:
        return {"schema": PROJECT_SCHEMA, **asdict(self)}


# ------------------------------------------------------------------------------------------- ids


def nfc(text: str) -> str:
    """Project ids are NFC: folders copied through Finder or unzipped from macOS archives are often
    named in NFD, and APFS opens either form."""
    return unicodedata.normalize("NFC", text)


def check_project_id(pid: str) -> str:
    """The NFC form of a valid project id, else ValueError: one path component of word characters
    (any script), combining marks that have no precomposed form, '-' and '.'; no leading '.', no '..'."""
    if not isinstance(pid, str) or not pid:
        raise ValueError("the project id is empty")
    pid = nfc(pid)
    bare = "".join(c for c in pid if not unicodedata.category(c).startswith("M"))
    if (not _ID_RE.fullmatch(bare) or unicodedata.category(pid[0]).startswith("M") or ".." in pid
            or len(pid.encode("utf-8")) > _ID_MAX_BYTES):
        raise ValueError(f"invalid project id {pid!r}: use letters, digits, '_', '-' and '.' "
                         f"(no leading '.', no '..', at most {_ID_MAX_BYTES} bytes)")
    return pid


def slugify(title: str) -> str:
    """lyricprompter.songfile.slugify, with 'project' as the fallback and a length cap."""
    s = unicodedata.normalize("NFKC", title).strip().lower()
    s = re.sub(r"[^\w\-]+", "-", s, flags=re.UNICODE).strip("-")
    while len(s.encode("utf-8")) > _SLUG_MAX_BYTES:
        s = s[:-1]
    s = s.rstrip("-")
    try:
        return check_project_id(s)
    except ValueError:
        return "project"


def new_project_id(title: str, root: Path) -> str:
    """Slug of the title, with -2, -3 ... when that folder is taken (case-insensitively on APFS)."""
    base, root = slugify(title), Path(root)
    pid, n = base, 2
    while (root / pid).exists():
        pid, n = f"{base}-{n}", n + 1
    return pid


def project_dir(root: Path, pid: str) -> Path:
    """`root / pid` for a valid id, or the existing project folder whose NFC name is `pid` (a folder named
    in NFD on a file system that, unlike APFS, does not match the two forms)."""
    pid, root = check_project_id(pid), Path(root)
    pdir = root / pid
    if not (pdir / META_FILE).is_file() and root.is_dir():
        for d in root.iterdir():
            if d.name != pid and nfc(d.name) == pid and (d / META_FILE).is_file():
                return d
    return pdir


# ----------------------------------------------------------------------------------- file helpers


def _atomic_write(path: Path, data: bytes) -> None:
    """Write `data` to a unique temp file next to `path`, fsync it and rename it over `path`."""
    path = Path(path)
    if path.is_symlink():                # a cue file linked from elsewhere: update the file, keep the link
        path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _strip_bom(text: str) -> str:
    return text.removeprefix("\ufeff")


def _sweep_tmp(root: Path) -> None:
    """Remove build folders of killed runs (ones still in progress elsewhere are recent: left alone)."""
    now = time.time()
    for d in root.glob(f"{TMP_PREFIX}*"):
        try:
            if d.is_dir() and now - d.stat().st_mtime > STALE_TMP_S:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


# ------------------------------------------------------------------------------------- audio steps


def estimate_tuning(mix_wav: Path) -> float:
    """Tuning deviation of the mix from A440 in fractional semitones, [-0.5, 0.5); see TUNING_PARAMS.

    The value of librosa.estimate_tuning on the whole 16 kHz mono mix (load_mono), computed in bounded
    memory (a whole-show recording must not take gigabytes in the server): the mix is read, downmixed and
    resampled block by block, and piptrack, which works frame by frame, runs on chunks of the frames the
    whole-signal STFT has (centred, zero padding at both ends). Only the pitch peaks are kept for the
    median magnitude threshold over the whole mix."""
    import librosa
    import soxr

    p = TUNING_PARAMS
    sr, n_fft, hop = p["sr"], p["n_fft"], p["hop_length"]
    pitches: list[np.ndarray] = []
    mags: list[np.ndarray] = []
    buf = np.zeros(n_fft // 2, np.float32)     # the padded signal from the next frame to analyse on
    n, sound = 0, False                         # 16 kHz samples so far; any of them not zero

    def analyse(final: bool) -> None:
        nonlocal buf
        while len(buf) >= n_fft:
            k = min((len(buf) - n_fft) // hop + 1, _TUNING_CHUNK_FRAMES)   # frames with a whole window
            if k < _TUNING_CHUNK_FRAMES and not final:
                return
            pitch, mag = librosa.piptrack(y=buf[:(k - 1) * hop + n_fft], sr=sr, n_fft=n_fft, hop_length=hop,
                                          center=False)
            peak = pitch > 0
            pitches.append(pitch[peak])
            mags.append(mag[peak])
            buf = buf[k * hop:]

    def push(y: np.ndarray) -> None:
        nonlocal buf, n, sound
        n += y.size
        sound = sound or bool(np.any(y))
        buf = np.concatenate([buf, y.astype(np.float32, copy=False)])
        analyse(False)

    file_sr = int(sf.info(str(mix_wav)).samplerate)
    stream = soxr.ResampleStream(file_sr, sr, 1, dtype="float32", quality="HQ") if file_sr != sr else None
    for block in sf.blocks(str(mix_wav), blocksize=_BLOCK, dtype="float32", always_2d=True):
        mono = block.mean(axis=1, dtype=np.float32) if block.shape[1] > 1 else block[:, 0]   # as load_mono
        push(stream.resample_chunk(mono) if stream is not None else mono)
    if stream is not None:
        push(stream.resample_chunk(np.zeros(0, np.float32), last=True))
    if n < n_fft or not sound:
        return 0.0
    buf = np.concatenate([buf, np.zeros(n_fft // 2, np.float32)])
    analyse(True)
    pitch, mag = np.concatenate(pitches), np.concatenate(mags)
    threshold = np.median(mag) if mag.size else 0.0             # as librosa.estimate_tuning, over all frames
    t = librosa.pitch_tuning(pitch[mag >= threshold], resolution=p["resolution"],
                             bins_per_octave=p["bins_per_octave"])
    return round(float(t), 3) if np.isfinite(t) else 0.0


def _fit_to(src: Path, dst: Path, n_frames: int) -> None:
    """Copy a canonical WAV to `dst` cut / zero-padded to exactly `n_frames` frames (lossless: int32 I/O)."""
    with sf.SoundFile(str(src)) as fin, sf.SoundFile(str(dst), "w", samplerate=fin.samplerate,
                                                      channels=fin.channels, subtype="PCM_24",
                                                      format="WAV") as fout:
        left = n_frames
        while left > 0:
            block = fin.read(min(_BLOCK, left), dtype="int32", always_2d=True)
            if len(block) == 0:
                break
            fout.write(block)
            left -= len(block)
        while left > 0:
            k = min(_BLOCK, left)
            fout.write(np.zeros((k, fin.channels), np.int32))
            left -= k


def _import_vocals(vocals: Path, mix_wav: Path, dst: Path) -> list[str]:
    """Convert `vocals` to the canonical format and fit it to the mix length at `dst` (atomically)."""
    warnings: list[str] = []
    conv = dst.with_name(f".{dst.stem}-{uuid.uuid4().hex}.wav")
    fitted = dst.with_name(f".{dst.stem}-{uuid.uuid4().hex}.fit.wav")
    try:
        to_canonical_wav(vocals, conv)
        n_mix, n_voc = sf.info(str(mix_wav)).frames, sf.info(str(conv)).frames
        diff = (n_voc - n_mix) / config.CANONICAL_SR
        if abs(diff) > LENGTH_WARN_S:
            warnings.append(f"the vocal stem is {abs(diff):.2f} s {'longer' if diff > 0 else 'shorter'} than the "
                            f"mix: it was {'cut' if diff > 0 else 'padded with silence'} to the mix length "
                            "(both must start at the same moment)")
        if n_voc == n_mix:
            os.replace(conv, dst)
        else:
            _fit_to(conv, fitted, n_mix)
            os.replace(fitted, dst)
    finally:
        conv.unlink(missing_ok=True)
        fitted.unlink(missing_ok=True)
    return warnings


# ---------------------------------------------------------------------------------------- create


def create_project(root: Path, *, title: str, audio: Path, cues_text: str = "", vocals: Path | None = None,
                   style: str = "classic", project_id: str | None = None,
                   cues_name: str | None = None) -> tuple[Path, list[str]]:
    """Build projects/<id>/ from a reference recording and a cue file text; returns (folder, warnings).

    `project_id` None takes the slug of the title (-2, -3 ... when taken); a given id must be free
    (ValueError otherwise). The cue text is stored as given (only a leading BOM is dropped); `cues_name` is
    the original cue file name for project.json. Missing files raise FileNotFoundError, audio ffmpeg cannot
    read raises audio_io.AudioError, bad titles / styles / ids raise ValueError.
    """
    root, audio = Path(root), Path(audio)
    title = (title or "").strip()
    if not title:
        raise ValueError("the project title is empty")
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; expected one of {', '.join(STYLES)}")
    if not isinstance(cues_text, str):
        raise TypeError("the cue text must be a string")
    if not audio.is_file():
        raise FileNotFoundError(f"audio file not found: {audio}")
    if vocals is not None and not Path(vocals).is_file():
        raise FileNotFoundError(f"vocals file not found: {vocals}")
    if project_id is not None:
        project_id = check_project_id(project_id)
        if (root / project_id).exists():
            raise ValueError(f"project {project_id!r} already exists in {root}")
    root.mkdir(parents=True, exist_ok=True)
    _sweep_tmp(root)

    work = root / f"{TMP_PREFIX}{project_id or slugify(title)}-{uuid.uuid4().hex}"
    work.mkdir()
    warnings: list[str] = []
    try:
        mix = to_canonical_wav(audio, stem_path(work, "mix"))
        dur = duration(mix)
        stems = ["mix"]
        if vocals is not None:
            warnings += _import_vocals(Path(vocals), mix, stem_path(work, "vocals"))
            stems.append("vocals")
        tuning = estimate_tuning(mix)
        _atomic_write(work / CUES_FILE, _strip_bom(cues_text).encode("utf-8"))
        (work / "cache").mkdir()
        (work / "logs").mkdir()
        meta = ProjectMeta(id="", title=title, created=datetime.now(UTC).isoformat(timespec="seconds"),
                           duration=round(dur, 6), tuning=tuning, style=style,
                           sources={"audio": audio.name, "vocals": Path(vocals).name if vocals else None,
                                    "cues": cues_name},
                           stems=stems)
        pdir = _commit(work, root, meta, project_id)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return pdir, warnings


def _commit(work: Path, root: Path, meta: ProjectMeta, project_id: str | None) -> Path:
    """Write project.json with the final id and rename the build folder into place.

    The id is claimed with an exclusive mkdir and the build folder renamed over that empty folder, so two
    concurrent creations never share an id and a project never appears without its project.json.
    """
    for _ in range(1000):
        pid = project_id or new_project_id(meta.title, root)
        meta.id = pid
        save_meta(work, meta)
        pdir = root / pid
        try:
            pdir.mkdir()
        except FileExistsError:
            if project_id is not None:
                raise ValueError(f"project {project_id!r} already exists in {root}") from None
            continue
        try:
            os.rename(work, pdir)
        except BaseException:
            with contextlib.suppress(OSError):      # not ours any more if something was written into it
                pdir.rmdir()
            raise
        return pdir
    raise ValueError(f"could not allocate a project id for {meta.title!r} in {root}")


# ------------------------------------------------------------------------------------------ meta


def _num(v: Any, default: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return default
    return float(v)


def load_meta(pdir: Path) -> ProjectMeta:
    """project.json of a project folder. FileNotFoundError without one; ValueError when it is not JSON or
    has another schema. Missing or malformed optional fields get defaults (duration: measured from the mix,
    stems: the stem files present); the id is always the folder name."""
    pdir = Path(pdir)
    path = pdir / META_FILE
    if not path.is_file():
        raise FileNotFoundError(f"not a project (no {META_FILE}): {pdir}")
    try:
        d = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from None
    schema = d.get("schema") if isinstance(d, dict) else type(d).__name__
    if schema != PROJECT_SCHEMA:
        raise ValueError(f"unsupported {META_FILE} schema {schema!r} in {pdir} "
                         f"(expected {PROJECT_SCHEMA})")
    pid = nfc(pdir.name)
    title = d.get("title") if isinstance(d.get("title"), str) and d.get("title").strip() else pid
    created = d.get("created") if isinstance(d.get("created"), str) else ""
    dur = _num(d.get("duration"), -1.0)
    if dur < 0:
        try:
            dur = duration(stem_path(pdir, "mix"))
        except (OSError, AudioError):
            dur = 0.0
    style = d.get("style") if d.get("style") in STYLES else "classic"
    raw_src = d.get("sources") if isinstance(d.get("sources"), dict) else {}
    sources = {k: raw_src.get(k) if isinstance(raw_src.get(k), str) else None
               for k in ("audio", "vocals", "cues")}
    sources.update({k: v for k, v in raw_src.items() if k not in sources})
    stems = d.get("stems")
    if not (isinstance(stems, list) and stems and all(s in STEM_FILES for s in stems)):
        stems = [s for s in ("mix", "vocals") if stem_path(pdir, s).is_file()] or ["mix"]
    return ProjectMeta(id=pid, title=title, created=created, duration=dur, tuning=_num(d.get("tuning"), 0.0),
                       style=style, sources=sources, stems=list(dict.fromkeys(stems)))


def save_meta(pdir: Path, meta: ProjectMeta) -> None:
    """Write project.json atomically (ValueError for a meta that load_meta would not read back)."""
    if meta.style not in STYLES:
        raise ValueError(f"unknown style {meta.style!r}; expected one of {', '.join(STYLES)}")
    if not all(s in STEM_FILES for s in meta.stems):
        raise ValueError(f"unknown stems {meta.stems!r}; expected some of {', '.join(STEM_FILES)}")
    if not (math.isfinite(meta.duration) and meta.duration >= 0 and math.isfinite(meta.tuning)):
        raise ValueError(f"invalid duration / tuning: {meta.duration!r} / {meta.tuning!r}")
    data = json.dumps(meta.to_dict(), ensure_ascii=False, indent=1) + "\n"
    _atomic_write(Path(pdir) / META_FILE, data.encode("utf-8"))


def list_projects(root: Path) -> list[Path]:
    """Project folders (those holding project.json) under `root`, sorted by name; dot-folders
    (.trash, build folders) are skipped."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.parent for p in root.glob(f"*/{META_FILE}")
                  if not p.parent.name.startswith(".") and p.is_file())


# ------------------------------------------------------------------------------------------ cues


def cues_path(pdir: Path) -> Path:
    return Path(pdir) / CUES_FILE


def read_cues(pdir: Path) -> str:
    """The cue file text exactly as stored (line endings kept), without a leading BOM; "" when the
    project has no cue file. ValueError when the file is not UTF-8."""
    path = cues_path(pdir)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return ""
    try:
        return _strip_bom(data.decode("utf-8"))
    except UnicodeDecodeError as e:
        raise ValueError(f"{path} is not UTF-8 text (byte {e.start}): save it as UTF-8") from None


def write_cues(pdir: Path, text: str) -> None:
    """Replace the cue file atomically with `text` (UTF-8, line endings as given, no BOM)."""
    if not isinstance(text, str):
        raise TypeError("the cue text must be a string")
    pdir = Path(pdir)
    if not pdir.is_dir():
        raise FileNotFoundError(f"project folder not found: {pdir}")
    _atomic_write(cues_path(pdir), _strip_bom(text).encode("utf-8"))


# ---------------------------------------------------------------------------------------- stems


def _drop_stem_caches(pdir: Path, stem: str) -> None:
    """Housekeeping only: every cache is keyed by its stem's size and mtime, so stale ones are never used."""
    cache = Path(pdir) / "cache"
    for f in [*cache.glob(f"feat_*_{stem}.npz"), *cache.glob(f"peaks_{stem}_*.json")]:
        f.unlink(missing_ok=True)


def set_vocals(pdir: Path, path: Path) -> list[str]:
    """Add or replace the project's vocal stem; returns warnings (length mismatch)."""
    pdir, path = Path(pdir), Path(path)
    meta = load_meta(pdir)
    if not path.is_file():
        raise FileNotFoundError(f"vocals file not found: {path}")
    mix = stem_path(pdir, "mix")
    if not mix.is_file():
        raise FileNotFoundError(f"the project has no reference mix: {mix}")
    warnings = _import_vocals(path, mix, stem_path(pdir, "vocals"))
    _drop_stem_caches(pdir, "vocals")
    meta.stems = ["mix", "vocals"]
    meta.sources = {**meta.sources, "vocals": path.name}
    save_meta(pdir, meta)
    return warnings


def _peaks(path: Path, points: int) -> tuple[float, np.ndarray, np.ndarray]:
    """(duration, min, max) of the channel mean over `points` equal bins, read block by block."""
    info = sf.info(str(path))
    n, sr = int(info.frames), int(info.samplerate)
    lo = np.full(points, np.inf, np.float32)
    hi = np.full(points, -np.inf, np.float32)
    if n <= 0:
        return 0.0, np.zeros(points, np.float32), np.zeros(points, np.float32)
    pos = 0
    for block in sf.blocks(str(path), blocksize=_BLOCK, dtype="float32", always_2d=True):
        m = block.mean(axis=1) if block.shape[1] > 1 else block[:, 0]
        b = np.minimum(np.arange(pos, pos + len(m), dtype=np.int64) * points // n, points - 1)
        starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
        bins = b[starts]
        lo[bins] = np.minimum(lo[bins], np.minimum.reduceat(m, starts))
        hi[bins] = np.maximum(hi[bins], np.maximum.reduceat(m, starts))
        pos += len(m)
    filled = np.isfinite(lo)
    if not filled.all():                 # fewer frames than points (or a short read): repeat the bin before
        src = np.maximum.accumulate(np.where(filled, np.arange(points), 0))
        lo, hi = lo[src], hi[src]
        lo[~np.isfinite(lo)] = 0.0
        hi[~np.isfinite(hi)] = 0.0
    return n / sr, lo, hi


def waveform(pdir: Path, stem: str = "mix", points: int = 2000) -> dict:
    """Min / max peaks of a stem (channel mean) over `points` equal bins (clamped to 100..20000):
    {"duration", "points", "min": [...], "max": [...]}, values rounded to 4 decimals. Cached in
    cache/peaks_<stem>_<points>.json, recomputed when the stem's size or mtime changes."""
    if stem not in STEM_FILES:
        raise ValueError(f"unknown stem {stem!r}; expected one of {', '.join(STEM_FILES)}")
    points = int(min(max(int(points), WAVEFORM_POINTS[0]), WAVEFORM_POINTS[1]))
    path = stem_path(pdir, stem)
    try:
        st = path.stat()
    except FileNotFoundError:
        raise FileNotFoundError(f"the project has no {stem} stem: {path}") from None
    key = [PEAKS_VERSION, st.st_size, st.st_mtime_ns]
    cache = Path(pdir) / "cache" / f"peaks_{stem}_{points}.json"
    try:
        d = json.loads(cache.read_text(encoding="utf-8"))
        if d.get("key") == key and len(d.get("min", ())) == points and len(d.get("max", ())) == points:
            return {k: d[k] for k in ("duration", "points", "min", "max")}
    except (OSError, ValueError, AttributeError):
        pass
    dur, lo, hi = _peaks(path, points)
    out = {"duration": round(dur, 6), "points": points,
           "min": np.round(lo.astype(np.float64), 4).tolist(), "max": np.round(hi.astype(np.float64), 4).tolist()}
    try:
        _atomic_write(cache, json.dumps({"key": key, **out}, separators=(",", ":")).encode("utf-8"))
    except OSError as e:
        log.warning("could not write waveform cache %s: %s", cache, e)
    return out


# ----------------------------------------------------------------------------------------- trash


def trash_project(root: Path, pid: str) -> Path:
    """Move a project to <root>/.trash/<id>-<YYYYmmdd-HHMMSS> (-2, -3 ... if taken); returns the new path."""
    root = Path(root)
    src = project_dir(root, pid)
    if not src.is_dir():
        raise FileNotFoundError(f"project {nfc(pid)!r} not found in {root}")
    trash = root / TRASH_DIR
    trash.mkdir(exist_ok=True)
    base = f"{nfc(src.name)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    for n in range(1, 1000):
        dst = trash / (base if n == 1 else f"{base}-{n}")
        if dst.exists():
            continue
        try:
            os.rename(src, dst)          # onto a missing name: never merges into another trashed folder
            return dst
        except OSError:
            if dst.exists():             # taken in between
                continue
            raise
    raise OSError(f"could not move {src} to {trash}")


# ------------------------------------------------------------------------------------------- CLI


def _read_text_file(path: Path, what: str) -> str:
    try:
        return path.read_bytes().decode("utf-8-sig")
    except FileNotFoundError:
        raise FileNotFoundError(f"{what} not found: {path}") from None
    except UnicodeDecodeError as e:
        raise ValueError(f"{what} {path} is not UTF-8 text (byte {e.start}): save it as UTF-8") from None


def _cue_summary(text: str) -> tuple[int | None, list]:
    """(number of cues, issues) when the cue parser is available, else (None, [])."""
    try:
        from .cues import parse_cues

        sheet = parse_cues(text)
    except NotImplementedError:
        return None, []
    return len(sheet.cues), list(sheet.issues)


def _fmt_dur(s: float) -> str:
    s = max(0.0, float(s))
    return f"{int(s // 60)}:{s % 60:04.1f}"


def _merge_declared_targets(root: Path, cues_text: str) -> list[str]:
    """Add the cue file's @target declarations the console settings lack (as the server does for a cue file
    it receives); returns warnings."""
    from .cues import parse_cues
    from .settings import load_settings, merge_declared, save_settings, settings_path

    path = settings_path(root)
    try:
        new, added = merge_declared(load_settings(path), parse_cues(cues_text).targets)
    except ValueError as e:
        return [f"the cue file's @target lines were not added to the OSC targets: {e}"]
    if added and not save_settings(path, new):
        return [f"could not save the OSC targets {', '.join(added)} to {path}"]
    if added:
        print(f"added OSC targets from the cue file: {', '.join(added)}")
    return []


def cli_add(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="oscsender add",
                                 description="Create a project from a reference recording and a cue file.")
    ap.add_argument("--audio", required=True, type=Path, help="reference recording (any ffmpeg-readable format)")
    ap.add_argument("--cues", type=Path, default=None,
                    help="cue file (UTF-8 text, format: README.md); default: an empty cue file to fill in the console")
    ap.add_argument("--title", required=True)
    ap.add_argument("--vocals", type=Path, default=None,
                    help="vocal stem of the same recording (enables vocal-mode tracking)")
    ap.add_argument("--style", default="classic", choices=STYLES,
                    help="tracking style: classic (flexible tempo) or pop (steady band tempo)")
    ap.add_argument("--id", dest="project_id", default=None,
                    help="project id (default: slug of the title, with -2, -3 ... if taken)")
    ap.add_argument("--projects-dir", type=Path, default=None,
                    help=f"projects root (default: {config.PROJECTS_DIR})")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:              # argparse: --help -> 0, usage error -> 2
        return int(e.code or 0)
    root = a.projects_dir or config.PROJECTS_DIR
    try:
        text = _read_text_file(a.cues, "cue file") if a.cues is not None else ""
        pdir, warnings = create_project(root, title=a.title, audio=a.audio, cues_text=text, vocals=a.vocals,
                                        style=a.style, project_id=a.project_id,
                                        cues_name=a.cues.name if a.cues is not None else None)
        meta = load_meta(pdir)
    except (OSError, ValueError, AudioError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    n_cues, issues = _cue_summary(text)
    warnings += _merge_declared_targets(root, text)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    for iss in issues:
        print(f"cues.txt line {iss.line}: {iss.severity}: {iss.message}", file=sys.stderr)
    cue_info = "" if n_cues is None else f", {n_cues} cues"
    print(f"created {meta.id!r} ({meta.title}): {_fmt_dur(meta.duration)}, tuning {meta.tuning:+.2f}, "
          f"style {meta.style}, stems {'+'.join(meta.stems)}{cue_info}")
    print(Path(pdir).resolve())
    return 0


def _summary(pdir: Path) -> dict:
    try:
        meta = load_meta(pdir)
    except (OSError, ValueError) as e:
        return {"id": nfc(pdir.name), "error": str(e)}
    d = asdict(meta)
    try:
        n_cues, issues = _cue_summary(read_cues(pdir))
        d["n_cues"] = n_cues
        d["n_errors"] = None if n_cues is None else sum(1 for i in issues if i.severity == "error")
    except (OSError, ValueError) as e:
        d.update(n_cues=None, n_errors=None, error=str(e))
    return d


def cli_list(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="oscsender list", description="List projects.")
    ap.add_argument("--projects-dir", type=Path, default=None,
                    help=f"projects root (default: {config.PROJECTS_DIR})")
    ap.add_argument("--json", action="store_true", help="print a JSON list")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    root = a.projects_dir or config.PROJECTS_DIR
    rows = [_summary(p) for p in list_projects(root)]
    if a.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return 0
    if not rows:
        print(f"no projects in {root}")
        return 0
    print(f"{'id':<28} {'duration':>8} {'style':<7} {'stems':<10} {'cues':<14} title")
    for r in rows:
        if "title" not in r:
            print(f"{r['id']:<28} error: {r['error']}")
            continue
        cues = "?" if r.get("n_cues") is None else str(r["n_cues"])
        if r.get("n_errors"):
            cues += f" ({r['n_errors']} err)"
        print(f"{r['id']:<28} {_fmt_dur(r['duration']):>8} {r['style']:<7} {'+'.join(r['stems']):<10} "
              f"{cues:<14} {r['title']}")
    return 0
