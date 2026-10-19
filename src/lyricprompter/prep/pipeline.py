"""Offline preparation of a song package (DESIGN §3/§4) and the `lyricprompter prep` command.

Stages, in order (stage names as reported to `progress`):
  convert   ffmpeg -> audio/mix.wav (44.1 kHz stereo PCM24; an identical existing mix.wav is kept)
  parse     lyrics -> Lines/Words/Syllables/AlignUnits (before separation, so bad lyrics fail fast)
  separate  audio-separator -> audio/{vocals,instrumental[,lead]}.wav (reused when audio/stems.json matches;
            skipped when vocals are supplied)
  align     MMS_FA (+ optional Qwen refinement) on the 16 kHz vocal stem
  finalize  tuning estimate, song.json, prep_report.json

`progress(stage, percent, message)` receives an *overall* percent (0..100, never decreasing); each stage
owns a fixed band of it (see `STAGE_BANDS`).

Re-preparing an existing package (one with song.json) builds the new version in songs/<id>/.prep-*/,
seeded with hard links of the current audio/ so stems can still be reused, and moves it into place only
when every stage succeeded. A failed or cancelled re-prep leaves the old package untouched, including
its prep_report.json (which keeps describing the live package), and writes the failure to
prep_report.failed.json next to it; the new stems it computed are discarded.

A first prep works in place; prep_report.json says "running" until the run ends. If it fails or is
cancelled before any stem exists (bad audio, unusable lyrics, a model that does not load), the song
directory it created is removed again, so the id stays free for a retry. Once stems exist they are kept
with prep_report.json {"status": "failed" | "cancelled"}, and a retry with the same --id reuses them
(such a directory has no song.json, so listings that require one do not show it).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__, config
from ..audio_io import duration, load_mono, read_audio, to_canonical_wav
from ..songfile import STEM_FILES, new_song_id, save_song, song_dir, stem_path
from ..text.model import STYLES, Line, Song, SongMeta
from . import Progress
from .separate import (
    DEFAULT_SEP_MODEL,
    LEAD_SEP_MODEL,
    MANIFEST,
    cached_stems,
    file_sha256,
    import_vocals,
    read_manifest,
    release_separators,
    separate_vocals,
)

LANGS = ("auto", "ko", "ja", "en")
REFINE_MODES = ("auto", "none", "qwen")
STAGE_BANDS: dict[str, tuple[float, float]] = {
    "convert": (0.0, 4.0), "parse": (4.0, 6.0), "separate": (6.0, 60.0),
    "align": (60.0, 95.0), "finalize": (95.0, 100.0),
}
REPORT_FILE = "prep_report.json"
FAILED_REPORT_FILE = "prep_report.failed.json"   # failed re-prep of a package that stays in place
REPORT_SCHEMA = 1
# Samples at digital full scale in the PCM24 mix (a clipped master, or lossy-decoder overshoot that the
# conversion clipped): always counted, a warning above this fraction of all samples.
FULL_SCALE = 1.0 - 2.0 ** -23
CLIP_WARN_FRACTION = 1e-4
# librosa.estimate_tuning on the 16 kHz mono mix: 4096-point FFT (3.9 Hz bins, piptrack refines by
# parabolic interpolation), hop 1024, 0.01-semitone histogram resolution, default fmin/fmax 150-4000 Hz.
TUNING_PARAMS = {"sr": config.ANALYSIS_SR, "n_fft": 4096, "hop_length": 1024, "resolution": 0.01,
                 "bins_per_octave": 12}
_PACKAGES = ("numpy", "scipy", "librosa", "soundfile", "soxr", "torch", "torchaudio", "transformers",
             "audio-separator", "uroman", "pyopenjtalk-plus", "cmudict", "pyphen")
_SONG_ID_RE = re.compile(r"\w[\w.\-]*")


class PrepError(RuntimeError):
    """Preparation failed for a reason the user can fix (bad input, unusable lyrics ...)."""


class Terminated(KeyboardInterrupt):
    """SIGTERM (the server cancelling a job, or shutting down), raised so cleanup and the report run."""


# ------------------------------------------------------------------------------- stage call-outs
# Thin indirections so tests (and alternative front-ends) can substitute the heavy steps.

def _parse(raw: str, default_lang: str) -> list[Line]:
    from ..text import parse_lyrics

    return parse_lyrics(raw, default_lang)


def _align(lines: list[Line], vocals: np.ndarray, sr: int, refine: str, progress: Progress) -> Any:
    from . import align

    try:
        return align.align_lines(lines, vocals, sr, refine=refine, progress=progress)
    finally:
        release = getattr(align, "release_models", None)   # not part of the contract; free MPS if offered
        if callable(release):
            release()


# ------------------------------------------------------------------------------------- progress

class _OverallProgress:
    """Maps stage-local percents onto STAGE_BANDS and forwards a monotonic overall percent."""

    def __init__(self, sink: Progress | None):
        self.sink = sink
        self.pct = 0.0

    def __call__(self, stage: str, local_pct: float, msg: str) -> None:
        lo, hi = STAGE_BANDS[stage]
        local = min(max(float(local_pct), 0.0), 100.0) if math.isfinite(local_pct) else 0.0
        self.pct = max(self.pct, lo + (hi - lo) * local / 100.0)
        if self.sink is not None:
            self.sink(stage, round(self.pct, 1), msg)

    def stage(self, stage: str) -> Progress:
        """Callback for a sub-step; its own stage names become a message prefix."""
        def cb(sub: str, local_pct: float, msg: str) -> None:
            self(stage, local_pct, msg if sub == stage or not sub else f"{sub}: {msg}")
        return cb


# --------------------------------------------------------------------------------------- helpers

def estimate_tuning(mix_wav: Path) -> float:
    """Tuning deviation of the mix from A440 in fractional semitones, [-0.5, 0.5); see TUNING_PARAMS."""
    import librosa

    p = TUNING_PARAMS
    y = load_mono(mix_wav, p["sr"])
    if y.size < p["n_fft"] or not np.any(y):
        return 0.0
    t = librosa.estimate_tuning(y=y, sr=p["sr"], n_fft=p["n_fft"], hop_length=p["hop_length"],
                                resolution=p["resolution"], bins_per_octave=p["bins_per_octave"])
    return round(float(t), 3) if np.isfinite(t) else 0.0


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"lyricprompter": __version__}
    for dist in _PACKAGES:
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    try:
        head = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False).stdout
        out["ffmpeg"] = head.split()[2] if head.startswith("ffmpeg version") else None
    except OSError:
        out["ffmpeg"] = None
    return out


def _jsonable(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(o)
    return str(o)


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=_jsonable), encoding="utf-8")
    tmp.replace(path)


def _free_torch() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _claim_song_dir(title: str, song_id: str | None, root: Path) -> tuple[str, Path, bool]:
    """(validated id, its directory, whether this call created it); a fresh id is claimed atomically."""
    root.mkdir(parents=True, exist_ok=True)
    if song_id is not None:
        song_id = unicodedata.normalize("NFC", song_id)      # the server addresses packages by NFC id
        if not _SONG_ID_RE.fullmatch(song_id) or ".." in song_id:
            raise PrepError(f"invalid song id {song_id!r} (letters, digits, '_', '-', '.'; no leading '.', "
                            "no '..')")
        sdir = song_dir(song_id, root)
        try:
            sdir.mkdir(parents=True)
            return song_id, sdir, True
        except FileExistsError:
            if not sdir.is_dir():
                raise PrepError(f"{sdir} exists and is not a directory") from None
            return song_id, sdir, False
    for _ in range(100):
        sid = new_song_id(title, root)
        try:
            (sdir := song_dir(sid, root)).mkdir(parents=True)
            return sid, sdir, True
        except FileExistsError:
            continue
    raise PrepError(f"could not allocate a song id for {title!r} in {root}")


def _existing_style(sdir: Path) -> str:
    """Tracking style of the package being re-prepared ("classic" when unset or unreadable)."""
    try:
        style = json.loads((sdir / "song.json").read_text(encoding="utf-8")).get("meta", {}).get("style")
    except (OSError, ValueError, AttributeError):
        return "classic"
    return style if style in STYLES else "classic"


def _has_stems(audio_dir: Path) -> bool:
    """Whether a (partial) package holds anything worth keeping for a retry."""
    names = ("vocals.wav", "instrumental.wav", "lead.wav", MANIFEST)
    return any((audio_dir / name).is_file() for name in names)


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _stage(sdir: Path) -> Path:
    """Private work dir for re-preparing an existing package, seeded with links to its audio/ files.

    Every writer replaces files (temp + rename), so the links never modify the live package's inodes.
    """
    for old in [*sdir.glob(".prep-*"), *sdir.glob("audio/.sep-*")]:     # left by a killed run
        shutil.rmtree(old, ignore_errors=True)
    work = Path(tempfile.mkdtemp(prefix=".prep-", dir=sdir))
    (work / "audio").mkdir()
    if (sdir / "audio").is_dir():
        for f in (sdir / "audio").iterdir():
            if f.is_file() and not f.name.startswith("."):
                _link_or_copy(f, work / "audio" / f.name)
    return work


def _commit(work: Path, sdir: Path, *, drop_cache: bool) -> None:
    """Move a finished staged package over the live one.

    song.json is removed first and written last: readers never pair new audio with old timings, and a
    crash mid-way leaves an incomplete package rather than a wrong one.
    """
    (sdir / "song.json").unlink(missing_ok=True)
    live = sdir / "audio"
    live.mkdir(exist_ok=True)
    staged = {f.name: f for f in (work / "audio").iterdir() if f.is_file() and not f.name.startswith(".")}
    for name in {Path(rel).name for rel in STEM_FILES.values()} | {MANIFEST}:
        if name not in staged:
            (live / name).unlink(missing_ok=True)        # e.g. a lead that is no longer valid
    for name, f in staged.items():
        os.replace(f, live / name)
    if drop_cache:
        shutil.rmtree(sdir / "cache", ignore_errors=True)
    for name in ("lyrics.txt", REPORT_FILE):
        os.replace(work / name, sdir / name)
    (sdir / FAILED_REPORT_FILE).unlink(missing_ok=True)
    os.replace(work / "song.json", sdir / "song.json")


def _convert_mix(audio: Path, mix: Path) -> Path:
    """to_canonical_wav, keeping a byte-identical existing mix.wav (its mtime keys feature caches)."""
    if not mix.is_file():
        return to_canonical_wav(audio, mix)
    new = mix.with_name(".mix.new.wav")
    try:
        to_canonical_wav(audio, new)
        if file_sha256(new) != file_sha256(mix):
            os.replace(new, mix)
    finally:
        new.unlink(missing_ok=True)
    return mix


def _full_scale_samples(mix: Path) -> tuple[int, int]:
    """(samples at digital full scale, all samples) of a canonical mix."""
    y, _ = read_audio(mix)
    return int(np.count_nonzero(np.abs(y) >= FULL_SCALE)), int(y.size)


def _check_alignment(lines: list[Line], warnings: list[str]) -> None:
    prev = -math.inf
    for ln in lines:
        if not ln.words:
            continue
        if ln.start is None or ln.end is None:
            warnings.append(f"line {ln.idx} has no timing after alignment")
            continue
        if ln.start < prev - 1e-6:
            warnings.append(f"line {ln.idx} starts at {ln.start:.3f}s before the previous line")
        prev = ln.start


def _f(x: float | None) -> float | None:
    return None if x is None else float(x)


def _plain_numbers(lines: list[Line]) -> None:
    """numpy scalars from the aligner -> Python floats (song.json is written with plain json.dumps)."""
    for ln in lines:
        ln.start, ln.end, ln.confidence = _f(ln.start), _f(ln.end), _f(ln.confidence)
        ln.needs_review = bool(ln.needs_review)
        for w in ln.words:
            w.needs_review = bool(w.needs_review)
            for s in w.syllables:
                s.start, s.end = _f(s.start), _f(s.end)
                for u in s.units:
                    u.start, u.end, u.score = _f(u.start), _f(u.end), _f(u.score)


def _line_rows(lines: list[Line], confidences: list[float] | None) -> list[dict]:
    rows = []
    for i, ln in enumerate(lines):
        conf = confidences[i] if confidences is not None and i < len(confidences) else ln.confidence
        rows.append({"idx": ln.idx, "text": ln.text, "start": ln.start, "end": ln.end,
                     "confidence": None if conf is None else round(float(conf), 4),
                     "needs_review": ln.needs_review, "flags": list(ln.flags),
                     "syllables": len(ln.syllables())})
    return rows


# ------------------------------------------------------------------------------------------ main

def prepare_song(audio: Path, lyrics_text: str, *, title: str, artist: str = "", default_lang: str = "auto",
                 song_id: str | None = None, songs_root: Path | None = None,
                 sep_model: str = DEFAULT_SEP_MODEL, lead: bool = False, refine: str = "auto",
                 vocals: Path | None = None, style: str | None = None,
                 progress: Progress | None = None) -> Path:
    """Build songs/<id>/ from an audio file and lyrics text; returns the song directory.

    `style` is the song's live tracking style ("classic" | "pop", SongMeta.style); None keeps the style of
    the package being re-prepared ("classic" for a new song).

    With `song_id` an existing package is re-prepared (stems are reused when the mix and the separation
    model are unchanged) and replaced only on success; a failed re-prep writes prep_report.failed.json
    and leaves the package, prep_report.json included, as it was. A failed first prep removes the
    directory it created unless stems exist, which are kept with prep_report.json
    {"status": "failed" | "cancelled", "stage": ..., "error": ...} for a retry under the same id.
    """
    audio = Path(audio)
    if not audio.is_file():
        raise FileNotFoundError(f"audio file not found: {audio}")
    if vocals is not None and not Path(vocals).is_file():
        raise FileNotFoundError(f"vocals file not found: {vocals}")
    if not lyrics_text or not lyrics_text.strip():
        raise PrepError("lyrics are empty")
    if default_lang not in LANGS:
        raise PrepError(f"unknown language {default_lang!r}; expected one of {LANGS}")
    if refine not in REFINE_MODES:
        raise PrepError(f"unknown refine mode {refine!r}; expected one of {REFINE_MODES}")
    if style is not None and style not in STYLES:
        raise PrepError(f"unknown style {style!r}; expected one of {STYLES}")

    prog = _OverallProgress(progress)
    t_start = time.perf_counter()
    root = Path(songs_root) if songs_root else config.SONGS_DIR
    sid, sdir, fresh_dir = _claim_song_dir(title, song_id, root)
    staged = (sdir / "song.json").is_file()
    work = _stage(sdir) if staged else sdir
    audio_dir = work / "audio"
    created = datetime.now(UTC).isoformat(timespec="seconds")
    timings: dict[str, float] = {}
    warnings: list[str] = []
    options = {"default_lang": default_lang, "sep_model": None if vocals else sep_model, "lead": lead,
               "refine": refine, "vocals": str(vocals) if vocals else None}
    report: dict[str, Any] = {"schema": REPORT_SCHEMA, "song_id": sid, "status": "running", "created": created,
                              "source": {"audio": str(audio), "lyrics_chars": len(lyrics_text)},
                              "options": options, "timings": timings, "warnings": warnings}
    stage = "convert"
    try:
        audio_dir.mkdir(parents=True, exist_ok=True)
        _write_json(work / REPORT_FILE, report)
        (work / "lyrics.txt").write_text(lyrics_text, encoding="utf-8")

        # 1. convert -----------------------------------------------------------------------------
        t0 = time.perf_counter()
        prog(stage, 0, f"converting {audio.name}")
        mix = _convert_mix(audio, stem_path(work, "mix"))
        dur = duration(mix)
        report["duration"] = round(dur, 3)
        clipped, total = _full_scale_samples(mix)
        report["source"]["full_scale_samples"] = clipped
        if clipped > CLIP_WARN_FRACTION * total:
            warnings.append(f"mix.wav has {clipped} samples at digital full scale ({100 * clipped / total:.3f}%): "
                            "the source is clipped, or its decoded peaks exceeded full scale")
        timings[stage] = time.perf_counter() - t0
        prog(stage, 100, f"mix.wav {dur:.1f}s")

        # 2. parse -------------------------------------------------------------------------------
        stage, t0 = "parse", time.perf_counter()
        prog(stage, 0, "parsing lyrics")
        lines = _parse(lyrics_text, default_lang)
        n_units = sum(1 for ln in lines for s in ln.syllables() for u in s.units if u.roman)
        languages = dict(Counter(w.lang for ln in lines for w in ln.words).most_common())
        if not lines or n_units == 0:
            raise PrepError("lyrics contain no alignable text")
        timings[stage] = time.perf_counter() - t0
        prog(stage, 100, f"{len(lines)} lines, {n_units} acoustic units")

        # 3. separate ----------------------------------------------------------------------------
        stage, t0 = "separate", time.perf_counter()
        sep_cb = prog.stage(stage)
        try:
            if vocals is not None:
                stems = import_vocals(Path(vocals), mix, audio_dir, lead=lead, progress=sep_cb)
                reused = False
            else:
                reused = cached_stems(audio_dir, mix, sep_model, lead) is not None
                stems = separate_vocals(mix, audio_dir, model=sep_model, lead=lead, progress=sep_cb)
        finally:
            release_separators()
            _free_torch()
        manifest = read_manifest(audio_dir) or {}
        if vocals is not None:              # offset / level / length of the supplied stem (import_vocals)
            warnings.extend(manifest.get("warnings") or [])
        report["separation"] = {"model": manifest.get("model"), "lead_model": manifest.get("lead_model"),
                                "reused": reused, "derived": manifest.get("derived"),
                                "subtypes": manifest.get("subtypes"), "timings": manifest.get("timings"),
                                "vocals_check": manifest.get("check")}
        timings[stage] = time.perf_counter() - t0
        prog(stage, 100, "reused cached stems" if reused else "stems ready")

        # 4. align -------------------------------------------------------------------------------
        stage, t0 = "align", time.perf_counter()
        prog(stage, 0, "loading vocal stem")
        y = load_mono(stems["vocals"], config.ANALYSIS_SR)
        align_report = _align(lines, y, config.ANALYSIS_SR, refine, prog.stage(stage))
        del y
        _free_torch()
        _plain_numbers(lines)
        _check_alignment(lines, warnings)
        timings[stage] = time.perf_counter() - t0
        prog(stage, 100, f"aligned with {getattr(align_report, 'method', '?')}")

        # 5. finalize ----------------------------------------------------------------------------
        stage, t0 = "finalize", time.perf_counter()
        prog(stage, 0, "estimating tuning")
        tuning = estimate_tuning(mix)
        confidences = list(getattr(align_report, "line_confidence", []) or []) or None
        rows = _line_rows(lines, confidences)
        confs = [r["confidence"] for r in rows if r["confidence"] is not None]
        flagged = sorted(set(getattr(align_report, "flagged", []) or [])
                         | {r["idx"] for r in rows if r["needs_review"]})
        align_info = {"method": getattr(align_report, "method", None), "flagged": flagged,
                      "timings": getattr(align_report, "timings", {}), "details": getattr(align_report, "details", {})}
        stems_rel = {"mix": "audio/mix.wav", **{k: Path(p).relative_to(work).as_posix() for k, p in stems.items()}}
        timings[stage] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - t_start
        rounded = {k: round(v, 3) for k, v in timings.items()}
        if style is None:           # read now: the style may have been changed while this re-prep ran
            style = _existing_style(sdir) if staged else "classic"
        meta = SongMeta(
            id=sid, title=title, artist=artist, default_lang=default_lang, duration=round(dur, 6),
            tuning=tuning, stems=stems_rel, style=style,
            prep={"created": created, "version": __version__, "options": options,
                  "models": {"separation": manifest.get("model"), "lead": manifest.get("lead_model"),
                             "align": align_info["method"]},
                  "languages": languages, "align_stem": "vocals", "tuning_params": TUNING_PARAMS,
                  "packages": _versions(),
                  "timings": rounded})
        save_song(Song(meta=meta, lines=lines), work)
        report.update(status="ok", tuning=tuning, stems=stems_rel, timings=rounded, align=align_info,
                      summary={"lines": len(lines), "acoustic_units": n_units, "languages": languages,
                               "flagged": len(flagged),
                               "mean_confidence": round(float(np.mean(confs)), 4) if confs else None,
                               "min_confidence": round(float(np.min(confs)), 4) if confs else None},
                      lines=rows)
        _write_json(work / REPORT_FILE, report)
        # Feature caches are keyed by each stem's size and mtime (features/extractor.py), so this is only
        # housekeeping: drop caches of stems that were just replaced.
        if staged:
            _commit(work, sdir, drop_cache=not reused)
        elif not reused:
            shutil.rmtree(sdir / "cache", ignore_errors=True)
        prog(stage, 100, f"done: {sdir}")
        return sdir
    except BaseException as e:
        if fresh_dir and not _has_stems(audio_dir):
            shutil.rmtree(sdir, ignore_errors=True)     # nothing worth a retry: free the id again
            raise
        report.update(status="cancelled" if isinstance(e, KeyboardInterrupt) else "failed", stage=stage,
                      error=f"{type(e).__name__}: {e}", timings={k: round(v, 3) for k, v in timings.items()})
        if staged:
            report["package"] = "previous version kept"
        try:
            _write_json(sdir / (FAILED_REPORT_FILE if staged else REPORT_FILE), report)
        except OSError:
            pass
        raise
    finally:
        if staged:
            shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lyricprompter prep",
        description="Separate, parse and align a song into songs/<id>/. Prints PROGRESS lines, then the "
                    "song directory.")
    p.add_argument("--audio", required=True, type=Path, help="original song audio (any ffmpeg-readable format)")
    p.add_argument("--lyrics", required=True, help="lyrics text file (UTF-8), or '-' for stdin")
    p.add_argument("--title", required=True)
    p.add_argument("--artist", default="")
    p.add_argument("--lang", default="auto", choices=LANGS, help="default lyrics language")
    p.add_argument("--id", dest="song_id", default=None, help="song id (default: slug of the title, with -2, -3... if taken; "
                        "pass an existing id to re-prepare that song)")
    p.add_argument("--songs-dir", type=Path, default=None, help=f"songs root (default: {config.SONGS_DIR})")
    p.add_argument("--sep-model", default=DEFAULT_SEP_MODEL, help="audio-separator model filename")
    p.add_argument("--lead", action="store_true", help=f"also extract the lead vocal ({LEAD_SEP_MODEL})")
    p.add_argument("--refine", default="auto", choices=REFINE_MODES, help="Qwen3 windowed refinement")
    p.add_argument("--vocals", type=Path, default=None,
                   help="pre-separated vocal stem of this mix (skips separation)")
    p.add_argument("--style", default=None, choices=STYLES,
                   help="live tracking style: classic (flexible tempo, follows every syllable) or pop (steady "
                        "band tempo, keeps moving when a line is not sung); default: keep the style of a "
                        "re-prepared song, else classic")
    return p


def _print_progress(stage: str, pct: float, msg: str) -> None:
    msg = " ".join(str(msg).split())
    print(f"PROGRESS {stage} {round(pct)} {msg}", flush=True)


def _on_sigterm(signum: int, frame: object) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)     # a second SIGTERM terminates outright
    raise Terminated()


def cli(argv: list[str]) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:   # argparse: --help -> 0, usage error -> 2
        return int(e.code or 0)
    try:
        previous = signal.signal(signal.SIGTERM, _on_sigterm)
    except ValueError:        # not the main thread: leave signals alone
        previous = None
    try:
        if args.lyrics == "-":
            lyrics = sys.stdin.read()
        else:
            lyrics = Path(args.lyrics).read_text(encoding="utf-8-sig")
        sdir = prepare_song(args.audio, lyrics, title=args.title, artist=args.artist, default_lang=args.lang,
                            song_id=args.song_id, songs_root=args.songs_dir, sep_model=args.sep_model,
                            lead=args.lead, refine=args.refine, vocals=args.vocals, style=args.style,
                            progress=_print_progress)
        print(Path(sdir).resolve(), flush=True)
        return 0
    except Terminated:
        print("error: terminated", file=sys.stderr)
        return 128 + signal.SIGTERM
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001  (CLI boundary: report, don't dump a traceback)
        if os.environ.get("LYRICPROMPTER_DEBUG"):
            traceback.print_exc()
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
