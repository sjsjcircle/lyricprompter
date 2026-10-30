"""Offline cue firing (`oscsender simulate`) and its accuracy on synthetic live takes (`oscsender eval`).

`simulate_cues` follows a recording with exactly the tracker the live session builds (`session.
build_tracking`) and drives the cue scheduler on a **virtual clock = live time**: the audio goes through in
blocks of `block` samples (20 ms by default, like a sound card), a block that completes a feature frame is
one tracker update at the block's end time t_k, and between t_k and the next block's end every cue fires at
its exact due time, the scheduler's documented recipe:

    after the update at t_k:  while (nd := next_due_time()) is not None and nd <= t_next:
                                  fires += due(max(nd, t_k))

So a fire's `at` is the live time it would have been sent at in real time, with the realtime runner's
sub-frame timing and none of its thread jitter. There is no audio-input latency on this clock, so the
tracker's input latency compensation defaults to 0 (whatever the console's setting).

`track_offline` (tracker only) and `fire_offline` (scheduler only) are the two halves: one tracking run
can be scheduled with several settings (the scheduler never influences the tracker).

`eval` makes synthetic live takes of a project's reference (synth.make_synthetic: piecewise tempo, noise,
reverb, EQ, a noise lead-in) with an exact live <-> reference map, fires the project's cues with every
target latency 0, and reports per cue: error = fire live time - ground-truth live time of the cue time.
Cues at the start position (they fire at the start by design) and after the reference end are not scored.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from .. import project
from ..audio_io import load_mono
from ..config import ANALYSIS_SR, MODES, PROJECTS_DIR, STYLES
from ..cues import Cue, parse_cues
from ..scheduler import CueScheduler, Fire, FireRecord, SchedulerParams, TrackPoint
from ..settings import Settings, latency_s, load_settings, merge_declared, settings_path
from ..songfile import stem_path
from .session import SessionDeps, build_tracking, reference_stem, track_point

TOLERANCES_MS = (50, 100, 200)
DUE_LOOP_MAX = 100_000           # cues fired between two blocks at most (a safety bound, never reached)


# --------------------------------------------------------------------------- offline tracking + firing


@dataclass
class TrackRun:
    """One offline tracking run: the start point (clock 0; the scheduler starts it with `start_time` as
    the boundary) and, per input block, its end time (live s: the virtual clock) with the tracker update it
    produced (None: the block completed no frame)."""
    start: TrackPoint
    steps: list[tuple[float, TrackPoint | None]]
    block_s: float
    duration: float              # live seconds of audio
    mode: str
    style: str
    stem: str
    start_time: float
    transpose: int = 0           # the tracker's key at the end
    key_auto: bool = False
    meta: project.ProjectMeta | None = None
    wall_s: float = 0.0          # processing time


def _key_preset(tracker: Any, transpose: int | None, auto_key: bool) -> None:
    """lyricprompter's rules: a preset key turns automatic detection off unless `auto_key` (then it only
    seeds the search); no preset and no detection rehearses at the original key."""
    if transpose is not None or not auto_key:
        tracker.set_transpose(int(transpose or 0))
        if auto_key:
            tracker.set_auto_key(True)


def track_offline(project_dir: Path, live_y16k: np.ndarray, *, mode: str = "mix", style: str | None = None,
                  start_time: float = 0.0, transpose: int | None = None, auto_key: bool = True,
                  input_latency_ms: float = 0.0, block: int = 320, params: Any | None = None,
                  deps: SessionDeps | None = None) -> TrackRun:
    """Run the session's tracker over 16 kHz mono audio in `block`-sample blocks (see the module doc)."""
    if block <= 0:
        raise ValueError(f"block must be > 0 samples, got {block}")
    t_wall = time.perf_counter()
    trk = build_tracking(Path(project_dir), mode, style, params, deps)
    tracker, ext, fps = trk.tracker, trk.extractor, trk.fps
    y = np.ascontiguousarray(np.asarray(live_y16k, dtype=np.float32).reshape(-1))
    start_time = max(0.0, float(start_time or 0.0))
    _key_preset(tracker, transpose, auto_key)
    tracker.set_latency(float(input_latency_ms) / 1000.0)
    tracker.start(start_time)
    start = track_point(tracker, mode, fps, tracker.state, 0.0)
    steps: list[tuple[float, TrackPoint | None]] = []
    for i in range(0, len(y), block):
        chunk = y[i:i + block]
        frames = ext.push(chunk)
        t_k = (i + len(chunk)) / ANALYSIS_SR
        if len(frames):
            st = tracker.push(frames)
            # the point is stamped at the end of its newest frame (live_time), as the session does; it is
            # observed at the block end t_k
            steps.append((t_k, track_point(tracker, mode, fps, st, min(st.live_time, t_k))))
        else:
            steps.append((t_k, None))
    st = tracker.state
    return TrackRun(start=start, steps=steps, block_s=block / ANALYSIS_SR, duration=len(y) / ANALYSIS_SR,
                    mode=mode, style=trk.style, stem=trk.stem, start_time=start_time,
                    transpose=int(getattr(st, "transpose", 0)), key_auto=bool(getattr(st, "key_auto", False)),
                    meta=trk.meta, wall_s=time.perf_counter() - t_wall)


def _fire_dict(f: Fire, default_target: str) -> dict:
    c = f.cue
    return {"cue_id": c.id, "cue_time": c.time, "target": c.target or default_target, "address": c.address,
            "message": c.message, "at": f.at, "ref_time": f.ref_time, "reason": f.reason, "send": f.send,
            "ref_late_ms": f.ref_late * 1000.0}


def fire_offline(run: TrackRun, cues: Sequence[Cue], settings: Settings | None = None) -> list[dict]:
    """Schedule `cues` over a tracking run on its virtual clock; one dict per fire, in firing order
    (cue_id, cue_time, target, address, message, at (live s), ref_time, reason, send, ref_late_ms)."""
    s = settings or Settings()
    sched = CueScheduler(cues, partial(latency_s, s),
                         SchedulerParams(catchup_s=s.catchup_s, skip_policy=s.skip_policy))
    sched.set_default_target(s.default_target)
    fires: list[Fire] = list(sched.start(run.start, boundary=run.start_time))

    def drain(t_k: float, t_next: float) -> None:
        for _ in range(DUE_LOOP_MAX):
            nd = sched.next_due_time()
            if nd is None or nd > t_next:
                return
            got = sched.due(max(nd, t_k))
            if not got:                       # next_due_time and due disagree: never spin
                return
            fires.extend(got)

    drain(0.0, run.block_s)
    for t_k, pt in run.steps:
        if pt is not None:
            fires.extend(sched.update(pt))
        drain(t_k, t_k + run.block_s)
    return [_fire_dict(f, s.default_target) for f in fires]


def simulate_cues(project_dir: Path, live_y16k: np.ndarray, *, mode: str = "mix", style: str | None = None,
                  settings: Settings | None = None, cues: Sequence[Cue] | None = None, start_time: float = 0.0,
                  transpose: int | None = None, auto_key: bool = True, input_latency_ms: float = 0.0,
                  block: int = 320, params: Any | None = None, deps: SessionDeps | None = None) -> list[dict]:
    """Fire a project's cues while following `live_y16k` (16 kHz mono) offline (see the module doc).

    `cues` None: the project's cue file. `settings` None: the defaults plus the targets the cue file
    declares (their latencies). Returns one dict per fire (see `fire_offline`)."""
    pdir = Path(project_dir)
    if cues is None or settings is None:
        sheet = parse_cues(project.read_cues(pdir))
        if cues is None:
            cues = sheet.cues
        if settings is None:
            settings, _ = merge_declared(Settings(), sheet.targets)
    run = track_offline(pdir, live_y16k, mode=mode, style=style, start_time=start_time, transpose=transpose,
                        auto_key=auto_key, input_latency_ms=input_latency_ms, block=block, params=params,
                        deps=deps)
    return fire_offline(run, cues, settings)


# --------------------------------------------------------------------------- evaluation


def first_fires(fires: Sequence[dict]) -> dict[int, dict]:
    """cue id -> its first sent fire."""
    out: dict[int, dict] = {}
    for f in fires:
        if f["send"] and f["cue_id"] not in out:
            out[f["cue_id"]] = f
    return out


def cue_errors(cues: Sequence[Cue], fires: Sequence[dict], gt_live: dict[int, float]) -> dict:
    """Per scored cue (the ids in `gt_live`): its first sent fire's `at` minus the ground-truth live time.
    Returns {"errors_ms": {id: ms}, "missed": [ids never sent], "refired": extra sends of scored cues}."""
    first = first_fires(fires)
    sends: dict[int, int] = {}
    for f in fires:
        if f["send"]:
            sends[f["cue_id"]] = sends.get(f["cue_id"], 0) + 1
    errors, missed = {}, []
    for c in cues:
        if c.id not in gt_live:
            continue
        f = first.get(c.id)
        if f is None:
            missed.append(c.id)
        else:
            errors[c.id] = (f["at"] - gt_live[c.id]) * 1000.0
    refired = sum(n - 1 for cid, n in sends.items() if cid in gt_live and n > 1)
    return {"errors_ms": errors, "missed": missed, "refired": refired}


def error_stats(errors_ms: Sequence[float], n_cues: int, missed: int, refired: int = 0) -> dict:
    """Summary of signed timing errors (ms): median |err|, p90, max, signed median (bias), % within."""
    e = np.asarray(list(errors_ms), dtype=np.float64)
    a = np.abs(e)
    out: dict[str, Any] = {"cues": int(n_cues), "fired": len(e), "missed": int(missed),
                           "refired": int(refired)}
    if len(e):
        out.update({"median_ms": float(np.median(a)), "p90_ms": float(np.percentile(a, 90)),
                    "max_ms": float(a.max()), "bias_ms": float(np.median(e))})
        for tol in TOLERANCES_MS:
            out[f"within_{tol}"] = float(np.mean(a <= tol) * 100.0)
    else:
        out.update({"median_ms": None, "p90_ms": None, "max_ms": None, "bias_ms": None,
                    **{f"within_{tol}": None for tol in TOLERANCES_MS}})
    return out


def eval_project(project_dir: Path, *, seeds: Sequence[int] = (0, 1, 2), mode: str = "mix",
                 style: str | None = None, tempo: tuple[float, float] = (0.8, 1.25), transpose: int = 0,
                 key_preset: bool = False, cues: Sequence[Cue] | None = None, block: int = 320,
                 spec_overrides: dict | None = None, progress=None) -> dict:
    """Synthetic live takes of the project's reference (one per seed), cue firing with every latency 0,
    errors against the exact ground truth. Returns {"project", "mode", "style", "takes": [...],
    "overall": {...}} (see `error_stats`; each take also has seed, live_s, wall_s, key)."""
    from ..synth import SynthSpec, make_synthetic, ref_to_live

    pdir = Path(project_dir)
    stem = reference_stem(pdir, mode)
    if cues is None:
        cues = parse_cues(project.read_cues(pdir)).cues
    y_ref = load_mono(stem_path(pdir, stem), ANALYSIS_SR)
    ref_dur = len(y_ref) / ANALYSIS_SR
    scored = [c for c in cues if 0.0 < c.time <= ref_dur]
    settings = Settings()                         # every target latency 0, input latency 0
    takes, pooled, n_all, missed_all, refired_all = [], [], 0, 0, 0
    style_used = style
    for seed in seeds:
        spec = SynthSpec(seed=int(seed), tempo=(float(tempo[0]), float(tempo[1])), transpose=int(transpose),
                         **(spec_overrides or {}))
        live, live_pts, ref_pts = make_synthetic(y_ref, ANALYSIS_SR, spec)
        run = track_offline(pdir, live, mode=mode, style=style, block=block,
                            transpose=int(transpose) if key_preset else None, auto_key=True)
        style_used = run.style
        fires = fire_offline(run, cues, settings)
        gt = {c.id: float(ref_to_live(np.array([c.time]), live_pts, ref_pts)[0]) for c in scored}
        res = cue_errors(cues, fires, gt)
        stats = error_stats(list(res["errors_ms"].values()), len(scored), len(res["missed"]), res["refired"])
        take = {"seed": int(seed), "live_s": round(run.duration, 3), "wall_s": round(run.wall_s, 2),
                "key": run.transpose, **stats, "missed_ids": res["missed"],
                "errors_ms": {str(k): round(v, 2) for k, v in res["errors_ms"].items()}}
        takes.append(take)
        pooled += list(res["errors_ms"].values())
        n_all += len(scored)
        missed_all += len(res["missed"])
        refired_all += res["refired"]
        if progress is not None:
            progress(take)
    return {"project": pdir.name, "mode": mode, "style": style_used, "tempo": list(tempo),
            "transpose": int(transpose), "excluded": len(cues) - len(scored), "takes": takes,
            "overall": error_stats(pooled, n_all, missed_all, refired_all)}


# --------------------------------------------------------------------------- cli helpers


def _fmt_t(t: float | None) -> str:
    if t is None or not math.isfinite(t):
        return "     --.---"
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):4d}:{s:06.3f}"


def _project_dir(ap: argparse.ArgumentParser, args: argparse.Namespace) -> Path:
    if args.project_dir is not None:
        pdir = Path(args.project_dir)
    else:
        try:
            pdir = project.project_dir(args.projects_dir, args.project)
        except ValueError as e:
            ap.error(str(e))
    if not (pdir / project.META_FILE).is_file():
        ap.error(f"not a project (no {project.META_FILE}): {pdir}")
    return pdir


def _add_project_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--project", help="project id under --projects-dir")
    g.add_argument("--project-dir", type=Path, help="project folder")
    ap.add_argument("--projects-dir", type=Path, default=PROJECTS_DIR)


def _console_settings(pdir: Path, sheet: Any) -> Settings:
    """The console's saved settings (next to the project) plus the targets the cue file declares."""
    s = load_settings(settings_path(pdir.parent))
    s, _ = merge_declared(s, sheet.targets)
    return s


def _fire_line(at: float | None, cue_time: float, ref: float, reason: str, target: str, message: str,
               note: str = "") -> str:
    return (f"live {_fmt_t(at)}  cue {_fmt_t(cue_time)}  ref {_fmt_t(ref)}  {reason:<8} {target:<12} "
            f"{message}{note}")


# --------------------------------------------------------------------------- simulate cli


def cli_simulate(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="oscsender simulate",
                                 description="Track a recording as if it were live and print the cues it fires")
    _add_project_args(ap)
    ap.add_argument("--file", type=Path, required=True, help="recording to play as the live input")
    ap.add_argument("--mode", choices=MODES, default="mix",
                    help="mix: the input is the full band; vocal: the vocal mic only (needs a vocals stem)")
    ap.add_argument("--style", choices=STYLES, default=None, help="tracking style (default: the project's)")
    ap.add_argument("--start-time", type=float, default=0.0, help="anchor the tracker at this reference time (s)")
    ap.add_argument("--transpose", type=int, default=None,
                    help="key preset: semitones the live take is above the reference (turns automatic key "
                         "detection off unless --auto-key is also given)")
    ap.add_argument("--auto-key", action=argparse.BooleanOptionalAction, default=None,
                    help="automatic key detection (default: on unless --transpose is given)")
    ap.add_argument("--latency-ms", type=float, default=None,
                    help="tracker input latency compensation (ms; default 0 offline, the console's setting "
                         "with --realtime)")
    ap.add_argument("--realtime", action="store_true",
                    help="play the file in real time through a live session (cue runner, wall clock)")
    ap.add_argument("--speed", type=float, default=1.0, help="pacing factor for --realtime")
    ap.add_argument("--monitor", action="store_true", help="also play the file (implies --realtime)")
    ap.add_argument("--send", action="store_true",
                    help="with --realtime: really send the OSC messages to the console's targets (else dry run)")
    args = ap.parse_args(argv)
    pdir = _project_dir(ap, args)
    if not args.file.is_file():
        ap.error(f"file not found: {args.file}")
    realtime = args.realtime or args.monitor
    if args.send and not realtime:
        ap.error("--send needs --realtime (an offline run is faster than the music)")
    try:
        reference_stem(pdir, args.mode)
    except (ValueError, FileNotFoundError) as e:
        ap.error(str(e))
    sheet = parse_cues(project.read_cues(pdir))
    for issue in sheet.issues:
        print(f"cues.txt line {issue.line}: {issue.severity}: {issue.message}", file=sys.stderr)
    settings = _console_settings(pdir, sheet)
    auto = args.auto_key if args.auto_key is not None else args.transpose is None
    if realtime:
        return _simulate_realtime(args, pdir, sheet.cues, settings, auto)

    t0 = time.perf_counter()
    live = load_mono(args.file, ANALYSIS_SR)
    run = track_offline(pdir, live, mode=args.mode, style=args.style, start_time=args.start_time,
                        transpose=args.transpose, auto_key=auto, input_latency_ms=args.latency_ms or 0.0)
    fires = fire_offline(run, sheet.cues, settings)
    print(f"project {pdir.name!r}: {len(sheet.cues)} cues, reference stem {run.stem!r}, {run.style} style, "
          f"{run.duration:.1f}s of live audio", flush=True)
    for f in fires:
        note = "" if f["send"] else "  (not sent)"
        print(_fire_line(f["at"], f["cue_time"], f["ref_time"], f["reason"], f["target"], f["message"], note))
    sent = {f["cue_id"] for f in fires if f["send"]}
    skipped = sum(1 for f in fires if f["reason"] == "skipped")
    never = [c.id for c in sheet.cues if c.id not in sent and not any(f["cue_id"] == c.id for f in fires)]
    print(f"done: {len(fires)} fires ({len(sent)} cues sent, {skipped} skipped, {len(never)} never reached) "
          f"in {time.perf_counter() - t0:.1f}s, final key {run.transpose:+d}"
          + (" (automatic detection)" if run.key_auto else ""))
    return 0


def _simulate_realtime(args: argparse.Namespace, pdir: Path, cues: list[Cue], settings: Settings,
                       auto: bool) -> int:
    from ..osc import OscSender
    from .audio_input import FileInput
    from .session import OscSession

    settings = settings.copy()
    settings.armed = bool(args.send)
    sender = OscSender()
    if args.send:
        for name, err in sender.set_targets(settings.targets).items():
            if err:
                print(f"target {name}: {err}", file=sys.stderr)

    def on_fire(rec: FireRecord) -> None:
        state = "sent" if rec.sent else ("dry run" if rec.dry_run else ("error" if rec.error else "not sent"))
        note = f"  ({state}{': ' + rec.error if rec.error else ''})"
        print(_fire_line(rec.live_time, rec.time, rec.ref_time, rec.reason, rec.target, rec.message, note),
              flush=True)

    source = FileInput(args.file, realtime=True, speed=args.speed, monitor=args.monitor)
    if args.latency_ms is not None:
        settings.input_latency_ms = float(args.latency_ms)
    sess = OscSession(pdir, args.mode, source, cues=cues, sender=sender, settings=settings, style=args.style,
                      start_time=args.start_time, on_fire=on_fire)
    print(f"project {sess.project_id!r}: {len(cues)} cues, reference stem {sess.stem!r}, {sess.style} style, "
          f"{'sending' if args.send else 'dry run'}", flush=True)
    if args.transpose is not None or not auto:
        sess.set_transpose(args.transpose or 0)
        if auto:
            sess.set_auto_key(True)
    sess.start()
    try:
        while not source.wait(0.25):
            pass
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    finally:
        sess.stop()
        sender.close()
    counts = sess.runner.counts()
    print(f"done: {counts['sent']} sent, {counts['dry']} dry run, {counts['skipped']} skipped, "
          f"{counts['errors']} errors" + (f", last error {sess.error}" if sess.error else ""))
    return 0 if sess.error is None else 1


# --------------------------------------------------------------------------- eval cli


def _parse_tempo(text: str) -> tuple[float, float]:
    try:
        lo, hi = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected LO,HI (e.g. 0.8,1.25), got {text!r}") from None
    if not (0.25 <= lo <= hi <= 4.0):
        raise argparse.ArgumentTypeError(f"tempo range must satisfy 0.25 <= LO <= HI <= 4, got {text!r}")
    return lo, hi


def _ms(v: float | None, width: int = 7) -> str:
    return f"{'--':>{width}}" if v is None else f"{v:{width}.1f}"


def _pct(v: float | None) -> str:
    return f"{'--':>6}" if v is None else f"{v:5.1f}%"


TABLE_HEADER = (f"{'take':<8}{'cues':>5}{'fired':>6}{'missed':>7}{'median':>8}{'p90':>8}{'max':>8}{'bias':>8}"
                f"{'<=50':>7}{'<=100':>7}{'<=200':>7}")


def _table_line(name: str, s: dict) -> str:
    return (f"{name:<8}{s['cues']:>5}{s['fired']:>6}{s['missed']:>7}{_ms(s['median_ms'], 8)}"
            f"{_ms(s['p90_ms'], 8)}{_ms(s['max_ms'], 8)}{_ms(s['bias_ms'], 8)} {_pct(s['within_50'])}"
            f"{_pct(s['within_100'])}{_pct(s['within_200'])}")


def cli_eval(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="oscsender eval",
                                 description="Measure cue timing accuracy on synthetic live takes of a project")
    _add_project_args(ap)
    ap.add_argument("--seeds", type=int, default=3, help="number of synthetic takes (seeds 0..N-1)")
    ap.add_argument("--mode", choices=MODES, default="mix")
    ap.add_argument("--style", choices=STYLES, default=None, help="tracking style (default: the project's)")
    ap.add_argument("--tempo", type=_parse_tempo, default=(0.8, 1.25),
                    help="tempo range of the takes, reference s per live s (LO,HI; default 0.8,1.25)")
    ap.add_argument("--transpose", type=int, default=0, help="the takes are this many semitones above the reference")
    ap.add_argument("--key-preset", action="store_true",
                    help="tell the tracker the key (--transpose) instead of letting it detect it")
    ap.add_argument("--json", action="store_true", help="print the results as JSON")
    args = ap.parse_args(argv)
    pdir = _project_dir(ap, args)
    if args.seeds < 1:
        ap.error("--seeds must be >= 1")
    try:
        reference_stem(pdir, args.mode)
    except (ValueError, FileNotFoundError) as e:
        ap.error(str(e))
    cues = parse_cues(project.read_cues(pdir)).cues
    if not cues:
        ap.error(f"project {pdir.name!r} has no cues to evaluate")

    def progress(take: dict) -> None:
        if not args.json:
            print(_table_line(f"seed {take['seed']}", take)
                  + f"   ({take['live_s']:.0f}s live, {take['wall_s']:.1f}s, key {take['key']:+d})", flush=True)

    if not args.json:
        print(f"project {pdir.name!r}: {len(cues)} cues, mode {args.mode}, tempo {args.tempo[0]:g}..{args.tempo[1]:g}"
              f", transpose {args.transpose:+d}; errors in ms (fire - ground truth)", flush=True)
        print(TABLE_HEADER, flush=True)
    res = eval_project(pdir, seeds=range(args.seeds), mode=args.mode, style=args.style, tempo=args.tempo,
                       transpose=args.transpose, key_preset=args.key_preset, cues=cues, progress=progress)
    if args.json:
        print(json.dumps(res, indent=1, allow_nan=False))
    else:
        print(_table_line("overall", res["overall"]))
        excluded = res["excluded"]
        if excluded:
            print(f"({excluded} cue{'' if excluded == 1 else 's'} at the start position or after the reference "
                  f"end not scored; style {res['style']})")
        else:
            print(f"(style {res['style']})")
    return 0
