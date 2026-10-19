"""`lyricprompter eval-align`: offline lyrics-alignment accuracy on the evaluation datasets.

GT tokens (words / syllables / lines) are matched to our display syllables by character-level sequence
alignment of their normalised text (difflib), so any granularity and tokenisation works. A GT onset maps to
the start of the syllable containing its first matched character (linearly interpolated inside the
syllable when the GT token starts mid-syllable); a GT offset maps to the end of the syllable containing its
last character.

Metrics per dataset: AAE, MedAE, mean signed error (bias), p90, PCO@0.3/0.5/1.0 s on onsets; the same for
offsets when the GT has them; line-start error stats; and recall / precision of `needs_review` for lines
whose worst onset error exceeds 1 s.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from .. import config
from ..text.model import Line
from .datasets import AlignDataset, dataset_names, load_dataset, norm_chars

EMISSION_CACHE = config.CACHE_DIR / "eval" / "emissions"
EMISSION_VERSION = 1                 # bump when prep.align.compute_emission changes its output
TOLS = (0.3, 0.5, 1.0)
BIG_ERR_S = 1.0


# ----------------------------------------------------------------------------------------- matching / metrics
@dataclass
class CharMap:
    chars: str
    start: list[float]               # onset time of each character (interpolated inside its syllable)
    end: list[float]                 # end time of the syllable containing the character
    line: list[int]


def predicted_chars(lines: list[Line]) -> CharMap:
    chars, st, en, li = [], [], [], []
    for ln in lines:
        for syl in ln.syllables():
            c = norm_chars(syl.text)
            if not c or syl.start is None:
                continue
            s, e = float(syl.start), float(syl.end if syl.end is not None else syl.start)
            for k, ch in enumerate(c):
                chars.append(ch)
                st.append(s + (e - s) * k / len(c))
                en.append(e)
                li.append(ln.idx)
    return CharMap("".join(chars), st, en, li)


@dataclass
class Matched:
    gt_t: np.ndarray
    pred_t: np.ndarray
    gt_end: np.ndarray               # NaN where the GT has no offset
    pred_end: np.ndarray
    gt_line: np.ndarray
    pred_line: np.ndarray
    backing: np.ndarray              # GT token is a backing vocal
    n_gt: int


def match_tokens(ds: AlignDataset, lines: list[Line]) -> Matched:
    pm = predicted_chars(lines)
    gchars = [norm_chars(g.text) for g in ds.gt]
    seq = "".join(gchars)
    m: dict[int, int] = {}
    for a, b, n in SequenceMatcher(None, seq, pm.chars, autojunk=False).get_matching_blocks():
        for k in range(n):
            m[a + k] = b + k
    rows = []
    pos = 0
    for g, c in zip(ds.gt, gchars):
        idx = [m[pos + k] for k in range(len(c)) if pos + k in m]
        pos += len(c)
        if not idx:
            continue
        first, last = idx[0], idx[-1]
        rows.append((g.t, pm.start[first], np.nan if g.end is None else g.end, pm.end[last], g.line,
                     pm.line[first], float(g.backing)))
    a = np.array(rows, dtype=float).reshape(-1, 7)
    return Matched(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4].astype(int), a[:, 5].astype(int),
                   a[:, 6].astype(bool), len(ds.gt))


def error_stats(err: np.ndarray) -> dict:
    err = err[~np.isnan(err)]
    if len(err) == 0:
        return {"n": 0}
    ab = np.abs(err)
    return {"n": len(err), "AAE": round(float(ab.mean()), 3), "MedAE": round(float(np.median(ab)), 3),
            "bias": round(float(np.mean(err)), 3), "med_signed": round(float(np.median(err)), 3),
            "p90": round(float(np.percentile(ab, 90)), 3), "max": round(float(ab.max()), 3),
            **{f"PCO@{t}": round(float(np.mean(ab <= t)), 3) for t in TOLS}}


def evaluate(ds: AlignDataset, lines: list[Line]) -> dict:
    mt = match_tokens(ds, lines)
    err = mt.pred_t - mt.gt_t
    out = {"granularity": ds.granularity, "n_gt": mt.n_gt, "n_matched": len(err),
           "onset": error_stats(err)}
    if mt.backing.any():
        out["onset_lead"] = error_stats(err[~mt.backing])
        out["onset_backing"] = error_stats(err[mt.backing])
    if np.any(~np.isnan(mt.gt_end)):
        out["offset"] = error_stats(mt.pred_end - mt.gt_end)
    # line starts: first matched token of every GT line vs. start of the predicted line it maps to
    by_line = {ln.idx: ln for ln in lines}
    ls = []
    for gl in np.unique(mt.gt_line):
        k = int(np.flatnonzero(mt.gt_line == gl)[0])
        pl = by_line.get(int(mt.pred_line[k]))
        if pl is not None and pl.start is not None:
            ls.append(pl.start - mt.gt_t[k])
    out["line_start"] = error_stats(np.array(ls, dtype=float))
    # review recall: predicted lines whose worst onset error is big
    worst: dict[int, float] = {}
    for pl, e in zip(mt.pred_line, np.abs(err)):
        worst[int(pl)] = max(worst.get(int(pl), 0.0), float(e))
    bad = {li for li, e in worst.items() if e > BIG_ERR_S}
    flagged = {ln.idx for ln in lines if ln.needs_review}
    out["review"] = {"n_lines": len(worst), "n_bad": len(bad), "n_flagged": len(flagged),
                     "n_caught": len(bad & flagged),
                     "recall": round(len(bad & flagged) / len(bad), 3) if bad else None,
                     "precision": round(len(bad & flagged) / len(flagged), 3) if flagged else None}
    out["per_line"] = [{"line": li, "conf": by_line[li].confidence, "worst_err": round(e, 3),
                        "flagged": by_line[li].needs_review} for li, e in sorted(worst.items())]
    return out


# ----------------------------------------------------------------------------------------- configs
@dataclass
class EvalConfig:
    name: str
    options: dict = field(default_factory=dict)          # AlignOptions overrides
    refine: str = "none"                                 # none | qwen (flagged lines) | all
    refine_options: dict = field(default_factory=dict)   # RefineOptions overrides


def default_configs(refine: str) -> list[EvalConfig]:
    cfgs = [EvalConfig("mms_gate")]
    if refine in ("qwen", "auto"):
        cfgs.append(EvalConfig("hybrid", refine="qwen"))
    return cfgs


def select_configs(refine: str, ablations: bool, qwen_available: Callable[[], bool] | None = None
                   ) -> list[EvalConfig]:
    """Configs for the CLI: Qwen configs only with refine qwen / auto (auto: only if the model is cached)."""
    configs = default_configs(refine)
    if ablations:
        configs += [c for c in ABLATIONS if c.name not in {x.name for x in configs}]
    if refine == "auto":
        if qwen_available is None:
            from ..prep.refine import qwen_cached as qwen_available
        if not qwen_available():
            print("Qwen3-ForcedAligner not cached locally: skipping hybrid configs")
            refine = "none"
    return configs if refine != "none" else [c for c in configs if c.refine == "none"]


ABLATIONS = [
    EvalConfig("mms_nogate", {"gate": False}),
    EvalConfig("gate_fixed_-35dB", {"gate_noise_margin_db": None}),
    EvalConfig("snap_none", {"snap_onsets": "none"}),
    EvalConfig("snap_all", {"snap_onsets": "all"}),
    EvalConfig("snap_on_gate_mask_1s", {"snap_rel_db": None, "max_snap_s": 1.0}),
    EvalConfig("backing_included", {"include_backing": True}),
    EvalConfig("backing_no_star", {"star_for_backing": False}),
    EvalConfig("stars_between_words", {"star_between_words": True}),
    EvalConfig("stars_between_words_pen1", {"star_between_words": True, "star_penalty": 1.0}),
    EvalConfig("no_stars_between_lines", {"star_between_lines": False}),
    EvalConfig("ends_raw", {"end_mode": "raw"}),
    EvalConfig("ends_close", {"end_mode": "close"}),
    EvalConfig("ends_sustain_gate_mask", {"end_rel_db": None}),
    EvalConfig("review_0.10", {"review_threshold": 0.10}),
    EvalConfig("hybrid", refine="qwen"),
    EvalConfig("hybrid_refine_0.15", {"refine_threshold": 0.15}, refine="qwen"),
    EvalConfig("hybrid_no_anchors", refine="qwen", refine_options={"anchors": False}),
    EvalConfig("hybrid_win60", refine="qwen", refine_options={"max_win_s": 60.0}),
    EvalConfig("hybrid_adopt_any_conf", refine="qwen", refine_options={"min_qwen_conf": 0.0}),
    EvalConfig("qwen_windows_all", refine="all"),
]


# ----------------------------------------------------------------------------------------- runner
def _parser():
    try:
        from ..text import parse_lyrics
    except ImportError as e:                                # module A not available yet
        raise RuntimeError("lyricprompter.text.parse_lyrics is required for eval-align") from e
    return parse_lyrics


def _load_vocals(path: Path) -> np.ndarray:
    from ..audio_io import load_mono
    return load_mono(Path(path), config.ANALYSIS_SR)


def emission_key(vocals: Path, win_s: float, ctx_s: float) -> str:
    """Cache key: vocals file (path, size, mtime), window, model, torch / torchaudio and emission code version."""
    import torch
    import torchaudio

    from ..prep.align import MMS_MODEL_NAME
    st = Path(vocals).stat()
    parts = (Path(vocals).resolve(), st.st_size, st.st_mtime_ns, win_s, ctx_s, MMS_MODEL_NAME, EMISSION_VERSION,
             torch.__version__, torchaudio.__version__)
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def cached_emission(ds: AlignDataset, y: np.ndarray, win_s: float, ctx_s: float, use_cache: bool = True):
    """MMS_FA emission for a dataset, cached on disk (see emission_key)."""
    import torch

    from ..prep.align import Emission, compute_emission, mms_labels
    key = emission_key(Path(ds.vocals), win_s, ctx_s)
    path = EMISSION_CACHE / f"{ds.name}-{key}.pt"
    if use_cache and path.is_file():
        return Emission(log_probs=torch.load(path), labels=mms_labels())
    em = compute_emission(y, config.ANALYSIS_SR, win_s=win_s, ctx_s=ctx_s)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(em.log_probs, path)
    return em


def run_eval(names: list[str], configs: list[EvalConfig], *, use_cache: bool = True,
             local_only: bool = True, log=print) -> dict:
    """Evaluate every config on every dataset (emissions computed once per dataset; MMS and Qwen are never
    loaded at the same time). `local_only=False` lets the Qwen configs download the model."""
    from ..prep import align as A
    from ..prep import refine as R
    parse_lyrics = _parser()
    results: dict = {"datasets": {}, "configs": [asdict(c) for c in configs]}
    prepared = []
    for name in names:                                       # pass 1: emissions (MMS loaded once)
        ds = load_dataset(name)
        y = _load_vocals(ds.vocals)
        t0 = time.perf_counter()
        em = cached_emission(ds, y, A.AlignOptions().win_s, A.AlignOptions().ctx_s, use_cache)
        lines0 = parse_lyrics(ds.lyrics, ds.default_lang)
        prepared.append((ds, y, em, lines0, round(time.perf_counter() - t0, 2)))
        log(f"[{name}] {len(y) / config.ANALYSIS_SR:.1f}s audio, {len(lines0)} lines, {len(ds.gt)} GT "
            f"{ds.granularity}s, emission {prepared[-1][4]}s")
    A.release_models()
    for ds, y, em, lines0, t_em in prepared:                 # pass 2: Viterbi variants (+ Qwen)
        entry = {"language": ds.language, "granularity": ds.granularity, "n_gt": len(ds.gt),
                 "n_lines": len(lines0), "duration": round(len(y) / config.ANALYSIS_SR, 2), "notes": ds.notes,
                 "emission_s": t_em, "configs": {}}
        for cfg in configs:
            opts = replace(A.AlignOptions(), **cfg.options)
            ropts = replace(R.RefineOptions(), **cfg.refine_options)
            masks = A.vocal_masks(y, em.n_frames, config.ANALYSIS_SR, opts)
            lines = copy.deepcopy(lines0)
            t0 = time.perf_counter()
            report, raw = A.align_with_emission(lines, em, masks=masks, options=opts)
            if cfg.refine != "none":
                targets = (A.refine_targets([ln.confidence for ln in lines], opts.refine_threshold)
                           if cfg.refine == "qwen" else [i for i, ln in enumerate(lines) if ln.syllables()])
                if targets:
                    res = R.refine_lines(lines, y, targets, options=ropts, local_only=local_only,
                                         active=masks.active, active_frame_s=A.FRAME_S)
                    A.apply_refinement(lines, raw, res, report, masks=masks, options=opts)
            dt = time.perf_counter() - t0
            ev = evaluate(ds, lines)
            ev.update(method=report.method, flagged=report.flagged, align_s=round(dt, 3))
            if "refine" in report.details:
                ev["refine"] = {k: v for k, v in report.details["refine"].items() if k != "windows"}
            entry["configs"][cfg.name] = ev
            o = ev["onset"]
            rv = ev["review"]
            log(f"  {ds.name:22s} {cfg.name:24s} AAE={o.get('AAE')} Med={o.get('MedAE')} "
                f"bias={o.get('bias')} PCO.3={o.get('PCO@0.3')} PCO1={o.get('PCO@1.0')} "
                f"review={rv['n_caught']}/{rv['n_bad']} (flag {rv['n_flagged']}) {dt:.1f}s")
        results["datasets"][ds.name] = entry
    R.release_qwen()
    results["summary"] = summarize(results)
    return results


def summarize(results: dict) -> dict:
    """Per config: means over datasets and pooled-onset numbers."""
    out: dict = {}
    for name, entry in results["datasets"].items():
        for cfg, ev in entry["configs"].items():
            s = out.setdefault(cfg, {"datasets": {}, "AAE": [], "MedAE": [], "PCO@0.3": [], "PCO@1.0": []})
            o = ev["onset"]
            keys = ("AAE", "MedAE", "bias", "PCO@0.3", "PCO@0.5", "PCO@1.0")
            s["datasets"][name] = {k: o.get(k) for k in keys}
            for k in ("AAE", "MedAE", "PCO@0.3", "PCO@1.0"):
                if o.get(k) is not None:
                    s[k].append(o[k])
    for s in out.values():
        for k in ("AAE", "MedAE", "PCO@0.3", "PCO@1.0"):
            s[f"mean_{k}"] = round(float(np.mean(s.pop(k))), 3) if s.get(k) else None
    return out


def print_table(results: dict, log=print) -> None:
    cfgs = [c["name"] for c in results["configs"]]
    log(f"{'dataset':22s} {'config':24s} {'AAE':>6s} {'MedAE':>6s} {'bias':>6s} {'P@.3':>5s} {'P@.5':>5s} "
        f"{'P@1':>5s} {'lineMed':>7s} {'rev':>7s}")
    for name, entry in results["datasets"].items():
        for c in cfgs:
            ev = entry["configs"].get(c)
            if not ev:
                continue
            o, ls, rv = ev["onset"], ev["line_start"], ev["review"]
            log(f"{name:22s} {c:24s} {o.get('AAE', 0):6.3f} {o.get('MedAE', 0):6.3f} {o.get('bias', 0):6.3f} "
                f"{o.get('PCO@0.3', 0):5.2f} {o.get('PCO@0.5', 0):5.2f} {o.get('PCO@1.0', 0):5.2f} "
                f"{ls.get('MedAE', 0):7.3f} {rv['n_caught']:>3d}/{rv['n_bad']:<3d}")


def cli(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lyricprompter eval-align", description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default="all",
                    help="comma list of datasets or groups (all, ref, jamendo, tts)")
    ap.add_argument("--refine", default="none", choices=["none", "qwen", "auto"],
                    help="also evaluate the MMS+Qwen hybrid on flagged lines (qwen: may download the model, "
                         "auto: only if cached); with 'none' no Qwen config runs")
    ap.add_argument("--ablations", action="store_true",
                    help="run the ablation set (gate, stars, backing, ends; Qwen variants with --refine)")
    ap.add_argument("--json", type=Path, help="write raw results JSON here")
    ap.add_argument("--no-cache", action="store_true", help="recompute MMS emissions")
    args = ap.parse_args(argv)
    configs = select_configs(args.refine, args.ablations)
    results = run_eval(dataset_names(args.datasets), configs, use_cache=not args.no_cache,
                       local_only=args.refine != "qwen")
    print()
    print_table(results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0
