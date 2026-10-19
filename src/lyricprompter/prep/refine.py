"""Windowed Qwen3-ForcedAligner-0.6B refinement of low-confidence lines (MMS + Qwen hybrid).

Qwen3-FA is accurate on <=30-60 s windows but drifts on whole songs, while gated MMS is globally robust.
So MMS gives the global placement and Qwen re-aligns only the flagged lines, inside the gap between their
trusted (high-confidence) neighbours. Units are the display syllables (a Japanese kanji group is one
unit), so Qwen times map 1:1 onto `Line.syllables()`.

Decision per refined line (picked on the evaluation data, docs/eval/ALIGNMENT.md):
- agree (median |Qwen - MMS| syllable-onset difference <= disagree_s): keep the MMS times (20 ms resolution
  beats Qwen's 80 ms) and clear the review flag: two independent aligners agree.
- disagree: adopt the Qwen times only when Qwen is confident (mean timestamp-class probability >=
  min_qwen_conf), its times stay inside the gap between the trusted neighbours and, when trusted neighbour
  lines fit in the window as anchors, Qwen reproduces their MMS times. The line keeps needing review.
- degenerate Qwen output never agrees and is never adopted (the line keeps needing review): the line
  collapsed (most onsets inside one 80 ms Qwen bin, or the whole line shorter than 50 ms per syllable),
  most onsets pinned to a window edge (Qwen clamps timestamps it cannot place to the window start / end),
  or most onsets away from vocal activity. Qwen's confidence stays ~0.15-0.19 on such output, so the
  confidence threshold alone does not catch it.

The HF processor's `language` only selects its word splitter (nagisa / soynlp / whitespace); the model
prompt itself is language-free. We pass our own units, and record the dominant language per window.
"""
from __future__ import annotations

import time
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from .. import config
from ..text.model import Line, Word

Progress = Callable[[str, float, str], None]

QWEN_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
QWEN_FILES = ("model.safetensors", "config.json", "processor_config.json", "tokenizer.json",
              "tokenizer_config.json", "chat_template.jinja")
QWEN_BIN_S = 0.08                    # Qwen3-FA timestamp resolution
SR = config.ANALYSIS_SR
LANG_NAMES = {"ko": "Korean", "ja": "Japanese", "en": "English"}


@dataclass
class RefineOptions:
    max_win_s: float = 30.0
    pad_s: float = 0.5
    anchors: bool = True             # add trusted neighbour lines to the window as context + sanity check
    anchor_tol_s: float = 1.0        # median |Qwen-MMS| on anchor syllables above this -> window rejected
    accept_tol_s: float = 0.5        # Qwen times must lie within [lo - tol, hi + tol] of the trusted gap
    disagree_s: float = 0.5          # |Qwen-MMS| onset difference that counts as disagreement
    disagree_stat: str = "median"    # "median" | "frac": per-line statistic compared to disagree_s / frac
    disagree_frac: float = 0.3
    min_qwen_conf: float = 0.15      # Qwen times replace MMS only above this mean timestamp confidence
    min_win_s: float = 1.0
    # degenerate-output checks (a degenerate line neither agrees nor is adopted)
    min_span_per_syl_s: float = 0.05  # Qwen line span below this x n_syllables -> "collapsed"
    max_pile_frac: float = 0.5       # more onsets than this within one Qwen bin (80 ms) -> "collapsed"
    max_pinned_frac: float = 0.5     # more onsets than this within one Qwen bin of a window edge -> "pinned"
    min_active_frac: float = 0.5     # fewer onsets than this near vocal activity -> "off_vocals"
    active_tol_s: float = 0.1        # "near": within this of an active frame


@dataclass
class Window:
    t0: float
    t1: float
    lines: list[int]                 # lines to refine
    anchors: list[int]               # trusted neighbour lines included as context
    lo: float                        # end of the previous trusted line (or 0)
    hi: float                        # start of the next trusted line (or song end)


@dataclass
class LineRefinement:
    line: int
    syl_times: list[tuple[float, float] | None]   # per Line.syllables(); None = not timed by Qwen
    qwen_conf: float
    disagreement: float              # median |Qwen - MMS| syllable onset (s)
    frac_disagree: float             # fraction of syllables with |Qwen - MMS| > disagree_s
    disagree: bool                   # True also when Qwen could not time the line
    adopted: bool                    # Qwen times replace the MMS times
    reason: str
    window: tuple[float, float]
    language: str


@dataclass
class RefineResult:
    lines: dict[int, LineRefinement] = field(default_factory=dict)
    windows: list[dict] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    model: str = QWEN_ID

    def summary(self) -> dict:
        rows = list(self.lines.values())
        return {"model": self.model, "n_lines": len(rows),
                "n_adopted": sum(r.adopted for r in rows), "n_disagree": sum(r.disagree for r in rows),
                "timings": self.timings, "windows": self.windows,
                "lines": {str(r.line): {k: v for k, v in asdict(r).items() if k not in ("syl_times", "line")}
                          for r in rows}}


# ----------------------------------------------------------------------------------------- model
_QWEN: tuple[object, object, str] | None = None


def qwen_cached() -> bool:
    """True if every Qwen3-ForcedAligner file the processor and model load is in the local HF cache (never
    downloads; an interrupted download with only the weights counts as not cached)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        return all(isinstance(try_to_load_from_cache(QWEN_ID, f), str) for f in QWEN_FILES)
    except (ImportError, OSError, ValueError):
        return False


def load_qwen(local_only: bool = True, device: str | None = None):
    global _QWEN
    if _QWEN is None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor
        dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        proc = AutoProcessor.from_pretrained(QWEN_ID, local_files_only=local_only)
        model = AutoModelForTokenClassification.from_pretrained(QWEN_ID, dtype=torch.float32,
                                                                local_files_only=local_only)
        _QWEN = (proc, model.to(dev).eval(), dev)
    return _QWEN


def release_qwen() -> None:
    global _QWEN
    if _QWEN is not None:
        import gc

        import torch
        _QWEN = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def qwen_align(seg: np.ndarray, units: list[str], *, local_only: bool = True
               ) -> tuple[list[tuple[float, float]], list[float]]:
    """Align `units` (non-empty strings) to a 16 kHz segment -> ([(start, end)] relative s, [confidence])."""
    import torch
    proc, model, dev = load_qwen(local_only)
    conv = [[{"role": "user", "content": [{"type": "audio", "audio": np.asarray(seg, dtype=np.float32)}]
              + [{"type": "text", "text": u} for u in units]}]]
    inputs = proc.apply_chat_template(conv, tokenize=True, return_dict=True).to(dev)
    with torch.inference_mode():
        logits = model(**inputs).logits
    ts_id = model.config.timestamp_token_id
    out = proc.decode_forced_alignment(logits=logits, input_ids=inputs["input_ids"], word_lists=[units],
                                       timestamp_token_id=ts_id)[0]
    mask = inputs["input_ids"][0] == ts_id
    pmax = logits[0][mask].float().softmax(-1).max(-1).values.cpu().numpy()
    times = [(float(o["start_time"]), float(o["end_time"])) for o in out]
    conf = [float(np.mean(pmax[2 * i: 2 * i + 2])) if 2 * i + 1 < len(pmax) else 0.0
            for i in range(len(units))]
    return times, conf


# ----------------------------------------------------------------------------------------- helpers
def clean_unit(text: str) -> str:
    """Characters the Qwen aligner keeps: letters, digits, apostrophes (CJK ideographs are letters)."""
    return "".join(c for c in unicodedata.normalize("NFKC", text)
                   if c == "'" or unicodedata.category(c)[0] in ("L", "N"))


def line_language(line: Line) -> str:
    """Dominant language of a line by syllable count (ko/ja/en); 'x' if none."""
    counts: dict[str, int] = {}
    for w in line.words:
        if w.lang in LANG_NAMES:
            counts[w.lang] = counts.get(w.lang, 0) + max(len(w.syllables), 1)
    return max(counts, key=lambda k: counts[k]) if counts else "x"


def _timed(line: Line) -> bool:
    return bool(line.syllables()) and line.start is not None and line.end is not None


def plan_windows(lines: list[Line], targets: list[int], duration: float, opts: RefineOptions) -> list[Window]:
    """Group flagged lines into <= max_win_s windows bounded by their trusted neighbours.

    Every window contains the MMS [start, end] of each line it refines (a line longer than the window is not
    refined). Gaps longer than max_win_s are split between consecutive flagged lines."""
    tset = {t for t in targets if _timed(lines[t])}
    trusted = [_timed(ln) and i not in tset for i, ln in enumerate(lines)]
    runs: list[list[int]] = []
    for t in sorted(tset):
        if runs and not any(trusted[k] for k in range(runs[-1][-1] + 1, t)):
            runs[-1].append(t)
        else:
            runs.append([t])
    pad, mw = opts.pad_s, opts.max_win_s
    out: list[Window] = []
    for run in runs:
        p = next((k for k in range(run[0] - 1, -1, -1) if trusted[k]), None)
        q = next((k for k in range(run[-1] + 1, len(lines)) if trusted[k]), None)
        lo = lines[p].end if p is not None else 0.0
        hi = lines[q].start if q is not None else duration
        if hi - lo + 2 * pad <= mw:
            t0, t1, anchors = lo - pad, hi + pad, []
            if opts.anchors and p is not None and t1 - (lines[p].start - pad) <= mw:
                t0, anchors = lines[p].start - pad, [p]
            if opts.anchors and q is not None and (lines[q].end + pad) - t0 <= mw:
                t1, anchors = lines[q].end + pad, [*anchors, q]
            out.append(Window(max(0.0, t0), min(duration, t1), run, sorted(anchors), lo, hi))
            continue
        # long gap: split the run at the midpoints between the (coarse) MMS line times; a window starts at
        # the split point unless its first line would not fit, then it starts just early enough for it
        i = 0
        while i < len(run):
            first = lines[run[i]]
            left = lo - pad if i == 0 else 0.5 * (lines[run[i - 1]].end + first.start)
            if first.end - first.start + 2 * pad > mw:       # longer than a window: not refined
                i += 1
                continue
            ws = max(left, first.end + pad - mw)
            j = i
            while j + 1 < len(run) and lines[run[j + 1]].end + pad - ws <= mw:
                j += 1
            right = hi + pad if j == len(run) - 1 else 0.5 * (lines[run[j]].end + lines[run[j + 1]].start)
            out.append(Window(max(0.0, ws), min(duration, right, ws + mw), run[i: j + 1], [], lo, hi))
            i = j + 1
    for w in out:                                    # the processor needs >= 0.5 s of audio
        if w.t1 - w.t0 < opts.min_win_s:
            c = 0.5 * (w.t0 + w.t1)
            w.t0, w.t1 = max(0.0, c - opts.min_win_s / 2), min(duration, c + opts.min_win_s / 2)
    return out


# ----------------------------------------------------------------------------------------- refinement
def refine_lines(lines: list[Line], y: np.ndarray, targets: list[int], *,
                 options: RefineOptions | None = None, local_only: bool = True,
                 active: np.ndarray | None = None, active_frame_s: float = 0.02,
                 progress: Progress | None = None) -> RefineResult:
    """Re-align `targets` (flagged line indices, times already filled by MMS) with Qwen3-FA windows.

    `y` is the 16 kHz mono vocal stem; `active` an optional per-frame vocal-activity mask (frame length
    `active_frame_s`) used to reject Qwen onsets placed in silence. Lines are not modified; see
    align.apply_refinement."""
    opts = options or RefineOptions()
    duration = len(y) / SR
    res = RefineResult()
    t0 = time.perf_counter()
    windows = plan_windows(lines, targets, duration, opts)
    res.timings["plan"] = round(time.perf_counter() - t0, 3)
    t_load = time.perf_counter()
    if windows:
        load_qwen(local_only)
    res.timings["load"] = round(time.perf_counter() - t_load, 3)
    t_run = time.perf_counter()
    for wi, w in enumerate(windows):
        members = sorted(set(w.lines) | set(w.anchors))
        keys: list[tuple[int, int]] = []
        texts: list[str] = []
        for li in members:
            for si, syl in enumerate(lines[li].syllables()):
                word = _word_of(lines[li], si)
                if lines[li].backing or (word is not None and word.kind == "backing"):
                    continue
                txt = clean_unit(syl.text)
                if txt:
                    keys.append((li, si))
                    texts.append(txt)
        lang = _dominant([line_language(lines[i]) for i in w.lines])
        info = {"t0": round(w.t0, 3), "t1": round(w.t1, 3), "lines": w.lines, "anchors": w.anchors,
                "language": lang, "n_units": len(texts)}
        if not texts:
            res.windows.append(info)
            continue
        seg = y[int(w.t0 * SR): int(w.t1 * SR)]
        times, conf = qwen_align(seg, texts, local_only=local_only)
        q = {k: (w.t0 + a, w.t0 + b, c) for k, (a, b), c in zip(keys, times, conf)}
        anchor_err = _median_onset_diff(lines, [k for k in keys if k[0] in w.anchors], q)
        info["anchor_err"] = None if anchor_err is None else round(anchor_err, 3)
        window_ok = anchor_err is None or anchor_err <= opts.anchor_tol_s
        for li in w.lines:
            res.lines[li] = _line_result(lines[li], li, q, w, window_ok, lang, opts, active, active_frame_s)
        res.windows.append(info)
        if progress:
            progress("align", 65 + 30 * (wi + 1) / len(windows), f"qwen window {wi + 1}/{len(windows)}")
    res.timings["run"] = round(time.perf_counter() - t_run, 3)
    return res


def _word_of(line: Line, syl_idx: int) -> Word | None:
    k = 0
    for w in line.words:
        if syl_idx < k + len(w.syllables):
            return w
        k += len(w.syllables)
    return None


def _dominant(langs: list[str]) -> str:
    langs = [x for x in langs if x != "x"] or ["x"]
    return max(set(langs), key=langs.count)


def _median_onset_diff(lines: list[Line], keys: list[tuple[int, int]], q: dict) -> float | None:
    d = [abs(q[k][0] - lines[k[0]].syllables()[k[1]].start) for k in keys
         if lines[k[0]].syllables()[k[1]].start is not None]
    return float(np.median(d)) if d else None


def degenerate_reason(timed: list[tuple[float, float]], w: Window, opts: RefineOptions,
                      active: np.ndarray | None = None, frame_s: float = 0.02) -> str | None:
    """Why a line's Qwen times are unusable, or None. `timed` are absolute (start, end) per syllable."""
    on = np.array([a for a, _ in timed])
    en = np.array([b for _, b in timed])
    if len(on) >= 2:
        srt = np.sort(on)
        pile = int(np.max(np.searchsorted(srt, srt + QWEN_BIN_S + 1e-6, side="right") - np.arange(len(srt))))
        if en.max() - on.min() < opts.min_span_per_syl_s * len(on) or pile > opts.max_pile_frac * len(on):
            return "collapsed"
    pinned = (on < w.t0 + QWEN_BIN_S) | (on > w.t1 - QWEN_BIN_S)
    if pinned.mean() > opts.max_pinned_frac:
        return "pinned_to_window_edge"
    if active is not None and len(active):
        tol = round(opts.active_tol_s / frame_s)
        frames = np.round(on / frame_s).astype(int)
        near = [bool(active[max(f - tol, 0): f + tol + 1].any()) for f in frames]
        if np.mean(near) < opts.min_active_frac:
            return "off_vocals"
    return None


def _line_result(line: Line, li: int, q: dict, w: Window, window_ok: bool, lang: str,
                 opts: RefineOptions, active: np.ndarray | None = None,
                 active_frame_s: float = 0.02) -> LineRefinement:
    sylls = line.syllables()
    syl_times: list[tuple[float, float] | None] = [None] * len(sylls)
    confs, diffs = [], []
    for si, syl in enumerate(sylls):
        if (li, si) not in q:
            continue
        a, b, c = q[(li, si)]
        syl_times[si] = (a, max(a, b))
        confs.append(c)
        if syl.start is not None:
            diffs.append(abs(a - syl.start))
    timed = [t for t in syl_times if t is not None]
    inside = bool(timed) and all(w.lo - opts.accept_tol_s <= a and b <= w.hi + opts.accept_tol_s
                                 for a, b in timed)
    med = float(np.median(diffs)) if diffs else float("inf")
    frac = float(np.mean(np.array(diffs) > opts.disagree_s)) if diffs else 1.0
    disagree = (med > opts.disagree_s) if opts.disagree_stat == "median" else (frac > opts.disagree_frac)
    qconf = float(np.mean(confs)) if confs else 0.0
    bad = degenerate_reason(timed, w, opts, active, active_frame_s) if timed else None
    if not timed:
        reason = "no_units"
    elif bad is not None:
        reason, disagree = bad, True                 # no evidence either way: the line stays flagged
    elif not disagree:
        reason = "agree"
    elif not window_ok:
        reason = "anchor_mismatch"
    elif not inside:
        reason = "outside_gap"
    elif qconf < opts.min_qwen_conf:
        reason = "low_qwen_conf"
    else:
        reason = "adopted"
    return LineRefinement(line=li, syl_times=syl_times, qwen_conf=round(qconf, 4),
                          disagreement=round(med, 3) if diffs else -1.0, frac_disagree=round(frac, 3),
                          disagree=bool(disagree), adopted=reason == "adopted", reason=reason,
                          window=(round(w.t0, 3), round(w.t1, 3)), language=lang)
