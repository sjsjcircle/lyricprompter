"""Gated global CTC forced alignment of lyrics to the vocal stem (torchaudio MMS_FA, with star token).

Pipeline (docs/eval/ALIGNMENT.md has the evidence behind every default):

1. Emissions: MMS_FA(with_star) log-posteriors at 50 fps from the 16 kHz vocal stem, on MPS in 30 s windows
   with 2 s of context on each side (model cached in-process; `release_models()` frees it).
2. Gate: real-letter log-probs get -15 on frames where the stem is silent. "Silent" is adaptive: below
   noise floor (p10) + 20 dB, clipped to [loud - 55, loud - 35] dB, dilated by 0.3 s.
3. Tokens: `AlignUnit.roman` letters in sung order, '*' at both ends and between lines. Backing words /
   lines are skipped (a '*' keeps their place) and get interpolated times.
4. Viterbi: `torchaudio.functional.forced_align` (CPU only) -> one TokenSpan per target token.
5. Times: TokenSpan -> AlignUnit start/end/score -> Syllable -> Line. Units without tokens are interpolated
   (in a long gap: line-final / line-initial units stay within 1 s of their own line, skipped backing starts
   at the first vocal onset inside the gap); times are made non-decreasing; a line's first onset is snapped
   back (<= 0.5 s) to the start of its loud vocal run when CTC fired late inside a held note; syllable ends
   are extended while the stem stays within 20 dB of its loud level (sustained vowels, reverb tails
   excluded), then gaps < 0.15 s to the next unit are closed; ends before a pause are kept.
6. Confidence: mean unit posterior per line. conf < 0.15 -> needs_review; conf < 0.10 -> re-checked by
   Qwen3-ForcedAligner in a window between trusted neighbours (`refine.py`), which clears the review flag
   when both aligners agree and replaces the MMS times only when a confident, non-degenerate Qwen alignment
   disagrees. With refine="auto" a Qwen failure keeps the MMS result (details["refine"]["error"]).
"""
from __future__ import annotations

import gc
import itertools
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .. import config
from ..text.model import AlignUnit, Line, Syllable
from .vad import vocal_activity

if TYPE_CHECKING:
    import torch

    from .refine import RefineOptions, RefineResult

Progress = Callable[[str, float, str], None]

SR = config.ANALYSIS_SR
FRAME_HOP = 320                      # wav2vec2 stride at 16 kHz
FRAME_S = FRAME_HOP / SR             # 20 ms
MMS_MODEL_NAME = "torchaudio.pipelines.MMS_FA(with_star=True)"
INTERP_UNIT_S = 0.25                 # nominal duration of an interpolated unit that has no neighbour
INLINE_PAUSE_S = 1.0                 # longest pause searched for a line-final / line-initial unaligned unit


@dataclass
class AlignOptions:
    gate: bool = True
    gate_penalty: float = 15.0       # subtracted from real-letter log-probs on inactive frames
    gate_rel_db: float = -35.0              # strictest threshold re. the stem's loud level (p99)
    gate_noise_margin_db: float | None = 20.0   # adaptive: noise floor (p10) + margin, clipped to
    gate_min_rel_db: float = -55.0          # [loud + gate_min_rel_db, loud + gate_rel_db]; None = fixed
    gate_dilate_s: float = 0.3
    star_between_lines: bool = True
    star_between_words: bool = False
    star_penalty: float = 0.0        # subtracted from the '*' log-prob (0 = free garbage frames)
    include_backing: bool = False    # align backing lines / words (default: skip + interpolate)
    star_for_backing: bool = True    # skipped backing words leave a '*' in their place
    time_offset_s: float = 0.0       # constant added to every CTC time (bias calibration)
    snap_onsets: str = "line"        # "none" | "line" | "all": move late CTC onsets to the vocal-run start
    max_snap_s: float = 0.5
    snap_rel_db: float | None = -25.0  # onset-run threshold re. loud level (None: the gate's activity mask)
    end_mode: str = "activity"       # "raw" | "close" | "activity"
    close_gap_s: float = 0.15
    max_extend_s: float = 1.5        # cap on activity-based end extension
    end_rel_db: float | None = -20.0  # sustain threshold re. loud level for end extension (None: gate mask)
    review_threshold: float = 0.15   # line confidence below this -> needs_review
    refine_threshold: float = 0.10   # line confidence below this -> re-checked by Qwen3-FA (refine)
    win_s: float = 30.0
    ctx_s: float = 2.0


@dataclass
class AlignReport:
    method: str                      # e.g. "mms_fa+gate" or "mms_fa+gate+qwen"
    line_confidence: list[float | None]  # per Line, 0..1; None = no lead acoustic unit (= Line.confidence)
    flagged: list[int]               # line indices needing review
    timings: dict[str, float]        # seconds spent per stage
    details: dict = field(default_factory=dict)


@dataclass
class Emission:
    log_probs: torch.Tensor          # [T, C] float32 on CPU, log domain, last column = '*'
    labels: tuple[str, ...]
    frame_s: float = FRAME_S

    @property
    def n_frames(self) -> int:
        return int(self.log_probs.shape[0])

    @property
    def duration(self) -> float:
        return self.n_frames * self.frame_s


@dataclass
class UnitRef:
    line: int
    word: int
    syl_idx: int                     # index into lines[line].syllables()
    syl: Syllable
    unit: AlignUnit
    aligned: bool                    # contributes acoustic tokens (False: backing skipped / empty roman)
    skipped_backing: bool = False


@dataclass
class RawAlignment:
    """Per-unit CTC result in sung order before interpolation / clean-up (NaN = no tokens)."""
    refs: list[UnitRef]
    start: np.ndarray
    end: np.ndarray
    score: np.ndarray
    n_tokens: int
    duration: float

    def override(self, idx: list[int], start: np.ndarray, end: np.ndarray) -> None:
        self.start[idx] = start
        self.end[idx] = end


# ----------------------------------------------------------------------------------------- model / emission
_MMS: tuple[object, str] | None = None


def best_device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _mms_model(device: str | None = None):
    global _MMS
    if _MMS is None:
        from torchaudio.pipelines import MMS_FA
        dev = device or best_device()
        _MMS = (MMS_FA.get_model(with_star=True).to(dev).eval(), dev)
    return _MMS


def mms_labels() -> tuple[str, ...]:
    from torchaudio.pipelines import MMS_FA
    return tuple(MMS_FA.get_labels(star="*"))


def release_models() -> None:
    """Free the cached MMS_FA model (keep at most one heavy model per process)."""
    global _MMS
    _MMS = None
    _free_device_memory()


def _free_device_memory() -> None:
    import torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def to_16k(y: np.ndarray, sr: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:                                  # (n, ch) or (ch, n) -> mono
        y = y.mean(axis=1 if y.shape[0] > y.shape[1] else 0)
    if sr != SR:
        import soxr
        y = soxr.resample(y, sr, SR, quality="HQ").astype(np.float32)
    return np.ascontiguousarray(y)


def compute_emission(y: np.ndarray, sr: int = SR, *, win_s: float = 30.0, ctx_s: float = 2.0,
                     device: str | None = None, progress: Progress | None = None) -> Emission:
    """Chunked MMS_FA log-posteriors; frame k covers samples [320k, 320k+400) of the 16 kHz signal."""
    import torch
    y = to_16k(y, sr)
    model, dev = _mms_model(device)
    win = max(FRAME_HOP, int(win_s * SR) // FRAME_HOP * FRAME_HOP)
    ctx = int(ctx_s * SR) // FRAME_HOP * FRAME_HOP
    n_total = max(1, len(y) // FRAME_HOP)
    wav = torch.from_numpy(y)
    outs = []
    starts = list(range(0, max(len(y), 1), win))
    with torch.inference_mode():
        for ci, start in enumerate(starts):
            a, b = max(0, start - ctx), min(len(y), start + win + ctx)
            x = wav[a:b]
            if len(x) < 400:
                x = torch.nn.functional.pad(x, (0, 400 - len(x)))
            em, _ = model(x.unsqueeze(0).to(dev))
            em = em[0].float().cpu()
            lead = (start - a) // FRAME_HOP
            n = min(win, len(y) - start) // FRAME_HOP
            outs.append(em[lead: lead + n])
            if progress:
                progress("align", 5 + 45 * (ci + 1) / len(starts), "emissions")
    em = torch.cat(outs) if outs else torch.zeros(0, 29)
    if em.shape[0] < n_total:                        # last window can come out one frame short
        pad = em[-1:].expand(n_total - em.shape[0], -1) if em.shape[0] else torch.zeros(n_total, 29)
        em = torch.cat([em, pad])
    return Emission(log_probs=em[:n_total].contiguous(), labels=mms_labels())


@dataclass
class VocalMasks:
    """Per-emission-frame vocal activity derived from the stem."""
    gate: np.ndarray                 # dilated activity: CTC gate
    active: np.ndarray               # undilated activity: interpolation
    sustain: np.ndarray              # stricter activity (no reverb tails): syllable-end extension
    onset: np.ndarray | None = None  # activity used for onset snapping (default: `active`)


def vocal_masks(y: np.ndarray, n_frames: int, sr: int = SR, opts: AlignOptions | None = None) -> VocalMasks:
    opts = opts or AlignOptions()
    y = to_16k(y, sr)
    kw = {"rel_db": opts.gate_rel_db, "noise_margin_db": opts.gate_noise_margin_db,
          "min_rel_db": opts.gate_min_rel_db}
    gate = vocal_activity(y, SR, hop=FRAME_HOP, dilate_s=opts.gate_dilate_s, **kw)
    active = vocal_activity(y, SR, hop=FRAME_HOP, dilate_s=0.0, **kw)
    sustain = active if opts.end_rel_db is None else vocal_activity(
        y, SR, hop=FRAME_HOP, rel_db=opts.end_rel_db, dilate_s=0.0)
    onset = active if opts.snap_rel_db is None else vocal_activity(
        y, SR, hop=FRAME_HOP, rel_db=opts.snap_rel_db, dilate_s=0.0)
    return VocalMasks(_fit(gate, n_frames), _fit(active, n_frames), _fit(sustain, n_frames),
                      _fit(onset, n_frames))


def _fit(mask: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    out[: min(n, len(mask))] = mask[:n]
    return out


# ----------------------------------------------------------------------------------------- tokens / viterbi
def unit_refs(lines: list[Line], include_backing: bool = False) -> list[UnitRef]:
    refs = []
    for li, ln in enumerate(lines):
        si = 0
        for wi, w in enumerate(ln.words):
            skip = (ln.backing or w.kind == "backing") and not include_backing
            for s in w.syllables:
                for u in s.units:
                    refs.append(UnitRef(li, wi, si, s, u, aligned=bool(u.roman) and not skip,
                                        skipped_backing=skip and bool(u.roman)))
                si += 1
    return refs


def build_tokens(refs: list[UnitRef], labels: tuple[str, ...], *, star_between_lines: bool = True,
                 star_between_words: bool = False, star_for_backing: bool = True
                 ) -> tuple[list[int], list[int]]:
    """Target label ids and their owner (index into refs, -1 for '*')."""
    d = {c: i for i, c in enumerate(labels)}
    star = d["*"]
    toks, owner = [star], [-1]
    prev: UnitRef | None = None
    pending_star = False                  # a skipped backing word since the previous aligned unit
    for k, r in enumerate(refs):
        if not r.aligned:
            pending_star |= star_for_backing and r.skipped_backing
            continue
        ids = [d[c] for c in r.unit.roman if c in d]
        if not ids:
            continue
        if prev is not None and toks[-1] != star and (
                (star_between_lines and r.line != prev.line)
                or (star_between_words and r.line == prev.line and r.word != prev.word)
                or pending_star):
            toks.append(star)
            owner.append(-1)
        pending_star = False
        toks += ids
        owner += [k] * len(ids)
        prev = r
    if toks[-1] != star:
        toks.append(star)
        owner.append(-1)
    return toks, owner


def viterbi(log_probs: torch.Tensor, tokens: list[int]) -> list:
    """forced_align on CPU -> one TokenSpan per target token."""
    import torch
    import torchaudio.functional as taf
    T = int(log_probs.shape[0])
    repeats = sum(1 for a, b in itertools.pairwise(tokens) if a == b)
    if T < len(tokens) + repeats:
        raise ValueError(f"audio too short for the lyrics: {T} frames < {len(tokens) + repeats} required")
    em = log_probs.detach().to("cpu", torch.float32).unsqueeze(0).contiguous()
    tg = torch.tensor([tokens], dtype=torch.int32)
    path, scores = taf.forced_align(em, tg, blank=0)
    spans = taf.merge_tokens(path[0], scores[0].exp(), blank=0)
    if len(spans) != len(tokens):
        raise RuntimeError(f"forced_align returned {len(spans)} spans for {len(tokens)} tokens")
    return spans


def apply_gate(emission: Emission, gate: np.ndarray | None, penalty: float) -> torch.Tensor:
    """Copy of the log-probs with real letters penalised where the vocal stem is inactive."""
    import torch
    em = emission.log_probs.clone()
    if gate is not None and penalty > 0:
        inactive = torch.from_numpy((~_fit(gate, em.shape[0])).astype(np.float32))
        pen = torch.tensor([0.0 if c in ("-", "*") else penalty for c in emission.labels])
        em -= inactive[:, None] * pen[None, :]
    return em


def mms_align(lines: list[Line], emission: Emission, *, gate: np.ndarray | None = None,
              options: AlignOptions | None = None) -> RawAlignment:
    opts = options or AlignOptions()
    refs = unit_refs(lines, opts.include_backing)
    toks, owner = build_tokens(refs, emission.labels, star_between_lines=opts.star_between_lines,
                               star_between_words=opts.star_between_words,
                               star_for_backing=opts.star_for_backing)
    n = len(refs)
    start, end, score = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    if len(toks) > 1:
        lp = apply_gate(emission, gate if opts.gate else None, opts.gate_penalty)
        if opts.star_penalty:
            lp[:, emission.labels.index("*")] -= opts.star_penalty
        spans = viterbi(lp, toks)
        per_unit: dict[int, list] = {}
        for sp, o in zip(spans, owner, strict=True):
            if o >= 0:
                per_unit.setdefault(o, []).append(sp)
        fs, off = emission.frame_s, opts.time_offset_s
        for k, ss in per_unit.items():
            start[k] = max(0.0, ss[0].start * fs + off)
            end[k] = max(start[k], ss[-1].end * fs + off)
            score[k] = float(np.mean([s.score for s in ss]))
    return RawAlignment(refs=refs, start=start, end=end, score=score, n_tokens=len(toks),
                        duration=emission.duration)


# ----------------------------------------------------------------------------------------- post-processing
def interpolate_missing(start: np.ndarray, end: np.ndarray, duration: float,
                        active: np.ndarray | None = None, frame_s: float = FRAME_S, *,
                        line: np.ndarray | None = None, floating: np.ndarray | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN runs between known units, with at most INTERP_UNIT_S per unit in a long gap.

    In a long gap between two lines (needs `line`), the units ending the previous line start at a vocal onset
    at most INLINE_PAUSE_S after the previous unit, else right after it (a line-final 'っ' / unread word),
    and the units starting the next line begin at a vocal onset at most INLINE_PAUSE_S before the next
    unit, else right before it. Everything else (skipped backing = `floating`, runs inside one line, lines
    without any aligned unit) starts at the first vocal onset inside the gap, else right after the
    previous unit."""
    s, e = start.astype(float).copy(), end.astype(float).copy()
    n = len(s)
    known = ~np.isnan(s)
    if not known.any():
        a, b = 0.0, max(duration, 0.0)
        if active is not None and active.any():
            idx = np.flatnonzero(active)
            a, b = idx[0] * frame_s, (idx[-1] + 1) * frame_s
        edges = np.linspace(a, b, n + 1)
        return edges[:-1], edges[1:]
    flt = np.zeros(n, bool) if floating is None else np.asarray(floating, bool)

    def fill(i0: int, i1: int, a0: float, b0: float) -> None:
        if i1 > i0:
            edges = np.linspace(a0, b0, i1 - i0 + 1)
            s[i0:i1], e[i0:i1] = edges[:-1], edges[1:]

    def onset(a0: float, b0: float, last: bool = False) -> float | None:
        return None if active is None else _onset_in(active, a0, b0, frame_s, last)

    i = 0
    while i < n:
        if known[i]:
            i += 1
            continue
        j = i
        while j < n and not known[j]:
            j += 1
        k = j - i
        if i == 0:
            b0 = float(s[j])
            fill(i, j, max(0.0, b0 - k * INTERP_UNIT_S), b0)
        elif j == n:
            a0 = float(e[i - 1])
            fill(i, j, a0, min(max(duration, a0), a0 + k * INTERP_UNIT_S))
        else:
            a0, b0 = float(e[i - 1]), float(max(e[i - 1], s[j]))
            if b0 - a0 <= k * INTERP_UNIT_S:
                fill(i, j, a0, b0)
            else:                                    # long gap
                head, tail = i, j
                if line is not None and line[i - 1] != line[j]:
                    while head < j and not flt[head] and line[head] == line[i - 1]:
                        head += 1
                    while tail > head and not flt[tail - 1] and line[tail - 1] == line[j]:
                        tail -= 1
                ha, tb = a0, b0
                if head > i:                         # end of the previous line
                    hs = onset(a0, min(b0, a0 + INLINE_PAUSE_S))
                    hs = a0 if hs is None else hs
                    ha = min(b0, hs + (head - i) * INTERP_UNIT_S)
                    fill(i, head, hs, ha)
                if tail < j:                         # start of the next line
                    ts = onset(max(ha, b0 - INLINE_PAUSE_S), b0, last=True)
                    tb = max(ha, b0 - (j - tail) * INTERP_UNIT_S) if ts is None else ts
                    fill(tail, j, tb, b0)
                if tail > head:
                    ma = ha
                    if tb - ha > (tail - head) * INTERP_UNIT_S:
                        mo = onset(ha, tb)
                        ma = ha if mo is None else mo
                    fill(head, tail, ma, min(tb, ma + (tail - head) * INTERP_UNIT_S))
        i = j
    return s, e


def _onset_in(active: np.ndarray, a: float, b: float, frame_s: float, last: bool = False) -> float | None:
    """Time of the first (or last) inactive->active transition inside (a, b], or None."""
    fa, fb = int(np.ceil(a / frame_s)), int(np.floor(b / frame_s))
    seg = active[max(fa - 1, 0):fb]
    onsets = np.flatnonzero(seg[1:] & ~seg[:-1]) + 1 if len(seg) > 1 else []
    if not len(onsets):
        return None
    return (max(fa - 1, 0) + int(onsets[-1 if last else 0])) * frame_s


def make_monotonic(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """start_k <= end_k <= start_{k+1}: overlaps cut the previous end (starts are the reliable part)."""
    s, e = start.astype(float).copy(), end.astype(float).copy()
    for i in range(len(s)):
        if i > 0 and s[i] < e[i - 1]:
            if s[i] >= s[i - 1]:
                e[i - 1] = s[i]
            else:
                s[i] = e[i - 1]
        e[i] = max(e[i], s[i])
    return s, e


def snap_onsets(start: np.ndarray, end: np.ndarray, active: np.ndarray, *, eligible: np.ndarray | None = None,
                max_snap_s: float = 1.0, frame_s: float = FRAME_S) -> np.ndarray:
    """CTC often fires late inside a sustained or soft onset (a held pickup note, a breathy attack). If a
    unit starts inside a vocal-activity run that began after the previous unit had ended (an inactive
    frame lies in between) and at most `max_snap_s` earlier, move the start back to the run onset."""
    s = start.astype(float).copy()
    n_act = len(active)
    for i in range(len(s)):
        if eligible is not None and not eligible[i]:
            continue
        f = round(s[i] / frame_s)
        if f <= 0 or f >= n_act or not active[f]:
            continue
        prev_end = end[i - 1] if i > 0 else 0.0
        lo = max(int(np.ceil(prev_end / frame_s)), f - round(max_snap_s / frame_s), 0)
        g = f
        while g > lo and active[g - 1]:
            g -= 1
        if g > lo:
            s[i] = g * frame_s
    return s


def shape_ends(start: np.ndarray, end: np.ndarray, *, mode: str = "activity", close_gap_s: float = 0.15,
               max_extend_s: float = 1.5, active: np.ndarray | None = None, frame_s: float = FRAME_S
               ) -> np.ndarray:
    """Syllable-end policy for karaoke wipes (input must be monotonic; output stays monotonic)."""
    e = end.astype(float).copy()
    if mode == "raw":
        return e
    n = len(e)
    for i in range(n):
        nxt = start[i + 1] if i + 1 < n else np.inf
        if mode == "activity" and active is not None and len(active):
            limit = min(nxt, e[i] + max_extend_s)
            f = round(e[i] / frame_s)
            while f < len(active) and active[f] and (f + 1) * frame_s <= limit:
                f += 1
            e[i] = max(e[i], min(f * frame_s, limit))
        if nxt - e[i] < close_gap_s:
            e[i] = max(e[i], nxt)
    return e


def finalize(lines: list[Line], raw: RawAlignment, *, masks: VocalMasks | None = None,
             options: AlignOptions | None = None) -> list[float | None]:
    """Write unit / syllable / line times and line confidences in place; returns per-line confidence
    (mean posterior of the line's lead units; None for lines without any)."""
    opts = options or AlignOptions()
    active = masks.active if masks is not None else None
    s, e = interpolate_missing(raw.start, raw.end, raw.duration, active,
                               line=np.array([r.line for r in raw.refs], dtype=int),
                               floating=np.array([r.skipped_backing for r in raw.refs], dtype=bool))
    s, e = make_monotonic(s, e)
    if opts.snap_onsets != "none" and active is not None:
        known = ~np.isnan(raw.start)
        first = np.array([k == 0 or raw.refs[k].line != raw.refs[k - 1].line for k in range(len(raw.refs))])
        elig = known & (first if opts.snap_onsets == "line" else True)
        onset = masks.onset if masks is not None and masks.onset is not None else active
        s = snap_onsets(s, e, onset, eligible=elig, max_snap_s=opts.max_snap_s)
    e = shape_ends(s, e, mode=opts.end_mode, close_gap_s=opts.close_gap_s, max_extend_s=opts.max_extend_s,
                   active=masks.sustain if masks is not None else None)
    s, e = np.round(s, 3), np.round(e, 3)
    syl_bounds: dict[int, list[float]] = {}
    for k, r in enumerate(raw.refs):
        r.unit.start, r.unit.end = float(s[k]), float(e[k])
        r.unit.score = None if np.isnan(raw.score[k]) else round(float(raw.score[k]), 4)
        b = syl_bounds.setdefault(id(r.syl), [s[k], e[k]])
        b[1] = e[k]
    scores: dict[int, list[float]] = {}
    for k, r in enumerate(raw.refs):
        if not r.skipped_backing and not np.isnan(raw.score[k]):
            scores.setdefault(r.line, []).append(float(raw.score[k]))
    conf: list[float | None] = []
    prev = 0.0
    for li, ln in enumerate(lines):
        for syl in ln.syllables():
            b = syl_bounds.get(id(syl))
            # syllables without units (pure symbols) sit at the previous syllable's end
            syl.start, syl.end = (float(b[0]), float(b[1])) if b else (prev, prev)
            prev = syl.end
        sylls = ln.syllables()
        if sylls:
            ln.start, ln.end = sylls[0].start, sylls[-1].end
        else:                                        # text without syllables: a point in time
            ln.start, ln.end = (prev, prev) if ln.text.strip() or ln.words else (None, None)
        sc = scores.get(li)
        ln.confidence = round(float(np.mean(sc)), 4) if sc else None
        conf.append(ln.confidence)
    return conf


def mark_review(lines: list[Line], conf: list[float | None], threshold: float) -> list[int]:
    flagged = []
    for li, (ln, c) in enumerate(zip(lines, conf)):
        ln.needs_review = c is not None and c < threshold
        if ln.needs_review:
            flagged.append(li)
            if "low_confidence" not in ln.flags:
                ln.flags.append("low_confidence")
    return flagged


def refine_targets(conf: list[float | None], threshold: float) -> list[int]:
    """Lines sent to the Qwen refinement: aligned lines whose confidence is below `threshold`."""
    return [i for i, c in enumerate(conf) if c is not None and c < threshold]


def syllable_unit_index(raw: RawAlignment) -> dict[tuple[int, int], list[int]]:
    """(line, syllable index in line) -> unit indices into raw."""
    out: dict[tuple[int, int], list[int]] = {}
    for k, r in enumerate(raw.refs):
        out.setdefault((r.line, r.syl_idx), []).append(k)
    return out


def apply_syllable_times(raw: RawAlignment, times: dict[tuple[int, int], tuple[float, float]]) -> None:
    """Override raw unit times from syllable-level (start, end); a syllable's span is shared among its
    units in proportion to their roman length."""
    index = syllable_unit_index(raw)
    for key, (a, b) in times.items():
        ks = index.get(key)
        if not ks:
            continue
        w = np.array([max(len(raw.refs[k].unit.roman), 1) for k in ks], dtype=float)
        edges = a + (b - a) * np.concatenate([[0.0], np.cumsum(w) / w.sum()])
        raw.override(ks, edges[:-1], edges[1:])


# ----------------------------------------------------------------------------------------- entry points
def align_with_emission(lines: list[Line], emission: Emission, *, masks: VocalMasks | None = None,
                        options: AlignOptions | None = None) -> tuple[AlignReport, RawAlignment]:
    """MMS alignment from precomputed emissions (used by align_lines and the evaluation ablations)."""
    opts = options or AlignOptions()
    gate = masks.gate if masks is not None and opts.gate else None
    t0 = time.perf_counter()
    raw = mms_align(lines, emission, gate=gate, options=opts)
    t1 = time.perf_counter()
    conf = finalize(lines, raw, masks=masks, options=opts)
    flagged = mark_review(lines, conf, opts.review_threshold)
    t2 = time.perf_counter()
    method = "mms_fa" + ("+gate" if opts.gate and gate is not None else "")
    details = {"model": MMS_MODEL_NAME, "n_frames": emission.n_frames, "frame_s": emission.frame_s,
               "n_tokens": raw.n_tokens, "n_units": len(raw.refs),
               "n_units_aligned": int(np.sum(~np.isnan(raw.start))),
               "gate_active_frac": None if gate is None else round(float(np.mean(gate)), 3),
               "options": asdict(opts)}
    report = AlignReport(method=method, line_confidence=list(conf),
                         flagged=flagged, timings={"viterbi": round(t1 - t0, 3), "post": round(t2 - t1, 3)},
                         details=details)
    return report, raw


def align_lines(lines: list[Line], vocals: np.ndarray, sr: int = 16000, *, refine: str = "auto",
                progress: Progress | None = None, options: AlignOptions | None = None,
                refine_options: RefineOptions | None = None) -> AlignReport:
    """Align `lines` (from text.parse_lyrics) to the separated vocal stem; fills times in place."""
    if refine not in ("none", "qwen", "auto"):
        raise ValueError(f"refine must be none|qwen|auto, got {refine!r}")
    opts = options or AlignOptions()
    t0 = time.perf_counter()
    y = to_16k(vocals, sr)
    if progress:
        progress("align", 2, "loading MMS_FA")
    em = compute_emission(y, SR, win_s=opts.win_s, ctx_s=opts.ctx_s, progress=progress)
    t_em = time.perf_counter() - t0
    masks = vocal_masks(y, em.n_frames, SR, opts)
    if progress:
        progress("align", 55, "forced alignment")
    report, raw = align_with_emission(lines, em, masks=masks, options=opts)
    report.timings = {"emission": round(t_em, 3), **report.timings}
    report.details["device"] = _MMS[1] if _MMS else None
    targets = refine_targets([ln.confidence for ln in lines], opts.refine_threshold)
    if refine != "none" and targets:
        from . import refine as rf
        if refine == "qwen" or rf.qwen_cached():
            release_models()
            if progress:
                progress("align", 65, f"refining {len(targets)} low-confidence lines (Qwen3-FA)")
            t1 = time.perf_counter()
            try:
                res = rf.refine_lines(lines, y, targets, options=refine_options, local_only=refine == "auto",
                                      active=masks.active, active_frame_s=FRAME_S, progress=progress)
            except Exception as e:                   # "auto" is best effort: keep the MMS alignment
                if refine == "qwen":
                    raise
                report.details["refine"] = {"error": f"{type(e).__name__}: {e}"}
                res = None
            finally:
                rf.release_qwen()
            if res is not None:
                apply_refinement(lines, raw, res, report, masks=masks, options=opts)
            report.timings["refine"] = round(time.perf_counter() - t1, 3)
        else:
            report.details["refine"] = {"skipped": "Qwen3-ForcedAligner not cached locally"}
    if progress:
        progress("align", 100, f"aligned {len(lines)} lines, {len(report.flagged)} need review")
    return report


def apply_refinement(lines: list[Line], raw: RawAlignment, res: RefineResult, report: AlignReport, *,
                     masks: VocalMasks | None, options: AlignOptions) -> None:
    """Merge a refine.RefineResult into the raw alignment, re-finalize and recompute review flags."""
    times: dict[tuple[int, int], tuple[float, float]] = {}
    for li, lr in res.lines.items():
        if lr.adopted:
            for si, t in enumerate(lr.syl_times):
                if t is not None:
                    times[(li, si)] = t
    apply_syllable_times(raw, times)
    conf = finalize(lines, raw, masks=masks, options=options)
    flagged = []
    for li, (ln, c) in enumerate(zip(lines, conf)):
        lr = res.lines.get(li)
        if lr is None:
            ln.needs_review = c is not None and c < options.review_threshold
        else:
            drop = ("qwen", "mms_qwen_agree", "mms_qwen_disagree") + (() if lr.disagree else ("low_confidence",))
            ln.flags = [f for f in ln.flags if f not in drop]
            ln.flags.append("mms_qwen_disagree" if lr.disagree else "mms_qwen_agree")
            if lr.adopted:
                ln.flags.append("qwen")
            ln.needs_review = lr.disagree
        if ln.needs_review:
            flagged.append(li)
    report.flagged = flagged
    report.line_confidence = list(conf)
    report.method += "+qwen"
    report.details["refine"] = res.summary()
