"""Live-tracking evaluation: synthetic warped "lives" with exact ground truth + real performance pairs.

Every run streams the live audio through FeatureExtractor.push() in 1024-sample blocks (exactly as
the live input does, but faster than real time) into a LiveTracker, records the published ref_time
after each block and scores it at ground-truth lyric onsets with display-oriented, causal metrics.

Pairs (reference data from config.ref_repo_dir(), private: evaluation only)
- edelweiss       ref Edelweiss.wav, live Edelweiss_short.wav, GT = the 26 shared LRC tokens.
- kkotbat_r2s     ref 꽃밭에서_ref.wav, live 꽃밭에서_score.wav (MuseScore render); GT = 144 syllables
  kkotbat_s2r     mapped through resources/alignments/꽃밭에서_offline.csv (offline DTW, 25 fps) and
                  the reverse direction. The path itself is algorithmic (pseudo ground truth).
- synth_<song>    piecewise time-stretched (0.8-1.25x, 2-8 s segments) + optional transposition,
                  EQ tilt, reverb, pink noise at 20 dB SNR, gain change and a 1 s noise lead-in,
                  generated from the reference itself (mix, or the vocal stem with instrumental bleed).
- steady_<song>   pop / band performances (PopSpec, make_pop_live): one steady tempo ratio (0.92-1.08,
  pop_<song>      drift <= 1 %), the live mix rebuilt from the separated stems; pop_ adds performance
                  perturbations on the vocal only (dropped lines the crowd sings, early / late lines,
                  ad-libs). Ground truth follows the band. Songs: JamendoLyrics pop + J-pop in songs/.
Vocal mode uses data/stems/<name>/vocals.wav on both sides (synthetic: the reference vocal stem).
`--styles classic,pop` runs every pair with each tracking style (TrackerParams.style).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import typing
import unicodedata
import warnings
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np

from ..config import DATA_DIR, SONGS_DIR, ref_repo_dir
from ..features.extractor import FeatureConfig, FeatureExtractor, FeatureFrames, compute_offline, load_mono
from ..tracking.tracker import STYLES, LiveTracker, TrackerParams

SR = 16000
BLOCK = 1024
PATH_FPS = 25.0
TOLERANCES_MS = (200, 300, 500, 1000)


# ============================================================== ground truth loaders
def parse_lrc(path: Path) -> list[dict]:
    """LRC flavour of the reference repo: '[mm:ss.xx]tok  <mm:ss.xx>tok  <mm:ss.xx> tok'. A tag whose
    text starts with a space begins a new word. -> [{t, text, word_start, line}] (line = LRC line no.)"""
    out = []
    for li, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)", raw)
        if not m:
            continue
        mi, se, rest = m.groups()
        rest = rest.strip()
        if not rest:
            continue
        if not rest.startswith("<"):
            rest = f"<{mi}:{se}> " + rest
        for a, b, txt in re.findall(r"<(\d+):(\d+(?:\.\d+)?)>([^<]*)", rest):
            if txt.strip():
                out.append({"t": int(a) * 60 + float(b), "text": txt.strip(), "word_start": txt.startswith(" "), "line": li})
    return out


def lrc_line_starts(tokens: list[dict]) -> list[float]:
    seen, out = set(), []
    for tok in tokens:
        if tok["line"] not in seen:
            seen.add(tok["line"])
            out.append(tok["t"])
    return out


def strict_monotonic(wp: np.ndarray) -> np.ndarray:
    """Keep only warping-path steps that advance both axes (plus the end points)."""
    keep = np.r_[True, (np.diff(wp[:, 0]) > 0) & (np.diff(wp[:, 1]) > 0)]
    keep[-1] = True
    return wp[keep]


def score_csv_to_ref_gt(score_csv: Path, wp_csv: Path, sec_per_quarter: float = 0.5) -> list[dict]:
    """annots/<song>_score.csv + alignments/<song>_offline.csv (rows = score frame, ref frame at 25 fps)
    -> syllables [{t_score, t_ref, text}]."""
    with open(score_csv, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["lyric"]]
    wp = strict_monotonic(np.loadtxt(wp_csv, delimiter=",")) / PATH_FPS
    t_score = np.array([float(r["offset"]) * sec_per_quarter for r in rows])
    t_ref = np.interp(t_score, wp[:, 0], wp[:, 1])
    return [{"t_score": float(a), "t_ref": float(b), "text": unicodedata.normalize("NFC", r["lyric"])}
            for r, a, b in zip(rows, t_score, t_ref)]


def group_syllables_by_lines(lyrics_txt: Path, n_syl: int) -> list[int]:
    """Index of the first syllable of each non-empty lyrics line (Hangul syllables counted in order)."""
    firsts, k = [], 0
    for line in Path(lyrics_txt).read_text(encoding="utf-8").splitlines():
        n = sum(1 for ch in unicodedata.normalize("NFC", line) if "가" <= ch <= "힣")
        if n:
            if k < n_syl:
                firsts.append(k)
            k += n
    return firsts


# ============================================================== metrics
def prompter_metrics(t_live, pos_ref, gt_live, gt_ref, tolerances_ms=TOLERANCES_MS) -> dict:
    """Display-based causal metrics. t_live: times (s) at which positions were published; pos_ref:
    published ref position (s). gt_live/gt_ref: lyric-unit onsets in live / ref time.
    - pos error: published position sampled-and-held at each GT live onset minus the GT ref onset
    - latency: first publish time the (running-max) position reaches gt_ref[k], minus gt_live[k]
    - lost_frac: share of onsets with |error| > 2 s"""
    t_live, pos_ref = np.asarray(t_live, float), np.asarray(pos_ref, float)
    gt_live, gt_ref = np.asarray(gt_live, float), np.asarray(gt_ref, float)
    idx = np.clip(np.searchsorted(t_live, gt_live, side="right") - 1, 0, len(t_live) - 1)
    err = pos_ref[idx] - gt_ref
    run_max = np.maximum.accumulate(pos_ref)
    k = np.searchsorted(run_max, gt_ref, side="left")
    lat = np.where(k < len(t_live), t_live[np.minimum(k, len(t_live) - 1)] - gt_live, np.nan)
    ae = np.abs(err)
    out = {"n": len(err), "pos_AAE_ms": 1e3 * float(ae.mean()), "pos_MedAE_ms": 1e3 * float(np.median(ae)),
               "pos_bias_ms": 1e3 * float(err.mean()),
               "lat_median_ms": 1e3 * float(np.nanmedian(lat)) if np.isfinite(lat).any() else math.nan,
               "lat_p90_ms": 1e3 * float(np.nanpercentile(lat, 90)) if np.isfinite(lat).any() else math.nan,
               "lost_frac": float(np.mean(ae > 2.0))}
    out.update({f"PCO@{t}": 100.0 * float(np.mean(ae <= t / 1e3)) for t in tolerances_ms})
    return out


def confidence_auroc(t_live, pos_ref, conf, gt_live, gt_ref, tol_s: float = 0.5) -> float:
    """AUROC of the published confidence for 'error <= tol_s' at the GT onsets (nan if one class)."""
    idx = np.clip(np.searchsorted(np.asarray(t_live), gt_live, side="right") - 1, 0, len(t_live) - 1)
    ok = np.abs(np.asarray(pos_ref)[idx] - gt_ref) <= tol_s
    c = np.asarray(conf)[idx]
    if ok.all() or not ok.any():
        return math.nan
    from scipy.stats import rankdata

    r = rankdata(c)
    n1, n0 = ok.sum(), (~ok).sum()
    return float((r[ok].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ============================================================== synthetic lives
@dataclass(frozen=True)
class SynthSpec:
    seed: int = 0
    tempo: tuple[float, float] = (0.8, 1.25)   # ref seconds per live second, log-uniform per segment
    seg_s: tuple[float, float] = (2.0, 8.0)
    transpose: int = 0                         # semitones (live above ref)
    snr_db: float | None = 20.0                # pink noise
    rt60_s: float | None = 0.6                 # exponential-decay noise reverb
    reverb_wet: float = 0.25
    eq_tilt_db: float = 6.0                    # +-tilt across 100 Hz..8 kHz (random sign)
    gain_db: tuple[float, float] = (-18.0, -6.0)
    lead_s: float = 1.0                        # noise-only lead-in before the music starts
    bleed_db: float | None = None              # instrumental bleed relative to the vocal (vocal mode)
    jumps: tuple[tuple[float, float], ...] = ()  # structural edits: at ref time a, continue at a + delta


def synth_time_map(dur: float, spec: SynthSpec, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear map as breakpoints (live_pts, ref_pts) in seconds, including the lead-in."""
    live, ref = [0.0], [0.0]
    if spec.lead_s > 0:
        live.append(spec.lead_s)
        ref.append(0.0)
    lo, hi = np.log(spec.tempo[0]), np.log(spec.tempo[1])
    jumps = sorted(spec.jumps)
    while ref[-1] < dur - 1e-9:
        seg = min(rng.uniform(*spec.seg_s), dur - ref[-1])
        jump = None
        if jumps and ref[-1] < jumps[0][0] <= ref[-1] + seg:
            jump = jumps.pop(0)
            seg = jump[0] - ref[-1]
        rho = float(np.exp(rng.uniform(lo, hi)))
        ref.append(ref[-1] + seg)
        live.append(live[-1] + seg / rho)
        if jump is not None:                           # skip (delta > 0) or repeat (delta < 0)
            ref.append(float(np.clip(jump[0] + jump[1], 0.0, dur)))
            live.append(live[-1] + 1e-3)
    return np.array(live), np.array(ref)


def pv_warp(y: np.ndarray, sr: int, live_pts: np.ndarray, ref_pts: np.ndarray,
            n_fft: int = 2048, hop: int = 256) -> np.ndarray:
    """Phase-vocoder time warp following a monotone live->ref map (output sample at live time u holds
    the input around ref time map(u)). Standard (non-phase-locked) vocoder: smears transients and adds
    some phasiness; no artefacts at segment boundaries because the rate changes frame by frame."""
    import librosa

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    T = D.shape[1]
    n_out = round(live_pts[-1] * sr)
    u = np.arange(n_out // hop + 1) * hop / sr
    s = np.clip(np.interp(u, live_pts, ref_pts) * sr / hop, 0.0, T - 1.000001)
    Dp = np.concatenate([D, np.zeros((D.shape[0], 2), D.dtype)], axis=1)
    mag, ang = np.abs(Dp), np.angle(Dp)
    adv = np.linspace(0, np.pi * hop, D.shape[0])
    out = np.empty((D.shape[0], len(s)), np.complex64)
    phase = ang[:, 0].copy()
    for k, st in enumerate(s):
        i = int(st)
        a = st - i
        out[:, k] = ((1 - a) * mag[:, i] + a * mag[:, i + 1]) * np.exp(1j * phase)
        dp = ang[:, i + 1] - ang[:, i] - adv
        dp -= 2 * np.pi * np.round(dp / (2 * np.pi))
        phase += adv + dp                      # instantaneous frequency x synthesis hop
    return librosa.istft(out, hop_length=hop, n_fft=n_fft, length=n_out).astype(np.float32)


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.maximum(np.fft.rfftfreq(n), 1.0 / n)
    y = np.fft.irfft(X / np.sqrt(f), n)
    return (y / (np.std(y) + 1e-12)).astype(np.float32)


def degrade(y: np.ndarray, sr: int, spec: SynthSpec, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y, np.float64)
    if spec.eq_tilt_db:
        X = np.fft.rfft(y)
        f = np.maximum(np.fft.rfftfreq(len(y), 1.0 / sr), 1.0)
        tilt = spec.eq_tilt_db * (1 if rng.random() < 0.5 else -1)
        g_db = tilt * np.clip(np.log2(f / 900.0) / np.log2(8000.0 / 100.0) * 2, -1, 1)
        y = np.fft.irfft(X * 10 ** (g_db / 20), len(y))
    if spec.rt60_s:
        from scipy.signal import fftconvolve

        n = int(spec.rt60_s * sr)
        ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
        ir /= np.sqrt(np.sum(ir ** 2))
        wet = fftconvolve(y, ir)[:len(y)]
        y = (1 - spec.reverb_wet) * y + spec.reverb_wet * wet
    if spec.snr_db is not None:
        sig = np.sqrt(np.mean(y ** 2)) + 1e-12
        y = y + _pink(len(y), rng) * sig * 10 ** (-spec.snr_db / 20)
    y = y / (np.max(np.abs(y)) + 1e-12) * 10 ** (rng.uniform(*spec.gain_db) / 20)
    return y.astype(np.float32)


def make_synthetic(y_ref: np.ndarray, sr: int, spec: SynthSpec, bleed: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (live audio, live_pts, ref_pts): exact piecewise-linear live->ref ground truth."""
    import soxr

    rng = np.random.default_rng(spec.seed)
    src = np.asarray(y_ref, np.float32)
    if bleed is not None and spec.bleed_db is not None:
        g = np.sqrt(np.mean(src ** 2)) / (np.sqrt(np.mean(bleed ** 2)) + 1e-12) * 10 ** (spec.bleed_db / 20)
        src = src + g * bleed[:len(src)]
    live_pts, ref_pts = synth_time_map(len(src) / sr, spec, rng)
    music_live = live_pts[1:] - spec.lead_s if spec.lead_s > 0 else live_pts
    music_ref = ref_pts[1:] if spec.lead_s > 0 else ref_pts
    f = 2.0 ** (spec.transpose / 12.0)
    if spec.transpose:
        z = pv_warp(src, sr, music_live * f, music_ref)
        warped = soxr.resample(z, sr * f, sr, quality="HQ").astype(np.float32)
    else:
        warped = pv_warp(src, sr, music_live, music_ref)
    y = np.concatenate([np.zeros(round(spec.lead_s * sr), np.float32), warped])
    return degrade(y, sr, spec, rng), live_pts, ref_pts


def ref_to_live(t_ref: np.ndarray, live_pts: np.ndarray, ref_pts: np.ndarray) -> np.ndarray:
    """Inverse of a monotone map. A flat run (the lead-in: ref stays at 0 while live advances) keeps
    its last point, where the music actually starts."""
    keep = np.r_[np.diff(ref_pts) > 0, True]
    return np.interp(t_ref, ref_pts[keep], live_pts[keep])


def map_events(t_ref: np.ndarray, live_pts: np.ndarray, ref_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference events -> (live times, ref times) for a map with structural jumps: every monotone
    piece contributes the events inside its reference span (repeated sections occur twice, skipped
    ones not at all)."""
    t_ref = np.asarray(t_ref, float)
    cuts = np.flatnonzero((np.diff(live_pts) < 0.01) & (np.abs(np.diff(ref_pts)) > 1e-9)) + 1
    gl, gr = [], []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(ref_pts)]):
        lp, rp = live_pts[a:b], ref_pts[a:b]
        if len(rp) < 2:
            continue
        ev = t_ref[(t_ref > rp[0]) & (t_ref < rp[-1])]
        gl.append(ref_to_live(ev, lp, rp))
        gr.append(ev)
    if not gl:
        return np.zeros(0), np.zeros(0)
    gl_a, gr_a = np.concatenate(gl), np.concatenate(gr)
    order = np.argsort(gl_a, kind="stable")
    return gl_a[order], gr_a[order]


# ============================================================== pop (steady-tempo band) lives
@dataclass(frozen=True)
class PopSpec:
    """A band performance at a steady tempo: one global tempo ratio with a slow linear drift, the live mix
    rebuilt from the separated stems, performance perturbations on the VOCAL only (the band keeps the time:
    the ground truth follows the band's time map, also through lines the singer drops), then the
    degradations of SynthSpec (lead-in, EQ tilt, reverb, pink noise, gain; its tempo / jump fields unused)."""
    seed: int = 0
    tempo: tuple[float, float] = (0.92, 1.08)    # global ratio (ref s per live s), log-uniform
    drift: float = 0.01                          # the ratio changes linearly by up to +-drift over the song
    drops: int = 0                               # passages of drop_lines consecutive lines left to the crowd
    drop_lines: tuple[int, int] = (1, 3)
    crowd_db: float | None = -10.0               # crowd singing a dropped line (mix; vocal mode: -10 dB more)
    shift_p: float = 0.0                         # share of lines sung early / late ...
    shift_s: tuple[float, float] = (0.2, 0.4)    # ... by this much (never into a neighbouring line)
    adlibs: int = 0                              # lines replaced by an ad-lib: the line pitch-shifted by 3-5
                                                 # semitones, or another line's vocal fitted to its span
    degrade: SynthSpec = field(default_factory=SynthSpec)

    @property
    def perturbed(self) -> bool:
        return bool(self.drops or self.shift_p > 0 or self.adlibs)


def pop_time_map(dur: float, spec: PopSpec, rng: np.random.Generator, step_s: float = 2.0
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Steady-tempo live->ref map as breakpoints (live_pts, ref_pts) in seconds, including the lead-in."""
    rho0 = float(np.exp(rng.uniform(np.log(spec.tempo[0]), np.log(spec.tempo[1]))))
    d = float(rng.uniform(-spec.drift, spec.drift))
    lead = spec.degrade.lead_s
    live, ref = ([0.0, lead], [0.0, 0.0]) if lead > 0 else ([0.0], [0.0])
    while ref[-1] < dur - 1e-9:
        seg = min(step_s, dur - ref[-1])
        rho = rho0 * (1.0 + d * ((ref[-1] + seg / 2) / dur - 0.5))
        ref.append(ref[-1] + seg)
        live.append(live[-1] + seg / rho)
    return np.array(live), np.array(ref)


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64)))) if len(y) else 0.0


def _crowd(seg: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A crowd singing along: delayed copies of the line (0-80 ms), low-passed, in a large reverberant room."""
    from scipy.signal import butter, fftconvolve, sosfilt

    out = np.zeros(len(seg) + round(0.08 * sr))
    for _ in range(4):
        k = int(rng.integers(0, round(0.08 * sr)))
        out[k:k + len(seg)] += seg * rng.uniform(0.6, 1.0)
    out = sosfilt(butter(2, 2500.0, fs=sr, output="sos"), out)
    n = int(1.5 * sr)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    out = 0.3 * out + 0.7 * fftconvolve(out, ir / np.sqrt(np.sum(ir ** 2)))[:len(out)]
    return out[:len(seg)]


def perform_vocal(v: np.ndarray, sr: int, spans: list[tuple[float, float]], spec: PopSpec,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """Perturb a vocal (already on the band's live time axis) line by line; spans = the lines' sung
    [start, end] in live seconds. Each line owns the audio between the midpoints of its neighbouring gaps:
    moved as a block (early / late), silenced (dropped: a crowd version is returned separately) or
    replaced (ad-lib). -> (vocal, crowd, info with the line indices / shifts used)."""
    import librosa

    n = len(v)
    a = np.array([s for s, _ in spans])
    b = np.array([e for _, e in spans])
    K = len(spans)
    cut = np.empty(K + 1)
    cut[0], cut[K] = max(0.0, a[0] - 0.5), min(n / sr, b[-1] + 0.5)
    for k in range(1, K):
        cut[k] = min(a[k], 0.5 * (b[k - 1] + a[k]))
    cs = np.clip(np.round(cut * sr).astype(int), 0, n)
    # early / late phrasing, never into the (shifted) neighbouring lines
    shift = np.zeros(K)
    for k in range(K):
        if rng.random() >= spec.shift_p:
            continue
        mag, sign = rng.uniform(*spec.shift_s), (1.0 if rng.random() < 0.5 else -1.0)
        room = {-1.0: a[k] - (b[k - 1] + shift[k - 1] if k else 0.0) - 0.05,
                1.0: (a[k + 1] if k + 1 < K else n / sr) - b[k] - 0.05}
        for sg in (sign, -sign):
            m = min(mag, room[sg])
            if m >= spec.shift_s[0]:
                shift[k] = sg * m
                break
    # dropped passages (never the first line: the tracker has locked once) and ad-libs
    ok = [k for k in range(1, K) if b[k] - a[k] >= 0.8]
    dropped: list[int] = []
    for _ in range(spec.drops):
        cand = [k for k in ok if all(abs(k - d) > 3 for d in dropped)]
        if not cand:
            break
        k0 = int(rng.choice(cand))
        dropped += [k for k in range(k0, min(K, k0 + int(rng.integers(spec.drop_lines[0], spec.drop_lines[1] + 1))))]
    adlib: dict[int, str] = {}
    for _ in range(spec.adlibs):
        cand = [k for k in ok if k not in dropped and k not in adlib and b[k] - a[k] >= 1.0]
        if not cand:
            break
        adlib[int(rng.choice(cand))] = "pitch" if rng.random() < 0.5 else "other"
    fade = max(1, round(0.01 * sr))
    out = np.zeros(n, np.float64)
    crowd = np.zeros(n, np.float64)
    out[:cs[0]] = v[:cs[0]]
    out[cs[K]:] = v[cs[K]:]
    for k in range(K):
        seg = np.asarray(v[cs[k]:cs[k + 1]], np.float64).copy()
        if len(seg) == 0:
            continue
        if k in adlib:
            i0, i1 = round(a[k] * sr) - cs[k], round(b[k] * sr) - cs[k]
            i0, i1 = max(0, i0), min(len(seg), i1)
            sung = seg[i0:i1]
            if adlib[k] == "pitch":
                st = float(rng.choice([-5, -4, -3, 3, 4, 5]))
                with warnings.catch_warnings():          # librosa 1.0: deprecated internal arguments and a
                    warnings.simplefilter("ignore", FutureWarning)      # spurious cast warning in its phasor
                    warnings.simplefilter("ignore", RuntimeWarning)
                    new = librosa.effects.pitch_shift(sung.astype(np.float32), sr=sr, n_steps=st)
            else:
                j = int(rng.choice([x for x in range(K) if x != k and b[x] - a[x] >= 0.5]))
                src = v[round(a[j] * sr):round(b[j] * sr)]
                dj, dk = len(src) / sr, len(sung) / sr
                new = pv_warp(np.asarray(src, np.float32), sr, np.array([0.0, dk]), np.array([0.0, dj]))
                new = new[:len(sung)] * (_rms(sung) / max(_rms(new), 1e-9))
            seg[i0:i0 + len(new)] = new[:i1 - i0]
        w = np.ones(len(seg))
        f = min(fade, len(seg) // 2)
        if f:
            w[:f], w[-f:] = np.linspace(0, 1, f), np.linspace(1, 0, f)
        if k in dropped:
            if spec.crowd_db is not None:
                crowd[cs[k]:cs[k + 1]] += _crowd(seg, sr, rng) * w
            continue
        p = cs[k] + round(shift[k] * sr)
        lo, hi = max(0, p), min(n, p + len(seg))
        if hi > lo:
            out[lo:hi] += (seg * w)[lo - p:hi - p]
    info = {"dropped": sorted(dropped), "adlib": {int(k): t for k, t in sorted(adlib.items())},
            "shift": [round(float(x), 3) for x in shift]}
    return out.astype(np.float32), crowd.astype(np.float32), info


def make_pop_live(vocals: np.ndarray, inst: np.ndarray, sr: int, spec: PopSpec, spans_ref: list[tuple[float, float]],
                  mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """-> (live audio, live_pts, ref_pts, perturbation info). The band (instrumental stem) and the vocal stem
    follow the same steady time map; the vocal is then perturbed per line (perform_vocal). mix: band + vocal
    (+ crowd); vocal: the vocal with the band bleeding in at -20 dB (+ the crowd 10 dB under the mix level).
    The time map and the degradations come from their own random streams: a clean and a perturbed spec with
    the same seed differ only in the vocal perturbations."""
    rng_map, rng_perf, rng_deg = (np.random.default_rng([spec.seed, i]) for i in range(3))
    n = min(len(vocals), len(inst))
    live_pts, ref_pts = pop_time_map(n / sr, spec, rng_map)
    lead = spec.degrade.lead_s
    music_live = live_pts[1:] - lead if lead > 0 else live_pts
    music_ref = ref_pts[1:] if lead > 0 else ref_pts
    wi = pv_warp(np.asarray(inst[:n], np.float32), sr, music_live, music_ref)
    wv = pv_warp(np.asarray(vocals[:n], np.float32), sr, music_live, music_ref)
    info: dict = {"tempo": float((ref_pts[-1] - ref_pts[0]) / (live_pts[-1] - live_pts[0] - lead))}
    crowd = np.zeros_like(wv)
    if spec.perturbed:
        sp = sorted((s, e) for s, e in spans_ref if s is not None and e is not None and e > s)
        spans = [(float(ref_to_live(np.array([s]), music_live, music_ref)[0]),
                  float(ref_to_live(np.array([e]), music_live, music_ref)[0])) for s, e in sp]
        wv, crowd, pinfo = perform_vocal(wv, sr, spans, spec, rng_perf)
        info.update(pinfo, spans_live=[(round(s + lead, 3), round(e + lead, 3)) for s, e in spans])
    cg = 0.0 if spec.crowd_db is None else 10 ** ((spec.crowd_db - (10.0 if mode == "vocal" else 0.0)) / 20)
    crowd = crowd * (_rms(wv) / max(_rms(crowd[crowd != 0]), 1e-9)) * cg if crowd.any() else crowd
    if mode == "mix":
        y = wi + wv + crowd
    else:
        y = wv + wi * (_rms(wv) / max(_rms(wi), 1e-9)) * 10 ** (-20.0 / 20) + crowd
    y = np.concatenate([np.zeros(round(lead * sr), np.float32), y.astype(np.float32)])
    return degrade(y, sr, spec.degrade, rng_deg), live_pts, ref_pts, info


# ============================================================== pairs
@dataclass
class Pair:
    name: str
    kind: str                         # "real" | "synth" | "pop"
    ref_audio: dict[str, Path]        # mode -> reference audio
    live_audio: dict[str, Path]       # mode -> live audio (real pairs)
    gt_ref: np.ndarray
    gt_live: np.ndarray | None = None  # real pairs; synthetic: derived from the map
    line_starts: list[float] = field(default_factory=list)
    synth: SynthSpec | None = None
    bleed: Path | None = None         # instrumental stem for vocal-mode synthetic bleed
    transpose_preset: int = 0         # operator-preset transposition given to the tracker
    note: str = ""
    modes: tuple[str, ...] = ("mix", "vocal")
    pop: PopSpec | None = None        # pop pairs: the live is rebuilt from the stems (make_pop_live) ...
    stems: dict[str, Path] = field(default_factory=dict)              # ... "vocals" + "instrumental"
    line_spans: list[tuple[float, float]] = field(default_factory=list)   # ... sung [start, end] per line (ref s)


def _stems(name: str) -> Path:
    return DATA_DIR / "stems" / name


# pop songs: JamendoLyrics (EN, htdemucs stems; events = word starts) and two J-pop songs prepared in songs/
# (Yorushika; read-only: BS-RoFormer stems, events = the aligned syllable starts). Tuning used embers, feel,
# hila and replicant; badside and rokugatsu were held out (docs/eval/TRACKING.md §8).
JAMENDO_POP = {"embers": "Avercage_-_Embers", "feel": "Cortez_-_Feel__Stripped_",
               "hila": "HILA_-_Give_Me_the_Same", "badside": "Rxbyn_-_Bad_Side"}
SONG_POP = {"replicant": "06-レプリカント", "rokugatsu": "07-六月は雨上がりの街を書く"}
POP_SCENARIOS = {                     # name prefix -> spec (seed added per song)
    "steady": PopSpec(),                                          # constant tempo (+ drift), clean performance
    "pop": PopSpec(drops=2, shift_p=0.5, adlibs=2),               # + dropped lines, early / late lines, ad-libs
}


def _pop_songs() -> dict[str, dict]:
    """Pop songs available locally -> {mix, vocals, instrumental, events, line_starts, line_spans, note}."""
    out: dict[str, dict] = {}
    jam = DATA_DIR / "research-testsets" / "jamendo"
    for short, stem in JAMENDO_POP.items():
        sep = jam / "sep" / "htdemucs" / stem
        mp3, voc, inst = jam / "mp3" / f"{stem}.mp3", sep / "vocals.wav", sep / "no_vocals.wav"
        if not (mp3.is_file() and voc.is_file() and inst.is_file()):
            continue
        with open(jam / "words" / f"{stem}.csv", encoding="utf-8") as fh:
            words = np.array([float(r["word_start"]) for r in csv.DictReader(fh)])
        with open(jam / "lines" / f"{stem}.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        out[short] = {"mix": mp3, "vocals": voc, "instrumental": inst, "events": words,
                      "line_starts": [float(r["start_time"]) for r in rows],
                      "line_spans": [(float(r["start_time"]), float(r["end_time"])) for r in rows],
                      "note": "JamendoLyrics pop (EN)"}
    from ..songfile import load_song, stem_path

    for short, sid in SONG_POP.items():
        sdir = SONGS_DIR / sid
        paths = {k: stem_path(sdir, k) for k in ("mix", "vocals", "instrumental")}
        if not ((sdir / "song.json").is_file() and all(p.is_file() for p in paths.values())):
            continue
        try:
            song = load_song(sdir)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        timed = [ln for ln in song.lines if ln.start is not None and ln.end is not None]
        events = np.array(sorted(s.start for ln in timed for s in ln.syllables() if s.start is not None))
        if len(events) == 0:
            continue
        out[short] = {**paths, "events": events, "line_starts": [float(ln.start) for ln in timed],
                      "line_spans": [(float(ln.start), float(ln.end)) for ln in timed],
                      "note": f"J-pop (JA), {sid}"}
    return out


def pop_pairs() -> dict[str, Pair]:
    """steady_<song> / pop_<song> for every pop song available locally (see POP_SCENARIOS)."""
    pairs: dict[str, Pair] = {}
    for i, (short, s) in enumerate(_pop_songs().items()):
        seed = 61 + i if short in JAMENDO_POP else 71 + list(SONG_POP).index(short)
        for pre, spec in POP_SCENARIOS.items():
            k = f"{pre}_{short}"
            what = "steady tempo" + (", dropped / early-late / ad-lib lines" if spec.perturbed else "")
            pairs[k] = Pair(k, "pop", {"mix": s["mix"], "vocal": s["vocals"]}, {}, s["events"], None,
                            s["line_starts"], pop=replace(spec, seed=seed),
                            stems={"vocals": s["vocals"], "instrumental": s["instrumental"]},
                            line_spans=s["line_spans"], note=f"{s['note']}, {what}")
    return pairs


def available_pairs() -> dict[str, Pair]:
    """All pairs whose audio exists locally."""
    res = (ref_repo_dir() or Path("/nonexistent")) / "resources"
    pairs: dict[str, Pair] = {}
    audio, lyr = res / "audio", res / "lyrics"
    if (audio / "Edelweiss.wav").is_file() and (audio / "Edelweiss_short.wav").is_file():
        full, short = parse_lrc(lyr / "Edelweiss.lrc"), parse_lrc(lyr / "Edelweiss_short.lrc")
        n = 26
        assert [x["text"] for x in full[:n]] == [x["text"] for x in short[:n]]
        ls = lrc_line_starts(full)
        ref_a = {"mix": audio / "Edelweiss.wav", "vocal": _stems("edelweiss") / "vocals.wav"}
        pairs["edelweiss"] = Pair("edelweiss", "real", ref_a,
                                  {"mix": audio / "Edelweiss_short.wav",
                                   "vocal": _stems("edelweiss_short") / "vocals.wav"},
                                  np.array([x["t"] for x in full[:n]]), np.array([x["t"] for x in short[:n]]),
                                  ls, transpose_preset=2,
                                  note="second performance (+2 semitones: C vs Bb major), 26 shared LRC tokens")
        for k, (spec, tp) in {"synth_edelweiss": (SynthSpec(seed=11), 0),
                              "synth_edelweiss_t+1": (SynthSpec(seed=12, transpose=1), 1)}.items():
            pairs[k] = Pair(k, "synth", ref_a, {}, np.array([x["t"] for x in full]), None, ls, spec,
                            bleed=_stems("edelweiss") / "instrumental.wav", transpose_preset=tp)
    ks = "꽃밭에서"
    if (audio / f"{ks}_ref.wav").is_file() and (audio / f"{ks}_score.wav").is_file():
        syl = score_csv_to_ref_gt(res / "annots" / f"{ks}_score.csv", res / "alignments" / f"{ks}_offline.csv")
        firsts = group_syllables_by_lines(lyr / f"{ks}_lyrics.txt", len(syl))
        t_ref = np.array([s["t_ref"] for s in syl])
        t_score = np.array([s["t_score"] for s in syl])
        ref_a = {"mix": audio / f"{ks}_ref.wav", "vocal": _stems("kkotbat_ref") / "vocals.wav"}
        sc_a = {"mix": audio / f"{ks}_score.wav", "vocal": _stems("kkotbat_score") / "vocals.wav"}
        pairs["kkotbat_r2s"] = Pair("kkotbat_r2s", "real", ref_a, sc_a, t_ref, t_score,
                                    [float(t_ref[i]) for i in firsts], modes=("mix",),
                                    note="live = MuseScore render (no sung voice); path GT")
        pairs["kkotbat_s2r"] = Pair("kkotbat_s2r", "real", sc_a, ref_a, t_score, t_ref,
                                    [float(t_score[i]) for i in firsts], modes=("mix",),
                                    note="ref = MuseScore render (no sung voice); path GT")
        for k, (spec, tp) in {"synth_kkotbat": (SynthSpec(seed=21), 0),
                              "synth_kkotbat_t-1": (SynthSpec(seed=22, transpose=-1), -1)}.items():
            pairs[k] = Pair(k, "synth", ref_a, {}, t_ref, None, [float(t_ref[i]) for i in firsts], spec,
                            bleed=_stems("kkotbat_ref") / "instrumental.wav", transpose_preset=tp)
    jam = DATA_DIR / "research-testsets" / "jamendo"
    for i, (short, stem) in enumerate({"embers": "Avercage_-_Embers", "feel": "Cortez_-_Feel__Stripped_",
                                       "hila": "HILA_-_Give_Me_the_Same", "badside": "Rxbyn_-_Bad_Side"}.items()):
        mp3, voc = jam / "mp3" / f"{stem}.mp3", jam / "sep" / "htdemucs" / stem / "vocals.wav"
        if not (mp3.is_file() and voc.is_file()):
            continue
        with open(jam / "words" / f"{stem}.csv", encoding="utf-8") as fh:
            words = np.array([float(r["word_start"]) for r in csv.DictReader(fh)])
        with open(jam / "lines" / f"{stem}.csv", encoding="utf-8") as fh:
            ls = [float(r["start_time"]) for r in csv.DictReader(fh)]
        k = f"synth_{short}"
        bleed = jam / "sep" / "htdemucs" / stem / "no_vocals.wav"
        pairs[k] = Pair(k, "synth", {"mix": mp3, "vocal": voc}, {}, words, None, ls, SynthSpec(seed=31 + i),
                        bleed=bleed, note="JamendoLyrics pop (EN)")
        edit = {"hila": ("skip", ((80.0, 25.0),)), "badside": ("repeat", ((120.0, -30.0),))}.get(short)
        if edit:
            k = f"synth_{short}_{edit[0]}"
            pairs[k] = Pair(k, "synth", {"mix": mp3, "vocal": voc}, {}, words, None, ls,
                            SynthSpec(seed=41 + i, jumps=edit[1]), bleed=bleed,
                            note=f"JamendoLyrics pop (EN), structural edit: {edit[0]} {edit[1][0][1]:+.0f} s")
    pairs.update(pop_pairs())
    return pairs


# ============================================================== running
_AUDIO: dict[tuple, np.ndarray] = {}
_TUNING: dict[tuple, float] = {}


def _load(path: Path) -> np.ndarray:
    key = (str(path),)
    if key not in _AUDIO:
        _AUDIO[key] = load_mono(path, SR)
    return _AUDIO[key]


_SYNTH: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
POP_INFO: dict[tuple, dict] = {}      # (pair, mode) -> perturbation info of a generated pop live


def live_and_gt(pair: Pair, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (live audio, gt_live, gt_ref) restricted to events inside both signals."""
    if pair.kind == "real":
        y = _load(pair.live_audio[mode])
        gl, gr = pair.gt_live, pair.gt_ref
    elif pair.kind == "pop":
        key = (pair.name, mode)
        if key not in _SYNTH:
            y, lp, rp, info = make_pop_live(_load(pair.stems["vocals"]), _load(pair.stems["instrumental"]), SR,
                                            pair.pop or PopSpec(), pair.line_spans, mode)
            _SYNTH[key], POP_INFO[key] = (y, lp, rp), info
        y, lp, rp = _SYNTH[key]
        gl, gr = map_events(pair.gt_ref, lp, rp)
    else:
        key = (pair.name, mode)
        if key not in _SYNTH:
            spec = pair.synth or SynthSpec()
            bleed = None
            if mode == "vocal" and pair.bleed is not None and pair.bleed.is_file():
                bleed = _load(pair.bleed)
                spec = replace(spec, bleed_db=-20.0)
            _SYNTH[key] = make_synthetic(_load(pair.ref_audio[mode]), SR, spec, bleed)
        y, lp, rp = _SYNTH[key]
        gl, gr = map_events(pair.gt_ref, lp, rp)
    ok = (gl < len(y) / SR) & (gl >= 0)
    return y, np.asarray(gl)[ok], np.asarray(gr)[ok]


def song_tuning(y: np.ndarray) -> float:
    import librosa

    return float(librosa.estimate_tuning(y=y, sr=SR, n_fft=2048))


def pair_tuning(pair: Pair) -> float:
    """Tuning of the pair's reference mix (cached per file)."""
    tkey = (str(pair.ref_audio["mix"]),)
    if tkey not in _TUNING:
        _TUNING[tkey] = song_tuning(_load(pair.ref_audio["mix"]))
    return _TUNING[tkey]


@dataclass
class StreamBlocks:
    t_pub: np.ndarray                 # publish time (s) after each block
    frames: list[FeatureFrames]
    feat_s: np.ndarray                # extractor time per block


def stream_features(y: np.ndarray, cfg: FeatureConfig, block: int = BLOCK) -> StreamBlocks:
    ext = FeatureExtractor(cfg)
    t_pub, frames, dt = [], [], []
    for i in range(0, len(y), block):
        t0 = time.perf_counter()
        ff = ext.push(y[i:i + block])
        dt.append(time.perf_counter() - t0)
        frames.append(ff)
        t_pub.append(min(i + block, len(y)) / SR)
    return StreamBlocks(np.array(t_pub), frames, np.array(dt))


def run_tracker(ref: FeatureFrames, blocks: StreamBlocks, params: TrackerParams, fps: float, *,
                line_starts: list[float] | None = None, transpose: int = 0, start_s: float = 0.0) -> dict:
    trk = LiveTracker(ref, fps, params, line_starts=line_starts)
    if transpose:
        trk.set_transpose(transpose)
    trk.start(start_s)
    pos, conf, status, dt, nfr = [], [], [], [], []
    for ff in blocks.frames:
        t0 = time.perf_counter()
        st = trk.push(ff)
        dt.append(time.perf_counter() - t0)
        nfr.append(len(ff))
        pos.append(st.ref_time)
        conf.append(st.confidence)
        status.append(st.status)
    return {"pos": np.array(pos), "conf": np.array(conf), "status": np.array(status), "trk_s": np.array(dt),
            "n_frames": np.array(nfr), "relocations": trk.relocations, "key_switches": trk.key_switches,
            "final_key": trk.state.transpose, "snaps": trk.snaps, "reanchors": trk.reanchors}


_REF: dict[tuple, FeatureFrames] = {}
_LIVE: dict[tuple, StreamBlocks] = {}


def evaluate_pair(pair: Pair, mode: str, fcfg: FeatureConfig | None = None, tparams: TrackerParams | None = None,
                  *, transpose: int | None = None, keep_trace: bool = False) -> dict:
    """Stream one pair through the tracker and score it. `fcfg` is used as given (its tuning
    included); None means the mode defaults with the tuning estimated from the reference mix."""
    y_ref = _load(pair.ref_audio[mode])
    if fcfg is None:
        fcfg = FeatureConfig.for_mode(mode, tuning=pair_tuning(pair))
    tparams = tparams or TrackerParams.for_mode(mode)
    y_live, gt_live, gt_ref = live_and_gt(pair, mode)
    rk = (pair.name, mode, fcfg.key(), "ref")
    if rk not in _REF:
        _REF[rk] = compute_offline(y_ref, fcfg)
    lk = (pair.name, mode, fcfg.key(), "live")
    if lk not in _LIVE:
        _LIVE[lk] = stream_features(y_live, fcfg)
    ref, blocks = _REF[rk], _LIVE[lk]
    tp = pair.transpose_preset if transpose is None else transpose
    r = run_tracker(ref, blocks, tparams, fcfg.fps, line_starts=pair.line_starts, transpose=tp)
    m = prompter_metrics(blocks.t_pub, r["pos"], gt_live, gt_ref)
    m["conf_auroc"] = confidence_auroc(blocks.t_pub, r["pos"], r["conf"], gt_live, gt_ref)
    m["relocations"], m["snaps"], m["reanchors"] = r["relocations"], r["snaps"], r["reanchors"]
    m["key_switches"], m["final_key"], m["key_preset"] = r["key_switches"], r["final_key"], tp
    has = r["n_frames"] > 0
    per_frame_feat = blocks.feat_s[has] / r["n_frames"][has]
    per_frame_trk = r["trk_s"][has] / r["n_frames"][has]
    m.update(pair=pair.name, mode=mode, style=tparams.style, kind=pair.kind, live_s=round(len(y_live) / SR, 2),
             feat_ms_mean=1e3 * float(per_frame_feat.mean()), feat_ms_p99=1e3 * float(np.percentile(per_frame_feat, 99)),
             trk_ms_mean=1e3 * float(per_frame_trk.mean()), trk_ms_p99=1e3 * float(np.percentile(per_frame_trk, 99)),
             conf_mean=float(r["conf"].mean()),
             coast_frac=float(np.mean(r["status"] == "coasting")))
    if keep_trace:
        m["trace"] = {"t_pub": blocks.t_pub, "pos": r["pos"], "conf": r["conf"], "status": r["status"],
                          "gt_live": gt_live, "gt_ref": gt_ref}
    return m


def summarize(rows: list[dict]) -> dict:
    """Means per mode ("mix" / "vocal"), per "style/mode" when the rows hold more than one style."""
    keys = ["pos_AAE_ms", "pos_MedAE_ms", "lat_median_ms", "lost_frac"] + [f"PCO@{t}" for t in TOLERANCES_MS]
    styles = {r.get("style", "classic") for r in rows}
    out = {}
    for style in sorted(styles):
        for mode in sorted({r["mode"] for r in rows}):
            rs = [r for r in rows if r["mode"] == mode and r.get("style", "classic") == style]
            if not rs:
                continue
            k = mode if len(styles) == 1 else f"{style}/{mode}"
            out[k] = {c: float(np.nanmean([r[c] for r in rs])) for c in keys}
            out[k]["n_pairs"] = len(rs)
    return out


def _parse_value(v: str, tp: object) -> object:
    """Parse a CLI string for a dataclass field annotation (bool/int/float/str, tuples, X | None)."""
    args = typing.get_args(tp)
    if type(None) in args:                                   # Optional[...] / X | None
        if v.strip().lower() in ("", "none", "null"):
            return None
        inner = [a for a in args if a is not type(None)]
        return _parse_value(v, inner[0] if len(inner) == 1 else str)
    origin = typing.get_origin(tp)
    if origin is tuple:
        elem = args[0] if args else float
        return tuple(_parse_value(x.strip(), elem) for x in v.split(","))
    if tp is bool:
        if v.strip().lower() not in ("1", "0", "true", "false", "yes", "no"):
            raise ValueError(f"not a boolean: {v!r}")
        return v.strip().lower() in ("1", "true", "yes")
    if tp in (int, float, str):
        return tp(v)
    return v


def _parse_overrides(items: list[str], cls) -> dict:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        if k not in names:
            raise SystemExit(f"unknown {cls.__name__} field {k!r}")
        try:
            out[k] = _parse_value(v, hints[k])
        except ValueError as e:
            raise SystemExit(f"bad value for {cls.__name__}.{k}: {e}") from None
    return out


def format_table(rows: list[dict]) -> str:
    cols = [("pair", "{:<20}"), ("mode", "{:<5}"), ("style", "{:<7}"), ("n", "{:>4}"), ("pos_AAE_ms", "{:>8.0f}"),
            ("pos_MedAE_ms", "{:>7.0f}"), ("pos_bias_ms", "{:>7.0f}"), ("lat_median_ms", "{:>7.0f}"),
            ("PCO@200", "{:>6.1f}"), ("PCO@300", "{:>6.1f}"), ("PCO@500", "{:>6.1f}"), ("PCO@1000", "{:>6.1f}"),
            ("lost_frac", "{:>5.2f}"), ("conf_mean", "{:>5.2f}"), ("trk_ms_p99", "{:>6.2f}"),
            ("final_key", "{:>+4d}"), ("key_switches", "{:>3d}")]
    lines = [" ".join(c[:8] for c, _ in cols)]
    for r in rows:
        lines.append(" ".join(f.format(r[c]) if c in r else "-" for c, f in cols))
    return "\n".join(lines)


def cli(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lyricprompter eval-track", description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", default="all", help="comma-separated pair names (see --list) or 'all'")
    ap.add_argument("--modes", default="mix,vocal")
    ap.add_argument("--json", type=Path, help="write per-pair results + summary as JSON")
    ap.add_argument("--list", action="store_true", help="list available pairs and exit")
    ap.add_argument("--feat", action="append", default=[], metavar="K=V", help="FeatureConfig override")
    ap.add_argument("--trk", action="append", default=[], metavar="K=V", help="TrackerParams override")
    ap.add_argument("--no-key-preset", action="store_true",
                    help="ignore the pairs' operator key presets (tests automatic key detection)")
    ap.add_argument("--styles", default="classic",
                    help="comma-separated tracking styles to run every pair with (classic, pop)")
    a = ap.parse_args(argv)
    pairs = available_pairs()
    if a.list:
        for k, p in pairs.items():
            print(f"{k:<22} {p.kind:<6} events={len(p.gt_ref):<4} {p.note}")
        return 0
    names = list(pairs) if a.pairs == "all" else [x.strip() for x in a.pairs.split(",") if x.strip()]
    missing = [n for n in names if n not in pairs]
    if missing:
        print(f"unknown / unavailable pairs: {missing}; available: {list(pairs)}", file=sys.stderr)
        return 2
    styles = [x.strip() for x in a.styles.split(",") if x.strip()]
    bad = [x for x in styles if x not in STYLES]
    if bad or not styles:
        print(f"unknown styles: {bad}; expected some of {list(STYLES)}", file=sys.stderr)
        return 2
    fo, to = _parse_overrides(a.feat, FeatureConfig), _parse_overrides(a.trk, TrackerParams)
    rows = []
    for style in styles:
        for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
            for n in names:
                p = pairs[n]
                if mode not in p.modes:
                    continue
                live_ok = (p.live_audio[mode].is_file() if p.kind == "real" else
                           all(x.is_file() for x in p.stems.values()) if p.kind == "pop" else True)
                if not (p.ref_audio[mode].is_file() and live_ok):
                    print(f"skip {n}/{mode}: audio missing", file=sys.stderr)
                    continue
                fcfg = replace(FeatureConfig.for_mode(mode, tuning=pair_tuning(p)), **fo)
                tpar = replace(TrackerParams.for_mode(mode, style=style), **to)
                rows.append(evaluate_pair(p, mode, fcfg, tpar, transpose=0 if a.no_key_preset else None))
                rows[-1].setdefault("style", style)
                print(format_table([rows[-1]]).splitlines()[-1], flush=True)
    print(format_table(rows))
    summ = summarize(rows)
    print(json.dumps(summ, indent=1))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"rows": rows, "summary": summ,
                                      "feature_overrides": fo, "tracker_overrides": to}, indent=1, default=str))
    return 0


__all__ = [
    "Pair",
    "SynthSpec",
    "available_pairs",
    "cli",
    "evaluate_pair",
    "make_synthetic",
    "parse_lrc",
    "prompter_metrics",
    "pv_warp",
    "score_csv_to_ref_gt",
    "synth_time_map",
]
