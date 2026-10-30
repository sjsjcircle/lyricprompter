"""Synthetic audio with exact ground truth, for tests, demos and `oscsender eval`.

Two independent parts:

* **Synthetic live takes of a real reference** (`SynthSpec`, `synth_time_map`, `pv_warp`, `_pink`, `degrade`,
  `make_synthetic`, `ref_to_live`, `map_events`), copied unchanged from lyricprompter.eval.track_eval: a
  piecewise-constant tempo (time-stretched with a phase vocoder), optional transposition, EQ tilt, reverb, pink
  noise, a gain change and a noise lead-in. The breakpoints (live_pts, ref_pts) give the live <-> reference
  ground truth.
* **Synthetic music** (`make_score`, `render_song`): a deterministic song per seed (triad harmony that changes
  every 0.5-1.5 s, a bass line, a melody, kick / snare / hi-hat, a few full-band rests) rendered by additive
  wavetable synthesis. `render_song(score, tempo_map=(score_pts, perf_pts))` renders the *same notes* at
  performance times instead of warping audio: every note's onset and end go through the piecewise-linear map
  (`score_to_perf`), so a "live" take has an exact ground truth (`perf_to_score`) without phase-vocoder
  artefacts. Fixed gains, no normalisation (a re-rendered take keeps the reference's levels): the mix sits
  around -18 dBFS RMS with peaks around -3 dBFS (below -1 dBFS for every seed tried).
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np

VOICES = ("pad", "bass", "lead", "kick", "snare", "hat")
PITCHED = frozenset({0, 1, 2})          # voice indices with a pitch (transposed by render_song)
NOTE_DTYPE = np.dtype([("start", "f8"), ("dur", "f8"), ("midi", "f8"), ("amp", "f8"), ("voice", "i1")])

_TABLE_N = 2048                         # wavetable length (one fundamental period)
_NOISE_SEED = 0x5EED                    # drum noise bank: identical for every render of a note
# voice -> (harmonic amplitude law, max harmonics, attack s, decay tau s, sustain level, release s)
_PITCHED_VOICES = {
    0: (lambda h: h ** -1.5, 6, 0.03, 0.40, 0.55, 0.08),                        # pad: soft, sustained
    1: (lambda h: h ** -2.0, 5, 0.005, 0.20, 0.45, 0.05),                       # bass
    2: (lambda h: (1.0 if h % 2 else 0.45) / h, 8, 0.01, 0.25, 0.50, 0.05),    # lead: reedy (odd partials)
}
_DRUM_RELEASE_S = 0.005


@dataclass(eq=False)
class Score:
    """Notes on the score time axis (seconds). `render_song` turns it into audio."""
    notes: np.ndarray                    # NOTE_DTYPE records, sorted by start
    duration: float                      # s; the rendered length is score_to_perf(duration)
    bpm: float = 120.0
    key: int = 0                         # tonic pitch class (0 = C)
    minor: bool = False
    seed: int | None = None
    rests: tuple[tuple[float, float], ...] = ()     # full-band rests [start, end)
    chords: np.ndarray = field(default_factory=lambda: np.zeros(0))   # harmony change times

    def onsets(self, voice: str | None = None) -> np.ndarray:
        """Sorted unique note start times (of one voice, or of all)."""
        n = self.notes if voice is None else self.notes[self.notes["voice"] == VOICES.index(voice)]
        return np.unique(n["start"])


def make_notes(rows) -> np.ndarray:
    """[(start, dur, midi, amp, voice name), ...] -> a NOTE_DTYPE array sorted by start (for hand-made scores)."""
    arr = np.array([(float(s), float(d), float(m), float(a), VOICES.index(v)) for s, d, m, a, v in rows],
                   dtype=NOTE_DTYPE)
    return arr[np.argsort(arr["start"], kind="stable")]


# ------------------------------------------------------------------------------------- tempo maps


def _map_points(tempo_map) -> tuple[np.ndarray, np.ndarray] | None:
    if tempo_map is None:
        return None
    try:
        xs, ys = (np.asarray(p, dtype=np.float64) for p in tempo_map)
    except (TypeError, ValueError):
        raise ValueError("a tempo map is (score_pts, perf_pts)") from None
    if xs.ndim != 1 or xs.shape != ys.shape or len(xs) < 2:
        raise ValueError("a tempo map is (score_pts, perf_pts): two equally long lists of >= 2 times")
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        raise ValueError("tempo map times must be finite")
    if not (np.all(np.diff(xs) > 0) and np.all(np.diff(ys) > 0)):
        raise ValueError("tempo map points must be strictly increasing on both axes")
    return xs, ys


def _piecewise(t, xp: np.ndarray, fp: np.ndarray):
    """Piecewise-linear through (xp, fp), extended linearly beyond both ends (first / last slope)."""
    ta = np.asarray(t, dtype=np.float64)
    y = np.interp(ta, xp, fp)
    y = np.where(ta < xp[0], fp[0] + (ta - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]), y)
    y = np.where(ta > xp[-1], fp[-1] + (ta - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), y)
    return float(y) if np.ndim(t) == 0 else y


def score_to_perf(t, tempo_map=None):
    """Score time(s) -> performance time(s) under `tempo_map` = (score_pts, perf_pts) (None: identity)."""
    pts = _map_points(tempo_map)
    if pts is None:
        return float(t) if np.ndim(t) == 0 else np.asarray(t, dtype=np.float64).copy()
    return _piecewise(t, *pts)


def perf_to_score(t, tempo_map=None):
    """Inverse of score_to_perf: performance time(s) -> score time(s)."""
    pts = _map_points(tempo_map)
    if pts is None:
        return float(t) if np.ndim(t) == 0 else np.asarray(t, dtype=np.float64).copy()
    return _piecewise(t, pts[1], pts[0])


# ---------------------------------------------------------------------------------------- score


def make_score(seconds: float, seed: int = 0) -> Score:
    """A deterministic song of `seconds` (score time) for seed `seed`.

    A random key (major / natural minor) and tempo (100-124 bpm); diatonic triads changing every 0.5-1.5 s
    (on the eighth-note grid, never the same chord twice in a row) played by a pad, a bass line on the beats
    (root / fifth), a melody (random walk on the scale, eighth to whole-beat notes, a few melodic rests),
    kick on 1 and 3 (plus random off-beat kicks), snare on 2 and 4 (plus fills), hi-hats on most eighths,
    and one short full-band rest (1-2 beats) per 25 s.
    """
    seconds = float(seconds)
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"the score length must be > 0 s, got {seconds!r}")
    rng = np.random.default_rng(seed)
    bpm = float(rng.uniform(100.0, 124.0))
    beat = 60.0 / bpm
    half = beat / 2
    key = int(rng.integers(0, 12))
    minor = bool(rng.random() < 0.4)
    scale = np.array([0, 2, 3, 5, 7, 8, 10] if minor else [0, 2, 4, 5, 7, 9, 11])
    rows: list[tuple[float, float, float, float, int]] = []

    def vel() -> float:
        return float(rng.uniform(0.8, 1.0))

    # harmony: pad triads + bass
    lengths = [k for k in range(1, 8) if 0.5 - 1e-9 <= k * half <= 1.5 + 1e-9]
    degrees, probs = np.arange(6), np.array([0.24, 0.12, 0.10, 0.20, 0.20, 0.14])
    chords, t, prev = [], 0.0, -1
    while t < seconds - 1e-9:
        d = int(rng.choice(lengths)) * half
        p = np.where(degrees == prev, 0.0, probs)
        deg = int(rng.choice(degrees, p=p / p.sum()))
        prev = deg
        chords.append(t)
        pcs = [(key + scale[(deg + i) % 7]) % 12 for i in (0, 2, 4)]
        for pc in pcs:
            rows.append((t, d, 55 + (pc - 55) % 12, 0.08 * vel(), 0))
        root = 36 + (pcs[0] - 36) % 12
        fifth = 36 + (pcs[2] - 36) % 12
        bt, i = t, 0
        while bt < t + d - 1e-9:
            slot = min(beat, t + d - bt)
            rows.append((bt, 0.9 * slot, root if i % 2 == 0 else fifth, 0.17 * vel(), 1))
            bt, i = bt + slot, i + 1
        t += d

    # melody: random walk on the scale over two octaves
    pool = np.array([m for m in range(60, 85) if (m - key) % 12 in scale and 62 <= m <= 81])
    idx = len(pool) // 2
    t = float(rng.choice([0.0, beat, 2 * beat]))
    while t < seconds - 1e-9:
        k = int(rng.choice([1, 2, 2, 3, 4], p=[0.25, 0.3, 0.2, 0.15, 0.1]))
        d = k * half
        if rng.random() >= 0.12:
            idx = int(np.clip(idx + rng.choice([-3, -2, -1, -1, 0, 1, 1, 2, 3]), 0, len(pool) - 1))
            rows.append((t, 0.9 * d, float(pool[idx]), 0.13 * vel(), 2))
        t += d

    # drums
    n_beats = int(np.ceil(seconds / beat))
    for b in range(n_beats):
        tb, pos, bar = b * beat, b % 4, b // 4
        if pos in (0, 2):
            rows.append((tb, 0.3, 0.0, 0.30 * vel(), 3))
        if pos == 2 and rng.random() < 0.3:
            rows.append((tb + half, 0.3, 0.0, 0.23 * vel(), 3))
        if pos in (1, 3):
            rows.append((tb, 0.25, 0.0, 0.20 * vel(), 4))
        if pos == 3 and (bar % 4 == 3 or rng.random() < 0.15):
            rows.append((tb + half, 0.2, 0.0, 0.15 * vel(), 4))
        for h in (0.0, half):
            if rng.random() < 0.85:
                rows.append((tb + h, 0.08, 0.0, 0.065 * vel(), 5))

    notes = np.array(rows, dtype=NOTE_DTYPE)
    # full-band rests on bar lines, at least 8 s apart, away from both ends
    rests: list[tuple[float, float]] = []
    bar_s = 4 * beat
    cands = [k * bar_s for k in range(1, int(seconds / bar_s)) if 6.0 <= k * bar_s <= seconds - 6.0]
    for _ in range(int(seconds // 25)):
        free = [c for c in cands if all(abs(c - r0) >= 8.0 for r0, _ in rests)]
        if not free:
            break
        r0 = float(rng.choice(free))
        rests.append((r0, r0 + int(rng.integers(1, 3)) * beat))
    rests.sort()
    keep = np.ones(len(notes), bool)
    for r0, r1 in rests:
        keep &= ~((notes["start"] >= r0 - 1e-9) & (notes["start"] < r1 - 1e-9))
        cut = (notes["start"] < r0) & (notes["start"] + notes["dur"] > r0)
        notes["dur"][cut] = r0 - notes["start"][cut]
    keep &= notes["start"] < seconds - 1e-9
    notes = notes[keep]
    notes["dur"] = np.minimum(notes["dur"], seconds - notes["start"])
    notes = notes[notes["dur"] >= 0.02]
    notes = notes[np.lexsort((notes["midi"], notes["voice"], notes["start"]))]
    chord_t = np.array([c for c in chords if not any(r0 <= c < r1 for r0, r1 in rests)])
    return Score(notes=notes, duration=seconds, bpm=bpm, key=key, minor=minor, seed=seed,
                 rests=tuple(rests), chords=chord_t)


# --------------------------------------------------------------------------------------- render


@functools.lru_cache(maxsize=64)
def _table(voice: int, n_harm: int) -> np.ndarray:
    """One period of the voice's waveform with `n_harm` harmonics, peak 1, plus a wrap-around sample."""
    law = _PITCHED_VOICES[voice][0]
    ph = np.arange(_TABLE_N + 1) * (2 * np.pi / _TABLE_N)
    w = sum(law(h) * np.sin(h * ph) for h in range(1, n_harm + 1))
    w = np.asarray(w, dtype=np.float64)
    return w / np.max(np.abs(w))


def _pitched(voice: int, midi: float, n: int, sr: int) -> np.ndarray:
    """A note of `n` samples (onset to note-off) plus its release."""
    _, h_max, att, tau, sus, rel = _PITCHED_VOICES[voice]
    f = 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
    n_harm = max(1, min(h_max, int(0.45 * sr / f)))          # keep every partial below Nyquist
    tab = _table(voice, n_harm)
    n_rel = max(1, round(rel * sr))
    k = np.arange(n + n_rel)
    ph = (k * (f * _TABLE_N / sr)) % _TABLE_N
    i = ph.astype(np.int64)
    w = tab[i] + (ph - i) * (tab[i + 1] - tab[i])
    t = k / sr
    env = sus + (1.0 - sus) * np.exp(-np.maximum(t - att, 0.0) / tau)
    n_att = min(max(1, round(att * sr)), n)
    env[:n_att] *= k[:n_att] / n_att
    env[n:] = env[n - 1] * (1.0 - (k[n:] - n + 1) / n_rel)
    return w * env


def _drum(voice: int, n: int, sr: int, noise: np.ndarray, offset: int) -> np.ndarray:
    n_rel = max(1, round(_DRUM_RELEASE_S * sr))
    m = n + n_rel
    t = np.arange(m) / sr
    nz = noise[offset:offset + m]
    if voice == 3:      # kick: pitch drop 120 -> 45 Hz, plus a click
        y = np.sin(2 * np.pi * (45.0 * t + 75.0 * 0.035 * (1.0 - np.exp(-t / 0.035)))) * np.exp(-t / 0.15)
        y += 0.3 * nz * np.exp(-t / 0.004)
    elif voice == 4:    # snare: high-passed noise + a 185 Hz body
        hp = np.diff(nz, prepend=nz[0]) * 0.5
        y = hp * np.exp(-t / 0.07) + 0.5 * np.sin(2 * np.pi * 185.0 * t) * np.exp(-t / 0.05)
    else:               # hi-hat: twice-differenced noise, very short
        hp = np.diff(nz, n=2, prepend=nz[:2]) * 0.25
        y = hp * np.exp(-t / 0.02)
    env = np.ones(m)
    env[n:] = 1.0 - np.arange(1, n_rel + 1) / n_rel
    return y * env


def render_song(score: Score, sr: int = 16000, tempo_map=None, transpose: float = 0) -> np.ndarray:
    """Float32 mono audio of `score` at `sr`.

    `tempo_map` = (score_pts, perf_pts) renders the same notes at performance times: note k starts at sample
    round(score_to_perf(start_k) * sr) and is held until score_to_perf(start_k + dur_k); envelopes, timbres,
    drum noise and releases do not depend on the map. `transpose` shifts the pitched voices by semitones.
    The output lasts score_to_perf(score.duration) seconds.
    """
    pts = _map_points(tempo_map)
    notes = score.notes
    t0 = np.asarray(score_to_perf(notes["start"], pts), dtype=np.float64)
    t1 = np.asarray(score_to_perf(notes["start"] + notes["dur"], pts), dtype=np.float64)
    total = float(score_to_perf(score.duration, pts))
    if len(notes) and t0.min() < -1e-9:
        raise ValueError("the tempo map puts notes before performance time 0")
    n_out = max(0, round(total * sr))
    y = np.zeros(n_out + sr, np.float64)               # room for the last releases (trimmed below)
    s0 = np.round(t0 * sr).astype(np.int64)
    lens = np.maximum(np.round((t1 - t0) * sr).astype(np.int64), 1)
    noise = np.random.default_rng(_NOISE_SEED).standard_normal(2 * sr)
    for i in range(len(notes)):
        s = int(s0[i])
        if s >= n_out:
            continue
        v = int(notes["voice"][i])
        n = int(min(lens[i], n_out - s))
        if v in PITCHED:
            w = _pitched(v, float(notes["midi"][i]) + transpose, n, sr)
        else:
            n = min(n, sr // 2)
            w = _drum(v, n, sr, noise, (i * 7919) % (len(noise) - n - sr // 4))
        e = min(s + len(w), len(y))
        y[s:e] += float(notes["amp"][i]) * w[:e - s]
    return y[:n_out].astype(np.float32)


# ===================================================================== copied from lyricprompter.eval.track_eval
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
