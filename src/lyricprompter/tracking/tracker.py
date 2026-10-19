"""LiveTracker: state machine, tempo, confidence and operator controls around FrameSyncOLTW.

Positions are reference frame indices; frame i of a FeatureFrames stream is stamped (i+1)/fps
(the moment it becomes available), so ref_time = (pos + 1) / fps and live_time = frames / fps.

Modes
- mix:   every live frame updates the OLTW.
- vocal: live activity (rms_db against an adaptive noise floor) gates the tracker.
  * Pauses shorter than `coast_after_s` (grace): the DP keeps following; its argmin runs through a
    silent reference rest to the rest's end, where the singer resumes. Longer pauses: COASTING, the DP
    is frozen and the position advances at the clamped tempo.
  * The published position never passes the next phrase onset (the coast limit): the next line start
    (without line starts: rising edges of the reference activity; optionally also the ends of
    reference rests inside a line, `onset_rest_max_s`). An onset at the current position (start, jump
    or hold release at a line start) keeps the display there until the voice comes.
  * When the voice returns the DP is re-anchored with a soft prior around the predicted position
    (back `relock_back_s`, but not before where the voice stopped; forward to the next onset; silent
    reference frames penalised) and the buffered frames since the voice onset are replayed. For
    `relock_free_s` the published position may follow the argmin backwards, so a coasting overshoot
    is undone instead of frozen in. The relock is provisional until the voice has been active for
    `relock_confirm_s`: meanwhile the display advances at most at the coasting rate past the coast
    limit, and if the activity stops again (a bleed burst in an interlude) coasting is restored.

Global re-localisation: while confidence stays low, the recent live history is matched against the
whole reference (coarse subsequence DTW, tracking/relocate.py); a candidate clearly better than the
tracker's neighbourhood (after a distance penalty that protects against repeated choruses) is
verified with a shadow FrameSyncOLTW replay before the tracker switches to it.

The published position is monotonic except for explicit jump/nudge, vocal relock settling (and an
aborted provisional relock), verified re-localisations, key corrections and a latency change, optionally
smoothed with a tempo-predictive first-order filter; `latency_s` (+ `lag_comp_s`) adds rate * latency to
the reported ref_time, where `rate` (reported as `tempo`) is the speed at which the published position
advances: the tempo estimate while tracking, the coasting rate while coasting (capped at the coast limit)
and 0 when it does not move. The reported ref_time never steps back otherwise: when the rate drops (hold,
a vocal pause near the coast limit) the compensated time already shown is kept, not withdrawn.

Styles (`TrackerParams.style`, switchable live with `set_style`)
- classic: everything above. The display follows the DP syllable by syllable (rubato, fermatas, tempo
  changes); where the DP has nothing to follow it stops, and in vocal mode it waits at the next line.
- pop: the band keeps a steady tempo, so the tempo is the strong cue and the DP a noisy measurement of the
  phase. `tempo` is a slow estimate: a confidence-weighted running mean of the local OLTW tempo turning into
  an EMA (pop_tempo_*), fed only once locked and while the DP agrees with the display. The published
  position is a constant-velocity predictor (+tempo per frame) pulled toward the DP position, its speed
  changing by at most pop_rate x tempo x confidence and only while the DP itself moves at about the tempo
  (a DP holding through a line it cannot follow is no evidence); it snaps to the DP only after a persistent,
  confident, large forward deviation and is monotonic otherwise. A DP stuck far from the prediction with low
  confidence is re-anchored softly around it. "Locked" = the DP follows the music with confidence >=
  pop_lock_conf (mix: and advanced pop_lock_s past the last start / jump / release, so room noise before the
  band does not move the display; vocal: the voice, not intro bleed); until then the display follows the
  DP like classic. Vocal mode once locked: no coast limit (pauses and dropped lines are coasted through at the tempo),
  the relock prior is centred on the predicted position; after a start / jump / hold release into a silent
  singer it waits for the voice like classic. Global re-localisation, once locked, adopts a target only when
  it is verified again pop_reloc_confirm_s later (an ad-lib lasts a line, a skipped section goes on). The
  automatic key detection works as in classic.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from ..features.extractor import FeatureFrames
from ..text.model import STYLES
from .oltw import FrameSyncOLTW
from .relocate import Relocator

WAITING, TRACKING, COASTING, HOLD, STOPPED = "waiting", "tracking", "coasting", "hold", "stopped"


@dataclass
class TrackerParams:
    mode: str = "mix"                 # "mix" | "vocal"
    style: str = "classic"            # "classic" | "pop" (steady tempo; see the module doc)
    latency_s: float = 0.0            # added (x tempo) to the reported ref_time (operator: I/O + display)
    lag_comp_s: float = 0.0           # tuned compensation of the tracker's own lag, added like latency_s
    # OLTW
    win_s: float = 10.0
    max_adv: int = 3
    weights: tuple[float, float, float] = (2.0, 1.0, 1.0)   # diagonal, hold, ref-only step weights
    start_search_s: float = 2.0       # soft start prior reaches this far past the start position
    anchor_beta: float = 0.3          # prior cost per second of distance from the predicted position
    # tempo
    tempo_every: int = 5
    tempo_look_s: float = 6.0
    tempo_skip_s: float = 1.0
    tempo_alpha: float = 0.3
    tempo_min: float = 0.5
    tempo_max: float = 2.0
    # output smoothing (1.0 = raw clamped OLTW position)
    smooth: float = 1.0
    snap_s: float = 1.0
    # live activity gate
    act_on_db: float = 10.0
    act_off_db: float = 6.0
    act_rel_db: float = 40.0          # also require level > running peak - act_rel_db
    act_min_db: float = -65.0
    floor_init_db: float | None = None   # optional assumed noise floor before any audio was seen
    floor_win_s: float = 8.0
    peak_decay_db_s: float = 2.0
    # vocal mode
    coast_after_s: float = 0.3
    resume_frames: int = 2
    coast_tempo_min: float = 0.8
    coast_tempo_max: float = 1.25
    relock_back_s: float = 3.0        # prior reach before the predicted position (undoes coasting overshoot),
    relock_stop_back_s: float = 0.5   # but at most this far before where the voice stopped / the operator
    relock_fwd_s: float = 3.0         # anchored (bleed bursts in an interlude must not pull it back)
    relock_free_s: float = 0.5        # after a relock the published position may follow the argmin back
    relock_confirm_s: float = 0.3     # a relock is provisional until the voice was active this long; a
                                      # pause longer than the grace period before that undoes it
    relock_inactive_cost: float = 0.3
    onset_margin_s: float = 0.0
    onset_rest_max_s: float = 0.0     # > 0: reference rests up to this long inside a line also bound
    inline_wait_s: float = 1.0        # coasting, for at most inline_wait_s (off: docs/eval/TRACKING.md)
    ref_rel_db: float = 25.0          # reference activity threshold below its 99th percentile level
    ref_min_gap_s: float = 0.3
    ref_min_on_s: float = 0.1
    # confidence
    conf_tau_s: float = 1.5
    conf_lo: float = 0.45             # d(pos)/mean(d) ratio giving confidence 1
    conf_hi: float = 0.9              # ratio giving confidence 0
    coast_conf_tau_s: float = 10.0
    # global re-localisation (coarse subsequence DTW + shadow-tracker verification)
    reloc: bool = True
    reloc_conf: float = 0.35          # check while confidence stays below this (vocal: 0.6, see for_mode) ...
    reloc_after_s: float = 2.0        # ... for at least this long
    reloc_every_s: float = 0.5
    reloc_query_s: float = 6.0        # live history matched against the whole reference
    reloc_verify_s: float = 3.0       # shadow tracker replay used to verify a candidate
    reloc_ratio: float = 0.75         # candidate cost must be below ratio x current cost (coarse and fine)
    reloc_dist_penalty: float = 0.5   # coarse cost x (1 + penalty * distance / 60 s): repeats are far away
    reloc_tie: float = 1.05           # among candidates within this factor of the best, take the nearest
    reloc_min_jump_s: float = 2.0
    reloc_cands: int = 5
    # automatic key detection: after start() shadow trackers follow the same frames with the reference
    # chroma rolled to every other key; switch when one is clearly and repeatedly better
    auto_key: bool = True
    key_range: int = 6                # candidates: -key_range+1 .. +key_range semitones (all keys at 6)
    key_detect_s: float = 6.0         # tracked seconds before the first decision
    key_every_s: float = 1.0
    key_max_s: float = 40.0           # stop searching (and free the shadows) after this many seconds of evidence
    key_ratio: float = 0.9            # winner's cumulative mean distance below ratio x every other key's
    key_confirm: int = 3              # (incl. the current one) at this many consecutive decisions
    key_onset_s: float = 1.0          # mix: only live activity is evidence (room noise before the band is not):
                                      # gate-active frames until the gate was on this long without a break (the
                                      # music started), then every frame (decisions on frames above the room level)
    key_quiet_s: float = 4.0          # mix: this long (nearly) at the room level (the activity was applause / talk,
                                      # or the music stopped) drops the evidence, the budget and the decisions
    key_suggest_ratio: float = 0.95   # key_suggest: lead below ratio x the current key's and every other
    key_suggest_confirm: int = 5      # candidate's cumulative distance at this many consecutive decisions
    # pop style (only read when style == "pop"; docs/eval/TRACKING.md §8)
    pop_tempo_alpha: float = 0.01     # steady tempo: running mean of the local OLTW tempo (every tempo_every
                                      # frames, weighted by confidence) that becomes an EMA with this weight
    pop_tempo_prior: float = 15.0     # weight of the initial tempo 1.0 in that running mean
    pop_tempo_min: float = 0.7
    pop_tempo_max: float = 1.4
    pop_tempo_gate_s: float = 0.5     # local tempo samples count only while the DP is this close to the display
    pop_tempo_huber: float = 0.1      # and with a bounded influence beyond this deviation from the estimate
    pop_gain: float = 0.08            # per-frame pull of the predicted position toward the DP position ...
    pop_rate: float = 0.5             # ... changing the display speed by at most pop_rate x tempo x confidence,
    pop_vel_s: float = 0.5            # and only while the DP itself moved at the tempo (within a factor of
    pop_vel_gate: float = 0.35        # 1 +- pop_vel_gate) over the last pop_vel_s (a holding DP is no evidence)
    pop_snap_s: float = 1.5           # DP ahead of the prediction by more than this ...
    pop_snap_after_s: float = 2.0     # ... for this long with confidence >= pop_snap_conf: jump to the DP
    pop_snap_conf: float = 0.7
    pop_stuck_s: float = 3.0          # DP this far from the prediction ...
    pop_stuck_after_s: float = 2.0    # ... for this long with confidence < pop_stuck_conf: re-anchor it softly
    pop_stuck_conf: float = 0.5
    pop_prior_s: float = 2.0          # reach of the relock / re-anchor prior around the predicted position
    pop_lock_conf: float = 0.5        # the display runs on at the tempo once the DP follows the music with this
    pop_lock_s: float = 1.0           # confidence (vocal: not bleed in the intro) and, mix, advanced this far
                                      # past the last start / jump / release (it holds in room noise)
    pop_reloc_confirm_s: float = 5.0  # a verified relocation target must be verified again this much later
    pop_reloc_tol_s: float = 1.5      # (at its first position + elapsed x tempo, within this) to be adopted

    @classmethod
    def for_mode(cls, mode: str, style: str = "classic") -> TrackerParams:
        """Tuned defaults per mode and style (see docs/eval/TRACKING.md)."""
        if style not in STYLES:
            raise ValueError(f"unknown style {style!r} (expected one of {STYLES})")
        if mode in ("mix", "vocal"):
            # diagonal bias + mild hold penalty: ambiguous passages advance instead of stalling
            p = cls(mode=mode, weights=(1.8, 1.2, 1.0))
            # vocal features are less distinctive: the confidence of a tracker on the wrong verse (same melody)
            # stays around 0.35-0.6, so a skipped verse was only re-localised after 14-32 s
            p = replace(p, reloc_conf=0.6) if mode == "vocal" else p
            return replace(p, **_POP_PRESET[mode]) if style == "pop" else p
        raise ValueError(f"unknown mode {mode!r} (expected 'mix' or 'vocal')")


# for_mode(mode, "pop") = for_mode(mode, "classic") + these (set_style switches exactly these fields)
_POP_PRESET: dict[str, dict] = {"mix": {"style": "pop"}, "vocal": {"style": "pop"}}


def style_fields(mode: str, style: str) -> dict:
    """The TrackerParams fields that differ between the styles' presets, with `style`'s preset values."""
    ref = TrackerParams.for_mode(mode, style)
    return {k: getattr(ref, k) for k in _POP_PRESET[mode]}


@dataclass
class TrackerState:
    status: str                  # waiting | tracking | coasting | hold | stopped
    ref_time: float
    tempo: float
    confidence: float            # 0..1
    live_time: float
    frame: int
    active: bool = False         # live activity (music / voice present) on the last frame
    transpose: int = 0
    key_auto: bool = False       # transpose is (being) chosen by automatic key detection
    key_suggest: int | None = None   # auto search: a key that fits better than `transpose` (and than every
                                     # other key) consistently, but not clearly enough to switch; kept after
                                     # the search ended until the operator sets a key or restarts the search


class LevelGate:
    """Live activity from per-frame levels: above max(noise floor + on_db, running peak - rel_db,
    min_db) with hysteresis. The floor is the minimum level of the last `floor_win_s` (optionally
    seeded with `floor_init_db`), the peak decays slowly."""

    def __init__(self, fps: float, p: TrackerParams):
        self.p = p
        self._blk = max(1, round(fps / 4))                      # 0.25 s blocks of minima
        self._mins: deque[float] = deque(maxlen=max(1, round(p.floor_win_s * fps / self._blk)))
        if p.floor_init_db is not None:
            self._mins.append(p.floor_init_db)
        self._cur, self._n = math.inf, 0
        self._decay = p.peak_decay_db_s / fps
        self.peak = -math.inf
        self.active = False

    @property
    def floor(self) -> float:
        return min(min(self._mins, default=math.inf), self._cur)

    def update(self, level_db: float) -> bool:
        p = self.p
        self._cur = min(self._cur, level_db)
        self._n += 1
        if self._n >= self._blk:
            self._mins.append(self._cur)
            self._cur, self._n = math.inf, 0
        self.peak = max(level_db, self.peak - self._decay)
        floor = self.floor
        on = max(floor + p.act_on_db, self.peak - p.act_rel_db, p.act_min_db)
        off = max(floor + p.act_off_db, self.peak - p.act_rel_db - (p.act_on_db - p.act_off_db), p.act_min_db)
        self.active = level_db > (off if self.active else on)
        return self.active


def ref_activity(rms_db: np.ndarray, fps: float, rel_db: float = 35.0, min_gap_s: float = 0.3,
                 min_on_s: float = 0.1) -> np.ndarray:
    """Reference activity (bool per frame): level above p99 - rel_db, gaps < min_gap_s closed,
    blips < min_on_s removed."""
    from scipy.ndimage import binary_closing, binary_opening

    rms_db = np.asarray(rms_db, dtype=np.float64)
    if rms_db.size == 0:
        return np.zeros(0, bool)
    act = rms_db > max(float(np.percentile(rms_db, 99)) - rel_db, -100.0)
    g = round(min_gap_s * fps)
    if g > 1:
        act = binary_closing(np.pad(act, g), structure=np.ones(g, bool))[g:-g]
    o = round(min_on_s * fps)
    if o > 1:
        act = binary_opening(act, structure=np.ones(o, bool))
    return act


def phrase_onsets(ref_active: np.ndarray, line_frames: np.ndarray, max_rest: int) -> np.ndarray:
    """Reference frames that bound vocal-mode coasting. Without line starts: every rising edge of the
    reference activity. With line starts: the line starts plus the rising edges that end an in-line
    rest, i.e. activity runs chained to the line's first run by gaps of at most `max_rest` frames and
    starting before the next line (activity in a long interlude, often bleed, is not chained)."""
    act = np.asarray(ref_active, bool)
    rises = np.flatnonzero(act & ~np.r_[False, act[:-1]])
    lines = np.unique(np.asarray(line_frames, dtype=np.int64))
    if len(lines) == 0:
        return rises.astype(np.int64)
    falls = np.flatnonzero(act & ~np.r_[act[1:], False])          # last frame of each run
    out = list(lines)
    for k, s in enumerate(lines):
        e = lines[k + 1] if k + 1 < len(lines) else len(act)
        j = int(np.searchsorted(falls, s, side="left"))           # first run still active at/after s
        if j >= len(rises) or rises[j] >= e:
            continue
        end = falls[j]
        for r in range(j + 1, len(rises)):
            if rises[r] >= e or rises[r] - end - 1 > max_rest:
                break
            out.append(rises[r])
            end = falls[r]
    return np.unique(np.asarray(out, dtype=np.int64))


@dataclass(frozen=True)
class _Provisional:
    """Coasting state saved by a provisional relock (restored if the activity was only a burst)."""
    out: float
    limit: float
    inline: bool
    wait: int
    floor: float
    frame: int
    tempo: float
    ratio: float
    conf: float


def roll_chroma(R0: np.ndarray, chroma: slice, semitones: int) -> np.ndarray:
    """Reference features for a live performance `semitones` above the reference."""
    R, k = R0.copy(), int(semitones) % 12
    if k:
        R[:, chroma] = np.roll(R0[:, chroma], k, axis=1)
    return R


class _KeySearch:
    """Automatic key detection: one shadow FrameSyncOLTW per other key, fed the same frames and anchored
    like the main tracker (every anchor of the main is applied to the shadows too, a relocation replays them
    over the same frames). The chroma is the only key-dependent group, so the key whose path has the lowest
    mean local distance is the live key.

    Evidence is the cumulative distance since the search (re)started, compared at equal frame counts. In mix
    mode every frame is tracked, also room noise / applause / talk before the band starts, and stationary
    noise can fit one rolled reference consistently, so evidence follows the live activity:
    - before the music only frames on which the activity gate is on count;
    - once the gate was on for `key_onset_s` without a break the music started (`room`: the gate's noise
      floor before it) and every frame counts; decisions are taken only on loud frames (gate on, or more than
      act_on_db above `room`: the gate's adaptive floor rises to the music's own quiet frames);
    - when at most a tenth of the last `key_quiet_s` was loud (back at the room level: the activity was
      applause or talk, or the music stopped) the evidence, the budget and the decisions so far are dropped
      and the search waits for the music again. A shorter soft passage of the music keeps its evidence.
    Vocal mode tracks only while the voice is active, so every tracked frame counts. A verified relocation
    restarts the evidence too (it was collected at the wrong place).

    A switch needs a clear (key_ratio) and repeated (key_confirm) advantage over every other key incl. the
    current one; the search runs until key_max_s seconds of evidence so an early wrong choice between
    neighbouring keys can still be corrected. A smaller but consistent advantage (key_suggest_ratio over the
    current key and over every other candidate, key_suggest_confirm decisions in a row) is published as a
    suggestion for the operator."""

    def __init__(self, trk: LiveTracker):
        p, fps = trk.params, trk.fps
        cur = trk._transpose % 12
        keys: dict[int, int] = {}
        for k in sorted(range(-p.key_range, p.key_range + 1), key=abs):   # prefer the smaller shift
            if k % 12 != cur:
                keys.setdefault(k % 12, k)
        self.shadows = {k: FrameSyncOLTW(roll_chroma(trk._R0, trk._chroma, k), fps=fps, win_s=p.win_s,
                                         max_adv=p.max_adv, weights=p.weights, hist_s=trk.oltw.K / fps)
                        for k in keys.values()}
        self.first = max(1, round(p.key_detect_s * fps))
        self.every = max(1, round(p.key_every_s * fps))
        self.max_steps = round(p.key_max_s * fps)
        self.ratio, self.confirm = p.key_ratio, p.key_confirm
        self.s_ratio, self.s_confirm = p.key_suggest_ratio, max(1, p.key_suggest_confirm)
        self.mix = p.mode == "mix"
        self.onset = max(1, round(p.key_onset_s * fps))
        self.above_db = p.act_on_db
        self.started = not self.mix             # mix: the music started (see the class doc)
        self.room: float | None = None          # live level before the music started (dB)
        self._loud: deque[bool] = deque(maxlen=max(1, round(p.key_quiet_s * fps)))   # recent frames: loud?
        self._n_loud = 0
        self.restart(budget=True)

    def restart(self, budget: bool = False) -> None:
        """Drop the evidence (with `budget` also the search budget and the decisions) collected so far."""
        self.sums = {k: 0.0 for k in self.shadows}
        self.sum_main = 0.0
        self.n = 0                              # evidence frames since the restart
        if budget:
            self.steps = 0                      # evidence frames counted against key_max_s
            self.evals = 0                      # decisions on this evidence (relocation waits for the first)
        self._due = False                       # a decision is due (taken on the next loud frame)
        self.cand: int | None = None
        self.count = 0
        self._lead: int | None = None
        self._lead_n = 0
        self.lead: int | None = None            # published suggestion (see key_suggest)

    @property
    def done(self) -> bool:
        return self.steps >= self.max_steps

    def anchor(self, i0: int, prior: np.ndarray | None, free: int = 0) -> None:
        for sh in self.shadows.values():
            sh.anchor(i0, prior, free=free)

    def replay(self, i0: int, prior: np.ndarray | None, X: np.ndarray) -> None:
        """Anchor every shadow like a relocated main (i0, prior) and replay the frames it was verified on."""
        for sh in self.shadows.values():
            sh.anchor(i0, prior)
            for x in X:
                sh.step(x)

    def swap(self, k: int, old_key: int, old_main: FrameSyncOLTW) -> FrameSyncOLTW:
        """Key k becomes the main hypothesis; the old main becomes the shadow of old_key."""
        new = self.shadows.pop(k)
        self.shadows[old_key] = old_main
        self.sums[old_key], self.sum_main = self.sum_main, self.sums.pop(k)
        self.cand, self.count = None, 0
        self._lead, self._lead_n, self.lead = None, 0, None
        return new

    def step(self, x: np.ndarray, d_main: float, active: bool = True, act_run: int = 0,
             level: float | None = None, floor: float | None = None) -> int | None:
        """Advance every shadow (always, to stay in sync with the main); add the frame to the evidence
        (see the class doc; `level` / `floor`: the frame's live level and the activity gate's noise floor,
        mix mode); returns the key to switch to, if any."""
        ds = {k: sh.step(x).d_pos for k, sh in self.shadows.items()}
        loud = True
        if self.mix and level is not None:
            if not self.started and act_run >= self.onset:
                self.started, self.room = True, floor
            loud = bool(active or (self.started and self.room is not None and level > self.room + self.above_db))
            w = self._loud
            if len(w) == w.maxlen:
                self._n_loud -= w[0]
            w.append(loud)
            self._n_loud += loud
            if (self.started or self.n) and len(w) == w.maxlen and self._n_loud * 10 <= w.maxlen:
                # back at the room level (at most a few blips): the activity so far was a burst (applause,
                # talk) or the music stopped; its evidence and the decisions on it must not carry over
                self.started, self.room = False, None
                self.restart(budget=True)
                return None
            if not (loud or self.started):
                return None                     # before the music only live activity is evidence
        for k, d in ds.items():
            self.sums[k] += d
        self.sum_main += d_main
        self.n += 1
        self.steps += 1
        if self.n >= self.first and (self.n - self.first) % self.every == 0:
            self._due = True
        # decisions only on live activity: frames at the room level after the onset count provisionally (a
        # soft passage of the music) but are dropped with the rest if the level stays there (see above)
        if not (self._due and loud):
            return None
        self._due = False
        self.evals += 1
        best = min(self.sums, key=self.sums.get)
        others = [v for k, v in self.sums.items() if k != best]
        runner_up = min([self.sum_main, *others])
        # same step count for every hypothesis: compare sums. The winner must beat the current key AND
        # the runner-up clearly; neighbouring keys that stay close (a lone voice) are left to the operator.
        if self.sums[best] < self.ratio * runner_up:
            self.count = self.count + 1 if best == self.cand else 1
            self.cand = best
        else:
            self.cand, self.count = None, 0
        # suggestion: smaller margin, but over the current key AND every other candidate, held for a while
        sug = best if self.sums[best] < self.s_ratio * min([self.sum_main, *others]) else None
        if sug is not None and sug == self._lead:
            self._lead_n += 1
        else:
            self._lead, self._lead_n = sug, int(sug is not None)
        self.lead = self._lead if self._lead_n >= self.s_confirm else None
        return best if self.count >= self.confirm else None


class LiveTracker:
    def __init__(self, ref: FeatureFrames, fps: float, params: TrackerParams, *,
                 ref_active: np.ndarray | None = None, line_starts: list[float] | None = None):
        if params.mode not in ("mix", "vocal"):
            raise ValueError(f"unknown mode {params.mode!r}")
        if params.style not in STYLES:
            raise ValueError(f"unknown style {params.style!r}")
        if len(ref) == 0:
            raise ValueError("empty reference features")
        self.params, self.fps, self.ref = params, float(fps), ref
        self.N = len(ref)
        self._R0 = np.ascontiguousarray(ref.feats, dtype=np.float32)
        self._chroma = ref.group("chroma")
        self.oltw = FrameSyncOLTW(self._R0, fps=self.fps, win_s=params.win_s, max_adv=params.max_adv,
                                  weights=params.weights, hist_s=max(8.0, params.tempo_look_s + 2.0))
        if ref_active is None:
            act = ref_activity(ref.rms_db, self.fps, params.ref_rel_db, params.ref_min_gap_s, params.ref_min_on_s)
        else:
            a = np.asarray(ref_active).astype(bool)
            act = a if len(a) == self.N else a[np.minimum((np.arange(self.N) * len(a)) // self.N, len(a) - 1)]
        self.ref_active = act
        self.set_line_starts(line_starts)
        self._reloc = Relocator(self._R0, self.fps) if params.reloc else None
        self._transpose = 0
        self._key_auto = bool(params.auto_key) and self._chroma is not None
        self._keys: _KeySearch | None = None
        self._kept_suggest: int | None = None  # last suggestion of a finished search (until the operator acts)
        self.key_switches = 0
        self._pub: float | None = None   # last reported ref_time (published-position floor)
        self._pub_reset = True           # the next report may step back (jump, relocation, key switch, ...)
        self._gate = LevelGate(self.fps, params)
        self._level: float | None = None  # live level of the frame being processed
        self._status = WAITING
        self._frame = 0
        self._out = 0.0                  # published position (ref frames, float)
        self._tempo = 1.0
        self._conf = 0.0
        self._ratio = params.conf_hi
        self.relocations = 0
        self.snaps = 0                   # pop: jumps of the display to the DP (persistent forward deviation)
        self.reanchors = 0               # pop: DP re-anchored around the prediction (stuck far from it)
        self._reset_runs()

    # ------------------------------------------------------------------ controls
    def set_line_starts(self, line_starts: list[float] | None) -> None:
        """Phrase onsets bounding vocal-mode coasting: lyric line starts when known (alignment is more
        reliable than stem levels, which carry bleed; with onset_rest_max_s > 0 also the ends of
        reference rests inside lines), else every rising edge of the reference activity."""
        ls = [round(t * self.fps) - 1 for t in (line_starts or []) if t is not None and math.isfinite(t)]
        lines = np.unique(np.clip(np.array(ls, dtype=np.int64), 0, self.N - 1))
        rest = round(self.params.onset_rest_max_s * self.fps)
        self.onsets = np.clip(phrase_onsets(self.ref_active, lines, rest), 0, self.N - 1).astype(np.int64)
        self._onset_inline = ~np.isin(self.onsets, lines) if len(lines) else np.zeros(len(self.onsets), bool)

    def start(self, ref_time: float = 0.0) -> None:
        p = self.params
        self._frame = 0
        self._tempo = 1.0
        self._conf, self._ratio = 0.0, p.conf_hi
        self._reset_runs()
        i0 = self._frame_of(ref_time)
        self._out = self._lock_from = float(i0)
        self._pub_reset = True
        self._keys = _KeySearch(self) if self._key_auto else None
        self._kept_suggest = None
        if p.mode == "vocal":
            self._status = COASTING                                   # nothing to follow until the voice starts
            self._coast_begin()
            self._floor = self._out - p.relock_stop_back_s * self.fps
        else:
            self._status = TRACKING
            lo, hi = i0, min(self.N, i0 + round(p.start_search_s * self.fps) + 1)
            prior = np.full(self.N, np.inf)
            prior[lo:hi] = p.anchor_beta * (np.arange(lo, hi) - i0) / self.fps
            self._anchor(i0, prior)

    def stop(self) -> None:
        self._status = STOPPED

    def hold(self, on: bool) -> None:
        """Freeze (on) / release (off) the published position; frames pushed while held are not
        tracked and the release re-anchors softly around the held position."""
        if self._status in (WAITING, STOPPED):
            return
        if on:
            self._status = HOLD
        elif self._status == HOLD:
            self._resume_at(self._out, back_s=1.0, fwd_s=self.params.relock_fwd_s)

    def jump(self, ref_time: float) -> None:
        pos = float(np.clip(float(ref_time) * self.fps - 1.0, 0.0, self.N - 1.0))
        self._out = pos
        self._pub_reset = True
        if self._status in (WAITING, STOPPED, HOLD):
            return
        self._resume_at(pos, back_s=self.params.relock_stop_back_s, fwd_s=self.params.relock_fwd_s)

    def nudge(self, dt: float) -> None:
        self.jump((self._out + 1.0) / self.fps + float(dt))

    def set_transpose(self, semitones: int) -> None:
        """Live is `semitones` above the reference: roll the reference chroma up by that amount.
        Needs a reference with a known chroma group (FeatureFrames.groups) unless semitones % 12 == 0.
        An operator-set key turns automatic key detection off (set_auto_key(True) turns it back on)."""
        k = int(semitones) % 12
        if self._chroma is None:
            if k:
                raise ValueError("set_transpose: the reference features have no 'chroma' group "
                                 "(FeatureFrames.groups is empty), cannot transpose")
            self._transpose = int(semitones)
            return
        self._key_auto, self._keys, self._kept_suggest = False, None, None
        self._transpose = int(semitones)
        R = roll_chroma(self._R0, self._chroma, k)
        self.oltw.set_reference(R)
        if self._reloc is not None:
            self._reloc.set_reference(R)

    def set_auto_key(self, on: bool) -> None:
        """Automatic key detection on/off. Turning it on while running starts a new search around the
        current position (the current key stays unless another one is clearly better): while the DP runs
        the main tracker and the new shadows are re-anchored there the same way (coasting / held: the next
        relock or release anchors them all)."""
        self._key_auto = bool(on) and self._chroma is not None
        self._kept_suggest = None
        if not self._key_auto:
            self._keys = None
        elif self._status not in (WAITING, STOPPED):
            self._keys = _KeySearch(self)
            if self._status == TRACKING:
                self._anchor(round(self._out), self._prior(self._out, 1.0, self.params.relock_fwd_s))
                self._steps = 0

    def set_latency(self, seconds: float) -> None:
        self.params = replace(self.params, latency_s=float(seconds))
        self._pub_reset = True                         # a smaller latency may move the report back once

    def set_style(self, style: str) -> None:
        """Switch classic / pop (also while running): only the fields of the style presets change (other
        overrides and the latency stay). The display goes on from where it is: to pop with the current tempo
        as the start of the steady estimate (coasting through pauses once the voice was followed), to classic
        with the coast limit at the next onset again."""
        if style not in STYLES:
            raise ValueError(f"unknown style {style!r} (expected one of {STYLES})")
        if style == self.params.style:
            return
        p = self.params = replace(self.params, **style_fields(self.params.mode, style))
        self._snap_run = self._stuck_run = 0
        if style == "pop":
            self._tempo = float(np.clip(self._tempo, p.pop_tempo_min, p.pop_tempo_max))
            self._tempo_w = p.pop_tempo_prior
        if self._status in (COASTING, TRACKING):
            self._coast_begin()                        # pop & locked: no limit; classic: the next onset

    # ------------------------------------------------------------------ streaming
    def push(self, frames: FeatureFrames) -> TrackerState:
        feats, levels = frames.feats, frames.rms_db
        if len(frames) and feats.shape[1] != self._R0.shape[1]:
            raise ValueError(f"live feature dim {feats.shape[1]} != reference dim {self._R0.shape[1]}")
        for k in range(len(frames)):
            self._step(feats[k], float(levels[k]))
        return self.state

    @property
    def state(self) -> TrackerState:
        st, p = self._status, self.params
        out, cap = self._out, None
        if self._prov is not None:                    # provisional relock: at most coasting past the limit
            cap = self._prov_cap()
            out = min(out, cap)
        elif st == COASTING or (st == TRACKING and self._grace):
            cap = max(self._out, self._coast_limit)   # never display past the coast limit
        t = (out + 1.0) / self.fps
        rate = 0.0                                    # speed of the published position (ref s per live s)
        if st == TRACKING and not self._grace:
            rate = self._tempo
        elif st in (TRACKING, COASTING) and self._out < self._coast_limit - 1e-6:
            rate = self._coast_rate()
        if rate > 0.0:
            t += rate * (p.latency_s + p.lag_comp_s)
            if cap is not None:
                t = min(t, (cap + 1.0) / self.fps)
        t = float(min(max(t, 0.0), self.N / self.fps))
        if st != WAITING and not self._pub_reset and self._pub is not None:
            t = max(t, self._pub)                     # keep what was shown (hold, pause near the limit)
        self._pub, self._pub_reset = t, False
        return TrackerState(status=st, ref_time=t, tempo=float(rate), confidence=float(self._conf),
                            live_time=self._frame / self.fps, frame=self._frame, active=self._gate.active,
                            transpose=self._transpose, key_auto=self._key_auto,
                            key_suggest=self._keys.lead if self._keys is not None else self._kept_suggest)

    @property
    def position(self) -> int:
        """Current (unsmoothed) OLTW reference frame."""
        return self.oltw.pos

    # ------------------------------------------------------------------ internals
    def _reset_runs(self) -> None:
        self._act_run = 0
        self._inact_run = 0
        self._grace = False              # vocal grace period (silent live, still TRACKING): capped at the limit
        self._floor = 0.0                # relock prior lower bound: where the voice stopped / operator anchor
        self._prov: _Provisional | None = None   # unconfirmed relock
        self._confirm_left = 0
        self._coast_limit = float(self.N - 1)
        self._limit_inline = False       # the coast limit is an in-line onset (waited at for inline_wait_s)
        self._wait = 0
        self._steps = 0
        self._buf: deque[tuple[np.ndarray, float]] = deque(maxlen=max(2, self.params.resume_frames + 2))
        self._hist: deque[tuple[np.ndarray, int]] = deque(maxlen=max(2, round(self.params.reloc_query_s * self.fps)))
        self._low_run = 0
        self._since_check = 0
        self._locked = False             # the DP follows the music (pop: the display runs on at the tempo from
                                         # then on): confidently, vocal: the voice, mix: the DP advanced
        self._lock_from = 0.0            # pop_lock_s past the last start / jump / release (not room noise)
        self._lock_frame = 0             # live frame of the lock
        self._tempo_w = self.params.pop_tempo_prior   # pop: weight behind the steady tempo estimate
        self._snap_run = 0               # pop: frames the DP has been far ahead of the prediction (confidently)
        self._stuck_run = 0              # pop: frames the DP has been far from the prediction (low confidence)
        self._reloc_pending: tuple[float, int, int] | None = None   # pop: verified target awaiting its
                                                                     # confirmation (position, frame, key)
        self._dp_hist: deque[float] = deque(maxlen=max(2, round(self.params.pop_vel_s * self.fps) + 1))

    @property
    def _free_run(self) -> bool:
        """Pop style, locked: the position runs on at the tempo (no coast limit), the DP only corrects it."""
        return self._locked and self.params.style == "pop"

    def _frame_of(self, ref_time: float) -> int:
        return int(np.clip(round(float(ref_time) * self.fps) - 1, 0, self.N - 1))

    def _step(self, x: np.ndarray, level: float) -> None:
        active = self._gate.update(level)
        self._level = level
        if self._status in (WAITING, STOPPED):
            return
        self._frame += 1
        if self._status == HOLD:
            return
        if active:
            self._act_run, self._inact_run = self._act_run + 1, 0
        else:
            self._act_run, self._inact_run = 0, self._inact_run + 1
        self._buf.append((x, self._out))
        p = self.params
        stepped = True
        if p.mode == "vocal" and self._status == COASTING:
            if self._act_run >= p.resume_frames:
                self._relock(self._act_run)
            else:
                self._coast_step()
                stepped = False
        elif p.mode == "vocal" and not active:
            # silent live while tracking (grace period): the DP keeps following. Its argmin runs through
            # a silent reference rest to the rest's end, where the singer will resume (evaluated: rate
            # limiting this costs ~1.5 PCO@300 points), but never past the coast limit
            if not self._grace:
                self._grace = True
                self._floor = self._out - p.relock_stop_back_s * self.fps
                self._coast_begin()
            self._track(x, cap=max(self._out, self._coast_limit))
            if self._inact_run > round(p.coast_after_s * self.fps):
                if self._prov is not None:
                    self._abort_relock()                     # a burst, not the voice
                else:
                    self._status = COASTING                  # from now on the DP is frozen
                self._grace = False
        else:
            self._grace = False
            self._track(x, active=active)
            if self._prov is not None:
                self._confirm_left -= 1
                if self._confirm_left <= 0:
                    self._prov = None
            if self._prov is None and not self._locked and self._conf >= p.pop_lock_conf and (
                    p.mode == "vocal" or self.oltw.pos - self._lock_from >= p.pop_lock_s * self.fps):
                self._locked, self._lock_frame = True, self._frame   # (pop: from now on the display runs on)
                if self._free_run:
                    self._coast_limit, self._limit_inline = float(self.N - 1), False
        self._hist.append((x, self.oltw.pos if stepped else round(self._out)))
        if self._reloc is not None and stepped and self._status == TRACKING:
            self._low_run = self._low_run + 1 if self._conf < p.reloc_conf else 0
            if self._low_run == 0:
                self._reloc_pending = None               # pop: the confidence recovered, nothing to confirm
            self._since_check += 1
            key_pending = self._keys is not None and self._keys.evals == 0   # a wrong key also looks "lost"
            if (self._low_run >= p.reloc_after_s * self.fps and self._since_check >= p.reloc_every_s * self.fps
                    and len(self._hist) == self._hist.maxlen and self._steps >= self._hist.maxlen
                    and not key_pending):
                self._since_check = 0
                self._try_relocate()

    def _track(self, x: np.ndarray, cap: float | None = None, active: bool = True) -> None:
        p = self.params
        r = self.oltw.step(x)
        self._steps += 1
        ratio = r.d_pos / max(r.d_mean, 1e-6)
        a = 1.0 / max(1.0, p.conf_tau_s * self.fps)
        self._ratio += a * (ratio - self._ratio)
        self._conf = float(np.clip((p.conf_hi - self._ratio) / (p.conf_hi - p.conf_lo), 0.0, 1.0))
        pop = p.style == "pop"
        if self._steps % p.tempo_every == 0:
            if pop:
                self._pop_tempo()
            else:
                rho = self.oltw.tempo(p.tempo_look_s, p.tempo_skip_s)
                if rho is not None:
                    rho = float(np.clip(rho, p.tempo_min, p.tempo_max))
                    self._tempo += p.tempo_alpha * (rho - self._tempo)
        raw = float(r.pos)
        if pop and self._locked:
            new = self._pop_position(raw)
        elif r.free:                                  # settling after a relock: may move backwards
            new = raw
            self._pub_reset = True
        elif p.smooth >= 1.0:
            new = max(self._out, raw)
        else:
            pred = self._out + self._tempo
            new = pred + p.smooth * (raw - pred)
            if abs(raw - new) > p.snap_s * self.fps:
                new = raw
            new = max(self._out, min(new, self.N - 1.0))
        self._out = new if cap is None else min(new, cap)       # cap >= _out: only limits advancing
        if self._keys is not None:
            k = self._keys.step(x, r.d_pos, active, self._act_run, self._level, self._gate.floor)
            if k is not None and cap is None:
                self._switch_key(k)
            elif self._keys.done:
                self._kept_suggest = self._keys.lead      # stays visible until the operator acts
                self._keys = None
        if pop and self._locked and not r.free and self._prov is None:
            # a DP stuck far from the prediction (lost in an ad-lib / a solo) is re-anchored around it
            far = abs(float(self.oltw.pos) - self._out) > p.pop_stuck_s * self.fps and self._conf < p.pop_stuck_conf
            self._stuck_run = self._stuck_run + 1 if far else 0
            if self._stuck_run >= p.pop_stuck_after_s * self.fps:
                self._stuck_run = 0
                self._anchor(round(self._out), self._pop_prior(self._out))
                self.reanchors += 1

    def _pop_tempo(self) -> None:
        """Pop steady tempo: confidence-weighted running mean of the local OLTW tempo, an EMA once the weight
        reaches 1 / pop_tempo_alpha. Only once locked, over the path since the lock (the DP holding in room
        noise before the band is no tempo), while the DP agrees with the display (a DP holding through a
        dropped line must not slow the band down), Huber-weighted around the estimate."""
        p = self.params
        look = min(p.tempo_look_s, (self._frame - self._lock_frame) / self.fps)
        if not self._locked or look < 2.0 * p.tempo_skip_s:
            return
        if abs(float(self.oltw.pos) - self._out) > p.pop_tempo_gate_s * self.fps:
            return
        rho = self.oltw.tempo(look, p.tempo_skip_s)
        if rho is None:
            return
        e = float(np.clip(rho, p.pop_tempo_min, p.pop_tempo_max)) - self._tempo
        w = max(self._conf, 0.05) * min(1.0, p.pop_tempo_huber / max(abs(e), 1e-9))
        self._tempo_w = min(self._tempo_w + w, 1.0 / p.pop_tempo_alpha)
        self._tempo += w / self._tempo_w * e

    def _pop_position(self, raw: float) -> float:
        """Pop: constant-velocity prediction (+tempo per frame) pulled toward the DP position `raw`, the speed
        changing by at most pop_rate x tempo x confidence and only while the DP moves at about the tempo; a
        persistent, confident, large forward deviation snaps to the DP. Monotonic."""
        p = self.params
        pred = self._out + self._tempo
        h = self._dp_hist
        h.append(raw)
        lim = p.pop_rate * self._tempo * self._conf
        if len(h) < 2 or abs((h[-1] - h[0]) / (len(h) - 1) - self._tempo) > p.pop_vel_gate * self._tempo:
            lim = 0.0                                # the DP holds / races: no evidence about the phase
        new = pred + float(np.clip(p.pop_gain * (raw - pred), -lim, lim))
        if raw - pred > p.pop_snap_s * self.fps and self._conf >= p.pop_snap_conf:
            self._snap_run += 1
            if self._snap_run >= p.pop_snap_after_s * self.fps:
                new, self._snap_run = raw, 0
                self.snaps += 1
        else:
            self._snap_run = 0
        return min(max(self._out, new), self.N - 1.0)

    def _pop_prior(self, center: float) -> np.ndarray:
        """Pop: soft prior beta*|i - center| over center +- pop_prior_s (+ relock_inactive_cost on silent
        reference frames in vocal mode): the steady tempo makes the predicted position reliable."""
        p = self.params
        lo = max(0, int(center - p.pop_prior_s * self.fps))
        hi = min(self.N, math.ceil(center + p.pop_prior_s * self.fps) + 1)
        prior = np.full(self.N, np.inf)
        prior[lo:hi] = p.anchor_beta * np.abs(np.arange(lo, hi) - center) / self.fps
        if p.mode == "vocal":
            prior[lo:hi] += p.relock_inactive_cost * (~self.ref_active[lo:hi])
        return prior

    def _anchor(self, i0: int, prior: np.ndarray | None, free: int = 0) -> None:
        """Anchor the main DP and, while the key search runs, every shadow the same way."""
        self._dp_hist.clear()                        # (pop: the DP's speed restarts from here)
        self.oltw.anchor(i0, prior, free=free)
        if self._keys is not None:
            self._keys.anchor(i0, prior, free=free)

    def _switch_key(self, k: int) -> None:
        """Adopt the shadow tracker of key k (it followed the same frames under that key): its DP state
        and position replace the main one; the old main becomes the shadow of the old key."""
        new = self._keys.swap(k, self._transpose, self.oltw)
        self.oltw = new
        self._transpose = k
        if self._reloc is not None:
            self._reloc.set_reference(new.R)
        self._prov = None
        self._out = float(new.pos)                   # a key correction may move the display backwards
        self._pub_reset = True
        self._ratio = 0.5 * (self.params.conf_lo + self.params.conf_hi)
        self._low_run = self._snap_run = self._stuck_run = 0
        self.key_switches += 1

    def _next_onset(self, pos: float) -> float:
        """First onset strictly after pos (N - 1 if none)."""
        k = int(np.searchsorted(self.onsets, pos, side="right"))
        return float(self.onsets[k]) if k < len(self.onsets) else float(self.N - 1)

    def _coast_begin(self, after: bool = False) -> None:
        """Set the coast limit: the first onset at or after the current position (within half a
        frame: after a start/jump/release at a line start the display waits there for the voice), or
        strictly after it (`after`, leaving an in-line onset that was waited at long enough).
        Pop style once locked: no limit (pauses and dropped lines are coasted through at the tempo)."""
        if self._free_run:
            self._coast_limit, self._limit_inline, self._wait = float(self.N - 1), False, 0
            return
        k = int(np.searchsorted(self.onsets, self._out + 0.5 if after else self._out - 0.5,
                                side="right" if after else "left"))
        if k < len(self.onsets):
            nxt, self._limit_inline = float(self.onsets[k]), bool(self._onset_inline[k])
        else:
            nxt, self._limit_inline = float(self.N - 1), False
        self._coast_limit = max(self._out, nxt - self.params.onset_margin_s * self.fps)
        self._wait = 0

    def _coast_rate(self) -> float:
        p = self.params
        if p.style == "pop":                         # the steady tempo (already within its own clamp)
            return float(np.clip(self._tempo, p.pop_tempo_min, p.pop_tempo_max))
        return float(np.clip(self._tempo, p.coast_tempo_min, p.coast_tempo_max))

    def _coast_step(self) -> None:
        p = self.params
        if self._limit_inline and self._out >= self._coast_limit - 1e-6:
            # waiting at the end of a reference rest inside a line: the live singer may be singing
            # softly (missed by the gate) -> continue after inline_wait_s
            self._wait += 1
            if self._wait > round(p.inline_wait_s * self.fps):
                self._coast_begin(after=True)
        self._out = min(self._out + self._coast_rate(), max(self._out, self._coast_limit))
        self._conf *= math.exp(-1.0 / max(1.0, p.coast_conf_tau_s * self.fps))

    def _prior(self, center: float, back_s: float, fwd_s: float, floor: float = 0.0) -> np.ndarray:
        """Soft prior beta*|i - center| over [max(center - back_s, floor), max(center + fwd_s, next onset
        + 1 s)], plus relock_inactive_cost on silent reference frames in vocal mode."""
        p = self.params
        c = round(center)
        hi_t = max(center + fwd_s * self.fps, self._next_onset(center + 0.5) + self.fps)
        lo = max(0, int(c - back_s * self.fps), min(c, math.ceil(floor)))
        hi = min(self.N, math.ceil(hi_t) + 1)
        idx = np.arange(lo, hi)
        prior = np.full(self.N, np.inf)
        prior[lo:hi] = p.anchor_beta * np.abs(idx - center) / self.fps
        if p.mode == "vocal":
            prior[lo:hi] += p.relock_inactive_cost * (~self.ref_active[lo:hi])
        return prior

    def _relock(self, n_replay: int) -> None:
        """Voice resumed n_replay frames ago: soft re-anchor around the position predicted for the
        onset frame, replay the buffered frames and let the published position settle (also backwards,
        within the prior) for relock_free_s. Provisional until confirmed (see _abort_relock)."""
        p = self.params
        n_replay = min(n_replay, len(self._buf))
        frames = list(self._buf)[-n_replay:]
        center = frames[0][1] if frames else self._out
        self._prov = _Provisional(self._out, self._coast_limit, self._limit_inline, self._wait, self._floor,
                                  self._frame, self._tempo, self._ratio, self._conf)
        self._confirm_left = round(p.relock_confirm_s * self.fps) - len(frames)
        free = len(frames) + round(p.relock_free_s * self.fps) if p.relock_free_s > 0 else 0
        run = self._free_run
        prior = self._pop_prior(center) if run else self._prior(center, p.relock_back_s, p.relock_fwd_s, self._floor)
        self._anchor(round(center), prior, free=free)
        self._status = TRACKING
        # _steps (tracked frames since the last start / jump / release / relocation) goes on: the relocation
        # history spans the pauses with the coasted positions, and restarting the count at every relock kept a
        # vocal tracker that lost its place after a skipped verse from checking for 6 s of unbroken singing
        out = self._out
        for x, _ in frames:
            self._track(x)
        if run:                       # pop: coasting already advanced the prediction over these frames: keep
            self._out = max(out, self._out - len(frames) * self._tempo)     # only the corrections
        if self._confirm_left <= 0:
            self._prov = None

    def _prov_cap(self) -> float:
        """Display bound of a provisional relock: the coast limit (or the coasted position if beyond),
        advancing at the coasting rate per active frame since the relock (the voice may have started
        the line), so an unconfirmed burst moves the display at most relock_confirm_s past the limit."""
        n_active = round(self.params.relock_confirm_s * self.fps) - self._confirm_left
        return max(self._prov.out, self._prov.limit) + max(0, n_active) * self._coast_rate()

    def _abort_relock(self) -> None:
        """The activity that triggered the relock ended before it was confirmed: restore coasting as
        if the relock had not happened (position advanced along the coasting trajectory)."""
        pv, self._prov = self._prov, None
        self._status = COASTING
        self._coast_limit, self._limit_inline, self._wait, self._floor = pv.limit, pv.inline, pv.wait, pv.floor
        self._tempo, self._ratio, self._conf = pv.tempo, pv.ratio, pv.conf
        traj = min(pv.out + (self._frame - pv.frame) * self._coast_rate(), max(pv.out, pv.limit))
        if self._free_run:                           # pop: the prediction ran on at the tempo meanwhile
            self._out = max(self._out, traj)
            return
        self._out = traj
        self._pub_reset = True                       # undo: back onto the coasting trajectory

    def _try_relocate(self) -> bool:
        """Coarse global search over the recent live history; switch to a candidate only if it is
        clearly better both coarsely and in a shadow-tracker replay of the last reloc_verify_s.

        While the key search runs the coarse search covers every candidate key (a wrong start in a
        transposed take is wrong in both): the neighbourhood baseline is the best key around the tracker,
        a candidate under another key is verified under that key and adopts it. All hypotheses are then
        replayed from the same anchor and the key evidence restarts."""
        p, rl = self.params, self._reloc
        X = np.stack([h[0] for h in self._hist])
        P = np.array([h[1] for h in self._hist])
        keys = [self._transpose]
        if self._keys is not None:
            keys += list(self._keys.shadows)
        costs = rl.end_costs_keys(X, self._chroma, [k - self._transpose for k in keys])
        if not np.isfinite(costs).any():
            return False
        main = self.oltw.pos
        jump = p.reloc_min_jump_s * self.fps
        j = int(np.clip(round((main - rl.win + 1) / rl.hop), 0, costs.shape[1] - 1))
        r = round((jump + rl.win) / rl.hop)
        c_near = float(costs[:, max(0, j - r):j + r + 1].min())   # best coarse match around the tracker
        pen = [(f, c * (1.0 + p.reloc_dist_penalty * abs(f - main) / (60.0 * self.fps)), key)
               for key, row in zip(keys, costs) for f, c in rl.candidates_from_costs(row, k=p.reloc_cands)]
        far = [fck for fck in pen if abs(fck[0] - main) > jump and fck[1] <= p.reloc_ratio * c_near]
        if not far:
            return False                             # nothing clearly better than the neighbourhood
        best = min(c for _, c, _ in far)
        far = [fck for fck in far if fck[1] <= best * p.reloc_tie + 1e-9]
        # nearest of the equally good ones (the current key first)
        f, _, key = min(far, key=lambda fck: (abs(fck[0] - main), fck[2] != self._transpose))
        V = min(len(X), round(p.reloc_verify_s * self.fps))
        start = f - V * self._tempo
        R_key = self.oltw.R if key == self._transpose else self._keys.shadows[key].R
        sh = FrameSyncOLTW(R_key, fps=self.fps, win_s=p.win_s, max_adv=p.max_adv, weights=p.weights,
                           hist_s=self.oltw.K / self.fps)
        lo, hi = max(0, round(start - self.fps)), min(self.N, round(start + self.fps) + 1)
        if hi <= lo:
            return False
        prior = np.full(self.N, np.inf)
        prior[lo:hi] = p.anchor_beta * np.abs(np.arange(lo, hi) - start) / self.fps
        sh.anchor(lo, prior)
        d_sh = [sh.step(x).d_pos for x in X[-V:]]
        d_main = 1.0 - np.einsum("ij,ij->i", self.oltw.R[P[-V:]], X[-V:])
        h = V // 2
        if not (np.mean(d_sh[-h:]) < p.reloc_ratio * float(np.mean(d_main[-h:]))
                and abs(sh.pos - main) > p.reloc_min_jump_s * self.fps):
            return False
        if self._free_run and p.pop_reloc_confirm_s > 0:
            # pop, locked: a singer's ad-lib / a line sung in the wrong place lasts a line, a skipped / repeated
            # section of the band goes on: switch only when the same target (moving on at the tempo, same
            # key) is verified again pop_reloc_confirm_s later
            pend = self._reloc_pending
            if (pend is None or pend[2] != key
                    or abs(sh.pos - (pend[0] + (self._frame - pend[1]) * self._tempo)) > p.pop_reloc_tol_s * self.fps):
                self._reloc_pending = (float(sh.pos), self._frame, key)
                return False
            if self._frame - pend[1] < p.pop_reloc_confirm_s * self.fps:
                return False
        self._reloc_pending = None
        if key != self._transpose:                   # the old main becomes the shadow of its key
            self._keys.swap(key, self._transpose, self.oltw)
            self._transpose = key
            self._reloc.set_reference(sh.R)
            self.key_switches += 1
        self.oltw = sh
        if self._keys is not None:                   # every hypothesis anchored and fed like the new main
            self._keys.replay(lo, prior, X[-V:])
            self._keys.restart()
        self._prov = None
        self._out = float(sh.pos)                    # a verified global jump may move backwards
        self._pub_reset = True
        self._ratio = 0.5 * (p.conf_lo + p.conf_hi)
        self._low_run = self._snap_run = self._stuck_run = 0
        self._steps = 0
        self.relocations += 1
        return True

    def _resume_at(self, pos: float, back_s: float, fwd_s: float) -> None:
        """Operator re-sync / hold release: soft anchor at pos (published immediately). In vocal mode
        with a silent singer: wait (coast) there; the next relock reaches back at most back_s. Pop: mix
        follows the DP from here until it locks again, vocal waits for the voice (singer silent)."""
        self._out = self._lock_from = pos
        self._grace = False
        self._prov = None
        self._snap_run = self._stuck_run = 0
        self._reloc_pending = None
        if self.params.mode == "mix":
            self._locked = False                       # pop: follow the DP until it moves on from here
        elif not self._gate.active:
            self._status = COASTING
            self._locked = False                       # pop too: wait at the target for the voice
            self._coast_begin()
            self._floor = pos - back_s * self.fps
            return
        self._anchor(round(pos), self._prior(pos, back_s, fwd_s))
        self._status = TRACKING
        self._steps = 0
