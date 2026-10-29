"""When each cue fires: a pure scheduler on an injected clock, and the realtime runner that drives it.

`CueScheduler` (pure, deterministic, not thread-safe) is fed tracker updates as `TrackPoint`s and asked
which cues are due by a clock time `now`. Between updates it extrapolates the reference position at the
published rate, at most `max_extrap_s` past the last update:

    P(now) = p.ref_time + v * clip(now - p.t, 0, max_extrap_s)     v = p.rate if p.status in MOVING else 0

so a cue fires at its sub-frame time instead of at the next 40 ms tracker frame, while a stalled tracker
never carries the position (and the cues) further than `max_extrap_s`. A cue whose target has an output
latency L is sent L live seconds early: it is due when P(now) + v * L >= cue.time (reference look-ahead
v * L; nothing is looked ahead while the position does not move).

Every update is compared with the previous point extrapolated to the update's time (E):

    forward by more than catchup_s       a jump: the unfired cues in (E, new] go through the skip policy
                                         (skip: none sent; latest: the latest per (target, address) sent;
                                         all: every one sent, late)
    backward by rearm_back_s or more     a new pass: cues after the new position fire again
    anything else                        tracking noise: crossed cues fire through `due` (late, "catchup");
                                         a small backward settle never fires a cue twice

Coasting (vocal mode, the singer paused) is no evidence of where the music is: when the voice comes back
the tracker relocks and may settle back over what it coasted through (up to its relock reach, more than
rearm_back_s). A run of points that do not follow the music (status "coasting", a vocal guard `hold_from`,
or `settling`: the relock is provisional or may still step back) and the first followed point after it
measure a backward move from where that run began (the position last followed), not from the coasted
position: only a move rearm_back_s or more behind that is a new pass; a relock settling back over coasted
positions re-fires nothing.

Cues are handled once per pass: a fired or skipped cue stays handled until a jump back, an operator jump
or `start` re-arms it. Unfired cues are kept per latency value ("lane") as sorted indices into the
(time, id)-sorted cue list; within a lane the due time grows with the cue time, so `due` and
`next_due_time` only look at the head of each lane: O(lanes + k), not O(all cues).

A tracker jump back may be a wrong re-localisation that a later update undoes. The cues it re-arms were
all handled in this pass (every handled cue lies at or before the furthest position, look-ahead included,
that the pass reached: its *reach*), so they are remembered as *restorable*. A later forward move of at
least rearm_back_s (a jump, or a catch-up move that large when catchup_s > rearm_back_s) is a return: if
it lands less than rearm_back_s behind the reach (a settle against where the pass was) or beyond it, every
restorable cue is marked handled again, silently (no Fire at all: they were sent before the glitch,
look-ahead lanes included); landing further behind (a new pass from there), or a smaller forward jump, only
the restorable cues at or before the landing position. Only the other cues it crosses go through the skip
policy or fire late. A restorable cue that the position reaches normally (the band really went back) fires
as usual and stops being restorable, and so does every restorable cue on the (target, address) of a cue
sent after the jump back (the device no longer holds what they set: "latest" chases them on the return).
An operator jump or `start` begins a new pass: no cue stays restorable.

`start` and `jump` take the requested time as a separate `boundary`: the tracker publishes it plus its
input-latency look-ahead. Cues before the boundary are jumped over (skip policy); cues from the boundary
to the published position are due at once as "track"; a point behind the boundary (a negative input
latency) is moved up to it, so a cue at the requested time always fires at once.

The guard (`hold_from`, vocal mode waiting for the voice) can hold cues the position has already reached.
A forward jump hands those to the skip policy together with the cues in (E, new]; they never fire late
after the jump's records.

`CueRunner` is the realtime driver: a timer thread sleeps on a `threading.Condition` until the next due
time (at most 50 ms, in shrinking steps because long timed waits overshoot by milliseconds on macOS;
sends land ~0.1 ms after the ideal time) and fires what is due; `feed` (audio thread) first fires what the
previous point had made due by the update's time (a timer that has not woken yet), then applies the
update and fires what became due at once. One lock guards the scheduler and is held only to work out the
fires; they are then sent in order outside it by one thread at a time (whichever finds no send in flight;
the others only queue), so a slow `send` never blocks `feed` or the views, and `send` may take a lock
that the caller of `feed` holds. `on_fire` is called outside every runner lock, in record order, by one
calling thread at a time.
"""
from __future__ import annotations

import bisect
import collections
import dataclasses
import heapq
import itertools
import logging
import math
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .cues import Cue
from .osc import SendResult
from .settings import SKIP_POLICIES

log = logging.getLogger(__name__)

MOVING = ("tracking", "coasting")
REASONS = ("track", "catchup", "chase", "skipped", "start", "manual")

DUE_EPS = 1e-9      # ref s: a cue is due when the position is within this of its time
POS_EPS = 1e-6      # ref s: start / jump / re-arm boundaries ("at the position" counts as ahead)
FRAME_S = 0.04      # a fire whose ideal time was more than one tracker frame before the update is "catchup"
TIMER_MAX_WAIT_S = 0.05   # the timer thread re-checks at least this often
TIMER_FINE_S = 0.001      # waits shorter than this are not split (they overshoot by ~0.1 ms)


@dataclass
class SchedulerParams:
    catchup_s: float = 1.0      # forward move <= this: every crossed cue fires (late); larger: a jump
    rearm_back_s: float = 1.0   # backward move >= this re-arms the cues after the new position
    max_extrap_s: float = 0.1   # extrapolate at most this far past the last tracker update
    skip_policy: str = "latest"  # what a jump forward does with the cues it skipped

    def validated(self) -> SchedulerParams:
        """A copy, or ValueError for an unknown policy or a negative / non-finite duration."""
        if self.skip_policy not in SKIP_POLICIES:
            raise ValueError(f"skip_policy must be one of {', '.join(SKIP_POLICIES)}: {self.skip_policy!r}")
        for name in ("catchup_s", "rearm_back_s", "max_extrap_s"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be a finite number >= 0: {v!r}")
        return dataclasses.replace(self)


@dataclass(frozen=True)
class TrackPoint:
    """One tracker update as the scheduler sees it."""
    ref_time: float                  # published reference time (s)
    rate: float                      # TrackerState.tempo (0 = not moving)
    status: str
    t: float                         # clock time of the update (same clock as `now` everywhere)
    hold_from: float | None = None   # cues with time >= hold_from wait (vocal mode waiting for the voice)
    settling: bool = False           # a relock may still move the position back over what it coasted through


@dataclass(frozen=True)
class Fire:
    cue: Cue
    at: float                   # clock time it fires
    ref_time: float             # extrapolated position (without latency look-ahead) at `at`
    reason: str                 # "track" | "catchup" | "chase" | "skipped" | "start" (see REASONS)
    send: bool = True           # False for "skipped"
    ref_late: float = 0.0       # ref s past the cue at `at`: position + rate * latency - cue.time (>= 0)


def _point_ok(pt: TrackPoint) -> bool:
    return math.isfinite(pt.ref_time) and math.isfinite(pt.t)


def _followed(pt: TrackPoint) -> bool:
    """The position follows the music: not coasting, not waiting for the voice, no relock settling."""
    return pt.status != "coasting" and pt.hold_from is None and not pt.settling


def _bad_point(pt: TrackPoint) -> ValueError:
    return ValueError(f"track point without a finite position / time: {pt}")


def _rate(p: TrackPoint) -> float:
    """Rate at which the position moves after `p` (also the look-ahead rate): 0 unless moving."""
    if p.status not in MOVING:
        return 0.0
    r = float(p.rate)
    return r if math.isfinite(r) and r > 0 else 0.0


class CueScheduler:
    """Pure cue scheduler (see the module docstring). Not thread-safe: `CueRunner` serialises access."""

    def __init__(self, cues: Iterable[Cue], latency_of: Callable[[Cue], float],
                 params: SchedulerParams | None = None):
        self._params = (params or SchedulerParams()).validated()
        self._latency_of = latency_of
        self._default_target: str | None = None
        self._p: TrackPoint | None = None
        self._cues: list[Cue] = []
        self._times: list[float] = []
        self._lat: list[float] = []
        self._fired: list[bool] = []
        self._restorable: set[int] = set()   # unfired cues a tracker jump back re-armed (see the docstring)
        self._reach = 0.0                    # furthest position of the pass the restorable cues were handled in
        self._coast_from: float | None = None   # where a stretch not following the music began (None: none)
        self._track_from: float | None = None   # start / jump boundary of the current point (see `due`)
        self._lanes: dict[float, list[int]] = {}
        sorted_cues = self._sorted(cues)
        self._install(sorted_cues, [False] * len(sorted_cues), set())

    # -- setup -------------------------------------------------------------------------------------
    @staticmethod
    def _sorted(cues: Iterable[Cue]) -> list[Cue]:
        out = []
        for c in cues:
            if math.isfinite(c.time):
                out.append(c)
            else:
                log.warning("cue %s has no finite time (%r): ignored", c.id, c.time)
        out.sort(key=lambda c: (c.time, c.id))
        return out

    def _latency(self, cue: Cue) -> float:
        try:
            v = float(self._latency_of(cue))
        except Exception as e:  # noqa: BLE001 - a bad latency map must not stop the show: send on time
            log.warning("latency of cue %s: %s (using 0)", cue.id, e)
            return 0.0
        # lanes are keyed by latency: values equal to the microsecond (100 ms as 0.1 or as 0.3 - 0.2) share
        # one, so equal-time cues on them become due together and fire in id order
        return round(v, 6) if math.isfinite(v) and v > 0 else 0.0

    def _install(self, cues: list[Cue], fired: list[bool], restorable: set[int]) -> None:
        self._cues = cues
        self._times = [float(c.time) for c in cues]
        self._lat = [self._latency(c) for c in cues]
        self._fired = fired
        self._restorable = restorable
        self._rebuild_lanes()

    def _rebuild_lanes(self) -> None:
        lanes: dict[float, list[int]] = {}
        for i, (lat, fired) in enumerate(zip(self._lat, self._fired)):
            if not fired:
                lanes.setdefault(lat, []).append(i)
        self._lanes = lanes

    def set_cues(self, cues: Iterable[Cue], now: float) -> None:
        """Replace the cues (file reloaded / edited). Fired and restorable state carry over by `Cue.key()`;
        a new cue behind the position (time < P(now) - 1e-6) is marked fired without sending: never fire it
        late."""
        new = self._sorted(cues)
        carried: dict[tuple, collections.deque[tuple[bool, bool]]] = {}
        for i, (c, f) in enumerate(zip(self._cues, self._fired)):
            carried.setdefault(c.key(), collections.deque()).append((f, i in self._restorable))
        behind = None if self._p is None else self._pos(self._p, now) - POS_EPS
        fired: list[bool] = []
        restorable: set[int] = set()
        for j, c in enumerate(new):
            q = carried.get(c.key())
            if q:
                f, r = q.popleft()
                if r:
                    restorable.add(j)
            else:
                f = behind is not None and c.time < behind
            fired.append(f)
        self._install(new, fired, restorable)

    def set_params(self, params: SchedulerParams) -> None:
        self._params = params.validated()

    @property
    def params(self) -> SchedulerParams:
        return dataclasses.replace(self._params)

    def set_latency_of(self, fn: Callable[[Cue], float]) -> None:
        self._latency_of = fn
        self._lat = [self._latency(c) for c in self._cues]
        self._rebuild_lanes()

    def set_default_target(self, name: str | None) -> None:
        """Target name that cues without a target go to: "latest" groups them with that target's cues."""
        self._default_target = name

    # -- position model ----------------------------------------------------------------------------
    def _pos(self, p: TrackPoint, now: float) -> float:
        v = _rate(p)
        if v == 0.0:
            return p.ref_time
        return p.ref_time + v * min(max(now - p.t, 0.0), self._params.max_extrap_s)

    def position(self, now: float) -> float:
        """Extrapolated reference position at clock time `now` (0 before `start`)."""
        return 0.0 if self._p is None else self._pos(self._p, now)

    def _due_time(self, i: int, p: TrackPoint, v: float) -> float:
        """Clock time at which cue i becomes due under point p: the inverse of the position model, so
        `due(now)` fires exactly the cues with due time <= now. -inf: due regardless of time (the position
        does not move and is at / past the cue); +inf: not before the next update."""
        lead = self._times[i] - DUE_EPS - p.ref_time - v * self._lat[i]
        if v == 0.0:
            return -math.inf if lead <= 0 else math.inf
        t = p.t + lead / v
        return t if t <= p.t + self._params.max_extrap_s else math.inf

    def _hold_end(self, p: TrackPoint) -> int:
        """Index of the first cue held by the guard (cues with time >= hold_from wait)."""
        h = p.hold_from
        if h is None or math.isnan(h):
            return len(self._times)
        return bisect.bisect_left(self._times, h)

    def _crossed_from(self, p: TrackPoint, e: float) -> int:
        """First index a forward move from p (extrapolated to e) crosses: the cues after e, and before them
        the cues p's guard still held although the position had reached them (only held cues there are
        unfired: the others were due)."""
        return min(bisect.bisect_right(self._times, e), self._hold_end(p))

    # -- tracker updates ---------------------------------------------------------------------------
    @staticmethod
    def _bounded(pt: TrackPoint, boundary: float | None) -> tuple[TrackPoint, float]:
        """(point, boundary) of a start / jump: the boundary is the requested time (default: the point's
        position); a point behind it (a negative input latency) is moved up to it."""
        if not _point_ok(pt):
            raise _bad_point(pt)
        if boundary is None:
            return pt, pt.ref_time
        b = float(boundary)
        if not math.isfinite(b):
            raise ValueError(f"start / jump boundary must be finite: {boundary!r}")
        return (dataclasses.replace(pt, ref_time=b) if pt.ref_time < b else pt), b

    def start(self, pt: TrackPoint, boundary: float | None = None) -> list[Fire]:
        """Arm every cue and start at `pt`; `boundary` = the requested start time (default: pt.ref_time; see
        the module docstring). Cues before it are jumped over (skip policy, reasons "start" / "skipped");
        cues from it to the start position are due at once (returned by `due`, reason "track")."""
        pt, b = self._bounded(pt, boundary)
        self._fired = [False] * len(self._cues)
        self._restorable = set()
        self._rebuild_lanes()
        self._p = pt
        self._track_from = b
        self._coasting_at(pt)
        end = bisect.bisect_left(self._times, b - POS_EPS)
        return self._policy(self._take(0, end), pt, "start")

    def update(self, pt: TrackPoint) -> list[Fire]:
        """Apply a tracker update; returns only the jump-forward records (`at` = pt.t). Callers then call
        `due(now)`. An update before `start` starts; a point without a finite position is ignored."""
        if not _point_ok(pt):
            log.warning("ignoring a track point without a finite position / time: %s", pt)
            return []
        p = self._p
        if p is None:
            return self.start(pt)
        prm = self._params
        e = self._pos(p, pt.t)
        d = pt.ref_time - e
        fires: list[Fire] = []
        new_pass = False
        if d > prm.catchup_s or (d > 0 and d >= prm.rearm_back_s):
            if self._restorable:
                self._restore_on_return(pt.ref_time, d)
            if d > prm.catchup_s:
                hi = bisect.bisect_right(self._times, pt.ref_time)
                fires = self._policy(self._take(self._crossed_from(p, e), hi), pt, "chase")
        elif d < 0 and -d >= prm.rearm_back_s and (
                self._coast_from is None or self._coast_from - pt.ref_time >= prm.rearm_back_s):
            self._reach = max(self._reach, e) if self._restorable else e
            self._restorable.update(self._rearm(bisect.bisect_right(self._times, pt.ref_time + POS_EPS)))
            new_pass = True
        self._follow_coasting(pt, e, new_pass)
        self._p = pt
        self._track_from = None
        return fires

    def jump(self, pt: TrackPoint, boundary: float | None = None) -> list[Fire]:
        """Operator jump (Go / jump to time / jump to cue); `pt` = tracker state after the jump, `boundary`
        = the requested time (default: pt.ref_time; see the module docstring). Every cue at or after the
        boundary is re-armed (a jump exactly to a cue's time makes it due at once); moving forward, the
        unfired cues between the old position and the boundary go through the skip policy."""
        pt, b = self._bounded(pt, boundary)
        p = self._p
        if p is None:
            return self.start(pt, b)
        e = self._pos(p, pt.t)
        i_b = bisect.bisect_left(self._times, b - POS_EPS)
        self._restorable = set()
        fires: list[Fire] = []
        if b > e:
            fires = self._policy(self._take(self._crossed_from(p, e), i_b), pt, "chase")
        self._rearm(i_b)
        self._p = pt
        self._track_from = b
        self._coasting_at(pt)
        return fires

    def _coasting_at(self, pt: TrackPoint) -> None:
        """A new pass begins at `pt`: a stretch that does not follow the music, if any, begins there."""
        self._coast_from = None if _followed(pt) else pt.ref_time

    def _follow_coasting(self, pt: TrackPoint, e: float, new_pass: bool) -> None:
        """A run of points that do not follow the music (coasting, waiting for the voice, a relock settling)
        begins where the position was last followed (`_coast_from`; e = the previous point extrapolated to
        pt.t) and ends at the next followed point, which `update` still judges against it (the relock's last
        settling step may move back)."""
        if new_pass:
            self._coasting_at(pt)
        elif _followed(pt):
            self._coast_from = None
        elif self._coast_from is None:
            self._coast_from = min(e, pt.ref_time)

    def _restore_on_return(self, landing: float, d: float) -> None:
        """A forward move after a tracker jump back (see the module docstring): mark restorable cues handled
        again, silently; all of them when a return of at least rearm_back_s lands less than rearm_back_s
        behind the reach of the pass (or beyond it), else those at or before the landing position."""
        back = self._params.rearm_back_s
        if d >= back and self._reach - landing < back:
            hi = len(self._times)
        else:
            hi = bisect.bisect_right(self._times, landing + POS_EPS)
        self._restore(0, hi)

    def _take(self, lo: int, hi: int) -> list[int]:
        """Mark the unfired cues with index in [lo, hi) fired; their indices in (time, id) order."""
        if lo >= hi:
            return []
        out: list[int] = []
        for lane in self._lanes.values():
            a, b = bisect.bisect_left(lane, lo), bisect.bisect_left(lane, hi)
            if a < b:
                out.extend(lane[a:b])
                del lane[a:b]
        out.sort()
        for i in out:
            self._fired[i] = True
        if self._restorable:
            self._restorable.difference_update(out)
        return out

    def _restore(self, lo: int, hi: int) -> None:
        """Silently mark the restorable cues with index in [lo, hi) handled again."""
        r = self._restorable
        back = [i for i in r if lo <= i < hi]
        for i in back:
            lane = self._lanes.get(self._lat[i], [])
            k = bisect.bisect_left(lane, i)
            if k < len(lane) and lane[k] == i:
                del lane[k]
            self._fired[i] = True
        r.difference_update(back)

    def _key(self, i: int) -> tuple:
        """What a cue sets on the receiving side: (target, address); "latest" chases one cue per key."""
        c = self._cues[i]
        return (c.target or self._default_target, c.address)

    def _overwritten(self, sent: Iterable[int]) -> None:
        """Cues just sent set their (target, address) anew: the restorable cues on those keys no longer
        describe what the devices hold, so a return does not restore them silently (they fire again or go
        through the skip policy)."""
        r = self._restorable
        if not r:
            return
        sent = list(sent)
        r.difference_update(sent)
        keys = {self._key(i) for i in sent}
        r.difference_update([j for j in r if self._key(j) in keys])

    def _rearm(self, i0: int) -> list[int]:
        """Every cue from index i0 on becomes unfired (a new pass over that part of the song); returns the
        indices that were fired."""
        added: dict[float, list[int]] = {}
        out: list[int] = []
        for i in range(i0, len(self._cues)):
            if self._fired[i]:
                self._fired[i] = False
                added.setdefault(self._lat[i], []).append(i)
                out.append(i)
        for lat, idx in added.items():
            lane = self._lanes.setdefault(lat, [])
            lane.extend(idx)
            lane.sort()
        return out

    def _policy(self, idx: list[int], pt: TrackPoint, chase_reason: str) -> list[Fire]:
        """Records for cues jumped over (already marked fired): sent ones first, each group in time order."""
        if not idx:
            return []
        policy = self._params.skip_policy
        if policy == "all":
            chase = set(idx)
        elif policy == "skip":
            chase = set()
        else:
            latest: dict[tuple, int] = {}
            for i in idx:                         # (time, id) order: the last one per key wins
                latest[self._key(i)] = i
            chase = set(latest.values())
        self._overwritten(chase)
        v = _rate(pt)

        def rec(i: int, reason: str, send: bool) -> Fire:
            c = self._cues[i]
            late = max(0.0, pt.ref_time + v * self._lat[i] - c.time)
            return Fire(c, pt.t, pt.ref_time, reason, send, late)

        return ([rec(i, chase_reason, True) for i in idx if i in chase]
                + [rec(i, "skipped", False) for i in idx if i not in chase])

    # -- firing ------------------------------------------------------------------------------------
    def due(self, now: float) -> list[Fire]:
        """Cues due by `now` (marked fired), in (time, id) order."""
        p = self._p
        if p is None:
            return []
        v = _rate(p)
        hold_end = self._hold_end(p)
        picked: list[int] = []
        for lane in self._lanes.values():
            k = 0
            while k < len(lane) and lane[k] < hold_end and self._due_time(lane[k], p, v) <= now:
                k += 1
            if k:
                picked.extend(lane[:k])
                del lane[:k]
        if not picked:
            return []
        picked.sort()
        self._overwritten(picked)
        pos = self._pos(p, now)
        tf = self._track_from
        out = []
        for i in picked:
            self._fired[i] = True
            c, lat = self._cues[i], self._lat[i]
            late_at_point = p.ref_time + v * lat - c.time
            # ideal due time earlier than the update by more than a frame: the update moved past it (a cue
            # from a start / jump boundary on is at the requested position: "track")
            catchup = late_at_point > (v if v > 0 else 1.0) * FRAME_S and (tf is None or c.time < tf - POS_EPS)
            out.append(Fire(c, now, pos, "catchup" if catchup else "track", True,
                            max(0.0, pos + v * lat - c.time)))
        return out

    def next_due_time(self) -> float | None:
        """Clock time of the next due cue under the current point (<= now when one is already due), None
        when nothing can become due before the next update."""
        p = self._p
        if p is None:
            return None
        v = _rate(p)
        hold_end = self._hold_end(p)
        best = math.inf
        for lane in self._lanes.values():
            if lane and lane[0] < hold_end:           # a lane's head is its earliest due cue
                best = min(best, self._due_time(lane[0], p, v))
        if best == math.inf:
            return None
        return p.t if best == -math.inf else best

    def upcoming(self, now: float, n: int = 3) -> list[tuple[Cue, float | None]]:
        """The next `n` unfired cues in (time, id) order with their ETA in clock seconds (0 when due; None
        when unknown: not started, not moving, or held by the guard)."""
        if n <= 0:
            return []
        idx = list(itertools.islice(heapq.merge(*self._lanes.values()), n))
        p = self._p
        if p is None:
            return [(self._cues[i], None) for i in idx]
        v = _rate(p)
        hold_end = self._hold_end(p)
        pos = self._pos(p, now)
        out: list[tuple[Cue, float | None]] = []
        for i in idx:
            eta: float | None = None
            if i < hold_end:
                if self._due_time(i, p, v) <= now:
                    eta = 0.0
                elif v > 0:
                    eta = max(0.0, (self._times[i] - DUE_EPS - v * self._lat[i] - pos) / v)
            out.append((self._cues[i], eta))
        return out

    def fired_ids(self) -> set[int]:
        """Ids of the cues handled in the current pass (fired or skipped)."""
        return {c.id for c, f in zip(self._cues, self._fired) if f}

    @property
    def cues(self) -> list[Cue]:
        return list(self._cues)

    @property
    def point(self) -> TrackPoint | None:
        return self._p


# ------------------------------------------------------------------------------------------- runner


@dataclass
class FireRecord:
    """What the console / show log shows for one fire."""
    seq: int                    # increasing from 1 per runner
    cue_id: int
    time: float                 # the cue's reference time
    target: str
    address: str
    message: str
    reason: str
    sent: bool
    error: str | None
    wall: float                 # time.time() when fired
    ref_time: float
    ref_late_ms: float
    live_time: float | None
    dry_run: bool

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _message_text(cue: Cue) -> str:
    if cue.message:
        return cue.message
    return " ".join([cue.address, *(repr(v) if tag == "s" else str(v) for tag, v in cue.args)])


class CueRunner:
    """Realtime driver of a CueScheduler: timer thread + sending + fire log (see the module docstring).

    `send(cue)` is called without any runner lock held, by one thread at a time for scheduled fires (in
    fire order); a `fire_manual` send may run at the same time as one of those. `live_time()` is called
    before the runner's lock is taken. `timer=False` runs no thread: the caller drives `poll()` (tests,
    offline tools). Fires still waiting to be sent when `stop()` is called are dropped.
    """

    def __init__(self, scheduler: CueScheduler, send: Callable[[Cue], SendResult], *,
                 clock: Callable[[], float] = time.perf_counter, armed: bool = True,
                 default_target: str = "main", on_fire: Callable[[FireRecord], None] | None = None,
                 live_time: Callable[[], float | None] | None = None, log_size: int = 500,
                 timer: bool = True):
        self._sched = scheduler
        self._send = send
        self._clock = clock
        self._armed = bool(armed)
        self._default = default_target
        scheduler.set_default_target(default_target)
        self._on_fire = on_fire
        self._live_time = live_time
        self._use_timer = timer
        self._cond = threading.Condition(threading.RLock())   # re-entrant: callbacks may call back in
        self._log: collections.deque[FireRecord] = collections.deque(maxlen=max(1, int(log_size)))
        self._seq = 0
        self._counts = {"sent": 0, "skipped": 0, "errors": 0, "dry": 0}
        self._state = "idle"                     # idle | running | stopped
        self._gen = 0                            # timer thread generation (a stale thread exits)
        self._thread: threading.Thread | None = None
        self._pending: collections.deque[tuple[Fire, float | None]] = collections.deque()   # to send
        self._sending = False                    # a thread is sending `_pending`
        self._outbox: collections.deque[FireRecord] = collections.deque()   # waiting for on_fire
        self._delivering = False

    # -- lifecycle ---------------------------------------------------------------------------------
    def start(self, pt: TrackPoint, boundary: float | None = None) -> None:
        """(Re)start the pass at `pt` (scheduler.start; `boundary` = the requested start time), fire what is
        due at once, start the timer thread."""
        live = self._live()
        with self._cond:
            self._start_locked(pt, live, boundary)
        self._flush()
        self._deliver()

    def _start_locked(self, pt: TrackPoint, live: float | None, boundary: float | None = None) -> None:
        fires = self._sched.start(pt, boundary)
        fires += self._sched.due(self._clock())
        if self._use_timer and (self._state != "running" or self._thread is None or not self._thread.is_alive()):
            self._gen += 1
            self._thread = threading.Thread(target=self._run, args=(self._gen,), name="cue-runner", daemon=True)
            # started under the lock: a stop() cannot come between creating and starting it (join would
            # fail); the thread itself waits for the lock
            self._thread.start()
        self._state = "running"
        self._queue(fires, live)
        self._cond.notify_all()

    def stop(self) -> None:
        """Stop firing (later `feed` / `jump` calls are ignored; fires not sent yet are dropped) and join
        the timer thread (1 s)."""
        with self._cond:
            self._state = "stopped"
            self._gen += 1
            thread, self._thread = self._thread, None
            self._pending.clear()
            self._cond.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    @property
    def running(self) -> bool:
        with self._cond:
            return self._state == "running"

    # -- tracker / operator input ------------------------------------------------------------------
    def feed(self, pt: TrackPoint) -> None:
        """Audio thread: apply a tracker update and fire what is due now (starts the pass when called
        before `start`; ignored after `stop`)."""
        self._apply(pt, jump=False)

    def jump(self, pt: TrackPoint, boundary: float | None = None) -> None:
        """Operator jump; `pt` = tracker state after the jump, `boundary` = the requested time."""
        self._apply(pt, jump=True, boundary=boundary)

    def _apply(self, pt: TrackPoint, *, jump: bool, boundary: float | None = None) -> None:
        if not jump and not _point_ok(pt):           # never raise on the audio thread
            log.warning("ignoring a track point without a finite position / time: %s", pt)
            return
        live = self._live()
        with self._cond:
            if self._state == "stopped":
                return
            if self._state == "idle":
                self._start_locked(pt, live, boundary)
            else:
                # before anything is fired: nothing gets lost
                if not _point_ok(pt):
                    raise _bad_point(pt)
                if boundary is not None and not math.isfinite(boundary):
                    raise ValueError(f"start / jump boundary must be finite: {boundary!r}")
                now = self._clock()
                # what the current point made due by the update's time goes first (the timer may not have
                # woken yet): after a jump it would come after the jump's records, out of time order
                fires = self._sched.due(min(pt.t, now))
                fires += self._sched.jump(pt, boundary) if jump else self._sched.update(pt)
                fires += self._sched.due(now)
                self._queue(fires, live)
                self._cond.notify_all()
        self._flush()
        self._deliver()

    def poll(self) -> list[FireRecord]:
        """Fire every cue due now (what the timer thread does when it wakes); returns the records this call
        made (fires queued behind a send in flight on another thread are recorded by that thread)."""
        return self._fire_due(None)[1]

    def fire_manual(self, cue: Cue) -> FireRecord:
        """Send `cue` now regardless of `armed` (reason "manual"); the scheduler's pass is not changed."""
        live = self._live()
        with self._cond:
            pos = self._sched.position(self._clock())
        sent, error = self._do_send(cue)
        with self._cond:
            rec = self._record(cue, "manual", sent, error, pos, 0.0, live, dry_run=False)
        self._deliver()
        return rec

    # -- settings ----------------------------------------------------------------------------------
    @property
    def armed(self) -> bool:
        with self._cond:
            return self._armed

    @armed.setter
    def armed(self, on: bool) -> None:
        with self._cond:
            self._armed = bool(on)

    def set_cues(self, cues: Iterable[Cue]) -> None:
        with self._cond:
            self._sched.set_cues(cues, self._clock())
            self._cond.notify_all()

    def set_params(self, params: SchedulerParams) -> None:
        with self._cond:
            self._sched.set_params(params)
            self._cond.notify_all()

    def set_latency_of(self, fn: Callable[[Cue], float]) -> None:
        with self._cond:
            self._sched.set_latency_of(fn)
            self._cond.notify_all()

    def set_default_target(self, name: str) -> None:
        with self._cond:
            self._default = name
            self._sched.set_default_target(name)

    # -- views -------------------------------------------------------------------------------------
    def records(self, after_seq: int = 0, limit: int = 100) -> list[FireRecord]:
        """The latest `limit` records with seq > after_seq, oldest first."""
        if limit <= 0:
            return []
        with self._cond:
            recs = [r for r in self._log if r.seq > after_seq]
        return recs[-limit:]

    @property
    def last_seq(self) -> int:
        with self._cond:
            return self._seq

    def upcoming(self, n: int = 3) -> list[tuple[Cue, float | None]]:
        with self._cond:
            return self._sched.upcoming(self._clock(), n)

    def position(self) -> float:
        with self._cond:
            return self._sched.position(self._clock())

    def counts(self) -> dict[str, int]:
        with self._cond:
            return dict(self._counts)

    # -- internals ---------------------------------------------------------------------------------
    def _live(self) -> float | None:
        """Live time for the records; read before taking the runner lock (it may take the session's)."""
        if self._live_time is None:
            return None
        try:
            v = self._live_time()
            return None if v is None or not math.isfinite(v) else float(v)
        except Exception as e:  # noqa: BLE001 - the record is still useful without it
            log.warning("live_time() failed: %s", e)
            return None

    def _do_send(self, cue: Cue) -> tuple[bool, str | None]:
        try:
            res = self._send(cue)
        except Exception as e:  # noqa: BLE001 - logged on the record; never stops the show
            log.warning("sending cue %s failed: %r", cue.id, e)
            return False, f"send failed: {e!r}"
        if getattr(res, "ok", False):
            return True, None
        return False, getattr(res, "error", None) or "send failed"

    def _record(self, cue: Cue, reason: str, sent: bool, error: str | None, ref_time: float,
                ref_late: float, live: float | None, *, dry_run: bool) -> FireRecord:
        """Log one fire (called with the lock held)."""
        self._seq += 1
        rec = FireRecord(self._seq, cue.id, cue.time, cue.target or self._default, cue.address,
                         _message_text(cue), reason, sent, error, time.time(), ref_time, ref_late * 1000.0,
                         live, dry_run)
        self._log.append(rec)
        self._counts["sent"] += sent
        self._counts["errors"] += error is not None
        self._counts["dry"] += dry_run
        self._counts["skipped"] += reason == "skipped"
        if self._on_fire is not None:
            self._outbox.append(rec)
        return rec

    def _queue(self, fires: list[Fire], live: float | None) -> None:
        """Queue fires for `_flush` in fire order (called with the lock held)."""
        self._pending.extend((f, live) for f in fires)

    def _flush(self) -> list[FireRecord]:
        """Send (when armed) and record the queued fires in order, without holding the lock during a send.
        One thread sends at a time; fires queued meanwhile by other threads are sent by it. Returns the
        records this call made."""
        out: list[FireRecord] = []
        with self._cond:
            if self._sending or not self._pending:
                return out
            self._sending = True
        try:
            while True:
                with self._cond:
                    if not self._pending:
                        self._sending = False
                        return out
                    f, live = self._pending.popleft()
                    armed = self._armed
                sent, error, dry = False, None, False
                if f.send:
                    if armed:
                        sent, error = self._do_send(f.cue)
                    else:
                        dry = True
                with self._cond:
                    out.append(self._record(f.cue, f.reason, sent, error, f.ref_time, f.ref_late, live,
                                            dry_run=dry))
        except BaseException:
            with self._cond:
                self._sending = False
            raise

    def _deliver(self) -> None:
        """Pass queued records to on_fire in seq order, without holding any runner lock. One thread
        delivers at a time; records queued meanwhile by other threads are delivered by it."""
        if self._on_fire is None:
            return
        with self._cond:
            if self._delivering:
                return
            self._delivering = True
        try:
            while True:
                with self._cond:
                    if not self._outbox:
                        self._delivering = False
                        return
                    rec = self._outbox.popleft()
                try:
                    self._on_fire(rec)
                except Exception:
                    log.exception("on_fire callback failed")
        except BaseException:
            with self._cond:
                self._delivering = False
            raise

    def _fire_due(self, gen: int | None) -> tuple[int, list[FireRecord]]:
        """Fire what is due now: (number of fires, records this call made)."""
        live = self._live()
        with self._cond:
            if self._state != "running" or (gen is not None and gen != self._gen):
                return 0, []
            fires = self._sched.due(self._clock())
            self._queue(fires, live)
        recs = self._flush()
        self._deliver()
        return len(fires), recs

    def _run(self, gen: int) -> None:
        while True:
            with self._cond:
                if gen != self._gen or self._state != "running":
                    return
                try:
                    nd = self._sched.next_due_time()
                except Exception:
                    log.exception("cue runner: next_due_time failed")
                    nd = None
                now = self._clock()
                if nd is None:
                    self._cond.wait(TIMER_MAX_WAIT_S)
                    continue
                if nd > now:
                    # timed waits overshoot by up to ~1/4 of their length (macOS timer coalescing): close in
                    # on the due time in shrinking steps and finish with a sub-millisecond wait
                    rem = nd - now
                    self._cond.wait(min(TIMER_MAX_WAIT_S, rem if rem <= TIMER_FINE_S else rem * 0.75))
                    continue
            try:
                n = self._fire_due(gen)[0]
            except Exception:
                log.exception("cue runner: firing failed")
                n = 0
            if not n:
                # another thread fired it first, or due() disagrees with next_due_time(): never spin
                with self._cond:
                    if gen == self._gen and self._state == "running":
                        nd = self._sched.next_due_time()
                        if nd is not None and nd <= self._clock():
                            self._cond.wait(0.001)
