"""Reference time -> lyric cursor (current line / syllable). The prompter JS mirrors these rules.

Rules (all times are reference seconds):
- Line i is "current" from its start until the next line's start; `line_idx` is the last line whose
  start <= t (-1 before the first line). `in_gap` is True before the first line, after the current
  line's end (until the next line starts) and after the last line.
- Within a line that is not in a gap, `syl_idx` is the last syllable whose start <= t (-1 before the
  first one) and `syl_progress` = (t - start) / (end - start) clipped to [0, 1]; in a gap syl_idx = -1.
- Missing times are filled defensively: a line's start/end fall back to its first/last timed syllable;
  a line without any time is zero-length at the previous timed line's end (or the next line's start);
  a line without an end ends where the next line starts; syllables without times inherit the
  neighbouring syllable / line boundaries. Starts are forced non-decreasing so binary search is valid.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from .text.model import Line


@dataclass
class Cursor:
    line_idx: int                # current or most recently finished line, -1 before the first
    syl_idx: int                 # index into line.syllables(), -1 if none active
    syl_progress: float          # 0..1 within the active syllable
    next_line_idx: int           # -1 if none
    in_gap: bool                 # between lines (instrumental / breath)


def _fill(values: list[float | None], fallback: float) -> list[float]:
    """Forward-fill None (backward-fill a leading run, `fallback` if nothing is known)."""
    first = next((v for v in values if v is not None), fallback)
    out, last = [], first
    for v in values:
        last = v if v is not None else last
        out.append(last)
    return out


class Timeline:
    def __init__(self, lines: list[Line]):
        self.lines = list(lines)
        starts: list[float | None] = []
        ends: list[float | None] = []
        for ln in self.lines:
            syl = ln.syllables()
            s0 = next((s.start for s in syl if _ok(s.start)), None)
            e0 = next((s.end for s in reversed(syl) if _ok(s.end)), None)
            starts.append(ln.start if _ok(ln.start) else s0)
            ends.append(ln.end if _ok(ln.end) else e0)
        n = len(self.lines)
        # untimed lines: zero length at the previous timed line's end (or the next line's start)
        prev_end: list[float | None] = []
        last = None
        for s, e in zip(starts, ends):
            prev_end.append(last)
            if s is not None or e is not None:
                last = e if e is not None else s
        nxt = _fill(starts[::-1], 0.0)[::-1]          # next known start at or after i
        fs, fe = [], []
        for i in range(n):
            s, e = starts[i], ends[i]
            if s is None and e is None:
                s = e = prev_end[i] if prev_end[i] is not None else nxt[i]
            elif s is None:
                s = e
            elif e is None:
                e = next((x for x in starts[i + 1:] if x is not None), s)
            fs.append(float(s))
            fe.append(float(e))
        for i in range(1, n):                          # non-decreasing starts, end >= start
            fs[i] = max(fs[i], fs[i - 1])
        self._starts = fs
        self._ends = [max(e, s) for s, e in zip(fs, fe)]
        self._syl: list[tuple[list[float], list[float]]] = []
        for i, ln in enumerate(self.lines):
            syl = ln.syllables()
            # boundaries on a "syllable axis" (start_k at k, end_k at k+1); missing ones are linearly
            # interpolated between the nearest known boundaries (line start / end as outer anchors)
            n_s = len(syl)
            pts: list[tuple[float, float | None]] = []
            for k, s in enumerate(syl):
                pts += [(k, s.start if _ok(s.start) else None), (k + 1, s.end if _ok(s.end) else None)]
            known = [(0.0, self._starts[i])] + [(a, v) for a, v in pts if v is not None] + \
                    [(float(n_s), self._ends[i])]
            vals = [v if v is not None else _interp(a, known) for a, v in pts]
            ss, se = vals[0::2], vals[1::2]
            for k in range(1, n_s):
                ss[k] = max(ss[k], ss[k - 1])
            self._syl.append((ss, [max(e, s) for s, e in zip(ss, se)]))

    def __len__(self) -> int:
        return len(self.lines)

    def line_start(self, idx: int) -> float:
        return self._starts[idx]

    def line_end(self, idx: int) -> float:
        return self._ends[idx]

    def locate(self, ref_time: float | None) -> Cursor:
        n = len(self._starts)
        t = -math.inf if ref_time is None or math.isnan(ref_time) else float(ref_time)
        li = bisect.bisect_right(self._starts, t) - 1
        if li < 0:
            return Cursor(-1, -1, 0.0, 0 if n else -1, True)
        nxt = li + 1 if li + 1 < n else -1
        if t >= self._ends[li]:
            return Cursor(li, -1, 0.0, nxt, True)
        ss, se = self._syl[li]
        si = bisect.bisect_right(ss, t) - 1
        if si < 0:
            return Cursor(li, -1, 0.0, nxt, False)
        dur = se[si] - ss[si]
        prog = 1.0 if dur <= 0 else min(1.0, max(0.0, (t - ss[si]) / dur))
        return Cursor(li, si, prog, nxt, False)


def _ok(v: float | None) -> bool:
    return v is not None and math.isfinite(v)


def _interp(a: float, known: list[tuple[float, float]]) -> float:
    """Linear interpolation at axis position `a` from (axis, value) pairs (nearest known value outside)."""
    prev = max((p for p in known if p[0] <= a), key=lambda p: p[0], default=None)
    nxt = min((p for p in known if p[0] >= a), key=lambda p: p[0], default=None)
    if prev is None or nxt is None:
        return (prev or nxt)[1]
    if nxt[0] == prev[0]:
        return prev[1]
    return prev[1] + (nxt[1] - prev[1]) * (a - prev[0]) / (nxt[0] - prev[0])
