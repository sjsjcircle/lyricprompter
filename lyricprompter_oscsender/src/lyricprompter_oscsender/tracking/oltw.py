"""Frame-synchronous on-line time warping (Arzt / Brazier style), vectorised per live frame.

For every live frame t the column of accumulated costs over a reference window [lo, hi) is

    D_t[i] = min(D_{t-1}[i-1] + wd*d[i],     # diagonal  (ref +1, live +1)
                 D_{t-1}[i]   + wh*d[i],     # hold      (ref +0, live +1): slower / fermata
                 D_t[i-1]     + wv*d[i])     # ref-only  (ref +1, live +0): faster

The in-column dependency is a min-plus prefix scan: with B = min(diag, hold) and S = cumsum(wv*d),
D_t = S + cummin(B - S); the argmin of the running minimum gives the start of each ref-only chain,
which is how path starts (and back-pointers) are propagated without a Python loop.

With the symmetric weights (2, 1, 1) every monotone path from start cell s (at the last anchor) to
(i, t) has weight (i - s) + t, so costs are normalised by (i - s + t + 1) and are comparable across
cells, also after a soft (prior-cost) restart where different cells start different paths. Other
weights keep that normalisation and act as a tempo prior: wd < 2 favours advancing at the reference
tempo, wh > 1 penalises holding (the tuned trackers use (1.8, 1.2, 1.0)).
The published position is the argmin of the normalised cost, clamped to move monotonically by at
most `max_adv` reference frames per live frame. For the first `free` steps after an anchor only the
forward limit applies, so it may follow the argmin backwards (settling after a soft re-anchor whose
prior also reaches back; no path can start before the prior's first finite cell).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import FPS

DIAG, HOLD, VERT = 0, 1, 2


@dataclass
class StepResult:
    pos: int            # published (clamped, monotonic) reference frame
    best: int           # unclamped argmin of the normalised cost
    cost: float         # normalised accumulated cost at `best`
    d_pos: float        # local distance of this live frame at `pos`
    d_mean: float       # mean local distance over the window (an "off-path" baseline)
    free: bool = False  # settling step after an anchor: pos may move backwards


class FrameSyncOLTW:
    def __init__(self, ref: np.ndarray, *, fps: float = FPS, win_s: float = 10.0, max_adv: int = 3,
                 weights: tuple[float, float, float] = (2.0, 1.0, 1.0), hist_s: float = 8.0):
        self.fps = float(fps)
        self.R = np.ascontiguousarray(ref, dtype=np.float32)
        if self.R.ndim != 2 or len(self.R) == 0:
            raise ValueError("reference features must be a non-empty [N, D] array")
        self.N = len(self.R)
        self.win = max(1, round(win_s * self.fps))
        self.max_adv = int(max_adv)
        self.wd, self.wh, self.wv = (float(w) for w in weights)
        self.K = max(2, round(hist_s * self.fps))           # back-pointer history (live frames)
        self._bp = np.zeros((self.K, 2 * self.win + 1), np.int8)
        self._lo_hist = np.zeros(self.K, np.int64)
        self._n_hist = np.zeros(self.K, np.int64)
        self._ar = np.arange(2 * self.win + 1, dtype=np.int64)
        self.D = np.full(self.N, np.inf)
        self.src = np.zeros(self.N, np.int64)
        self.anchor(0)

    def set_reference(self, ref: np.ndarray) -> None:
        """Swap reference features in place (same shape), e.g. after a chroma roll; keeps the DP."""
        ref = np.ascontiguousarray(ref, dtype=np.float32)
        if ref.shape != self.R.shape:
            raise ValueError(f"reference shape {ref.shape} != {self.R.shape}")
        self.R = ref

    def anchor(self, i0: int, prior: np.ndarray | None = None, *, free: int = 0) -> None:
        """Restart the DP at reference frame i0 (hard, Brazier-style), or softly from a prior cost
        vector over all N reference frames (np.inf = not allowed). The published position becomes i0;
        during the next `free` steps it may move backwards (still at most max_adv forwards)."""
        i0 = int(np.clip(i0, 0, self.N - 1))
        if prior is None:
            self.D.fill(np.inf)
            self.D[i0] = 0.0
        else:
            p = np.asarray(prior, dtype=np.float64)
            if p.shape != (self.N,):
                raise ValueError(f"prior must have shape ({self.N},)")
            self.D[:] = p
            if not np.isfinite(self.D).any():
                self.D[i0] = 0.0
        self.src[:] = np.arange(self.N)
        self.i0 = self.pos = self.best = i0
        self.t = 0                        # live frames since the anchor
        self.free = max(0, int(free))     # remaining steps without the backward clamp
        self._lo, self._hi = 0, self.N    # cells that may hold finite costs

    def step(self, x: np.ndarray) -> StepResult:
        pos, win = self.pos, self.win
        lo, hi = max(0, pos - win), min(self.N, pos + win + 1)
        n = hi - lo
        ar = self._ar[:n]
        d = 1.0 - (self.R[lo:hi] @ np.asarray(x, dtype=np.float32)).astype(np.float64)
        np.clip(d, 0.0, 2.0, out=d)
        d[~np.isfinite(d)] = 1.0
        Dp, Sp = self.D[lo:hi], self.src[lo:hi]
        hold = Dp + self.wh * d
        if self.t == 0:                   # first frame after an anchor: start cells consume it
            use_diag = np.zeros(n, bool)
            B, Bsrc = hold, Sp
        else:
            Dsh = np.empty(n)
            Ssh = np.empty(n, np.int64)
            Dsh[0] = self.D[lo - 1] if lo > 0 else np.inf
            Ssh[0] = self.src[lo - 1] if lo > 0 else 0
            Dsh[1:], Ssh[1:] = Dp[:-1], Sp[:-1]
            diag = Dsh + self.wd * d
            use_diag = diag <= hold
            B = np.where(use_diag, diag, hold)
            Bsrc = np.where(use_diag, Ssh, Sp)
        S = np.cumsum(self.wv * d)
        Z = B - S
        M = np.minimum.accumulate(Z)
        Dn = S + M
        with np.errstate(invalid="ignore"):
            rec = Z <= M                  # running-minimum records = starts of ref-only chains
        k = np.maximum.accumulate(np.where(rec, ar, 0))
        srcn = Bsrc[k]
        slot = self.t % self.K
        self._bp[slot, :n] = np.where(k < ar, VERT, np.where(use_diag, DIAG, HOLD))
        self._lo_hist[slot], self._n_hist[slot] = lo, n
        # write back: only [lo, hi) stays finite
        self.D[self._lo:self._hi] = np.inf
        self.D[lo:hi] = Dn
        self.src[lo:hi] = srcn
        self._lo, self._hi = lo, hi
        fin = np.isfinite(Dn)
        length = (lo + ar - srcn) + self.t + 1.0
        norm = np.where(fin, Dn / np.where(fin, length, 1.0), np.inf)
        j = int(np.argmin(norm))
        self.best = lo + j
        free = self.free > 0
        if free:
            self.free -= 1
        self.pos = int(min(self.best, pos + self.max_adv, self.N - 1)) if free else \
            int(np.clip(self.best, pos, min(pos + self.max_adv, self.N - 1)))
        self.t += 1
        return StepResult(self.pos, self.best, float(norm[j]), float(d[self.pos - lo]), float(d.mean()), free)

    def backward_path(self, n_frames: int) -> np.ndarray:
        """Back-track from the published position over at most the last n live frames since the
        anchor -> int array [(t, i)] in forward order (t counts live frames since the anchor)."""
        t, i, path = self.t - 1, self.pos, []
        stop = max(0, self.t - min(int(n_frames), self.K))
        while t >= stop:
            slot = t % self.K
            j = i - self._lo_hist[slot]
            if j < 0 or j >= self._n_hist[slot]:
                break
            path.append((t, i))
            c = self._bp[slot, j]
            if c == DIAG:
                i, t = i - 1, t - 1
            elif c == HOLD:
                t -= 1
            else:
                i -= 1
        return np.array(path[::-1], dtype=np.int64).reshape(-1, 2)

    def tempo(self, look_s: float = 6.0, skip_s: float = 1.0) -> float | None:
        """Relative tempo rho = d(ref)/d(live) from the slope of the backward path over the last
        look_s seconds, ignoring the most recent skip_s (still unsettled). None if too little path."""
        p = self.backward_path(int(look_s * self.fps))
        if len(p) < self.fps:
            return None
        p = p[p[:, 0] <= p[-1, 0] - int(skip_s * self.fps)]
        if len(p) < self.fps // 2 or np.ptp(p[:, 0]) < self.fps // 2:
            return None
        return float(np.polyfit(p[:, 0].astype(float), p[:, 1].astype(float), 1)[0])
