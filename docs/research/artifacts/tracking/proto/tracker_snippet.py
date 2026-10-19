import numpy as np


def cosine_dist(R, x):            # R rows and x already L2-normalised per feature group
    return 1.0 - R @ x


class FrameSyncOLTW:
    """Arzt/Brazier-style frame-synchronous on-line time warping, fully vectorised.

    Per live frame t, over reference frames i in a window around the current position:
        D_t[i] = min(D_{t-1}[i-1] + wd*d[i],   # diagonal  (ref+1, live+1)
                     D_{t-1}[i]   + wh*d[i],   # hold      (ref+0, live+1)  -> fermata / slower
                     D_t[i-1]     + wv*d[i])   # ref-only  (ref+1, live+0)  -> faster
    The in-column dependency is solved with a min-plus prefix scan:
        B = min(diag, hold);  S = cumsum(wv*d);  D_t = S + cummin(B - S)
    (verified identical to the cell-by-cell loop).  With (wd, wh, wv) = (2, 1, 1) every path
    from the anchor has weight (i - i0) + (t - t0), so D / (|i-i0| + t-t0 + 1) is comparable.
    """
    DIAG, HOLD, VERT = 0, 1, 2

    def __init__(self, ref, fps=25, win_s=10.0, max_adv=3, w=(2.0, 1.0, 1.0), hist_s=8.0,
                 dist=cosine_dist):
        self.R = np.ascontiguousarray(ref, dtype=np.float32)   # [N, D]
        self.N, self.fps, self.dist = len(self.R), fps, dist
        self.win, self.max_adv = int(win_s * fps), max_adv
        self.wd, self.wh, self.wv = w
        self.K = int(hist_s * fps)                              # backpointer history (frames)
        self.bp = np.zeros((self.K, 2 * self.win + 1), np.int8)
        self.lo_hist = np.zeros(self.K, np.int64)
        self.n_hist = np.zeros(self.K, np.int64)
        self.anchor(0)

    def anchor(self, i0, prior=None):
        """Hard restart at ref frame i0, or soft restart with a prior cost vector [N]
        (e.g. prior = beta * |i - i_pred| / fps, inf outside a +-3 s region)."""
        self.D = np.full(self.N, np.inf) if prior is None else np.asarray(prior, float).copy()
        if prior is None:
            self.D[i0] = 0.0
        self.i0, self.t_rel, self.pos, self.first = int(i0), 0, int(i0), True

    def step(self, x):
        lo, hi = max(0, self.pos - self.win), min(self.N, self.pos + self.win + 1)
        d = self.dist(self.R[lo:hi], x)
        Dp = self.D[lo:hi]
        Dp_prev = np.concatenate(([self.D[lo - 1] if lo > 0 else np.inf], Dp[:-1]))
        if self.first:                       # anchor cells start the path
            diag, hold, self.first = np.full_like(d, np.inf, dtype=float), Dp + d, False
        else:
            diag, hold = Dp_prev + self.wd * d, Dp + self.wh * d
        B = np.minimum(diag, hold)
        S = np.cumsum(self.wv * d)
        Dn = S + np.minimum.accumulate(B - S)
        k = self.t_rel % self.K               # backpointers for the backward path
        self.bp[k, :hi - lo] = np.where(Dn < B - 1e-9, self.VERT,
                                        np.where(diag <= hold, self.DIAG, self.HOLD))
        self.lo_hist[k], self.n_hist[k] = lo, hi - lo
        self.D.fill(np.inf)
        self.D[lo:hi] = Dn
        idx = np.arange(lo, hi)
        norm = Dn / (np.abs(idx - self.i0) + self.t_rel + 1.0)
        j = int(np.argmin(norm))
        cand = lo + j
        self.pos = int(np.clip(cand, self.pos, self.pos + self.max_adv))   # Arzt step clamp
        self.t_rel += 1
        return self.pos, float(norm[j]), norm

    def backward_path(self, n_frames):
        """Backtrack from the current position over the last n live frames -> [(t_rel, i)]."""
        t, i, path = self.t_rel - 1, self.pos, []
        stop = max(0, self.t_rel - min(n_frames, self.K))
        while t >= stop:
            k = t % self.K
            j = i - self.lo_hist[k]
            if j < 0 or j >= self.n_hist[k]:
                break
            path.append((t, i))
            c = self.bp[k, j]
            if c == self.DIAG:
                i, t = i - 1, t - 1
            elif c == self.HOLD:
                t -= 1
            else:
                i -= 1
        return np.array(path[::-1])

    def tempo(self, look_s=6.0, skip_s=1.0):
        """Relative tempo rho = d(ref)/d(live) from the backward path, ignoring the last skip_s
        (Arzt & Widmer 2010 use onsets >= 1 s in the past)."""
        p = self.backward_path(int(look_s * self.fps))
        if len(p) < self.fps:
            return 1.0
        p = p[p[:, 0] <= p[-1, 0] - int(skip_s * self.fps)]
        if len(p) < self.fps // 2:
            return 1.0
        slope = np.polyfit(p[:, 0], p[:, 1], 1)[0]
        return float(np.clip(slope, 0.5, 2.0))


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(3)
    fps, N, dim = 25, 240 * 25, 24
    base = rng.standard_normal((N // 5 + 2, dim))
    R = np.repeat(base, 5, axis=0)[:N] + 0.3 * rng.standard_normal((N, dim))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    rho_true = 0.85                                        # live 15 % slower than reference
    gt = np.arange(0, N - 1, rho_true)
    L = R[np.round(gt).astype(int)] + 0.6 * rng.standard_normal((len(gt), dim)) / np.sqrt(dim)
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    trk = FrameSyncOLTW(R, fps=fps)
    est, rhos, tt = [], [], []
    for t, x in enumerate(L):
        t0 = time.perf_counter()
        p, c, _ = trk.step(x.astype(np.float32))
        tt.append(time.perf_counter() - t0)
        est.append(p)
        if t % 50 == 49:
            rhos.append(trk.tempo())
    err = np.abs(np.array(est) - gt) / fps * 1000
    print("PCO<=200ms %.1f%%  median %.1f ms  rho est median %.3f (true %.2f)  step p50 %.0f us" % (
        100 * np.mean(err <= 200), np.median(err), np.median(rhos[2:]), rho_true, 1e6 * np.median(tt)))
