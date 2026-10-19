"""Prototype: frame-synchronous OLTW (Arzt/Brazier-style column update) vectorized with a
cumulative-min trick, plus a numba reference loop to verify equivalence and benchmark.

Recursion for each new live frame t over reference indices i in window [lo, hi):
    D_t[i] = min( D_{t-1}[i-1] + wd*d[i],   # diagonal  (ref +1, live +1)
                  D_{t-1}[i]   + wh*d[i],   # hold      (ref +0, live +1)
                  D_t[i-1]     + wv*d[i] )  # ref-only  (ref +1, live +0)
Let B[i] = min(first two terms) and S = cumsum(wv*d). Then
    D_t[i] = S[i] + cummin_k<=i (B[k] - S[k])
which is pure numpy (np.minimum.accumulate), no Python loop.
With (wd, wh, wv) = (2, 1, 1) (Sakoe-Chiba symmetric) every monotone path from the anchor
(i0, t0) to (i, t) has total weight (i - i0) + (t - t0), so normalized cost = D / (i-i0 + t-t0 + 1)
is comparable across i (Dixon 2005 / Arzt normalisation).
"""
import time
import numpy as np
import numba as nb


class VecOLTW:
    def __init__(self, ref_feats, win=250, max_adv=3, wd=2.0, wh=1.0, wv=1.0):
        self.R = np.ascontiguousarray(ref_feats, dtype=np.float32)  # [N, D], rows L2-normalised
        self.N = len(self.R)
        self.win, self.max_adv = win, max_adv
        self.wd, self.wh, self.wv = wd, wh, wv
        self.anchor(0, 0)

    def anchor(self, i0, t0, spread=0):
        """(Re)start the DP at ref frame i0 / live frame t0 (Brazier-style +inf everywhere, 0 at anchor)."""
        self.D = np.full(self.N, np.inf)
        lo, hi = max(0, i0 - spread), min(self.N, i0 + spread + 1)
        self.D[lo:hi] = 0.0
        self.i0, self.t0, self.t, self.pos = i0, t0, t0, i0
        self.first = True

    def step(self, x):
        lo = max(0, self.pos - self.win)
        hi = min(self.N, self.pos + self.win)
        d = 1.0 - self.R[lo:hi] @ x                        # cosine distance (x L2-normalised)
        Dp = self.D[lo:hi]
        Dp_shift = np.empty_like(Dp)
        Dp_shift[0] = self.D[lo - 1] if lo > 0 else np.inf
        Dp_shift[1:] = Dp[:-1]
        if self.first:   # first live frame: anchor cells start the path (weight 1)
            B = np.where(np.isfinite(Dp), Dp + d, np.inf)
            self.first = False
        else:
            B = np.minimum(Dp_shift + self.wd * d, Dp + self.wh * d)
        S = np.cumsum(self.wv * d)
        Dn = S + np.minimum.accumulate(B - S)
        newD = np.full(self.N, np.inf)
        newD[lo:hi] = Dn
        self.D = newD
        idx = np.arange(lo, hi)
        norm = Dn / ((idx - self.i0) + (self.t - self.t0) + 1.0)
        norm[idx < self.i0] = np.inf
        j = int(np.argmin(norm))
        cand = lo + j
        self.pos = int(np.clip(cand, self.pos, self.pos + self.max_adv))  # Arzt step-size clamp
        self.t += 1
        return self.pos, norm[j]


@nb.njit(cache=True)
def column_loop(Dprev, d, wd, wh, wv, first):
    n = d.shape[0]
    out = np.empty(n)
    for i in range(n):
        if first:
            b = Dprev[i + 1] + d[i]
        else:
            b = min(Dprev[i] + wd * d[i], Dprev[i + 1] + wh * d[i])
        if i > 0:
            b = min(b, out[i - 1] + wv * d[i])
        out[i] = b
    return out


def verify_equivalence(n=600, trials=200, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        d = rng.random(n)
        Dprev = rng.random(n + 1) * 50
        Dprev[rng.random(n + 1) < 0.1] = np.inf
        B = np.minimum(Dprev[:-1] + 2 * d, Dprev[1:] + 1 * d)
        S = np.cumsum(d)
        vec = S + np.minimum.accumulate(B - S)
        ref = column_loop(Dprev, d, 2.0, 1.0, 1.0, False)
        fin = np.isfinite(ref)
        assert np.array_equal(fin, np.isfinite(vec))
        assert np.allclose(vec[fin], ref[fin], rtol=1e-9, atol=1e-9)
    return True


def synth_test(seed=1, fps=25, dur=240, dim=24):
    """Random smooth 'reference', warped by a piecewise tempo curve + noise -> live. Track it."""
    rng = np.random.default_rng(seed)
    N = dur * fps
    base = rng.standard_normal((N // 5 + 2, dim))
    R = np.repeat(base, 5, axis=0)[:N] + 0.3 * rng.standard_normal((N, dim))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    # piecewise tempo ratio in [0.8, 1.25] incl. a 2 s fermata (ratio 0.3)
    ratios, t_ref, gt = [], 0.0, []
    while t_ref < N - 2:
        seg = rng.integers(2 * fps, 8 * fps)
        r = rng.uniform(0.8, 1.25)
        if rng.random() < 0.1:
            r, seg = 0.3, 2 * fps
        for _ in range(seg):
            gt.append(t_ref)
            t_ref += r
            if t_ref >= N - 1:
                break
    gt = np.array(gt)
    L = R[np.round(gt).astype(int)] + 0.6 * rng.standard_normal((len(gt), dim)) / np.sqrt(dim)
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    trk = VecOLTW(R, win=10 * fps, max_adv=3)
    est, tt = [], []
    for x in L:
        t0 = time.perf_counter()
        p, _ = trk.step(x.astype(np.float32))
        tt.append(time.perf_counter() - t0)
        est.append(p)
    err_ms = np.abs(np.array(est) - gt) / fps * 1000
    tol = {f"<= {t}ms": round(float(np.mean(err_ms <= t)) * 100, 1) for t in (200, 300, 500, 1000)}
    return tol, round(float(np.median(err_ms)), 1), round(1e6 * float(np.median(tt)), 1), round(1e6 * float(np.percentile(tt, 99)), 1)


if __name__ == "__main__":
    print("equivalence:", verify_equivalence())
    print("synthetic tracking (PCO, median err ms, step us p50, p99):", synth_test())
    # micro-benchmark: numba loop vs numpy cummin for window 2*10s*50fps = 1000 cells
    d = np.random.rand(1000); Dprev = np.random.rand(1001) * 100
    column_loop(Dprev, d, 2.0, 1.0, 1.0, False)
    t0 = time.perf_counter()
    for _ in range(10000):
        column_loop(Dprev, d, 2.0, 1.0, 1.0, False)
    t_nb = (time.perf_counter() - t0) / 10000
    t0 = time.perf_counter()
    for _ in range(10000):
        B = np.minimum(Dprev[:-1] + 2 * d, Dprev[1:] + d); S = np.cumsum(d)
        _ = S + np.minimum.accumulate(B - S)
    t_np = (time.perf_counter() - t0) / 10000
    print(f"W=1000 column update: numba {t_nb*1e6:.1f} us, numpy-cummin {t_np*1e6:.1f} us")
    # pure-python double-loop cost estimate (reference repo style) for 4 x 125 cells
    t0 = time.perf_counter()
    for _ in range(200):
        acc = np.zeros((125, 125))
        for i in range(4):
            for j in range(125):
                c = [acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1] * 0.98]
                k = int(np.argmin(c)); acc[i, j] = 0.5 + c[k]
    print(f"python double loop 4x125 cells: {(time.perf_counter()-t0)/200*1e3:.2f} ms per update")
