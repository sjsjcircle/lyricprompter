"""Independent check of the min-plus prefix-scan column update for frame-synchronous OLTW.
Recursion: D_t[i] = min(Dp[i-1] + wd*d[i], Dp[i] + wh*d[i], D_t[i-1] + wv*d[i]).
"""
import time
import numpy as np

rng = np.random.default_rng(1)


def loop_col(Dp, d, wd, wh, wv, Dp_before=np.inf):
    n = len(d)
    D = np.empty(n)
    for i in range(n):
        diag = (Dp[i - 1] if i > 0 else Dp_before) + wd * d[i]
        hold = Dp[i] + wh * d[i]
        vert = (D[i - 1] + wv * d[i]) if i > 0 else np.inf
        D[i] = min(diag, hold, vert)
    return D


def vec_col(Dp, d, wd, wh, wv, Dp_before=np.inf):
    Dprev_shift = np.concatenate(([Dp_before], Dp[:-1]))
    B = np.minimum(Dprev_shift + wd * d, Dp + wh * d)
    S = np.cumsum(wv * d)
    return S + np.minimum.accumulate(B - S)


worst = 0.0
for trial in range(1000):
    n = int(rng.integers(2, 400))
    d = rng.random(n) * rng.choice([0.01, 1, 5])
    Dp = rng.random(n) * rng.uniform(0, 300)
    # inject infinities (outside-window / restart) in random places
    mask = rng.random(n) < rng.uniform(0, 0.7)
    Dp[mask] = np.inf
    wd, wh, wv = rng.choice([(2., 1., 1.), (1., 1., 1.), (2., 1., 3.)])
    before = np.inf if rng.random() < 0.5 else float(rng.random() * 100)
    a = loop_col(Dp, d, wd, wh, wv, before)
    b = vec_col(Dp, d, wd, wh, wv, before)
    fin = np.isfinite(a)
    assert np.array_equal(fin, np.isfinite(b)), trial
    if fin.any():
        worst = max(worst, float(np.max(np.abs(a[fin] - b[fin]) / (1 + np.abs(a[fin])))))
print("1000 random trials incl. inf + 3 weight sets OK; max rel err", worst)

# speed (numpy) for 1000-cell column and 501-cell (+-10 s @ 25 fps) column
for n in (501, 1000):
    d = rng.random(n); Dp = rng.random(n) * 100
    for _ in range(200):
        vec_col(Dp, d, 2., 1., 1.)
    t = time.perf_counter(); K = 20000
    for _ in range(K):
        vec_col(Dp, d, 2., 1., 1.)
    print(f"numpy vec column n={n}: {(time.perf_counter() - t) / K * 1e6:.2f} us")

# full step incl. cosine distances against a 501x(12+5) window
R = rng.random((6000, 17)).astype(np.float32); R /= np.linalg.norm(R, axis=1, keepdims=True)
x = rng.random(17).astype(np.float32); x /= np.linalg.norm(x)
t = time.perf_counter(); K = 20000
for _ in range(K):
    lo, hi = 2000, 2501
    dd = 1.0 - R[lo:hi] @ x
    Dn = vec_col(Dp[:501], dd.astype(np.float64), 2., 1., 1.)
    norm = Dn / (np.abs(np.arange(lo, hi) - 2250) + 10 + 1.0)
    j = int(np.argmin(norm))
print(f"dist+column+argmin step (501 cells): {(time.perf_counter() - t) / K * 1e6:.2f} us")
