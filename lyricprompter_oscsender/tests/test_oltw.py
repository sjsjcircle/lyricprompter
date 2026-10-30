from __future__ import annotations

import numpy as np
import pytest

from lyricprompter_oscsender.tracking.oltw import DIAG, HOLD, VERT, FrameSyncOLTW


class LoopOLTW:
    """Cell-by-cell reference implementation of the same recursion / normalisation / clamp."""

    def __init__(self, R, win, max_adv, w):
        self.R, self.N, self.win, self.max_adv = R.astype(np.float32), len(R), win, max_adv
        self.wd, self.wh, self.wv = w

    def anchor(self, i0, prior=None):
        self.D = np.full(self.N, np.inf)
        if prior is None:
            self.D[i0] = 0.0
        else:
            self.D[:] = prior
        self.src = np.arange(self.N)
        self.pos, self.t, self.i0 = i0, 0, i0

    def step(self, x):
        lo, hi = max(0, self.pos - self.win), min(self.N, self.pos + self.win + 1)
        newD, newsrc, bp = np.full(self.N, np.inf), np.zeros(self.N, int), {}
        for i in range(lo, hi):
            d = float(np.clip(1.0 - float(self.R[i] @ x.astype(np.float32)), 0.0, 2.0))
            hold = (self.D[i] + self.wh * d, self.src[i], HOLD)
            if self.t == 0:
                best = hold
            else:
                diag = ((self.D[i - 1] if i > 0 else np.inf) + self.wd * d, self.src[i - 1] if i > 0 else 0, DIAG)
                best = diag if diag[0] <= hold[0] else hold
            if i > lo:
                vert = (newD[i - 1] + self.wv * d, newsrc[i - 1], VERT)
                if np.isfinite(vert[0]) and (not np.isfinite(best[0])
                                             or vert[0] < best[0] - 1e-12 * max(1.0, abs(vert[0]))):
                    best = vert
            newD[i], newsrc[i], bp[i] = best
        norm = np.full(self.N, np.inf)
        for i in range(lo, hi):
            if np.isfinite(newD[i]):
                norm[i] = newD[i] / ((i - newsrc[i]) + self.t + 1.0)
        best_i = int(np.argmin(norm))
        self.pos = int(np.clip(best_i, self.pos, min(self.pos + self.max_adv, self.N - 1)))
        self.D, self.src = newD, newsrc
        self.t += 1
        return self.pos, bp


def _rand_ref(rng, n, dim):
    R = rng.standard_normal((n, dim))
    return (R / np.linalg.norm(R, axis=1, keepdims=True)).astype(np.float32)


@pytest.mark.parametrize("seed", range(6))
def test_vectorised_equals_loop(seed):
    rng = np.random.default_rng(seed)
    N, dim, win = 220, 8, 15
    R = _rand_ref(rng, N, dim)
    w = [(2.0, 1.0, 1.0), (1.0, 1.0, 1.0), (2.0, 1.0, 3.0)][seed % 3]
    vec = FrameSyncOLTW(R, fps=25, win_s=win / 25, max_adv=3, weights=w, hist_s=4.0)
    ref = LoopOLTW(R, win=win, max_adv=3, w=w)
    i0 = int(rng.integers(0, 40))
    if seed % 2:
        prior = np.full(N, np.inf)
        prior[i0:i0 + 12] = 0.05 * np.abs(np.arange(12) - 4)
        vec.anchor(i0, prior)
        ref.anchor(i0, prior)
    else:
        vec.anchor(i0)
        ref.anchor(i0)
    for t in range(120):
        # live drifts through the reference with noise so the window moves
        x = R[min(N - 1, i0 + int(t * 1.3))] + 0.3 * rng.standard_normal(dim)
        x = (x / np.linalg.norm(x)).astype(np.float32)
        r = vec.step(x)
        p, bp = ref.step(x)
        assert r.pos == p
        fin = np.isfinite(ref.D)
        assert np.array_equal(fin, np.isfinite(vec.D))
        np.testing.assert_allclose(vec.D[fin], ref.D[fin], rtol=1e-9, atol=1e-9)
        assert np.array_equal(vec.src[fin], ref.src[fin])
        slot = (vec.t - 1) % vec.K
        lo = vec._lo_hist[slot]
        for i, c in bp.items():
            assert vec._bp[slot, i - lo] == c


def test_hard_anchor_first_step_starts_at_anchor():
    rng = np.random.default_rng(1)
    R = _rand_ref(rng, 100, 6)
    o = FrameSyncOLTW(R, fps=25, win_s=1.0)
    o.anchor(40)
    o.step(R[40])
    assert np.isfinite(o.D[40]) and not np.isfinite(o.D[:40]).any()
    assert o.pos == 40


def test_monotone_and_clamped():
    rng = np.random.default_rng(2)
    R = _rand_ref(rng, 400, 12)
    o = FrameSyncOLTW(R, fps=25, win_s=4.0, max_adv=3)
    prev = o.pos
    for _ in range(200):
        r = o.step(_rand_ref(rng, 1, 12)[0])
        assert prev <= r.pos <= prev + 3
        prev = r.pos


def _warped(rng, N=3000, dim=16, rho=0.85, noise=0.4):
    base = rng.standard_normal((N // 5 + 2, dim))
    R = np.repeat(base, 5, axis=0)[:N] + 0.3 * rng.standard_normal((N, dim))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    gt = np.arange(0, N - 1, rho)
    L = R[np.round(gt).astype(int)] + noise * rng.standard_normal((len(gt), dim)) / np.sqrt(dim)
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    return R.astype(np.float32), L.astype(np.float32), gt


@pytest.mark.parametrize("rho", [0.8, 1.25])
def test_tracks_constant_tempo_and_estimates_it(rho):
    rng = np.random.default_rng(3)
    R, L, gt = _warped(rng, rho=rho)
    o = FrameSyncOLTW(R, fps=25)
    est, rhos = [], []
    for t, x in enumerate(L):
        est.append(o.step(x).pos)
        if t % 50 == 49 and t > 300:
            rhos.append(o.tempo())
    err_ms = np.abs(np.array(est) - gt) / 25 * 1000
    assert np.mean(err_ms <= 200) > 0.95
    assert abs(np.median([r for r in rhos if r is not None]) - rho) < 0.03


def test_backward_path_is_monotone_and_ends_at_pos():
    rng = np.random.default_rng(4)
    R, L, _ = _warped(rng, N=600)
    o = FrameSyncOLTW(R, fps=25, hist_s=4.0)
    for x in L[:300]:
        o.step(x)
    p = o.backward_path(1000)
    assert len(p) > 0 and p[-1, 0] == o.t - 1 and p[-1, 1] == o.pos
    assert (np.diff(p[:, 0]) >= 0).all() and (np.diff(p[:, 1]) >= 0).all()
    assert p[0, 0] >= o.t - o.K


def test_soft_prior_restart_prefers_matching_region():
    rng = np.random.default_rng(5)
    R, _, _ = _warped(rng, N=1000, rho=1.0)
    o = FrameSyncOLTW(R, fps=25)
    true = 520
    prior = np.full(len(R), np.inf)
    prior[450:600] = 0.02 * np.abs(np.arange(450, 600) - 500) / 25
    o.anchor(450, prior)
    for k in range(40):
        o.step(R[true + k])
    assert abs(o.pos - (true + 39)) <= 2


def test_set_reference_shape_check():
    o = FrameSyncOLTW(np.eye(5, dtype=np.float32), fps=25, win_s=0.2)
    with pytest.raises(ValueError):
        o.set_reference(np.eye(4, dtype=np.float32))


def test_free_steps_after_anchor_may_move_back_within_the_prior():
    rng = np.random.default_rng(5)
    R, L, _ = _warped(rng, N=600, rho=1.0, noise=0.2)
    prior = np.full(len(R), np.inf)
    prior[150:260] = 0.3 * np.abs(np.arange(150, 260) - 200) / 25
    live = L[160:200]                                       # the performer is at 160, behind the anchor
    o = FrameSyncOLTW(R, fps=25, win_s=4.0, max_adv=3)
    o.anchor(200, prior, free=10)
    res = [o.step(x) for x in live]
    pos = np.array([r.pos for r in res])
    assert [r.free for r in res] == [True] * 10 + [False] * 30
    assert pos.min() >= 150                                 # no path starts before the prior's support
    assert pos[9] < 200 and abs(pos[-1] - 199) <= 2         # moved back and then follows
    assert (np.diff(pos) <= 3).all()                        # forward limit also while free
    assert (np.diff(pos[10:]) >= 0).all()                   # monotone again after the free steps
    clamped = FrameSyncOLTW(R, fps=25, win_s=4.0, max_adv=3)
    clamped.anchor(200, prior)
    assert min(clamped.step(x).pos for x in live) >= 200    # default: never below the anchor
