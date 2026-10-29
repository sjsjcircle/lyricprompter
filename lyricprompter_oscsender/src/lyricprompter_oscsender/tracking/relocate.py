"""Coarse global re-localisation (after Arzt & Widmer's rough position estimator).

Reference and live features are pooled with a Hann window (`win_s`, hop `hop_s`) and L2-normalised.
A subsequence DTW of the recent live query against the whole pooled reference (steps (1,1) w=1,
(2,1) and (1,2) w=2, i.e. local tempo 0.5..2) gives a length-normalised cost for every possible end
position; its local minima are candidate positions. Candidates are verified by the caller with a
shadow FrameSyncOLTW before switching.

Pooling is linear per feature column and the L2 norm is invariant under a permutation of columns, so
the pooled reference of a chroma-rolled reference is the rolled pooled reference; `end_costs_keys`
uses that to search under several keys at once (rolling the short query instead of the reference).
"""
from __future__ import annotations

import numpy as np


def pool_features(F: np.ndarray, hop: int, win: int) -> np.ndarray:
    """Hann-weighted sums of `win` frames every `hop` frames, L2-normalised -> [(n - win)//hop + 1, D].
    Coarse frame j covers fine frames [j*hop, j*hop + win)."""
    F = np.asarray(F, dtype=np.float32)
    n = len(F)
    if n < win:
        return np.zeros((0, F.shape[1]), np.float32)
    w = np.hanning(win + 2)[1:-1].astype(np.float32)
    starts = np.arange(0, n - win + 1, hop)
    view = np.lib.stride_tricks.sliding_window_view(F, win, axis=0)[starts]     # [m, D, win]
    out = view @ w
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def subsequence_end_costs(C: np.ndarray) -> np.ndarray:
    """C: [..., M query, N ref] local costs -> normalised cost of the best path ending at (M-1, n), any
    start, per leading index (a batch of cost matrices is processed in one pass)."""
    C = np.asarray(C)
    M, N = C.shape[-2:]
    batch = C.shape[:-2]
    inf = np.inf
    Dm2 = np.full((*batch, N), inf)
    Lm2 = np.ones((*batch, N))
    Dm1 = C[..., 0, :].astype(np.float64)          # free start anywhere
    Lm1 = np.ones((*batch, N))
    for m in range(1, M):
        c = C[..., m, :].astype(np.float64)
        a = np.full((*batch, N), inf)
        la = np.ones((*batch, N))
        a[..., 1:] = Dm1[..., :-1] + c[..., 1:]                  # (1,1)
        la[..., 1:] = Lm1[..., :-1] + 1
        b = np.full((*batch, N), inf)
        lb = np.ones((*batch, N))
        b[..., 1:] = Dm2[..., :-1] + 2 * c[..., 1:]              # query +2, ref +1
        lb[..., 1:] = Lm2[..., :-1] + 2
        e = np.full((*batch, N), inf)
        le = np.ones((*batch, N))
        e[..., 2:] = Dm1[..., :-2] + 2 * c[..., 2:]              # query +1, ref +2
        le[..., 2:] = Lm1[..., :-2] + 2
        stack = np.stack([a, b, e])
        k = np.argmin(stack, axis=0)
        D = np.choose(k, stack)
        L = np.choose(k, [la, lb, le])
        Dm2, Lm2, Dm1, Lm1 = Dm1, Lm1, D, L
    return Dm1 / Lm1


class Relocator:
    def __init__(self, ref_feats: np.ndarray, fps: float, *, hop_s: float = 0.32, win_s: float = 0.64):
        self.hop = max(1, round(hop_s * fps))
        self.win = max(self.hop, round(win_s * fps))
        self.fps = float(fps)
        self.set_reference(ref_feats)

    def set_reference(self, ref_feats: np.ndarray) -> None:
        self.C = pool_features(ref_feats, self.hop, self.win)

    def end_frame(self, j: int | np.ndarray) -> int | np.ndarray:
        """Fine reference frame at the end of coarse frame j."""
        return j * self.hop + self.win - 1

    def _pool_query(self, query: np.ndarray) -> np.ndarray:
        n = len(query)
        return pool_features(query[(n - self.win) % self.hop:], self.hop, self.win)  # align the end

    def end_costs(self, query: np.ndarray) -> np.ndarray:
        """Normalised cost per coarse reference end index for a fine-frame live query [M, D]."""
        return self.end_costs_keys(query, None, (0,))[0]

    def end_costs_keys(self, query: np.ndarray, chroma: slice | None, shifts) -> np.ndarray:
        """end_costs under the reference chroma additionally rolled by each of `shifts` semitones
        -> [len(shifts), coarse frames] (shift 0 = the reference as set)."""
        shifts = [int(s) for s in shifts]
        q = self._pool_query(query)
        if len(q) < 2 or len(self.C) < 3:
            return np.full((len(shifts), len(self.C)), np.inf)
        Q = np.repeat(q[None], len(shifts), axis=0)
        for i, s in enumerate(shifts):
            if s % 12:
                if chroma is None:
                    raise ValueError("end_costs_keys: a key shift needs the chroma columns")
                Q[i][:, chroma] = np.roll(q[:, chroma], -s, axis=1)   # q . roll(C, s) == roll(q, -s) . C
        return subsequence_end_costs(np.clip(1.0 - Q @ self.C.T, 0.0, 2.0))

    def candidates_from_costs(self, cost: np.ndarray, k: int = 3, sep_s: float = 4.0) -> list[tuple[int, float]]:
        """Up to k (fine end frame, cost) local minima of an end_costs row, best first, sep_s apart."""
        out: list[tuple[int, float]] = []
        sep = max(1, round(sep_s * self.fps / self.hop))
        c = np.array(cost, dtype=np.float64)
        for _ in range(k):
            j = int(np.argmin(c))
            if not np.isfinite(c[j]):
                break
            out.append((int(self.end_frame(j)), float(cost[j])))
            c[max(0, j - sep):j + sep + 1] = np.inf
        return out

    def candidates(self, query: np.ndarray, k: int = 3, sep_s: float = 4.0) -> list[tuple[int, float]]:
        """Up to k (fine end frame, cost) local minima, best first, at least sep_s apart."""
        return self.candidates_from_costs(self.end_costs(query), k, sep_s)
