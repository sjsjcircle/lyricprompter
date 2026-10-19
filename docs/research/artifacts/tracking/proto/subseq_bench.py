"""Benchmark: subsequence DTW of the last ~9 s of live features against the whole reference
(global re-localisation layer). Step set {(1,1),(1,2),(2,1)} (Mueller), open begin/end on ref."""
import time
import numpy as np
import numba as nb


@nb.njit(cache=True)
def subseq_dtw_end_costs(C):
    """C: [M (query), N (ref)] local costs. Returns normalised accumulated cost of the best
    path ending at each ref frame (open begin in ref)."""
    M, N = C.shape
    D = np.full((M + 2, N + 2), np.inf)
    L = np.zeros((M + 2, N + 2))
    for n in range(N):
        D[2, n + 2] = C[0, n]
        L[2, n + 2] = 1.0
    for m in range(1, M):
        for n in range(N):
            c = C[m, n]
            a = D[m + 1, n + 1] + c          # (1,1)
            b = D[m, n + 1] + 2 * c          # (2,1): query advances 2 (weight 2 -> also covers skipped)
            e = D[m + 1, n] + 2 * c          # (1,2): ref advances 2
            if a <= b and a <= e:
                D[m + 2, n + 2] = a; L[m + 2, n + 2] = L[m + 1, n + 1] + 1
            elif b <= e:
                D[m + 2, n + 2] = b; L[m + 2, n + 2] = L[m, n + 1] + 2
            else:
                D[m + 2, n + 2] = e; L[m + 2, n + 2] = L[m + 1, n] + 2
    return D[M + 1, 2:] / np.maximum(L[M + 1, 2:], 1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for fps, label in ((25, "fine 25 fps"), (1 / 0.3, "coarse 300 ms hop")):
        N = int(240 * fps); M = int(9 * fps)
        R = rng.standard_normal((N, 24)); R /= np.linalg.norm(R, axis=1, keepdims=True)
        start = N // 2
        Q = R[start:start + M] + 0.3 * rng.standard_normal((M, 24)); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        C = 1 - Q @ R.T
        subseq_dtw_end_costs(C)
        t0 = time.perf_counter(); costs = subseq_dtw_end_costs(C); dt = time.perf_counter() - t0
        print(f"{label}: query {M} x ref {N} = {M*N/1e6:.2f} M cells, {dt*1e3:.2f} ms, "
              f"best end {int(np.argmin(costs))} (true {start+M-1})")
