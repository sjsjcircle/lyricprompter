"""Check that pymatchmaker 0.3.0 OnlineTimeWarpingArztFrame can be used audio-to-audio
(reference = feature matrix of a recording, 'beats' = reference seconds), and time it."""
import time
import numpy as np

t_imp = time.perf_counter()
from matchmaker.dp.oltw_arzt import OnlineTimeWarpingArztFrame  # noqa: E402
print(f"import ok ({time.perf_counter()-t_imp:.2f}s)")

rng = np.random.default_rng(1)
fps, dur, dim = 25, 240, 24
N = dur * fps
base = rng.standard_normal((N // 5 + 2, dim))
R = (np.repeat(base, 5, axis=0)[:N] + 0.3 * rng.standard_normal((N, dim))).astype(np.float32)
R /= np.linalg.norm(R, axis=1, keepdims=True)
gt, t_ref = [], 0.0
while t_ref < N - 2:
    r = rng.uniform(0.8, 1.25)
    for _ in range(rng.integers(2 * fps, 8 * fps)):
        gt.append(t_ref); t_ref += r
        if t_ref >= N - 1:
            break
gt = np.array(gt)
L = (R[np.round(gt).astype(int)] + 0.6 * rng.standard_normal((len(gt), dim)) / np.sqrt(dim)).astype(np.float32)
L /= np.linalg.norm(L, axis=1, keepdims=True)

ref_sec = np.arange(N, dtype=np.float32) / fps
trk = OnlineTimeWarpingArztFrame(
    reference_features=R,
    score_positions=ref_sec,          # "beats" == reference seconds
    ref_frame_to_beat=ref_sec,
    window_size=10,                   # seconds (each side)
    step_size=3,                      # max ref frames advanced per live frame
    start_window_size=2,              # seconds
    distance_func="Cosine",
    frame_rate=fps,
)
est, tt = [], []
for t, x in enumerate(L):
    t0 = time.perf_counter()
    est.append(trk(x[None, :], t / fps))   # returns reference time (s)
    tt.append(time.perf_counter() - t0)
err_ms = np.abs(np.array(est) - gt / fps) * 1000
print({f"<= {k}ms": round(float(np.mean(err_ms <= k)) * 100, 1) for k in (200, 300, 500, 1000)},
      "median err ms", round(float(np.median(err_ms)), 1),
      "step us p50/p99", round(1e6 * float(np.median(tt)), 1), round(1e6 * float(np.percentile(tt, 99)), 1))
