"""Measure PESTO 2.0.1 streaming behaviour at 16 kHz: block size semantics, per-call latency, and
algorithmic lag (how many ms after a pitch change the streamed output follows)."""
import time
import numpy as np
import torch
from pesto import load_model

SR = 16000
for step_ms in (20.0, 10.0):
    hop = int(SR * step_ms / 1000)
    m = load_model("mir-1k_g7", step_size=step_ms, sampling_rate=SR, streaming=True, max_batch_size=1)
    m.eval()
    # HCQT kernels are built lazily on first call (sampling rate inferred) -> run one block first
    with torch.inference_mode():
        m(torch.zeros(1, hop), return_activations=False)
    widths = []
    for mod in m.modules():
        if hasattr(mod, "kernel_width"):
            widths.append(int(mod.kernel_width))
    widths = widths or [0]
    print(f"step {step_ms} ms hop {hop}: CQT kernel widths (samples) {sorted(set(widths))} -> max {max(widths)/SR*1000:.0f} ms")
    # signal: 2 s of 220 Hz (A3=57) then 2 s of 330 Hz (E4=64)
    t = np.arange(int(4 * SR)) / SR
    f = np.where(t < 2, 220.0, 330.0)
    y = (0.3 * np.sin(2 * np.pi * np.cumsum(f) / SR)).astype(np.float32)
    preds, times, lat = [], [], []
    with torch.inference_mode():
        for b in (hop, 2 * hop):
            out = m(torch.from_numpy(y[:b])[None], return_activations=False)
            print(f"  block {b} samples -> pitch shape {tuple(out[0].shape)}")
        m = load_model("mir-1k_g7", step_size=step_ms, sampling_rate=SR, streaming=True, max_batch_size=1).eval()
        for i in range(0, len(y) - hop + 1, hop):
            t0 = time.perf_counter()
            p, c, a = m(torch.from_numpy(y[i:i + hop])[None], return_activations=False)
            lat.append(time.perf_counter() - t0)
            preds.append(float(p.reshape(-1)[-1])); times.append((i + hop) / SR)
    preds = np.array(preds); times = np.array(times)
    # first time after 2.0 s where prediction is within 0.5 semitone of 64
    idx = np.where((times > 2.0) & (np.abs(preds - 64) < 0.5))[0]
    lag = (times[idx[0]] - 2.0) * 1000 if len(idx) else None
    print(f"  pitch before switch ~{np.median(preds[(times>1)&(times<2)]):.2f} (expect 57); lag to follow switch: {lag} ms")
    print(f"  per-call CPU latency p50 {np.median(lat)*1e3:.2f} ms p99 {np.percentile(lat,99)*1e3:.2f} ms")
