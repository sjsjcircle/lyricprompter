"""Vocal activity from a separated vocal stem (energy gate used to constrain the forced aligner).

A frame is active when its RMS is within `rel_db` of the stem's loud level (99th percentile of frame dB)
and above an absolute floor (so digital silence never counts as singing).

Adaptive mode (`noise_margin_db` set): the threshold becomes `noise + noise_margin_db` (noise = the
`noise_pct` percentile of frame dB, i.e. the separation-bleed floor), clipped to
[loud + min_rel_db, loud + rel_db]. Soft passages of clean stems stay active while the higher bleed floor of
weaker separators is still rejected (see docs/eval/ALIGNMENT.md).
"""
from __future__ import annotations

import numpy as np

ABS_FLOOR_DB = -70.0


def frame_rms_db(y: np.ndarray, hop: int) -> np.ndarray:
    """dBFS RMS of consecutive non-overlapping `hop`-sample frames (last partial frame zero-padded)."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    n = int(np.ceil(len(y) / hop)) if len(y) else 0
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    x = np.zeros(n * hop, dtype=np.float32)
    x[: len(y)] = y
    ms = np.mean(np.square(x.reshape(n, hop), dtype=np.float64), axis=1)
    return (10.0 * np.log10(ms + 1e-12)).astype(np.float32)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation of a 1-D bool mask by `radius` frames on each side."""
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0 or not mask.any():
        return mask.copy()
    c = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
    idx = np.arange(len(mask))
    lo = np.clip(idx - radius, 0, len(mask))
    hi = np.clip(idx + radius + 1, 0, len(mask))
    return (c[hi] - c[lo]) > 0


def activity_threshold(db: np.ndarray, rel_db: float = -35.0, floor_db: float = ABS_FLOOR_DB,
                       noise_margin_db: float | None = None, min_rel_db: float = -55.0,
                       noise_pct: float = 10.0) -> float:
    """dB threshold for `vocal_activity` given the frame levels of the whole stem."""
    loud = float(np.percentile(db, 99))
    thr = loud + rel_db
    if noise_margin_db is not None:
        noise = float(np.percentile(db, noise_pct))
        thr = float(np.clip(noise + noise_margin_db, loud + min_rel_db, loud + rel_db))
    return max(thr, floor_db)


def vocal_activity(y: np.ndarray, sr: int = 16000, hop: int = 320, rel_db: float = -35.0,
                   dilate_s: float = 0.3, floor_db: float = ABS_FLOOR_DB,
                   noise_margin_db: float | None = None, min_rel_db: float = -55.0,
                   noise_pct: float = 10.0) -> np.ndarray:
    """Bool per `hop` frame (frame k covers samples [k*hop, (k+1)*hop)): is the vocal stem singing?"""
    db = frame_rms_db(y, hop)
    if len(db) == 0:
        return np.zeros(0, dtype=bool)
    thr = activity_threshold(db, rel_db, floor_db, noise_margin_db, min_rel_db, noise_pct)
    return dilate(db > thr, round(dilate_s * sr / hop))
