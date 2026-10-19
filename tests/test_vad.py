from __future__ import annotations

import numpy as np
import pytest

from lyricprompter.prep.vad import activity_threshold, dilate, frame_rms_db, vocal_activity

SR = 16000
HOP = 320


def tone(seconds: float, amp: float, f: float = 250.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_frame_rms_db_levels_and_length():
    y = np.concatenate([tone(1.0, 0.5), np.zeros(SR // 2, np.float32), tone(0.01, 0.5)])
    db = frame_rms_db(y, HOP)
    assert len(db) == int(np.ceil(len(y) / HOP))
    # sine RMS = amp / sqrt(2) -> 20*log10(0.3536) = -9.03 dB
    assert db[10] == pytest.approx(-9.03, abs=0.05)
    assert db[60] < -100
    assert frame_rms_db(np.zeros(0, np.float32), HOP).shape == (0,)


def test_dilate():
    m = np.zeros(20, bool)
    m[10] = True
    d = dilate(m, 3)
    assert np.flatnonzero(d).tolist() == list(range(7, 14))
    assert dilate(m, 0).tolist() == m.tolist()
    assert not dilate(np.zeros(5, bool), 2).any()


def test_vocal_activity_burst_with_dilation():
    y = np.concatenate([np.zeros(SR, np.float32), tone(1.0, 0.3), np.zeros(SR, np.float32)])
    act = vocal_activity(y, SR, hop=HOP, dilate_s=0.0)
    assert act.dtype == bool and len(act) == 150
    assert act[50:100].all() and not act[:50].any() and not act[100:].any()
    act_d = vocal_activity(y, SR, hop=HOP, dilate_s=0.3)
    assert act_d[35:115].all() and not act_d[:34].any() and not act_d[116:].any()


def test_digital_silence_is_never_active():
    assert not vocal_activity(np.zeros(SR, np.float32)).any()
    assert not vocal_activity(np.full(SR, 1e-5, np.float32)).any()     # -100 dBFS < absolute floor


def test_relative_threshold_rejects_quiet_bleed():
    y = np.concatenate([tone(1.0, 0.5), tone(1.0, 0.5 * 10 ** (-40 / 20))])   # bleed 40 dB below
    act = vocal_activity(y, SR, hop=HOP, rel_db=-35.0, dilate_s=0.0)
    assert act[:50].all() and not act[50:].any()


def test_adaptive_threshold_tracks_noise_floor_within_bounds():
    def levels(floor: float) -> np.ndarray:                    # loud level -10 dB, 20 % of frames at `floor`
        return np.concatenate([np.full(800, -10.0), np.full(200, floor)])

    assert activity_threshold(levels(-75.0), noise_margin_db=20.0) == pytest.approx(-55.0)   # floor + 20
    assert activity_threshold(levels(-90.0), noise_margin_db=20.0) == pytest.approx(-65.0)   # clip: loud - 55
    assert activity_threshold(levels(-60.0), noise_margin_db=20.0) == pytest.approx(-45.0)   # clip: loud - 35
    assert activity_threshold(levels(-90.0)) == pytest.approx(-45.0)                       # fixed: loud - 35
    assert activity_threshold(np.full(10, -200.0), noise_margin_db=20.0) == pytest.approx(-70.0)  # abs floor


def test_adaptive_mode_keeps_soft_singing_of_clean_stem():
    soft = tone(1.0, 0.5 * 10 ** (-45 / 20))                  # 45 dB below the loud part
    y = np.concatenate([tone(2.0, 0.5), soft, np.zeros(SR, np.float32)])
    fixed = vocal_activity(y, SR, hop=HOP, dilate_s=0.0)
    adaptive = vocal_activity(y, SR, hop=HOP, dilate_s=0.0, noise_margin_db=20.0, min_rel_db=-55.0)
    assert not fixed[105:145].any()
    assert adaptive[105:145].all() and not adaptive[155:].any()
