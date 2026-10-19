from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lyricprompter.features.extractor import FeatureFrames
from lyricprompter.tracking.tracker import (
    COASTING,
    HOLD,
    STOPPED,
    TRACKING,
    WAITING,
    LevelGate,
    LiveTracker,
    TrackerParams,
    phrase_onsets,
    ref_activity,
)

FPS = 25.0
DIM = 16


def _unit(x):
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def _ref_feats(rng, n, seg=5):
    base = rng.standard_normal((n // seg + 2, DIM))
    return _unit(np.repeat(base, seg, axis=0)[:n] + 0.3 * rng.standard_normal((n, DIM)))


def _live_from(R, gt, rng, noise=0.5):
    return _unit(R[np.clip(np.round(gt).astype(int), 0, len(R) - 1)] +
                 noise * rng.standard_normal((len(gt), DIM)) / np.sqrt(DIM))


def _piecewise_gt(rng, n_ref, lo=0.8, hi=1.25):
    gt, t = [], 0.0
    while t < n_ref - 2:
        seg, rho = rng.integers(2 * FPS, 8 * FPS), np.exp(rng.uniform(np.log(lo), np.log(hi)))
        for _ in range(int(seg)):
            gt.append(t)
            t += rho
            if t >= n_ref - 2:
                break
    return np.array(gt)


def _frames(feats, levels):
    return FeatureFrames(np.asarray(feats, np.float32), np.asarray(levels, np.float32))


def _run(trk, feats, levels, chunk=2):
    out = []
    for i in range(0, len(feats), chunk):
        st = trk.push(_frames(feats[i:i + chunk], levels[i:i + chunk]))
        out.extend([st] * len(feats[i:i + chunk]))
    return out


@pytest.mark.parametrize("seed", [0, 1])
def test_follows_synthetic_warped_sequence(seed):
    rng = np.random.default_rng(seed)
    R = _ref_feats(rng, 150 * 25)
    gt = _piecewise_gt(rng, len(R))
    L = _live_from(R, gt, rng)
    trk = LiveTracker(_frames(R, np.full(len(R), -20.0)), FPS, TrackerParams.for_mode("mix"))
    trk.start(0.0)
    states = _run(trk, L, np.full(len(L), -20.0))
    est = np.array([s.ref_time for s in states])
    err = np.abs(est - (gt + 1) / FPS)                     # ref_time of frame i is (i+1)/fps
    assert np.mean(err <= 0.3) > 0.9
    assert all(s.status == TRACKING for s in states)
    assert np.all(np.diff(est) >= 0)                        # monotonic output
    assert 0.75 < states[-1].tempo < 1.35
    assert np.mean([s.confidence for s in states[len(states) // 2:]]) > 0.5


def test_step_time_budget():
    rng = np.random.default_rng(3)
    R = _ref_feats(rng, 240 * 25)
    trk = LiveTracker(_frames(R, np.full(len(R), -20.0)), FPS, TrackerParams.for_mode("mix"))
    trk.start(0.0)
    L = _live_from(R, np.arange(0, 1500, 1.0), rng)
    import time
    t0 = time.perf_counter()
    _run(trk, L, np.full(len(L), -20.0), chunk=1)
    assert (time.perf_counter() - t0) / len(L) < 2e-3


def _vocal_song(rng, voice_s=(4.0, 4.0, 4.0), gap_s=(2.0, 3.0, 3.0)):
    """Reference: [gap][voice][gap][voice]... with silent frames as a fixed 'uniform' vector."""
    silence = _unit(np.ones(DIM))
    feats, levels, onsets = [], [], []
    for g, v in zip(gap_s, voice_s):
        n_g, n_v = round(g * FPS), round(v * FPS)
        feats.append(np.repeat(silence[None], n_g, 0))
        levels.append(np.full(n_g, -90.0))
        onsets.append(sum(len(x) for x in levels))
        feats.append(_ref_feats(rng, n_v))
        levels.append(np.full(n_v, -20.0))
    return np.concatenate(feats), np.concatenate(levels), onsets, silence


def test_vocal_mode_coasts_through_gaps_and_relocks():
    rng = np.random.default_rng(5)
    R, lv, onsets, silence = _vocal_song(rng)
    line_starts = [(o + 1) / FPS for o in onsets]
    trk = LiveTracker(_frames(R, lv), FPS, TrackerParams.for_mode("vocal"), line_starts=line_starts)
    trk.start(0.0)
    # live: singer enters 1 s late for phrase 1, 1 s early for phrase 2, on time for phrase 3
    live_gaps = [3.0, 2.0, 3.0]
    feats, levels, gt = [], [], []
    for k, g in enumerate(live_gaps):
        n_g = round(g * FPS)
        feats.append(np.repeat(silence[None], n_g, 0))
        levels.append(np.full(n_g, -90.0))
        gt.append(np.full(n_g, np.nan))
        seg = np.arange(onsets[k], onsets[k] + 4 * FPS)
        feats.append(_live_from(R, seg, rng, noise=0.4))
        levels.append(np.full(len(seg), -20.0))
        gt.append(seg.astype(float))
    feats, levels, gt = np.concatenate(feats), np.concatenate(levels), np.concatenate(gt)
    states = _run(trk, feats, levels, chunk=1)
    status = np.array([s.status for s in states])
    est = np.array([s.ref_time for s in states])
    gaps = np.isnan(gt)
    # silence -> coasting (after the 0.3 s grace period), voice -> tracking
    assert np.mean(status[gaps] == COASTING) > 0.8
    assert np.mean(status[~gaps] == TRACKING) > 0.9
    # coasting never runs past the next phrase onset
    starts = np.flatnonzero(gaps & ~np.r_[False, gaps[:-1]])
    assert len(starts) == 3
    for k, s in enumerate(starts):
        e = s + round(live_gaps[k] * FPS)
        assert est[s:e].max() <= line_starts[k] + 1e-6
    # after re-locking, position error is small 0.5 s into each phrase
    voiced = ~gaps
    ok = np.zeros(len(gt), bool)
    for k in range(3):
        idx = np.flatnonzero(voiced)[k * 100:(k + 1) * 100][12:]
        ok[idx] = True
    err = np.abs(est[ok] - (gt[ok] + 1) / FPS)
    assert np.mean(err <= 0.3) > 0.9


def test_waiting_until_start_and_stop():
    rng = np.random.default_rng(6)
    R = _ref_feats(rng, 500)
    trk = LiveTracker(_frames(R, np.full(500, -20.0)), FPS, TrackerParams())
    st = trk.push(_frames(R[:50], np.full(50, -20.0)))
    assert st.status == WAITING and st.ref_time == pytest.approx(1 / FPS) and st.frame == 0
    trk.start(4.0)
    st = trk.push(_frames(R[99:110], np.full(11, -20.0)))
    assert st.status == TRACKING and st.frame == 11 and st.live_time == pytest.approx(11 / FPS)
    assert st.ref_time >= 4.0
    trk.stop()
    before = trk.state
    st = trk.push(_frames(R[110:150], np.full(40, -20.0)))
    assert st.status == STOPPED and st.ref_time == before.ref_time


def test_hold_jump_nudge_semantics():
    rng = np.random.default_rng(7)
    R = _ref_feats(rng, 2000)
    lv = np.full(len(R), -20.0)
    trk = LiveTracker(_frames(R, lv), FPS, TrackerParams())
    trk.start(0.0)
    _run(trk, R[:250], lv[:250])
    t_hold = trk.state.ref_time
    assert abs(t_hold - 10.0) < 0.3
    trk.hold(True)
    st = trk.push(_frames(R[250:400], lv[250:400]))         # performer moves on, display frozen
    assert st.status == HOLD and st.ref_time == t_hold and st.frame == 400 and st.tempo == 0.0
    trk.jump(30.0)                                          # jump while held sets the held position
    assert trk.state.ref_time == pytest.approx(30.0) and trk.state.status == HOLD
    trk.hold(False)
    assert trk.state.status == TRACKING
    states = _run(trk, R[750:1000], lv[750:1000])         # live continues from ref 30 s
    assert abs(states[-1].ref_time - 40.0) < 0.3
    trk.jump(20.0)                                          # backward operator jump is allowed
    assert trk.state.ref_time == pytest.approx(20.0)
    states = _run(trk, R[500:600], lv[500:600])
    assert abs(states[-1].ref_time - 24.0) < 0.3
    trk.nudge(-1.0)
    assert trk.state.ref_time == pytest.approx(states[-1].ref_time - 1.0)
    trk.nudge(+2.5)
    assert trk.state.ref_time == pytest.approx(states[-1].ref_time + 1.5)


def test_latency_compensation():
    rng = np.random.default_rng(8)
    R = _ref_feats(rng, 600)
    lv = np.full(len(R), -20.0)
    a = LiveTracker(_frames(R, lv), FPS, TrackerParams())
    b = LiveTracker(_frames(R, lv), FPS, TrackerParams(latency_s=0.2))
    for t in (a, b):
        t.start(0.0)
        _run(t, R[:300], lv[:300])
    assert b.state.ref_time - a.state.ref_time == pytest.approx(0.2 * a.state.tempo, abs=1e-6)
    a.set_latency(0.2)
    assert a.state.ref_time == pytest.approx(b.state.ref_time)


def test_set_transpose_rolls_reference_chroma():
    rng = np.random.default_rng(9)
    chroma = _unit(rng.random((300, 12)))
    other = _unit(rng.standard_normal((300, 4)))
    ff = FeatureFrames(np.concatenate([chroma, other], 1), np.full(300, -20.0, np.float32),
                       (("chroma", 0, 12), ("mfcc", 12, 16)))
    trk = LiveTracker(ff, FPS, TrackerParams())
    trk.set_transpose(2)
    np.testing.assert_allclose(trk.oltw.R[:, :12], np.roll(chroma, 2, axis=1))
    np.testing.assert_allclose(trk.oltw.R[:, 12:], other)
    assert trk.state.transpose == 2
    trk.set_transpose(0)
    np.testing.assert_allclose(trk.oltw.R, ff.feats)
    # a live performance 2 semitones up is followed once the operator presets the key
    live = np.concatenate([np.roll(chroma, 2, axis=1), other], 1)[:200]
    trk.set_transpose(2)
    trk.start(0.0)
    st = trk.push(FeatureFrames(live, np.full(200, -20.0, np.float32)))
    assert abs(st.ref_time - 200 / FPS) < 0.2


def test_global_relocation_recovers_from_wrong_start():
    rng = np.random.default_rng(10)
    R = _ref_feats(rng, 120 * 25)
    lv = np.full(len(R), -20.0)
    trk = LiveTracker(_frames(R, lv), FPS, TrackerParams())
    trk.start(70.0)                                          # operator started at the wrong place
    gt = np.arange(20 * 25, 45 * 25, 1.0)                    # performer is at 20..45 s
    states = _run(trk, _live_from(R, gt, rng), lv[:len(gt)])
    assert trk.relocations >= 1
    est = np.array([s.ref_time for s in states])
    assert np.abs(est[-100:] - (gt[-100:] + 1) / FPS).max() < 0.3


def test_level_gate_adapts_and_has_hysteresis():
    g = LevelGate(FPS, TrackerParams())
    noise = [g.update(-50.0 + 0.5 * np.sin(i)) for i in range(100)]
    assert not any(noise[30:])                               # steady noise floor -> inactive
    assert g.update(-20.0)                                   # 30 dB above the floor
    assert g.update(-42.0)                                   # still above floor + off threshold
    assert not g.update(-48.0)


def test_ref_activity_closes_short_gaps():
    lv = np.full(200, -80.0)
    lv[20:60] = -20.0
    lv[63:100] = -20.0                                       # 3-frame gap closed
    lv[150] = -20.0                                          # 1-frame blip removed
    a = ref_activity(lv, FPS, rel_db=25.0, min_gap_s=0.3, min_on_s=0.1)
    assert a[20:100].all() and not a[:20].any() and not a[100:].any()


def test_bad_inputs():
    R = _unit(np.random.default_rng(0).standard_normal((50, DIM)))
    with pytest.raises(ValueError):
        LiveTracker(_frames(R, np.zeros(50)), FPS, TrackerParams(mode="karaoke"))
    trk = LiveTracker(_frames(R, np.zeros(50)), FPS, TrackerParams())
    trk.start()
    with pytest.raises(ValueError):
        trk.push(_frames(np.zeros((2, DIM + 1)), np.zeros(2)))
    assert TrackerParams.for_mode("vocal").mode == "vocal"


# ---------------------------------------------------------------------- vocal-mode regressions
def _blocks(rng, spec):
    """Reference from (n_frames, voiced) blocks: voiced = random phrase features, else silence."""
    sil = _unit(np.ones(DIM))
    feats = [_ref_feats(rng, n) if v else np.repeat(sil[None], n, 0) for n, v in spec]
    levels = [np.full(n, -20.0 if v else -90.0) for n, v in spec]
    return np.concatenate(feats), np.concatenate(levels), sil


@pytest.mark.parametrize("how", ["start", "jump", "release"])
@pytest.mark.parametrize("wait_s", [0.5, 4.0])
def test_vocal_start_jump_release_at_line_start_wait_for_the_voice(how, wait_s):
    rng = np.random.default_rng(11)
    R, lv, onsets, silence = _vocal_song(rng, voice_s=(6.0, 6.0, 6.0), gap_s=(2.0, 2.0, 2.0))
    line_starts = [(o + 1) / FPS for o in onsets]
    target = line_starts[1]
    trk = LiveTracker(_frames(R, lv), FPS, TrackerParams.for_mode("vocal"), line_starts=line_starts)
    if how == "start":
        trk.start(target)
    else:
        trk.start(0.0)
        seg = np.arange(onsets[0], onsets[0] + 150)                    # phrase 0 sung, then a pause
        pre = np.concatenate([np.repeat(silence[None], 50, 0), _live_from(R, seg, rng, 0.4),
                              np.repeat(silence[None], 10, 0)])
        _run(trk, pre, np.r_[np.full(50, -90.0), np.full(150, -20.0), np.full(10, -90.0)], chunk=1)
        if how == "jump":
            trk.jump(target)
        else:
            trk.hold(True)
            trk.jump(target)
            trk.hold(False)
    n_w = round(wait_s * FPS)
    sts = _run(trk, np.repeat(silence[None], n_w, 0), np.full(n_w, -90.0), chunk=1)
    assert all(s.status == COASTING and s.ref_time == pytest.approx(target) for s in sts)   # waits there
    assert sts[-1].tempo == 0.0                                       # parked: clients must not extrapolate
    seg = np.arange(onsets[1], onsets[1] + 6 * FPS)
    sts = _run(trk, _live_from(R, seg, rng, 0.4), np.full(len(seg), -20.0), chunk=1)
    err = np.array([s.ref_time for s in sts]) - (seg + 1) / FPS
    assert np.mean(np.abs(err) <= 0.3) > 0.9 and np.abs(err[12:]).max() < 0.5
    assert trk.relocations == 0


# ref: 2 s silence | line 0: 3 s voice, 1 s rest, 3 s voice | 2 s silence | line 1: 4 s voice
_REST_SONG = [(50, False), (75, True), (25, False), (75, True), (50, False), (100, True)]
_REST_LINES = [(50 + 1) / FPS, (275 + 1) / FPS]


def _sing_rest_song(trk, R, sil, rng, rest):
    L = np.concatenate([np.repeat(sil[None], 50, 0), _live_from(R, np.arange(50, 125), rng, 0.4),
                        np.repeat(sil[None], rest, 0), _live_from(R, np.arange(150, 225), rng, 0.4)])
    lvl = np.r_[np.full(50, -90.0), np.full(75, -20.0), np.full(rest, -90.0), np.full(75, -20.0)]
    trk.start(0.0)
    return np.array([s.ref_time for s in _run(trk, L, lvl, chunk=1)])


@pytest.mark.parametrize("extra_s", [-0.4, 0.0, 1.0, 2.0])
def test_vocal_relock_undoes_coasting_overshoot(extra_s):
    """The singer holds an in-line rest longer than the reference: coasting overshoots into the rest
    of the line, and the relock must move the published position back (it used to stay ahead)."""
    rng = np.random.default_rng(1)
    R, lv, sil = _blocks(rng, _REST_SONG)
    trk = LiveTracker(_frames(R, lv), FPS, TrackerParams.for_mode("vocal"), line_starts=_REST_LINES)
    rest = 25 + round(extra_s * FPS)
    est = _sing_rest_song(trk, R, sil, rng, rest)
    i2 = 125 + rest                                                   # live frame where the voice resumes
    err = est[i2:] - (np.arange(150, 225) + 1) / FPS
    assert np.mean(np.abs(err[12:]) <= 0.3) > 0.9, err[:30]
    assert est[:i2].max() <= _REST_LINES[1] + 1e-9                   # never past the next line start
    assert trk.relocations == 0


def test_vocal_inline_onsets_opt_in():
    """onset_rest_max_s > 0: coasting also waits at the end of a reference rest inside a line, for
    at most inline_wait_s (then it continues and the relock corrects)."""
    rng = np.random.default_rng(1)
    R, lv, sil = _blocks(rng, _REST_SONG)
    rest_end = (150 + 1) / FPS
    for wait_s, extra_s in ((5.0, 1.5), (0.4, 1.5)):
        p = replace(TrackerParams.for_mode("vocal"), onset_rest_max_s=2.5, inline_wait_s=wait_s)
        trk = LiveTracker(_frames(R, lv), FPS, p, line_starts=_REST_LINES)
        assert 150 in trk.onsets                                      # end of the in-line rest
        rest = 25 + round(extra_s * FPS)
        est = _sing_rest_song(trk, R, sil, np.random.default_rng(2), rest)
        i2 = 125 + rest
        if wait_s > extra_s + 1.0:
            assert est[125:i2].max() == pytest.approx(rest_end)       # parked at the rest's end
        else:
            assert est[i2 - 1] > rest_end + 0.3                       # gave up waiting, kept coasting
        err = est[i2:] - (np.arange(150, 225) + 1) / FPS
        assert np.mean(np.abs(err[12:]) <= 0.3) > 0.9


def test_phrase_onsets_chain_inline_rests_but_not_interludes():
    act = np.zeros(400, bool)
    for a, b in ((10, 20), (25, 35), (40, 50), (200, 205), (300, 320)):
        act[a:b] = True
    assert phrase_onsets(act, np.array([], np.int64), 5).tolist() == [10, 25, 40, 200, 300]
    assert phrase_onsets(act, np.array([10, 300]), 5).tolist() == [10, 25, 40, 300]
    assert phrase_onsets(act, np.array([10, 300]), 0).tolist() == [10, 300]
    assert phrase_onsets(act, np.array([10, 30, 300]), 5).tolist() == [10, 25, 30, 40, 300]


@pytest.mark.parametrize("seed", range(4))
def test_vocal_interlude_with_bleed_bursts_does_not_ratchet(seed):
    """30 s instrumental interlude: the live gate opens on short bleed bursts (< relock_confirm_s).
    The display may run ahead inside the interlude (no lyrics there) but must not enter the next line
    (at most the confirmation time while a burst lasts) and then follows it. The tracker used to run
    through the whole rest of the song here."""
    rng = np.random.default_rng(seed)
    spec = [(50, False), (100, True), (750, False), (150, True), (50, False)]
    R, lv, sil = _blocks(rng, spec)
    line_starts = [(50 + 1) / FPS, (900 + 1) / FPS]
    p = TrackerParams.for_mode("vocal")
    trk = LiveTracker(_frames(R, lv), FPS, p, line_starts=line_starts)
    trk.start(0.0)
    inter = np.repeat(sil[None], 750, 0).copy()
    ilv = np.full(750, -90.0)
    k = 10
    while k < 730:                                                    # bursts of 0.12-0.28 s every 1-2 s
        n = int(rng.integers(3, 8))
        inter[k:k + n] = _unit(rng.standard_normal((n, DIM)))
        ilv[k:k + n] = -40.0
        k += n + int(rng.integers(25, 50))
    L = np.concatenate([np.repeat(sil[None], 50, 0), _live_from(R, np.arange(50, 150), rng, 0.4), inter,
                        _live_from(R, np.arange(900, 1050), rng, 0.4)])
    lvl = np.r_[np.full(50, -90.0), np.full(100, -20.0), ilv, np.full(150, -20.0)]
    est = np.array([s.ref_time for s in _run(trk, L, lvl, chunk=1)])
    assert est[:900].max() <= line_starts[1] + p.relock_confirm_s * p.coast_tempo_max + 1e-9
    assert line_starts[1] - 0.5 <= est[899] <= line_starts[1] + 1e-9   # at (or coasting to) line B
    err = est[900:] - (np.arange(900, 1050) + 1) / FPS
    assert np.mean(np.abs(err[12:]) <= 0.3) > 0.9
    assert trk.relocations == 0


def test_vocal_latency_never_displays_past_the_coast_limit():
    rng = np.random.default_rng(5)
    R, lv, onsets, silence = _vocal_song(rng)
    line_starts = [(o + 1) / FPS for o in onsets]
    trk = LiveTracker(_frames(R, lv), FPS, replace(TrackerParams.for_mode("vocal"), latency_s=0.3),
                      line_starts=line_starts)
    trk.start(0.0)
    seg = np.arange(onsets[0], onsets[0] + 4 * FPS)
    _run(trk, np.repeat(silence[None], 50, 0), np.full(50, -90.0), chunk=1)
    sts = _run(trk, _live_from(R, seg, rng, 0.4), np.full(len(seg), -20.0), chunk=1)
    st = sts[-1]
    assert st.status == TRACKING and st.ref_time == pytest.approx((trk._out + 1) / FPS + 0.3 * st.tempo)
    sts = _run(trk, np.repeat(silence[None], 150, 0), np.full(150, -90.0), chunk=1)   # 6 s pause
    assert max(s.ref_time for s in sts) <= line_starts[1] + 1e-9
    assert sts[-1].status == COASTING and sts[-1].ref_time == pytest.approx(line_starts[1])
    assert sts[-1].tempo == 0.0


def test_set_transpose_needs_a_chroma_group():
    R = _unit(np.random.default_rng(0).standard_normal((50, DIM)))
    trk = LiveTracker(_frames(R, np.zeros(50)), FPS, TrackerParams())   # contract-shaped: no groups
    trk.set_transpose(12)                                              # an octave is a no-op
    assert trk.state.transpose == 12
    with pytest.raises(ValueError, match="chroma"):
        trk.set_transpose(2)
    assert trk.state.transpose == 12


def test_set_line_starts_updates_the_coast_limits():
    rng = np.random.default_rng(5)
    R, lv, onsets, _ = _vocal_song(rng)
    trk = LiveTracker(_frames(R, lv), FPS, TrackerParams.for_mode("vocal"), line_starts=[(onsets[0] + 1) / FPS])
    assert trk.onsets.tolist() == [onsets[0]]
    trk.set_line_starts([(o + 1) / FPS for o in onsets] + [None, float("nan")])
    assert trk.onsets.tolist() == onsets
    trk.set_line_starts(None)                                          # -> reference activity edges
    assert trk.onsets.tolist() == onsets


def _chroma_ref(rng, n, seg=5):
    """Reference with a 12-d chroma group (non-negative, segment-wise) + 4-d timbre group."""
    chroma = np.repeat(rng.random((n // seg + 2, 12)) ** 3, seg, axis=0)[:n] + 0.05 * rng.random((n, 12))
    other = np.repeat(rng.standard_normal((n // seg + 2, 4)), seg, axis=0)[:n]
    feats = np.concatenate([_unit(chroma), 0.5 * _unit(other)], 1).astype(np.float32)
    return FeatureFrames(feats, np.full(n, -20.0, np.float32), (("chroma", 0, 12), ("mfcc", 12, 16)))


def _transposed_live(ref, gt, k, rng, noise=0.15):
    idx = np.clip(np.round(gt).astype(int), 0, len(ref) - 1)
    live = ref.feats[idx].copy()
    live[:, :12] = np.roll(live[:, :12], k, axis=1)
    live += noise * rng.standard_normal(live.shape).astype(np.float32) / 4
    return live.astype(np.float32)


@pytest.mark.parametrize("mode", ["mix", "vocal"])
@pytest.mark.parametrize("k", [3, -2])
def test_auto_key_detects_a_transposed_performance(mode, k):
    rng = np.random.default_rng(20 + k)
    ref = _chroma_ref(rng, 90 * 25)
    gt = _piecewise_gt(rng, len(ref) - 10, lo=0.9, hi=1.1)
    live = _transposed_live(ref, gt, k, rng)
    pre = 25                                                 # 1 s of silence: the level gate needs a floor
    feats = np.concatenate([_unit(rng.random((pre, 16)) + 1e-3), live]).astype(np.float32)
    levels = np.r_[np.full(pre, -80.0), np.full(len(live), -20.0)]
    trk = LiveTracker(ref, FPS, TrackerParams.for_mode(mode))
    trk.start(0.0)
    states = _run(trk, feats, levels, chunk=4)[pre:]
    assert trk.state.transpose == k and trk.key_switches == 1 and trk.state.key_auto
    if mode == "mix":     # (vocal: a constant level eventually reads as the noise floor -> coasting)
        est = np.array([s.ref_time for s in states])
        late = slice(len(gt) // 2, None)                     # after detection: followed under the right key
        assert np.mean(np.abs(est[late] - (gt[late] + 1) / FPS) <= 0.3) > 0.9


def test_auto_key_does_not_switch_without_transposition():
    rng = np.random.default_rng(31)
    ref = _chroma_ref(rng, 60 * 25)
    gt = _piecewise_gt(rng, len(ref) - 10, lo=0.85, hi=1.15)
    live = _transposed_live(ref, gt, 0, rng, noise=0.3)
    # 1 s of silence first: a level that never changes is indistinguishable from stationary room noise,
    # which (by design) is not key evidence until the live gate has seen the music start
    pre = 25
    live = np.concatenate([_unit(rng.random((pre, 16)) + 1e-3), live]).astype(np.float32)
    trk = LiveTracker(ref, FPS, TrackerParams.for_mode("mix"))
    trk.start(0.0)
    _run(trk, live, np.r_[np.full(pre, -80.0), np.full(len(live) - pre, -20.0)], chunk=4)
    assert trk.state.transpose == 0 and trk.key_switches == 0
    assert trk._keys is None                                 # search ended (key_max_s) and freed its shadows


def test_operator_key_turns_auto_key_off_and_back_on():
    rng = np.random.default_rng(32)
    ref = _chroma_ref(rng, 30 * 25)
    trk = LiveTracker(ref, FPS, TrackerParams.for_mode("mix"))
    assert trk.state.key_auto
    trk.set_transpose(2)
    assert not trk.state.key_auto and trk.state.transpose == 2
    trk.start(0.0)
    assert trk._keys is None                                 # no search after an operator key
    trk.set_auto_key(True)
    assert trk.state.key_auto and trk._keys is not None and 2 not in trk._keys.shadows
    assert len(trk._keys.shadows) == 11                      # every other key
    no_chroma = LiveTracker(FeatureFrames(ref.feats, ref.rms_db), FPS, TrackerParams.for_mode("mix"))
    assert not no_chroma.state.key_auto                      # nothing to roll: auto key unavailable


# ---------------------------------------------------------------------- auto key: evidence, anchoring, controls
def _chords(rng, n_chords=8):
    return rng.random((n_chords, 12)) ** 3 + 0.02, rng.standard_normal((n_chords, 4))


def _chord_feats(C, M, idx, k=0):
    """Frames of chord idx (chroma rolled by k semitones) + a 4-d timbre group, unit norm."""
    ch = np.roll(C[idx], k, axis=-1)
    return np.concatenate([np.sqrt(0.8) * _unit(ch), np.sqrt(0.2) * _unit(M[idx])], -1).astype(np.float32)


_GROUPS = (("chroma", 0, 12), ("mfcc", 12, 16))
_SEG = 5                                                     # frames per chord


def _chord_ref(C, M, n):
    """A reference cycling through the chords: the tracking is the same at every phase, so lead-ins that
    move the tracker around do not matter, only the key evidence does."""
    return FeatureFrames(_chord_feats(C, M, (np.arange(n) // _SEG) % len(C)), np.full(n, -20.0, np.float32), _GROUPS)


def _chord_live(C, M, n, k, rng, start=0):
    idx = ((start + np.arange(n)) // _SEG) % len(C)
    x = _chord_feats(C, M, idx, k) + 0.05 * rng.standard_normal((n, 16)).astype(np.float32) / 4
    return x.astype(np.float32), -20.0 + 2.0 * rng.standard_normal(n)


def _room(C, M, n, rng, k_fit=5, level=-60.0):
    """Stationary room noise that fits one rolled reference (chord 0 under key k_fit) much better than the
    others: evidence for a wrong key if it were counted."""
    v = _chord_feats(C, M, np.zeros(n, int), k_fit) + 0.01 * rng.standard_normal((n, 16)).astype(np.float32)
    return v.astype(np.float32), level + 0.5 * rng.standard_normal(n)


def _applause(n, rng, level=-25.0):
    return _unit(rng.random((n, 16)) + 0.2), level + 1.0 * rng.standard_normal(n)


def _keys_of(states):
    return np.array([s.transpose for s in states])


@pytest.mark.parametrize("k", [0, -2])
def test_auto_key_ignores_room_noise_and_a_burst_before_the_music(k):
    """Mix mode, Start pressed before the band: room noise, a loud burst (applause / talk, longer than
    key_onset_s), more room noise, then the music. Room noise that fits a wrong key must not be evidence
    (it used to be counted once the burst had 'started' the music: false switch during the noise), and the
    music's key is found after the lead-in."""
    rng = np.random.default_rng(40)
    C, M = _chords(rng)
    ref = _chord_ref(C, M, 120 * 25)
    parts = [_room(C, M, 250, rng), _applause(100, rng), _room(C, M, 375, rng), _chord_live(C, M, 1250, k, rng)]
    feats = np.concatenate([f for f, _ in parts])
    levels = np.concatenate([lv for _, lv in parts])
    music = 725
    trk = LiveTracker(ref, FPS, TrackerParams.for_mode("mix"))
    trk.start(0.0)
    keys = _keys_of(_run(trk, feats, levels, chunk=4))
    assert (keys[:music] == 0).all(), np.flatnonzero(keys[:music])[:5]       # nothing decided on noise
    assert trk.state.transpose == k and trk.key_switches == (k != 0)
    if k:
        first = int(np.flatnonzero(keys == k)[0])
        assert first - music < 20 * FPS                                    # found after the music started


def test_auto_key_a_key_change_before_the_music_keeps_the_evidence_consistent():
    """A key adopted before the music started (a relocation under another key during a lead-in) used to
    leave evidence keyed by the old hypotheses, and the music's onset crashed the tracker (KeyError)."""
    rng = np.random.default_rng(41)
    C, M = _chords(rng)
    ref = _chord_ref(C, M, 60 * 25)
    trk = LiveTracker(ref, FPS, TrackerParams.for_mode("mix"))
    trk.start(0.0)
    _run(trk, *_room(C, M, 50, rng))
    feats, levels = _chord_live(C, M, 500, 0, rng)
    _run(trk, feats[:10], levels[:10])                        # the music starts ...
    assert not trk._keys.started
    trk._switch_key(-1)                                       # ... as a relocation adopts key -1
    _run(trk, feats[10:], levels[10:])                        # the onset (key_onset_s of activity) follows
    ks = trk._keys
    assert ks.started and set(ks.sums) == set(ks.shadows) and len(ks.shadows) == 11
    assert trk.state.transpose == 0 and trk.key_switches == 2  # ... and the search went back to the live key


def _anchors(trk):
    return {(h.i0, h.t, h.free) for h in (trk.oltw, *trk._keys.shadows.values())}


def test_auto_key_hypotheses_are_anchored_like_the_main():
    rng = np.random.default_rng(42)
    C, M = _chords(rng)
    ref = _chord_ref(C, M, 90 * 25)
    trk = LiveTracker(ref, FPS, TrackerParams.for_mode("mix"))
    trk.start(3.0)
    assert len(_anchors(trk)) == 1
    _run(trk, *_room(C, M, 25, rng, level=-80.0))
    _run(trk, *_chord_live(C, M, 150, 0, rng, start=75))
    for op in (lambda: trk.jump(20.0), lambda: trk.nudge(-1.5), lambda: trk.hold(True), lambda: trk.hold(False)):
        op()
        assert len(_anchors(trk)) == 1
        _run(trk, *_chord_live(C, M, 60, 0, rng))
        assert len(_anchors(trk)) == 1
    i0 = trk.oltw.i0
    trk.jump(30.0)
    assert len(_anchors(trk)) == 1 and trk.oltw.i0 != i0 and trk.oltw.t == 0
    # Auto turned on mid-run (after an operator key): the new hypotheses start where the display is
    trk.set_transpose(0)
    _run(trk, *_chord_live(C, M, 100, 0, rng))
    out = trk._out
    trk.set_auto_key(True)
    assert len(_anchors(trk)) == 1 and trk.oltw.i0 == round(out) and trk.oltw.t == 0
    assert len(trk._keys.shadows) == 11


def _spy_relocation(monkeypatch):
    """Record (live frame, transpose before, transpose after, success) of every relocation attempt and the
    live frame of the first key decision."""
    from lyricprompter.tracking import tracker as T

    log = {"tries": [], "first_decision": None}
    orig_try, orig_step = LiveTracker._try_relocate, T._KeySearch.step

    def spy_try(self):
        k0 = self._transpose
        ok = orig_try(self)
        log["tries"].append((self._frame, k0, self._transpose, ok))
        return ok

    def spy_step(self, *a, **kw):
        e = self.evals
        out = orig_step(self, *a, **kw)
        if self.evals > e and log["first_decision"] is None:
            log["first_decision"] = log["frame"]()
        return out

    monkeypatch.setattr(LiveTracker, "_try_relocate", spy_try)
    monkeypatch.setattr(T._KeySearch, "step", spy_step)
    return log


def _wrong_start(rng, ref, k):
    """Operator started at 70 s, the performer is at 20..45 s (k semitones up); 1 s of silence first."""
    gt = np.arange(20 * 25, 45 * 25, 1.0)
    pre = 25
    feats = np.concatenate([_unit(rng.random((pre, 16)) + 1e-3), _transposed_live(ref, gt, k, rng)])
    return gt, pre, feats, np.r_[np.full(pre, -80.0), np.full(len(gt), -20.0)]


def test_auto_key_relocation_waits_for_a_key_decision_and_reanchors_every_hypothesis(monkeypatch):
    """Before the first key decision a wrong key also looks lost, so relocation waits for it; a verified
    relocation re-anchors and replays every hypothesis like the new main."""
    rng = np.random.default_rng(10)
    ref = _chroma_ref(rng, 120 * 25)
    log = _spy_relocation(monkeypatch)
    trk = LiveTracker(ref, FPS, TrackerParams.for_mode("mix"))
    log["frame"] = lambda: trk._frame
    trk.start(70.0)
    gt, pre, feats, levels = _wrong_start(rng, ref, 0)
    states = _run(trk, feats, levels)[pre:]
    assert trk.relocations >= 1 and log["first_decision"] is not None
    assert log["tries"] and min(f for f, *_ in log["tries"]) >= log["first_decision"]
    est = np.array([s.ref_time for s in states])
    assert np.abs(est[-100:] - (gt[-100:] + 1) / FPS).max() < 0.3
    assert trk._keys is not None and len(_anchors(trk)) == 1    # still searching: all anchored alike


def test_auto_key_relocation_searches_every_key_and_adopts_the_candidates(monkeypatch):
    """A wrong start in a transposed take is wrong in both: the relocation searches under every candidate key
    and adopts the key it verified the candidate under."""
    rng = np.random.default_rng(11)
    ref = _chroma_ref(rng, 120 * 25)
    log = _spy_relocation(monkeypatch)
    trk = LiveTracker(ref, FPS, TrackerParams.for_mode("mix"))
    log["frame"] = lambda: trk._frame
    trk.start(70.0)
    gt, pre, feats, levels = _wrong_start(rng, ref, 3)
    states = _run(trk, feats, levels)[pre:]
    ok = [t for t in log["tries"] if t[3]]
    assert ok and ok[0][1:3] == (0, 3)                         # the first relocation found the place AND the key
    assert trk.state.transpose == 3
    est = np.array([s.ref_time for s in states])
    assert np.abs(est[-100:] - (gt[-100:] + 1) / FPS).max() < 0.3


def test_auto_key_switch_adopts_the_hypothesis_position_even_backwards():
    """A key correction may move the display backwards: the adopted hypothesis' position is published."""
    rng = np.random.default_rng(43)
    C, M = _chords(rng)
    ref = _chord_ref(C, M, 60 * 25)
    trk = LiveTracker(ref, FPS, TrackerParams.for_mode("mix"))
    trk.start(0.0)
    _run(trk, *_room(C, M, 25, rng, level=-80.0))
    _run(trk, *_chord_live(C, M, 125, 0, rng))
    sh = trk._keys.shadows[3]
    trk._out = float(sh.pos + 200)                            # the display ran ahead of that hypothesis
    assert trk.state.ref_time > (sh.pos + 1) / FPS
    trk._switch_key(3)
    st = trk.state
    assert trk.oltw is sh and st.transpose == 3 and trk.key_switches == 1
    assert st.ref_time == pytest.approx((sh.pos + 1) / FPS)


class _FakeShadow:
    def __init__(self, d):
        self.d = d

    def step(self, x):
        return type("R", (), {"d_pos": self.d})()


def _fake_search(d_shadows, mode="vocal"):
    """The key search of a started tracker with shadows of constant local distance (vocal: every tracked
    frame is evidence)."""
    rng = np.random.default_rng(44)
    trk = LiveTracker(_chroma_ref(rng, 30 * 25), FPS, TrackerParams.for_mode(mode))
    trk.start(0.0)
    ks = trk._keys
    ks.shadows = {k: _FakeShadow(d_shadows.get(k, 0.5)) for k in ks.shadows}
    ks.restart(budget=True)
    return trk, ks


def _feed(ks, d_main, n):
    x = np.zeros(16, np.float32)
    return [ks.step(x, d_main) for _ in range(n)]


def test_key_search_switches_on_a_clear_repeated_lead_and_only_suggests_a_small_one():
    p = TrackerParams.for_mode("vocal")
    first, every = round(p.key_detect_s * FPS), round(p.key_every_s * FPS)
    _, ks = _fake_search({3: 0.40})                           # 20 % better than every other key
    out = _feed(ks, 0.5, first + 2 * every)
    assert out[-1] == 3 and not any(o is not None for o in out[:-1])   # the key_confirm-th decision
    _, ks = _fake_search({3: 0.47})                           # 6 %: consistent but not clear -> suggestion
    out = _feed(ks, 0.5, first + (p.key_suggest_confirm - 1) * every - 1)
    assert ks.lead is None
    out += _feed(ks, 0.5, 1)
    assert ks.lead == 3 and not any(o is not None for o in out)
    _, ks = _fake_search({3: 0.49})                           # 2 %: no suggestion
    _feed(ks, 0.5, first + 10 * every)
    assert ks.lead is None
    _, ks = _fake_search({3: 0.47})                           # the current key fits best: nothing
    out = _feed(ks, 0.45, first + 10 * every)
    assert ks.lead is None and not any(o is not None for o in out)


def test_key_suggestion_outlives_the_search_until_the_operator_acts():
    rng = np.random.default_rng(45)
    C, M = _chords(rng)
    trk = LiveTracker(_chord_ref(C, M, 60 * 25), FPS, TrackerParams.for_mode("mix"))
    trk.start(0.0)
    ks = trk._keys
    ks.started, ks.room, ks.lead = True, -90.0, 3
    ks.steps = ks.max_steps - 1                               # the search ends with the next evidence frame
    _run(trk, *_chord_live(C, M, 1, 0, rng))
    assert trk._keys is None and trk.state.key_suggest == 3 and trk.state.transpose == 0
    trk.set_auto_key(True)                                    # a new search drops the old suggestion
    assert trk.state.key_suggest is None
    trk.start(0.0)
    trk._keys.lead = 5
    assert trk.state.key_suggest == 5
    trk.set_transpose(5)                                      # the operator accepted it
    assert trk.state.key_suggest is None and not trk.state.key_auto


def _vocal_chord_song(C, M, rng, n_phr=10, voice=100, gap=50):
    """Phrases of chords separated by silence (uniform chroma, constant timbre)."""
    sil = np.r_[np.full(12, np.sqrt(0.8 / 12)), np.sqrt(0.2), 0.0, 0.0, 0.0].astype(np.float32)
    feats, levels, onsets, pos = [], [], [], 0
    for _ in range(n_phr):
        feats.append(np.repeat(sil[None], gap, 0))
        levels.append(np.full(gap, -90.0))
        onsets.append(pos + gap)
        idx = rng.integers(0, len(C), voice // _SEG).repeat(_SEG)
        feats.append(_chord_feats(C, M, idx))
        levels.append(np.full(voice, -20.0))
        pos += gap + voice
    return np.concatenate(feats), np.concatenate(levels), onsets, sil


def test_auto_key_vocal_take_with_pauses_and_a_transposition():
    """Vocal mode: the voice pauses between phrases (coasting, relocks); the hypotheses are re-anchored
    with every relock and the transposition is still found, then followed under the right key."""
    rng = np.random.default_rng(46)
    C, M = _chords(rng)
    R, lv, onsets, sil = _vocal_chord_song(C, M, rng)
    ref = FeatureFrames(R, lv.astype(np.float32), _GROUPS)
    trk = LiveTracker(ref, FPS, TrackerParams.for_mode("vocal"), line_starts=[(o + 1) / FPS for o in onsets])
    trk.start(0.0)
    k = 2
    feats, levels, gt = [], [], []
    for j, o in enumerate(onsets):
        n_g = 50 + (15 if j % 2 else -15)                      # the singer pauses longer / shorter
        feats.append(np.repeat(sil[None], n_g, 0))
        levels.append(np.full(n_g, -90.0))
        gt.append(np.full(n_g, np.nan))
        seg = np.arange(o, o + 100)
        x = R[seg].copy()
        x[:, :12] = np.roll(x[:, :12], k, axis=1)
        feats.append(x + 0.05 * rng.standard_normal(x.shape).astype(np.float32) / 4)
        levels.append(np.full(100, -20.0))
        gt.append(seg.astype(float))
    feats, levels, gt = np.concatenate(feats).astype(np.float32), np.concatenate(levels), np.concatenate(gt)
    states = _run(trk, feats, levels, chunk=1)
    assert trk.state.transpose == k and trk.key_switches == 1 and trk.relocations == 0
    if trk._keys is not None:
        assert len(_anchors(trk)) == 1
    est = np.array([s.ref_time for s in states])
    late = ~np.isnan(gt) & (np.arange(len(gt)) > len(gt) // 2)
    assert np.mean(np.abs(est[late] - (gt[late] + 1) / FPS) <= 0.3) > 0.85


def test_hold_keeps_the_latency_compensated_position():
    """The reported ref_time includes rate x latency; Hold (rate 0) must keep what was shown instead of
    stepping back by the compensation (the prompter's wipe would un-wipe)."""
    rng = np.random.default_rng(47)
    R = _ref_feats(rng, 1000)
    lv = np.full(len(R), -20.0)
    trk = LiveTracker(_frames(R, lv), FPS, TrackerParams(latency_s=0.3))
    trk.start(0.0)
    sts = _run(trk, R[:250], lv[:250])
    shown = sts[-1].ref_time
    assert shown == pytest.approx((trk._out + 1) / FPS + 0.3 * sts[-1].tempo) and sts[-1].tempo > 0.5
    trk.hold(True)
    st = trk.push(_frames(R[250:300], lv[250:300]))
    assert st.status == HOLD and st.tempo == 0.0 and st.ref_time == pytest.approx(shown)
    trk.hold(False)
    sts = _run(trk, R[300:400], lv[300:400])
    assert min(s.ref_time for s in sts) >= shown - 1e-9


def _verses(seed, n_verse=5, phr=4, voice=100, gap=25, w_timbre=0.5):
    """Vocal reference: verses that share the melody (chord sequence) but not the words (timbre group),
    phrases separated by silence. Returns feats, levels, phrase onsets."""
    rng = np.random.default_rng(seed)
    C, _ = _chords(rng)
    melody = [rng.integers(0, len(C), voice // _SEG).repeat(_SEG) for _ in range(phr)]
    sil = np.r_[np.full(12, np.sqrt(0.8 / 12)), np.sqrt(0.2), 0.0, 0.0, 0.0].astype(np.float32)
    feats, levels, onsets, pos = [], [], [], 0
    for _ in range(n_verse):
        for j in range(phr):
            feats.append(np.repeat(sil[None], gap, 0))
            levels.append(np.full(gap, -90.0))
            onsets.append(pos + gap)
            words = _unit(rng.standard_normal((voice // _SEG, 4)).repeat(_SEG, 0))
            feats.append(np.concatenate([np.sqrt(1 - w_timbre) * _unit(C[melody[j]]), np.sqrt(w_timbre) * words], 1))
            levels.append(np.full(voice, -20.0))
            pos += gap + voice
    return np.concatenate(feats).astype(np.float32), np.concatenate(levels), onsets


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("skip", [True, False])
def test_vocal_relocates_soon_after_a_skipped_verse(seed, skip):
    """The singers skip two verses of the same melody. On the wrong verse the confidence of a vocal tracker
    stays moderate (about 0.4-0.6) and every phrase gap relocked it, which restarted the relocation's
    'since the last anchor' count: it stayed on the wrong verse for 15-30 s or for good. Now it relocates
    within a few seconds; without a skip nothing is relocated."""
    R, lv, onsets = _verses(seed)
    ref = FeatureFrames(R, lv.astype(np.float32), _GROUPS)
    trk = LiveTracker(ref, FPS, TrackerParams.for_mode("vocal"), line_starts=[(o + 1) / FPS for o in onsets])
    trk.start(0.0)
    a, b = (onsets[4] - 25, onsets[12] - 25) if skip else (0, 0)        # verse 0, then verse 3 on
    idx = np.r_[np.arange(0, a), np.arange(b, len(R))]
    rng = np.random.default_rng(100 + seed)
    live = R[idx] + 0.05 * rng.standard_normal((len(idx), 16)).astype(np.float32) / 4
    est = np.array([s.ref_time for s in _run(trk, live, lv[idx], chunk=1)])
    err = np.abs(est - (idx + 1) / FPS)
    voiced = lv[idx] > -50
    late = voiced & (np.arange(len(idx)) >= a + 10 * FPS)
    assert err[late].max() < 0.5                              # on the right verse 10 s after the skip
    assert trk.relocations == int(skip)
