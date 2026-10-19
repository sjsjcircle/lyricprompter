"""Pop tracking style (TrackerParams.style == "pop"): steady-tempo prediction corrected by the DP.

The classic style is covered by tests/test_tracker.py; here: pop keeps moving through what the DP cannot
follow (a dropout in the mix, a line the singer drops), waits where classic waits before it locked, does
not jump on a short ad-lib that matches another place but still follows a real skip, and the live
set_style switch."""
from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

from lyricprompter.features.extractor import FeatureFrames
from lyricprompter.tracking.tracker import (
    COASTING,
    STYLES,
    TRACKING,
    LiveTracker,
    TrackerParams,
    style_fields,
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


def _frames(feats, levels):
    return FeatureFrames(np.asarray(feats, np.float32), np.asarray(levels, np.float32))


def _run(trk, feats, levels, chunk=2):
    out = []
    for i in range(0, len(feats), chunk):
        st = trk.push(_frames(feats[i:i + chunk], levels[i:i + chunk]))
        out.extend([st] * len(feats[i:i + chunk]))
    return out


def _mix_tracker(R, style):
    trk = LiveTracker(_frames(R, np.full(len(R), -20.0)), FPS, TrackerParams.for_mode("mix", style))
    trk.start(0.0)
    return trk


# ----------------------------------------------------------------------------- params


def test_for_mode_styles_and_style_fields():
    for mode in ("mix", "vocal"):
        classic, pop = TrackerParams.for_mode(mode), TrackerParams.for_mode(mode, "pop")
        assert classic == TrackerParams.for_mode(mode, "classic") and classic.style == "classic"
        assert pop.style == "pop" and pop.mode == mode
        # the presets differ only in the fields set_style switches
        diff = {f.name for f in fields(TrackerParams) if getattr(classic, f.name) != getattr(pop, f.name)}
        assert diff <= set(style_fields(mode, "pop")) and "style" in diff
        assert replace(classic, **style_fields(mode, "pop")) == pop
        assert replace(pop, **style_fields(mode, "classic")) == classic
    assert STYLES == ("classic", "pop")
    with pytest.raises(ValueError, match="style"):
        TrackerParams.for_mode("mix", "jazz")
    rng = np.random.default_rng(0)
    R = _ref_feats(rng, 100)
    with pytest.raises(ValueError, match="style"):
        LiveTracker(_frames(R, np.full(100, -20.0)), FPS, TrackerParams(style="rock"))


@pytest.mark.parametrize("mode", ["mix", "vocal"])
def test_classic_ignores_the_pop_fields(mode):
    """The classic path never reads pop_* (its numbers stay those of the code before the styles)."""
    rng = np.random.default_rng(1)
    R = _ref_feats(rng, 60 * 25)
    lv = np.where((np.arange(len(R)) // 100) % 4 == 3, -90.0, -20.0)          # vocal: pauses
    gt = np.arange(0, len(R) - 30, 1.07)
    L, lvl = _live_from(R, gt, rng), lv[np.round(gt).astype(int)]
    odd = {f.name: (3.0 if f.type in (float, "float") else getattr(TrackerParams(), f.name))
           for f in fields(TrackerParams) if f.name.startswith("pop_")}
    runs = []
    for p in (TrackerParams.for_mode(mode), replace(TrackerParams.for_mode(mode), **odd)):
        trk = LiveTracker(_frames(R, lv), FPS, p)
        trk.start(0.0)
        runs.append([(s.ref_time, s.status, s.tempo, s.confidence) for s in _run(trk, L, lvl)])
    assert runs[0] == runs[1]


# ----------------------------------------------------------------------------- mix


def test_pop_follows_a_steady_tempo_and_reports_it():
    rng = np.random.default_rng(2)
    R = _ref_feats(rng, 100 * 25)
    rho = 1.06
    gt = np.arange(0, (len(R) - 30) / rho) * rho
    trk = _mix_tracker(R, "pop")
    states = _run(trk, _live_from(R, gt, rng), np.full(len(gt), -20.0))
    est = np.array([s.ref_time for s in states])
    err = np.abs(est - (gt + 1) / FPS)
    assert np.mean(err <= 0.2) > 0.95 and err[10 * 25:].max() < 0.3
    assert np.all(np.diff(est) >= 0)                                       # monotonic
    assert all(s.status == TRACKING for s in states)
    assert abs(states[-1].tempo / rho - 1) < 0.03                          # the steady estimate is reported
    assert trk.snaps == 0 and trk.relocations == 0


def test_pop_keeps_moving_through_a_mix_dropout():
    """4 s the DP cannot follow (the live is unrelated to the reference, at full level): the display goes on
    at the tempo and is on time when the music is recognisable again."""
    rng = np.random.default_rng(3)
    R = _ref_feats(rng, 80 * 25)
    rho = 0.95
    gt = np.arange(0, (len(R) - 30) / rho) * rho
    L = _live_from(R, gt, rng)
    a, b = 30 * 25, 34 * 25
    L[a:b] = _unit(rng.standard_normal((b - a, DIM)))
    trk = _mix_tracker(R, "pop")
    est = np.array([s.ref_time for s in _run(trk, L, np.full(len(gt), -20.0))])
    err = np.abs(est - (gt + 1) / FPS)
    assert err[a:b + 50].max() < 0.3                                      # through the dropout and after it
    assert np.mean(np.diff(est[a:b])) * FPS == pytest.approx(rho, abs=0.08)   # moving at the tempo


@pytest.mark.parametrize("seed", [0, 1])
def test_pop_does_not_jump_on_an_adlib_that_matches_another_place(seed):
    """4 s of another part of the song (a line sung in the wrong place) while the band goes on: classic
    re-localises to it (and back), pop keeps the time; the relocation needs a confirmation 5 s later."""
    rng = np.random.default_rng(seed)
    R = _ref_feats(rng, 150 * 25)
    rho = 1.04
    gt = np.arange(0, 100 * 25 / rho) * rho
    L = _live_from(R, gt, rng)
    a, n = 40 * 25, 4 * 25
    L[a:a + n] = _live_from(R, np.arange(n) + 110 * 25, rng)
    err = {}
    for style in STYLES:
        trk = _mix_tracker(R, style)
        est = np.array([s.ref_time for s in _run(trk, L, np.full(len(L), -20.0))])
        err[style] = (np.abs(est - (gt + 1) / FPS), trk.relocations)
    assert err["classic"][1] >= 1 and err["classic"][0][a:a + n + 250].max() > 10.0
    assert err["pop"][1] == 0 and err["pop"][0][a:].max() < 0.3


def test_pop_still_follows_a_real_skip():
    """The band skips 40 s: the relocation is confirmed after pop_reloc_confirm_s and adopted."""
    rng = np.random.default_rng(4)
    R = _ref_feats(rng, 150 * 25)
    rho = 0.96
    gt = np.r_[np.arange(0, 30 * 25 / rho) * rho, 70 * 25 + np.arange(0, 40 * 25 / rho) * rho]
    trk = _mix_tracker(R, "pop")
    est = np.array([s.ref_time for s in _run(trk, _live_from(R, gt, rng), np.full(len(gt), -20.0))])
    a = round(30 * 25 / rho)
    err = np.abs(est - (gt + 1) / FPS)
    assert trk.relocations == 1
    assert err[a + 15 * 25:].max() < 0.3                                  # on target 15 s after the skip


def test_pop_does_not_run_ahead_before_the_music():
    """Mix: until the DP follows the music (confidently, past the start) pop publishes exactly what classic
    does, so room noise before the band does not move the display at the tempo."""
    rng = np.random.default_rng(5)
    R = _ref_feats(rng, 60 * 25)
    noise = _unit(rng.standard_normal((8 * 25, DIM)))
    gt = np.arange(0, 40 * 25, 1.0)
    L = np.concatenate([noise, _live_from(R, gt, rng)])
    lv = np.r_[np.full(len(noise), -70.0), np.full(len(gt), -20.0)]
    est = {}
    for style in STYLES:
        trk = _mix_tracker(R, style)
        est[style] = np.array([s.ref_time for s in _run(trk, L, lv)])
    n = len(noise)
    np.testing.assert_array_equal(est["pop"][:n], est["classic"][:n])
    assert np.abs(est["pop"][n + 10 * 25:] - (gt[10 * 25:] + 1) / FPS).max() < 0.3


# ----------------------------------------------------------------------------- vocal


def _vocal_song(rng, n_lines=5, voice_s=4.0, gap_s=1.0):
    """[gap][voice] x n_lines; silent frames are a fixed 'uniform' vector."""
    sil = _unit(np.ones(DIM))
    feats, levels, onsets = [], [], []
    for _ in range(n_lines):
        n_g, n_v = round(gap_s * FPS), round(voice_s * FPS)
        feats.append(np.repeat(sil[None], n_g, 0))
        levels.append(np.full(n_g, -90.0))
        onsets.append(sum(len(x) for x in levels))
        feats.append(_ref_feats(rng, n_v))
        levels.append(np.full(n_v, -20.0))
    return np.concatenate(feats), np.concatenate(levels), onsets, sil


def _sing(R, lv, sil, rng, drop=(), rho=1.0, lead_s=0.0):
    """Live at a steady tempo rho (ref frames per live frame); lines in `drop` are not sung (silence)."""
    gt = np.arange(0, (len(R) - 2) / rho) * rho
    L = _live_from(R, gt, rng, 0.4)
    lvl = lv[np.round(gt).astype(int)].copy()
    for k in drop:
        m = (gt >= drop[k][0]) & (gt < drop[k][1])
        L[m], lvl[m] = sil, -90.0
    n = round(lead_s * FPS)
    gt = np.r_[np.full(n, np.nan), gt]
    return np.concatenate([np.repeat(sil[None], n, 0), L]), np.r_[np.full(n, -90.0), lvl], gt


def test_pop_vocal_coasts_through_a_dropped_line_where_classic_waits():
    rng = np.random.default_rng(6)
    R, lv, onsets, sil = _vocal_song(rng)
    starts = [(o + 1) / FPS for o in onsets]
    L, lvl, gt = _sing(R, lv, sil, rng, drop={2: (onsets[2] - 5, onsets[3] - 25)}, rho=1.03)
    i3 = int(np.flatnonzero(gt >= onsets[3])[0])                       # live frame where line 3 starts
    i2 = int(np.flatnonzero(gt >= onsets[2])[0])
    res = {}
    for style in STYLES:
        trk = LiveTracker(_frames(R, lv), FPS, TrackerParams.for_mode("vocal", style), line_starts=starts)
        trk.start(0.0)
        res[style] = _run(trk, L, lvl, chunk=1)
    classic = np.array([s.ref_time for s in res["classic"]])
    pop = np.array([s.ref_time for s in res["pop"]])
    truth = (gt + 1) / FPS
    # classic: parked at the start of the dropped line until the voice comes back (the user's complaint)
    assert classic[i2 + 25:i3].max() <= starts[2] + 1e-6
    assert res["classic"][i3 - 5].tempo == 0.0
    # pop: through the dropped line at the tempo, on time when line 3 starts and after it
    assert np.abs(pop[i2:i3] - truth[i2:i3]).max() < 0.3
    assert res["pop"][i3 - 5].status == COASTING and res["pop"][i3 - 5].tempo == pytest.approx(1.03, abs=0.04)
    assert np.abs(pop[i3:] - truth[i3:]).max() < 0.3


def test_pop_vocal_waits_for_the_first_voice():
    """Before the voice was followed (start, or a jump into silence) pop waits at the line like classic."""
    rng = np.random.default_rng(7)
    R, lv, onsets, sil = _vocal_song(rng, gap_s=2.0)
    starts = [(o + 1) / FPS for o in onsets]
    L, lvl, gt = _sing(R, lv, sil, rng, lead_s=6.0)                      # the singer starts 6 s late
    trk = LiveTracker(_frames(R, lv), FPS, TrackerParams.for_mode("vocal", "pop"), line_starts=starts)
    trk.start(0.0)
    sts = _run(trk, L[:7 * 25], lvl[:7 * 25], chunk=1)
    assert sts[-1].status == COASTING and sts[-1].ref_time == pytest.approx(starts[0]) and sts[-1].tempo == 0.0
    sts = _run(trk, L[7 * 25:], lvl[7 * 25:], chunk=1)
    err = np.abs(np.array([s.ref_time for s in sts]) - (gt[7 * 25:] + 1) / FPS)
    sung = gt[7 * 25:] >= onsets[0]                                     # (NaN before the singer: False)
    assert np.mean(err[sung] <= 0.3) > 0.95
    # a jump into a silent stretch (the singer stopped): waits at the target again until the voice
    n = 3 * 25
    _run(trk, np.repeat(sil[None], 25, 0), np.full(25, -90.0), chunk=1)
    trk.jump(starts[2])
    sts = _run(trk, np.repeat(sil[None], n, 0), np.full(n, -90.0), chunk=1)
    assert all(s.ref_time == pytest.approx(starts[2]) for s in sts) and sts[-1].tempo == 0.0


# ----------------------------------------------------------------------------- set_style


def test_set_style_switches_live_and_keeps_the_rest():
    rng = np.random.default_rng(8)
    R, lv, onsets, sil = _vocal_song(rng)
    starts = [(o + 1) / FPS for o in onsets]
    L, lvl, gt = _sing(R, lv, sil, rng, drop={2: (onsets[2] - 5, onsets[3] - 25)})
    trk = LiveTracker(_frames(R, lv), FPS, replace(TrackerParams.for_mode("vocal"), latency_s=0.1, reloc_every_s=0.7),
                      line_starts=starts)
    trk.start(0.0)
    i2 = int(np.flatnonzero(gt >= onsets[2])[0])
    _run(trk, L[:i2 + 25], lvl[:i2 + 25], chunk=1)                     # classic: parked at line 2
    parked = trk.state
    assert parked.status == COASTING and parked.tempo == 0.0
    trk.set_style("pop")
    assert trk.params.style == "pop" and trk.params.latency_s == 0.1 and trk.params.reloc_every_s == 0.7
    sts = _run(trk, L[i2 + 25:i2 + 50], lvl[i2 + 25:i2 + 50], chunk=1)
    assert sts[-1].ref_time > parked.ref_time + 0.8 and sts[-1].tempo > 0.8   # runs on at the tempo
    trk.set_style("classic")                                          # back: waits at the next line start
    i3 = int(np.flatnonzero(gt >= onsets[3])[0])
    sts = _run(trk, L[i2 + 50:i3 - 2], lvl[i2 + 50:i3 - 2], chunk=1)
    assert max(s.ref_time for s in sts) <= starts[3] + 1e-6
    trk.set_style("classic")                                          # no-op
    with pytest.raises(ValueError, match="style"):
        trk.set_style("metal")
