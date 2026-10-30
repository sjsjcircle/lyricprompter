"""Synthetic music (make_score / render_song / tempo maps) and the synthetic-live helpers copied from
lyricprompter's evaluation (unchanged; compared with the originals when the parent checkout is present)."""
from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np
import pytest

from lyricprompter_oscsender import synth as S

SR = 16000
UPSTREAM = Path(__file__).resolve().parents[2] / "src" / "lyricprompter" / "eval" / "track_eval.py"
COPIED = ["SynthSpec", "synth_time_map", "pv_warp", "_pink", "degrade", "make_synthetic", "ref_to_live",
          "map_events"]
TEMPO_MAP = ([0.0, 20.0, 40.0, 60.0], [0.0, 21.5, 44.0, 63.0])


def first_nonzero(y: np.ndarray) -> int:
    return int(np.flatnonzero(np.abs(y) > 0)[0])


def last_nonzero(y: np.ndarray) -> int:
    return int(np.flatnonzero(np.abs(y) > 0)[-1])


# ------------------------------------------------------------------------------------------ score


def test_score_is_deterministic_per_seed():
    a, b, c = S.make_score(30, 7), S.make_score(30, 7), S.make_score(30, 8)
    assert np.array_equal(a.notes, b.notes) and a.bpm == b.bpm and a.rests == b.rests
    assert not (len(a.notes) == len(c.notes) and np.array_equal(a.notes, c.notes))
    ya, yb, yc = S.render_song(a), S.render_song(b), S.render_song(c)
    assert ya.dtype == np.float32 and np.array_equal(ya, yb) and not np.array_equal(ya, yc[:len(ya)])


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_score_content(seed):
    sc = S.make_score(60, seed)
    n = sc.notes
    assert sc.duration == 60 and 100 <= sc.bpm <= 124
    assert np.all(np.diff(n["start"]) >= 0) and np.all(n["dur"] > 0)
    assert np.all(n["start"] + n["dur"] <= 60 + 1e-9) and n["start"].min() >= 0
    counts = {v: int(np.sum(n["voice"] == i)) for i, v in enumerate(S.VOICES)}
    assert counts["pad"] >= 100 and counts["bass"] >= 50 and counts["lead"] >= 40
    assert counts["kick"] >= 40 and counts["snare"] >= 40 and counts["hat"] >= 100
    # harmony changes every 0.5-1.5 s (rests excepted), with real changes of chord
    starts = np.unique(n["start"][n["voice"] == 0])
    gaps = np.diff(starts)
    in_rest = np.array([any(t < r1 and t + g > r0 for r0, r1 in sc.rests) for t, g in zip(starts, gaps)])
    assert np.all((gaps[~in_rest] >= 0.5 - 1e-6) & (gaps[~in_rest] <= 1.5 + 1e-6))
    chords = {tuple(sorted(n["midi"][(n["voice"] == 0) & (n["start"] == t)] % 12)) for t in starts}
    assert len(chords) >= 5
    lead = n["midi"][n["voice"] == 2]
    assert len(np.unique(lead)) >= 6 and lead.min() >= 62 and lead.max() <= 81
    # a couple of short full-band rests: nothing sounds inside them
    assert len(sc.rests) == 2
    for r0, r1 in sc.rests:
        assert 0.4 <= r1 - r0 <= 1.3
        assert not np.any((n["start"] < r1 - 1e-9) & (n["start"] + n["dur"] > r0 + 1e-9))
    assert len(S.make_score(20, seed).rests) == 0


def test_score_level_and_rests_are_silent():
    for seed in range(3):
        sc = S.make_score(60, seed)
        y = S.render_song(sc)
        assert len(y) == 60 * SR
        rms = 20 * np.log10(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
        peak = 20 * np.log10(np.max(np.abs(y)))
        assert -22 < rms < -14 and peak < -1.0
        for r0, r1 in sc.rests:
            seg = y[round((r0 + 0.1) * SR):round(r1 * SR)]       # after the longest release (80 ms)
            assert np.max(np.abs(seg)) == 0.0


def test_bad_score_length():
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            S.make_score(bad, 0)


def test_speed():
    t0 = time.perf_counter()
    sc = S.make_score(60, 3)
    y = S.render_song(sc, tempo_map=TEMPO_MAP, transpose=1)
    dt = time.perf_counter() - t0
    assert len(y) == 63 * SR
    assert dt < 1.0, f"60 s song took {dt:.2f} s"


# ------------------------------------------------------------------------------------- tempo maps


def test_tempo_map_helpers():
    assert S.score_to_perf(20.0, TEMPO_MAP) == pytest.approx(21.5)
    assert S.score_to_perf(30.0, TEMPO_MAP) == pytest.approx(21.5 + 10 * 22.5 / 20)
    assert S.perf_to_score(44.0, TEMPO_MAP) == pytest.approx(40.0)
    assert S.score_to_perf(70.0, TEMPO_MAP) == pytest.approx(63.0 + 10 * 19.0 / 20)     # extrapolated
    assert S.perf_to_score(-1.0, TEMPO_MAP) == pytest.approx(-20.0 / 21.5)                # first slope
    t = np.linspace(-5, 80, 1001)
    np.testing.assert_allclose(S.perf_to_score(S.score_to_perf(t, TEMPO_MAP), TEMPO_MAP), t, atol=1e-9)
    assert isinstance(S.score_to_perf(3.0, TEMPO_MAP), float) and isinstance(S.score_to_perf(3.0), float)
    np.testing.assert_array_equal(S.score_to_perf(t), t)
    for bad in [([0, 1], [0]), ([0, 1, 1], [0, 1, 2]), ([0, 1], [1, 0]), ([0], [0]), ([0, np.nan], [0, 1]),
                "x"]:
        with pytest.raises(ValueError):
            S.score_to_perf(1.0, bad)


@pytest.mark.parametrize("voice,midi", [("lead", 72.0), ("pad", 60.0), ("bass", 40.0), ("kick", 0.0),
                                        ("snare", 0.0), ("hat", 0.0)])
def test_note_onset_and_end_follow_the_map(voice, midi):
    notes = S.make_notes([(10.0, 1.0, midi, 0.2, voice)])
    sc = S.Score(notes=notes, duration=20.0)
    tm = ([0.0, 8.0, 20.0], [0.0, 9.0, 25.0])                 # 1.125x then 1.33x slower
    y = S.render_song(sc, tempo_map=tm)
    on, off = S.score_to_perf(10.0, tm), S.score_to_perf(11.0, tm)
    assert len(y) == 25 * SR
    assert abs(first_nonzero(y) - round(on * SR)) <= 1
    if voice in ("lead", "pad", "bass"):                      # held to the mapped end, then released
        rel = {"lead": 0.05, "pad": 0.08, "bass": 0.05}[voice]
        assert round(off * SR) <= last_nonzero(y) <= round((off + rel) * SR)
        assert np.max(np.abs(y[round((off - 0.05) * SR):round(off * SR)])) > 0.02
    else:                                                     # drums decay but never outlive the note
        assert last_nonzero(y) <= round((off + 0.006) * SR)


def test_real_score_notes_land_at_mapped_times():
    sc = S.make_score(60, 5)
    n = sc.notes
    for voice in ("lead", "kick", "pad"):
        pick = n[n["voice"] == S.VOICES.index(voice)][7]
        single = S.Score(notes=pick.reshape(1), duration=sc.duration)
        y = S.render_song(single, tempo_map=TEMPO_MAP)
        assert abs(first_nonzero(y) - round(S.score_to_perf(pick["start"], TEMPO_MAP) * SR)) <= 1


def test_pure_shift_map_delays_the_reference_exactly():
    sc = S.make_score(30, 2)
    ref = S.render_song(sc)
    shift = 1.5                                               # 24000 samples, durations unchanged
    take = S.render_song(sc, tempo_map=([0.0, 30.0], [shift, 30.0 + shift]))
    k = round(shift * SR)
    assert len(take) == len(ref) + k
    assert not take[:k].any()
    np.testing.assert_allclose(take[k:], ref, atol=1e-6)


def test_transpose_moves_pitch_only():
    notes = S.make_notes([(0.0, 1.0, 69.0, 0.2, "lead"), (0.0, 0.3, 0.0, 0.2, "kick")])
    sc = S.Score(notes=notes, duration=1.0)
    lead_only = S.Score(notes=notes[notes["voice"] == 2], duration=1.0)
    spec = np.abs(np.fft.rfft(S.render_song(lead_only, transpose=2)))
    f0 = np.argmax(spec) * SR / (2 * (len(spec) - 1))
    assert f0 == pytest.approx(440 * 2 ** (2 / 12), abs=2.0)
    kick_only = S.Score(notes=notes[notes["voice"] == 3], duration=1.0)
    assert np.array_equal(S.render_song(kick_only, transpose=5), S.render_song(kick_only))
    assert len(S.render_song(sc, transpose=-3)) == SR


def test_map_before_zero_is_rejected():
    sc = S.Score(notes=S.make_notes([(0.0, 0.5, 60.0, 0.1, "pad")]), duration=1.0)
    with pytest.raises(ValueError):
        S.render_song(sc, tempo_map=([1.0, 2.0], [0.0, 2.0]))     # score 0 -> perf -1


def test_empty_score_renders_silence():
    sc = S.Score(notes=S.make_notes([]), duration=2.0)
    y = S.render_song(sc, sr=8000)
    assert y.shape == (16000,) and not y.any()


# --------------------------------------------------------------------------- tracking on synthetic songs


@pytest.mark.parametrize("style", ["classic", "pop"])
def test_tracker_locks_on_a_rerendered_take(style):
    """The point of the generator: lyricprompter's tracker follows a tempo-mapped take of it closely."""
    from lyricprompter_oscsender.features.extractor import FeatureConfig, FeatureExtractor, compute_offline
    from lyricprompter_oscsender.tracking.tracker import LiveTracker, TrackerParams

    tm = ([0.0, 15.0, 30.0, 40.0], [0.0, 16.0, 33.5, 43.0])
    sc = S.make_score(40, 11)
    ref, live = S.render_song(sc), S.render_song(sc, tempo_map=tm)
    cfg = FeatureConfig.for_mode("mix", tuning=0.0)
    trk = LiveTracker(compute_offline(ref, cfg), cfg.fps, TrackerParams.for_mode("mix", style))
    trk.start(0.0)
    ext = FeatureExtractor(cfg)
    t_live, pos = [], []
    for i in range(0, len(live), 1024):
        st = trk.push(ext.push(live[i:i + 1024]))
        t_live.append(min(i + 1024, len(live)) / SR)
        pos.append(st.ref_time)
    t_live, pos = np.array(t_live), np.array(pos)
    err = np.abs(pos - S.perf_to_score(t_live, tm))[t_live > 4.0]
    assert np.median(err) < 0.08 and np.percentile(err, 90) < 0.25 and err.max() < 1.0


# ------------------------------------------------------------------------- copied synthetic-live helpers


def _sources(path: Path, names: list[str]) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            out[node.name] = "\n".join(text.splitlines()[start - 1:node.end_lineno])
    return out


@pytest.mark.skipif(not UPSTREAM.is_file(), reason="lyricprompter checkout not found next to this project")
def test_copied_helpers_match_lyricprompter():
    ours = _sources(Path(S.__file__), COPIED)
    theirs = _sources(UPSTREAM, COPIED)
    assert set(theirs) == set(COPIED) == set(ours)
    for name in COPIED:
        assert ours[name] == theirs[name], f"synth.{name} differs from lyricprompter's"


def test_make_synthetic_ground_truth_on_a_synthetic_song():
    y = S.render_song(S.make_score(8, 4))
    spec = S.SynthSpec(seed=3, tempo=(0.9, 1.1), seg_s=(2.0, 3.0), snr_db=30.0, rt60_s=0.3, lead_s=0.5)
    live, live_pts, ref_pts = S.make_synthetic(y, SR, spec)
    assert live.dtype == np.float32 and np.all(np.isfinite(live))
    assert len(live) == pytest.approx(round(live_pts[-1] * SR), abs=2)
    assert ref_pts[0] == 0 and ref_pts[-1] == pytest.approx(8.0)
    assert np.all(np.diff(live_pts) > 0) and np.all(np.diff(ref_pts) >= 0)
    gl, gr = S.map_events(np.array([1.0, 4.0, 7.5]), live_pts, ref_pts)
    np.testing.assert_allclose(gr, [1.0, 4.0, 7.5])
    np.testing.assert_allclose(gl, S.ref_to_live(gr, live_pts, ref_pts))
    assert np.all(gl > spec.lead_s)
