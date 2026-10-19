from __future__ import annotations

import numpy as np
import pytest

from lyricprompter.config import ref_repo_dir
from lyricprompter.eval import track_eval as E
from lyricprompter.features.extractor import FeatureConfig, compute_offline
from lyricprompter.tracking.tracker import TrackerParams

SR = 16000


def test_parse_lrc(tmp_path):
    p = tmp_path / "x.lrc"
    p.write_text("[ti:x]\n\n[00:06.82]E  <00:07.87>del  <00:08.33>weiss,  <00:10.00> e\n"
                 "[01:02.50]Small  <01:03.00> and\n", encoding="utf-8")
    toks = E.parse_lrc(p)
    assert [t["text"] for t in toks] == ["E", "del", "weiss,", "e", "Small", "and"]
    assert toks[0]["t"] == pytest.approx(6.82) and toks[4]["t"] == pytest.approx(62.5)
    assert [t["word_start"] for t in toks] == [True, False, False, True, True, True]
    assert E.lrc_line_starts(toks) == [pytest.approx(6.82), pytest.approx(62.5)]


def test_score_csv_to_ref_gt_and_line_groups(tmp_path):
    (tmp_path / "s.csv").write_text("start,pitch,pitchclass,pitchname,offset,instrument,lyric,syllabic\n"
                                    "0.5,60,0,C4,1.0,voice,꽃,begin\n0.75,60,0,C4,1.5,voice,,none\n"
                                    "1.0,62,2,D4,2.0,voice,밭,end\n", encoding="utf-8")
    # path rows (score frame, ref frame) at 25 fps: ref runs at half speed, with a non-monotone step
    wp = np.array([[0, 0], [0, 1], [25, 50], [50, 100]], float)
    np.savetxt(tmp_path / "wp.csv", wp, delimiter=",")
    g = E.score_csv_to_ref_gt(tmp_path / "s.csv", tmp_path / "wp.csv")
    assert [x["text"] for x in g] == ["꽃", "밭"]
    assert g[0]["t_score"] == pytest.approx(0.5) and g[0]["t_ref"] == pytest.approx(1.0)
    assert g[1]["t_ref"] == pytest.approx(2.0)
    (tmp_path / "l.txt").write_text("꽃밭에 앉아\n\n고운 빛\n", encoding="utf-8")
    assert E.group_syllables_by_lines(tmp_path / "l.txt", 7) == [0, 5]


def test_prompter_metrics_perfect_and_delayed():
    gt_live = np.array([1.0, 2.0, 3.0, 4.0])
    gt_ref = gt_live * 1.5
    t = np.arange(0, 5, 0.04)
    m = E.prompter_metrics(t, np.interp(t, gt_live, gt_ref), gt_live, gt_ref)
    assert m["pos_AAE_ms"] < 61 and m["PCO@200"] == 100.0 and m["lost_frac"] == 0.0
    late = 1.5 * (t - 0.2)                              # published 0.2 s late
    m = E.prompter_metrics(t, late, gt_live, gt_ref)
    assert m["pos_bias_ms"] == pytest.approx(-300, abs=61)
    assert m["lat_median_ms"] == pytest.approx(200, abs=41)
    far = np.interp(t, gt_live, gt_ref) + 3.0
    assert E.prompter_metrics(t, far, gt_live, gt_ref)["lost_frac"] == 1.0


def test_synth_time_map_is_piecewise_and_in_range():
    spec = E.SynthSpec(seed=3, lead_s=1.0)
    lp, rp = E.synth_time_map(60.0, spec, np.random.default_rng(3))
    assert lp[0] == rp[0] == 0.0 and lp[1] == 1.0 and rp[1] == 0.0
    assert rp[-1] == pytest.approx(60.0)
    rho = np.diff(rp[1:]) / np.diff(lp[1:])
    assert (rho >= 0.8 - 1e-9).all() and (rho <= 1.25 + 1e-9).all()
    seg = np.diff(rp[1:])
    assert (seg[:-1] >= 2.0).all() and (seg <= 8.0 + 1e-9).all()
    np.testing.assert_allclose(E.ref_to_live(rp[2:], lp, rp), lp[2:])


@pytest.mark.parametrize("seed", range(5))
def test_ref_to_live_inverts_the_map_in_the_first_segment_after_a_lead_in(seed):
    """The flat lead-in (ref stays 0 while live advances) used to be inverted from its first point,
    making first-segment GT live times up to lead_s too early."""
    spec = E.SynthSpec(seed=seed, lead_s=1.0)
    lp, rp = E.synth_time_map(60.0, spec, np.random.default_rng(seed))
    ev = np.linspace(0.05, rp[2] - 0.05, 9)                      # inside the first music segment
    gl = E.ref_to_live(ev, lp, rp)
    assert (gl >= 1.0).all()                                     # nothing is sung during the lead-in
    np.testing.assert_allclose(np.interp(gl, lp, rp), ev, atol=1e-9)
    gl2, gr2 = E.map_events(ev, lp, rp)
    np.testing.assert_allclose(np.interp(gl2, lp, rp), gr2, atol=1e-9)


def test_parse_overrides_uses_field_annotations():
    from lyricprompter.features.extractor import FeatureConfig as FC

    o = E._parse_overrides(["floor_init_db=-60", "weights=1.8,1.2,1", "reloc=false", "max_adv=4",
                            "relock_free_s=0"], TrackerParams)
    assert o == {"floor_init_db": -60.0, "weights": (1.8, 1.2, 1.0), "reloc": False, "max_adv": 4,
                 "relock_free_s": 0.0}
    assert isinstance(o["max_adv"], int) and isinstance(o["relock_free_s"], float)
    assert E._parse_overrides(["floor_init_db=none"], TrackerParams) == {"floor_init_db": None}
    from lyricprompter.tracking.tracker import LevelGate

    LevelGate(25.0, TrackerParams(**o)).update(-50.0)              # used to fail on a str floor
    assert E._parse_overrides(["kind=vocal", "tuning=0"], FC) == {"kind": "vocal", "tuning": 0.0}
    for bad in (["nope=1"], ["reloc=maybe"], ["max_adv=x"]):
        with pytest.raises(SystemExit):
            E._parse_overrides(bad, TrackerParams)


def test_evaluate_pair_keeps_an_explicit_tuning(monkeypatch):
    """An explicit FeatureConfig (tuning=0 included) is used as given, not replaced by the estimate."""
    seen = []

    class Stop(Exception):
        pass

    def fake_offline(y, cfg):
        seen.append(cfg.tuning)
        raise Stop

    monkeypatch.setattr(E, "_load", lambda path: np.zeros(SR, np.float32))
    monkeypatch.setattr(E, "song_tuning", lambda y: 0.37)
    monkeypatch.setattr(E, "live_and_gt", lambda pair, mode: (np.zeros(SR, np.float32), np.ones(1), np.ones(1)))
    monkeypatch.setattr(E, "compute_offline", fake_offline)
    monkeypatch.setattr(E, "_REF", {})
    monkeypatch.setattr(E, "_TUNING", {})
    pair = E.Pair("fake", "real", {"mix": E.Path("/x/mix.wav")}, {"mix": E.Path("/x/live.wav")}, np.ones(1))
    for fcfg, want in ((FeatureConfig.for_mode("mix", tuning=0.0), 0.0), (None, 0.37)):
        with pytest.raises(Stop):
            E.evaluate_pair(pair, "mix", fcfg)
        assert seen[-1] == want


def _tone_steps(freqs, seg_s=1.0, amp=0.3):
    t = np.arange(int(seg_s * SR)) / SR
    return np.concatenate([amp * np.sin(2 * np.pi * f * t) for f in freqs]).astype(np.float32)


def _dominant_freq(y, t_centres, n=2048):
    out = []
    for c in t_centres:
        i = int(c * SR)
        seg = y[max(0, i - n // 2):i + n // 2] * np.hanning(n)
        out.append(np.argmax(np.abs(np.fft.rfft(seg, n))) * SR / n)
    return np.array(out)


def test_pv_warp_follows_the_time_map():
    freqs = [300, 450, 600, 750, 900, 525, 375, 675]
    y = _tone_steps(freqs)
    lp = np.array([0.0, 2.5, 6.0, 9.0])
    rp = np.array([0.0, 2.0, 5.0, 8.0])                 # rho 0.8, then ~0.857, then 1.0
    w = E.pv_warp(y, SR, lp, rp)
    assert len(w) == round(lp[-1] * SR)
    probe = np.arange(0.3, 8.7, 0.1)
    ref_t = np.interp(probe, lp, rp)
    keep = np.abs(ref_t - np.round(ref_t)) > 0.15          # away from tone boundaries
    expect = np.array(freqs)[np.minimum(ref_t.astype(int), len(freqs) - 1)]
    got = _dominant_freq(w, probe)
    assert np.all(np.abs(got - expect)[keep] < 12)


def test_make_synthetic_lead_in_transpose_and_gain():
    y = _tone_steps([440.0] * 6)
    spec = E.SynthSpec(seed=1, transpose=1, lead_s=1.0, rt60_s=None, snr_db=30.0, eq_tilt_db=0.0)
    live, lp, _rp = E.make_synthetic(y, SR, spec)
    assert len(live) == pytest.approx(lp[-1] * SR, abs=SR * 0.01)
    lead, music = live[:SR], live[int(1.5 * SR):int(3 * SR)]
    assert np.sqrt(np.mean(lead ** 2)) < 0.1 * np.sqrt(np.mean(music ** 2))
    f = _dominant_freq(live, [2.0, 3.0, 4.0], n=8192)
    assert np.all(np.abs(f - 440 * 2 ** (1 / 12)) < 4)
    assert np.abs(live).max() <= 10 ** (-6 / 20) + 1e-6


def test_end_to_end_synthetic_audio_tracking():
    rng = np.random.default_rng(0)
    notes = rng.choice([262, 294, 330, 349, 392, 440, 494, 523], size=60)
    durs = rng.choice([0.25, 0.5, 0.75], size=60)
    t_parts = []
    for f, d in zip(notes, durs):
        t = np.arange(int(d * SR)) / SR
        env = np.minimum(1.0, t / 0.01) * np.exp(-2.0 * t)
        t_parts.append(env * (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t)))
    y = (0.3 * np.concatenate(t_parts)).astype(np.float32)
    live, lp, rp = E.make_synthetic(y, SR, E.SynthSpec(seed=2, lead_s=0.5))
    cfg = FeatureConfig.for_mode("mix")
    ref = compute_offline(y, cfg)
    blocks = E.stream_features(live, cfg)
    r = E.run_tracker(ref, blocks, TrackerParams.for_mode("mix"), cfg.fps)
    onsets_ref = np.cumsum(np.r_[0, durs[:-1]])[1:-2]
    m = E.prompter_metrics(blocks.t_pub, r["pos"], E.ref_to_live(onsets_ref, lp, rp), onsets_ref)
    assert m["PCO@300"] > 75 and m["PCO@1000"] > 95 and m["lost_frac"] == 0.0


@pytest.mark.skipif(ref_repo_dir() is None, reason="reference repo data not available")
def test_available_pairs_and_cli_list(capsys):
    pairs = E.available_pairs()
    assert "edelweiss" in pairs and len(pairs["edelweiss"].gt_ref) == 26
    assert pairs["edelweiss"].transpose_preset == 2
    if "kkotbat_r2s" in pairs:
        assert len(pairs["kkotbat_r2s"].gt_ref) == 144 and pairs["kkotbat_r2s"].modes == ("mix",)
        assert np.all(np.diff(pairs["kkotbat_r2s"].gt_ref) >= 0)
    assert E.cli(["--list"]) == 0
    assert "edelweiss" in capsys.readouterr().out


@pytest.mark.slow
@pytest.mark.skipif(ref_repo_dir() is None, reason="reference repo data not available")
@pytest.mark.parametrize("name,mode,pco300,pco1000", [
    ("edelweiss", "mix", 75.0, 95.0),
    ("edelweiss", "vocal", 65.0, 85.0),
    ("kkotbat_s2r", "mix", 60.0, 90.0),
    ("synth_edelweiss", "mix", 90.0, 98.0),
    ("synth_edelweiss", "vocal", 90.0, 98.0),
    ("synth_kkotbat", "vocal", 75.0, 90.0),
    ("synth_hila_skip", "mix", 85.0, 92.0),           # needs the global re-localisation
    ("synth_badside_repeat", "vocal", 80.0, 88.0),
])
def test_real_and_synthetic_pairs(name, mode, pco300, pco1000):
    pairs = E.available_pairs()
    if name not in pairs:
        pytest.skip("pair not available")
    p = pairs[name]
    if mode not in p.modes or not p.ref_audio[mode].is_file():
        pytest.skip("pair / stems not available for this mode")
    m = E.evaluate_pair(p, mode)
    assert m["PCO@300"] >= pco300 and m["PCO@1000"] >= pco1000, m
    if name.endswith(("_skip", "_repeat")):
        assert m["relocations"] >= 1
    assert m["trk_ms_p99"] < 20.0 and m["feat_ms_mean"] < 2.0


def test_structural_jumps_in_map_and_events():
    spec = E.SynthSpec(seed=5, lead_s=0.0, jumps=((20.0, 10.0), (40.0, -15.0)))
    lp, rp = E.synth_time_map(60.0, spec, np.random.default_rng(5))
    assert (np.diff(lp) > 0).all()
    k = int(np.flatnonzero(np.isclose(rp, 20.0))[0])
    assert rp[k + 1] == pytest.approx(30.0) and lp[k + 1] - lp[k] < 0.01      # skip 20 -> 30 s
    events = np.arange(1.0, 60.0, 1.0)
    gl, gr = E.map_events(events, lp, rp)
    # path: 0..20, skip to 30..40, back to 25..60 -> 22 s never sung, 27 s once, 35 s twice
    assert np.sum(np.isclose(gr, 22.0)) == 0
    assert np.sum(np.isclose(gr, 27.0)) == 1
    assert np.sum(np.isclose(gr, 35.0)) == 2
    assert (np.diff(gl) >= 0).all()
    np.testing.assert_allclose(np.interp(gl, lp, rp), gr, atol=1e-6)


def _melody(seed=0, n=60):
    rng = np.random.default_rng(seed)
    notes = rng.choice([262, 294, 330, 349, 392, 440, 494, 523], size=n)
    durs = rng.choice([0.25, 0.5, 0.75], size=n)
    parts = []
    for f, d in zip(notes, durs):
        t = np.arange(int(d * SR)) / SR
        env = np.minimum(1.0, t / 0.01) * np.exp(-2.0 * t)
        parts.append(env * (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t)))
    return (0.3 * np.concatenate(parts)).astype(np.float32)


def test_run_tracker_reports_the_key_found_by_auto_key():
    """A take 2 semitones up without a key preset: the automatic key detection finds it (final_key, one
    key switch); with the operator preset nothing is switched."""
    y = _melody(0, 80)
    live, _lp, _rp = E.make_synthetic(y, SR, E.SynthSpec(seed=2, lead_s=0.5, transpose=2))
    cfg = FeatureConfig.for_mode("mix")
    ref = compute_offline(y, cfg)
    blocks = E.stream_features(live, cfg)
    r = E.run_tracker(ref, blocks, TrackerParams.for_mode("mix"), cfg.fps)
    assert r["final_key"] == 2 and r["key_switches"] == 1
    r = E.run_tracker(ref, blocks, TrackerParams.for_mode("mix"), cfg.fps, transpose=2)
    assert r["final_key"] == 2 and r["key_switches"] == 0


def test_cli_no_key_preset_ignores_the_pairs_key(monkeypatch, tmp_path, capsys):
    wav = tmp_path / "ref.wav"
    wav.write_bytes(b"")
    pair = E.Pair("fake", "synth", {"mix": wav}, {}, np.ones(1), transpose_preset=2, modes=("mix",))
    seen = []

    def fake_evaluate(p, mode, fcfg=None, tparams=None, *, transpose=None, keep_trace=False):
        seen.append(transpose)
        row = {k: 0.0 for k in ("pos_AAE_ms", "pos_MedAE_ms", "lat_median_ms", "lost_frac")}
        row.update({f"PCO@{t}": 100.0 for t in E.TOLERANCES_MS})
        tp = p.transpose_preset if transpose is None else transpose
        row.update(pair=p.name, mode=mode, final_key=3, key_switches=int(tp != 3), key_preset=tp)
        return row

    monkeypatch.setattr(E, "available_pairs", lambda: {"fake": pair})
    monkeypatch.setattr(E, "pair_tuning", lambda p: 0.0)
    monkeypatch.setattr(E, "evaluate_pair", fake_evaluate)
    assert E.cli(["--pairs", "fake", "--modes", "mix", "--no-key-preset"]) == 0
    assert seen == [0]                                       # the pair's preset (+2) is not given
    out = capsys.readouterr().out
    assert "final_ke" in out and "  +3   1" in out          # final key and switch count columns
    assert E.cli(["--pairs", "fake", "--modes", "mix"]) == 0
    assert seen == [0, None]                                 # default: evaluate_pair applies the preset
