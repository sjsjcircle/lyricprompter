from __future__ import annotations

import os
import time
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from lyricprompter_oscsender.features.extractor import (
    FeatureConfig,
    FeatureExtractor,
    FeatureFrames,
    compute_offline,
    load_ref_features,
)

SR = 16000


def _signal(seconds=3.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    y = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 659.25 * t * (1 + 0.1 * t))
    y += 0.05 * rng.standard_normal(len(t))
    y[SR // 2:SR] = 0.0                                   # a silent stretch
    return y.astype(np.float32)


CONFIGS = [FeatureConfig(), FeatureConfig(w_flux=1.0, mfcc_cmn_s=3.0, n_fft=4096),
           FeatureConfig(w_mfcc=0.0), FeatureConfig.for_mode("vocal", tuning=0.2)]


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.key())
def test_streaming_equals_offline_any_block_sizes(cfg):
    y = _signal()
    off = compute_offline(y, cfg)
    assert len(off) == len(y) // cfg.hop
    for seed in range(3):
        rng = np.random.default_rng(seed)
        ext = FeatureExtractor(cfg)
        parts, i = [], 0
        while i < len(y):
            b = int(rng.choice([1, 7, 333, 640, 1024, 5000]))
            parts.append(ext.push(y[i:i + b]))
            i += b
        s = FeatureFrames.concat(parts)
        assert s.feats.shape == off.feats.shape and s.groups == off.groups
        np.testing.assert_allclose(s.feats, off.feats, atol=2e-5)
        np.testing.assert_allclose(s.rms_db, off.rms_db, atol=1e-4)


def test_causal_framing():
    cfg = FeatureConfig()
    y = _signal()
    a = compute_offline(y, cfg)
    y2 = y.copy()
    k = 40
    y2[(k + 1) * cfg.hop:] = np.random.default_rng(1).standard_normal(len(y) - (k + 1) * cfg.hop)
    b = compute_offline(y2, cfg)
    np.testing.assert_array_equal(a.feats[:k + 1], b.feats[:k + 1])     # frame k ends at (k+1)*hop
    assert not np.allclose(a.feats[k + 1], b.feats[k + 1])


def test_unit_norm_groups_and_rms():
    cfg = FeatureConfig(w_chroma=2.0, w_mfcc=1.0, w_flux=1.0)
    t = np.arange(SR) / SR
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    ff = compute_offline(y, cfg)
    np.testing.assert_allclose(np.linalg.norm(ff.feats, axis=1), 1.0, atol=1e-5)
    ch = ff.feats[10:, ff.group("chroma")]
    assert (np.argmax(ch, axis=1) == 9).all()                          # A
    np.testing.assert_allclose(np.linalg.norm(ch, axis=1) ** 2, 0.5, atol=1e-5)   # weight 2 / 4
    np.testing.assert_allclose(ff.rms_db[5:], 10 * np.log10(0.125), atol=0.05)


def test_silence_is_uniform_chroma_and_matches_silence():
    cfg = FeatureConfig()
    ff = compute_offline(np.zeros(SR, np.float32), cfg)
    ch = ff.feats[:, ff.group("chroma")]
    np.testing.assert_allclose(ch / ch[0, 0], 1.0, atol=1e-5)
    assert np.allclose(1 - ff.feats[3] @ ff.feats[20], 0.0, atol=1e-6)


def test_transpose_rolls_chroma():
    y = _signal(1.0)
    a = compute_offline(y, FeatureConfig())
    b = compute_offline(y, FeatureConfig(transpose=2))
    sl = a.group("chroma")
    np.testing.assert_allclose(np.roll(a.feats[:, sl], 2, axis=1), b.feats[:, sl], atol=1e-6)
    np.testing.assert_allclose(a.feats[:, 12:], b.feats[:, 12:], atol=1e-6)


def test_key_is_stable_and_config_sensitive():
    assert FeatureConfig().key() == FeatureConfig().key()
    assert FeatureConfig().key() != FeatureConfig(tuning=0.1).key()
    assert FeatureConfig.for_mode("mix").kind == "mix" and FeatureConfig.for_mode("vocal").kind == "vocal"
    with pytest.raises(ValueError):
        FeatureConfig.for_mode("karaoke")


def test_load_ref_features_cache(tmp_path):
    song = tmp_path / "song"
    (song / "audio").mkdir(parents=True)
    y = _signal(2.0)
    sf.write(song / "audio" / "mix.wav", np.stack([y, y], 1), SR)
    cfg = FeatureConfig()
    a = load_ref_features(song, "mix", cfg)
    cache = song / "cache" / f"feat_{cfg.key()}_mix.npz"
    assert cache.is_file()
    mtime = cache.stat().st_mtime_ns
    b = load_ref_features(song, "mix", cfg)
    assert cache.stat().st_mtime_ns == mtime                            # served from cache
    np.testing.assert_array_equal(a.feats, b.feats)
    assert b.groups == a.groups
    sf.write(song / "audio" / "mix.wav", np.stack([y[:SR], y[:SR]], 1), SR)
    st = (song / "audio" / "mix.wav").stat()
    os.utime(song / "audio" / "mix.wav", ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    c = load_ref_features(song, "mix", cfg)
    assert len(c) == SR // cfg.hop                                      # recomputed after the change


@pytest.mark.parametrize("mode", ["mix", "vocal"])
def test_non_finite_input_does_not_poison_the_stream(mode):
    """One NaN/inf sample used to make every later frame NaN (causal CMN filter state)."""
    cfg = FeatureConfig.for_mode(mode)
    y = _signal(4.0)
    bad, clean = y.copy(), y.copy()
    for i, v in ((10 * 1024 + 5, np.nan), (20 * 1024, np.inf), (30 * 1024 + 7, -np.inf)):
        bad[i], clean[i] = v, 0.0
    bad[40 * 1024] = clean[40 * 1024] = 1e30                     # absurd but finite: clipped
    ext = FeatureExtractor(cfg)
    a = FeatureFrames.concat([ext.push(bad[i:i + 1024]) for i in range(0, len(bad), 1024)])
    b = compute_offline(clean, cfg)
    assert len(a) == len(b) == 100 and np.isfinite(a.feats).all() and np.isfinite(a.rms_db).all()
    k = 40 * 1024 // cfg.hop                                     # frames before the clipped spike:
    np.testing.assert_allclose(a.feats[:k], b.feats[:k], atol=2e-5)   # non-finite samples -> 0


def _song_with_mix(tmp_path, seconds=2.0):
    song = tmp_path / "song"
    (song / "audio").mkdir(parents=True)
    y = _signal(seconds)
    sf.write(song / "audio" / "mix.wav", np.stack([y, y], 1), SR)
    return song


@pytest.mark.parametrize("damage", ["truncate", "empty", "garbage"])
def test_load_ref_features_recovers_from_a_corrupt_cache(tmp_path, damage):
    song = _song_with_mix(tmp_path)
    cfg = FeatureConfig()
    a = load_ref_features(song, "mix", cfg)
    cache = song / "cache" / f"feat_{cfg.key()}_mix.npz"
    blob = cache.read_bytes()
    cache.write_bytes({"truncate": blob[:len(blob) // 2], "empty": b"", "garbage": b"PK\x03\x04" + b"x" * 100}[damage])
    b = load_ref_features(song, "mix", cfg)                             # recomputed instead of raising
    np.testing.assert_array_equal(a.feats, b.feats)
    assert cache.read_bytes() == blob                                   # and rewritten
    c = load_ref_features(song, "mix", cfg)
    np.testing.assert_array_equal(a.feats, c.feats)


def test_load_ref_features_concurrent_cold_loads(tmp_path):
    """Several sessions cold-loading the same song used to race on one shared temp file."""
    from concurrent.futures import ThreadPoolExecutor

    song = _song_with_mix(tmp_path)
    cfg = FeatureConfig()
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda _: load_ref_features(song, "mix", cfg), range(12)))
    for r in res[1:]:
        np.testing.assert_array_equal(r.feats, res[0].feats)
    assert sorted(x.name for x in (song / "cache").iterdir()) == [f"feat_{cfg.key()}_mix.npz"]
    np.testing.assert_array_equal(load_ref_features(song, "mix", cfg).feats, res[0].feats)


def test_cpu_budget_per_frame():
    for mode in ("mix", "vocal"):
        ext = FeatureExtractor(FeatureConfig.for_mode(mode))
        y = np.random.default_rng(0).standard_normal(SR * 10).astype(np.float32) * 0.1
        n, t0 = 0, time.perf_counter()
        for i in range(0, len(y), 1024):
            n += len(ext.push(y[i:i + 1024]))
        per_frame_ms = (time.perf_counter() - t0) / n * 1e3
        assert per_frame_ms < 2.0, per_frame_ms


def _phrases(seed=3):
    """Sung-like phrases (tones) separated by near-silence (a quiet noise floor at about -70 dBFS)."""
    rng = np.random.default_rng(seed)
    t = np.arange(SR) / SR
    parts = []
    for f in (330.0, 392.0, 440.0, 494.0):
        parts.append(0.1 * np.sin(2 * np.pi * f * t) * np.minimum(1.0, t / 0.05))
        parts.append(np.zeros(SR // 2))
    y = np.concatenate(parts)
    return (y + 3e-4 * rng.standard_normal(len(y))).astype(np.float32)


@pytest.mark.parametrize("dc", [0.003, 0.01, 0.03])
def test_dc_offset_is_neither_level_nor_feature(dc):
    """A constant DC offset (cheap interfaces: -50..-30 dBFS) must not read as signal: it used to set the
    activity gate's noise floor and vocal mode stopped tracking a quiet singer."""
    from lyricprompter_oscsender.tracking.tracker import LevelGate, TrackerParams

    cfg = FeatureConfig.for_mode("vocal")
    y = _phrases()
    a, b = compute_offline(y, cfg), compute_offline(y + np.float32(dc), cfg)
    # frames 0..2 still see the stream-start step 0 -> dc through the zero padding
    np.testing.assert_allclose(b.rms_db[3:], a.rms_db[3:], atol=0.05)
    assert a.rms_db[a.rms_db < -50].max() < -60                 # the pauses are still near-silent
    # the spectrum too (without the causal CMN, which still remembers that step for a while)
    nocmn = replace(cfg, mfcc_cmn_s=0.0)
    np.testing.assert_allclose(compute_offline(y + np.float32(dc), nocmn).feats[3:],
                               compute_offline(y, nocmn).feats[3:], atol=1e-3)
    p = TrackerParams.for_mode("vocal")
    ga, gb = LevelGate(cfg.fps, p), LevelGate(cfg.fps, p)
    act_a = np.array([ga.update(float(v)) for v in a.rms_db])
    act_b = np.array([gb.update(float(v)) for v in b.rms_db])
    assert act_a.mean() > 0.5 and np.array_equal(act_a, act_b)
