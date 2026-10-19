from __future__ import annotations

import json
import sys
import time
import types
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lyricprompter import config
from lyricprompter.audio_io import load_mono, read_audio, save_wav, to_canonical_wav
from lyricprompter.prep import separate as S

SR = 44100
StemFn = Callable[[np.ndarray], np.ndarray]


def tqdm(iterable, *args, **kwargs):   # stands in for the architecture module's tqdm (patched by S)
    return iterable


class FakeModelInstance:
    output_dir: str | None = None


class FakeSeparator:
    """Mimics audio_separator.Separator.separate(): default file naming, relative paths, ragged lengths."""

    def __init__(self, model: str, stems: dict[str, StemFn], length_delta: int = 0):
        self.model, self.stems, self.length_delta = model, stems, length_delta
        self.model_instance = FakeModelInstance()
        self.output_dir: str | None = None
        self.calls = 0

    def separate(self, path: str) -> list[str]:
        self.calls += 1
        for _ in tqdm(range(4)):
            pass
        y, sr = sf.read(path, dtype="float32", always_2d=True)
        outs = []
        for stem, fn in self.stems.items():
            z = fn(y)
            z = z[: len(z) + self.length_delta] if self.length_delta < 0 else np.pad(z, ((0, self.length_delta), (0, 0)))
            name = f"{Path(path).stem}_({stem})_{Path(self.model).stem}.wav"
            sf.write(Path(self.model_instance.output_dir) / name, z, sr, subtype="FLOAT")
            outs.append(name)
        return outs


def two_stem(model: str, voc: str = "vocals", inst: str = "other", delta: int = 0) -> FakeSeparator:
    return FakeSeparator(model, {inst: lambda y: 0.7 * y, voc: lambda y: 0.3 * y}, delta)


@pytest.fixture()
def mix(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    y = (0.2 * rng.standard_normal((SR * 2 + 17, 2))).astype(np.float32)
    return save_wav(tmp_path / "song" / "audio" / "mix.wav", np.clip(y, -0.9, 0.9), SR)


def vocal_like(n: int, sr: int, seed: int = 1) -> np.ndarray:
    """Stereo 'singing': gliding harmonic tone in syllable bursts plus a little breath noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    f0 = 220 + 80 * np.sin(2 * np.pi * 0.3 * t) + 40 * np.sin(2 * np.pi * 1.7 * t + 1.0)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    tone = sum(0.5 ** k * np.sin(k * phase) for k in range(1, 6))
    env = np.clip(np.sin(2 * np.pi * 1.3 * t + rng.uniform(0, 6)) * 1.5, 0, 1) ** 2
    y = 0.25 * env * tone + 0.003 * rng.standard_normal(n)
    return np.stack([y, 0.9 * y], axis=1).astype(np.float32)


@pytest.fixture()
def vocal_mix(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    """(mix.wav, vocals, accompaniment) with mix = vocals + accompaniment, 8 s at 44.1 kHz."""
    n = 8 * SR
    rng = np.random.default_rng(7)
    acc = (0.08 * rng.standard_normal((n, 2))).astype(np.float32)
    v = vocal_like(n, SR)
    mix = save_wav(tmp_path / "song" / "audio" / "mix.wav", v + acc, SR, subtype="FLOAT")
    return mix, v, acc


@pytest.fixture()
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeSeparator]:
    reg = {S.DEFAULT_SEP_MODEL: two_stem(S.DEFAULT_SEP_MODEL, delta=-5),
           "other_model.ckpt": two_stem("other_model.ckpt", "Vocals", "Instrumental", delta=3),
           S.LEAD_SEP_MODEL: FakeSeparator(S.LEAD_SEP_MODEL, {"Instrumental": lambda y: 0.8 * y,
                                                              "Vocals": lambda y: 0.2 * y})}
    monkeypatch.setattr(S, "get_separator", lambda model: reg[model])
    return reg


def test_stem_names_and_roles() -> None:
    assert S._stem_name("mix_(Vocals)_bs_roformer_vocals_resurrection_unwa.wav") == "vocals"
    assert S._stem_name("a (live)_(Instrumental)_m.wav") == "instrumental"
    assert S.split_roles(["other", "vocals"]) == ("vocals", ["other"])
    assert S.split_roles(["drums", "bass", "other", "vocals"]) == ("vocals", ["drums", "bass", "other"])
    assert S.split_roles(["instrumental"]) == (None, ["instrumental"])
    with pytest.raises(S.SeparationError):
        S.split_roles(["noreverb", "reverb"])
    with pytest.raises(S.SeparationError):
        S.split_roles(["vocals", "lead"])


def test_separate_writes_mix_length_stems_and_manifest(mix: Path, fakes: dict, tmp_path: Path) -> None:
    out = mix.parent
    events: list[tuple[str, float, str]] = []
    res = S.separate_vocals(mix, out, progress=lambda *a: events.append(a))
    assert set(res) == {"vocals", "instrumental"}
    m, _ = read_audio(mix)
    v, sr = read_audio(res["vocals"])
    i, _ = read_audio(res["instrumental"])
    assert sr == SR and v.shape == i.shape == m.shape          # fake output was 5 samples short
    assert sf.info(res["vocals"]).subtype == S.STEM_SUBTYPE
    np.testing.assert_allclose(v[:-5] + i[:-5], m[:-5], atol=1e-5)
    assert not np.any(v[-5:])                                   # zero padded
    assert not list(out.glob(".sep-*"))                         # temp dirs removed
    man = json.loads((out / S.MANIFEST).read_text())
    assert man["model"] == S.DEFAULT_SEP_MODEL and man["lead_model"] is None
    assert man["stems"] == {"vocals": "vocals.wav", "instrumental": "instrumental.wav"}
    assert man["mix_frames"] == m.shape[0] and man["derived"] == {"vocals": "vocals", "instrumental": "other"}
    pcts = [e[1] for e in events]
    assert {e[0] for e in events} == {"separate"} and pcts == sorted(pcts) and pcts[-1] == 100
    assert any(0 < p < 100 for p in pcts)                       # tqdm loop routed into progress


def test_reuse_model_change_and_lead(mix: Path, fakes: dict) -> None:
    out = mix.parent
    S.separate_vocals(mix, out)
    assert fakes[S.DEFAULT_SEP_MODEL].calls == 1
    again = S.separate_vocals(mix, out)
    assert fakes[S.DEFAULT_SEP_MODEL].calls == 1 and set(again) == {"vocals", "instrumental"}

    res = S.separate_vocals(mix, out, lead=True)                # vocals reused, lead pass only
    assert fakes[S.DEFAULT_SEP_MODEL].calls == 1 and fakes[S.LEAD_SEP_MODEL].calls == 1
    lead, _ = read_audio(res["lead"])
    np.testing.assert_allclose(lead, 0.2 * read_audio(mix)[0], atol=1e-5)
    assert S.read_manifest(out)["lead_model"] == S.LEAD_SEP_MODEL
    assert set(S.separate_vocals(mix, out)) == {"vocals", "instrumental", "lead"}   # lead kept

    res = S.separate_vocals(mix, out, model="other_model.ckpt")  # model change -> rerun, lead still valid
    assert fakes["other_model.ckpt"].calls == 1 and "lead" in res
    assert S.read_manifest(out)["derived"]["instrumental"] == "instrumental"
    assert sf.info(res["vocals"]).frames == sf.info(mix).frames  # fake output was 3 samples long

    S.separate_vocals(mix, out, model="other_model.ckpt", reuse=False)
    assert fakes["other_model.ckpt"].calls == 2


def test_mix_change_invalidates_cache_and_stale_lead(mix: Path, fakes: dict) -> None:
    out = mix.parent
    S.separate_vocals(mix, out, lead=True)
    y, _ = read_audio(mix)
    save_wav(mix, 0.5 * y, SR)
    assert S.cached_stems(out, mix, S.DEFAULT_SEP_MODEL) is None
    res = S.separate_vocals(mix, out)
    assert fakes[S.DEFAULT_SEP_MODEL].calls == 2
    assert "lead" not in res and not (out / "lead.wav").exists()


def test_vocals_derived_when_model_outputs_only_instrumental(mix: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSeparator("inst.ckpt", {"Instrumental": lambda y: 0.6 * y})
    monkeypatch.setattr(S, "get_separator", lambda model: fake)
    res = S.separate_vocals(mix, mix.parent, model="inst.ckpt")
    np.testing.assert_allclose(read_audio(res["vocals"])[0], 0.4 * read_audio(mix)[0], atol=1e-5)


def test_separation_failure_is_wrapped(mix: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = two_stem("x.ckpt")
    fake.separate = lambda path: (_ for _ in ()).throw(RuntimeError("MPS exploded"))
    monkeypatch.setattr(S, "get_separator", lambda model: fake)
    with pytest.raises(S.SeparationError, match="MPS exploded"):
        S.separate_vocals(mix, mix.parent, model="x.ckpt")
    assert not list(mix.parent.glob(".sep-*"))


def test_get_separator_cache_scopes_warnings_and_evicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    class Sep:
        def __init__(self, **kw):
            assert Path(kw["model_file_dir"]).is_dir() and kw["use_soundfile"] is True
            assert kw["normalization_threshold"] == 1.0 and kw["use_autocast"] is S.USE_AUTOCAST
            warnings.filterwarnings("ignore")        # what the real constructor does, process-wide
            self.model_instance = object()

        def load_model(self, model_filename: str) -> None:
            if model_filename == "bad.ckpt":
                raise ValueError("not in supported model files")
            created.append(model_filename)

    pkg = types.ModuleType("audio_separator")
    mod = types.ModuleType("audio_separator.separator")
    mod.Separator = Sep
    monkeypatch.setitem(sys.modules, "audio_separator", pkg)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", mod)
    monkeypatch.setenv("AUDIO_SEPARATOR_MODEL_DIR", str(tmp_path / "models" / "sep"))
    monkeypatch.setattr(S, "_loaded", type(S._loaded)())

    before = list(warnings.filters)
    a = S.get_separator("a.ckpt")
    assert warnings.filters == before
    assert S.get_separator("a.ckpt") is a and created == ["a.ckpt"]
    S.get_separator("b.ckpt")
    assert list(S._loaded) == ["b.ckpt"] and a.model_instance is None    # one heavy model per process
    with pytest.raises(S.SeparationError, match="unknown separation model"):
        S.get_separator("bad.ckpt")
    S.release_separators()
    assert not S._loaded
    assert S.model_dir() == tmp_path / "models" / "sep"


def test_failed_model_load_keeps_reusable_stems(mix: Path, fakes: dict,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """stems.json survives a model that cannot load (or run): the next run reuses everything."""
    out = mix.parent
    S.separate_vocals(mix, out, lead=True)
    before = (out / S.MANIFEST).read_text()

    def get(model: str) -> FakeSeparator:
        if model not in fakes:
            raise S.SeparationError(f"could not load separation model {model!r}: network down")
        return fakes[model]

    monkeypatch.setattr(S, "get_separator", get)
    with pytest.raises(S.SeparationError, match="network down"):
        S.separate_vocals(mix, out, model="broken.ckpt", lead=True)
    assert (out / S.MANIFEST).read_text() == before
    res = S.separate_vocals(mix, out, lead=True)
    assert fakes[S.DEFAULT_SEP_MODEL].calls == 1 and fakes[S.LEAD_SEP_MODEL].calls == 1 and "lead" in res


def test_lead_model_failure_keeps_fresh_vocals(mix: Path, fakes: dict,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    out = mix.parent
    lead_fake = fakes.pop(S.LEAD_SEP_MODEL)
    monkeypatch.setattr(S, "get_separator", lambda model: fakes[model] if model in fakes else
                        (_ for _ in ()).throw(S.SeparationError("lead model unavailable")))
    with pytest.raises(S.SeparationError, match="lead model unavailable"):
        S.separate_vocals(mix, out, lead=True)
    man = S.read_manifest(out)
    assert man is not None and set(man["stems"]) == {"vocals", "instrumental"} and man["lead_model"] is None
    fakes[S.LEAD_SEP_MODEL] = lead_fake
    res = S.separate_vocals(mix, out, lead=True)
    assert fakes[S.DEFAULT_SEP_MODEL].calls == 1 and lead_fake.calls == 1 and "lead" in res


def test_import_vocals_lead_run_and_reuse(vocal_mix: tuple, fakes: dict, tmp_path: Path) -> None:
    mix, v, acc = vocal_mix
    out = mix.parent
    voc = save_wav(tmp_path / "v.wav", v, SR)
    res = S.import_vocals(voc, mix, out, lead=True)
    assert set(res) == {"vocals", "instrumental", "lead"} and fakes[S.LEAD_SEP_MODEL].calls == 1
    man = S.read_manifest(out)
    assert (man["model"], man["lead_model"]) == (S.EXTERNAL_MODEL, S.LEAD_SEP_MODEL)
    assert man["warnings"] == [] and man["check"]["matched"] and man["check"]["lag"] == 0
    assert man["check"]["gain"] == 1.0 and man["check"]["gain_how"] == "unity"
    np.testing.assert_allclose(read_audio(res["instrumental"])[0], acc, atol=1e-4)
    res = S.import_vocals(voc, mix, out, lead=True)              # lead of the same mix is kept
    assert fakes[S.LEAD_SEP_MODEL].calls == 1 and "lead" in res


def test_import_vocals_unreadable_stem_keeps_manifest(vocal_mix: tuple, fakes: dict, tmp_path: Path) -> None:
    mix, v, _ = vocal_mix
    out = mix.parent
    S.import_vocals(save_wav(tmp_path / "v.wav", v, SR), mix, out)
    before = (out / S.MANIFEST).read_text()
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(b"not audio at all" * 100)
    with pytest.raises(Exception, match="junk.mp3"):
        S.import_vocals(junk, mix, out)
    assert (out / S.MANIFEST).read_text() == before


@pytest.mark.parametrize(("shift_s", "scale"), [(0.4, 0.8), (-0.25, 1.0), (0.0, 0.5)])
def test_import_vocals_corrects_offset_and_level(vocal_mix: tuple, tmp_path: Path, shift_s: float,
                                                 scale: float) -> None:
    mix, v, _ = vocal_mix
    out = mix.parent
    k = round(shift_s * SR)
    supplied = scale * (np.concatenate([np.zeros((k, 2), np.float32), v]) if k >= 0 else v[-k:])
    res = S.import_vocals(save_wav(tmp_path / "v.flac", supplied, SR), mix, out)
    man = S.read_manifest(out)
    chk = man["check"]
    assert chk["matched"] and chk["lag"] == k and chk["dominance"] >= S.MATCH_MIN_DOMINANCE
    assert chk["gain"] == pytest.approx(1 / scale, rel=0.01) if scale != 1.0 else chk["gain"] == 1.0
    got, _ = read_audio(res["vocals"])
    lo, hi = max(0, -k), len(v) - max(0, k)              # region the supplied stem covers
    leak = np.sum((got[lo:hi] - v[lo:hi]) ** 2) / np.sum(v[lo:hi] ** 2)
    assert 10 * np.log10(leak) < -35                      # vocals.wav lines up with the mix at its level
    joined = " | ".join(man["warnings"])
    assert ("offset by" in joined) == (k != 0) and ("rescaled" in joined) == (scale != 1.0)
    if k:
        assert f"{shift_s:+.3f}s" in joined


def test_import_vocals_unrelated_stem_is_flagged(vocal_mix: tuple, tmp_path: Path) -> None:
    mix, v, _ = vocal_mix
    other = vocal_like(len(v), SR, seed=99)[::-1].copy()      # a different "song"
    S.import_vocals(save_wav(tmp_path / "v.wav", other, SR), mix, mix.parent)
    man = S.read_manifest(mix.parent)
    assert not man["check"]["matched"] and man["check"]["lag"] == 0 and man["check"]["gain"] == 1.0
    assert any("could not be lined up" in w for w in man["warnings"])


def test_import_vocals_recognizes_separator_normalization(tmp_path: Path) -> None:
    """A stem from a separator that scaled its input to peak 0.9 is restored exactly (mix peak / 0.9)."""
    n = 8 * SR
    v = vocal_like(n, SR)
    acc = (0.15 * np.random.default_rng(3).standard_normal((n, 2))).astype(np.float32)
    y = v + acc
    y *= 0.98 / np.max(np.abs(y))
    mix = save_wav(tmp_path / "audio" / "mix.wav", y, SR, subtype="FLOAT")
    v_in_mix = v * (0.98 / np.max(np.abs(v + acc)))
    norm = 0.9 / float(np.max(np.abs(read_audio(mix)[0])))
    S.import_vocals(save_wav(tmp_path / "v.wav", norm * v_in_mix, SR, subtype="FLOAT"), mix, mix.parent)
    chk = S.read_manifest(mix.parent)["check"]
    assert chk["gain_how"] == "separator-normalization" and chk["gain"] == pytest.approx(1 / norm, rel=1e-5)


def test_match_vocals_degenerate_inputs() -> None:
    silent = np.zeros((SR, 2), np.float32)
    m = S.match_vocals(silent, vocal_like(SR, SR), SR)
    assert not m.matched and m.lag == 0 and m.gain == 1.0
    assert not S.match_vocals(vocal_like(100, SR), vocal_like(SR, SR), SR).matched     # too short


def test_truncated_checkpoint_is_quarantined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDIO_SEPARATOR_MODEL_DIR", str(tmp_path))
    good = tmp_path / "good.ckpt"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("archive/data.pkl", b"x" * 100_000)
    data = good.read_bytes()
    (tmp_path / "cut.ckpt").write_bytes(data[: len(data) // 2])
    (tmp_path / "legacy.ckpt").write_bytes(b"\x80\x02" + b"y" * 1000)       # pre-zip torch pickle
    assert S.is_model_cached("good.ckpt") and S.is_model_cached("legacy.ckpt")
    assert not S.is_model_cached("cut.ckpt") and not S.is_model_cached("missing.ckpt")
    assert S._quarantine_truncated(good) is None and S._quarantine_truncated(tmp_path / "legacy.ckpt") is None
    assert S._quarantine_truncated(tmp_path / "cut.ckpt") == tmp_path / "cut.ckpt.corrupt"
    assert not (tmp_path / "cut.ckpt").exists() and good.is_file()


def test_model_download_is_atomic_silent_and_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      capfd: pytest.CaptureFixture) -> None:
    payload = b"\x80\x02" + bytes(range(256)) * 12_000            # ~3 MB, not a zip
    chunk = 1 << 18

    class Sep:
        fail_after: int | None = None

        def __init__(self, **kw):
            self.dir = Path(kw["model_file_dir"])
            self.model_instance = object()

        def download_file_if_not_exists(self, url: str, output_path: str) -> None:
            """Like audio-separator's: writes to output_path, module-level tqdm(total=...) bar."""
            bar = sys.modules[type(self).__module__].tqdm(total=len(payload), unit="iB")
            with open(output_path, "wb") as f:
                for i in range(0, len(payload), chunk):
                    if Sep.fail_after is not None and i >= Sep.fail_after:
                        raise ConnectionError("connection dropped")
                    f.write(payload[i:i + chunk])
                    bar.update(len(payload[i:i + chunk]))
            bar.close()

        def load_model(self, model_filename: str) -> None:
            self.download_file_if_not_exists(f"https://example.invalid/{model_filename}",
                                             str(self.dir / model_filename))

    pkg = types.ModuleType("audio_separator")
    mod = types.ModuleType("audio_separator.separator")
    mod.Separator = Sep
    monkeypatch.setitem(sys.modules, "audio_separator", pkg)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", mod)
    monkeypatch.setenv("AUDIO_SEPARATOR_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(S, "_loaded", type(S._loaded)())

    events: list[tuple[str, float, str]] = []
    S._load("m.ckpt", lambda *a: events.append(a), 0.0, 5.0)
    assert (tmp_path / "models/m.ckpt").read_bytes() == payload
    assert not list((tmp_path / "models").glob("*.part"))
    msgs, pcts = [e[2] for e in events], [e[1] for e in events]
    assert msgs[0] == "loading m.ckpt" and msgs[-1] == "downloading m.ckpt 3/3 MB"
    assert 3 <= len(msgs) <= 60 and pcts == sorted(pcts) and pcts[-1] == 5.0
    assert tqdm.__name__ == "tqdm"                                  # the module's tqdm is restored
    assert capfd.readouterr().err == ""                             # no progress bar on stderr

    S.release_separators()
    Sep.fail_after = 2 * chunk
    with pytest.raises(S.SeparationError, match="connection dropped"):
        S._load("n.ckpt", None, 0.0, 5.0)
    assert sorted(p.name for p in (tmp_path / "models").iterdir()) == ["m.ckpt"]   # no partial file


# --------------------------------------------------------------------------------------------- slow

@pytest.mark.slow
def test_real_separation_of_15s_excerpt(tmp_path: Path) -> None:
    ref = config.ref_repo_dir()
    if ref is None or not S.is_model_cached(S.DEFAULT_SEP_MODEL):
        pytest.skip("reference audio or cached separation model missing")
    src = tmp_path / "full.wav"
    to_canonical_wav(ref / "resources/audio/Edelweiss_short.wav", src)
    y, sr = read_audio(src)
    mix = save_wav(tmp_path / "audio" / "mix.wav", y[int(6.5 * sr): int(21.5 * sr)], sr)
    try:
        t0 = time.perf_counter()
        res = S.separate_vocals(mix, tmp_path / "audio")
        took = time.perf_counter() - t0
    finally:
        S.release_separators()
    m, _ = read_audio(mix)
    v, _ = read_audio(res["vocals"])
    i, _ = read_audio(res["instrumental"])
    assert v.shape == i.shape == m.shape == (15 * sr, 2)
    resid = m - v - i
    snr = 10 * np.log10(np.sum(m**2) / max(np.sum(resid**2), 1e-12))
    vocal_share = np.sum(v**2) / np.sum(m**2)
    # the excerpt is sung throughout: vocals carry real energy, and agree with the pre-separated stem
    full_v = load_mono(Path(config.DATA_DIR / "stems/edelweiss_short/vocals.wav"), 16000) \
        if (config.DATA_DIR / "stems/edelweiss_short/vocals.wav").is_file() else None
    print(f"\nseparation 15s: {took:.1f}s, reconstruction SNR {snr:.1f} dB, vocal energy share {vocal_share:.2f}")
    assert snr > 30
    assert 0.02 < vocal_share < 0.98
    if full_v is not None:
        ours = load_mono(res["vocals"], 16000)
        theirs = full_v[int(6.5 * 16000): int(6.5 * 16000) + len(ours)]
        corr = float(np.corrcoef(np.abs(ours), np.abs(theirs))[0, 1])
        print(f"envelope correlation with data/stems vocals: {corr:.3f}")
        assert corr > 0.8
    assert S.cached_stems(tmp_path / "audio", mix, S.DEFAULT_SEP_MODEL) is not None


@pytest.mark.slow
def test_import_real_presepared_vocals(tmp_path: Path) -> None:
    """data/stems were made from an `ffmpeg -ac 2` upmix of the mono source (-3 dB): the level is restored."""
    ref = config.ref_repo_dir()
    stem = config.DATA_DIR / "stems/edelweiss_short/vocals.wav"
    if ref is None or not stem.is_file():
        pytest.skip("reference audio or pre-separated stem missing")
    mix = to_canonical_wav(ref / "resources/audio/Edelweiss_short.wav", tmp_path / "audio" / "mix.wav")
    res = S.import_vocals(stem, mix, mix.parent)
    chk = S.read_manifest(mix.parent)["check"]
    print(f"\nedelweiss_short data/stems vocals: {chk}")
    assert chk["matched"] and chk["lag"] == 0 and chk["gain"] == pytest.approx(np.sqrt(2), rel=0.01)
    m, _ = read_audio(mix)
    v, _ = read_audio(res["vocals"])
    theirs, _ = read_audio(stem)
    # the derived instrumental keeps (almost) no vocal: its projection on the true-level vocal stem
    inst = m - v
    true_v = np.sqrt(2) * theirs
    leak = float(np.sum(inst * true_v) / np.sum(true_v**2)) - float(
        np.sum((m - true_v) * true_v) / np.sum(true_v**2))
    assert abs(leak) < 0.01
