"""Project store: ids, atomic creation (wav / mp3 / m4a, vocal stems), meta, cue file I/O, waveform peaks,
trash and the `add` / `list` commands. Tests that convert audio need ffmpeg (skipped without it)."""
from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import time
import unicodedata
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lyricprompter_oscsender import config
from lyricprompter_oscsender import project as P
from lyricprompter_oscsender.audio_io import AudioError, to_canonical_wav
from lyricprompter_oscsender.cli import main as cli_main

FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg / ffprobe not on PATH")

CUES = "\ufeff# show\r\n@target lights 127.0.0.1:9000\r\n0:00.500 /a 1\r\n1.25  lights /b \"x y\"\r\n"


def tone(path: Path, seconds: float = 2.0, sr: int = 22050, channels: int = 1, freq: float = 440.0,
         amp: float = 0.3) -> Path:
    t = np.arange(round(seconds * sr)) / sr
    y = amp * np.sin(2 * np.pi * freq * t)
    if channels > 1:
        y = np.stack([y * (1.0 - 0.1 * c) for c in range(channels)], axis=1)
    sf.write(str(path), y, sr)
    return path


def visible(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir()) if root.is_dir() else []


def make_bare_project(root: Path, pid: str = "p", seconds: float = 1.0, signal=None, sr: int = 44100) -> Path:
    """A project folder written directly (no ffmpeg): canonical-rate mix + project.json."""
    pdir = root / pid
    (pdir / "audio").mkdir(parents=True)
    y = signal if signal is not None else 0.25 * np.sin(2 * np.pi * 220 * np.arange(round(seconds * sr)) / sr)
    sf.write(str(pdir / "audio" / "mix.wav"), np.stack([y, y], axis=1), sr, subtype="PCM_24")
    P.save_meta(pdir, P.ProjectMeta(id=pid, title=pid.upper(), created="2026-01-01T00:00:00+00:00",
                                    duration=len(y) / sr, tuning=0.0))
    P.write_cues(pdir, "")
    return pdir


# ------------------------------------------------------------------------------------------ ids


@pytest.mark.parametrize("pid", ["abc", "My-Song_2.v1", "한글-곡", "曲.1", "x" * 200])
def test_valid_ids(pid):
    assert P.check_project_id(pid) == pid


def test_id_is_nfc():
    nfd = unicodedata.normalize("NFD", "가나")
    assert P.check_project_id(nfd) == "가나"
    assert P.check_project_id("カ\u3099") == "ガ"           # composes in NFC


@pytest.mark.parametrize("pid", ["", ".hidden", "a/b", "a..b", "..", "a b", "x\n", "-x", "\u0301a",
                                 "x" * 201, "가" * 70, "a\\b", None])
def test_invalid_ids(pid):
    with pytest.raises(ValueError):
        P.check_project_id(pid)


def test_new_project_id_slug_and_suffixes(tmp_path):
    assert P.new_project_id("  My Song: Live!  ", tmp_path) == "my-song-live"
    (tmp_path / "my-song-live").mkdir()
    assert P.new_project_id("My Song: Live!", tmp_path) == "my-song-live-2"
    (tmp_path / "my-song-live-2").mkdir()
    assert P.new_project_id("My Song: Live!", tmp_path) == "my-song-live-3"
    assert P.new_project_id("!!!", tmp_path) == "project"
    assert P.new_project_id("봄날 (Live)", tmp_path) == "봄날-live"
    long = P.new_project_id("가" * 300, tmp_path)
    assert len(long.encode()) <= 96 and P.check_project_id(long) == long


def test_project_dir_validates(tmp_path):
    assert P.project_dir(tmp_path, "abc") == tmp_path / "abc"
    with pytest.raises(ValueError):
        P.project_dir(tmp_path, "../x")


# ---------------------------------------------------------------------------------------- create


@needs_ffmpeg
def test_create_from_wav(tmp_path):
    src = tone(tmp_path / "ref.wav", seconds=2.5, sr=22050, channels=1, freq=440.0 * 2 ** (0.25 / 12))
    root = tmp_path / "projects"
    pdir, warnings = P.create_project(root, title="  First Song ", audio=src, cues_text=CUES, cues_name="show.txt")
    assert warnings == []
    assert pdir == root / "first-song" and visible(root) == ["first-song"]
    info = sf.info(str(pdir / "audio" / "mix.wav"))
    assert (info.samplerate, info.channels, info.subtype) == (44100, 2, "PCM_24")
    assert abs(info.frames / 44100 - 2.5) < 0.01
    assert (pdir / "cues.txt").read_bytes() == CUES[1:].encode("utf-8")     # BOM dropped, CRLF kept
    assert (pdir / "cache").is_dir() and (pdir / "logs").is_dir()
    assert not (pdir / "audio" / "vocals.wav").exists()
    d = json.loads((pdir / "project.json").read_text(encoding="utf-8"))
    assert d["schema"] == 1 and d["id"] == "first-song" and d["title"] == "First Song"
    assert d["style"] == "classic" and d["stems"] == ["mix"]
    assert d["sources"] == {"audio": "ref.wav", "vocals": None, "cues": "show.txt"}
    assert abs(d["duration"] - 2.5) < 0.01
    assert abs(d["tuning"] - 0.25) < 0.05                    # librosa estimate of a +25 cent tone
    assert d["created"].endswith("+00:00")
    meta = P.load_meta(pdir)
    assert meta.id == "first-song" and meta.title == "First Song" and meta.stems == ["mix"]
    assert P.list_projects(root) == [pdir]


@needs_ffmpeg
@pytest.mark.parametrize("ext,codec", [("mp3", "libmp3lame"), ("m4a", "aac")])
def test_create_from_compressed(tmp_path, ext, codec):
    wav = tone(tmp_path / "src.wav", seconds=3.0, sr=48000, channels=2)
    enc = tmp_path / f"ref.{ext}"
    proc = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(wav), "-c:a", codec,
                           "-b:a", "128k", str(enc)], capture_output=True, check=False)
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg cannot encode {codec}: {proc.stderr.decode(errors='replace')[-200:]}")
    pdir, _ = P.create_project(tmp_path / "projects", title="Compressed", audio=enc, style="pop")
    info = sf.info(str(pdir / "audio" / "mix.wav"))
    assert (info.samplerate, info.channels) == (44100, 2)
    assert abs(info.frames / 44100 - 3.0) < 0.1
    meta = P.load_meta(pdir)
    assert meta.style == "pop" and meta.sources["audio"] == f"ref.{ext}" and meta.sources["cues"] is None
    assert P.read_cues(pdir) == ""


@needs_ffmpeg
@pytest.mark.parametrize("voc_s,warn", [(2.0, None), (2.3, None), (1.0, "shorter"), (3.5, "longer")])
def test_create_with_vocals_fits_mix_length(tmp_path, voc_s, warn):
    mix = tone(tmp_path / "mix.wav", seconds=2.0, sr=44100)
    voc = tone(tmp_path / "voc.flac", seconds=voc_s, sr=32000, freq=660.0, amp=0.2)
    pdir, warnings = P.create_project(tmp_path / "projects", title="V", audio=mix, vocals=voc)
    if warn is None:
        assert warnings == []
    else:
        assert len(warnings) == 1 and warn in warnings[0]
    n_mix = sf.info(str(pdir / "audio" / "mix.wav")).frames
    v, sr = sf.read(str(pdir / "audio" / "vocals.wav"), dtype="int32")
    info = sf.info(str(pdir / "audio" / "vocals.wav"))
    assert (len(v), sr, info.channels, info.subtype) == (n_mix, 44100, 2, "PCM_24")
    ref, _ = sf.read(str(to_canonical_wav(voc, tmp_path / "ref_voc.wav")), dtype="int32")
    k = min(len(ref), n_mix)
    assert np.array_equal(v[:k], ref[:k])                    # cut / padded losslessly
    assert not v[k:].any()
    meta = P.load_meta(pdir)
    assert meta.stems == ["mix", "vocals"] and meta.sources["vocals"] == "voc.flac"


@needs_ffmpeg
def test_id_collisions(tmp_path):
    root = tmp_path / "projects"
    src = tone(tmp_path / "a.wav", seconds=1.0)
    a, _ = P.create_project(root, title="Same", audio=src)
    b, _ = P.create_project(root, title="Same", audio=src)
    assert (a.name, b.name) == ("same", "same-2")
    assert P.load_meta(b).id == "same-2"
    c, _ = P.create_project(root, title="Other", audio=src, project_id="custom.id")
    assert c.name == "custom.id"
    with pytest.raises(ValueError, match="already exists"):
        P.create_project(root, title="Other", audio=src, project_id="custom.id")
    with pytest.raises(ValueError, match="already exists"):
        P.create_project(root, title="Other", audio=src, project_id="same")
    with pytest.raises(ValueError, match="invalid project id"):
        P.create_project(root, title="Other", audio=src, project_id="../evil")
    assert visible(root) == ["custom.id", "same", "same-2"]


@needs_ffmpeg
def test_id_taken_while_converting(tmp_path, monkeypatch):
    """The id is claimed only at the end: a folder that appeared meanwhile is never overwritten."""
    root = tmp_path / "projects"
    src = tone(tmp_path / "a.wav", seconds=1.0)

    def squat(name):
        def fake_tuning(mix):
            (root / name).mkdir()
            (root / name / "keep.txt").write_text("other")
            return 0.0
        return fake_tuning

    monkeypatch.setattr(P, "estimate_tuning", squat("race"))
    pdir, _ = P.create_project(root, title="Race", audio=src)
    assert pdir.name == "race-2" and P.load_meta(pdir).id == "race-2"
    assert json.loads((pdir / "project.json").read_text())["id"] == "race-2"
    monkeypatch.setattr(P, "estimate_tuning", squat("fixed"))
    with pytest.raises(ValueError, match="already exists"):
        P.create_project(root, title="Fixed", audio=src, project_id="fixed")
    assert (root / "fixed" / "keep.txt").read_text() == "other"
    assert visible(root) == ["fixed", "race", "race-2"]


@needs_ffmpeg
def test_failed_conversion_leaves_nothing(tmp_path):
    root = tmp_path / "projects"
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"this is not audio at all" * 100)
    with pytest.raises(AudioError):
        P.create_project(root, title="Broken", audio=bad, cues_text="1.0 /x")
    assert visible(root) == []
    good = tone(tmp_path / "ok.wav", seconds=1.0)
    with pytest.raises(AudioError):                           # vocals fail after the mix converted
        P.create_project(root, title="Broken", audio=good, vocals=bad)
    assert visible(root) == []


@needs_ffmpeg
def test_failure_after_conversion_leaves_nothing(tmp_path, monkeypatch):
    root = tmp_path / "projects"

    def boom(mix):
        raise RuntimeError("tuning crashed")

    monkeypatch.setattr(P, "estimate_tuning", boom)
    with pytest.raises(RuntimeError, match="tuning crashed"):
        P.create_project(root, title="X", audio=tone(tmp_path / "a.wav", seconds=1.0))
    assert visible(root) == []


# ------------------------------------------------------------------------------------------ tuning


def music(seconds: float, cents: float = 25.0, sr: int = 44100, seed: int = 0) -> np.ndarray:
    """Detuned harmonic notes switching on and off over quiet noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * sr)) / sr
    y = 0.02 * rng.standard_normal(t.size)
    for k, midi in enumerate((57, 60, 64, 69)):
        f = 440.0 * 2 ** ((midi - 69 + cents / 100) / 12)
        gate = np.sin(2 * np.pi * (0.3 + 0.1 * k) * t) > 0
        for h in (1, 2, 3):
            y += gate * (0.05 / h) * np.sin(2 * np.pi * f * h * t)
    return y


def whole_mix_tuning(mix: Path) -> float:
    """The estimate on the whole mix in memory (what estimate_tuning computed before it streamed)."""
    import librosa

    from lyricprompter_oscsender.audio_io import load_mono

    p = P.TUNING_PARAMS
    y = load_mono(mix, p["sr"])
    if y.size < p["n_fft"] or not np.any(y):
        return 0.0
    return round(float(librosa.estimate_tuning(y=y, sr=p["sr"], n_fft=p["n_fft"], hop_length=p["hop_length"],
                                               resolution=p["resolution"],
                                               bins_per_octave=p["bins_per_octave"])), 3)


@pytest.mark.parametrize("seconds, cents", [(0.05, 0.0), (0.3, 30.0), (7.3, 25.0), (31.0, -40.0)])
def test_tuning_is_the_whole_mix_estimate(tmp_path, monkeypatch, seconds, cents):
    mix = make_bare_project(tmp_path, signal=music(seconds, cents)) / "audio" / "mix.wav"
    expected = whole_mix_tuning(mix)
    assert P.estimate_tuning(mix) == expected
    monkeypatch.setattr(P, "_TUNING_CHUNK_FRAMES", 7)                 # many chunk boundaries
    assert P.estimate_tuning(mix) == expected
    if seconds > 1:
        assert abs(expected - cents / 100) < 0.05


def test_tuning_of_silence_or_a_tiny_mix_is_zero(tmp_path):
    assert P.estimate_tuning(make_bare_project(tmp_path, "a", signal=np.zeros(44100 * 3)) / "audio" / "mix.wav") == 0
    assert P.estimate_tuning(make_bare_project(tmp_path, "b", signal=music(0.05)) / "audio" / "mix.wav") == 0


def test_tuning_memory_does_not_grow_with_the_mix_length(tmp_path):
    """A whole-show mix must not load the whole file and its spectrogram into the server (it took about
    85 MB per minute of audio)."""
    import tracemalloc

    short = make_bare_project(tmp_path, "short", signal=music(2.0)) / "audio" / "mix.wav"
    long = make_bare_project(tmp_path, "long", signal=music(150.0)) / "audio" / "mix.wav"
    P.estimate_tuning(short)                                          # imports, JIT compilation
    tracemalloc.start()
    try:
        P.estimate_tuning(long)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 40e6, f"{peak / 1e6:.0f} MB"


def test_bad_input_is_rejected_before_any_work(tmp_path):
    root = tmp_path / "projects"
    audio = tone(tmp_path / "a.wav", seconds=0.5)
    with pytest.raises(FileNotFoundError, match="audio file not found"):
        P.create_project(root, title="T", audio=tmp_path / "missing.wav")
    with pytest.raises(FileNotFoundError, match="vocals file not found"):
        P.create_project(root, title="T", audio=audio, vocals=tmp_path / "missing.wav")
    with pytest.raises(ValueError, match="title"):
        P.create_project(root, title="   ", audio=audio)
    with pytest.raises(ValueError, match="style"):
        P.create_project(root, title="T", audio=audio, style="jazz")
    with pytest.raises(ValueError, match="invalid project id"):
        P.create_project(root, title="T", audio=audio, project_id=".x")
    with pytest.raises(TypeError):
        P.create_project(root, title="T", audio=audio, cues_text=None)
    assert visible(root) == []


@needs_ffmpeg
def test_stale_build_folders_are_swept(tmp_path):
    root = tmp_path / "projects"
    old, recent = root / ".tmp-old-123", root / ".tmp-new-456"
    old.mkdir(parents=True)
    recent.mkdir()
    long_ago = time.time() - P.STALE_TMP_S - 60
    os.utime(old, (long_ago, long_ago))
    P.create_project(root, title="T", audio=tone(tmp_path / "a.wav", seconds=0.5))
    assert not old.exists() and recent.exists()


# ------------------------------------------------------------------------------------------ meta


def test_meta_roundtrip_and_defaults(tmp_path):
    pdir = make_bare_project(tmp_path, "song", seconds=0.5)
    meta = P.load_meta(pdir)
    assert (meta.id, meta.title, meta.style, meta.stems) == ("song", "SONG", "classic", ["mix"])
    meta.style, meta.tuning = "pop", -0.12
    P.save_meta(pdir, meta)
    again = P.load_meta(pdir)
    assert again == meta
    assert not [p for p in pdir.iterdir() if p.name.endswith(".tmp")]

    (pdir / "project.json").write_text(json.dumps({"schema": 1}))
    m = P.load_meta(pdir)
    assert (m.id, m.title, m.created, m.tuning, m.style) == ("song", "song", "", 0.0, "classic")
    assert abs(m.duration - 0.5) < 1e-6                       # measured from the mix
    assert m.stems == ["mix"] and m.sources == {"audio": None, "vocals": None, "cues": None}

    (pdir / "project.json").write_text(json.dumps({"schema": 1, "id": "elsewhere", "title": 3,
                                                   "duration": "x", "tuning": None,
                                                   "style": "metal", "stems": ["mix", "drums"],
                                                   "sources": {"audio": "a.wav", "extra": 1}}))
    m = P.load_meta(pdir)
    assert m.id == "song" and m.title == "song" and m.style == "classic" and m.stems == ["mix"]
    assert m.sources["audio"] == "a.wav" and m.sources["extra"] == 1


@pytest.mark.parametrize("content,match", [("{not json", "not valid JSON"), ("[1, 2]", "schema"),
                                           ('{"schema": 2}', "schema"), ('{"title": "x"}', "schema")])
def test_meta_errors(tmp_path, content, match):
    pdir = tmp_path / "p"
    pdir.mkdir()
    (pdir / "project.json").write_text(content)
    with pytest.raises(ValueError, match=match):
        P.load_meta(pdir)


def test_meta_missing_and_invalid_save(tmp_path):
    with pytest.raises(FileNotFoundError):
        P.load_meta(tmp_path)
    meta = P.ProjectMeta(id="x", title="x", created="", duration=1.0, tuning=0.0, style="nope")
    with pytest.raises(ValueError):
        P.save_meta(tmp_path, meta)
    with pytest.raises(ValueError):
        P.save_meta(tmp_path, P.ProjectMeta(id="x", title="x", created="", duration=1.0, tuning=0.0,
                                            stems=["mix", "bass"]))
    assert not (tmp_path / "project.json").exists()


def test_list_projects(tmp_path):
    root = tmp_path / "projects"
    assert P.list_projects(root) == []
    for pid in ("b", "a", "c"):
        make_bare_project(root, pid, seconds=0.1)
    (root / "no-meta").mkdir()
    (root / ".trash" / "old").mkdir(parents=True)
    (root / ".trash" / "old" / "project.json").write_text('{"schema": 1}')
    (root / ".tmp-x-1").mkdir()
    (root / ".tmp-x-1" / "project.json").write_text('{"schema": 1}')
    (root / "file.json").write_text("{}")
    assert [p.name for p in P.list_projects(root)] == ["a", "b", "c"]


# ------------------------------------------------------------------------------------------ cues


def test_cues_roundtrip_keeps_text(tmp_path):
    pdir = make_bare_project(tmp_path, "p", seconds=0.1)
    assert P.cues_path(pdir) == pdir / "cues.txt"
    text = "1.0 /a\r\n2.0 /b # 한글 note\n\n3.0\t/c"
    P.write_cues(pdir, text)
    assert P.cues_path(pdir).read_bytes() == text.encode("utf-8")
    assert P.read_cues(pdir) == text
    P.write_cues(pdir, "\ufeff1.0 /a\n")
    assert P.cues_path(pdir).read_bytes() == b"1.0 /a\n"
    P.cues_path(pdir).write_bytes(b"\xef\xbb\xbf0.5 /x\r\n")   # BOM written by an editor
    assert P.read_cues(pdir) == "0.5 /x\r\n"
    assert not [p.name for p in pdir.iterdir() if p.name.startswith(".")]


def test_cues_errors_and_missing(tmp_path):
    pdir = make_bare_project(tmp_path, "p", seconds=0.1)
    P.cues_path(pdir).unlink()
    assert P.read_cues(pdir) == ""
    P.cues_path(pdir).write_bytes(b"1.0 /caf\xe9\n")
    with pytest.raises(ValueError, match="not UTF-8"):
        P.read_cues(pdir)
    with pytest.raises(TypeError):
        P.write_cues(pdir, b"bytes")
    with pytest.raises(FileNotFoundError):
        P.write_cues(tmp_path / "nope", "x")


def test_write_cues_through_symlink(tmp_path):
    pdir = make_bare_project(tmp_path, "p", seconds=0.1)
    real = tmp_path / "shared-cues.txt"
    real.write_text("old")
    P.cues_path(pdir).unlink()
    P.cues_path(pdir).symlink_to(real)
    P.write_cues(pdir, "new\n")
    assert P.cues_path(pdir).is_symlink() and real.read_text() == "new\n"


# ---------------------------------------------------------------------------------------- vocals


@needs_ffmpeg
def test_set_vocals_adds_and_replaces(tmp_path):
    root = tmp_path / "projects"
    pdir, _ = P.create_project(root, title="S", audio=tone(tmp_path / "m.wav", seconds=2.0))
    (pdir / "cache" / "feat_abc_vocals.npz").write_bytes(b"x")
    (pdir / "cache" / "feat_abc_mix.npz").write_bytes(b"x")
    (pdir / "cache" / "peaks_vocals_2000.json").write_text("{}")
    warnings = P.set_vocals(pdir, tone(tmp_path / "v1.wav", seconds=2.0, freq=330.0))
    assert warnings == []
    meta = P.load_meta(pdir)
    assert meta.stems == ["mix", "vocals"] and meta.sources["vocals"] == "v1.wav"
    assert sorted(p.name for p in (pdir / "cache").iterdir()) == ["feat_abc_mix.npz"]
    warnings = P.set_vocals(pdir, tone(tmp_path / "v2.wav", seconds=1.0, freq=330.0))
    assert len(warnings) == 1 and "shorter" in warnings[0]
    assert sf.info(str(pdir / "audio" / "vocals.wav")).frames == sf.info(str(pdir / "audio" / "mix.wav")).frames
    assert P.load_meta(pdir).sources["vocals"] == "v2.wav"
    assert sorted(p.name for p in (pdir / "audio").iterdir()) == ["mix.wav", "vocals.wav"]
    with pytest.raises(FileNotFoundError):
        P.set_vocals(pdir, tmp_path / "missing.wav")
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"junk" * 50)
    before = (pdir / "audio" / "vocals.wav").read_bytes()
    with pytest.raises(AudioError):
        P.set_vocals(pdir, bad)
    assert (pdir / "audio" / "vocals.wav").read_bytes() == before
    assert sorted(p.name for p in (pdir / "audio").iterdir()) == ["mix.wav", "vocals.wav"]


def test_set_vocals_needs_a_project(tmp_path):
    with pytest.raises(FileNotFoundError):
        P.set_vocals(tmp_path, tone(tmp_path / "v.wav", seconds=0.2))


# -------------------------------------------------------------------------------------- waveform


def test_waveform_peaks(tmp_path):
    sr = 44100
    y = np.zeros(sr * 2)
    y[sr:] = 0.5 * np.sin(2 * np.pi * 100 * np.arange(sr) / sr)     # silence, then a 100 Hz tone
    pdir = make_bare_project(tmp_path, "w", signal=y)
    wf = P.waveform(pdir, points=200)
    assert set(wf) == {"duration", "points", "min", "max"}
    assert wf["points"] == 200 and len(wf["min"]) == len(wf["max"]) == 200
    assert wf["duration"] == pytest.approx(2.0)
    assert all(isinstance(v, float) for v in wf["min"] + wf["max"])
    assert max(abs(v) for v in wf["max"][:99] + wf["min"][:99]) == 0.0
    assert all(0.49 <= v <= 0.5 for v in wf["max"][101:])
    assert all(-0.5 <= v <= -0.49 for v in wf["min"][101:])
    assert all(lo <= hi for lo, hi in zip(wf["min"], wf["max"]))
    assert all(v == round(v, 4) for v in wf["max"])


def test_waveform_matches_brute_force(tmp_path):
    rng = np.random.default_rng(3)
    n, points = 44100 * 3 + 777, 1234
    y = 0.4 * rng.standard_normal(n).clip(-2, 2) / 2
    pdir = make_bare_project(tmp_path, "w", signal=y)
    wf = P.waveform(pdir, points=points)
    mono = sf.read(str(pdir / "audio" / "mix.wav"), dtype="float32")[0].mean(axis=1)
    edges = -(-np.arange(points + 1) * n // points)          # bin k: frames f with f * points // n == k
    lo = [mono[a:b].min() for a, b in itertools.pairwise(edges)]
    hi = [mono[a:b].max() for a, b in itertools.pairwise(edges)]
    np.testing.assert_allclose(wf["min"], lo, atol=6e-5)
    np.testing.assert_allclose(wf["max"], hi, atol=6e-5)


def test_waveform_short_file_and_clamping(tmp_path):
    pdir = make_bare_project(tmp_path, "w", signal=np.linspace(-0.5, 0.5, 30))
    wf = P.waveform(pdir, points=50)                          # clamped up to 100 bins > 30 frames
    assert wf["points"] == 100 and len(wf["max"]) == 100
    assert all(np.isfinite(wf["min"])) and all(np.isfinite(wf["max"]))
    assert wf["min"][0] == pytest.approx(-0.5, abs=1e-3) and wf["max"][-1] == pytest.approx(0.5, abs=1e-3)
    assert P.waveform(pdir, points=10 ** 6)["points"] == 20000


def test_waveform_cache_and_invalidation(tmp_path, monkeypatch):
    pdir = make_bare_project(tmp_path, "w", seconds=1.0)
    first = P.waveform(pdir, points=300)
    cache = pdir / "cache" / "peaks_mix_300.json"
    assert cache.is_file()
    calls = []
    real = P._peaks
    monkeypatch.setattr(P, "_peaks", lambda path, points: calls.append(points) or real(path, points))
    assert P.waveform(pdir, points=300) == first and calls == []          # served from the cache
    P.waveform(pdir, points=400)
    assert calls == [400]
    sr = 44100
    louder = 0.8 * np.sin(2 * np.pi * 220 * np.arange(sr // 2) / sr)
    sf.write(str(pdir / "audio" / "mix.wav"), np.stack([louder, louder], 1), sr, subtype="PCM_24")
    second = P.waveform(pdir, points=300)
    assert calls == [400, 300]
    assert second["duration"] == pytest.approx(0.5) and max(second["max"]) > 0.79
    cache.write_text("{broken")
    assert P.waveform(pdir, points=300) == second and calls == [400, 300, 300]


def test_waveform_errors(tmp_path):
    pdir = make_bare_project(tmp_path, "w", seconds=0.2)
    with pytest.raises(ValueError, match="unknown stem"):
        P.waveform(pdir, stem="drums")
    with pytest.raises(FileNotFoundError):
        P.waveform(pdir, stem="vocals")


# ----------------------------------------------------------------------------------------- trash


def test_trash_project(tmp_path):
    root = tmp_path / "projects"
    make_bare_project(root, "gone", seconds=0.1)
    make_bare_project(root, "kept", seconds=0.1)
    dst = P.trash_project(root, "gone")
    assert dst.parent == root / ".trash" and dst.name.startswith("gone-") and (dst / "project.json").is_file()
    assert [p.name for p in P.list_projects(root)] == ["kept"]
    make_bare_project(root, "gone", seconds=0.1)
    again = P.trash_project(root, "gone")
    assert again != dst and again.is_dir() and dst.is_dir()
    with pytest.raises(FileNotFoundError):
        P.trash_project(root, "gone")
    with pytest.raises(ValueError):
        P.trash_project(root, ".trash")
    with pytest.raises(ValueError):
        P.trash_project(root, "../projects")


# ------------------------------------------------------------------------------------------- CLI


@needs_ffmpeg
def test_cli_add_and_list(tmp_path, capsys):
    root = tmp_path / "projects"
    audio = tone(tmp_path / "ref.wav", seconds=1.0)
    cues = tmp_path / "show.txt"
    cues.write_bytes(CUES.encode("utf-8"))
    rc = P.cli_add(["--audio", str(audio), "--cues", str(cues), "--title", "Show One", "--style", "pop",
                    "--projects-dir", str(root)])
    out = capsys.readouterr()
    assert rc == 0, out.err
    pdir = root / "show-one"
    assert out.out.strip().splitlines()[-1] == str(pdir.resolve())
    assert (pdir / "cues.txt").read_bytes() == CUES[1:].encode("utf-8")
    meta = P.load_meta(pdir)
    assert meta.style == "pop" and meta.sources["cues"] == "show.txt"

    rc = P.cli_add(["--audio", str(audio), "--cues", str(cues), "--title", "X", "--id", "show-one",
                    "--projects-dir", str(root)])
    assert rc == 1 and "already exists" in capsys.readouterr().err
    rc = P.cli_add(["--audio", str(tmp_path / "nope.wav"), "--cues", str(cues), "--title", "X",
                    "--projects-dir", str(root)])
    assert rc == 1 and "not found" in capsys.readouterr().err
    rc = P.cli_add(["--audio", str(audio), "--cues", str(tmp_path / "nope.txt"), "--title", "X",
                    "--projects-dir", str(root)])
    assert rc == 1 and "cue file not found" in capsys.readouterr().err
    latin = tmp_path / "latin1.txt"
    latin.write_bytes(b"1.0 /caf\xe9\n")
    rc = P.cli_add(["--audio", str(audio), "--cues", str(latin), "--title", "X", "--projects-dir", str(root)])
    assert rc == 1 and "UTF-8" in capsys.readouterr().err
    assert P.cli_add(["--audio", str(audio)]) == 2           # argparse usage error (no --title)
    assert P.cli_add(["--audio", str(audio), "--title", "No cues yet", "--projects-dir", str(root)]) == 0
    assert P.read_cues(root / "no-cues-yet") == ""           # an empty cue file, filled in the console
    shutil.rmtree(root / "no-cues-yet")
    capsys.readouterr()
    assert visible(root) == [".settings.json", "show-one"]     # + the console settings with the @target added

    assert P.cli_list(["--projects-dir", str(root)]) == 0
    out = capsys.readouterr().out
    assert "show-one" in out and "Show One" in out and "pop" in out
    assert P.cli_list(["--projects-dir", str(root), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["id"] for r in rows] == ["show-one"]
    assert rows[0]["title"] == "Show One" and rows[0]["stems"] == ["mix"]


def test_cli_add_adds_the_declared_targets_to_the_console_settings(tmp_path, capsys):
    from lyricprompter_oscsender.settings import load_settings, settings_path

    root = tmp_path / "projects"
    audio = tone(tmp_path / "ref.wav", seconds=1.0)
    cues = tmp_path / "show.txt"
    cues.write_text("@target lights 10.0.0.5:8000 latency=20ms\n0:00.5 lights /cue/go 1\n", encoding="utf-8")
    assert P.cli_add(["--audio", str(audio), "--cues", str(cues), "--title", "T", "--projects-dir", str(root)]) == 0
    assert "added OSC targets from the cue file: lights" in capsys.readouterr().out
    t = load_settings(settings_path(root)).target("lights")
    assert (t.host, t.port, t.latency_ms) == ("10.0.0.5", 8000, 20.0)


def test_cli_list_default_root_and_errors(tmp_path, monkeypatch, capsys):
    root = tmp_path / "projects"
    monkeypatch.setattr(config, "PROJECTS_DIR", root)
    assert P.cli_list([]) == 0
    assert "no projects" in capsys.readouterr().out
    make_bare_project(root, "good", seconds=0.1)
    bad = root / "bad"
    bad.mkdir()
    (bad / "project.json").write_text("{oops")
    assert cli_main(["list"]) == 0
    out = capsys.readouterr().out
    assert "good" in out and "bad" in out and "error" in out
    assert cli_main(["list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["id"] for r in rows] == ["bad", "good"] and "error" in rows[0]
    assert P.cli_list(["--help"]) == 0
