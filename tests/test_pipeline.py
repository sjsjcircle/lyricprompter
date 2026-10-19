from __future__ import annotations

import io
import json
import re
import shutil
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lyricprompter import cli as top_cli
from lyricprompter import config
from lyricprompter.audio_io import read_audio, save_wav
from lyricprompter.prep import pipeline as P
from lyricprompter.prep import separate as S
from lyricprompter.songfile import load_song
from lyricprompter.text.model import AlignUnit, Line, Syllable, Word

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

LYRICS = "Hello world\n\nsing it again\n"
TUNING = 0.2
PROGRESS_RE = re.compile(r"^PROGRESS (convert|parse|separate|align|finalize) (\d{1,3}) (.*)$")


# ------------------------------------------------------------------------------------------ fakes

def fake_parse(raw: str, default_lang: str = "auto") -> list[Line]:
    lines, stanza = [], 0
    for row in raw.splitlines():
        if not row.strip():
            stanza += 1
            continue
        words, pos = [], 0
        for w in row.split():
            a = row.index(w, pos)
            pos = a + len(w)
            unit = AlignUnit(reading=w, roman=re.sub(r"[^a-z']", "", w.lower()))
            words.append(Word(text=w, span=(a, pos), lang="en",
                              syllables=[Syllable(text=w, span=(a, pos), units=[unit])]))
        lines.append(Line(idx=len(lines), text=row, raw=row, stanza=stanza, words=words))
    return lines


@dataclass
class FakeReport:
    method: str = "fake"
    line_confidence: list[float] = field(default_factory=list)
    flagged: list[int] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    details: dict = field(default_factory=dict)


def fake_align(lines: list[Line], vocals: np.ndarray, sr: int, refine: str, progress) -> FakeReport:
    assert sr == config.ANALYSIS_SR and vocals.dtype == np.float32 and vocals.ndim == 1
    t = 0.5
    for li, ln in enumerate(lines):
        for w in ln.words:
            for s in w.syllables:
                s.start, s.end = t, t + 0.2
                for u in s.units:
                    u.start, u.end, u.score = np.float64(t), t + 0.2, np.float32(0.9)
                t += 0.25
        ln.start, ln.end = ln.words[0].start, ln.words[-1].end
        ln.confidence = 0.05 if li == 1 else 0.8
        ln.needs_review = li == 1
    for pct in (0, 50, 100):
        progress("emission", pct, f"emission {pct}")
    return FakeReport(line_confidence=[ln.confidence for ln in lines], flagged=[1],
                      timings={"emission": 0.01}, details={"refine": refine, "np": np.float32(1.5)})


def fake_separate(mix_wav: Path, out_dir: Path, *, model: str = S.DEFAULT_SEP_MODEL, lead: bool = False,
                  progress=None) -> dict[str, Path]:
    y, sr = read_audio(mix_wav)
    out = {"vocals": save_wav(out_dir / "vocals.wav", 0.5 * y, sr),
           "instrumental": save_wav(out_dir / "instrumental.wav", 0.5 * y, sr)}
    if lead:
        out["lead"] = save_wav(out_dir / "lead.wav", 0.25 * y, sr)
    if progress:
        progress("separate", 50, "half")
    S.write_manifest(out_dir, model=model, mix_wav=mix_wav, stems=out,
                     lead_model=S.LEAD_SEP_MODEL if lead else None)
    return out


class _Inst:
    output_dir: str | None = None


def fake_separator(calls: list[str]) -> object:
    """Fake audio-separator Separator for any model: '(Vocals)' = 0.25x and '(Other)' = 0.75x the input."""
    class Sep:
        model_instance = _Inst()

        def separate(self, path: str) -> list[str]:
            calls.append(path)
            y, sr = sf.read(path, dtype="float32")
            for stem, g in (("Vocals", 0.25), ("Other", 0.75)):
                sf.write(Path(self.model_instance.output_dir) / f"mix_({stem})_m.wav", g * y, sr)
            return ["mix_(Vocals)_m.wav", "mix_(Other)_m.wav"]
    return Sep()


@pytest.fixture()
def patched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(P, "_parse", fake_parse)
    monkeypatch.setattr(P, "_align", fake_align)
    monkeypatch.setattr(P, "separate_vocals", fake_separate)


@pytest.fixture()
def audio(tmp_path: Path) -> Path:
    """3 s mono 48 kHz tone at A440 + 0.2 semitone (exercises conversion and tuning)."""
    sr = 48000
    t = np.arange(3 * sr) / sr
    f = 440.0 * 2 ** (TUNING / 12)
    y = 0.3 * (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 1.5 * f * t))
    p = tmp_path / "in" / "song.wav"
    p.parent.mkdir()
    sf.write(p, y.astype(np.float32), sr)
    return p


# ------------------------------------------------------------------------------------------ tests

def test_prepare_song_package(patched: None, audio: Path, tmp_path: Path) -> None:
    events: list[tuple[str, float, str]] = []
    root = tmp_path / "songs"
    sdir = P.prepare_song(audio, LYRICS, title="Hello World!", artist="Me", default_lang="en", songs_root=root,
                          lead=True, refine="none", progress=lambda *a: events.append(a))
    assert sdir == root / "hello-world"
    assert (sdir / "lyrics.txt").read_text(encoding="utf-8") == LYRICS

    song = load_song(sdir)
    m = song.meta
    assert (m.id, m.title, m.artist, m.default_lang) == ("hello-world", "Hello World!", "Me", "en")
    assert m.duration == pytest.approx(3.0, abs=1e-3)
    assert m.tuning == pytest.approx(TUNING, abs=0.03)
    assert m.stems == {"mix": "audio/mix.wav", "vocals": "audio/vocals.wav",
                       "instrumental": "audio/instrumental.wav", "lead": "audio/lead.wav"}
    mix_frames = sf.info(sdir / "audio/mix.wav").frames
    for rel in m.stems.values():
        info = sf.info(sdir / rel)
        assert (info.samplerate, info.channels, info.frames) == (44100, 2, mix_frames)
    assert m.prep["options"] == {"default_lang": "en", "sep_model": S.DEFAULT_SEP_MODEL, "lead": True,
                                 "refine": "none", "vocals": None}
    assert m.prep["models"] == {"separation": S.DEFAULT_SEP_MODEL, "lead": S.LEAD_SEP_MODEL, "align": "fake"}
    assert m.prep["packages"]["numpy"] == np.__version__ and m.prep["packages"]["lyricprompter"]
    assert set(m.prep["timings"]) == {"convert", "parse", "separate", "align", "finalize", "total"}
    assert m.prep["tuning_params"]["sr"] == 16000 and m.prep["languages"] == {"en": 5}

    assert [ln.text for ln in song.lines] == ["Hello world", "sing it again"]
    assert song.lines[1].stanza == 1 and song.lines[1].needs_review
    assert song.lines[0].start == 0.5
    assert song.lines[1].words[2].syllables[0].units[0].score == pytest.approx(0.9)

    rep = json.loads((sdir / P.REPORT_FILE).read_text(encoding="utf-8"))
    assert rep["status"] == "ok" and rep["song_id"] == "hello-world"
    assert rep["align"]["flagged"] == [1] and rep["align"]["details"]["np"] == 1.5
    assert rep["summary"]["lines"] == 2 and rep["summary"]["flagged"] == 1
    assert rep["summary"]["min_confidence"] == 0.05
    assert [r["confidence"] for r in rep["lines"]] == [0.8, 0.05]
    assert rep["separation"]["model"] == S.DEFAULT_SEP_MODEL and rep["warnings"] == []

    stages = [e[0] for e in events]
    assert stages[0] == "convert" and stages[-1] == "finalize"
    assert [s for i, s in enumerate(stages) if i == 0 or stages[i - 1] != s] == list(P.STAGE_BANDS)
    pcts = [e[1] for e in events]
    assert pcts == sorted(pcts) and pcts[-1] == 100
    assert ("align", pytest.approx(77.5), "emission: emission 50") in events     # sub-progress mapped


def test_supplied_vocals_skip_separation(patched: None, audio: Path, tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(P, "separate_vocals", lambda *a, **k: pytest.fail("separation must be skipped"))
    voc = tmp_path / "in" / "vocals.flac"
    y, sr = sf.read(audio, dtype="float32")
    sf.write(voc, 0.4 * y[: -sr // 2], sr)                      # 0.5 s short, mono, 48 kHz
    sdir = P.prepare_song(audio, LYRICS, title="x", songs_root=tmp_path / "songs", vocals=voc)
    mix, _ = read_audio(sdir / "audio/mix.wav")
    v, _ = read_audio(sdir / "audio/vocals.wav")
    i, _ = read_audio(sdir / "audio/instrumental.wav")
    assert v.shape == i.shape == mix.shape
    np.testing.assert_allclose(v + i, mix, atol=1e-5)
    assert not np.any(v[-sr // 4:])
    song = load_song(sdir)
    assert set(song.meta.stems) == {"mix", "vocals", "instrumental"}
    assert song.meta.prep["models"]["separation"] == "external"
    assert song.meta.prep["options"]["vocals"] == str(voc)
    man = S.read_manifest(sdir / "audio")
    assert man["model"] == "external" and man["source"] == str(voc)
    assert S.cached_stems(sdir / "audio", sdir / "audio/mix.wav", "external") is not None
    rep = json.loads((sdir / P.REPORT_FILE).read_text())      # a pure tone has no unique lag: flagged
    assert not rep["separation"]["vocals_check"]["matched"]
    assert any("could not be lined up" in w for w in rep["warnings"])


def test_stems_reused_on_reprep(audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real separate_vocals with a fake Separator: re-running with --id reuses audio/stems.json."""
    monkeypatch.setattr(P, "_parse", fake_parse)
    monkeypatch.setattr(P, "_align", fake_align)
    calls: list[str] = []
    sep = fake_separator(calls)
    monkeypatch.setattr(S, "get_separator", lambda model: sep)
    root = tmp_path / "songs"
    first = P.prepare_song(audio, LYRICS, title="t", songs_root=root)
    assert len(calls) == 1
    feat = first / "cache" / "feat_x.npz"
    feat.parent.mkdir()
    feat.write_bytes(b"x")
    second = P.prepare_song(audio, "Hello again", title="t", song_id="t", songs_root=root)
    assert second == first and len(calls) == 1
    rep = json.loads((second / P.REPORT_FILE).read_text())
    assert rep["separation"]["reused"] is True
    assert [ln.text for ln in load_song(second).lines] == ["Hello again"]
    assert feat.exists()                                  # same audio: feature cache stays valid
    P.prepare_song(audio, LYRICS, title="t", song_id="t", songs_root=root, sep_model="other.ckpt")
    assert len(calls) == 2
    assert not feat.exists()                              # new stems: stale feature cache dropped


def test_ids_and_validation(patched: None, audio: Path, tmp_path: Path) -> None:
    root = tmp_path / "songs"
    a = P.prepare_song(audio, LYRICS, title="Same", songs_root=root)
    b = P.prepare_song(audio, LYRICS, title="Same", songs_root=root)
    assert (a.name, b.name) == ("same", "same-2")
    with pytest.raises(P.PrepError, match="invalid song id"):
        P.prepare_song(audio, LYRICS, title="x", song_id="../evil", songs_root=root)
    with pytest.raises(P.PrepError, match="empty"):
        P.prepare_song(audio, "  \n", title="x", songs_root=root)
    with pytest.raises(P.PrepError, match="language"):
        P.prepare_song(audio, LYRICS, title="x", default_lang="fr", songs_root=root)
    with pytest.raises(FileNotFoundError):
        P.prepare_song(tmp_path / "missing.wav", LYRICS, title="x", songs_root=root)
    assert sorted(p.name for p in root.iterdir()) == ["same", "same-2"]


def test_first_prep_failure_before_stems_frees_the_id(patched: None, audio: Path, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "songs"
    monkeypatch.setattr(P, "_parse", lambda raw, lang: [Line(idx=0, text="...", words=[])])
    with pytest.raises(P.PrepError, match="no alignable"):
        P.prepare_song(audio, "...", title="bad", songs_root=root)
    with pytest.raises(P.PrepError, match="no alignable"):
        P.prepare_song(audio, "...", title="bad", song_id="chosen", songs_root=root)
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(b"\x00garbage" * 500)
    monkeypatch.setattr(P, "_parse", fake_parse)
    with pytest.raises(Exception, match="junk.mp3"):
        P.prepare_song(junk, LYRICS, title="junk", song_id="junk", songs_root=root)
    monkeypatch.setattr(P, "separate_vocals", lambda *a, **k: (_ for _ in ()).throw(P.Terminated()))
    with pytest.raises(P.Terminated):                     # cancelled while the model loads
        P.prepare_song(audio, LYRICS, title="cancel", song_id="cancel", songs_root=root)
    assert list(root.iterdir()) == []                     # every id is free again

    keep = root / "keep"                                  # a directory this run did not create stays
    keep.mkdir()
    (keep / "notes.txt").write_text("mine")
    monkeypatch.setattr(P, "_parse", lambda raw, lang: [Line(idx=0, text="...", words=[])])
    with pytest.raises(P.PrepError):
        P.prepare_song(audio, "...", title="k", song_id="keep", songs_root=root)
    rep = json.loads((keep / P.REPORT_FILE).read_text())
    assert rep["status"] == "failed" and rep["stage"] == "parse" and "no alignable" in rep["error"]
    assert (keep / "notes.txt").read_text() == "mine"


def test_first_prep_failure_after_stems_keeps_them_for_retry(audio: Path, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    sep = fake_separator(calls)
    monkeypatch.setattr(S, "get_separator", lambda model: sep)
    monkeypatch.setattr(P, "_parse", fake_parse)
    monkeypatch.setattr(P, "_align", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("align crashed")))
    root = tmp_path / "songs"
    with pytest.raises(RuntimeError, match="align crashed"):
        P.prepare_song(audio, LYRICS, title="t", song_id="t", songs_root=root)
    sdir = root / "t"
    rep = json.loads((sdir / P.REPORT_FILE).read_text())
    assert rep["status"] == "failed" and rep["stage"] == "align" and "align crashed" in rep["error"]
    assert not (sdir / "song.json").exists() and (sdir / "audio/vocals.wav").is_file()
    monkeypatch.setattr(P, "_align", fake_align)
    assert P.prepare_song(audio, LYRICS, title="t", song_id="t", songs_root=root) == sdir
    assert len(calls) == 1                                # the retry reused the stems
    rep = json.loads((sdir / P.REPORT_FILE).read_text())
    assert rep["status"] == "ok" and rep["separation"]["reused"] is True


def test_failed_reprep_keeps_previous_package(patched: None, audio: Path, tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "songs"
    sdir = P.prepare_song(audio, LYRICS, title="t", songs_root=root, lead=True)
    snapshot = {name: (sdir / name).read_bytes() for name in ("song.json", "lyrics.txt", P.REPORT_FILE)}
    inodes = {f.name: f.stat().st_ino for f in (sdir / "audio").iterdir()}
    monkeypatch.setattr(P, "_align", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("align crashed")))
    with pytest.raises(RuntimeError, match="align crashed"):
        P.prepare_song(audio, "Other words", title="t", song_id="t", songs_root=root)
    assert {name: (sdir / name).read_bytes() for name in snapshot} == snapshot    # report still "ok"
    assert {f.name: f.stat().st_ino for f in (sdir / "audio").iterdir()} == inodes
    failed = json.loads((sdir / P.FAILED_REPORT_FILE).read_text())
    assert failed["status"] == "failed" and failed["stage"] == "align"
    assert failed["package"] == "previous version kept" and "align crashed" in failed["error"]
    assert not list(sdir.glob(".prep-*")) and not list(sdir.glob("audio/.sep-*"))
    assert load_song(sdir).meta.duration == pytest.approx(3.0, abs=1e-3)

    monkeypatch.setattr(P, "_align", fake_align)
    P.prepare_song(audio, "Other words", title="t", song_id="t", songs_root=root)
    assert not (sdir / P.FAILED_REPORT_FILE).exists() and not list(sdir.glob(".prep-*"))
    assert [ln.text for ln in load_song(sdir).lines] == ["Other words"]


def test_reprep_with_new_audio_drops_stale_lead(audio: Path, tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    sep = fake_separator(calls)
    monkeypatch.setattr(S, "get_separator", lambda model: sep)
    monkeypatch.setattr(P, "_parse", fake_parse)
    monkeypatch.setattr(P, "_align", fake_align)
    root = tmp_path / "songs"
    sdir = P.prepare_song(audio, LYRICS, title="t", songs_root=root, lead=True)
    assert (sdir / "audio/lead.wav").is_file() and len(calls) == 2
    y, sr = sf.read(audio, dtype="float32")
    other = tmp_path / "in" / "other.wav"
    sf.write(other, 0.5 * y[::-1], sr)
    P.prepare_song(other, LYRICS, title="t", song_id="t", songs_root=root)
    assert len(calls) == 3                                # new mix: vocals separated again, no lead pass
    assert not (sdir / "audio/lead.wav").exists() and "lead" not in load_song(sdir).meta.stems
    assert S.read_manifest(sdir / "audio")["lead_model"] is None


_SIGTERM_CHILD = r"""
import sys, time
sys.path.insert(0, sys.argv[1])
import test_pipeline as T
from lyricprompter.prep import pipeline as P, separate as S


class Blocking:
    model_instance = T._Inst()

    def separate(self, path):
        print("SEPARATING", flush=True)
        time.sleep(60)
        raise AssertionError("SIGTERM did not arrive")


P._parse, P._align = T.fake_parse, T.fake_align
S.get_separator = lambda model: Blocking()
sys.exit(P.cli(sys.argv[2:]))
"""


def test_sigterm_cancels_reprep_and_keeps_package(audio: Path, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """The server cancels a job with SIGTERM: exit 143, report "cancelled", the live package untouched."""
    calls: list[str] = []
    sep = fake_separator(calls)
    monkeypatch.setattr(S, "get_separator", lambda model: sep)
    monkeypatch.setattr(P, "_parse", fake_parse)
    monkeypatch.setattr(P, "_align", fake_align)
    root = tmp_path / "songs"
    sdir = P.prepare_song(audio, LYRICS, title="t", songs_root=root)
    song_json = (sdir / "song.json").read_bytes()
    lyr = tmp_path / "lyrics.txt"
    lyr.write_text("Other words\n", encoding="utf-8")
    child = tmp_path / "child.py"
    child.write_text(_SIGTERM_CHILD, encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(child), str(Path(__file__).parent), "--audio", str(audio), "--lyrics", str(lyr),
         "--title", "t", "--id", "t", "--songs-dir", str(root), "--sep-model", "other.ckpt"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    watchdog = threading.Timer(60, proc.kill)
    watchdog.start()
    try:
        seen = []
        for line in proc.stdout:
            seen.append(line.rstrip())
            if line.startswith("SEPARATING"):
                break
        assert seen and seen[-1] == "SEPARATING", seen
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=30)
        err = proc.stderr.read()
    finally:
        watchdog.cancel()
        proc.kill()
    assert rc == 128 + signal.SIGTERM and "terminated" in err
    failed = json.loads((sdir / P.FAILED_REPORT_FILE).read_text())
    assert failed["status"] == "cancelled" and failed["stage"] == "separate"
    assert (sdir / "song.json").read_bytes() == song_json
    assert json.loads((sdir / P.REPORT_FILE).read_text())["status"] == "ok"
    assert not list(sdir.glob(".prep-*"))


def test_clipped_source_warns(patched: None, tmp_path: Path) -> None:
    t = np.arange(2 * 44100) / 44100
    y = np.clip(1.6 * np.sin(2 * np.pi * 220 * t), -1.0, 1.0).astype(np.float32)
    src = tmp_path / "loud.wav"
    sf.write(src, np.stack([y, y], 1), 44100, subtype="FLOAT")
    sdir = P.prepare_song(src, LYRICS, title="loud", songs_root=tmp_path / "songs")
    rep = json.loads((sdir / P.REPORT_FILE).read_text())
    assert rep["source"]["full_scale_samples"] > 0.3 * y.size
    assert any("digital full scale" in w for w in rep["warnings"])


def test_supplied_vocals_are_lined_up_and_reported(patched: None, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    from test_separate import vocal_like

    monkeypatch.setattr(P, "separate_vocals", lambda *a, **k: pytest.fail("separation must be skipped"))
    sr, n = 44100, 6 * 44100
    v = vocal_like(n, sr)
    acc = (0.08 * np.random.default_rng(5).standard_normal((n, 2))).astype(np.float32)
    src = tmp_path / "song.wav"
    sf.write(src, v + acc, sr, subtype="FLOAT")
    k = int(0.3 * sr)
    voc = tmp_path / "vocals.wav"
    sf.write(voc, 0.7 * np.concatenate([np.zeros((k, 2), np.float32), v]), sr, subtype="FLOAT")
    sdir = P.prepare_song(src, LYRICS, title="x", songs_root=tmp_path / "songs", vocals=voc)
    rep = json.loads((sdir / P.REPORT_FILE).read_text())
    chk = rep["separation"]["vocals_check"]
    assert chk["matched"] and chk["lag"] == k and chk["gain"] == pytest.approx(1 / 0.7, rel=0.01)
    assert any("offset by +0.300s" in w for w in rep["warnings"])
    assert any("rescaled" in w for w in rep["warnings"])
    got, _ = read_audio(sdir / "audio/vocals.wav")
    assert 10 * np.log10(np.sum((got - v) ** 2) / np.sum(v**2)) < -35


def test_tuning_estimate_on_silence(tmp_path: Path) -> None:
    p = save_wav(tmp_path / "s.wav", np.zeros((44100, 2), np.float32), 44100)
    assert P.estimate_tuning(p) == 0.0


# -------------------------------------------------------------------------------------------- CLI

def test_cli_success(patched: None, audio: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    lyr = tmp_path / "lyrics.txt"
    lyr.write_text("﻿" + LYRICS, encoding="utf-8")         # BOM is stripped
    rc = P.cli(["--audio", str(audio), "--lyrics", str(lyr), "--title", "Cli Song", "--lang", "en",
                "--songs-dir", str(tmp_path / "songs"), "--refine", "none"])
    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0
    assert all(PROGRESS_RE.match(x) for x in out[:-1]) and len(out) > 8
    assert [int(PROGRESS_RE.match(x).group(2)) for x in out[:-1]][-1] == 100
    assert Path(out[-1]) == (tmp_path / "songs/cli-song").resolve()
    assert (tmp_path / "songs/cli-song/lyrics.txt").read_text(encoding="utf-8") == LYRICS


def test_cli_via_dispatcher_and_stdin(patched: None, audio: Path, tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(LYRICS))
    rc = top_cli.main(["prep", "--audio", str(audio), "--lyrics", "-", "--title", "T", "--id", "my_id",
                       "--songs-dir", str(tmp_path / "songs")])
    assert rc == 0
    assert capsys.readouterr().out.strip().splitlines()[-1].endswith("my_id")
    assert load_song(tmp_path / "songs/my_id").meta.id == "my_id"


def test_cli_errors(patched: None, audio: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert P.cli(["--audio", str(audio)]) == 2                                  # argparse usage error
    assert "required" in capsys.readouterr().err
    assert P.cli(["--help"]) == 0
    capsys.readouterr()
    lyr = tmp_path / "l.txt"
    lyr.write_text(LYRICS, encoding="utf-8")
    rc = P.cli(["--audio", str(tmp_path / "nope.m4a"), "--lyrics", str(lyr), "--title", "x",
                "--songs-dir", str(tmp_path / "songs")])
    err = capsys.readouterr().err
    assert rc == 1 and "nope.m4a" in err and err.startswith("error:")
    assert P.cli(["--audio", str(audio), "--lyrics", str(tmp_path / "none.txt"), "--title", "x"]) == 1
    assert P.cli(["--audio", str(audio), "--lyrics", str(lyr), "--title", "x", "--refine", "bogus"]) == 2


# ------------------------------------------------------------------------------------------- slow

def _lrc_lines(path: Path) -> list[tuple[float, str, list[float]]]:
    """[(line start s, text, tag times)] from an enhanced LRC; '<t>x' continues a word, '<t> x' starts one."""
    out = []
    for row in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\[(\d+):(\d+(?:\.\d+)?)\](.*)$", row)
        if not m or not m.group(3).strip():
            continue
        start = int(m.group(1)) * 60 + float(m.group(2))
        tags = [int(a) * 60 + float(b) for a, b in re.findall(r"<(\d+):(\d+(?:\.\d+)?)>", m.group(3))]
        body = re.sub(r"\s*<\d+:\d+(?:\.\d+)?>(\s?)", lambda g: " " if g.group(1) else "", m.group(3))
        out.append((start, " ".join(body.split()), [start, *tags]))
    return out


def test_lrc_parser() -> None:
    ref = config.ref_repo_dir()
    if ref is None:
        pytest.skip("reference repo missing")
    rows = _lrc_lines(ref / "resources/lyrics/Edelweiss_short.lrc")
    assert [r[1] for r in rows] == ["Edelweiss, edelweiss", "Every morning you greet me",
                                    "Small and white, clean and bright", "You look happy to meet me"]
    assert rows[0][0] == pytest.approx(6.89) and len(rows[0][2]) == 6 and rows[0][2][1] == pytest.approx(8.25)


@pytest.mark.slow
@pytest.mark.parametrize("refine", ["none", "auto"])
def test_e2e_prep_edelweiss_short(refine: str, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Real `lyricprompter prep` (text + MMS_FA [+ Qwen]) on the reference recording vs its LRC."""
    ref = config.ref_repo_dir()
    stem = config.DATA_DIR / "stems/edelweiss_short/vocals.wav"
    try:
        from lyricprompter.prep import align  # noqa: F401
        from lyricprompter.text import parse_lyrics  # noqa: F401
    except ImportError as e:
        pytest.skip(f"text / align not implemented yet: {e}")
    if ref is None or not stem.is_file():
        pytest.skip("reference audio or pre-separated stem missing")
    rows = _lrc_lines(ref / "resources/lyrics/Edelweiss_short.lrc")
    lyr = tmp_path / "edelweiss.txt"
    lyr.write_text("\n".join(r[1] for r in rows) + "\n", encoding="utf-8")
    rc = P.cli(["--audio", str(ref / "resources/audio/Edelweiss_short.wav"), "--lyrics", str(lyr),
                "--title", "Edelweiss (short)", "--lang", "en", "--songs-dir", str(tmp_path / "songs"),
                "--vocals", str(stem), "--refine", refine])
    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0, out[-5:]
    assert all(PROGRESS_RE.match(x) for x in out[:-1])
    song = load_song(Path(out[-1]))
    assert [ln.text for ln in song.lines] == [r[1] for r in rows]
    line_err = [ln.start - r[0] for ln, r in zip(song.lines, rows, strict=True)]
    syl_err = [s.start - t for ln, r in zip(song.lines, rows, strict=True)
               if len(ln.syllables()) == len(r[2]) for s, t in zip(ln.syllables(), r[2], strict=True)]
    with capsys.disabled():
        print(f"\n[refine={refine}] line  lrc     ours    err   conf")
        for ln, r, e in zip(song.lines, rows, line_err, strict=True):
            print(f"{ln.idx:>18}  {r[0]:6.2f}  {ln.start:6.2f}  {e:+.2f}  {ln.confidence:.3f}")
        a = np.abs(syl_err)
        print(f"syllable onsets n={len(a)} AAE={a.mean():.3f} MedAE={np.median(a):.3f} "
              f"PCO@500={np.mean(a <= 0.5):.2f}; timings {song.meta.prep['timings']}")
    assert max(abs(e) for e in line_err) < 1.5
    assert len(syl_err) >= 20 and float(np.median(np.abs(syl_err))) < 0.5


@pytest.mark.parametrize("bad", ["take..2", "..", ".hidden", "a/b"])
def test_song_id_rules_match_the_server(tmp_path, bad) -> None:
    from lyricprompter.prep.pipeline import PrepError, _claim_song_dir
    with pytest.raises(PrepError):
        _claim_song_dir("t", bad, tmp_path)


def test_song_id_is_nfc(tmp_path) -> None:
    import unicodedata

    from lyricprompter.prep.pipeline import _claim_song_dir
    sid, sdir, created = _claim_song_dir("t", unicodedata.normalize("NFD", "がっこう"), tmp_path)
    assert sid == unicodedata.normalize("NFC", "がっこう") and created and sdir.name == sid
