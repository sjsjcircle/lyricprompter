"""Tracking style (classic / pop) plumbing: SongMeta.style, prep --style, LiveSession, server endpoints, console
form fields, CLI flags and the eval-track pop scenarios. The tracker behaviour itself: tests/test_tracker_pop.py."""
from __future__ import annotations

import json
import sys
import textwrap
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from lyricprompter.eval import track_eval as E
from lyricprompter.live import session as live_session
from lyricprompter.live.session import LiveSession, package_id, with_style
from lyricprompter.prep import pipeline as P
from lyricprompter.server import app as app_mod
from lyricprompter.server.app import create_app
from lyricprompter.server.jobs import JobManager, build_prep_argv
from lyricprompter.songfile import load_song, save_song
from lyricprompter.text.model import STYLES, AlignUnit, Line, Song, SongMeta, Syllable, Word
from lyricprompter.tracking.tracker import TrackerParams

STATIC = Path(app_mod.__file__).parent / "static"
TEST_HOSTS = ("testserver",)
SR = 16000


def make_line(idx: int, text: str, start: float, dur: float = 0.8) -> Line:
    syls, step = [], dur / len(text)
    for k, ch in enumerate(text):
        t0 = round(start + k * step, 4)
        syls.append(Syllable(text=ch, span=(k, k + 1), units=[AlignUnit(ch, "a", t0, t0 + step, 0.9)],
                             start=t0, end=round(t0 + step, 4)))
    word = Word(text=text, span=(0, len(text)), lang="ko", syllables=syls)
    return Line(idx=idx, text=text, words=[word], start=start, end=round(start + dur, 4), confidence=0.8)


@pytest.fixture
def song_root(tmp_path: Path) -> Path:
    root = tmp_path / "songs"
    for sid, style in (("demo", "classic"), ("band", "pop")):
        sdir = root / sid
        save_song(Song(meta=SongMeta(id=sid, title=sid.title(), duration=4.0, style=style),
                       lines=[make_line(0, "꽃밭에", 0.5), make_line(1, "앉아서", 1.5), make_line(2, "보니", 2.5)]),
                  sdir)
        (sdir / "audio").mkdir()
        t = np.arange(SR * 4) / SR
        y = (0.2 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)).astype(np.float32)
        for stem in ("mix", "vocals"):
            sf.write(sdir / "audio" / f"{stem}.wav", y, SR)
    return root


# ----------------------------------------------------------------------------- song model / prep


def test_song_meta_style_default_and_old_song_json(tmp_path: Path) -> None:
    assert STYLES == ("classic", "pop") and SongMeta(id="x").style == "classic"
    sdir = tmp_path / "old"
    sdir.mkdir()
    (sdir / "song.json").write_text(json.dumps({"schema": 1, "meta": {"id": "old", "title": "Old"}, "lines": []}))
    assert load_song(sdir).meta.style == "classic"                        # written before the styles existed
    song = load_song(sdir)
    song.meta.style = "pop"
    save_song(song, sdir)
    assert json.loads((sdir / "song.json").read_text())["meta"]["style"] == "pop"
    assert load_song(sdir).meta.style == "pop"


def test_prep_style_option(tmp_path: Path) -> None:
    args = P.build_parser().parse_args(["--audio", "a.wav", "--lyrics", "l.txt", "--title", "t", "--style", "pop"])
    assert args.style == "pop"
    assert P.build_parser().parse_args(["--audio", "a.wav", "--lyrics", "l.txt", "--title", "t"]).style is None
    with pytest.raises(SystemExit):
        P.build_parser().parse_args(["--audio", "a.wav", "--lyrics", "l.txt", "--title", "t", "--style", "jazz"])
    audio = tmp_path / "a.wav"
    sf.write(audio, np.zeros(SR, np.float32), SR)
    with pytest.raises(P.PrepError, match="style"):
        P.prepare_song(audio, "la la", title="t", songs_root=tmp_path / "songs", style="jazz")
    assert not (tmp_path / "songs" / "t").exists()                        # refused before anything was made
    # the style a re-prep keeps: the package's own (classic when unset / unreadable)
    sdir = tmp_path / "pkg"
    save_song(Song(meta=SongMeta(id="pkg", style="pop")), sdir)
    assert P._existing_style(sdir) == "pop"
    (sdir / "song.json").write_text('{"schema": 1, "meta": {"id": "pkg"}}')
    assert P._existing_style(sdir) == "classic"
    (sdir / "song.json").write_text("{broken")
    assert P._existing_style(sdir) == "classic"


def _fake_prep(monkeypatch: pytest.MonkeyPatch) -> None:
    from lyricprompter.audio_io import read_audio, save_wav
    from lyricprompter.prep import separate as S

    def parse(raw: str, default_lang: str = "auto") -> list[Line]:
        return [make_line(i, row.strip(), 0.5 + i) for i, row in enumerate(raw.splitlines()) if row.strip()]

    class Report:
        method, line_confidence, flagged, timings, details = "fake", [], [], {}, {}

    def separate(mix_wav: Path, out_dir: Path, *, model: str = "m", lead: bool = False, progress=None) -> dict:
        y, sr = read_audio(mix_wav)
        out = {"vocals": save_wav(out_dir / "vocals.wav", 0.5 * y, sr),
               "instrumental": save_wav(out_dir / "instrumental.wav", 0.5 * y, sr)}
        S.write_manifest(out_dir, model=model, mix_wav=mix_wav, stems=out, lead_model=None)
        return out

    monkeypatch.setattr(P, "_parse", parse)
    monkeypatch.setattr(P, "_align", lambda lines, *a, **k: Report())
    monkeypatch.setattr(P, "separate_vocals", separate)


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="ffmpeg not installed")
def test_prepare_song_sets_and_reprep_keeps_the_style(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_prep(monkeypatch)
    audio = tmp_path / "in.wav"
    sf.write(audio, (0.2 * np.sin(np.arange(2 * SR) * 0.07)).astype(np.float32), SR)
    root = tmp_path / "songs"
    sdir = P.prepare_song(audio, "가나다", title="t", songs_root=root)
    assert load_song(sdir).meta.style == "classic"                        # a new song
    P.prepare_song(audio, "가나다", title="t", song_id="t", songs_root=root, style="pop")
    assert load_song(sdir).meta.style == "pop"
    P.prepare_song(audio, "라마바", title="t", song_id="t", songs_root=root)    # re-prep without --style
    assert load_song(sdir).meta.style == "pop" and load_song(sdir).lines[0].text == "라마바"
    P.prepare_song(audio, "라마바", title="t", song_id="t", songs_root=root, style="classic")
    assert load_song(sdir).meta.style == "classic"


def test_prep_argv_passes_the_style() -> None:
    argv = build_prep_argv(Path("/a.wav"), Path("/l.txt"), title="t", song_id="t", songs_dir=Path("/s"), style="pop")
    assert "--style=pop" in argv and P.build_parser().parse_args(argv[1:]).style == "pop"
    assert not any(a.startswith("--style") for a in build_prep_argv(Path("/a"), Path("/l"), title="t", song_id="t",
                                                                     songs_dir=Path("/s")))


# ----------------------------------------------------------------------------- live session


class _Src:
    finished = False

    def start(self, cb) -> None:
        pass

    def stop(self) -> None:
        pass


def test_live_session_style_from_the_song_and_live_switch(song_root: Path) -> None:
    sess = LiveSession(song_root / "band", "vocal", _Src())
    assert sess.style == "pop" and sess.params.style == "pop" and sess.tracker.params.style == "pop"
    assert sess.snapshot()["style"] == "pop"
    sess = LiveSession(song_root / "demo", "mix", _Src())
    assert sess.style == "classic" and sess.tracker.params.style == "classic"
    sess = LiveSession(song_root / "demo", "mix", _Src(), style="pop",          # explicit: wins over the song
                       params=replace(TrackerParams.for_mode("mix"), latency_s=0.2, reloc_every_s=0.7))
    assert sess.tracker.params.style == "pop" and sess.tracker.params.reloc_every_s == 0.7
    sess.start()
    sess.set_style("classic")
    assert sess.style == "classic" and sess.tracker.params.style == "classic"
    assert sess.tracker.params.latency_s == 0.2 and sess.snapshot()["style"] == "classic"
    with pytest.raises(ValueError, match="style"):
        sess.set_style("jazz")
    with pytest.raises(ValueError, match="style"):
        LiveSession(song_root / "demo", "mix", _Src(), style="jazz")
    sess.stop()
    assert with_style(TrackerParams.for_mode("vocal"), "vocal", "pop") == TrackerParams.for_mode("vocal", "pop")
    assert with_style("not params", "mix", "pop") == "not params"


def test_simulate_cli_style(song_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    live = tmp_path / "live.wav"
    sf.write(live, (0.2 * np.sin(np.arange(2 * SR) * 0.07)).astype(np.float32), SR)
    assert live_session.cli(["--song-dir", str(song_root / "demo"), "--file", str(live), "--style", "pop"]) == 0
    assert "pop style" in capsys.readouterr().out
    assert live_session.cli(["--song-dir", str(song_root / "band"), "--file", str(live)]) == 0
    assert "pop style" in capsys.readouterr().out                          # default: the song's style


# ----------------------------------------------------------------------------- server


class StyleSession:
    def __init__(self, sdir: Path, req: app_mod.SessionStart):
        self.song_id, self.req, self.style, self.calls = package_id(sdir), req, req.style, []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def set_style(self, style: str) -> None:
        self.calls.append(style)
        self.style = style

    def reload_lines(self, song: Song) -> None:
        pass

    def snapshot(self) -> dict:
        return {"type": "state", "song_id": self.song_id, "mode": self.req.mode, "style": self.style,
                "status": "tracking", "ref_time": 0.0, "tempo": 1.0, "confidence": 0.8, "live_time": 0.0,
                "line_idx": 0, "syl_idx": 0, "t": time.time(), "age": 0.0}


FAKE_PREP = textwrap.dedent('''
    import json, pathlib, sys
    args = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    sdir = pathlib.Path(args["--songs-dir"]) / args["--id"]
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "argv.json").write_text(json.dumps(sys.argv[1:]))
    (sdir / "song.json").write_text(json.dumps({"schema": 1, "meta": {"id": args["--id"], "title": "x",
                                                "style": args.get("--style", "classic")}, "lines": []}))
    print(sdir, flush=True)
''')


@pytest.fixture
def server(song_root: Path, tmp_path: Path):
    sessions: list[StyleSession] = []

    def factory(sdir: Path, req: app_mod.SessionStart, root: Path) -> StyleSession:
        sessions.append(StyleSession(sdir, req))
        return sessions[-1]

    script = tmp_path / "fake_prep.py"
    script.write_text(FAKE_PREP)
    app = create_app(song_root, session_factory=factory, jobs=JobManager(command_prefix=[sys.executable, str(script)]),
                     devices=lambda refresh: [], allowed_hosts=TEST_HOSTS)
    with TestClient(app) as client:
        client.sessions, client.hub = sessions, app.state.hub
        yield client


def test_song_list_and_detail_include_the_style(server: TestClient) -> None:
    styles = {s["id"]: s["style"] for s in server.get("/api/songs").json()}
    assert styles == {"band": "pop", "demo": "classic"}
    d = server.get("/api/songs/band").json()
    assert d["style"] == "pop" and d["meta"]["style"] == "pop"


def test_put_song_style_saves_bumps_the_rev_and_switches_the_live_session(server: TestClient, song_root: Path) -> None:
    rev0 = server.get("/api/songs/demo").json()["rev"]
    r = server.put("/api/songs/demo/style", json={"style": "pop"})
    assert r.status_code == 200, r.text
    assert r.json() == {"song_id": "demo", "style": "pop", "rev": rev0 + 1, "session": False}
    assert load_song(song_root / "demo").meta.style == "pop"
    assert load_song(song_root / "demo").lines[1].text == "앉아서"           # lines untouched
    r = server.put("/api/songs/demo/style", json={"style": "pop"})          # unchanged: nothing saved
    assert r.json()["rev"] == rev0 + 1
    assert server.put("/api/songs/demo/style", json={"style": "jazz"}).status_code == 422
    assert server.put("/api/songs/nosuch/style", json={"style": "pop"}).status_code == 404
    # a live session of this song is switched (and only of this song)
    assert server.post("/api/session/start", json={"song_id": "demo", "input": {"kind": "file", "stem": "mix"}}
                       ).status_code == 200
    sess = server.sessions[-1]
    assert sess.req.style is None                                        # None: the song's style
    r = server.put("/api/songs/demo/style", json={"style": "classic"})
    assert r.json()["session"] is True and sess.calls == ["classic"]
    r = server.put("/api/songs/band/style", json={"style": "classic"})
    assert r.json()["session"] is False and sess.calls == ["classic"]


def test_session_style_endpoint_and_start_option(server: TestClient, song_root: Path) -> None:
    assert server.post("/api/session/style", json={"style": "pop"}).status_code == 409   # no session
    r = server.post("/api/session/start", json={"song_id": "band", "style": "classic",
                                                "input": {"kind": "file", "stem": "mix"}})
    assert r.status_code == 200, r.text
    sess = server.sessions[-1]
    assert sess.req.style == "classic"
    r = server.post("/api/session/style", json={"style": "pop"})
    assert r.status_code == 200 and sess.calls == ["pop"] and r.json()["style"] == "pop"
    assert load_song(song_root / "band").meta.style == "pop"              # the song is not touched
    assert server.post("/api/session/style", json={"style": "rock"}).status_code == 422
    assert server.post("/api/session/start", json={"song_id": "band", "style": "rock",
                                                   "input": {"kind": "file", "stem": "mix"}}).status_code == 422


def test_upload_passes_the_style_to_prep(server: TestClient, song_root: Path) -> None:
    def up(options: dict, title: str) -> dict:
        return server.post("/api/songs", data={"title": title, "lyrics": "가나다", "language": "ko",
                                               "options": json.dumps(options)},
                           files={"audio": ("a.wav", b"\x00" * 2048, "audio/wav")})

    r = up({"style": "pop"}, "Rock Song")
    assert r.status_code == 202, r.text
    sid = r.json()["song_id"]
    deadline = time.monotonic() + 15
    while not (song_root / sid / "argv.json").is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert "--style=pop" in json.loads((song_root / sid / "argv.json").read_text())
    assert up({"style": "polka"}, "Bad").status_code == 422


def test_default_session_factory_passes_the_style(song_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built: dict = {}

    class FakeLive:
        def __init__(self, sdir, mode, source, **kw):
            built.update(kw)

    monkeypatch.setattr(live_session, "LiveSession", FakeLive)
    for style in (None, "pop"):
        req = app_mod.SessionStart(song_id="demo", style=style, input=app_mod.InputSpec(kind="file", stem="mix"))
        app_mod.default_session_factory(song_root / "demo", req, song_root)
        assert built["style"] == style


def test_console_has_the_style_controls() -> None:
    html = (STATIC / "console.html").read_text(encoding="utf-8")
    js = (STATIC / "console.js").read_text(encoding="utf-8")
    for label in ("Classical — flexible tempo", "Pop &amp; band — steady tempo"):
        assert html.count(label) == 2                                     # add-song form + Session fieldset
    assert html.count('name="style"') == 2
    assert "/style`, { method: \"PUT\"" in js and "style: el.style.value" in js


# ----------------------------------------------------------------------------- eval-track


def test_pop_time_map_is_steady():
    spec = E.PopSpec(seed=3)
    lp, rp = E.pop_time_map(200.0, spec, np.random.default_rng(3))
    assert lp[1] == pytest.approx(spec.degrade.lead_s) and rp[1] == 0.0
    rho = np.diff(rp[1:]) / np.diff(lp[1:])
    assert spec.tempo[0] <= rho.min() and rho.max() <= spec.tempo[1]
    assert rho.max() / rho.min() - 1 <= spec.drift + 1e-9                  # slow drift only


def test_perform_vocal_drops_shifts_and_adlibs():
    sr = 8000
    spans = [(1.0 + 3.0 * k, 3.0 + 3.0 * k) for k in range(12)]           # 2 s lines, 1 s gaps
    rng = np.random.default_rng(0)
    v = np.zeros(int(40 * sr), np.float32)
    for a, b in spans:
        v[int(a * sr):int(b * sr)] = rng.standard_normal(int((b - a) * sr)) * 0.1
    spec = E.PopSpec(seed=1, drops=2, shift_p=0.6, adlibs=2)
    out, crowd, info = E.perform_vocal(v, sr, spans, spec, np.random.default_rng(1))
    assert len(out) == len(v) and info["dropped"] and len(info["adlib"]) == 2
    for k in info["dropped"]:
        a, b = spans[k]
        assert np.abs(out[int((a - 0.4) * sr):int((b + 0.4) * sr)]).max() == 0.0     # silent: the crowd sings
        assert np.abs(crowd[int(a * sr):int(b * sr)]).max() > 0
    sh = np.array(info["shift"])
    moved = sh[sh != 0]
    assert len(moved) and np.all((np.abs(moved) >= 0.2 - 1e-9) & (np.abs(moved) <= 0.4 + 1e-9))
    starts = np.array([a for a, _ in spans]) + sh
    ends = np.array([b for _, b in spans]) + sh
    assert np.all(starts[1:] >= ends[:-1] + 0.05 - 1e-9)                   # never into a neighbouring line
    clean, _, _ = E.perform_vocal(v, sr, spans, E.PopSpec(seed=1), np.random.default_rng(1))
    assert np.allclose(clean[int(1.1 * sr):int(2.9 * sr)], v[int(1.1 * sr):int(2.9 * sr)])


def test_make_pop_live_clean_and_perturbed_share_the_band(monkeypatch):
    sr = 8000
    rng = np.random.default_rng(2)
    n = 30 * sr
    inst = (0.1 * rng.standard_normal(n)).astype(np.float32)
    voc = np.zeros(n, np.float32)
    spans = [(2.0 + 4.0 * k, 4.5 + 4.0 * k) for k in range(6)]
    for a, b in spans:
        voc[int(a * sr):int(b * sr)] = (0.2 * rng.standard_normal(int((b - a) * sr))).astype(np.float32)
    clean = E.make_pop_live(voc, inst, sr, E.PopSpec(seed=5), spans, "mix")
    pert = E.make_pop_live(voc, inst, sr, E.PopSpec(seed=5, drops=1, shift_p=0.5, adlibs=1), spans, "vocal")
    np.testing.assert_array_equal(clean[1], pert[1])                       # same steady time map
    np.testing.assert_array_equal(clean[2], pert[2])
    assert "dropped" in pert[3] and "dropped" not in clean[3]
    assert 0.92 <= clean[3]["tempo"] <= 1.08
    assert len(clean[0]) == pytest.approx(clean[1][-1] * sr, abs=2)
    gl, gr = E.map_events(np.array([a for a, _ in spans]), clean[1], clean[2])
    np.testing.assert_allclose(np.interp(gl, clean[1], clean[2]), gr, atol=1e-6)


def test_pop_pairs_and_eval_cli_styles(monkeypatch, tmp_path, capsys):
    for k, p in E.pop_pairs().items():                                    # whatever is available locally
        assert p.kind == "pop" and p.pop is not None and k.split("_", 1)[0] in E.POP_SCENARIOS
        assert p.pop.perturbed == k.startswith("pop_") and set(p.stems) == {"vocals", "instrumental"}
    wav = tmp_path / "ref.wav"
    wav.write_bytes(b"")
    pair = E.Pair("fake", "synth", {"mix": wav}, {}, np.ones(1), modes=("mix",))
    seen = []

    def fake_evaluate(p, mode, fcfg=None, tparams=None, *, transpose=None, keep_trace=False):
        seen.append(tparams.style)
        row = {k: 0.0 for k in ("pos_AAE_ms", "pos_MedAE_ms", "lat_median_ms", "lost_frac")}
        row.update({f"PCO@{t}": 90.0 if tparams.style == "pop" else 80.0 for t in E.TOLERANCES_MS})
        row.update(pair=p.name, mode=mode, style=tparams.style, final_key=0, key_switches=0)
        return row

    monkeypatch.setattr(E, "available_pairs", lambda: {"fake": pair})
    monkeypatch.setattr(E, "pair_tuning", lambda p: 0.0)
    monkeypatch.setattr(E, "evaluate_pair", fake_evaluate)
    out_json = tmp_path / "r.json"
    assert E.cli(["--pairs", "fake", "--modes", "mix", "--styles", "classic,pop", "--json", str(out_json)]) == 0
    assert seen == ["classic", "pop"]
    summ = json.loads(out_json.read_text())["summary"]
    assert summ["classic/mix"]["PCO@300"] == 80.0 and summ["pop/mix"]["PCO@300"] == 90.0
    assert E.cli(["--pairs", "fake", "--modes", "mix"]) == 0 and seen[-1] == "classic"
    assert "mix" in E.summarize([{**fake_evaluate(pair, "mix", tparams=TrackerParams())}])
    assert E.cli(["--pairs", "fake", "--styles", "funk"]) == 2
