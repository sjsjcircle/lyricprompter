"""Offline cue firing end to end: a synthetic song (synth.render_song) as a project's reference, re-rendered live
takes with a known tempo map (exact ground truth), the session's tracker and the scheduler on the virtual
clock; the skip policies on a take with a skipped section; the `simulate` and `eval` commands."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lyricprompter_oscsender import project as P
from lyricprompter_oscsender import synth as S
from lyricprompter_oscsender.cli import main as cli_main
from lyricprompter_oscsender.cues import parse_cues
from lyricprompter_oscsender.live import simulate as sim
from lyricprompter_oscsender.live.simulate import (
    cue_errors,
    error_stats,
    fire_offline,
    simulate_cues,
    track_offline,
)
from lyricprompter_oscsender.osc import Target
from lyricprompter_oscsender.settings import Settings

FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
pytestmark = pytest.mark.skipif(not FFMPEG, reason="creating a project needs ffmpeg / ffprobe")

SR = 16000
SONG_S = 40.0
# the band plays 8% slower, then 7% faster, then 3% slower than the reference
TEMPO_MAP = ([0.0, 12.0, 28.0, 40.0], [0.0, 12.96, 27.84, 40.2])
VIDEO_LATENCY_MS = 200.0
FIRE_KEYS = {"cue_id", "cue_time", "target", "address", "message", "at", "ref_time", "reason", "send",
             "ref_late_ms"}


def cue_text(times, *, every_video: int = 0) -> str:
    """One cue per time; every `every_video`-th one goes to the declared "video" target."""
    lines = [f"@target video 127.0.0.1:7000 latency={VIDEO_LATENCY_MS:g}ms"]
    for k, t in enumerate(times):
        target = "video " if every_video and k % every_video == every_video - 1 else ""
        lines.append(f"{t:.3f}  {target}/cue {k + 1}")
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def song(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, S.Score]:
    root = tmp_path_factory.mktemp("sim")
    score = S.make_score(SONG_S, 5)
    wav = root / "reference.wav"
    sf.write(str(wav), S.render_song(score), SR)
    text = cue_text(np.arange(2.0, SONG_S - 0.5, 1.5), every_video=4)       # 25 cues
    pdir, warnings = P.create_project(root / "projects", title="Synthetic song", audio=wav, cues_text=text)
    assert warnings == []
    return pdir, score


@pytest.fixture(scope="module")
def live_take(song) -> np.ndarray:
    return S.render_song(song[1], tempo_map=TEMPO_MAP)


def gt_live(cue_time: float) -> float:
    return float(S.score_to_perf(cue_time, TEMPO_MAP))


# ----------------------------------------------------------------------------- accuracy


@pytest.mark.parametrize("style", ["classic", "pop"])
def test_mix_mode_every_cue_fires_once_near_ground_truth(song, live_take, style: str) -> None:
    pdir, _ = song
    cues = parse_cues(P.read_cues(pdir)).cues
    fires = simulate_cues(pdir, live_take, style=style, settings=Settings())       # every latency 0
    assert all(set(f) == FIRE_KEYS for f in fires)
    assert sorted(f["cue_id"] for f in fires) == [c.id for c in cues]               # exactly once each
    assert all(f["send"] and f["reason"] in ("track", "catchup") for f in fires)
    ats = [f["at"] for f in fires]
    assert ats == sorted(ats)
    res = cue_errors(cues, fires, {c.id: gt_live(c.time) for c in cues})
    st = error_stats(list(res["errors_ms"].values()), len(cues), len(res["missed"]), res["refired"])
    assert st["missed"] == 0 and st["refired"] == 0 and st["fired"] == 25
    assert st["median_ms"] <= 60.0 and st["p90_ms"] <= 150.0, st
    late = {f["cue_id"]: f["ref_late_ms"] for f in fires}
    assert all(0.0 <= v < 1000.0 for v in late.values())
    assert {f["target"] for f in fires} == {"main", "video"}


def test_declared_target_latency_sends_its_cues_earlier(song, live_take) -> None:
    """simulate_cues without settings uses the cue file's declared targets: "video" cues go 200 ms early."""
    pdir, _ = song
    cues = parse_cues(P.read_cues(pdir)).cues
    run = track_offline(pdir, live_take)
    plain = {f["cue_id"]: f["at"] for f in fire_offline(run, cues, Settings())}
    fires = simulate_cues(pdir, live_take)
    early = {f["cue_id"]: f["at"] for f in fires}
    assert sorted(early) == sorted(plain)
    video = [c.id for c in cues if c.target == "video"]
    others = [c.id for c in cues if c.target is None]
    shift = np.array([early[i] - plain[i] for i in video]) * 1000
    assert len(video) == 6 and np.median(shift) == pytest.approx(-VIDEO_LATENCY_MS, abs=40)
    assert all(early[i] == plain[i] for i in others)
    s = Settings(targets=[Target("main"), Target("video", latency_ms=VIDEO_LATENCY_MS)])
    assert fire_offline(run, cues, s) == fires                       # the same tracking run, re-scheduled


def test_start_time_and_input_latency(song, live_take) -> None:
    pdir, _ = song
    cues = parse_cues(P.read_cues(pdir)).cues
    t0 = gt_live(20.0)                                       # the take from the cue at 20 s
    fires = simulate_cues(pdir, live_take[round(t0 * SR):], start_time=20.0, settings=Settings(skip_policy="skip"))
    first = [(f["cue_id"], f["reason"], f["send"]) for f in fires[:13]]
    assert cues[12].time == 20.0                            # the 13th cue is at the start position: fires at once
    assert first == [(c.id, "skipped", False) for c in cues[:12]] + [(cues[12].id, "track", True)]
    assert fires[13]["cue_id"] == cues[13].id and fires[13]["at"] == pytest.approx(gt_live(21.5) - t0, abs=0.15)
    # input latency compensation runs the published position ahead: cues fire that much earlier
    at0 = {f["cue_id"]: f["at"] for f in fire_offline(track_offline(pdir, live_take), cues, Settings())}
    at1 = {f["cue_id"]: f["at"] for f in
           fire_offline(track_offline(pdir, live_take, input_latency_ms=100.0), cues, Settings())}
    assert sorted(at1) == sorted(at0)
    assert np.median([at1[i] - at0[i] for i in at0]) == pytest.approx(-0.1, abs=0.04)


def test_input_latency_at_the_start_fires_the_cues_it_looks_ahead_over(song, live_take) -> None:
    # the tracker publishes the start time plus a 400 ms look-ahead (longer than the 0.3 s catch-up window): the
    # cues from the start time to there fire at once, none is jumped over
    pdir, _ = song
    cues = parse_cues("19.000 /a\n20.000 /b\n20.150 /c\n20.300 /d\n21.000 /e\n").cues
    t0 = gt_live(20.0)
    fires = simulate_cues(pdir, live_take[round(t0 * SR):], start_time=20.0, cues=cues, input_latency_ms=400.0,
                          settings=Settings(skip_policy="skip", catchup_s=0.3))
    assert [(f["cue_id"], f["reason"], f["send"], f["at"]) for f in fires[:4]] == [
        (1, "skipped", False, 0.0), (2, "track", True, 0.0), (3, "track", True, 0.0), (4, "track", True, 0.0)]
    assert [(f["cue_id"], f["send"]) for f in fires[4:]] == [(5, True)]


# ----------------------------------------------------------------------------- skipped section


@pytest.fixture(scope="module")
def skip_song(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, np.ndarray, list]:
    """50 s reference with cues every 2 s alternating /scene and /light; a live take that leaves out
    reference 14..30 s (the band jumps from 14 s to 30 s)."""
    root = tmp_path_factory.mktemp("skip")
    y = S.render_song(S.make_score(50.0, 5))
    wav = root / "reference.wav"
    sf.write(str(wav), y, SR)
    times = np.arange(2.0, 49.0, 2.0)
    text = "".join(f"{t:.3f}  {('/scene', '/light')[k % 2]} {k + 1}\n" for k, t in enumerate(times))
    pdir, _ = P.create_project(root / "projects", title="Skip", audio=wav, cues_text=text)
    live = np.concatenate([y[:14 * SR], y[30 * SR:]])
    return pdir, live, parse_cues(text).cues


def test_skipped_section_goes_through_the_skip_policy(skip_song) -> None:
    pdir, live, cues = skip_song
    a, b = 14.0, 30.0

    def gt(t: float) -> float | None:
        return t if t <= a else (t - (b - a) if t >= b else None)

    run = track_offline(pdir, live)
    by_policy = {p: fire_offline(run, cues, Settings(skip_policy=p)) for p in ("skip", "latest", "all")}
    jump = [f for f in by_policy["skip"] if f["reason"] == "skipped"]
    ids = sorted(f["cue_id"] for f in jump)
    t_jump = jump[0]["at"]
    assert len({f["at"] for f in jump}) == 1 and a < t_jump < a + 8.0             # relocated within 8 s
    # the jump lands where the band is: it passes the cues the band has just played, not the next one
    assert gt(cues[ids[-1] - 1].time) <= t_jump + 0.1 < gt(cues[ids[-1]].time)
    assert len(ids) >= 5
    for policy, fires in by_policy.items():
        assert sorted(f["cue_id"] for f in fires) == [c.id for c in cues], policy    # each cue handled once
        group = [f for f in fires if f["reason"] in ("chase", "skipped")]
        assert sorted(f["cue_id"] for f in group) == ids and {f["at"] for f in group} == {t_jump}
        sent = {f["cue_id"] for f in group if f["send"]}
        if policy == "skip":
            assert sent == set() and all(f["reason"] == "skipped" for f in group)
        elif policy == "all":
            assert sent == set(ids) and all(f["reason"] == "chase" for f in group)
        else:                                       # the latest cue per address: where the show should be now
            latest = {}
            for c in cues:
                if c.id in ids:
                    latest[c.address] = c.id
            assert sent == set(latest.values()) and len(sent) == 2
            assert all((f["reason"] == "chase") == (f["cue_id"] in sent) for f in group)
        # cues before the cut and after the jump fire on time
        for f in fires:
            g = gt(f["cue_time"])
            if f["cue_id"] not in ids and g is not None and (f["cue_time"] < a or f["at"] > t_jump):
                assert abs(f["at"] - g) < 0.15, (policy, f)


# ----------------------------------------------------------------------------- commands


@pytest.fixture(scope="module")
def live_wav(tmp_path_factory: pytest.TempPathFactory, live_take) -> Path:
    path = tmp_path_factory.mktemp("live") / "take.wav"
    sf.write(str(path), live_take, SR)
    return path


def test_simulate_cli_offline(song, live_wav, capsys: pytest.CaptureFixture) -> None:
    pdir, _ = song
    assert cli_main(["simulate", "--project", pdir.name, "--projects-dir", str(pdir.parent),
                     "--file", str(live_wav), "--style", "classic"]) == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("live ")]
    assert len(lines) == 25 and "/cue 1" in lines[0] and "video" in lines[3]
    assert "done: 25 fires (25 cues sent, 0 skipped, 0 never reached)" in out
    assert out.splitlines()[0].startswith("project 'synthetic-song': 25 cues, reference stem 'mix', classic style")


def test_simulate_cli_realtime_dry_run(song, live_take, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    pdir, _ = song
    short = tmp_path / "short.wav"
    sf.write(str(short), live_take[:round(6.0 * SR)], SR)
    assert sim.cli_simulate(["--project-dir", str(pdir), "--file", str(short), "--realtime", "--speed", "6"]) == 0
    out = capsys.readouterr().out
    fired = [ln for ln in out.splitlines() if ln.startswith("live ")]
    assert len(fired) >= 2 and "/cue 1  (dry run)" in fired[0] and "/cue 2  (dry run)" in fired[1]
    assert "dry run" in out.splitlines()[0] and "done: 0 sent" in out


def test_simulate_cli_errors(song, live_wav, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    pdir, _ = song
    base = ["--project-dir", str(pdir), "--file", str(live_wav)]
    for argv, msg in [(base + ["--send"], "--send needs --realtime"),
                      (base + ["--mode", "vocal"], "has no vocal stem"),
                      (["--project-dir", str(tmp_path), "--file", str(live_wav)], "not a project"),
                      (["--project-dir", str(pdir), "--file", str(tmp_path / "nope.wav")], "file not found"),
                      (["--project", "../x", "--file", str(live_wav)], "invalid project id")]:
        with pytest.raises(SystemExit) as ei:
            sim.cli_simulate(argv)
        assert ei.value.code == 2
        assert msg in capsys.readouterr().err


def test_eval_cli(song, capsys: pytest.CaptureFixture) -> None:
    pdir, _ = song
    assert sim.cli_eval(["--project-dir", str(pdir), "--seeds", "1", "--json", "--tempo", "0.9,1.1"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["project"] == "synthetic-song" and res["mode"] == "mix" and res["style"] == "classic"
    assert len(res["takes"]) == 1 and res["takes"][0]["seed"] == 0 and res["excluded"] == 0
    o = res["overall"]
    assert o["cues"] == 25 and o["fired"] + o["missed"] == 25 and o["fired"] >= 23
    assert {"median_ms", "p90_ms", "max_ms", "bias_ms", "within_50", "within_100", "within_200"} <= set(o)
    assert o["median_ms"] < 150.0 and 0 <= o["within_200"] <= 100

    assert cli_main(["eval", "--project-dir", str(pdir), "--seeds", "1", "--style", "pop"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[1].split() == ["take", "cues", "fired", "missed", "median", "p90", "max", "bias", "<=50", "<=100",
                              "<=200"]
    assert out[2].startswith("seed 0") and out[3].startswith("overall") and "style pop" in out[-1]
    with pytest.raises(SystemExit):
        sim.cli_eval(["--project-dir", str(pdir), "--mode", "vocal"])
    with pytest.raises(SystemExit):
        sim.cli_eval(["--project-dir", str(pdir), "--tempo", "1.2,0.9"])


def test_error_stats() -> None:
    st = error_stats([-30.0, 10.0, 60.0, 250.0], n_cues=5, missed=1, refired=2)
    assert st["cues"] == 5 and st["fired"] == 4 and st["missed"] == 1 and st["refired"] == 2
    assert st["median_ms"] == 45.0 and st["max_ms"] == 250.0 and st["bias_ms"] == 35.0
    assert st["within_50"] == 50.0 and st["within_100"] == 75.0 and st["within_200"] == 75.0
    empty = error_stats([], 3, 3)
    assert empty["median_ms"] is None and empty["within_50"] is None and empty["fired"] == 0
    json.dumps(empty, allow_nan=False)
