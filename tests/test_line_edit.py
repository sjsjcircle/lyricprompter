"""Timing editor (feedback 2026-10-13, issue 1): line edits push / squeeze neighbours instead of
refusing, `end` edits, timing for untimed lines, undo, and the console's Lines panel."""
from __future__ import annotations

import copy
import itertools
import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lyricprompter.server import app as app_mod
from lyricprompter.server.app import MIN_LINE_DUR, create_app, fmt_clock, line_list, move_line
from lyricprompter.songfile import load_song, save_song
from lyricprompter.text.model import AlignUnit, Line, Song, SongMeta, Syllable, Word

STATIC = Path(app_mod.__file__).parent / "static"
Span = tuple[float | None, float | None]

# ----------------------------------------------------------------------------- helpers


def mk(idx: int, spans: list[Span], *, units: int = 1, flags: tuple[str, ...] = (), review: bool = False) -> Line:
    """A line with one syllable per (start, end) span, each split into `units` align units."""
    syls = []
    for k, (a, b) in enumerate(spans):
        us = []
        for u in range(units):
            if a is None or b is None:
                us.append(AlignUnit("x", "a"))
            else:
                us.append(AlignUnit("x", "a", round(a + (b - a) * u / units, 4),
                                    round(a + (b - a) * (u + 1) / units, 4), 0.5))
        syls.append(Syllable(text=chr(0xAC00 + k), span=(k, k + 1), units=us, start=a, end=b))
    text = "".join(s.text for s in syls)
    starts = [a for a, _ in spans if a is not None]
    ends = [b for _, b in spans if b is not None]
    return Line(idx=idx, text=text, words=[Word(text=text, span=(0, len(text)), lang="ko", syllables=syls)],
                start=starts[0] if starts else None, end=ends[-1] if ends else None, confidence=0.8,
                needs_review=review, flags=list(flags))


def even(idx: int, start: float | None, dur: float = 0.8, n: int = 3, **kw) -> Line:
    if start is None:
        return mk(idx, [(None, None)] * n, **kw)
    return mk(idx, [(round(start + dur * k / n, 4), round(start + dur * (k + 1) / n, 4)) for k in range(n)], **kw)


def song_of(*lines: Line, duration: float | None = 10.0) -> Song:
    return Song(meta=SongMeta(id="t", title="T", duration=duration), lines=list(lines))


def assert_consistent(song: Song) -> None:
    """Lines in order, none ending after the next timed one starts; inside each line every time
    lies within [start, end], each start <= its end and syllable / unit starts never go back."""
    eps = 1e-9
    timed = [ln for ln in song.lines if ln.start is not None]
    for a, b in itertools.pairwise(timed):
        assert a.start < b.start and a.end <= b.start + eps, (a.idx, a.start, a.end, b.idx, b.start)
    for ln in timed:
        assert ln.start <= ln.end
        prev_s = prev_u = ln.start
        for s in ln.syllables():
            assert ln.start - eps <= s.start <= s.end <= ln.end + eps and s.start >= prev_s - eps, (ln.idx, s)
            prev_s = s.start
            for u in s.units:
                assert ln.start - eps <= u.start <= u.end <= ln.end + eps and u.start >= prev_u - eps, (ln.idx, u)
                prev_u = u.start


def onsets(line: Line) -> list[float]:
    return [s.start for s in line.syllables()]


def rel(line: Line) -> list[float]:
    """Syllable boundaries as fractions of the line's span (its rhythm)."""
    a, b = line.start, line.end
    return [round((t - a) / (b - a), 3) for s in line.syllables() for t in (s.start, s.end)]


def songs07() -> Song:
    """The lock-up reported on songs/07 (lines 29-34, real times, fewer syllables): line 30 could not
    move later ("its last syllable would run into line 31"), line 31 not earlier ("line 30 is still
    being sung until 142.78 s"); line 31 had been squeezed by hand into 0.78 s."""
    return song_of(
        mk(0, [(129.22, 129.6), (129.6, 130.36), (130.36, 131.26), (131.26, 132.0), (132.0, 132.4)],
           flags=("low_confidence",), review=True),
        mk(1, [(132.98, 133.48), (133.76, 134.1), (134.46, 135.6), (135.6, 136.7), (136.7, 137.5), (137.5, 137.78)],
           units=2, flags=("mms_qwen_agree",)),
        mk(2, [(137.78, 138.22), (138.22, 138.74), (138.74, 139.58), (139.76, 140.84), (141.8, 142.32),
               (142.48, 142.78), (142.78, 142.83)], flags=("mms_qwen_agree",)),
        mk(3, [(142.83, 142.99), (142.99, 143.23), (143.23, 143.43), (143.43, 143.61)],
           flags=("low_confidence", "manual_timing")),
        mk(4, [(146.24, 146.48), (146.48, 147.0), (147.0, 147.4)], flags=("low_confidence",), review=True),
        mk(5, [(162.72, 164.0), (164.0, 165.68)]),
        duration=238.48)


# ----------------------------------------------------------------------------- move_line: songs/07


def test_songs07_line_moves_later_and_squeezes_into_its_room() -> None:
    song = songs07()
    before = copy.deepcopy(song)
    move = move_line(song, 2, start=139.0)           # was: "cannot start after 137.78 s"
    ln = song.lines[2]
    assert (move.shift, move.touched, move.pushed) == (1.22, [2], [])
    assert (ln.start, ln.end) == (139.0, 142.83)     # moved rigidly, then squeezed before line 31
    assert rel(ln) == pytest.approx(rel(before.lines[2]), abs=2e-3)       # keeps its rhythm
    assert "manual_timing" in ln.flags and not ln.needs_review
    for i in (0, 1, 3, 4, 5):
        assert song.lines[i] == before.lines[i], i
    assert move.before == {2: app_mod.asdict(before.lines[2])}
    assert_consistent(song)


def test_songs07_line_moves_earlier_and_squeezes_the_previous_line() -> None:
    song = songs07()
    before = copy.deepcopy(song)
    move = move_line(song, 3, start=141.0)           # was: "cannot start before 142.83 s"
    assert (move.shift, move.touched, move.pushed) == (-1.83, [2, 3], [])
    l2, l3 = song.lines[2], song.lines[3]
    assert (l3.start, l3.end) == (141.0, 141.78)     # rigid: keeps its 0.78 s
    assert (l2.start, l2.end) == (137.78, 141.0)     # its onsets ran past 140.95: scaled, not clipped
    assert rel(l2) == pytest.approx(rel(before.lines[2]), abs=2e-3)
    assert l2.flags == ["mms_qwen_agree"] and not l2.needs_review          # only squeezed: flags kept
    assert_consistent(song)


def test_songs07_moving_far_earlier_pushes_the_previous_line() -> None:
    song = songs07()
    move = move_line(song, 3, start=137.9)
    assert move.pushed == [2] and move.touched == [1, 2, 3]
    l1, l2 = song.lines[1], song.lines[2]
    assert l2.start == pytest.approx(137.9 - MIN_LINE_DUR) and l2.end == 137.9
    assert l2.needs_review and l2.flags == ["mms_qwen_agree", "pushed"]
    assert l1.start == 132.98 and l1.end == pytest.approx(137.65) and l1.flags == ["mms_qwen_agree"]
    assert not l1.needs_review
    assert_consistent(song)


def test_songs07_moving_far_later_pushes_the_next_lines() -> None:
    song = songs07()
    before = copy.deepcopy(song)
    move = move_line(song, 2, start=146.2)
    assert move.pushed == [3, 4] and move.touched == [2, 3, 4]
    l3, l4, l5 = song.lines[3], song.lines[4], song.lines[5]
    assert l3.start == pytest.approx(146.45) and l4.start == pytest.approx(146.7)
    assert l3.flags == ["low_confidence", "manual_timing", "pushed"] and l3.needs_review   # placeholder now
    # the last pushed line moved rigidly (room enough before line 34)
    assert [s.end - s.start for s in l4.syllables()] == pytest.approx(
        [s.end - s.start for s in before.lines[4].syllables()])
    assert l5 == before.lines[5]
    assert_consistent(song)


# ----------------------------------------------------------------------------- move_line: pushes


def test_push_later_cascades_only_as_far_as_needed() -> None:
    song = song_of(even(0, 1.0), even(1, 2.0), even(2, 2.3, 0.2), even(3, 2.6, 0.2), even(4, 5.0))
    move = move_line(song, 0, start=2.1)
    assert move.pushed == [1, 2, 3]
    assert [ln.start for ln in song.lines] == pytest.approx([2.1, 2.35, 2.6, 2.85, 5.0])
    assert all("pushed" in song.lines[i].flags and song.lines[i].needs_review for i in (1, 2, 3))
    assert song.lines[4].flags == [] and not song.lines[4].needs_review
    assert song.lines[3].end == pytest.approx(3.05)          # the last pushed line keeps its span
    assert_consistent(song)


def test_push_earlier_and_never_below_zero() -> None:
    song = song_of(even(0, 0.3, 0.2), even(1, 0.6, 0.2), even(2, 2.0))
    move = move_line(song, 2, start=0.7)
    assert move.pushed == [0, 1]
    assert [ln.start for ln in song.lines] == pytest.approx([0.2, 0.45, 0.7])
    assert_consistent(song)
    before = song.to_dict()
    with pytest.raises(ValueError, match=r"Line 3 cannot start at 0:00.40: lines 1–2 before it would have to start before"):
        move_line(song, 2, start=0.4)
    assert song.to_dict() == before                         # refused: nothing changed
    with pytest.raises(ValueError, match="before the beginning"):
        move_line(song, 0, shift=-0.3)


def test_push_past_the_song_end_is_refused() -> None:
    song = song_of(even(0, 1.0), even(1, 2.0), even(2, 3.0), duration=10.0)
    before = song.to_dict()
    with pytest.raises(ValueError, match=r"line 3 after it would be pushed past the end of the song \(0:10.00"):
        move_line(song, 0, start=9.6)
    with pytest.raises(ValueError, match="the song ends at 0:10.00"):
        move_line(song, 2, start=10.0)
    assert song.to_dict() == before
    move_line(song, 2, start=9.9)                            # the last line: fitted before the end
    assert song.lines[2].end == 10.0
    assert_consistent(song)
    open_end = song_of(even(0, 1.0), even(1, 2.0), duration=None)
    assert move_line(open_end, 0, start=500.0).pushed == [1]  # no duration: no upper bound


# ----------------------------------------------------------------------------- move_line: fit / end / untimed


def test_fit_clips_the_tails_or_scales_the_line() -> None:
    def setup() -> Song:
        return song_of(mk(0, [(1.0, 1.2), (1.2, 1.4), (1.4, 2.5)]), even(1, 3.0))
    song = setup()
    move = move_line(song, 1, start=2.0)                     # every onset of line 0 is before 1.95 s
    l0 = song.lines[0]
    assert move.touched == [0, 1] and onsets(l0) == [1.0, 1.2, 1.4] and l0.end == 2.0
    assert [(u.start, u.end) for s in l0.syllables() for u in s.units][-1] == (1.4, 2.0)
    song = setup()
    move_line(song, 1, start=1.3)                            # an onset (1.4 s) passes it: scaled
    l0 = song.lines[0]
    assert [(s.start, s.end) for s in l0.syllables()] == [(1.0, 1.04), (1.04, 1.08), (1.08, 1.3)]
    assert l0.flags == [] and not l0.needs_review
    assert_consistent(song)


def test_end_edit_scales_and_pushes_the_following_lines() -> None:
    song = song_of(even(0, 1.0, 0.8, 4, units=2), even(1, 2.0, 0.8, 2), even(2, 3.0))
    move = move_line(song, 0, end=1.4)
    assert (move.shift, move.touched, move.pushed) == (0.0, [0], [])
    assert [(s.start, s.end) for s in song.lines[0].syllables()] == [(1.0, 1.1), (1.1, 1.2), (1.2, 1.3), (1.3, 1.4)]
    assert song.lines[0].end == 1.4 and "manual_timing" in song.lines[0].flags
    move = move_line(song, 0, end=2.5)                        # past line 1's start: pushed to the new end
    assert move.pushed == [1] and song.lines[1].start == 2.5 and song.lines[2].start == 3.0
    assert song.lines[0].end == 2.5 and rel(song.lines[0]) == [0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0]
    assert onsets(song.lines[1]) == [2.5, 2.9] and song.lines[1].end == 3.0     # tail clipped at line 2
    move = move_line(song, 0, end=3.9)
    assert move.pushed == [1, 2] and song.lines[2].start == pytest.approx(3.9 + MIN_LINE_DUR)
    assert_consistent(song)
    with pytest.raises(ValueError, match="at least 0.1 s after its start"):
        move_line(song, 0, end=1.05)
    with pytest.raises(ValueError, match="the song ends at 0:10.00"):
        move_line(song, 2, end=10.5)


def test_untimed_line_gets_timing_from_a_start_edit() -> None:
    song = song_of(even(0, 1.0, 0.8, 2), even(1, None, n=6), even(2, 5.0, 0.8, 2), even(3, None, n=2),
                   duration=8.0)
    for bad in ({"shift": 0.1}, {"end": 3.0}):
        with pytest.raises(ValueError, match="Line 2 has no timing yet"):
            move_line(song, 1, **bad)
    move = move_line(song, 1, start=2.0)
    ln = song.lines[1]
    assert (move.touched, move.pushed) == ([1], [])
    assert (ln.start, ln.end) == (2.0, 3.5)                  # 0.25 s per syllable
    assert [(s.start, s.end) for s in ln.syllables()][:2] == [(2.0, 2.25), (2.25, 2.5)]
    assert all((u.start, u.end) == (s.start, s.end) for s in ln.syllables() for u in s.units)
    assert "manual_timing" in ln.flags and not ln.needs_review
    move = move_line(song, 3, start=7.5)                     # the last line: at least 1 s, but not past the end
    assert (song.lines[3].start, song.lines[3].end) == (7.5, 8.0)
    assert_consistent(song)
    song = song_of(even(0, 1.0, 0.8, 2), even(1, None, n=6), even(2, 5.0, 0.8, 2))
    move = move_line(song, 1, start=4.9)                     # just before line 3: pushes it, stays short
    assert move.pushed == [2] and song.lines[2].start == 5.15
    assert (song.lines[1].start, song.lines[1].end) == (4.9, 5.15)
    assert_consistent(song)


def test_flags_of_edited_pushed_and_squeezed_lines() -> None:
    song = song_of(even(0, 1.0, review=True), even(1, 2.0, flags=("manual_timing",)), even(2, 3.0, review=True))
    move_line(song, 0, start=1.9)                            # pushes line 1 (timed by hand before)
    l0, l1 = song.lines[0], song.lines[1]
    assert l0.flags == ["manual_timing"] and not l0.needs_review
    assert l1.flags == ["manual_timing", "pushed"] and l1.needs_review
    move_line(song, 1, shift=0.1)                            # checking a pushed line clears the flag
    assert l1.flags == ["manual_timing"] and not l1.needs_review
    move_line(song, 2, start=2.5)                            # line 1 only squeezed: keeps its flags
    assert l1.flags == ["manual_timing"] and not l1.needs_review and l1.end == 2.5
    assert song.lines[2].flags == ["manual_timing"] and not song.lines[2].needs_review
    assert_consistent(song)


def test_edit_validation_and_noop() -> None:
    song = song_of(even(0, 1.0), even(1, 2.0))
    for bad in ({}, {"shift": 0.1, "start": 1.0}, {"start": 1.0, "end": 2.0}):
        with pytest.raises(ValueError, match="exactly one"):
            move_line(song, 0, **bad)
    with pytest.raises(ValueError, match="finite"):
        move_line(song, 0, shift=float("nan"))
    with pytest.raises(IndexError):
        move_line(song, 5, shift=0.1)
    move_line(song, 1, start=2.0)                            # same start: the line is confirmed by hand
    assert "manual_timing" in song.lines[1].flags
    move = move_line(song, 1, start=2.0)                     # nothing left to change
    assert (move.touched, move.pushed, move.before) == ([], [], {})


def test_formatting_helpers() -> None:
    assert fmt_clock(141.2) == "2:21.20" and fmt_clock(59.999) == "1:00.00" and fmt_clock(0) == "0:00.00"
    assert line_list([31]) == "line 32" and line_list([31, 32, 34]) == "lines 32–33, 35"


# ----------------------------------------------------------------------------- HTTP: edit + undo


class FakeSession:
    def __init__(self, song_id: str) -> None:
        self.song_id, self.reloads = song_id, []

    def reload_lines(self, song: Song) -> None:
        self.reloads.append([ln.start for ln in song.lines])

    def snapshot(self) -> dict:
        return {"type": "state", "song_id": self.song_id, "mode": "mix", "status": "tracking", "ref_time": 0.0,
                "tempo": 1.0, "confidence": 1.0, "live_time": 0.0, "level_db": None, "line_idx": 0, "syl_idx": 0,
                "t": time.time(), "age": 0.0}

    def stop(self) -> None:
        pass


@pytest.fixture
def client(tmp_path: Path):
    root = tmp_path / "songs"
    save_song(song_of(even(0, 1.0), even(1, 2.0, review=True), even(2, 3.0), even(3, None), duration=6.0),
              root / "demo")
    app = create_app(root, session_factory=lambda *a: None, allowed_hosts=("testserver",))
    with TestClient(app) as c:
        c.hub, c.sdir = app.state.hub, root / "demo"
        yield c


def lines_on_disk(c: TestClient) -> list[dict]:
    return load_song(c.sdir).to_dict()["lines"]


def test_edit_endpoint_pushes_reloads_and_reports(client: TestClient) -> None:
    sess = client.hub.session = FakeSession("demo")
    r = client.put("/api/songs/demo/lines/2", json={"start": 1.2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pushed"] == [0, 1] and body["clamped"] is False and body["rev"] == 1 and body["undo_left"] == 1
    assert body["shift"] == pytest.approx(-1.8) and set(body["lines"]) == {"0", "1", "2"}
    assert body["lines"]["1"]["flags"] == ["pushed"] and body["lines"]["1"]["needs_review"] is True
    assert [ln["start"] for ln in lines_on_disk(client)] == [0.7, 0.95, 1.2, None]
    assert sess.reloads == [[0.7, 0.95, 1.2, None]]          # the running session got the new lines
    assert client.get("/api/songs/demo").json()["rev"] == 1
    r = client.put("/api/songs/demo/lines/3", json={"start": 4.0})          # the untimed line
    assert r.status_code == 200 and r.json()["lines"]["3"]["end"] == 5.0
    r = client.put("/api/songs/demo/lines/2", json={"end": 1.5}).json()
    assert r["pushed"] == [] and set(r["lines"]) == {"2"} and r["lines"]["2"]["end"] == 1.5
    # refused edits save nothing and add no undo step
    disk = lines_on_disk(client)
    for body, code in (({"start": -1}, 422), ({"start": 6.0}, 422), ({"end": 1.25}, 422), ({}, 422),
                       ({"shift": 1, "end": 2}, 422)):
        assert client.put("/api/songs/demo/lines/2", json=body).status_code == code, body
    assert client.put("/api/songs/demo/lines/9", json={"shift": 0.1}).status_code == 404
    assert "before the beginning" in client.put("/api/songs/demo/lines/0", json={"start": -1}).json()["detail"]
    assert lines_on_disk(client) == disk and client.get("/api/songs/demo").json()["rev"] == 3
    assert len(client.hub.undo["demo"]) == 3
    client.hub.session = None


def test_undo_restores_each_edit_in_turn(client: TestClient) -> None:
    states = [lines_on_disk(client)]
    for body in ({"start": 1.2}, {"end": 1.6}, {"shift": 0.1}):
        assert client.put("/api/songs/demo/lines/2", json=body).status_code == 200
        states.append(lines_on_disk(client))
    r = client.post("/api/songs/demo/undo")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["undone"] == "line 3 start" and body["undo_left"] == 2 and body["rev"] == 4
    assert set(body["lines"]) == {"2"} and lines_on_disk(client) == states[2]
    assert client.post("/api/songs/demo/undo").json()["undone"] == "line 3 end"
    assert lines_on_disk(client) == states[1]
    body = client.post("/api/songs/demo/undo").json()
    assert set(body["lines"]) == {"0", "1", "2"} and body["undo_left"] == 0      # the pushed lines come back
    assert lines_on_disk(client) == states[0]
    r = client.post("/api/songs/demo/undo")
    assert r.status_code == 409 and r.json()["detail"] == "Nothing to undo"
    # an edit after an undo starts a new branch; undoing it returns to the restored state
    client.put("/api/songs/demo/lines/0", json={"shift": 0.1})
    client.post("/api/songs/demo/undo")
    assert lines_on_disk(client) == states[0]
    assert client.post("/api/songs/nope/undo").status_code == 404
    assert client.post("/api/songs/..x/undo").status_code == 400


def test_undo_is_refused_after_the_lines_changed_on_disk(client: TestClient) -> None:
    client.put("/api/songs/demo/lines/1", json={"shift": 0.1})
    song = load_song(client.sdir)                             # a re-prep rewrites song.json
    song.lines[0].text = "REPREPPED"
    save_song(song, client.sdir)
    r = client.post("/api/songs/demo/undo")
    assert r.status_code == 409 and "changed on disk" in r.json()["detail"]
    assert load_song(client.sdir).lines[0].text == "REPREPPED"
    assert client.post("/api/songs/demo/undo").json()["detail"] == "Nothing to undo"     # history dropped
    # steps from before an on-disk change are dropped when the next edit is recorded
    client.put("/api/songs/demo/lines/1", json={"shift": 0.1})
    song = load_song(client.sdir)
    song.lines[0].start = 0.9
    save_song(song, client.sdir)
    client.put("/api/songs/demo/lines/2", json={"shift": 0.1})
    assert client.post("/api/songs/demo/undo").status_code == 200
    assert client.post("/api/songs/demo/undo").status_code == 409
    # a change of the song's meta only (e.g. its title or style) keeps the history
    client.put("/api/songs/demo/lines/2", json={"shift": 0.1})
    song = load_song(client.sdir)
    song.meta.title = "Renamed"
    save_song(song, client.sdir)
    assert client.post("/api/songs/demo/undo").status_code == 200
    assert load_song(client.sdir).meta.title == "Renamed"


def test_undo_history_is_bounded(client: TestClient) -> None:
    for k in range(app_mod.UNDO_DEPTH + 5):
        assert client.put("/api/songs/demo/lines/2", json={"shift": 0.01 if k % 2 else -0.01}).status_code == 200
    assert len(client.hub.undo["demo"]) == app_mod.UNDO_DEPTH


# ----------------------------------------------------------------------------- console Lines panel

LINES_HARNESS = r"""
// The real static/console.js with a stub DOM: Set start / Set end, Tap mode (T) and Undo (Cmd+Z).
const CONSOLE = process.argv[2];
const ELS = new Map(), DOC = {}, SCROLLS = [];
class El {
  constructor(sel) {
    this.sel = sel; this.dataset = {}; this.style = { setProperty() {} }; this.hidden = false; this.textContent = "";
    this._html = ""; this.value = ""; this.checked = false; this.disabled = false; this.title = ""; this.files = [];
    this.handlers = {}; this.attrs = {}; this.readyState = 0; this.src = ""; this.scrollTop = 0; this.currentTime = 0;
    const cls = new Set(); this.cls = cls;
    this.classList = { toggle(c, on) { (on ?? !cls.has(c)) ? cls.add(c) : cls.delete(c); }, add(c) { cls.add(c); },
                       remove(c) { cls.delete(c); }, contains(c) { return cls.has(c); } };
    this.elements = new Proxy({}, { get: (t, k) => (typeof k === "string" ? (t[k] ||= new El(`${sel}.${k}`)) : undefined) });
  }
  set innerHTML(h) { this._html = h; } get innerHTML() { return this._html; }
  addEventListener(t, f) { (this.handlers[t] ||= []).push(f); }
  async fire(t, ev = {}) { for (const f of this.handlers[t] || []) await f({ preventDefault() {}, target: this, currentTarget: this, ...ev }); }
  setAttribute(k, v) { this.attrs[k] = v; } getAttribute(k) { return this.attrs[k]; } removeAttribute(k) { delete this.attrs[k]; }
  querySelector(s) { return el(`${this.sel} ${s}`); }
  querySelectorAll(s) { return s === "tr.is-armed" ? [...ELS.values()].filter((e) => e.cls.has("is-armed")) : []; }
  closest() { return null; } append() {} focus() {} reset() {} play() { return Promise.resolve(); }
  scrollIntoView(o) { SCROLLS.push([this.sel.match(/data-idx="(\d+)"/)?.[1], o.block]); }
}
const el = (sel) => { if (!ELS.has(sel)) ELS.set(sel, new El(sel)); return ELS.get(sel); };
globalThis.document = {
  querySelector: el, querySelectorAll: () => [], createElement: (t) => new El(t), activeElement: null,
  addEventListener(t, f) { (DOC[t] ||= []).push(f); },
};
globalThis.requestAnimationFrame = () => 0;
globalThis.matchMedia = () => ({ matches: true });
globalThis.location = { protocol: "http:", host: "test:8000" };
let sock = null;
globalThis.WebSocket = class { constructor(url) { this.url = url; sock = this; } close() {} };

const LINES = [[0.5, 1.3], [1.5, 2.3], [null, null], [3.5, 4.3]];
const line = (i) => ({ idx: i, text: `line ${i}`, start: LINES[i][0], end: LINES[i][1], words: [], flags: [],
                       needs_review: false, confidence: 0.8 });
const CALLS = [], PUSHED = [];
let rev = 0, inflight = 0, maxInflight = 0;
globalThis.fetch = async (path, opts = {}) => {
  const body = opts.body ? JSON.parse(opts.body) : undefined;
  const method = opts.method || "GET";
  CALLS.push({ path, method, body });
  let data = {};
  const m = path.match(/^\/api\/songs\/(\w+)(?:\/lines\/(\d+)|\/(undo))?$/);
  if (path === "/api/songs") data = [{ id: "A", title: "Song A" }, { id: "B", title: "Song B" }];
  else if (m && method === "GET") data = { id: m[1], meta: { title: `Song ${m[1]}` }, stems: ["mix"], rev, lines: LINES.map((_, i) => line(i)) };
  else if (m && m[2] != null) {
    inflight++; maxInflight = Math.max(maxInflight, inflight);
    await new Promise((r) => setTimeout(r, 15));
    inflight--;
    const i = Number(m[2]);
    if (body.start != null) LINES[i] = [body.start, LINES[i][1] ?? body.start + 1];
    if (body.end != null) LINES[i] = [LINES[i][0], body.end];
    const pushed = PUSHED.shift() || [];
    for (const j of pushed) LINES[j] = [LINES[j][0] - 0.3, LINES[j][1] - 0.3];
    data = { song_id: m[1], rev: ++rev, pushed, shift: 0, clamped: false, undo_left: rev,
             lines: Object.fromEntries([i, ...pushed].map((j) => [j, line(j)])) };
  } else if (m && m[3]) data = { song_id: m[1], rev: ++rev, undone: "line 1 start", undo_left: 1, lines: { 0: line(0) } };
  else if (path === "/api/jobs" || path.startsWith("/api/devices")) data = [];
  return { ok: true, status: 200, headers: { get: () => "application/json" }, json: async () => data };
};

await import(CONSOLE);
const tick = (ms = 5) => new Promise((r) => setTimeout(r, ms));
await tick();
sock.onopen();
sock.onmessage({ data: JSON.stringify({ type: "state", status: "idle", song_id: null, mode: null, age: 0, ref_time: 0 }) });
const tbody = el("#t-lines tbody"), player = el("#player"), notice = () => el("#notice").textContent;
const select = async (id) => { await el("#songs").fire("click", { target: { closest: () => ({ dataset: { id } }) } }); await tick(); };
const button = async (idx, data) => {
  const btn = { dataset: data, closest: () => ({ dataset: { idx: String(idx) } }) };
  await tbody.fire("click", { target: { closest: (s) => (s === "button" ? btn : null) } });
};
const numberCell = async (idx) => {
  const cell = { closest: () => ({ dataset: { idx: String(idx) } }) };
  await tbody.fire("click", { target: { closest: (s) => (s === "td.c-idx" ? cell : null) } });
};
const key = async (k, extra = {}) => {
  for (const f of DOC.keydown || []) await f({ key: k, preventDefault() {}, target: { closest: () => null }, ...extra });
};
const puts = () => CALLS.filter((c) => c.method !== "GET").map((c) => [c.path.split("/").slice(3).join("/"), c.body ?? null]);
const armed = () => tbody.innerHTML.match(/data-idx="(\d+)" class="[^"]*is-armed/)?.[1] ?? null;
const out = {};

await select("A");
out.rendered = { setStart: (tbody.innerHTML.match(/data-set="start" type="button" (disabled)?/g) || []).length,
                 setEndDisabled: (tbody.innerHTML.match(/data-set="end" type="button" disabled/g) || []).length };
await button(2, { set: "start" });                         // nothing loaded in the player yet
out.notLoaded = notice();
Object.assign(player, { readyState: 4, currentTime: 2.756 });
PUSHED.push([1]);
await button(2, { set: "start" }); await tick(40);          // the untimed line gets timing
out.setStart = notice();
await button(0, { set: "end" }); await tick(40);
out.setEnd = notice();
// Tap mode: arms the first line at / after the playhead; T sets it and arms the next one
player.currentTime = 3.2;
await el("#b-tap").fire("click");
out.tapOn = { pressed: el("#b-tap").attrs["aria-pressed"], armed: tap(), notice: notice() };
function tap() { return [...ELS.values()].filter((e) => e.cls.has("is-armed")).map((e) => e.sel.match(/data-idx="(\d+)"/)[1]); }
player.currentTime = 3.61;
await key("t");
out.afterT = tap();
await numberCell(1);
out.afterCell = tap();
player.currentTime = 1.4; await key("t");
player.currentTime = 1.45; await key("T", { repeat: true });    // a held key does not repeat taps
player.currentTime = 2.9; await key("t");                      // a second tap before the first answer
await tick(80);
out.armedRendered = armed();
out.maxInflight = maxInflight;
await key("t", { target: { closest: () => ({}) } });          // typing in a field: no tap
await key("z", { metaKey: true }); await tick(40);
out.undo = notice();
await key("Escape");
out.tapOff = el("#b-tap").attrs["aria-pressed"];
await key("t"); await tick(40);
out.puts = puts();
await el("#b-tap").fire("click");
await select("B");
out.tapAfterSwitch = el("#b-tap").attrs["aria-pressed"];
console.log(JSON.stringify(out));
process.exit(0);
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_console_lines_panel_set_tap_and_undo(tmp_path: Path) -> None:
    path = tmp_path / "harness.mjs"
    path.write_text(LINES_HARNESS, encoding="utf-8")
    res = subprocess.run(["node", str(path), (STATIC / "console.js").as_uri()], capture_output=True, text=True,
                         check=True, timeout=60)
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["rendered"] == {"setStart": 4, "setEndDisabled": 1}      # Set start works on untimed rows
    assert "Play the recording first" in out["notLoaded"]
    assert out["setStart"] == "Line 3 now starts at 0:02.76 (line 2 pushed earlier: check it)"
    assert out["setEnd"] == "Line 1 now ends at 0:02.76"
    assert out["tapOn"]["pressed"] == "true" and out["tapOn"]["armed"] == ["3"]
    assert "press T when line 4 starts" in out["tapOn"]["notice"]
    assert out["afterT"] == [] and out["afterCell"] == ["1"]          # line 4 was the last one
    assert out["armedRendered"] == "3"
    assert out["maxInflight"] == 1                                     # taps reach the server one by one
    assert out["undo"] == "Undid the line 1 start edit (1 more to undo)"
    assert out["tapOff"] == "false"
    assert out["puts"] == [["A/lines/2", {"start": 2.76}], ["A/lines/0", {"end": 2.76}],
                           ["A/lines/3", {"start": 3.61}], ["A/lines/1", {"start": 1.4}],
                           ["A/lines/2", {"start": 2.9}], ["A/undo", None]]
    assert out["tapAfterSwitch"] == "false"
