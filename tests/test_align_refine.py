"""Qwen3-FA refinement logic without the model: window planning, agreement / adoption decisions."""
from __future__ import annotations

import numpy as np
import pytest

from lyricprompter.prep import refine as R
from lyricprompter.text.model import AlignUnit, Line, Syllable, Word


def timed_line(idx: int, starts: list[float], lang: str = "en", dur: float = 0.3) -> Line:
    sylls = [Syllable(f"s{k}", (k, k + 1), [AlignUnit(f"s{k}", "ta")], start=t, end=t + dur)
             for k, t in enumerate(starts)]
    ln = Line(idx=idx, text="x" * len(starts), words=[Word("x", (0, len(starts)), lang, syllables=sylls)])
    ln.start, ln.end = sylls[0].start, sylls[-1].end
    return ln


def test_clean_unit_and_language():
    assert R.clean_unit("weiss,") == "weiss"
    assert R.clean_unit("名前") == "名前" and R.clean_unit("can't!") == "can't" and R.clean_unit("…") == ""
    ln = Line(idx=0, text="", words=[
        Word("a", (0, 1), "ko", syllables=[Syllable("a", (0, 1)), Syllable("b", (1, 2))]),
        Word("c", (2, 3), "en", syllables=[Syllable("c", (2, 3))])])
    assert R.line_language(ln) == "ko"
    assert R.line_language(Line(idx=0, text="")) == "x"


def test_plan_windows_between_trusted_neighbours_with_anchors():
    lines = [timed_line(0, [1.0, 1.5]), timed_line(1, [4.0, 4.5]), timed_line(2, [6.0, 6.5]),
             timed_line(3, [9.0, 9.5])]
    opts = R.RefineOptions(max_win_s=30.0, pad_s=0.5)
    (w,) = R.plan_windows(lines, [1, 2], duration=12.0, opts=opts)
    assert w.lines == [1, 2] and w.anchors == [0, 3]
    assert w.lo == pytest.approx(1.8) and w.hi == pytest.approx(9.0)
    assert (w.t0, w.t1) == pytest.approx((0.5, 10.3))
    (w,) = R.plan_windows(lines, [1, 2], 12.0, R.RefineOptions(max_win_s=30.0, anchors=False))
    assert w.anchors == [] and (w.t0, w.t1) == pytest.approx((1.3, 9.5))
    ws = R.plan_windows(lines, [1, 3], 12.0, opts)
    assert [x.lines for x in ws] == [[1], [3]]
    assert ws[1].hi == pytest.approx(12.0) and ws[1].t1 == pytest.approx(12.0)


def _covers(w: R.Window, ln: Line, pad: float = 0.0) -> bool:
    return w.t0 <= ln.start - pad + 1e-9 and ln.end + pad <= w.t1 + 1e-9


def test_plan_windows_splits_long_gaps():
    lines = [timed_line(i, [10.0 * i, 10.0 * i + 1]) for i in range(8)]
    ws = R.plan_windows(lines, list(range(1, 7)), duration=80.0, opts=R.RefineOptions(max_win_s=25.0))
    assert [x.lines for x in ws] == [[1, 2], [3, 4], [5, 6]]
    assert all(x.t1 - x.t0 <= 25.0 + 1e-9 for x in ws)
    assert all(_covers(w, lines[t]) for w in ws for t in w.lines)
    assert ws[0].t0 == pytest.approx(lines[0].end - 0.5)           # split points: gap start, midpoints
    assert ws[1].t0 == pytest.approx(0.5 * (lines[2].end + lines[3].start))


def test_plan_windows_contains_flagged_line_far_after_trusted_neighbour():
    # verse ends at 12 s, 35 s instrumental, flagged first line of verse 2 at 48 s, trusted line at 52 s
    lines = [timed_line(0, [10.0, 11.7]), timed_line(1, [48.0, 49.0, 50.0]), timed_line(2, [52.0, 53.0])]
    (w,) = R.plan_windows(lines, [1], duration=60.0, opts=R.RefineOptions())
    assert _covers(w, lines[1], pad=0.5) and (w.t0, w.t1) == pytest.approx((20.8, 50.8))
    assert (w.lo, w.hi) == pytest.approx((12.0, 52.0))
    # two flagged lines in the long gap, the first beyond 30 s of the gap start
    lines = [timed_line(0, [10.0, 11.7]), timed_line(1, [45.0, 46.0]), timed_line(2, [50.0, 51.0]),
             timed_line(3, [70.0, 71.0])]
    ws = R.plan_windows(lines, [1, 2], duration=80.0, opts=R.RefineOptions())
    assert [w.lines for w in ws] == [[1], [2]]
    assert _covers(ws[0], lines[1], pad=0.5) and _covers(ws[1], lines[2], pad=0.5)


def test_plan_windows_first_line_after_long_intro_and_overlong_line():
    lines = [timed_line(0, [40.0, 41.0, 42.0]), timed_line(1, [44.0, 45.0])]
    (w,) = R.plan_windows(lines, [0], duration=60.0, opts=R.RefineOptions())
    assert _covers(w, lines[0], pad=0.5) and (w.t0, w.t1) == pytest.approx((12.8, 42.8))
    long_line = timed_line(1, [20.0, 55.0])                         # MMS smeared it over 35 s
    lines = [timed_line(0, [1.0, 1.5]), long_line, timed_line(2, [70.0, 71.0]), timed_line(3, [75.0])]
    ws = R.plan_windows(lines, [1, 2], duration=80.0, opts=R.RefineOptions())
    assert [w.lines for w in ws] == [[2]] and _covers(ws[0], lines[2])


@pytest.mark.parametrize("seed", range(20))
def test_plan_windows_always_contains_its_lines(seed):
    rng = np.random.default_rng(seed)
    t, lines = 0.0, []
    for i in range(int(rng.integers(3, 15))):
        t += float(rng.choice([0.0, 0.5, 2.0, 15.0, 40.0]))
        n = int(rng.integers(1, 6))
        lines.append(timed_line(i, list(t + 0.4 * np.arange(n))))
        t = lines[-1].end
    targets = sorted(int(x) for x in rng.choice(len(lines), size=int(rng.integers(1, len(lines) + 1)),
                                                replace=False))
    duration = t + 5.0
    for opts in (R.RefineOptions(), R.RefineOptions(anchors=False, max_win_s=12.0)):
        ws = R.plan_windows(lines, targets, duration, opts)
        planned = [x for w in ws for x in w.lines]
        assert sorted(planned) == targets                              # every line once (all fit here)
        for w in ws:
            assert 0.0 <= w.t0 < w.t1 <= duration and w.t1 - w.t0 <= opts.max_win_s + 1e-9
            assert all(_covers(w, lines[x]) for x in w.lines)
            assert all(w.lo - 1e-9 <= lines[x].start and lines[x].end <= w.hi + 1e-9 for x in w.lines)


def _window(lo=0.0, hi=10.0, t0=None, t1=None):
    return R.Window(t0=lo if t0 is None else t0, t1=hi if t1 is None else t1, lines=[1], anchors=[], lo=lo, hi=hi)


def test_line_result_agree_keeps_mms():
    ln = timed_line(1, [2.0, 2.5, 3.0])
    q = {(1, 0): (2.1, 2.4, 0.05), (1, 1): (2.55, 2.9, 0.05), (1, 2): (3.2, 3.5, 0.05)}
    r = R._line_result(ln, 1, q, _window(), True, "en", R.RefineOptions())
    assert r.reason == "agree" and not r.disagree and not r.adopted
    assert r.disagreement == pytest.approx(0.1)


def test_line_result_disagree_adopts_only_confident_inside_gap():
    ln = timed_line(1, [2.0, 2.5, 3.0])
    shifted = {(1, k): (t + 2.0, t + 2.3, 0.4) for k, t in enumerate([2.0, 2.5, 3.0])}
    opts = R.RefineOptions()
    r = R._line_result(ln, 1, shifted, _window(), True, "en", opts)
    assert r.reason == "adopted" and r.adopted and r.disagree
    assert r.syl_times[0] == pytest.approx((4.0, 4.3))
    low = {k: (a, b, 0.05) for k, (a, b, _) in shifted.items()}
    assert R._line_result(ln, 1, low, _window(), True, "en", opts).reason == "low_qwen_conf"
    assert R._line_result(ln, 1, shifted, _window(0.0, 4.0, t1=10.0), True, "en", opts).reason == "outside_gap"
    assert R._line_result(ln, 1, shifted, _window(), False, "en", opts).reason == "anchor_mismatch"
    r = R._line_result(ln, 1, {}, _window(), True, "en", opts)
    assert r.reason == "no_units" and r.disagree and not r.adopted


def test_degenerate_qwen_output_is_never_adopted():
    # MMS put the line at 2.0-3.3 s; Qwen disagrees confidently but its output is unusable
    ln = timed_line(1, [2.0, 2.5, 3.0])
    opts, w = R.RefineOptions(), _window(0.0, 10.0)

    def res(times, active=None):
        q = {(1, k): (a, b, 0.4) for k, (a, b) in enumerate(times)}
        return R._line_result(ln, 1, q, w, True, "en", opts, active, 0.02)

    collapsed = res([(6.0, 6.0)] * 3)                                   # whole line at one instant
    assert collapsed.reason == "collapsed" and collapsed.disagree and not collapsed.adopted
    ln6 = timed_line(1, [2.0, 2.3, 2.6, 2.9, 3.2, 3.5])

    def res6(times):
        q = {(1, k): (a, b, 0.4) for k, (a, b) in enumerate(times)}
        return R._line_result(ln6, 1, q, w, True, "en", opts)

    piled = [(4.0, 4.0), (4.0, 7.6), (7.68, 7.76), (7.76, 7.76), (7.76, 7.76), (7.76, 7.8)]
    assert res6(piled).reason == "collapsed"                            # 4 of 6 onsets in one 80 ms bin
    spread = [(4.0, 4.0), (4.0, 4.2), (4.5, 4.6), (4.6, 4.6), (5.0, 5.2), (5.3, 5.6)]
    assert res6(spread).reason == "adopted"                             # pairs sharing a bin are fine
    edges = [(0.0, 0.0), (0.04, 0.3), (5.0, 5.3), (5.5, 5.8), (9.93, 9.96), (9.96, 10.0)]
    pinned = res6(edges)                                                # clamped to the window start / end
    assert pinned.reason == "pinned_to_window_edge" and pinned.disagree and not pinned.adopted
    assert res([(0.0, 0.0), (0.0, 0.08), (5.0, 5.3)]).reason == "collapsed"   # 2 of 3 in one bin
    active = np.zeros(500, bool)
    active[100:170] = True                                              # vocals only at 2.0-3.4 s
    off = res([(6.0, 6.3), (6.5, 6.8), (7.0, 7.3)], active)
    assert off.reason == "off_vocals" and not off.adopted
    active[300:370] = True                                              # vocals at 6.0-7.4 s too
    assert res([(6.0, 6.3), (6.5, 6.8), (7.0, 7.3)], active).reason == "adopted"
    # one pinned onset (a line that really starts right at the window start) is fine
    assert res([(0.0, 0.3), (5.0, 5.3), (5.5, 5.8)]).reason == "adopted"
    # degenerate output that happens to agree with MMS does not clear the review flag
    near = res([(2.3, 2.3), (2.3, 2.3), (2.3, 2.32)])
    assert near.reason == "collapsed" and near.disagree


def test_degenerate_reason_single_syllable():
    w = _window(0.0, 10.0)
    assert R.degenerate_reason([(4.0, 4.0)], w, R.RefineOptions()) is None   # 'oh': no span check
    assert R.degenerate_reason([(0.04, 0.2)], w, R.RefineOptions()) == "pinned_to_window_edge"


def test_refine_lines_with_fake_qwen(monkeypatch):
    lines = [timed_line(0, [1.0, 1.5]), timed_line(1, [4.0, 4.5]), timed_line(2, [8.0, 8.5])]
    seen = {}

    def fake(seg, units, local_only=True):
        seen["n"] = len(seg)
        seen["units"] = units
        # anchors reproduce MMS (window starts at 0.5 s); the target is 1 s later than MMS
        rel = [0.5, 1.0, 4.5, 5.0, 7.5, 8.0]
        return [(t, t + 0.3) for t in rel], [0.5] * len(units)

    monkeypatch.setattr(R, "qwen_align", fake)
    monkeypatch.setattr(R, "load_qwen", lambda local_only=True: None)
    res = R.refine_lines(lines, np.zeros(16000 * 10, np.float32), [1])
    assert seen["units"] == ["s0", "s1"] * 3 and seen["n"] == int(9.3 * 16000) - 8000
    lr = res.lines[1]
    assert lr.reason == "adopted" and lr.syl_times[0] == pytest.approx((5.0, 5.3))
    assert res.windows[0]["anchor_err"] == pytest.approx(0.0)
    summary = res.summary()
    assert summary["n_lines"] == 1 and summary["n_adopted"] == 1 and "syl_times" not in summary["lines"]["1"]


def test_qwen_cached_is_a_local_lookup(monkeypatch, tmp_path):
    import huggingface_hub.constants as hf_constants  # import before patching: constants read env at import

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    assert R.qwen_cached() is False                       # empty cache: no download attempted


def test_qwen_cached_needs_every_file(monkeypatch):
    import huggingface_hub

    present = {"model.safetensors"}                      # interrupted download: weights only
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo, f: f"/cache/{f}" if f in present else None)
    assert R.qwen_cached() is False
    present.update(R.QWEN_FILES)
    assert R.qwen_cached() is True
