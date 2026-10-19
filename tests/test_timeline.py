from __future__ import annotations

import math

from lyricprompter.text.model import Line, Syllable, Word
from lyricprompter.timeline import Timeline


def _line(idx, sylls, start=None, end=None, auto=True):
    """sylls: list of (text, start, end)."""
    ss = [Syllable(text=t, span=(k, k + 1), start=a, end=b) for k, (t, a, b) in enumerate(sylls)]
    w = Word(text="".join(s.text for s in ss), span=(0, len(ss)), lang="ko", syllables=ss)
    if auto:
        start = start if start is not None else next((s.start for s in ss if s.start is not None), None)
        end = end if end is not None else next((s.end for s in reversed(ss) if s.end is not None), None)
    return Line(idx=idx, text=w.text, words=[w] if ss else [], start=start, end=end)


def _song():
    return [_line(0, [("a", 1.0, 1.5), ("b", 1.5, 2.0), ("c", 2.5, 3.0)]),
            _line(1, [("d", 4.0, 5.0), ("e", 5.0, 6.0)]),
            _line(2, [("f", 6.0, 7.0)])]


def test_before_first_line():
    c = Timeline(_song()).locate(0.5)
    assert (c.line_idx, c.syl_idx, c.next_line_idx, c.in_gap) == (-1, -1, 0, True)


def test_inside_syllables_and_progress():
    tl = Timeline(_song())
    c = tl.locate(1.25)
    assert (c.line_idx, c.syl_idx, c.next_line_idx, c.in_gap) == (0, 0, 1, False)
    assert math.isclose(c.syl_progress, 0.5)
    c = tl.locate(2.2)                                  # between syllables inside the line
    assert (c.line_idx, c.syl_idx, c.in_gap) == (0, 1, False) and c.syl_progress == 1.0
    c = tl.locate(1.5)                                  # boundary belongs to the next syllable
    assert c.syl_idx == 1 and c.syl_progress == 0.0


def test_gap_between_lines_and_after_last():
    tl = Timeline(_song())
    c = tl.locate(3.5)
    assert (c.line_idx, c.syl_idx, c.next_line_idx, c.in_gap) == (0, -1, 1, True)
    c = tl.locate(6.0)                                  # line 1 ends exactly where line 2 starts
    assert (c.line_idx, c.syl_idx, c.in_gap) == (2, 0, False)
    c = tl.locate(10.0)
    assert (c.line_idx, c.next_line_idx, c.in_gap) == (2, -1, True)
    assert tl.line_start(1) == 4.0 and tl.line_end(1) == 6.0


def test_none_and_nan_times():
    tl = Timeline(_song())
    assert tl.locate(None).line_idx == -1
    assert tl.locate(float("nan")).line_idx == -1
    assert tl.locate(float("inf")).line_idx == 2


def test_untimed_and_partial_lines():
    lines = [_line(0, [("a", 1.0, 2.0)]),
             _line(1, [("x", None, None), ("y", None, None)]),          # never aligned
             _line(2, [("b", 3.0, None), ("c", None, 4.0)]),             # partial syllable times
             _line(3, [], start=None, end=None),                          # empty line
             _line(4, [("d", 5.0, 6.0)], start=None, end=None, auto=False)]
    tl = Timeline(lines)
    assert tl.line_start(1) == 2.0 and tl.line_end(1) == 2.0              # zero length at previous end
    c = tl.locate(2.5)
    assert c.line_idx == 1 and c.in_gap and c.next_line_idx == 2
    c = tl.locate(3.2)
    assert (c.line_idx, c.syl_idx, c.in_gap) == (2, 0, False)
    c = tl.locate(3.9)                                                    # 'c' starts at b's (missing) end
    assert c.syl_idx in (0, 1)
    assert tl.line_start(3) == 4.0 and tl.line_end(3) == 4.0
    c = tl.locate(5.5)                                                    # line times from syllables
    assert (c.line_idx, c.syl_idx, c.in_gap) == (4, 0, False)
    assert math.isclose(c.syl_progress, 0.5)


def test_non_monotonic_starts_are_repaired():
    lines = [_line(0, [("a", 2.0, 3.0)]), _line(1, [("b", 1.0, 1.5)]), _line(2, [("c", 4.0, 5.0)])]
    tl = Timeline(lines)
    assert tl.line_start(1) >= tl.line_start(0)
    assert tl.locate(4.5).line_idx == 2


def test_empty_timeline():
    c = Timeline([]).locate(1.0)
    assert (c.line_idx, c.next_line_idx, c.in_gap) == (-1, -1, True)


def test_line_without_end_runs_to_next_start():
    lines = [_line(0, [("a", 1.0, None)], end=None), _line(1, [("b", 3.0, 4.0)])]
    tl = Timeline(lines)
    assert tl.line_end(0) == 3.0
    c = tl.locate(2.5)
    assert (c.line_idx, c.syl_idx, c.in_gap) == (0, 0, False)
