"""Aligner tests on synthetic CTC emissions (no model): unit->time mapping, gating, stars, interpolation,
monotonicity, syllable-end shaping, confidence / review flags and the align_lines contract."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from lyricprompter.prep import align as A
from lyricprompter.prep import refine as R
from lyricprompter.text.model import AlignUnit, Line, Syllable, Word

LABELS = ("-", "a", "i", "e", "n", "o", "u", "t", "s", "r", "m", "k", "l", "d", "g", "h", "y", "b", "p", "w",
          "c", "v", "j", "z", "f", "'", "q", "x", "*")
FS = A.FRAME_S


def mk_line(idx: int, words: list[str], backing: bool = False) -> Line:
    """'na-mo' = one word, two syllables; '(oh)' = backing word; 'ー' (no letters) = empty-roman unit."""
    out, pos, texts = [], 0, []
    for w in words:
        kind = "backing" if w.startswith("(") else "lex"
        w = w.strip("()")
        start, sylls = pos, []
        for part in w.split("-"):
            roman = "".join(c for c in part.lower() if "a" <= c <= "z" or c == "'")
            sylls.append(Syllable(part, (pos, pos + len(part)), [AlignUnit(part, roman)]))
            pos += len(part)
        out.append(Word(w.replace("-", ""), (start, pos), "en", kind, syllables=sylls))
        texts.append(w.replace("-", ""))
        pos += 1
    return Line(idx=idx, text=" ".join(texts), words=out, backing=backing)


def synth_emission(n_frames: int, events: list[tuple[str, int, float]]) -> A.Emission:
    """Blank-dominated posteriors; each (letter, frame, prob) puts that letter's mass on one frame.
    Like MMS_FA(with_star) the '*' column is log 1 on every frame."""
    p = np.full((n_frames, len(LABELS) - 1), 1e-4)
    p[:, 0] = 1.0
    for c, f, prob in events:
        p[f, :] = 1e-4
        p[f, LABELS.index(c)] = prob
        p[f, 0] = 1.0 - prob
    p /= p.sum(axis=1, keepdims=True)
    lp = np.concatenate([np.log(p), np.zeros((n_frames, 1))], axis=1).astype(np.float32)
    return A.Emission(log_probs=torch.from_numpy(lp), labels=LABELS)


def spell(word: str, frame: int, prob: float = 0.9, step: int = 2) -> list[tuple[str, int, float]]:
    return [(c, frame + step * i, prob) for i, c in enumerate(word)]


RAW = A.AlignOptions(end_mode="raw", snap_onsets="none")


def run(lines, em, gate=None, active=None, opts=RAW):
    masks = None
    if gate is not None or active is not None:
        n = em.n_frames
        g = gate if gate is not None else np.ones(n, bool)
        a = active if active is not None else g
        masks = A.VocalMasks(gate=g, active=a, sustain=a)
    report, raw = A.align_with_emission(lines, em, masks=masks, options=opts)
    return report, raw


def all_units(lines):
    return [u for ln in lines for s in ln.syllables() for u in s.units]


def test_mms_labels_match_bundle():
    assert A.mms_labels() == LABELS


def test_unit_syllable_line_times_from_token_spans():
    lines = [mk_line(0, ["na-mo"]), mk_line(1, ["ri"])]
    em = synth_emission(300, spell("na", 50) + spell("mo", 100) + spell("ri", 200))
    report, _ = run(lines, em)
    s0, s1 = lines[0].syllables()
    assert (s0.start, s0.end) == pytest.approx((50 * FS, 53 * FS))
    assert (s1.start, s1.end) == pytest.approx((100 * FS, 103 * FS))
    assert lines[0].start == pytest.approx(1.0) and lines[0].end == pytest.approx(2.06)
    assert lines[1].start == pytest.approx(4.0)
    assert s0.units[0].score == pytest.approx(0.9, abs=0.05)
    assert report.method == "mms_fa" and report.flagged == []
    assert report.line_confidence[0] == pytest.approx(lines[0].confidence)


def test_gate_moves_line_out_of_silent_region():
    lines = [mk_line(0, ["na"])]
    em = synth_emission(200, spell("na", 20, 0.9) + spell("na", 120, 0.3))
    run(lines, em, opts=A.AlignOptions(gate=False, end_mode="raw", snap_onsets="none"))
    assert lines[0].start == pytest.approx(0.4)
    gate = np.zeros(200, bool)
    gate[100:150] = True
    report, _ = run(lines, em, gate=gate)
    assert lines[0].start == pytest.approx(2.4)
    assert report.method == "mms_fa+gate"


def test_build_tokens_stars_between_lines_and_words():
    lines = [mk_line(0, ["na", "mo"]), mk_line(1, ["ri"])]
    refs = A.unit_refs(lines)
    d = {c: i for i, c in enumerate(LABELS)}
    star = d["*"]
    toks, owner = A.build_tokens(refs, LABELS)
    assert toks == [star, d["n"], d["a"], d["m"], d["o"], star, d["r"], d["i"], star]
    assert owner == [-1, 0, 0, 1, 1, -1, 2, 2, -1]
    toks_w, _ = A.build_tokens(refs, LABELS, star_between_words=True)
    assert toks_w == [star, d["n"], d["a"], star, d["m"], d["o"], star, d["r"], d["i"], star]
    toks_n, _ = A.build_tokens(refs, LABELS, star_between_lines=False)
    assert toks_n == [star, d["n"], d["a"], d["m"], d["o"], d["r"], d["i"], star]


def test_backing_words_skipped_with_star_placeholder():
    lines = [mk_line(0, ["na", "(oh)", "mo"]), mk_line(1, ["(la)"], backing=True), mk_line(2, ["ri"])]
    refs = A.unit_refs(lines)
    assert [r.aligned for r in refs] == [True, False, True, False, True]
    assert [r.skipped_backing for r in refs] == [False, True, False, True, False]
    d = {c: i for i, c in enumerate(LABELS)}
    toks, _ = A.build_tokens(refs, LABELS)
    assert toks[:7] == [d["*"], d["n"], d["a"], d["*"], d["m"], d["o"], d["*"]]
    toks, _ = A.build_tokens(refs, LABELS, star_for_backing=False)
    assert toks[:5] == [d["*"], d["n"], d["a"], d["m"], d["o"]]


BACKING_EVENTS = spell("na", 50) + spell("oh", 80) + spell("mo", 120) + spell("la", 200) + spell("ri", 300)


def backing_lines():
    return [mk_line(0, ["na", "(oh)", "mo"]), mk_line(1, ["(la)"], backing=True), mk_line(2, ["ri"])]


def test_skipped_backing_is_interpolated_and_not_confidence():
    lines, em = backing_lines(), synth_emission(400, BACKING_EVENTS)
    report, _ = run(lines, em)
    na, oh, mo = lines[0].syllables()
    assert (na.start, mo.start) == pytest.approx((1.0, 2.4))
    assert na.end <= oh.start <= oh.end <= mo.start and oh.units[0].score is None
    assert lines[0].end <= lines[1].start <= lines[2].start
    assert lines[1].confidence is None and report.line_confidence[1] is None and not lines[1].needs_review
    assert report.line_confidence[0] == lines[0].confidence


def test_backing_included_in_main_pass():
    lines, em = backing_lines(), synth_emission(400, BACKING_EVENTS)
    run(lines, em, opts=A.AlignOptions(include_backing=True, end_mode="raw", snap_onsets="none"))
    assert lines[0].syllables()[1].start == pytest.approx(80 * FS)
    assert lines[1].start == pytest.approx(200 * FS)


def test_empty_roman_unit_is_interpolated_between_neighbours():
    lines = [mk_line(0, ["na", "ー", "mo"])]
    em = synth_emission(200, spell("na", 50) + spell("mo", 70))
    run(lines, em)
    na, bar, mo = lines[0].syllables()
    assert na.end <= bar.start <= bar.end <= mo.start
    assert bar.units[0].score is None


def test_line_final_empty_roman_unit_stays_with_its_line():
    """'ッ' (no letters) ends line 0; a long instrumental with a breath blip follows before line 1."""
    lines = [mk_line(0, ["na-mo-ッ"]), mk_line(1, ["ri-ka"])]
    em = synth_emission(700, spell("na", 50) + spell("mo", 56) + spell("ri", 500) + spell("ka", 506))
    active = np.zeros(700, bool)
    active[50:64] = True                       # line 0 sung 1.0-1.28 s
    active[300:315] = True                     # breath / bleed blip at 6.0 s
    active[490:515] = True                     # line 1 from 9.8 s (CTC fires at 10.0 s)
    for opts in (RAW, A.AlignOptions()):
        run(lines, em, active=active, opts=opts)
        mo, tsu = lines[0].syllables()[1:]
        assert tsu.start == pytest.approx(mo.end, abs=0.021) and tsu.end - tsu.start <= A.INTERP_UNIT_S + 0.021
        assert lines[0].end < 2.0 and lines[1].start >= 9.8 - 1e-9
    active[300:315] = False                    # no blip: still not glued to the next line
    run(lines, em, active=active)
    assert lines[0].end < 2.0


def test_line_initial_empty_roman_unit_precedes_its_line():
    lines = [mk_line(0, ["na-mo"]), mk_line(1, ["ー-ri-ka"])]
    em = synth_emission(700, spell("na", 50) + spell("mo", 56) + spell("ri", 500) + spell("ka", 506))
    active = np.zeros(700, bool)
    active[50:64] = True
    active[300:315] = True
    active[495:515] = True
    run(lines, em, active=active)
    bar, ri, _ = lines[1].syllables()
    assert bar.end == pytest.approx(ri.start) == pytest.approx(10.0)
    assert bar.start == pytest.approx(9.9)                                # the line's vocal onset
    active[495:500] = False                                              # onset after CTC: attach directly
    run(lines, em, active=active)
    bar, ri, _ = lines[1].syllables()
    assert (bar.start, bar.end) == pytest.approx((10.0 - A.INTERP_UNIT_S, 10.0))


def test_interpolate_missing_floating_vs_attached_units():
    nan = np.nan
    start, end = np.array([1.0, nan, nan, nan, 9.0]), np.array([1.2, nan, nan, nan, 9.2])
    active = np.zeros(500, bool)
    active[250:260] = True                     # a backing call at 5.0 s
    line = np.array([0, 0, 1, 2, 2])           # 'ッ' of line 0, backing line 1, 'ー' of line 2
    floating = np.array([False, False, True, False, False])
    s, e = A.interpolate_missing(start, end, 10.0, active, line=line, floating=floating)
    assert (s[1], e[1]) == pytest.approx((1.2, 1.2 + A.INTERP_UNIT_S))
    assert s[2] == pytest.approx(5.0) and e[2] == pytest.approx(5.0 + A.INTERP_UNIT_S)
    assert (s[3], e[3]) == pytest.approx((9.0 - A.INTERP_UNIT_S, 9.0))
    s0, _ = A.interpolate_missing(start, end, 10.0, active)            # no line info: all float
    assert s0[1] == pytest.approx(5.0)
    # a line-final unread word sung after a short in-line pause starts at that onset (<= INLINE_PAUSE_S)
    active[80:100] = True                                              # 1.6-2.0 s
    s1, e1 = A.interpolate_missing(start, end, 10.0, active, line=line, floating=floating)
    assert (s1[1], e1[1]) == pytest.approx((1.6, 1.6 + A.INTERP_UNIT_S)) and s1[2] == pytest.approx(5.0)


def test_all_units_monotonic_and_filled():
    lines = [mk_line(0, ["na-mo", "ー"]), mk_line(1, ["ri", "(oh)"]), mk_line(2, ["ka"])]
    em = synth_emission(300, spell("na", 30) + spell("mo", 40) + spell("ri", 100) + spell("ka", 250))
    for opts in (RAW, A.AlignOptions()):
        run(lines, em, active=np.ones(300, bool), opts=opts)
        times = [(u.start, u.end) for u in all_units(lines)]
        assert all(s is not None and e is not None and s <= e for s, e in times)
        flat = [t for se in times for t in se]
        assert flat == sorted(flat)
        for ln in lines:
            for syl in ln.syllables():
                assert syl.start <= syl.end


def test_viterbi_rejects_audio_too_short():
    em = synth_emission(4, [])
    with pytest.raises(ValueError, match="too short"):
        A.viterbi(em.log_probs, [28, 4, 1, 10, 5, 28])


def test_no_alignable_tokens_spreads_units_over_activity():
    lines = [mk_line(0, ["ー", "ー"])]
    em = synth_emission(100, [])
    active = np.zeros(100, bool)
    active[20:60] = True
    run(lines, em, active=active)
    s = lines[0].syllables()
    assert s[0].start == pytest.approx(0.4) and s[1].end == pytest.approx(1.2)


def test_make_monotonic_cuts_previous_end():
    s, e = A.make_monotonic(np.array([0.0, 0.5, 0.4]), np.array([1.0, 0.8, 0.3]))
    assert s.tolist() == [0.0, 0.5, 0.8] and e.tolist() == [0.5, 0.8, 0.8]


def test_interpolate_missing_edges_and_long_gaps():
    nan = np.nan
    s, e = A.interpolate_missing(np.array([nan, 2.0, nan, nan, 6.0, nan]),
                                 np.array([nan, 2.5, nan, nan, 6.5, nan]), duration=10.0)
    assert s[0] == pytest.approx(2.0 - A.INTERP_UNIT_S) and e[0] == pytest.approx(2.0)
    assert s[2] == pytest.approx(2.5) and e[3] == pytest.approx(2.5 + 2 * A.INTERP_UNIT_S)
    assert s[5] == pytest.approx(6.5) and e[5] == pytest.approx(6.5 + A.INTERP_UNIT_S)
    active = np.zeros(500, bool)
    active[100:130] = True                     # tail of the previous unit: not an onset in the gap
    active[200:250] = True                     # singing resumes at 4.0 s inside the 2.5..6.0 gap
    s2, _ = A.interpolate_missing(np.array([nan, 2.0, nan, nan, 6.0, nan]),
                                  np.array([nan, 2.5, nan, nan, 6.5, nan]), duration=10.0, active=active)
    assert s2[2] == pytest.approx(4.0)
    s3, e3 = A.interpolate_missing(np.array([1.0, nan, 1.2]), np.array([1.1, nan, 1.3]), duration=5.0)
    assert (s3[1], e3[1]) == pytest.approx((1.1, 1.2))


def test_shape_ends_close_and_activity():
    s = np.array([1.0, 1.5, 3.0])
    e = np.array([1.2, 1.6, 3.1])
    assert A.shape_ends(s, e, mode="raw").tolist() == e.tolist()
    closed = A.shape_ends(s, e, mode="close", close_gap_s=0.15)
    assert closed.tolist() == pytest.approx([1.2, 1.6, 3.1])          # gaps 0.3 / 1.4 kept
    closed2 = A.shape_ends(s, e, mode="close", close_gap_s=0.35)
    assert closed2.tolist() == pytest.approx([1.5, 1.6, 3.1])          # tiny gap closed
    active = np.zeros(200, bool)
    active[50:70] = True                                               # sustain to 1.4 s
    active[75:120] = True                                              # 1.5..2.4 s
    ext = A.shape_ends(s, e, mode="activity", close_gap_s=0.15, max_extend_s=1.5, active=active)
    assert ext[0] == pytest.approx(1.5)                                # 1.4 then gap 0.1 < 0.15 closed
    assert ext[1] == pytest.approx(2.4)                                # sustained until activity ends
    assert ext[2] == pytest.approx(3.1)


def test_snap_onsets_to_vocal_run_start():
    active = np.zeros(300, bool)
    active[100:180] = True                                             # held pickup note from 2.0 s
    s = A.snap_onsets(np.array([0.5, 3.0]), np.array([0.8, 3.2]), active, max_snap_s=1.5)
    assert s.tolist() == pytest.approx([0.5, 2.0])
    s = A.snap_onsets(np.array([0.5, 3.0]), np.array([0.8, 3.2]), active, max_snap_s=0.5)
    assert s[1] == pytest.approx(3.0)                                  # run onset too far back
    active[40:100] = True                                              # continuous since the previous unit
    s = A.snap_onsets(np.array([0.5, 3.0]), np.array([0.8, 3.2]), active, max_snap_s=1.5)
    assert s[1] == pytest.approx(3.0)


def test_confidence_and_review_threshold():
    lines = [mk_line(0, ["na"]), mk_line(1, ["mo"])]
    em = synth_emission(200, spell("na", 30, 0.9) + spell("mo", 120, 0.05))
    report, _ = run(lines, em)
    assert lines[0].confidence > 0.8 and lines[1].confidence < 0.1
    assert report.flagged == [1] and lines[1].needs_review and "low_confidence" in lines[1].flags
    assert not lines[0].needs_review


def test_apply_syllable_times_splits_units_by_roman_length():
    line = Line(idx=0, text="名前", words=[Word("名前", (0, 2), "ja", syllables=[
        Syllable("名前", (0, 2), [AlignUnit("な", "na"), AlignUnit("ま", "ma"), AlignUnit("え", "e")])])])
    raw = A.RawAlignment(refs=A.unit_refs([line]), start=np.full(3, np.nan), end=np.full(3, np.nan),
                         score=np.full(3, np.nan), n_tokens=0, duration=10.0)
    A.apply_syllable_times(raw, {(0, 0): (1.0, 2.0)})
    assert raw.start.tolist() == pytest.approx([1.0, 1.4, 1.8])
    assert raw.end.tolist() == pytest.approx([1.4, 1.8, 2.0])


def _tone_bursts(n_frames: int, frames: list[int]) -> np.ndarray:
    y = np.zeros(n_frames * A.FRAME_HOP, np.float32)
    t = np.arange(A.FRAME_HOP * 8) / A.SR
    for f in frames:
        y[f * A.FRAME_HOP: (f + 8) * A.FRAME_HOP] = 0.3 * np.sin(2 * np.pi * 250 * t)
    return y


@pytest.fixture
def fake_mms(monkeypatch):
    """compute_emission -> synthetic emission matching tone bursts in the 'vocals'."""
    events = spell("na", 50) + spell("mo", 60) + spell("ri", 150, 0.03) + spell("ka", 250)

    def fake(y, sr=16000, **kw):
        return synth_emission(len(A.to_16k(y, sr)) // A.FRAME_HOP, events)

    monkeypatch.setattr(A, "compute_emission", fake)
    monkeypatch.setattr(A, "release_models", lambda: None)
    y = _tone_bursts(320, [50, 60, 150, 250])
    return y, [mk_line(0, ["na-mo"]), mk_line(1, ["ri"]), mk_line(2, ["(oh)", "ka"])]


def test_align_lines_contract(fake_mms, monkeypatch):
    y, lines = fake_mms
    monkeypatch.setattr(R, "qwen_cached", lambda: False)
    progress = []
    report = A.align_lines(lines, y, 16000, refine="auto", progress=lambda *a: progress.append(a))
    assert report.method == "mms_fa+gate" and report.flagged == [1]
    assert report.details["refine"] == {"skipped": "Qwen3-ForcedAligner not cached locally"}
    assert len(report.line_confidence) == 3 and all(0.0 <= c <= 1.0 for c in report.line_confidence)
    assert {"emission", "viterbi", "post"} <= set(report.timings)
    prev = 0.0
    for ln in lines:
        assert ln.start is not None and ln.end is not None and ln.confidence is not None
        for syl in ln.syllables():
            assert prev <= syl.start <= syl.end
            prev = syl.end
    assert progress[-1][1] == 100
    with pytest.raises(ValueError):
        A.align_lines(lines, y, refine="bogus")
    y44 = np.repeat(y, 2)                                  # resampled input gives the same frame grid
    A.align_lines(lines, y44, 32000, refine="none")
    oh, ka = lines[2].syllables()
    assert ka.start == pytest.approx(5.0, abs=0.05)
    assert lines[1].end <= oh.start <= oh.end <= ka.start


def _fake_qwen(monkeypatch, rel_times, conf=0.6):
    """Qwen3-FA stand-in returning fixed (start, end) times relative to the window start."""
    calls = []

    def fake_qwen(seg, units, local_only=True):
        calls.append(units)
        return [(t, t + 0.1) for t in rel_times[: len(units)]], [conf] * len(units)

    monkeypatch.setattr(R, "qwen_align", fake_qwen)
    monkeypatch.setattr(R, "load_qwen", lambda local_only=True: None)
    monkeypatch.setattr(R, "release_qwen", lambda: None)
    return calls


def test_align_lines_qwen_hybrid_adopts_confident_disagreeing_qwen(fake_mms, monkeypatch):
    _, lines = fake_mms
    y = _tone_bursts(320, [50, 60, 100, 150, 250])           # also singing at 2.0 s (no CTC evidence)
    # window starts 0.5 s before line 0: anchors 'na' 'mo' 'ka' reproduce MMS, 'ri' at 2.0 s (MMS: 3.0 s)
    calls = _fake_qwen(monkeypatch, [0.5, 0.7, 1.5, 4.5])
    report = A.align_lines(lines, y, refine="qwen")
    assert calls == [["na", "mo", "ri", "ka"]]          # anchors + target, backing word left out
    lr = report.details["refine"]["lines"]["1"]
    assert lr["window"][0] == pytest.approx(0.5)
    assert lr["reason"] == "adopted" and lr["disagree"]
    assert report.method == "mms_fa+gate+qwen" and report.flagged == [1]
    assert "mms_qwen_disagree" in lines[1].flags and "qwen" in lines[1].flags
    assert lines[1].start == pytest.approx(2.0, abs=0.01)
    prev = 0.0
    for ln in lines:
        for syl in ln.syllables():
            assert prev <= syl.start <= syl.end
            prev = syl.end


@pytest.mark.parametrize("rel_ri", [0.0, 2.0])       # pinned to the window start / sung where it is silent
def test_align_lines_keeps_mms_when_qwen_output_is_degenerate(fake_mms, monkeypatch, rel_ri):
    y, lines = fake_mms
    A.align_lines(lines, y, refine="none")
    mms = [(ln.start, ln.end) for ln in lines]
    _fake_qwen(monkeypatch, [0.5, 0.7, rel_ri, 4.5])
    report = A.align_lines(lines, y, refine="qwen")
    lr = report.details["refine"]["lines"]["1"]
    assert lr["reason"] in ("pinned_to_window_edge", "off_vocals") and not lr["adopted"] and lr["disagree"]
    assert [(ln.start, ln.end) for ln in lines] == mms
    assert lines[1].needs_review and report.flagged == [1] and "qwen" not in lines[1].flags


def test_qwen_agreement_clears_review_and_low_confidence_flag(fake_mms, monkeypatch):
    y, lines = fake_mms
    _fake_qwen(monkeypatch, [0.5, 0.7, 2.5, 4.5])          # 'ri' at 3.0 s like MMS
    report = A.align_lines(lines, y, refine="qwen")
    assert report.details["refine"]["lines"]["1"]["reason"] == "agree"
    assert report.flagged == [] and not lines[1].needs_review
    assert lines[1].flags == ["mms_qwen_agree"]


def test_align_lines_auto_survives_qwen_failure(fake_mms, monkeypatch):
    y, lines = fake_mms
    released = []

    def broken(*a, **kw):
        raise OSError("Can't load tokenizer (local_files_only=True)")

    monkeypatch.setattr(R, "qwen_cached", lambda: True)
    monkeypatch.setattr(R, "load_qwen", broken)
    monkeypatch.setattr(R, "release_qwen", lambda: released.append(1))
    report = A.align_lines(lines, y, refine="auto")
    assert report.method == "mms_fa+gate" and report.flagged == [1]
    assert "OSError" in report.details["refine"]["error"] and released == [1]
    assert all(ln.start is not None for ln in lines) and "refine" in report.timings
    with pytest.raises(OSError):                            # an explicit request fails loudly
        A.align_lines(lines, y, refine="qwen")
    assert released == [1, 1]


def test_symbol_only_syllables_and_lines_get_times():
    sym = Line(idx=1, text="♪", words=[Word("♪", (0, 1), "x", "sym", syllables=[Syllable("♪", (0, 1))])])
    bare = Line(idx=2, text="...", words=[])
    lines = [mk_line(0, ["na"]), sym, bare, mk_line(3, ["mo"])]
    em = synth_emission(200, spell("na", 30) + spell("mo", 120))
    run(lines, em)
    assert sym.start == sym.end == lines[0].end
    assert bare.start == bare.end == lines[0].end
    assert lines[3].start == pytest.approx(2.4)
    empty = Line(idx=0, text="")
    run([empty, mk_line(1, ["na"])], em)
    assert empty.start is None
