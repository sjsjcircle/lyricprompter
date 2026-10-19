"""Evaluation datasets (LRC / score-CSV ground truth, registry) and the GT-to-syllable matching metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lyricprompter.eval import align_eval as E
from lyricprompter.eval import datasets as D
from lyricprompter.text.model import AlignUnit, Line, Syllable, Word

LRC = """[ti:Test]
[la:EN]

[00:06.82]E  <00:07.87>del  <00:08.33>weiss,  <00:10.00> e  <00:10.99>del  <00:11.49>weiss
[00:12.99]E  <00:14.10>very  <00:14.75> morn  <00:15.17>ing

[01:05.37]Bless  <01:06.33> my  <01:06.92> home  <01:07.50>land
"""


def test_parse_lrc_and_words(tmp_path: Path):
    p = tmp_path / "x.lrc"
    p.write_text(LRC, encoding="utf-8")
    lines = D.parse_lrc(p)
    assert [len(x) for x in lines] == [6, 4, 0, 4]
    assert lines[0][0] == (6.82, "E", True) and lines[0][1] == (7.87, "del", False)
    assert lines[0][3] == (10.0, "e", True)
    text, gt = D.lrc_words(p)
    assert text == "Edelweiss, edelweiss\nEvery morning\n\nBless my homeland"
    assert [(g.text, g.t, g.line) for g in gt[:3]] == [("Edelweiss,", 6.82, 0), ("edelweiss", 10.0, 0),
                                                      ("Every", 12.99, 1)]
    assert gt[-1].text == "homeland" and gt[-1].t == pytest.approx(66.92) and gt[-1].line == 2


def test_norm_chars_and_assign_lines():
    assert D.norm_chars("Weiss, ＡＢ ラーメン!") == "weissabラーメン"
    lyrics = "꽃밭에 앉아서\n\n고운 빛은"
    assert D.assign_lines(list("꽃밭에앉아서고운빛은"), lyrics) == [0] * 6 + [1] * 4


def test_kkotbat_gt_warps_score_onsets(tmp_path: Path):
    (tmp_path / "s.csv").write_text("start,pitch,pitchclass,pitchname,offset,instrument,lyric,syllabic\n"
                                    "0.5,60,0,C4,1.0,voice,꽃,begin\n1.0,60,0,C4,2.0,voice,,\n"
                                    "1.5,60,0,C4,3.0,voice,밭,end\n", encoding="utf-8")
    # warping path (score_frame, ref_frame) at 25 fps: ref runs 2x slower and starts 1 s later
    wp = np.array([[f, 25 + 2 * f] for f in range(200)], dtype=float)
    np.savetxt(tmp_path / "wp.csv", wp, delimiter=",")
    gt = D.kkotbat_gt(tmp_path / "s.csv", tmp_path / "wp.csv")
    assert [s for _, s in gt] == ["꽃", "밭"]
    assert [t for t, _ in gt] == pytest.approx([1.0 + 2 * 0.5, 1.0 + 2 * 1.5])


def test_dataset_names_groups():
    assert D.dataset_names("jamendo") == list(D.JAMENDO_SONGS)
    assert D.dataset_names("edelweiss,ref")[:3] == ["edelweiss", "edelweiss_short", "kkotbat"]
    assert len(D.dataset_names("all")) == len(D.LOADERS)
    with pytest.raises(KeyError):
        D.dataset_names("nope")


def mk_pred_lines(rows: list[list[tuple[str, float, float]]]) -> list[Line]:
    """rows[line] = [(syllable text, start, end), ...]; each syllable its own word."""
    out = []
    for li, row in enumerate(rows):
        words = [Word(t, (0, len(t)), "en",
                      syllables=[Syllable(t, (0, len(t)), [AlignUnit(t, t.lower())], s, e)])
                 for t, s, e in row]
        ln = Line(idx=li, text=" ".join(t for t, _, _ in row), words=words)
        ln.start, ln.end = row[0][1], row[-1][2]
        out.append(ln)
    return out


def test_match_tokens_any_granularity():
    lines = mk_pred_lines([[("Edel", 1.0, 1.4), ("weiss", 1.4, 2.0)],
                           [("Bless", 3.0, 3.5), ("me", 3.6, 4.0)]])
    gt = [D.GTToken("Edelweiss,", 1.1, 0, 2.1), D.GTToken("Bless me", 2.9, 1), D.GTToken("zzz", 9.0, 1)]
    ds = D.AlignDataset("t", Path("m"), Path("v"), "", gt, "word", "en")
    mt = E.match_tokens(ds, lines)
    assert mt.n_gt == 3 and len(mt.gt_t) == 2                      # 'zzz' unmatched
    assert mt.pred_t.tolist() == pytest.approx([1.0, 3.0])         # word onset = first syllable start
    assert mt.pred_end.tolist() == pytest.approx([2.0, 4.0])       # word end = last syllable end
    assert np.isnan(mt.gt_end[1]) and mt.pred_line.tolist() == [0, 1]
    # GT token starting mid-syllable is interpolated inside the syllable ('del' starts 1/4 into 'Edel')
    ds2 = D.AlignDataset("t", Path("m"), Path("v"), "", [D.GTToken("del", 0.0, 0)], "syllable", "en")
    assert E.match_tokens(ds2, lines).pred_t[0] == pytest.approx(1.1)


def test_evaluate_metrics_and_review_recall():
    lines = mk_pred_lines([[("a", 1.0, 1.2), ("b", 2.0, 2.2)], [("c", 5.0, 5.2), ("d", 9.0, 9.2)]])
    lines[1].needs_review = True
    gt = [D.GTToken("a", 1.1, 0, 1.3), D.GTToken("b", 2.0, 0, 2.2), D.GTToken("c", 5.4, 1, 5.5),
          D.GTToken("d", 6.0, 1, 6.2)]
    ev = E.evaluate(D.AlignDataset("t", Path("m"), Path("v"), "", gt, "word", "en"), lines)
    o = ev["onset"]
    assert o["n"] == 4 and o["MedAE"] == pytest.approx(0.25)
    assert o["AAE"] == pytest.approx((0.1 + 0 + 0.4 + 3) / 4)
    assert o["PCO@0.3"] == pytest.approx(0.5) and o["PCO@1.0"] == pytest.approx(0.75)
    assert ev["line_start"]["n"] == 2 and ev["line_start"]["MedAE"] == pytest.approx(0.25)
    assert ev["review"] == {"n_lines": 2, "n_bad": 1, "n_flagged": 1, "n_caught": 1, "recall": 1.0,
                            "precision": 1.0}
    assert ev["offset"]["n"] == 4
    assert [p["worst_err"] for p in ev["per_line"]] == pytest.approx([0.1, 3.0])


def test_backing_tokens_reported_separately():
    lines = mk_pred_lines([[("a", 1.0, 1.2), ("oh", 1.5, 1.7), ("b", 2.0, 2.2)]])
    gt = [D.GTToken("a", 1.0, 0), D.GTToken("(oh)", 1.6, 0, backing=True), D.GTToken("b", 2.0, 0)]
    ev = E.evaluate(D.AlignDataset("t", Path("m"), Path("v"), "", gt, "word", "en"), lines)
    assert ev["onset_lead"]["n"] == 2 and ev["onset_lead"]["AAE"] == 0.0
    assert ev["onset_backing"]["AAE"] == pytest.approx(0.1)


def test_select_configs_runs_qwen_only_when_asked():
    def no_qwen(cfgs):
        return all(c.refine == "none" for c in cfgs)

    assert [c.name for c in E.select_configs("none", False)] == ["mms_gate"]
    abl = E.select_configs("none", True)
    assert no_qwen(abl) and "mms_nogate" in {c.name for c in abl}          # --ablations alone: no Qwen
    assert no_qwen(E.select_configs("auto", True, qwen_available=lambda: False))
    auto = E.select_configs("auto", True, qwen_available=lambda: True)
    assert {"hybrid", "qwen_windows_all"} <= {c.name for c in auto}
    qwen = E.select_configs("qwen", True, qwen_available=lambda: False)     # explicit: may download
    assert [c.name for c in qwen][:2] == ["mms_gate", "hybrid"] and len(qwen) == len(E.ABLATIONS) + 1


def test_emission_key_tracks_file_window_and_code_version(tmp_path: Path, monkeypatch):
    f = tmp_path / "vocals.wav"
    f.write_bytes(b"x" * 10)
    k = E.emission_key(f, 30.0, 2.0)
    assert k == E.emission_key(f, 30.0, 2.0) and k != E.emission_key(f, 20.0, 2.0)
    monkeypatch.setattr(E, "EMISSION_VERSION", E.EMISSION_VERSION + 1)
    assert E.emission_key(f, 30.0, 2.0) != k
    monkeypatch.undo()
    f.write_bytes(b"y" * 11)
    assert E.emission_key(f, 30.0, 2.0) != k


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(D.LOADERS))
def test_real_datasets_load(name):
    try:
        ds = D.load_dataset(name)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    assert ds.gt and ds.lyrics.strip() and ds.granularity in ("word", "syllable", "line")
    ts = [g.t for g in ds.gt]
    assert ts == sorted(ts), "GT onsets must be in sung order"
    if ds.granularity == "line":
        assert len(ds.gt) == len([x for x in ds.lyrics.splitlines() if x.strip()])
