"""Real-audio alignment (MMS_FA on MPS, optional Qwen3-FA): needs the cached models and data/stems."""
from __future__ import annotations

import numpy as np
import pytest

from lyricprompter.eval import align_eval as E
from lyricprompter.eval import datasets as D

pytestmark = pytest.mark.slow


def _load(name: str) -> D.AlignDataset:
    try:
        return D.load_dataset(name)
    except FileNotFoundError as e:
        pytest.skip(str(e))


def _check_contract(lines) -> None:
    prev = 0.0
    for ln in lines:
        if not ln.words:
            continue
        assert ln.start is not None and ln.end is not None and ln.start <= ln.end
        for syl in ln.syllables():
            assert prev <= syl.start <= syl.end
            prev = syl.end
            for u in syl.units:
                assert syl.start <= u.start <= u.end <= syl.end


def test_edelweiss_short_word_onsets():
    from lyricprompter.audio_io import load_mono
    from lyricprompter.prep.align import align_lines, release_models
    from lyricprompter.text import parse_lyrics

    ds = _load("edelweiss_short")
    lines = parse_lyrics(ds.lyrics, ds.default_lang)
    report = align_lines(lines, load_mono(ds.vocals, 16000), 16000, refine="none")
    release_models()
    _check_contract(lines)
    ev = E.evaluate(ds, lines)
    assert ev["n_matched"] == len(ds.gt)
    assert ev["onset"]["MedAE"] < 0.3
    assert ev["onset"]["PCO@1.0"] == 1.0
    assert report.method == "mms_fa+gate" and len(report.line_confidence) == len(lines)


def test_hybrid_refine_fixes_tts_long():
    from lyricprompter.audio_io import load_mono
    from lyricprompter.prep import refine as R
    from lyricprompter.prep.align import align_lines, release_models
    from lyricprompter.text import parse_lyrics

    if not R.qwen_cached():
        pytest.skip("Qwen3-ForcedAligner not cached")
    ds = _load("tts_multilingual_long")
    y = load_mono(ds.vocals, 16000)
    base = parse_lyrics(ds.lyrics, ds.default_lang)
    align_lines(base, y, refine="none")
    hyb = parse_lyrics(ds.lyrics, ds.default_lang)
    report = align_lines(hyb, y, refine="auto")
    release_models()
    _check_contract(hyb)
    assert report.method == "mms_fa+gate+qwen" and report.details["refine"]["n_lines"] > 0
    e_base, e_hyb = E.evaluate(ds, base)["onset"], E.evaluate(ds, hyb)["onset"]
    assert e_hyb["AAE"] < e_base["AAE"]
    assert e_hyb["PCO@1.0"] >= e_base["PCO@1.0"]
    assert np.isfinite(report.timings["refine"])


def test_hybrid_after_long_intro_keeps_first_line_on_the_vocals():
    """A low-confidence first line 43 s into the song: its refine window must contain it, and Qwen output
    collapsed onto the window edge must not move it into the intro."""
    from lyricprompter.audio_io import load_mono
    from lyricprompter.prep import refine as R
    from lyricprompter.prep.align import align_lines, release_models
    from lyricprompter.text import parse_lyrics

    if not R.qwen_cached():
        pytest.skip("Qwen3-ForcedAligner not cached")
    ds = _load("edelweiss_short")
    y = load_mono(ds.vocals, 16000)
    first = min(g.t for g in ds.gt)
    pre = y[: int((first - 0.5) * 16000)]                    # the stem's own pre-vocal bleed
    intro = np.tile(pre, int(np.ceil(36.0 * 16000 / len(pre))))[: 36 * 16000]
    lines = parse_lyrics(ds.lyrics, ds.default_lang)
    report = align_lines(lines, np.concatenate([intro, y]), refine="auto")
    release_models()
    _check_contract(lines)
    assert abs(lines[0].start - (first + 36.0)) < 1.0 and lines[0].end - lines[0].start > 2.0
    for li, lr in report.details.get("refine", {}).get("lines", {}).items():
        w0, w1 = lr["window"]
        assert lr["reason"] != "adopted" or w0 <= lines[int(li)].start <= w1
