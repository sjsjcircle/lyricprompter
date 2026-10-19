"""Qwen3-ForcedAligner-0.6B (transformers>=5.19 native) with custom syllable-level units.

The stock processor splits ja with nagisa words and ko with soynlp words; we override
`split_words_for_alignment` so every Hangul syllable / Japanese morpheme surface / English
word-piece gets its own <timestamp> pair. Output resolution is 80 ms (timestamp_segment_time).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoProcessor

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lyric_units import line_units  # noqa: E402
from align_mms import load_mono16k  # noqa: E402

MID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def align(audio_path: str, lines: list[str], language: str = "Korean", dtype=torch.float32):
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MID)
    model = AutoModelForTokenClassification.from_pretrained(MID, dtype=dtype).to(DEV).eval()
    t_load = time.time() - t0

    units = [(li, u) for li, ln in enumerate(lines) for u in line_units(ln)]
    unit_texts = [u.display for _, u in units]
    proc.split_words_for_alignment = lambda text, language=None: list(unit_texts)  # custom granularity

    wav = load_mono16k(audio_path).numpy()
    t1 = time.time()
    inputs, word_lists = proc.prepare_forced_aligner_inputs(audio=wav, transcript=" ".join(lines), language=language)
    inputs = inputs.to(DEV)
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(dtype)
    with torch.inference_mode():
        out = model(**inputs)
    probs = out.logits.float().softmax(-1)
    ts = proc.decode_forced_alignment(logits=out.logits, input_ids=inputs["input_ids"], word_lists=word_lists,
                                      timestamp_token_id=model.config.timestamp_token_id)[0]
    # confidence: max class prob at each <timestamp> slot (start/end) per unit
    mask = inputs["input_ids"][0] == model.config.timestamp_token_id
    pmax = probs[0][mask].max(-1).values.cpu().numpy()
    t_run = time.time() - t1
    res = []
    for i, ((li, u), item) in enumerate(zip(units, ts)):
        res.append(dict(line=li, text=u.display, start=item["start_time"], end=item["end_time"],
                        conf=round(float(np.mean(pmax[2 * i: 2 * i + 2])), 3)))
    return res, dict(load_s=round(t_load, 1), run_s=round(t_run, 2), device=DEV, n_units=len(units))


if __name__ == "__main__":
    lyr = [l.strip() for l in open(sys.argv[2], encoding="utf-8") if l.strip()]
    res, timing = align(sys.argv[1], lyr, language=sys.argv[3] if len(sys.argv) > 3 else "Korean")
    print(timing)
    cur = -1
    for r in res:
        if r["line"] != cur:
            cur = r["line"]; print(f"\nLINE {cur}: {lyr[cur]}")
        print(f"  {r['text']}@{r['start']}-{r['end']}({r['conf']})", end="")
    print()
