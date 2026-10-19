"""Global CTC forced alignment of a whole song's lyrics with torchaudio MMS_FA (torchaudio 2.11 / torch 2.14).

- Emissions computed in overlapping windows (model on MPS if available), Viterbi on CPU
  (torchaudio::forced_align has no MPS kernel).
- '*' star (garbage) token at song edges and between lines absorbs intros, instrumental gaps, ad-libs.
- Returns per-unit (syllable / mora-group) start/end + mean posterior score; per-line confidence.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from torchaudio.pipelines import MMS_FA as BUNDLE

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lyric_units import line_units  # noqa: E402

SR = BUNDLE.sample_rate  # 16000
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load_mono16k(path: str) -> torch.Tensor:
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(1))
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav


@torch.inference_mode()
def emissions(model, wav: torch.Tensor, win_s=30.0, ctx_s=2.0) -> tuple[torch.Tensor, float]:
    """Chunked emission (T, C) in log-domain with star column; returns (emission_cpu, sec_per_frame)."""
    hop = 320  # wav2vec2 stride at 16 kHz -> 20 ms
    win, ctx = int(win_s * SR), int(ctx_s * SR)
    outs = []
    for start in range(0, len(wav), win):
        a, b = max(0, start - ctx), min(len(wav), start + win + ctx)
        x = wav[a:b].unsqueeze(0).to(DEV)
        em, _ = model(x)
        em = em[0].float().cpu()
        lead = (start - a) // hop
        n = min(win, len(wav) - start) // hop
        outs.append(em[lead: lead + n])
    em = torch.cat(outs)
    return em, hop / SR


def align_song(audio_path: str, lyrics_lines: list[str]):
    t0 = time.time()
    model = BUNDLE.get_model(with_star=True).to(DEV)
    t_load = time.time() - t0
    wav = load_mono16k(audio_path)
    t1 = time.time()
    em, spf = emissions(model, wav)
    t_em = time.time() - t1

    d = BUNDLE.get_dict()  # includes '*'
    star = d["*"]
    tokens, owner = [star], [(-1, -1)]  # owner = (line_idx, unit_idx) per token
    lines = [line_units(ln) for ln in lyrics_lines]
    for li, units in enumerate(lines):
        for ui, u in enumerate(units):
            for ch in u.roman:
                if ch in d:
                    tokens.append(d[ch]); owner.append((li, ui))
        tokens.append(star); owner.append((-1, -1))

    t2 = time.time()
    path, scores = F.forced_align(em.unsqueeze(0), torch.tensor([tokens], dtype=torch.int32), blank=0)
    spans = F.merge_tokens(path[0], scores[0].exp())  # one TokenSpan per target token, in order
    t_al = time.time() - t2
    assert len(spans) == len(tokens)

    out = []
    for li, units in enumerate(lines):
        line_res = []
        for ui, u in enumerate(units):
            ss = [s for s, o in zip(spans, owner) if o == (li, ui)]
            if not ss:
                line_res.append(dict(text=u.display, start=None, end=None, score=0.0)); continue
            line_res.append(dict(text=u.display, start=round(ss[0].start * spf, 3), end=round(ss[-1].end * spf, 3),
                                 score=round(float(np.mean([s.score for s in ss])), 3)))
        conf = float(np.mean([r["score"] for r in line_res])) if line_res else 0.0
        out.append(dict(line=lyrics_lines[li], start=line_res[0]["start"], end=line_res[-1]["end"],
                        confidence=round(conf, 3), units=line_res))
    timing = dict(load_s=round(t_load, 2), emission_s=round(t_em, 2), viterbi_s=round(t_al, 3),
                  audio_s=round(len(wav) / SR, 2), frames=em.shape[0], device=DEV)
    return out, timing


if __name__ == "__main__":
    audio = sys.argv[1]
    lyr = [l.strip() for l in open(sys.argv[2], encoding="utf-8") if l.strip()]
    res, timing = align_song(audio, lyr)
    print(timing)
    for r in res:
        print(f"[{r['start']}-{r['end']}] conf={r['confidence']} {r['line']}")
        print("    ", " ".join(f"{u['text']}@{u['start']}({u['score']})" for u in r["units"]))
