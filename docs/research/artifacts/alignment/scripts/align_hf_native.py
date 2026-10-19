"""Global CTC alignment with a native-script HF CTC model (e.g. kresnik/wav2vec2-large-xlsr-korean whose
vocab = 1205 Hangul syllables, or reazon-research/japanese-wav2vec2-large-rs35kh = kana+kanji).

Garbage/star column = max non-blank log-prob per frame minus a penalty (WhisperX-style wildcard),
used at edges and between lines. Unknown characters also map to the wildcard.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torchaudio.functional as F
from transformers import AutoModelForCTC, AutoProcessor

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from align_mms import load_mono16k  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


@torch.inference_mode()
def emissions(model, wav, win_s=30.0, ctx_s=2.0, sr=16000, hop=320):
    win, ctx, outs = int(win_s * sr), int(ctx_s * sr), []
    for start in range(0, len(wav), win):
        a, b = max(0, start - ctx), min(len(wav), start + win + ctx)
        x = wav[a:b]
        x = (x - x.mean()) / (x.std() + 1e-7)  # Wav2Vec2FeatureExtractor do_normalize
        lg = model(x.unsqueeze(0).to(DEV)).logits[0].float().cpu()
        lead = (start - a) // hop
        outs.append(lg[lead: lead + min(win, len(wav) - start) // hop])
    return torch.cat(outs).log_softmax(-1)


def align(audio, lines, model_id, star_penalty=1.0):
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id).to(DEV).eval()
    vocab = {k: v for k, v in proc.tokenizer.get_vocab().items()}
    blank = proc.tokenizer.pad_token_id
    wav = load_mono16k(audio)
    t1 = time.time()
    em = emissions(model, wav)
    nb = torch.ones(em.shape[1], dtype=torch.bool); nb[blank] = False
    star_col = em[:, nb].max(1).values - star_penalty
    em = torch.cat([em, star_col[:, None]], 1)
    star = em.shape[1] - 1
    toks, owner = [star], [(-1, -1)]
    for li, ln in enumerate(lines):
        chars = [c for c in ln if not c.isspace() and c not in ",.!?、。"]
        for ci, c in enumerate(chars):
            toks.append(vocab.get(c, star)); owner.append((li, ci))
        toks.append(star); owner.append((-1, -1))
    # CTC needs blank between identical consecutive labels; star-star repeats are fine since merge splits them
    path, sc = F.forced_align(em.unsqueeze(0), torch.tensor([toks], dtype=torch.int32), blank=blank)
    spans = F.merge_tokens(path[0], sc[0].exp(), blank=blank)
    spf = 320 / 16000
    out = []
    for li, ln in enumerate(lines):
        ss = [(s, o) for s, o in zip(spans, owner) if o[0] == li]
        out.append(dict(line=ln, start=round(ss[0][0].start * spf, 3), conf=round(float(np.mean([s.score for s, _ in ss])), 3),
                        units=[(ln_c, round(s.start * spf, 2)) for (s, _), ln_c in zip(ss, [c for c in ln if not c.isspace()])]))
    return out, dict(total_s=round(time.time() - t0, 1), align_s=round(time.time() - t1, 2), frames=em.shape[0])
