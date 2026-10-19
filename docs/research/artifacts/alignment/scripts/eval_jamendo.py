"""Word-onset evaluation on a few JamendoLyrics (English) songs: MMS_FA global vs Qwen3-FA (whole / windowed),
on full mix vs htdemucs vocal stem. Metrics: AAE (mean abs onset error, s), median, PCO@0.3s (fraction within 0.3 s).
"""
from __future__ import annotations

import csv
import re
import sys
import time

import numpy as np
import torch
import torchaudio.functional as F
from torchaudio.pipelines import MMS_FA as B

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import align_mms as A  # noqa: E402

ROOT = sys.argv[1]
SONGS = sys.argv[2].split(",")
DEV = A.DEV
_mms = None


def words_of(lines):
    out = []
    for li, ln in enumerate(lines):
        for w in ln.split():
            out.append((li, w))
    return out


def gt_onsets(song):
    with open(f"{ROOT}/words/{song}.csv") as f:
        return np.array([float(r["word_start"]) for r in csv.DictReader(f)])


def mms_global(wav, lines, star_between_lines=True):
    global _mms
    if _mms is None:
        _mms = B.get_model(with_star=True).to(DEV)
    em, spf = A.emissions(_mms, wav)
    d = B.get_dict()
    star = d["*"]
    toks, owner = [star], [-1]
    W = words_of(lines)
    prev_li = 0
    for wi, (li, w) in enumerate(W):
        if star_between_lines and li != prev_li:
            toks.append(star); owner.append(-1)
        prev_li = li
        for ch in re.sub(r"[^a-z']", "", w.lower()):
            toks.append(d[ch]); owner.append(wi)
    toks.append(star); owner.append(-1)
    path, sc = F.forced_align(em.unsqueeze(0), torch.tensor([toks], dtype=torch.int32), blank=0)
    spans = F.merge_tokens(path[0], sc[0].exp())
    on = np.full(len(W), np.nan); score = np.zeros(len(W))
    for wi in range(len(W)):
        ss = [s for s, o in zip(spans, owner) if o == wi]
        if ss:
            on[wi] = ss[0].start * spf; score[wi] = np.mean([s.score for s in ss])
    return on, score


_q = None


def qwen(wav_np, lines, t0=0.0):
    global _q
    from transformers import AutoModelForTokenClassification, AutoProcessor
    if _q is None:
        mid = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
        _q = (AutoProcessor.from_pretrained(mid),
              AutoModelForTokenClassification.from_pretrained(mid, dtype=torch.float32).to(DEV).eval())
    proc, model = _q
    W = [w for _, w in words_of(lines)]
    clean = [re.sub(r"[^A-Za-z0-9']", "", w) or "x" for w in W]
    proc.split_words_for_alignment = lambda text, language=None: list(clean)
    inputs, wl = proc.prepare_forced_aligner_inputs(audio=wav_np, transcript=" ".join(clean), language="English")
    inputs = inputs.to(DEV)
    with torch.inference_mode():
        out = model(**inputs)
    ts = proc.decode_forced_alignment(logits=out.logits, input_ids=inputs["input_ids"], word_lists=wl,
                                      timestamp_token_id=model.config.timestamp_token_id)[0]
    return np.array([t["start_time"] + t0 for t in ts])


def qwen_windowed(wav, lines, line_starts, line_ends, max_win=30.0, pad=1.0):
    """Coarse-to-fine: group lines into <=max_win windows using coarse (MMS) line times; run Qwen per window."""
    W = words_of(lines)
    res = np.full(len(W), np.nan)
    i = 0
    n = len(lines)
    while i < n:
        j = i
        while j + 1 < n and line_ends[j + 1] - line_starts[i] < max_win:
            j += 1
        a = max(0.0, line_starts[i] - pad)
        b = min(len(wav) / 16000, line_ends[j] + pad)
        seg = wav[int(a * 16000): int(b * 16000)].numpy()
        on = qwen(seg, lines[i: j + 1], t0=a)
        idx = [k for k, (li, _) in enumerate(W) if i <= li <= j]
        res[idx] = on[: len(idx)]
        i = j + 1
    return res


def metrics(pred, gt):
    m = min(len(pred), len(gt))
    e = np.abs(pred[:m] - gt[:m])
    e = e[~np.isnan(e)]
    return dict(AAE=round(float(e.mean()), 3), med=round(float(np.median(e)), 3), PCO03=round(float((e < 0.3).mean()), 3),
                n=int(m))


if __name__ == "__main__":
    for song in SONGS:
        lines = [l.strip() for l in open(f"{ROOT}/lyrics/{song}.txt", encoding="utf-8") if l.strip()]
        gt = gt_onsets(song)
        nW = len(words_of(lines))
        print(f"\n== {song}: {len(lines)} lines, {nW} words (gt {len(gt)})")
        mix = A.load_mono16k(f"{ROOT}/mp3/{song}.mp3") if False else None
        import subprocess, io, soundfile as sf  # decode mp3 with ffmpeg
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", f"{ROOT}/mp3/{song}.mp3", "-ac", "1", "-ar", "16000", "-f", "wav", "-"],
                             capture_output=True, check=True).stdout
        mix = torch.from_numpy(sf.read(io.BytesIO(raw), dtype="float32")[0])
        voc = A.load_mono16k(f"{ROOT}/sep/htdemucs/{song}/vocals.wav")
        for name, wav in [("mix", mix), ("vocals", voc)]:
            t = time.time(); on, sc = mms_global(wav, lines); dt = time.time() - t
            print(f"  MMS_FA global  {name:6s}", metrics(on, gt), f"{dt:.1f}s")
        # coarse line times from MMS on vocals
        on, sc = mms_global(voc, lines)
        W = words_of(lines)
        ls = [np.nanmin([on[k] for k, (li, _) in enumerate(W) if li == L]) for L in range(len(lines))]
        le = [np.nanmax([on[k] for k, (li, _) in enumerate(W) if li == L]) + 0.8 for L in range(len(lines))]
        t = time.time(); qw = qwen_windowed(voc, lines, ls, le); dt = time.time() - t
        print(f"  Qwen3FA win30  vocals", metrics(qw, gt), f"{dt:.1f}s")
        if len(voc) / 16000 <= 300:
            t = time.time(); qa = qwen(voc.numpy(), lines); dt = time.time() - t
            print(f"  Qwen3FA whole  vocals", metrics(qa, gt), f"{dt:.1f}s")
        # line-level confidence vs error (MMS vocals)
        err = np.abs(on[: len(gt)] - gt[: len(on)])
        line_conf = [np.mean([sc[k] for k, (li, _) in enumerate(W) if li == L]) for L in range(len(lines))]
        line_err = [np.nanmax([err[k] for k, (li, _) in enumerate(W) if li == L and k < len(err)] or [0]) for L in range(len(lines))]
        bad = np.array(line_err) > 1.0
        flag = np.array(line_conf) < 0.10
        print(f"  lines with max word err >1s: {bad.sum()}/{len(lines)}; flagged(conf<0.10): {flag.sum()}; caught {int((bad & flag).sum())}")
