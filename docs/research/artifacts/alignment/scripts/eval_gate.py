import sys; sys.path.insert(0,'scripts')
sys.argv=['x','jamendo','']
import re, numpy as np, torch, torchaudio.functional as F
import eval_jamendo as E, align_mms as A
from torchaudio.pipelines import MMS_FA as B

def vocal_gate(voc, n_frames, hop=320, rel_db=-35.0, dilate_s=0.3):
    x = voc[: n_frames*hop].reshape(-1, hop) if len(voc) >= n_frames*hop else torch.nn.functional.pad(voc,(0,n_frames*hop-len(voc))).reshape(-1,hop)
    rms = x.pow(2).mean(1).sqrt().numpy()
    db = 20*np.log10(rms+1e-9); thr = np.percentile(db, 99) + rel_db
    active = db > thr
    k = int(dilate_s/0.02); act = np.convolve(active.astype(float), np.ones(2*k+1), 'same') > 0
    return act

def mms_gated(em_wav, voc, lines, gate=True, star_pen=0.0):
    if E._mms is None: E._mms = B.get_model(with_star=True).to(A.DEV)
    em, spf = A.emissions(E._mms, em_wav)
    em = em.clone()
    if gate:
        act = vocal_gate(voc, em.shape[0])
        em[~torch.from_numpy(act), 1:28] -= 15.0   # real tokens nearly impossible where vocal stem is silent
    em[:, 28] -= star_pen
    d = B.get_dict(); star = d['*']; toks=[star]; owner=[-1]; W=E.words_of(lines); prev=0
    for wi,(li,w) in enumerate(W):
        if li!=prev: toks.append(star); owner.append(-1)
        prev=li
        for ch in re.sub(r"[^a-z']","",w.lower()): toks.append(d[ch]); owner.append(wi)
    toks.append(star); owner.append(-1)
    path, sc = F.forced_align(em.unsqueeze(0), torch.tensor([toks],dtype=torch.int32), blank=0)
    spans = F.merge_tokens(path[0], sc[0].exp())
    on=np.full(len(W),np.nan)
    for wi in range(len(W)):
        ss=[s for s,o in zip(spans,owner) if o==wi]
        if ss: on[wi]=ss[0].start*spf
    return on

if __name__ == '__main__':
 for song in ['HILA_-_Give_Me_the_Same','Avercage_-_Embers','Rxbyn_-_Bad_Side','Cortez_-_Feel__Stripped_']:
     lines=[l.strip() for l in open(f'jamendo/lyrics/{song}.txt') if l.strip()]
     gt=E.gt_onsets(song); voc=A.load_mono16k(f'jamendo/sep/htdemucs/{song}/vocals.wav')
     for gate, pen in [(False,0.0),(True,0.0),(True,1.0),(True,3.0)]:
         print(song[:22], 'gate' if gate else 'nogate', 'star_pen',pen, E.metrics(mms_gated(voc, voc, lines, gate, pen), gt))
