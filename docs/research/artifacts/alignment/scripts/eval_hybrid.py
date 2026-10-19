import sys; sys.path.insert(0,'scripts')
sys.argv=['x','jamendo','']
import re, numpy as np, torch, torchaudio.functional as F
import eval_jamendo as E, align_mms as A, eval_gate as G
from torchaudio.pipelines import MMS_FA as B

def mms_gated_scores(voc, lines):
    if E._mms is None: E._mms = B.get_model(with_star=True).to(A.DEV)
    em, spf = A.emissions(E._mms, voc); em = em.clone()
    act = G.vocal_gate(voc, em.shape[0]); em[~torch.from_numpy(act), 1:28] -= 15.0
    d = B.get_dict(); star = d['*']; toks=[star]; owner=[-1]; W=E.words_of(lines); prev=0
    for wi,(li,w) in enumerate(W):
        if li!=prev: toks.append(star); owner.append(-1)
        prev=li
        for ch in re.sub(r"[^a-z']","",w.lower()): toks.append(d[ch]); owner.append(wi)
    toks.append(star); owner.append(-1)
    path, sc = F.forced_align(em.unsqueeze(0), torch.tensor([toks],dtype=torch.int32), blank=0)
    spans = F.merge_tokens(path[0], sc[0].exp())
    on=np.full(len(W),np.nan); s=np.zeros(len(W))
    for wi in range(len(W)):
        ss=[x for x,o in zip(spans,owner) if o==wi]
        if ss: on[wi]=ss[0].start*spf; s[wi]=np.mean([x.score for x in ss])
    return on, s

for song in ['HILA_-_Give_Me_the_Same','Avercage_-_Embers','Rxbyn_-_Bad_Side','Cortez_-_Feel__Stripped_']:
    lines=[l.strip() for l in open(f'jamendo/lyrics/{song}.txt') if l.strip()]
    gt=E.gt_onsets(song); voc=A.load_mono16k(f'jamendo/sep/htdemucs/{song}/vocals.wav')
    W=E.words_of(lines); LI=np.array([li for li,_ in W])
    on, sc = mms_gated_scores(voc, lines)
    ls=[np.nanmin(on[LI==L]) for L in range(len(lines))]; le=[np.nanmax(on[LI==L])+0.8 for L in range(len(lines))]
    qw = E.qwen_windowed(voc, lines, ls, le)
    qa = E.qwen(voc.numpy(), lines) if len(voc)/16000<=300 else np.full(len(W),np.nan)
    lconf=np.array([sc[LI==L].mean() for L in range(len(lines))])
    # hybrid: per line, if MMS line conf >= 0.10 keep MMS; else take Qwen-whole for that line if its line start lies
    # within the gap between neighbouring trusted MMS lines, else Qwen-windowed
    hy=on.copy()
    trusted = lconf>=0.10
    for L in range(len(lines)):
        if trusted[L]: continue
        lo = max([le[k] for k in range(L) if trusted[k]] or [0]); hi = min([ls[k] for k in range(L+1,len(lines)) if trusted[k]] or [1e9])
        cand = qa[LI==L]
        if np.all((cand>=lo-0.5)&(cand<=hi+0.5)): hy[LI==L]=cand
        else: hy[LI==L]=qw[LI==L]
    agree = np.abs(on-qa)
    print(song[:20], 'MMSgate', E.metrics(on,gt), '| Qwin', E.metrics(qw,gt), '| Qwhole', E.metrics(qa,gt), '| hybrid', E.metrics(hy,gt))
    # agreement-based confidence: words where |MMS - Qwhole| > 0.5 s
    err=np.abs(on-gt[:len(on)]); dis=agree>0.5
    print('   disagree>0.5s words:', int(dis.sum()), 'of', len(on), '; of MMS errors>1s:', int((err>1).sum()), 'caught by disagreement:', int(((err>1)&dis).sum()))
