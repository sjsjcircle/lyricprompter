import time, pyphen, cmudict, re
words = "every morning I wake up wanna fire beautiful yesterday heaven baby tonight everything somebody hold forever ooh yeah oh-oh rhythm hours lyrics serenade gonna dreamin' you're I'll Seoul Tokyo".split()
d = pyphen.Pyphen(lang='en_US')
d1 = pyphen.Pyphen(lang='en_US', left=1, right=1)
t=time.time(); cd = cmudict.dict(); print("cmudict load", round(time.time()-t,2), len(cd))
def ncmu(w):
    p = cd.get(w.lower())
    return [sum(ph[-1].isdigit() for ph in pr) for pr in p] if p else None
for w in words:
    print(f"{w:12s} pyphen={d.inserted(w):16s} p(1,1)={d1.inserted(w):16s} cmu={ncmu(w)}")
print(pyphen.LANGUAGES.keys().__len__(), [k for k in pyphen.LANGUAGES if k.startswith('en')])
