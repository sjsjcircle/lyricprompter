import time
t=time.time()
import fugashi
tagger = fugashi.Tagger()  # unidic-lite
print("fugashi load", round(time.time()-t,2))
s = "君の名は。夢を見ていた　東京タワーで2人きり、行こうよ！ラブソング"
for w in tagger(s):
    f = w.feature
    print(w.surface, f.kana, f.pron, f.pos1, getattr(f,'lemma',None), w.white_space)
import cutlet
k = cutlet.Cutlet()  # hepburn
print(k.romaji(s))
k2 = cutlet.Cutlet(system='kunrei'); 
print(k2.romaji("東京タワーで2人きり"))
k.use_foreign_spelling = False
print(k.romaji("ラブソング"))
