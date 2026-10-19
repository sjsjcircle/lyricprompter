import time, fugashi, pyopenjtalk, jaconv
from sudachipy import Dictionary, SplitMode
tag = fugashi.Tagger()
t=time.time(); sud = Dictionary(dict="core").create(); print("sudachi load", round(time.time()-t,2))
lines = ["明日また会えるかな", "一人きりの夜に何を思う", "今日の日はさようなら", "君の心に触れたくて", "何でもないや", "見上げた空に星が降る", "生まれ変わっても", "風が吹いている", "微笑んで", "未来へ向かって走り出せ", "行こうよ", "大人になった", "抱きしめたい", "2人の思い出", "1000年先も"]
for s in lines:
    f = "".join((w.feature.kana or w.surface) for w in tag(s))
    fp = "".join((w.feature.pron or w.surface) for w in tag(s))
    o = pyopenjtalk.g2p(s, kana=True)
    su = "".join(m.reading_form() for m in sud.tokenize(s, SplitMode.C))
    print(f"{s}\n  unidic-lite kana: {f}\n  unidic-lite pron: {fp}\n  openjtalk+:       {o}\n  sudachi:          {su}")
