from uroman import Uroman
u = Uroman()
for s, l in [("사랑해 너를 보고 싶어", "kor"), ("君の名前を呼んだ 夜空に響く", "jpn"), ("Hello 世界", None), ("빛나는 별처럼 Shine on", "kor")]:
    print("uroman", l, repr(s), "->", u.romanize_string(s, lcode=l))
import pykakasi
k = pykakasi.kakasi()
print("pykakasi", [(x['orig'], x['hira'], x['hepburn']) for x in k.convert("君の名前を呼んだ 夜空に響く")])
import cutlet
c = cutlet.Cutlet()
print("cutlet", c.romaji("君の名前を呼んだ 夜空に響く"))
import pyopenjtalk
print("openjtalk g2p kana:", pyopenjtalk.g2p("君の名前を呼んだ 夜空に響く", kana=True))
print("openjtalk g2p phon:", pyopenjtalk.g2p("君の名前を呼んだ 夜空に響く"))
from fugashi import Tagger
t = Tagger()
print("fugashi", [(w.surface, w.feature.kana, w.feature.pron) for w in t("君の名前を呼んだ 夜空に響く")])
from g2pk2 import G2p
g = G2p()
print("g2pk2", g("빛나는 별처럼 같이 읽어요"))
