import time
samples = ["꽃잎을 날려", "같이 걸어갈까", "사랑해 baby 너를 원해", "좋아해 읽고 싶어", "2번째 봄, 우리 함께 할래"]
t=time.time()
from g2pk2 import G2p
g2p = G2p()
print("g2pk2 load", round(time.time()-t,2))
for s in samples:
    t=time.time(); out = g2p(s); dt=time.time()-t
    out2 = g2p(s, descriptive=True)
    print("g2pk2:", s, "->", out, "| desc:", out2, f"{dt*1000:.0f}ms")
from korean_romanizer.romanizer import Romanizer
for s in samples: print("korean-romanizer:", Romanizer(s).romanize())
from hangul_romanize import Transliter
from hangul_romanize.rule import academic
tr = Transliter(academic)
for s in samples: print("hangul-romanize(academic):", tr.translit(s))
import uroman as ur
u = ur.Uroman()
for s in samples: print("uroman:", u.romanize_string(s))
for s in samples: print("uroman(g2pk2):", u.romanize_string(g2p(s)))
