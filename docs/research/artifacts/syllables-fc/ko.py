import time, g2pk2
g=g2pk2.G2p()
for s in ["꽃잎을","같이","좋아해","읽고","24시간","2명의","10년","baby","R&B","꽃잎을 날려","날려","신라","국물","맑은 하늘","않아","좋다","밟고","닫히다","넓은","옷 입어","꽃 위에"]:
    t=time.time(); p=g(s); dt=(time.time()-t)*1000
    print(s,'->',p,f"{dt:.0f}ms")
import inspect; print(inspect.signature(g.__call__))
