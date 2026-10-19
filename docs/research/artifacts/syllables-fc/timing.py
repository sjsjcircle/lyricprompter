import time
t=time.time(); import pyopenjtalk; t1=time.time(); pyopenjtalk.run_frontend("明日"); t2=time.time()
print(f"import pyopenjtalk {t1-t:.2f}s first run_frontend {t2-t1:.2f}s")
t=time.time(); pyopenjtalk.run_frontend("東京タワーで2人きり"); print(f"warm {1000*(time.time()-t):.1f}ms")
t=time.time(); import g2pk2; g=g2pk2.G2p(); t1=time.time(); g("꽃잎을"); print(f"g2pk2 init {t1-t:.2f}s first call {time.time()-t1:.2f}s")
t=time.time(); g("같이"); print(f"g2pk2 warm {1000*(time.time()-t):.1f}ms")
t=time.time(); import cmudict; c=cmudict.dict(); print(f"cmudict {time.time()-t:.2f}s")
t=time.time(); import uroman; u=uroman.Uroman(); print(f"uroman init {time.time()-t:.2f}s")
