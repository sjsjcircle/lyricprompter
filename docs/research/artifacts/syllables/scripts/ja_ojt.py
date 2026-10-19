import time
t=time.time()
import pyopenjtalk
print(pyopenjtalk.__version__ if hasattr(pyopenjtalk,'__version__') else '')
s = "東京タワーで2人きり、行こうよ！ラブソング 3月14日 100%"
print(pyopenjtalk.g2p(s, kana=True)); print("t", round(time.time()-t,2))
print(pyopenjtalk.g2p(s))
for n in pyopenjtalk.run_frontend(s):
    print(n['string'], n['read'], n['pron'], n['mora_size'], n['pos'])
