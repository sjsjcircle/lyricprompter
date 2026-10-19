import pyphen, cmudict, inspect
print(inspect.signature(pyphen.Pyphen.__init__))
h=pyphen.Pyphen(lang='en_US'); h1=pyphen.Pyphen(lang='en_US',left=1,right=1)
print('defaults left/right:', h.left, h.right)
C=cmudict.dict()
for w in ["tonight","somebody","lyrics","beautiful","everything","every","morning","remember","emotion","Tokyo","gonna","rhythm","ohoh","fire","hour","heaven","forever","believe","dancing","wanna","yeah","oh"]:
    p=C.get(w.lower())
    print(w, h.inserted(w), h1.inserted(w), [sum(x[-1].isdigit() for x in pr) for pr in p] if p else None)
print(pyphen.__version__ if hasattr(pyphen,'__version__') else '')
