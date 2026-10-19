import pyopenjtalk, fugashi, pykakasi
from sudachipy import Dictionary, SplitMode
tests=["明日また会える","何をしても","走り出せ","2人きり","1000年先も","3月14日","今日の日","東京タワーで","本気で好き","未来へ","地球を回せ","一人きり","大人になる","今日は晴れ"]
fr=pyopenjtalk.run_frontend("明日また会える")
print(fr[0].keys() if isinstance(fr[0],dict) else type(fr[0]))
print([(n['string'],n['read'],n['pron'],n['mora_size']) for n in fr])
tg=fugashi.Tagger()
try:
    sd=Dictionary().create()
except Exception as e: print('sudachi create err',e); sd=None
try:
    sdt=Dictionary().tokenizer()
except Exception as e: print('no Dictionary.tokenizer():',e); sdt=None
kk=pykakasi.kakasi()
for t in tests:
    ojt="".join(n['read'] for n in pyopenjtalk.run_frontend(t))
    ojp="".join(n['pron'] for n in pyopenjtalk.run_frontend(t)).replace("’","")
    ud="".join((w.feature.kana or w.surface) if hasattr(w.feature,'kana') else w.surface for w in tg(t))
    su="".join(m.reading_form() for m in sd.tokenize(t, SplitMode.C)) if sd else '-'
    pk="".join(x['hira'] for x in kk.convert(t))
    print(f"{t}\tOJT={ojt}\tpron={ojp}\tunidic={ud}\tsudachi={su}\tkakasi={pk}")
print(pyopenjtalk.run_frontend("2人")[0]['string'])
