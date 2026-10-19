import pyopenjtalk
for t in ["1000年先も Forever","3月14日に会おう","ｶﾞｯｺｳの帰り道、走り出す！","東京の空〜あ〜あ","R&Bを聴いて100%"]:
    m=pyopenjtalk.g2p_mapping(t)
    print(t, [(x['surface'], x['char_span'], t[x['char_span'][0]:x['char_span'][1]], x['read'], x['pron'], x['is_ignored']) for x in m])
r=pyopenjtalk.run_mecab_nbest_features("明日の今日は何を", 4)
for p in r: print([ (mm['surface'], mm['features'][8] if len(mm['features'])>8 else None) for mm in p['morphs']])
