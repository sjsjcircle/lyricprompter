import pyopenjtalk, inspect
print([x for x in dir(pyopenjtalk) if not x.startswith('_')])
for f in ['g2p_mapping','run_mecab_nbest_features','run_frontend','update_global_jtalk_with_user_dict','mecab_dict_index']:
    if hasattr(pyopenjtalk,f): print(f, inspect.signature(getattr(pyopenjtalk,f)))
if hasattr(pyopenjtalk,'g2p_mapping'):
    try: print(pyopenjtalk.g2p_mapping("東京タワーで2人きり"))
    except Exception as e: print('g2p_mapping err', e)
if hasattr(pyopenjtalk,'run_mecab_nbest_features'):
    try:
        r=pyopenjtalk.run_mecab_nbest_features("明日も宇宙へ", 3)
        print(type(r), str(r)[:800])
    except Exception as e: print('nbest err', e)
