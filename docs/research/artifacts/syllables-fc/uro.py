import uroman as ur, regex as re, unicodedata, inspect
U=ur.Uroman()
print([x for x in dir(ur) if not x.startswith('_')][:40])
print(inspect.signature(U.romanize_string))
print(list(ur.RomFormat))
for s in ["꽃잎을","날려","너를","원해","같이","읽고","좋아해","사랑해","값","닭","앉아","없어","괜찮아","뷁","의사","희망","예쁜","얘기"]:
    whole=U.romanize_string(s); per="".join(U.romanize_string(c) for c in s)
    print(s, whole, per, "SAME" if whole==per else "DIFF")
print(U.romanize_string("東京タワー"), U.romanize_string("とうきょう"), U.romanize_string("トーキョー"))
e=U.romanize_string("ラーメン待って", rom_format=ur.RomFormat.EDGES)
print([(x.start,x.end,x.txt) for x in e])
e=U.romanize_string("꽃잎을", rom_format=ur.RomFormat.EDGES); print([(x.start,x.end,x.txt,getattr(x,'type',None)) for x in e])
# compare kana table
kana="あいうえおかきくけこがぎぐげごさしすせそざじずぜぞたちつてとだぢづでどなにぬねのはひふへほばびぶべぼぱぴぷぺぽまみむめもやゆよらりるれろわをんゔ"
outs=[(k,U.romanize_string(k)) for k in kana]
print(outs)
combos=["きゃ","しゃ","ちゃ","じゃ","にゃ","ふぁ","てぃ","でゅ","うぃ","ヴァ","ティ","ジェ","シェ","ツァ","ディ","デュ","フォ"]
print([(k,U.romanize_string(k)) for k in combos])
print(U.romanize_string("24시간 100%"), U.romanize_string("21살"))
