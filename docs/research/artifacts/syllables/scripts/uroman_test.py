import uroman as ur, inspect
u = ur.Uroman()
print([m for m in dir(ur) if not m.startswith('_')][:60])
print(inspect.signature(u.romanize_string))
print(list(ur.RomFormat))
for s in ["꽃잎을 날려", "とうきょう", "がっこう", "ちょっとまって", "らーめん", "しんゆう", "ヴァイオリン", "東京", "Hello, world!", "ティーンエイジャー", "きゃりーぱみゅぱみゅ"]:
    print(s, "->", u.romanize_string(s), "| ja:", u.romanize_string(s, lcode='jpn'))
e = u.romanize_string("꽃잎을 날려 とうきょう", rom_format=ur.RomFormat.EDGES)
print(e[:12])
# per-syllable vs whole
s="같이걸어갈까"
print("".join(u.romanize_string(c) for c in s), u.romanize_string(s))
s="ちょっとまって"
print([u.romanize_string(c) for c in s], u.romanize_string(s))
