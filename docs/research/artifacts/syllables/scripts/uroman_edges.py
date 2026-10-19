import uroman as ur
u = ur.Uroman()
for s in ["ちょっとまって", "らーめん", "しんゆう", "Every morning", "꽃잎을", "ティーンエイジャー", "2人", "Ｒ＆Ｂ"]:
    e = u.romanize_string(s, rom_format=ur.RomFormat.EDGES)
    print(s, [(x.start, x.end, x.txt, x.type) for x in e] if hasattr(e[0],'start') else e)
