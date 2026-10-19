import regex as re, unicodedata as u
for c in "ー〜・々〆ヶ":
    print(hex(ord(c)), u.name(c), 'Katakana' if re.match(r'\p{Katakana}',c) else '-', 'Hira' if re.match(r'\p{Hiragana}',c) else '-', 'Han' if re.match(r'\p{Han}',c) else '-','Common' if re.match(r'\p{Script=Common}',c) else '-', 'scxKana' if re.match(r'\p{scx=Katakana}',c) else '-')
for s in ["ｶﾞ","㌔","ㅋㅋ","ㅠ","～","〜","１２３","Ｒ＆Ｂ"]:
    n=u.normalize("NFKC",s); print(repr(s),len(s),'->',repr(n),len(n),[hex(ord(c)) for c in n])
print(re.__version__)
