import cutlet, inspect
print(inspect.signature(cutlet.Cutlet.__init__))
k=cutlet.Cutlet(); print(k.use_foreign_spelling, k.romaji("東京タワーでラブソング"))
k.use_foreign_spelling=False; print(k.romaji("東京タワーでラブソング"))
from num2words import num2words
print([num2words(n, lang='ko') for n in (2,21,24,100,1000)])
print(num2words(21, lang='en'), num2words(2026, lang='en', to='year'))
import kiwipiepy; from kiwipiepy import Kiwi
kw=Kiwi(); print([ (t.form,t.tag) for t in kw.tokenize("꽃잎을 날려 baby 21살")])
print([x for x in dir(kw) if 'join' in x or 'space' in x or 'pron' in x.lower()])
