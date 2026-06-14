import codecs
with open('demo.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('\u2014', '-').replace('\u2192', '->').replace('\u2208', 'in')

with open('demo.py', 'w', encoding='utf-8') as f:
    f.write(text)
