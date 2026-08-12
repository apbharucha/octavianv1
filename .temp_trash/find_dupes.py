import sys, os, re
sys.path.insert(0, os.path.abspath('.'))
src = open('ai_chatbot.py', encoding='utf-8').read()
lines = src.split('\n')

methods = []
for i, l in enumerate(lines, 1):
    s = l.strip()
    if s.startswith('def '):
        name = re.match(r'def (\w+)', s).group(1)
        methods.append((name, i))

from collections import Counter
counts = Counter(name for name, _ in methods)
print('=== DUPLICATE METHOD DEFINITIONS ===')
for name, cnt in counts.items():
    if cnt > 1:
        locs = [i for n, i in methods if n == name]
        print(f'{name}: {cnt}x at lines {locs}')
