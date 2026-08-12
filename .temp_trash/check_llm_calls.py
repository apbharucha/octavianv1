import re

src = open('financial_llm_engine.py', encoding='utf-8').read()
lines = src.split('\n')

print('=== all _call_llm call sites with context ===')
for i, l in enumerate(lines):
    if '_call_llm(' in l:
        print(f'line {i+1}: {l.strip()[:100]}')

print('\n=== check_llm_connectivity usage ===')
for i, l in enumerate(lines):
    if 'check_llm_connectivity' in l:
        print(f'line {i+1}: {l.strip()[:100]}')

print('\n=== function list ===')
for i, l in enumerate(lines):
    m = re.match(r'^def (\w+)', l.strip())
    if m:
        print(f'{i+1}: {m.group(1)}')
