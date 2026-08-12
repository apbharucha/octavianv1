import sys, os
sys.path.insert(0, os.path.abspath('.'))
lines = open('ai_chatbot.py', encoding='utf-8').read().split('\n')
for i in range(3044, 3110):
    print(f'{i}: {lines[i-1]}')

print('='*60)
src = open('portfolio_chatbot_context.py', encoding='utf-8').read()
pl = src.split('\n')
for i, l in enumerate(pl, 1):
    s = l.strip()
    if s.startswith('def ') or s.startswith('class '):
        print(f'{i}: {s[:100]}')
