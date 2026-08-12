import sys, os, re
sys.path.insert(0, os.path.abspath('.'))
lines = open('ai_chatbot.py', encoding='utf-8').read().split('\n')
# print lines 528-540 (start of dead method) and find next def after 531
print('=== around 528-545 ===')
for i in range(528, 545):
    print(f'{i}: {lines[i-1]}')
print()
print('=== next def after 531 ===')
for i in range(532, len(lines)):
    s = lines[i-1].strip()
    if s.startswith('def ') or s.startswith('async def '):
        print(f'{i}: {s[:110]}')
        break
