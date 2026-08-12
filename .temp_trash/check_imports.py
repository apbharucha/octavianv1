import sys, os
sys.path.insert(0, os.path.abspath('.'))
lines = open('ai_chatbot.py', encoding='utf-8').read().split('\n')
for i in range(0, 77):
    s = lines[i].strip()
    if s.startswith('import ') or s.startswith('from '):
        print(f'{i+1}: {s}')
