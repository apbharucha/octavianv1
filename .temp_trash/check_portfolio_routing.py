import sys, os
sys.path.insert(0, os.path.abspath('.'))
src = open('ai_chatbot.py', encoding='utf-8').read()
lines = src.split('\n')

for i, l in enumerate(lines, 1):
    if 'def process_query' in l or 'def process_unbiased_query' in l:
        print(f'>>> {i}: {l.strip()}')
        for j in range(i, min(i+50, len(lines))):
            print(f'{j}: {lines[j-1]}')
        print('='*60)
        if i > 3300:
            break
