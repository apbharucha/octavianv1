import re

# 1. ticker_universe caching
src = open('ticker_universe.py', encoding='utf-8').read()
lines = src.split('\n')
print('=== ticker_universe: get_ticker_universe + caching ===')
for i, l in enumerate(lines):
    if 'def get_ticker_universe' in l or 'cache' in l.lower() or '_UNIVERSE' in l:
        print(f'{i+1}: {l.strip()[:110]}')

# 2. _generate_opportunity_charts in ai_chatbot
src2 = open('ai_chatbot.py', encoding='utf-8').read()
lines2 = src2.split('\n')
print('\n=== ai_chatbot _generate_opportunity_charts (my earlier fix) ===')
for i, l in enumerate(lines2):
    if 'def _generate_opportunity_charts' in l:
        print(f'defined at line {i+1}')
        for j in range(i, min(i+45, len(lines2))):
            print(f'{j+1}: {lines2[j][:110]}')
        break
else:
    print('NOT FOUND')
