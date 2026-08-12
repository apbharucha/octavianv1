import re

src = open('ai_chatbot.py', encoding='utf-8').read()
lines = src.split('\n')

print('=== _create_advanced_price_chart signature ===')
for i, l in enumerate(lines):
    if 'def _create_advanced_price_chart' in l:
        print(f'{i+1}: {l.strip()[:130]}')
        for j in range(i, min(i+8, len(lines))):
            print(f'    {j+1}: {lines[j].strip()[:100]}')
        break

print('\n=== data_sources / get_stock imports in ai_chatbot ===')
for i, l in enumerate(lines):
    if 'from data_sources import' in l or 'import data_sources' in l or 'get_stock' in l:
        print(f'{i+1}: {l.strip()[:130]}')
