import re

src = open('trader_profile.py', encoding='utf-8').read()
lines = src.split('\n')
print('=== trader_profile top imports ===')
for i, l in enumerate(lines[:60]):
    if 'import' in l:
        print(f'{i+1}: {l.strip()[:110]}')

src2 = open('data_sources.py', encoding='utf-8').read()
lines2 = src2.split('\n')
print('\n=== data_sources get_stock + get_fresh_quote ===')
for i, l in enumerate(lines2):
    if 'def get_stock' in l or 'def get_fresh_quote' in l or 'def get_realtime_price' in l:
        print(f'{i+1}: {l.strip()[:110]}')
