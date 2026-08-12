import re

src = open('financial_llm_engine.py', encoding='utf-8').read()
lines = src.split('\n')

print('=== _fetch_live_data_for_tickers + data fetching functions ===')
for i, l in enumerate(lines):
    if 'def _fetch' in l or '_fetch_live_data' in l or 'yf.Ticker' in l or '.info' in l:
        print(f'{i+1}: {l.strip()}')

print('\n=== _fetch_live_data_for_tickers body ===')
found = False
for i, l in enumerate(lines):
    if 'def _fetch_live_data_for_tickers' in l:
        found = True
        for j in range(i, min(i+60, len(lines))):
            print(f'{j+1}: {lines[j]}')
        break
if not found:
    print('NOT FOUND')

print('\n=== how info is fetched per ticker (look for loops) ===')
idx = None
for i, l in enumerate(lines):
    if 'def _fetch_live_data_for_tickers' in l:
        idx = i
        break
if idx is not None:
    for j in range(idx, min(idx+100, len(lines))):
        pass
