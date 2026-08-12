import re

src = open('financial_llm_engine.py', encoding='utf-8').read()
lines = src.split('\n')

print('=== lines 30-75 (_call_llm region) ===')
for i in range(29, 75):
    print(f'{i+1}: {lines[i]}')

print('\n=== lines 370-416 (_fetch_live_data_for_tickers region) ===')
for i in range(369, 416):
    print(f'{i+1}: {lines[i]}')
