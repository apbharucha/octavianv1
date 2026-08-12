src = open('financial_llm_engine.py', encoding='utf-8').read()
lines = src.split('\n')
print('=== imports ===')
for i, l in enumerate(lines[:35]):
    if 'import' in l:
        print(f'{i+1}: {l.strip()[:100]}')

print('\n=== _fetch_sector_etf_data body (cache save check) ===')
for i, l in enumerate(lines):
    if 'def _fetch_sector_etf_data' in l:
        for j in range(i, min(i+30, len(lines))):
            print(f'{j+1}: {lines[j][:100]}')
        break
