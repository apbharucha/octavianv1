import sys, os, re
sys.path.insert(0, os.path.abspath('.'))
src = open('portfolio_analyzer.py', encoding='utf-8').read()
lines = src.split('\n')

# find loaded_positions construction
for i, l in enumerate(lines, 1):
    if 'loaded_positions' in l and ('append' in l or '=' in l):
        print(f'{i}: {l.strip()[:120]}')

print('='*60)
# asset_type references
for i, l in enumerate(lines, 1):
    if 'asset_type' in l:
        print(f'{i}: {l.strip()[:120]}')
