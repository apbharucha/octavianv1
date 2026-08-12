import sys, os, re
sys.path.insert(0, os.path.abspath('.'))
src = open('market_scanner.py', encoding='utf-8').read()
lines = src.split('\n')
for i, l in enumerate(lines, 1):
    if 'universe' in l.lower() or 'get_ticker' in l or 'TickerUniverse' in l:
        print(f'{i}: {l.strip()[:130]}')
