import sys, os
sys.path.insert(0, os.path.abspath('.'))
lines = open('paper_trading_system.py', encoding='utf-8').read().split('\n')
for i, l in enumerate(lines, 1):
    if 'def _update_option_position' in l:
        for j in range(i, min(i+40, len(lines))):
            print(f'{j}: {lines[j-1]}')
        break
