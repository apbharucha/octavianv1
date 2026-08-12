import sys, os
sys.path.insert(0, os.path.abspath('.'))
src = open('paper_trading_ui.py', encoding='utf-8').read()
lines = src.split('\n')
for i, l in enumerate(lines, 1):
    if 'execute_option_trade' in l:
        for j in range(max(1, i-10), min(i+20, len(lines))):
            print(f'{j}: {lines[j-1]}')
        print('='*60)
