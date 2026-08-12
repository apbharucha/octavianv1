import sys, os
sys.path.insert(0, os.path.abspath('.'))
src = open('paper_trading_system.py', encoding='utf-8').read()
lines = src.split('\n')

# OptionPosition class
for i, l in enumerate(lines, 1):
    if 'class OptionPosition' in l:
        for j in range(i, min(i+20, len(lines))):
            print(f'{j}: {lines[j-1]}')
        break

print('='*60)
# _update_option_position
for i, l in enumerate(lines, 1):
    if 'def _update_option_position' in l:
        for j in range(i, min(i+45, len(lines))):
            print(f'{j}: {lines[j-1]}')
        break
