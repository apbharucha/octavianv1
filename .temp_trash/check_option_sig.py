import sys, os, re
sys.path.insert(0, os.path.abspath('.'))
src = open('paper_trading_system.py', encoding='utf-8').read()
lines = src.split('\n')
for i, l in enumerate(lines, 1):
    if 'def execute_option_trade' in l:
        for j in range(i, min(i+40, len(lines))):
            print(f'{j}: {lines[j-1]}')
        break

print()
print('=== CALLERS OF execute_option_trade ===')
import subprocess
r = subprocess.run(['grep', '-rn', 'execute_option_trade', '--include=*.py', '.'], capture_output=True, text=True)
for line in r.stdout.split('\n'):
    if line and '__pycache__' not in line and '.temp_trash' not in line and 'paper_trading_system.py' not in line:
        print(line)
