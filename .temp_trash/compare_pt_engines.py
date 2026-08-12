import sys, os, re
sys.path.insert(0, os.path.abspath('.'))
os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')

print('='*70)
print('LEGACY paper_trading_engine.py — structure')
print('='*70)
src = open('paper_trading_engine.py', encoding='utf-8').read()
lines = src.split('\n')
print('lines:', len(lines))
for i, l in enumerate(lines, 1):
    if re.match(r'\s*(def|class) ', l):
        print(f'{i}: {l.strip()}')

print()
print('='*70)
print('WHO IMPORTS paper_trading_engine?')
print('='*70)
import subprocess
r = subprocess.run(['grep', '-rn', 'paper_trading_engine', '--include=*.py', '.'], capture_output=True, text=True)
for line in r.stdout.split('\n'):
    if line and '__pycache__' not in line and '.temp_trash' not in line:
        print(line)

print()
print('='*70)
print('MODERN paper_trading_system.py — public API')
print('='*70)
src2 = open('paper_trading_system.py', encoding='utf-8').read()
lines2 = src2.split('\n')
for i, l in enumerate(lines2, 1):
    if re.match(r'\s*(def|class) ', l):
        name = l.strip()
        # try to capture docstring first line
        doc = ''
        for j in range(i, min(i+3, len(lines2))):
            d = lines2[j].strip()
            if d.startswith('"""') or d.startswith("'''"):
                doc = d.strip('"\'')
                break
        print(f'{i}: {name}' + (f'   # {doc[:60]}' if doc else ''))
