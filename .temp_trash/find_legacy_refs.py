import subprocess, os

print('='*70)
print('REFS TO paper_trading_engine (any form)')
print('='*70)
r = subprocess.run(['grep', '-rn', 'paper_trading_engine', '--include=*.py', '.'], capture_output=True, text=True)
out = [l for l in r.stdout.split('\n') if l and '__pycache__' not in l and '.temp_trash' not in l]
print('\n'.join(out) if out else 'NONE')

print()
print('='*70)
print('REFS TO legacy class/func names: PaperTradingEngine, execute_trade, get_portfolio, TradingEngine')
print('='*70)
for pat in ['PaperTradingEngine', 'execute_trade', 'get_portfolio(', 'close_position', 'open_position', 'get_balance']:
    r = subprocess.run(['grep', '-rn', pat, '--include=*.py', '.'], capture_output=True, text=True)
    hits = [l for l in r.stdout.split('\n') if l and '__pycache__' not in l and '.temp_trash' not in l]
    print(f'--- {pat} ({len(hits)} hits)')
    for h in hits[:8]:
        print('   ', h)

print()
print('='*70)
print('Modern paper_trading_system.py public API')
print('='*70)
src = open('paper_trading_system.py', encoding='utf-8').read()
lines = src.split('\n')
for i, l in enumerate(lines, 1):
    s = l.strip()
    if s.startswith('def ') or s.startswith('class '):
        print(f'{i}: {s}')
