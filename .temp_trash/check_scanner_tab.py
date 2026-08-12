import sys, os, subprocess
sys.path.insert(0, os.path.abspath('.'))
lines = open('main.py', encoding='utf-8').read().split('\n')
print('=== scanner tab 537-545 ===')
for i in range(537, 546):
    print(f'{i}: {lines[i-1]}')
print()
# find what functions the scanner tab calls
r = subprocess.run(['grep', '-n', 'scan\\|_scan\\|MarketScanner\\|market_scanner', 'main.py'], capture_output=True, text=True)
hits = [l for l in r.stdout.split('\n') if l and 'def ' not in l][:40]
for h in hits[:40]:
    print(h)
