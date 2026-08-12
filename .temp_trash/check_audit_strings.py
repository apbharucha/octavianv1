import sys, os, subprocess
sys.path.insert(0, os.path.abspath('.'))

def check(fname, pat):
    src = open(fname, encoding='utf-8').read()
    found = pat in src
    print(f'{fname}: {pat!r} -> {"FOUND" if found else "MISSING"}')
    if not found:
        # show similar patterns
        pass
    return found

print('=== 1. ai_chatbot dynamic examples ===')
check('ai_chatbot.py', 'def _get_asset_examples')
r = subprocess.run(['grep', '-n', '_get_asset_examples', '--include=*.py', '.'], capture_output=True, text=True)
print(r.stdout[:1500] if r.stdout else '  (no refs anywhere)')

print('=== 2. main.py breaking trades dynamic ===')
check('main.py', 'tu.get_full_universe_sample(80)')
r = subprocess.run(['grep', '-n', 'get_full_universe_sample\\|breaking_trades\\|BreakingTrades', '--include=*.py', '.'], capture_output=True, text=True)
print(r.stdout[:2000] if r.stdout else '  (no refs)')

print('=== 3. quant_portal crowding ===')
check('quant_portal.py', 'get_crowding_engine()')
r = subprocess.run(['grep', '-n', 'crowding\\|Crowding', 'quant_portal.py'], capture_output=True, text=True)
print(r.stdout[:2000] if r.stdout else '  (no crowding refs)')
