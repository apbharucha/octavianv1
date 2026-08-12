import sys, os, subprocess
sys.path.insert(0, os.path.abspath('.'))

files = [
    'tests/test_integration_flows.py',
    'tests/test_before_start.py',
    'tests/test_audit_regressions.py',
    'tests/test_ib_models.py',
    'tests/test_paper_trading_system.py',
]
for f in files:
    if os.path.exists(f):
        src = open(f, encoding='utf-8').read()
        defs = [l.strip() for l in src.split('\n') if l.strip().startswith('def test_')]
        print(f'=== {f} ({len(defs)} tests) ===')
        for d in defs:
            print('   ', d[:100])
        print()
    else:
        print(f'=== {f} MISSING ===\n')
