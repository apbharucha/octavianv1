import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import expand_query_intents, _decompose_mega_query, _is_equity_reference

print('=== FALSE-POSITIVE GUARDS (should stay clean) ===')
neg = [
    # Concepts must NOT resolve as tickers
    ('what is the net loss for the quarter', ['NET', 'LOSS']),
    ('probabilities that sum to 100%', ['SUM']),
    ('key drivers of growth', ['KEY']),
    ('main risk is a recession', ['MAIN']),
    ('base case scenario', ['BASE']),
    ('AI infrastructure spending', ['AI']),
    ('competitive moat and switching costs', ['MOAT']),
    ('target price of 50', ['TGT']),
    ('what do they cost', ['COST']),
    ('low volatility stocks', ['LOW']),
    # Copulas must not split statements
    ('NVDA is a good buy', None),
    ('AMD can rally 20%', None),
    ('COST was up this week', None),
    ('the market what do you think', None),
    # Single letters NOT in equity context
    ('grade was a C', ['C']),
    ('plan B is better', ['B']),
    ('the F word', ['F']),
    ('V for vendetta', ['V']),
]
for q, bad in neg:
    i, t, s = expand_query_intents(q)
    if bad:
        hits = [b for b in bad if b in t]
        status = 'FAIL' if hits else 'ok'
        print(f"{status:4s} {q[:52]:54s} -> {t}" + (f'  <-- BAD: {hits}' if hits else ''))
    else:
        parts = _decompose_mega_query(q)
        status = 'FAIL' if parts else 'ok'
        print(f"{status:4s} {q[:52]:54s} -> decompose={'yes' if parts else 'no'}")

print()
print('=== POSITIVE (should still work) ===')
pos = [
    ('Is AAPL bullish?', 'AAPL'),
    ('Compare MSFT and GOOGL', 'MSFT'),
    ('What is the outlook for COST?', 'COST'),
    ('Should I buy AI stock?', 'AI'),
    ('TGT earnings preview', 'TGT'),
    ('How likely is LOW to drop 25%', 'LOW'),
]
for q, want in pos:
    i, t, s = expand_query_intents(q)
    ok = want in t
    print(f"{'PASS' if ok else 'FAIL'} {q[:50]:52s} -> {t}")
