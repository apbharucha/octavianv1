import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import expand_query_intents, _decompose_mega_query, _is_equity_reference

print('=== TICKER EXTRACTION ===')
cases = [
    ('What happened to COST this week?', 'COST'),
    ('Is the news flow bullish or bearish for COST?', 'COST'),
    ('Does LOW pay a dividend?', 'LOW'),
    ('Has LOW been growing its dividend?', 'LOW'),
    ('How should I size positions in LOW?', 'LOW'),
    ('Should I sell covered calls on TGT?', 'TGT'),
    ('Is a bull call spread on TGT a good idea?', 'TGT'),
    ('How does TGT compare to AAPL?', 'TGT'),
    ('Should I own BA or TGT?', 'TGT'),
    ('Analyze TGT competitive moat versus AAPL', 'TGT'),
    ('how likely is F to fall to $33 in 6 months', 'F'),
    ('how likely is C to fall to $110 in 12 months', 'C'),
    ('compare XLF and TGT what is the outlook for MSFT?', 'TGT'),
    ('What is the latest news on COST?', 'COST'),
    ('What is the dividend yield of LOW?', 'LOW'),
]
for q, want in cases:
    i, t, s = expand_query_intents(q)
    ok = want in t
    print(f"{'PASS' if ok else 'FAIL'} {q[:52]:54s} -> {t}")

print()
print('=== EQUITY REFERENCE DIRECT ===')
for q, tk in [('What happened to COST this week?', 'COST'),
              ('Analyze TGT competitive moat versus AAPL', 'TGT'),
              ('compare XLF and TGT what is the outlook for MSFT?', 'TGT'),
              ('Does LOW pay a dividend?', 'LOW')]:
    print(f"{_is_equity_reference(q, tk)!s:5s} {q[:55]:57s} {tk}")

print()
print('=== MEGA DECOMPOSITION ===')
for q in ['compare XLF and TGT what is the outlook for MSFT?',
          'how likely is F to fall to $33 in 6 months?. After that, give me a trade setup for SNAP with entry, stop and target',
          'what is the best options strategy for GM before earnings?. Now, how likely is C to fall to $110 in 12 months?']:
    parts = _decompose_mega_query(q)
    print(f'Q: {q[:70]}')
    if parts:
        for p in parts:
            _i, _t, _s = expand_query_intents(p)
            print(f'   [{p[:60]!r}] tickers={_t}')
    else:
        print('   (None)')
    print()
