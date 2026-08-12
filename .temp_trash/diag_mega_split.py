import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')

from financial_llm_engine import _decompose_mega_query, expand_query_intents

qs = [
    "how likely is F to fall to $33 in 6 months?. After that, give me a trade setup for SNAP with entry, stop and target",
    "compare XLF and TGT what is the outlook for MSFT?",
    "what is the best options strategy for GM before earnings?. Now, how likely is C to fall to $110 in 12 months?",
    "compare XLF and TGT. what is the outlook for MSFT?",
]

for q in qs:
    print('Q:', q[:95])
    parts = _decompose_mega_query(q)
    print('  fragments:', len(parts) if parts else None)
    for i, p in enumerate(parts or []):
        _i, _t, _s = expand_query_intents(p)
        print(f'   [{i}] {p[:80]!r} tickers={_t}')
    print()
