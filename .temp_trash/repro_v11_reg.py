import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import expand_query_intents, _decompose_mega_query, _build_mega_response

qs = [
    'what is the outlook for the VIX?. Next, what is the outlook for PYPL?',
    'what is the outlook for LI?. And since we are on the topic, what is the outlook for the Nasdaq?',
    'what is the outlook for GILD?. Meanwhile, what is the outlook for the VIX?',
]
for q in qs:
    print('Q:', q)
    parts = _decompose_mega_query(q)
    print('  parts:', parts)
    if parts:
        for p in parts:
            i, t, s = expand_query_intents(p)
            print(f'    {p[:60]!r} tickers={t} intents={ {k for k,v in i.items() if v} }')
        out = _build_mega_response(q, parts, None) or ''
        print('  response parts:', out.count('### Part '))
        for ln in out.splitlines():
            if ln.strip().startswith('### Part '):
                print('   ', ln.strip())
    print()
