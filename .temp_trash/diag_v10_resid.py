import sys, re
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import expand_query_intents, _decompose_mega_query, _is_equity_reference

qs = [
    'What is the best way to play TGT this week?',
    'What is the best way to protect gains in TGT?',
    'What is the best way to protect gains in LOW?',
]
for q in qs:
    i, t, s = expand_query_intents(q)
    print('Q:', q)
    print('  tickers:', t)
    print('  intents:', {k: v for k, v in i.items() if v})
    print()

print('=== MEGA SPLITS ===')
mq = [
    'what is the probability UBER reaches $41 in 24 months?. Meanwhile, give me a trade setup for BIIB with entry, stop and target compare JNJ and GOOG give me the key support and resistance',
    'should i own long-duration bonds right now?. On a different note, what is the outlook for quantitative tightening?',
]
for q in mq:
    print('Q:', q[:120])
    parts = _decompose_mega_query(q)
    if parts is None:
        print('  -> None (not decomposed)')
    else:
        for p in parts:
            print('  PART:', p[:110])
    print()
