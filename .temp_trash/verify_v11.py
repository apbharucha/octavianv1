import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import expand_query_intents, _decompose_mega_query, _is_equity_reference

print('=== POSITIVES (ticker extraction) ===')
cases = [
    ('What is the best way to play TGT this week?', ['TGT']),
    ('What is the best way to protect gains in TGT?', ['TGT']),
    ('What is the best way to protect gains in LOW?', ['LOW']),
]
all_ok = True
for q, want in cases:
    i, t, s = expand_query_intents(q)
    ok = sorted(t) == sorted(want)
    all_ok &= ok
    print(f"{'PASS' if ok else 'FAIL'}  {q[:60]!r} -> tickers={t}  want={want}")

print()
print('=== MEGA SPLITS ===')
mq = [
    'what is the probability UBER reaches $41 in 24 months?. Meanwhile, give me a trade setup for BIIB with entry, stop and target compare JNJ and GOOG give me the key support and resistance',
    'should i own long-duration bonds right now?. On a different note, what is the outlook for quantitative tightening?',
]
for q in mq:
    parts = _decompose_mega_query(q)
    print('Q:', q[:95])
    if parts is None:
        print('  -> None')
        continue
    for p in parts:
        i, t, s = expand_query_intents(p)
        print(f'  PART: {p[:95]!r}  tickers={t} intents={ {k for k,v in i.items() if v} }')
    print()

print('=== NEGATIVES (must stay False / no split) ===')
neg = [
    ('Analyze TGT competitive moat versus AAPL: pricing power', 'MOAT'),
    ('The company has a durable moat', 'MOAT'),
    ('What is the probability of a drawdown in the next 6 months', 'TGT'),
    ('net loss widened last quarter', 'LOSS'),
]
for q, tk in neg:
    r = _is_equity_reference(q, tk)
    print(f"{'PASS' if not r else 'FAIL'}  _is_equity_reference({q[:55]!r}, {tk}) -> {r}")

neg_split = [
    'What is the best way to play the market this week?',   # "play the market" - no ticker boundary
    'entry, stop and target are the key levels to watch',    # task words + prose, no split
]
for q in neg_split:
    parts = _decompose_mega_query(q)
    print(f"{'PASS' if parts is None else 'FAIL'}  no-split: {q[:60]!r} -> {parts}")
