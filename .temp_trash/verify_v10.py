import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import _decompose_mega_query, expand_query_intents

print('=== MEGA DECOMPOSITION v10 ===')
cases = [
    "give me a trade setup for XLI with entry, stop and target what is the outlook for the 10-year treasury yield?",
    "give me a trade setup for CAT with entry, stop and target what is the outlook for the 10-year treasury yield?",
    "give me a trade setup for ADBE with entry, stop and target which is the better investment, C or WFC?",
    "what is the outlook for the yield curve? Now: what is the outlook for the 10-year treasury yield?",
    "should i own long-duration bonds right now?. On a different note, what is the outlook for quantitative tightening?",
    "what is the correlation between platinum and the S&P 500?. Separately, which is the better investment, SPY or TGT?",
    "what is the probability UBER reaches $41 in 24 months?. Meanwhile, give me a trade setup for BIIB with entry, stop and target",
]
for q in cases:
    parts = _decompose_mega_query(q)
    print(f'Q: {q[:75]}')
    if parts:
        for p in parts:
            _i, _t, _s = expand_query_intents(p)
            print(f'   [{p[:58]!r}] tickers={_t}')
    else:
        print('   (None)')
    print()

print('=== SMALL/MEDIUM TICKER EXTRACTION ===')
small = [
    ('What are the chances of a drawdown in LOW?', 'LOW'),
    ('Would a straddle work on TGT?', 'TGT'),
    ('What sector is TGT in?', 'TGT'),
    ('How should I think about LOW in my portfolio?', 'LOW'),
    ('What are the chances of a drawdown in TGT?', 'TGT'),
    ('What is the probability of a recession in the next 12 months, and how should I position TGT defensively?', 'TGT'),
]
for q, want in small:
    i, t, s = expand_query_intents(q)
    ok = want in t
    print(f"{'PASS' if ok else 'FAIL'} {q[:60]:62s} -> {t}")
