import sys, random
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import expand_query_intents
from ticker_universe import get_ticker_universe

u = get_ticker_universe()
KNOWN = sorted(u.get_known_ticker_set())
EQUITY = [t for t in KNOWN if len(t) == 2 or (3 <= len(t) <= 5)]
EQUITY = [t for t in EQUITY if not t.endswith('-USD') and not t.endswith('=X')
          and not t.endswith('=F') and t.isalpha()]
AMB = ["TGT", "LOW", "COST", "AI", "MOAT", "NET", "DASH", "CASH", "SUM", "BASE"]
EQUITY = [t for t in EQUITY if t not in AMB and len(t) >= 3][:150]

fails = 0
for t in EQUITY:
    q = f"What is the outlook for {t}?"
    _, tk, _ = expand_query_intents(q)
    if t not in tk:
        fails += 1
        if fails <= 25:
            print(f"  FAIL: {t:8s} got={tk}")
print(f"total fails={fails}/{len(EQUITY)}")
