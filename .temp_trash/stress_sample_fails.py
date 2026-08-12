import sys, random
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import expand_query_intents, _COMMON_ENGLISH_WORDS
from ticker_universe import get_ticker_universe

random.seed(777)
u = get_ticker_universe()
KNOWN = sorted(u.get_known_ticker_set())

common = {t for t in KNOWN if t in _COMMON_ENGLISH_WORDS}
print("common-word tickers in known set:", sorted(common)[:40])

excluded = {"AI","BASE","MOAT","LOW","KEY","MAIN","SUM","CASH","WAVE",
           "DASH","PEAK","TRIP","WISH","HOPE","SEED","COST","TGT","NET","SIX",
           "LOSS","UNIT","LTM"}
STOCKS = [t for t in KNOWN if len(t) >= 2 and t not in excluded][:120]

fails = 0
tested = 0
for t in STOCKS:
    q = f"What is the outlook for {t}?"
    _, tk, _ = expand_query_intents(q)
    tested += 1
    if t not in tk:
        fails += 1
        print(f"  FAIL outlook-for: {t}  (common_word={t in _COMMON_ENGLISH_WORDS})")
        if fails >= 14:
            break
print(f"outlook-for tested={tested} fails={fails}")
