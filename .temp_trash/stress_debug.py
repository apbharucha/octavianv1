import sys, random, collections
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import expand_query_intents, _decompose_mega_query
from ticker_universe import get_ticker_universe

random.seed(777)
u = get_ticker_universe()
KNOWN = sorted(u.get_known_ticker_set())

EQUITY = [t for t in KNOWN if len(t) == 2 or (3 <= len(t) <= 5)]
EQUITY = [t for t in EQUITY if not t.endswith('-USD') and not t.endswith('=X')
          and not t.endswith('=F') and t.isalpha()]
AMB = ["TGT", "LOW", "COST", "AI", "MOAT", "NET", "DASH", "CASH", "SUM", "BASE"]
EQUITY = [t for t in EQUITY if t not in AMB and len(t) >= 3][:150]
print(f"EQUITY pool: {len(EQUITY)}")

templates = [
    "What is the outlook for {t}?",
    "Is {t} a good buy right now?",
    "What is the best way to play {t} this week?",
    "What is the best way to protect gains in {t}?",
    "Should I sell covered calls on {t}?",
    "What are the key support and resistance levels for {t}?",
    "What is the probability {t} reaches ${p} in 12 months?",
    "How likely is {t} to drop {pct}% over the next 6 months?",
    "Compare {t1} and {t2}.  What is the outlook for {t}?",
    "Analyze {t}.  In the same vein, should I own long-duration bonds?",
    "Give me a trade setup for {t} with entry, stop and target.",
    "Is the news flow bullish or bearish for {t}?",
    "What sector is {t} in?",
    "How does {t} compare to {t1}?",
    "Should I own {t} or {t1}?",
    "Hedge my {t} position against a market correction.",
    "What is {t}'s dividend yield?",
    "Does {t} pay a dividend?",
    "What happened to {t} this week?",
    "What would a drawdown in {t} look like?",
    "Would a straddle work on {t} before earnings?",
    "What is the probability UBER reaches ${p} in 24 months?. Meanwhile, give me a trade setup for BIIB with entry, stop and target compare JNJ and GOOG give me the key support and resistance",
    "should i own long-duration bonds right now?. On a different note, what is the outlook for quantitative tightening?",
    "what is the outlook for the VIX?. Next, what is the outlook for PYPL?",
    "Analyze the technicals for UAL, then if the trend is constructive give me a trade setup with entry, stop and target, and finally recommend how to hedge the position.",
]

fails_by_t = collections.Counter()
fails_by_tpl = collections.Counter()
count_by_tpl = collections.Counter()
N = 100000
for i in range(N):
    tpl = random.choice(templates)
    t = random.choice(EQUITY)
    t1 = random.choice(EQUITY)
    t2 = random.choice(EQUITY)
    p = random.randint(50, 1500)
    pct = random.randint(5, 50)
    q = tpl.format(t=t, t1=t1, t2=t2, p=p, pct=pct)
    count_by_tpl[tpl] += 1
    try:
        if "play {t}" in tpl:
            _, tk, _ = expand_query_intents(q)
            if t not in tk:
                fails_by_t[t] += 1
                fails_by_tpl[tpl] += 1
        elif "protect gains in {t}" in tpl:
            _, tk, _ = expand_query_intents(q)
            if t not in tk:
                fails_by_t[t] += 1
                fails_by_tpl[tpl] += 1
        elif "In the same vein" in q:
            parts = _decompose_mega_query(q)
            if not parts or len(parts) < 2:
                fails_by_tpl[tpl] += 1
        elif "UBER reaches" in q:
            parts = _decompose_mega_query(q)
            if not parts or len(parts) < 4:
                fails_by_tpl[tpl] += 1
        elif "outlook for the VIX" in q:
            parts = _decompose_mega_query(q)
            if not parts or len(parts) < 2:
                fails_by_tpl[tpl] += 1
        elif "the key levels" in q:
            parts = _decompose_mega_query(q)
            if parts is not None:
                fails_by_tpl[tpl] += 1
        else:
            _, tk, _ = expand_query_intents(q)
            if t not in tk:
                fails_by_t[t] += 1
                fails_by_tpl[tpl] += 1
    except Exception:
        fails_by_tpl[tpl] += 1

print("\nFailures by ticker (top 15):")
for tk_, n in fails_by_t.most_common(15):
    print(f"  {tk_:8s} {n}")
print("\nFailures by template:")
for tpl_, n in fails_by_tpl.most_common():
    print(f"  {n:5d}/{count_by_tpl[tpl_]:5d}  {tpl_[:80]}")
