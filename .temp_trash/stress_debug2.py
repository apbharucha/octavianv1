import sys, random, collections
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import expand_query_intents
from ticker_universe import get_ticker_universe

random.seed(777)
u = get_ticker_universe()
KNOWN = sorted(u.get_known_ticker_set())
EQUITY = [t for t in KNOWN if len(t) == 2 or (3 <= len(t) <= 5)]
EQUITY = [t for t in EQUITY if not t.endswith('-USD') and not t.endswith('=X')
          and not t.endswith('=F') and t.isalpha()]
AMB = ["TGT", "LOW", "COST", "AI", "MOAT", "NET", "DASH", "CASH", "SUM", "BASE"]
EQUITY = [t for t in EQUITY if t not in AMB and t != "ALL" and len(t) >= 3][:150]

templates = [
    "What is the outlook for {t}?",
    "Is {t} a good buy right now?",
    "Should I sell covered calls on {t}?",
    "What are the key support and resistance levels for {t}?",
    "What is the probability {t} reaches ${p} in 12 months?",
    "How likely is {t} to drop {pct}% over the next 6 months?",
    "Compare {t1} and {t2}.  What is the outlook for {t}?",
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
    _, tk, _ = expand_query_intents(q)
    if t not in tk:
        fails_by_t[t] += 1
        fails_by_tpl[tpl] += 1

print("Failures by ticker (top 20):")
for tk_, n in fails_by_t.most_common(20):
    print(f"  {tk_:8s} {n}")
print("\nFailures by template:")
for tpl_, n in fails_by_tpl.most_common():
    print(f"  {n:5d}/{count_by_tpl[tpl_]:5d}  {tpl_[:75]}")
