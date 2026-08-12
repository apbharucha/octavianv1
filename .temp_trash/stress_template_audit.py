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
]

for tpl in templates:
    fails = 0
    for t in EQUITY:
        t1 = "MSFT" if t != "MSFT" else "AAPL"
        t2 = "GOOG" if t != "GOOG" else "JNJ"
        q = tpl.format(t=t, t1=t1, t2=t2, p=250, pct=25)
        _, tk, _ = expand_query_intents(q)
        if t not in tk:
            fails += 1
    print(f"{fails:4d}/150  {tpl}")
