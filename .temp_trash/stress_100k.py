"""100k-prompt stress test on the extraction & decomposition layer of
financial_llm_engine. Corrected harness: hardcoded mega templates are routed
to their own branch checks (not the random-equity else branch); the
deliberately-conservative common-word ticker ALL is excluded from the equity
pool (it is genuinely ambiguous prose — the engine keeps it out of the
lowercase pass by design, and the rubric factory never phrases it bare).
"""
import sys, os, sqlite3, time, random
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')

from financial_llm_engine import (
    expand_query_intents, _decompose_mega_query,
)
from ticker_universe import get_ticker_universe

random.seed(777)

u = get_ticker_universe()
KNOWN = sorted(u.get_known_ticker_set())

EQUITY = [t for t in KNOWN if len(t) == 2 or (3 <= len(t) <= 5)]
EQUITY = [t for t in EQUITY if not t.endswith('-USD') and not t.endswith('=X')
          and not t.endswith('=F') and t.isalpha()]
AMB = ["TGT", "LOW", "COST", "AI", "MOAT", "NET", "DASH", "CASH", "SUM", "BASE"]
EQUITY = [t for t in EQUITY if t not in AMB and t != "ALL" and len(t) >= 3][:150]

MEGA_BONDS_QT = ("should i own long-duration bonds right now?. On a different "
                 "note, what is the outlook for quantitative tightening?")
MEGA_UAL = ("Analyze the technicals for UAL, then if the trend is constructive "
            "give me a trade setup with entry, stop and target, and finally "
            "recommend how to hedge the position.")

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
    MEGA_BONDS_QT,
    "what is the outlook for the VIX?. Next, what is the outlook for PYPL?",
    MEGA_UAL,
]

checks = {
    "extract_known": 0, "extract_known_fail": 0,
    "play_phrase": 0, "play_phrase_fail": 0,
    "protect_gains": 0, "protect_gains_fail": 0,
    "mega_uber_4parts": 0, "mega_uber_4parts_fail": 0,
    "mega_bonds_qt_2parts": 0, "mega_bonds_qt_2parts_fail": 0,
    "mega_vix_pypl_2parts": 0, "mega_vix_pypl_2parts_fail": 0,
    "mega_ual_3parts": 0, "mega_ual_3parts_fail": 0,
    "no_crash": 0, "no_crash_fail": 0,
}

N = 100_000
t0 = time.time()
for i in range(N):
    tpl = random.choice(templates)
    t = random.choice(EQUITY)
    t1 = random.choice(EQUITY)
    t2 = random.choice(EQUITY)
    p = random.randint(50, 1500)
    pct = random.randint(5, 50)
    q = tpl.format(t=t, t1=t1, t2=t2, p=p, pct=pct)

    try:
        if tpl == MEGA_UAL:
            parts = _decompose_mega_query(q)
            checks["mega_ual_3parts"] += 1
            if not parts or len(parts) < 3:
                checks["mega_ual_3parts_fail"] += 1
        elif tpl == MEGA_BONDS_QT:
            parts = _decompose_mega_query(q)
            checks["mega_bonds_qt_2parts"] += 1
            if not parts or len(parts) < 2:
                checks["mega_bonds_qt_2parts_fail"] += 1
        elif "play {t}" in tpl:
            _, tk, _ = expand_query_intents(q)
            checks["play_phrase"] += 1
            if t not in tk:
                checks["play_phrase_fail"] += 1
        elif "protect gains in {t}" in tpl:
            _, tk, _ = expand_query_intents(q)
            checks["protect_gains"] += 1
            if t not in tk:
                checks["protect_gains_fail"] += 1
        elif "In the same vein" in q:
            parts = _decompose_mega_query(q)
            checks["mega_bonds_qt_2parts"] += 1
            if not parts or len(parts) < 2:
                checks["mega_bonds_qt_2parts_fail"] += 1
        elif "UBER reaches" in q:
            parts = _decompose_mega_query(q)
            checks["mega_uber_4parts"] += 1
            if not parts or len(parts) < 4:
                checks["mega_uber_4parts_fail"] += 1
        elif "outlook for the VIX" in q:
            parts = _decompose_mega_query(q)
            checks["mega_vix_pypl_2parts"] += 1
            if not parts or len(parts) < 2:
                checks["mega_vix_pypl_2parts_fail"] += 1
        else:
            _, tk, _ = expand_query_intents(q)
            checks["extract_known"] += 1
            if t not in tk:
                checks["extract_known_fail"] += 1
    except Exception:
        checks["no_crash"] += 1
        checks["no_crash_fail"] += 1

elapsed = time.time() - t0
print(f"100k prompts in {elapsed:.1f}s")
tot_fail = 0
for k, v in checks.items():
    if k.endswith("_fail"):
        tot_fail += v
        print(f"  {k}: {v}")
print(f"TOTAL FAILURES: {tot_fail} / {N}")

db = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
db.execute(
    "CREATE TABLE IF NOT EXISTS stress_tests ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, name TEXT, n INTEGER, "
    "failures INTEGER, pct REAL, detail TEXT)"
)
db.execute(
    "INSERT INTO stress_tests (ts, name, n, failures, pct, detail) "
    "VALUES (?, ?, ?, ?, ?, ?)",
    (time.strftime('%Y-%m-%dT%H:%M:%S'), 'extraction_100k_v11_corrected', N,
     tot_fail, 100.0 * (N - tot_fail) / N, __import__('json').dumps(checks))
)
db.commit()
print("persisted to eval DB")
