"""
LLM Entity-Resolution & Decomposition Stress Harness
====================================================

Generates tens of thousands of prompts from combinatorial templates and
verifies hard invariants of the financial LLM's entity-resolution firewall:

  1. Semantic / finance-concept words (CAPEX, FCF, GPU, CUDA, DCF, EBITDA,
     MOAT, ...) NEVER appear in the extracted ticker list — regardless of
     casing or surrounding context.
  2. Real, known tickers (NVDA, AAPL, ...) DO appear when the user references
     them as securities (uppercase or lowercase).
  3. Ambiguous tokens that are BOTH tickers and concepts (AI, BASE, MOAT, ...)
     resolve to a ticker ONLY when referenced as a security ("AI stock"),
     never as a concept ("AI infrastructure").
  4. Number words (FIVE, TWENTY, ...) and prose written in ALL CAPS never
     resolve to tickers.
  5. Ratio shorthand (EV/EBITDA, P/E, EV/FCF) never produces a ticker from the
     metric side of the slash.
  6. Mega-query decomposition: a single deep-dive thesis is NOT exploded into
     dozens of repetitive parts; genuinely multi-part queries decompose into
     at most MAX_MEGA_PARTS sections with unique labels.

Run:  python3 llm_stress_test.py [num_prompts]      (default 50,000)
"""

import re
import sys
import time
import itertools

sys.path.insert(0, ".")

from financial_llm_engine import expand_query_intents, _decompose_mega_query

# ─────────────────────────────────────────────────────────────────────────────
# Pools
# ─────────────────────────────────────────────────────────────────────────────

# Finance concepts that must never resolve to tickers (the reported bug class).
CONCEPTS = [
    "CAPEX", "FCF", "DCF", "EBITDA", "EBIT", "EPS", "NOPAT", "ROIC", "ROCE",
    "CAGR", "WACC", "RSI", "MACD", "NIM", "MARGIN", "REVENUE", "GROWTH",
    "YIELD", "DIVIDEND", "PAYOUT", "LEVERAGE", "LIQUIDITY", "SOLVENCY", "BETA",
    "ALPHA", "SHARPE", "SORTINO", "VOL", "VOLATILITY", "DRAWDOWN", "PREMIUM",
    "DISCOUNT", "SPREAD", "NOTIONAL", "EXPIRY", "STRIKE", "IV", "IMPLIED",
    "THETA", "GAMMA", "VEGA", "DELTA", "RHO", "VANNA", "CHARM", "VOLGA",
    "GPU", "CPU", "CUDA", "MOAT", "ASIC", "TPU", "NPU", "FPGA", "INFERENCE",
    "TRAINING", "LLM", "MODEL", "DATACENTER", "HYPESCALER",
    "LTM", "NTM", "TTM", "FWD", "YTD", "QTD", "MOM", "YOY", "FY24", "FY25",
    "Q1", "Q2", "Q3", "Q4", "H1", "H2",
    "CONSENSUS", "ESTIMATE", "GUIDANCE", "BEAT", "MISS",
    # prose words that appeared in the reported NVDA prompt in ALL CAPS
    "MULTI", "GROSS", "BUILD", "FACTS", "ASPS", "LOSS", "UNIT", "CYCLE",
    "RATE", "SPEED", "PACE", "RANGE", "STAGE", "SCALE", "SCOPE", "TOTAL",
    "MAIN", "CORE", "KEY", "TOP", "BOTTOM", "END", "START", "MID", "HIGH",
    "LOW", "WIDE", "NARROW", "FAST", "SLOW", "BIG", "SMALL", "LARGE",
    "MEDIUM", "FULL",
]

# Ambiguous tokens — both ticker and concept.
AMBIGUOUS = [
    ("AI", ["AI stock", "AI shares", "AI price", "buy AI", "outlook for AI"],
     ["AI infrastructure", "AI capex", "AI demand", "AI spending", "AI models"]),
    ("BASE", ["BASE stock", "BASE earnings", "outlook for BASE"],
     ["base case", "base scenario", "base rate", "the base"]),
    ("MOAT", ["MOAT ETF", "buy MOAT"],
     ["durable moat", "competitive moat", "moat analysis"]),
    ("LOSS", ["LOSS stock"], ["net loss", "operating loss", "losses"]),
]

# Real tickers (validated against the universe at runtime).
TICKERS = [
    "NVDA", "AAPL", "MSFT", "AMD", "AMZN", "GOOGL", "META", "TSLA", "NFLX",
    "JPM", "BAC", "WMT", "XOM", "CVX", "KO", "PEP", "DIS", "INTC", "CRM",
    "ORCL", "ADBE", "AVGO", "QCOM", "TXN", "MU", "CSCO", "UBER", "ABNB",
    "SHOP", "SNOW", "PLTR", "COIN", "SQ", "PYPL", "V", "MA", "UNH", "PFE",
    "MRK", "LLY", "GS", "MS", "C", "WFC",
]

NUMBER_WORDS = [
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE",
    "TEN", "TWENTY", "THIRTY", "FORTY", "FIFTY", "HUNDRED", "THOUSAND",
    "MILLION", "BILLION", "FIRST", "SECOND", "THIRD", "HALF",
]

RATIOS = ["EV/EBITDA", "EV/Revenue", "EV/EBIT", "EV/FCF", "P/E", "P/FCF",
          "EV / EBITDA", "P / E"]

TEMPLATES = [
    "Analyze {t} and its {c} trends over the next 12 months",
    "What is the outlook for {t}? Focus on {c} and {c2}",
    "Is {t} a good long-term investment? Consider {c}",
    "Build a {c} model for {t} with {n} scenarios",
    "Compare {t} vs {t2} on {c} and {c2}",
    "How does {c} affect {t} in a rising-rate environment",
    "Should I buy {t}? Evaluate {c}, {c2}, and {c3}",
    "What would happen to {t} if {c} doubled",
    "Estimate the {c} sensitivity of {t}",
    "Reverse-engineer {t} using {c} and {c2}",
    "Give me a trade setup for {t} with entry, stop and target",
    "What is the probability {t} pulls back 25%? Consider {c}",
    "Hedge my {t} position with puts. What is the {c} impact",
    "Is {c} a leading indicator for {t}?",
    "Analyze {t} and {t2} {c} multiples",
    "What is {t} worth? Use {c} and DCF",
    "The market is volatile. Is {t} overvalued on {c}",
    "Run a full assessment of {t}: {c}, {c2}, {c3}, {c4}, risks and catalysts",
]


def _build_prompt_set(known: set):
    """Yield (prompt, expected_absent_set, expected_present_set)."""
    real_tickers = [t for t in TICKERS if t in known]

    # 1. Ticker × concept templates
    for t in real_tickers:
        for c in CONCEPTS:
            for c2 in CONCEPTS:
                if c == c2:
                    continue
                yield (TEMPLATES[0].format(t=t, c=c),
                       {c, c2}, {t})

    # 2. Two-ticker comparison templates
    for t1, t2 in itertools.combinations(real_tickers[:15], 2):
        for c in CONCEPTS[:20]:
            yield (TEMPLATES[4].format(t=t1, t2=t2, c=c, c2=c),
                   {c}, {t1, t2})

    # 3. Ratio shorthand must not yield metric-side tickers
    for t in real_tickers[:12]:
        for r in RATIOS:
            yield (f"What is the {r} for {t} and how does it compare to peers",
                   set(re.findall(r"[A-Z]+", r)), {t})

    # 4. Number words in caps must never resolve
    for t in real_tickers[:10]:
        for nw in NUMBER_WORDS:
            yield (f"Estimate {nw} scenarios for {t}",
                   {nw}, {t})

    # 5. Ambiguous tokens — security context resolves, concept context does not
    for tok, sec_ctx, con_ctx in AMBIGUOUS:
        for ctx in sec_ctx:
            for t in real_tickers[:6]:
                yield (f"{ctx} — compare with {t}", set(), {tok, t})
        for ctx in con_ctx:
            yield (ctx, {tok}, set())


def run(n_prompts: int) -> dict:
    from ticker_universe import get_ticker_universe

    universe = get_ticker_universe()
    known = set(universe.get_known_ticker_set())

    failures = []
    total = 0
    t0 = time.time()

    for prompt, absent, present in _build_prompt_set(known):
        if total >= n_prompts:
            break
        total += 1
        try:
            _intents, tickers, _sectors = expand_query_intents(prompt)
        except Exception as e:  # noqa: BLE001
            failures.append((prompt, "CRASH", str(e), []))
            continue

        ticker_set = set(tickers)
        for bad in absent:
            if bad in ticker_set:
                failures.append((prompt, "CONCEPT_LEAK", bad, sorted(ticker_set)))
        for good in present:
            if good not in ticker_set and good in known:
                failures.append((prompt, "MISSING_TICKER", good, sorted(ticker_set)))

    elapsed = time.time() - t0
    return {
        "total": total,
        "failures": failures,
        "elapsed": elapsed,
        "rate": total / elapsed if elapsed else 0,
    }


def run_decomposition(n_prompts: int) -> dict:
    """Deep-dive theses must not explode; multi-part queries must cap + dedupe."""
    deep_dives = [
        "Conduct a full fundamental, quantitative, macroeconomic and sentiment-driven "
        "investment assessment of NVIDIA (NVDA) over the next 12-24 months. "
        "Reconstruct revenue and earnings drivers by segment. Evaluate competitive "
        "position against AMD, custom ASICs and hyperscaler silicon. Determine whether "
        "CUDA represents a durable moat. Build a DCF using at least three explicit "
        "scenarios. Perform a reverse DCF. Compare using EV/Revenue, EV/EBITDA, P/E and "
        "FCF yield. Analyze sensitivity to Fed policy, real interest rates, USD strength "
        "and the semiconductor cycle. Construct a comprehensive risk matrix and catalyst "
        "analysis. Construct at least five scenarios and assign probabilities that sum "
        "to 100%. Identify the 10 strongest arguments against your conclusion. Establish "
        "a Bayesian updating framework. Give me probability-weighted fair value, "
        "12-month expected return, and the probability of a >30% drawdown.",
        "Analyze Apple (AAPL): reconstruct the revenue model by product and geography, "
        "estimate services growth and gross margin trajectory, build a three-stage DCF, "
        "compare against EV/EBITDA comps, run a reverse DCF on the current price, "
        "stress-test under rising rates, list the top risks and catalysts, and give a "
        "final probability-weighted fair value with confidence levels.",
        "Perform a full LBO underwriting of a software company: build the operating "
        "model, layer in the debt schedule with cash sweep, compute sponsor IRR and MOIC "
        "across entry and exit multiple sensitivities, run downside and upside cases, "
        "build the returns-attribution bridge, and conclude with an IC recommendation.",
    ]
    multi_part = [
        "Give me an outlook for AAPL, then a trade setup for MSFT, and what is the "
        "probability NVDA pulls back 25%",
        "Analyze MSFT earnings, hedge my AMD position with puts, and compare TSLA vs "
        "NFLX on growth",
        "What is the outlook for NVDA? Also give me a trade setup for AMD, and explain "
        "how rising rates affect tech stocks",
    ]

    failures = []
    total = 0
    for d in deep_dives:
        total += 1
        parts = _decompose_mega_query(d)
        if parts is not None:
            failures.append((d[:80], "DEEP_DIVE_EXPLODED", parts, None))

    for q in multi_part:
        total += 1
        parts = _decompose_mega_query(q)
        if parts is None or len(parts) > 6:
            failures.append((q[:80], "MULTI_NOT_DECOMPOSED", parts, None))
        elif len(set(parts)) != len(parts):
            failures.append((q[:80], "DUPLICATE_PARTS", parts, None))

    return {"total": total, "failures": failures}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    print(f"Running LLM stress harness on up to {n:,} prompts ...\n")

    res = run(n)
    print(f"Entity resolution: {res['total']:,} prompts in {res['elapsed']:.1f}s "
          f"({res['rate']:,.0f}/s)")
    print(f"  failures: {len(res['failures'])}")
    for prompt, kind, bad, tickers in res['failures'][:25]:
        print(f"  [{kind}] '{prompt[:70]}' -> bad={bad!r} tickers={tickers}")

    dec = run_decomposition(n)
    print(f"\nDecomposition: {dec['total']} prompts, failures: {len(dec['failures'])}")
    for prompt, kind, parts, _x in dec['failures'][:10]:
        print(f"  [{kind}] '{prompt}' parts={parts}")

    ok = not res["failures"] and not dec["failures"]
    print(f"\n{'PASS' if ok else 'FAIL'} — {res['total']:,} prompts, "
          f"{len(res['failures']) + len(dec['failures'])} failures")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
