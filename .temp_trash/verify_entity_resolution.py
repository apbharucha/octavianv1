"""Verify entity resolution: CAPEX/FCF/GPU/CUDA/MOAT/CYCLE/MULTI/BASE/BUILD/
FACTS/UNIT/ASPS/GROSS/LOSS/ASICS/LTM never become tickers; real tickers and
patterns still resolve."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_llm_engine import expand_query_intents

# The exact words that polluted the NVDA output
NEVER = {"CAPEX", "FCF", "GPU", "CUDA", "MOAT", "CYCLE", "MULTI", "BASE", "BUILD",
         "FACTS", "UNIT", "ASPS", "GROSS", "LOSS", "LOSSES", "ASICS", "LTM",
         "WACC", "EBITDA", "EBIT", "DCF", "RSI", "MACD", "NOPAT", "YTD", "TTM"}

NVDA_PROMPT = (
    "Conduct a full fundamental, quantitative, macroeconomic, market-structure, "
    "and sentiment-driven investment assessment of NVIDIA (NVDA). Reconstruct the "
    "company's revenue and earnings drivers by segment. Evaluate NVIDIA's "
    "competitive position against AMD and custom ASICs. Determine whether CUDA "
    "represents a durable moat. Quantify what level of market-share loss the current "
    "valuation could tolerate. Build a DCF using at least three explicit scenarios. "
    "Perform a reverse DCF. Compare the company using EV/Revenue, EV/EBITDA, P/E, "
    "FCF yield. Determine what the equity market is currently pricing in. Analyze "
    "the company's sensitivity to Federal Reserve policy, real interest rates, "
    "inflation, USD strength, economic growth, corporate capex cycles. Construct a "
    "comprehensive risk matrix covering probability, financial impact. Identify "
    "catalysts over 0-3 months, 3-6 months, 6-12 months, 12-24 months. Construct "
    "at least five scenarios: extreme bull, bull, base, bear, extreme bear. "
    "Establish your initial probability distribution. Give me probability-weighted "
    "fair value, current valuation versus fair value, 12-month expected return, "
    "24-month expected return, probability of permanent capital impairment, "
    "probability of a >30% drawdown, probability of a >50% drawdown."
)

intents, tickers, sectors = expand_query_intents(NVDA_PROMPT)
print("Tickers extracted:", tickers)
bad = [t for t in tickers if t.upper() in NEVER]
assert not bad, f"semantic concepts became tickers: {bad}"
assert "NVDA" in tickers, f"NVDA lost: {tickers}"
# AMD is a real competitor the user named — must be preserved (or at least NVDA)
assert "AMD" in tickers, f"AMD lost: {tickers}"
print("NVDA prompt: PASS (no semantic-concept tickers)")

# Also ensure 'AI' as a concept is dropped but 'AI stock' is kept
_, t1, _ = expand_query_intents("How is AI infrastructure spending trending?")
assert "AI" not in t1, f"AI concept leaked: {t1}"
_, t2, _ = expand_query_intents("Is AI a good stock to buy?")
assert "AI" in t2, f"AI equity reference dropped: {t2}"
print("AI disambiguation: PASS")

# Commodity/pattern anchors still resolve
_, t3, _ = expand_query_intents("How does oil affect the stock market?")
assert "CL=F" in t3, t3
_, t4, _ = expand_query_intents("What is the VIX doing?")
assert "^VIX" in t4, t4
_, t5, _ = expand_query_intents("EURUSD outlook")
assert "EURUSD=X" in t5, t5
print("Anchors: PASS")

# Real tickers still resolve
for q, exp in [("Is AAPL bullish?", "AAPL"), ("MSFT earnings", "MSFT"),
               ("What about COST?", "COST"), ("NVDA vs AMD", "AMD")]:
    _, tt, _ = expand_query_intents(q)
    assert exp in tt, f"{exp} lost in {q!r}: {tt}"
print("Real tickers: PASS")

# Wordy meta words in isolation must not create garbage
_, t6, _ = expand_query_intents("Multi-part analysis of the market")
assert "MULTI" not in t6, t6
print("ALL ENTITY RESOLUTION CHECKS PASSED")
