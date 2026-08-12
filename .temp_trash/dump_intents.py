import sys

sys.path.insert(0, ".")
from financial_llm_engine import expand_query_intents

# Show the full intent dict for a set of representative queries
qs = [
    "give me a trade setup for DAL with entry, stop and target",
    "what is the outlook for interest rates",
    "how should i hedge a long NVDA position",
    "what is the probability AAPL reaches $200 in 6 months",
    "how does the price of oil affect the stock market",
    "how would a war in the Middle East affect markets",
    "what did the latest CPI report show",
    "compare AAPL and MSFT",
    "what is the price forecast for gold",
    "what is the price outlook for bitcoin",
    "where is EUR/USD heading",
    "best biotech stocks to buy now",
    "is TSLA overvalued",
    "does WMT pay a dividend",
    "tell me about COST the company",
    "earnings preview for NFLX",
]
for q in qs:
    intents, tickers, sectors = expand_query_intents(q)
    active = {k: v for k, v in intents.items() if v}
    print(f"{q[:55]:<57} -> {sorted(active.keys())} | t={tickers} | s={sectors}")
