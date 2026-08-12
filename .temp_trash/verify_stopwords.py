import sys, os, time
sys.path.insert(0, os.path.abspath('.'))
import financial_llm_engine as f

tests = [
    "do an intensive market scan and pick 5 new and up and coming biotech stocks, present your reasoning why you think these stocks will rise",
    "what do you think about AAPL and MSFT",
    "analyze NVDA's latest earnings",
    "compare TSLA vs RIVN",
    "show me undervalued energy stocks",
    "is the market bullish today",
    "find good AI semiconductor plays",
]

for q in tests:
    intents, tickers, sectors = f.expand_query_intents(q)
    print(f'Q: {q[:70]}')
    print(f'   tickers={tickers}')
    print(f'   sectors={sectors}')
    print()
