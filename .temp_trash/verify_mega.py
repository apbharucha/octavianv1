import sys

sys.path.insert(0, ".")

from chatbot_eval.pipeline import mocked_pipeline, run_query

queries = [
    "earnings preview for F Now: what is the outlook for the 10-year treasury yield?",
    "where is GBP/USD heading?. On a different note, how does the price of coffee affect the stock market?. Now, what is the price outlook for ethereum?",
    "give me a trade setup for DAL with entry, stop and target. And since we are on the topic, what are the best materials stocks to buy now?",
    "is TSLA overvalued or undervalued?. And since we are on the topic, should i buy polygon at these levels?. And relatedly, how likely is NVDA to fall 30% in 12 months?",
    "Compare C and XLU, then tell me which has better momentum, then give me a covered call strategy on the winner.",
    "Analyze the technicals for META, then if the trend is constructive give me a trade setup with entry, stop and target, and finally recommend how to hedge the position.",
    "What is the outlook for oil? Then, how would a war in the Middle East affect energy stocks? Then give me a futures setup on CL=F with stops, and a hedge for my XOM position.",
    # single-task control queries — must NOT be decomposed
    "what is the outlook for AAPL?",
    "is AAPL a buy right now?",
]

with mocked_pipeline():
    for q in queries:
        intents, tickers, sectors, text, el = run_query(q)
        print("=" * 80)
        print("Q:", q[:95])
        print("tickers:", tickers)
        n_parts = text.count("### Part ")
        print(f"response parts: {n_parts}, len={len(text)}")
        print("head:", text[:150].replace("\n", " | "))
