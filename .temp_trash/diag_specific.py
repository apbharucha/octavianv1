import sys

sys.path.insert(0, ".")

from chatbot_eval.pipeline import mocked_pipeline, run_query

queries = [
    "Give me a sector view on autos, then the probability MS reaches $435 in 3 months, then a trade setup for MS with stops, and finally where USD/SGD is heading.",
    "how does the price of cotton affect the stock market? what is the price forecast for platinum?",
    "compare MU and DAL. what is the correlation between silver and the S&P 500?",
    "is AMZN overbought or oversold? Now: what is the outlook for nonfarm payrolls?",
]
with mocked_pipeline():
    for q in queries:
        intents, tickers, sectors, text, el = run_query(q)
        print("=" * 80)
        print("Q:", q[:100])
        print("tickers:", tickers)
        # find unable-to-fetch lines
        import re
        unf = re.findall(r"[A-Z0-9=.^-]{1,12}: Unable to fetch[^\n]*", text)
        print("unable-fetch lines:", unf[:5])
        print("--- response (first 1100 chars) ---")
        print(text[:1100])
