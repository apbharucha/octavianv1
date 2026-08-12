import sys

sys.path.insert(0, ".")

from chatbot_eval.pipeline import mocked_pipeline, run_query

q = "earnings preview for F Now: what is the outlook for the 10-year treasury yield?"
with mocked_pipeline():
    intents, tickers, sectors, text, el = run_query(q)
print("tickers:", tickers)
print("parts:", text.count("### Part "))
print("head:", text[:180].replace("\n", " | "))
