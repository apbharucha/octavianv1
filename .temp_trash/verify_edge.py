import sys

sys.path.insert(0, ".")

from chatbot_eval.pipeline import mocked_pipeline, run_query

qs = [
    "What is the outlook for the S&P?",
    "Where will the S&P be in 12 months?",
    "How are companies guiding for next quarter?",
]
with mocked_pipeline():
    for q in qs:
        intents, tickers, sectors, text, el = run_query(q)
        ql = q.lower()
        ok = True
        if "s&p" in ql:
            ok = ("s&p" in text.lower()) or ("^gspc" in text.lower())
        if "guiding" in ql:
            ok = "earnings" in text.lower()
        print(f"{'OK ' if ok else 'BAD'} t={tickers} | {q[:55]}")
