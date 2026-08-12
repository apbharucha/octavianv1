import sys

sys.path.insert(0, ".")

import financial_llm_engine as fle
from chatbot_eval.pipeline import mocked_pipeline

qs = [
    "What does the data say about nonfarm payrolls?",
    "Explain the jobs report and what it means for investors",
    "Is PCE a risk to markets right now?",
    "Explain PCE and what it means for investors",
    "Where is PCE heading over the next year?",
    "Explain quantitative easing and what it means for investors",
    "What is the outlook for interest rates?",
]
for q in qs:
    print("=" * 70)
    print("Q:", q)
    g = fle._macro_topic_grounding(q)
    print("grounding:", repr(g[:120]))
    with mocked_pipeline():
        _, tickers, sectors, text, _ = fle.expand_query_intents(q), [], [], "", 0
        i, t, s = fle.expand_query_intents(q)
        text = fle.generate_financial_analysis(q)
    ql = q.lower()
    for w in ("pce", "nonfarm", "jobs report", "quantitative easing"):
        if w in ql:
            print(f"  '{w}' in response: {w in text.lower()}")
    print("  head:", text[:160].replace("\n", " | "))
