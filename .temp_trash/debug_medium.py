import sys
sys.path.insert(0, ".")
import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")

import chatbot_eval.pipeline as pipeline
import chatbot_eval.prompt_factory as pf
import chatbot_eval.rubric as rubric

qs = [
    "How do rising yields transmit to tech stocks like XLY, and does duration explain most of the multiple compression?",
    "Analyze the outlook for MSFT over the next 12 months, including the key risks, earnings catalysts and a fair value estimate.",
    "What is the probability NVDA reaches $320 within 12 months, and what are the main upside and downside risks?",
]
corpus = pf.generate_sized_corpus("medium", 120, seed=11)
fails = [c for c in corpus if "transmit" in c["query"] or "probability" in c["query"]]
for c in fails[:6]:
    q = c["query"]
    with pipeline.mocked_pipeline():
        intents, tickers, sectors, text, elapsed = pipeline.run_query(q)
    scores = rubric.score_response(q, c["expectations"], intents, tickers, text, None)
    print("=" * 70)
    print("Q:", q)
    print("tickers:", tickers, "| sectors:", sectors)
    print("exp:", c["expectations"])
    print("scored: tf=%.1f rel=%.1f tc=%.1f dg=%.1f" % (
        scores["task_fulfillment"][0], scores["relevance"][0],
        scores["ticker_cleanliness"][0], scores["data_grounding"][0]))
    for crit in ("task_fulfillment", "relevance", "ticker_cleanliness", "data_grounding"):
        print(f"  {crit}: {scores[crit][0]} — {scores[crit][1][:110]}")
    print("resp head:", text[:220].replace("\n", " | "))
