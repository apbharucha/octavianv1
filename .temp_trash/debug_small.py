import sys
sys.path.insert(0, ".")
import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")

import chatbot_eval.pipeline as pipeline
import chatbot_eval.prompt_factory as pf
import chatbot_eval.rubric as rubric

corpus = pf.generate_sized_corpus("small", 600, seed=11)
shown = 0
for c in corpus:
    q = c["query"]
    with pipeline.mocked_pipeline():
        intents, tickers, sectors, text, elapsed = pipeline.run_query(q)
    scores = rubric.score_response(q, c["expectations"], intents, tickers, text, None)
    low = {k: v[0] for k, v in scores.items() if v[0] < 5.0}
    if not low:
        continue
    shown += 1
    if shown > 14:
        break
    print("=" * 70)
    print("Q:", q)
    print("tickers:", tickers, "| exp:", c["expectations"])
    print("low:", low)
    for k in ("task_fulfillment", "relevance", "ticker_cleanliness"):
        if scores[k][0] < 6:
            print(f"  {k}: {scores[k][0]} — {scores[k][1][:130]}")
    print("resp:", text[:180].replace("\n", " | "))
