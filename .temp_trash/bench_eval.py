import sys
sys.path.insert(0, ".")
import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")
import time

import chatbot_eval.pipeline as pipeline
import chatbot_eval.prompt_factory as pf
import chatbot_eval.rubric as rubric

corpus = pf.generate_sized_corpus("huge", 8, seed=3) + pf.generate_sized_corpus("small", 8, seed=3)

t0 = time.time()
with pipeline.mocked_pipeline():
    for c in corpus:
        intents, tickers, sectors, text, elapsed = pipeline.run_query(c["query"])
t1 = time.time()
print(f"run_query only: {(t1-t0)/len(corpus)*1000:.0f} ms/query")

t0 = time.time()
with pipeline.mocked_pipeline():
    for c in corpus:
        intents, tickers, sectors, text, elapsed = pipeline.run_query(c["query"])
        visuals = pipeline.visual_check(c["query"], tickers, c.get("category"))
t1 = time.time()
print(f"run_query + visual_check: {(t1-t0)/len(corpus)*1000:.0f} ms/query")
