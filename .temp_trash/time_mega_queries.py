import sys
import time

sys.path.insert(0, ".")

from chatbot_eval import prompt_factory
from chatbot_eval.pipeline import mocked_pipeline, run_query, visual_check

corpus = prompt_factory.generate_corpus(seed=42, max_per_category=400)
mega = [p for p in corpus if p["category"] == "mega_prompt"][:12]

with mocked_pipeline():
    for p in mega:
        t0 = time.time()
        intents, tickers, sectors, text, el = run_query(p["query"])
        vis = visual_check(p["query"], tickers, "mega_prompt")
        dt = time.time() - t0
        print(f"{dt:6.2f}s tickers={tickers} vis={len(vis['generated'])} | {p['query'][:80]}")
