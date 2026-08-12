import faulthandler
import sys
import time

sys.path.insert(0, ".")

faulthandler.dump_traceback_later(45, exit=True)

from chatbot_eval import prompt_factory
from chatbot_eval.pipeline import mocked_pipeline, run_query, visual_check

corpus = prompt_factory.generate_corpus(seed=42, max_per_category=400)
mega = [p for p in corpus if p["category"] == "mega_prompt"]

with mocked_pipeline():
    for i, p in enumerate(mega):
        if i < 20:
            continue
        t0 = time.time()
        print(f"--- [{i}] {p['query'][:95]}", flush=True)
        intents, tickers, sectors, text, el = run_query(p["query"])
        vis = visual_check(p["query"], tickers, "mega_prompt")
        print(f"    {time.time()-t0:.2f}s tickers={tickers}", flush=True)
