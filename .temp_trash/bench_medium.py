import sys
sys.path.insert(0, ".")
import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")
import time

import chatbot_eval.pipeline as pipeline
import chatbot_eval.prompt_factory as pf
import financial_llm_engine as fle

corpus = pf.generate_sized_corpus("medium", 30, seed=202)
with pipeline.mocked_pipeline():
    t_int = t_gen = 0.0
    for c in corpus:
        q = c["query"]
        t0 = time.time()
        fle.expand_query_intents(q)
        t_int += time.time() - t0
        t0 = time.time()
        fle.generate_financial_analysis(q)
        t_gen += time.time() - t0
    print(f"30 medium queries: intents {t_int:.1f}s, generate {t_gen:.1f}s ({(t_gen/30)*1000:.0f} ms/query)")
