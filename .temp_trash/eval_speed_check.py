import sys
import time

sys.path.insert(0, ".")

from chatbot_eval.pipeline import mocked_pipeline

with mocked_pipeline():
    from target_probability_engine import get_target_probability_engine

    eng = get_target_probability_engine()
    for sym in ["AAL", "ARKK", "SLV", "XLK", "MU", "PFE", "AAPL", "GLD"]:
        t0 = time.time()
        try:
            r = eng.analyze_target(sym, 100.0, False, 180)
            print(f"{sym}: {time.time()-t0:.2f}s prob={r.final_probability:.1f}%")
        except Exception as e:
            print(f"{sym}: {time.time()-t0:.2f}s ERROR {e!r}")
