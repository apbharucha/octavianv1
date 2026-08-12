import sys
import time

sys.path.insert(0, ".")

from chatbot_eval.pipeline import mocked_pipeline

with mocked_pipeline():
    # 1) Direct long-horizon check (30 years = 10950 days)
    from target_probability_engine import get_target_probability_engine

    eng = get_target_probability_engine()
    for days in (180, 365, 1095, 3650, 10950):
        t0 = time.time()
        r = eng.analyze_target("AAPL", 400.0, False, days)
        print(f"days={days:>6}: {time.time()-t0:5.2f}s prob={r.final_probability:.1f}%")

    # 2) Mega queries with new stopwords — check extraction cleanliness
    from chatbot_eval import prompt_factory
    from chatbot_eval.pipeline import run_query

    corpus = prompt_factory.generate_corpus(seed=42, max_per_category=400)
    mega = [p for p in corpus if p["category"] == "mega_prompt"]
    bad = []
    for p in mega[:60]:
        _, tickers, _, _, _ = run_query(p["query"])
        for t in tickers:
            if t.upper() in ("SINCE", "TOPIC", "NOTE", "PAY", "FALL", "PULLS",
                             "VIEW", "VEIN", "NEXT", "NOW"):
                bad.append((t, p["query"][:70]))
    print("leak count in first 60 mega:", len(bad))
    for t, q in bad[:8]:
        print("  LEAK:", t, "|", q)
