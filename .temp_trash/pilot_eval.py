import sys
sys.path.insert(0, ".")
import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")
import concurrent.futures as cf
from collections import Counter

import chatbot_eval.db as db
import chatbot_eval.pipeline as pipeline
import chatbot_eval.prompt_factory as pf
import chatbot_eval.rubric as rubric

DB = "/tmp/pilot_eval.db"
db.DB_PATH = DB
conn = db.init_db()

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def work(item):
    try:
        with pipeline.mocked_pipeline():
            intents, tickers, sectors, text, elapsed = pipeline.run_query(item["query"])
            visuals = pipeline.visual_check(item["query"], tickers, item.get("category"))
        scores = rubric.score_response(item["query"], item["expectations"],
                                       intents, tickers, text, visuals)
        ov = rubric.overall_score(scores)
        ok = rubric.passed(scores)
        tags = rubric.failure_tags(scores)
        return dict(text=text, scores=scores, ov=ov, ok=ok, tags=tags,
                    tickers=tickers, elapsed=elapsed)
    except Exception as e:
        return dict(text=f"[ERR] {e}", scores={}, ov=0.0, ok=False,
                    tags=["pipeline_error"], tickers=[], elapsed=0.0)


for b in ("small", "medium", "huge", "mega"):
    corpus = pf.generate_sized_corpus(b, N, seed=11)
    run_id = f"pilot-{b}"
    results = []
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for r in ex.map(work, corpus):
            results.append(r)
    for item, r in zip(corpus, results):
        db.insert_evaluation(
            conn, run_id=run_id, query=item["query"], category=item["category"],
            subcategory=item["subcategory"], asset_class=item["asset_class"],
            expected_intent=item["expected_intent"], extracted_tickers=r["tickers"],
            intents={}, response=r["text"], overall=r["ov"], criteria=r["scores"],
            passed=r["ok"], issue_tags=[item["category"]] + r["tags"],
            elapsed_ms=r["elapsed"], visuals=None)
    s = db.summary(conn, run_id)
    print(f"=== bucket={b}: n={s['total']} passed={s['passed']} "
          f"({100.0*s['passed']/max(s['total'],1):.1f}%) avg={s['avg']}")
    rows = conn.execute(
        "SELECT query, issue_tags FROM evaluations WHERE run_id=? AND passed=0",
        (run_id,)).fetchall()
    tagc = Counter()
    for r in rows:
        for tg in (r["issue_tags"] or "").split(","):
            tagc[tg.strip()] += 1
    for tg, n in tagc.most_common(10):
        print(f"    FAIL[{tg}]: {n}")
    if rows:
        print("    e.g.", rows[0]["query"][:110])
