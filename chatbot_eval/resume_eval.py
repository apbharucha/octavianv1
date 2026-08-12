"""Resume-capable bucket eval runner.

Continues a partially-completed sized-corpus run: generates the SAME
deterministic corpus (seed), skips queries already persisted under the run_id,
evaluates the rest, and persists each row as it finishes. Safe to re-invoke —
each invocation picks up where the last one left off.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import concurrent.futures as cf
import datetime
import logging
import time
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import chatbot_eval.db as db
import chatbot_eval.pipeline as pipeline
import chatbot_eval.prompt_factory as pf
import chatbot_eval.rubric as rubric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, choices=["small", "medium", "huge", "mega"])
    ap.add_argument("--size", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=404)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-seconds", type=int, default=500)
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    conn = db.init_db()
    existing = set()
    for r in conn.execute(
            "SELECT query FROM evaluations WHERE run_id=?", (args.run_id,)).fetchall():
        existing.add(r["query"])

    corpus = pf.generate_sized_corpus(args.bucket, args.size, seed=args.seed)
    todo = [c for c in corpus if c["query"] not in existing]
    print(f"bucket={args.bucket} total={len(corpus)} done={len(corpus)-len(todo)} "
          f"todo={len(todo)} (run {args.run_id})", flush=True)
    if not todo:
        print("nothing left to do", flush=True)
        return

    t0 = time.time()
    deadline = t0 + args.max_seconds
    done = 0

    def work(item):
        try:
            with pipeline.mocked_pipeline():
                intents, tickers, sectors, text, elapsed = pipeline.run_query(item["query"])
            scores = rubric.score_response(item["query"], item["expectations"],
                                           intents, tickers, text, None)
            ov = rubric.overall_score(scores)
            ok = rubric.passed(scores)
            tags = rubric.failure_tags(scores)
            return dict(tickers=tickers, text=text, ov=ov, ok=ok, tags=tags,
                        scores=scores, elapsed=elapsed)
        except Exception as e:
            return dict(tickers=[], text=f"[ERR] {e}", ov=0.0, ok=False,
                        tags=["pipeline_error"], scores={}, elapsed=0.0)

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        i = 0
        for item in todo:
            if time.time() > deadline and futs:
                break
            futs[ex.submit(work, item)] = item
            i += 1
        for fut in cf.as_completed(futs):
            item = futs[fut]
            r = fut.result()
            db.insert_evaluation(
                conn, run_id=args.run_id, query=item["query"], category=item["category"],
                subcategory=item["subcategory"], asset_class=item["asset_class"],
                expected_intent=item["expected_intent"], extracted_tickers=r["tickers"],
                intents={}, response=r["text"], overall=r["ov"], criteria=r["scores"],
                passed=r["ok"], issue_tags=[item["category"]] + r["tags"],
                elapsed_ms=r["elapsed"], visuals=None)
            done += 1
            if done % args.progress_every == 0:
                print(f"  ...{done} persisted ({time.time()-t0:.0f}s)", flush=True)

    s = db.summary(conn, args.run_id)
    print(f"DONE this invocation: +{done} | run total={s['total']} "
          f"passed={s['passed']} ({100.0*s['passed']/max(s['total'],1):.1f}%) "
          f"avg={s['avg']}", flush=True)


if __name__ == "__main__":
    main()
