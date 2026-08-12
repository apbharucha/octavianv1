"""Evaluation orchestrator.

Usage (from project root):
    python -m chatbot_eval.run_eval --limit 5000 --workers 8     # run an eval
    python -m chatbot_eval.run_eval --report                     # summary report
    python -m chatbot_eval.run_eval --rerun-failed --workers 8   # re-test failures
    python -m chatbot_eval.run_eval --sync-issues                # open/close issues
"""

import argparse
import concurrent.futures as cf
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot_eval import db, pipeline, prompt_factory, rubric  # noqa: E402

SEVERITY = {"critical": 3, "high": 2, "medium": 1, "low": 0}


def _severity_for(failed_frac: float, n: int) -> str:
    if failed_frac >= 0.5 and n >= 20:
        return "critical"
    if failed_frac >= 0.25:
        return "high"
    if failed_frac >= 0.1:
        return "medium"
    return "low"


def _run_batch(conn, run_id, items, workers, progress_every=250, no_visuals=False):
    """Run prompts through the mocked pipeline, score, persist."""
    total = len(items)
    results = [None] * total

    def _work(iq):
        i, item = iq
        try:
            with pipeline.mocked_pipeline():
                intents, tickers, sectors, text, elapsed = pipeline.run_query(item["query"])
                if no_visuals:
                    visuals = None
                else:
                    visuals = pipeline.visual_check(item["query"], tickers,
                                                    item.get("category"))
            scores = rubric.score_response(item["query"], item["expectations"],
                                           intents, tickers, text, visuals)
            ov = rubric.overall_score(scores)
            passed = rubric.passed(scores)
            tags = rubric.failure_tags(scores)
            return (i, {
                "intents": intents, "tickers": tickers, "sectors": sectors,
                "text": text, "scores": scores, "overall": ov,
                "passed": passed, "tags": tags, "elapsed": elapsed,
                "visuals": visuals,
            })
        except Exception as e:  # pipeline crash on this query — count as hard fail
            return (i, {
                "intents": {}, "tickers": [], "sectors": [],
                "text": f"[PIPELINE ERROR] {e}", "scores": {
                    c: (0.0, f"pipeline error: {e}") for c in rubric.CRITERIA
                }, "overall": 0.0, "passed": False,
                "tags": ["pipeline_error"], "elapsed": 0.0,
                "visuals": {"expected": False, "requested": False,
                             "generated": [], "candidates": []},
            })

    def _persist_one(i, res):
        item = items[i]
        issue_tags = [f"{item['category']}"] + res["tags"]
        db.insert_evaluation(
            conn, run_id=run_id, query=item["query"],
            category=item["category"], subcategory=item["subcategory"],
            asset_class=item["asset_class"],
            expected_intent=item["expected_intent"],
            extracted_tickers=res["tickers"], intents=res["intents"],
            response=res["text"], overall=res["overall"],
            criteria=res["scores"], passed=res["passed"],
            issue_tags=issue_tags, elapsed_ms=res["elapsed"],
            visuals=res.get("visuals"),
        )

    # Incremental persistence: each row is committed as its future completes,
    # so a hung/interrupted batch never loses already-finished evaluations.
    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_work, (i, item)): i
                   for i, item in enumerate(items)}
        for fut in cf.as_completed(futures):
            i, res = fut.result()
            _persist_one(i, res)
            done += 1
            if done % progress_every == 0:
                print(f"  ...{done}/{total}", flush=True)
    return results


def run(limit=5000, workers=8, categories=None, run_id=None, seed=42,
        bucket=None, bucket_size=10000, no_visuals=False):
    conn = db.init_db()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if bucket:
        run_id = run_id or f"bucket-{bucket}-{stamp}"
        print(f"Generating sized corpus bucket={bucket} size={bucket_size} "
              f"(seed={seed})...")
        corpus = prompt_factory.generate_sized_corpus(bucket, bucket_size, seed)
    else:
        run_id = run_id or f"run-{stamp}"
        print(f"Generating corpus (seed={seed}, limit={limit})...")
        corpus = prompt_factory.generate_corpus(seed=seed, max_per_category=400,
                                                include_categories=categories)
        if limit:
            corpus = corpus[:limit]
    print(f"Evaluating {len(corpus)} prompts ({run_id}) with {workers} workers...")
    t0 = datetime.datetime.now()
    _run_batch(conn, run_id, corpus, workers, no_visuals=no_visuals)
    dt = (datetime.datetime.now() - t0).total_seconds()
    print(f"Done in {dt:.1f}s ({dt / max(len(corpus), 1):.3f}s/query)")
    return run_id


def report(conn=None, run_id=None, top_n=12):
    conn = conn or db.init_db()
    s = db.summary(conn, run_id)
    print("=" * 78)
    print(f"EVALUATION SUMMARY  (run: {run_id or 'all runs'})")
    print(f"  total: {s['total']}  passed: {s['passed']} "
          f"({100.0 * s['passed'] / max(s['total'], 1):.1f}%)  avg overall: {s['avg']}/10")
    print("-" * 78)
    print("  Per-category:")
    for cat in sorted(s["per_category"], key=lambda c: -s["per_category"][c]["avg"]):
        pc = s["per_category"][cat]
        print(f"    {cat:<28} n={pc['n']:<5} passed={pc['passed']:<5} "
              f"avg={pc['avg']:>5}/10")
    print("-" * 78)
    print("  Per-criterion (all criteria, out of 10):")
    for c in rubric.CRITERIA:
        print(f"    {c:<28} {db.avg_criterion(conn, c, run_id):>5}/10")
    print("-" * 78)
    vq = ("SELECT COUNT(*) n, SUM(CASE WHEN visuals_json IS NOT NULL AND "
          "json_extract(visuals_json, '$.expected')=1 THEN 1 ELSE 0 END) exp,"
          " SUM(CASE WHEN visuals_json IS NOT NULL AND "
          "json_extract(visuals_json, '$.generated[0]') IS NOT NULL THEN 1 ELSE 0 END) gen"
          " FROM evaluations")
    vargs = ()
    if run_id:
        vq += " WHERE run_id=?"
        vargs = (run_id,)
    vr = conn.execute(vq, vargs).fetchone()
    if vr and vr["n"]:
        print("  Visuals: expected=%s generated=%s coverage=%.0f%%" % (
            vr["exp"] or 0, vr["gen"] or 0,
            100.0 * (vr["gen"] or 0) / max(vr["exp"] or 0, 1)))
    print("=" * 78)


def rerun_failed(workers=8, run_id=None, categories=None):
    """Re-evaluate previously failing queries (after fixes) and open/close issues."""
    conn = db.init_db()
    fails = db.failing_queries(conn, run_id=run_id)
    if not fails:
        print("No failing evaluations to re-run.")
        return None
    items = [{
        "query": r["query"], "category": r["category"],
        "subcategory": r["subcategory"], "asset_class": r["asset_class"],
        "expected_intent": r["expected_intent"],
        "expectations": _expectations_from_row(r),
    } for r in fails]
    new_run = run_id or f"fix-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Re-running {len(items)} failing queries as {new_run}...")
    _run_batch(conn, new_run, items, workers)
    # Close any open issue whose example query now passes in the new run
    _sync_issues(conn, new_run)
    report(conn, new_run)
    return new_run


def _expectations_from_row(row):
    import json as _json
    try:
        exp = {}
        meta = _json.loads(row["intents_json"]) if row["intents_json"] else {}
        exp["primary"] = _primary_from_tickers(row["extracted_tickers"])
        cat = row["category"] or ""
        exp["setup_query"] = cat.startswith("setup_")
        exp["risk_query"] = cat.startswith("setup_") or cat.startswith("hedging_") \
            or cat.startswith("risk_") or cat == "geopolitics_assets"
        exp["transmission_query"] = cat.startswith("transmission_")
        return exp
    except Exception:
        return {}


def _primary_from_tickers(tickers_json):
    try:
        tks = json.loads(tickers_json or "[]")
        return tks[0] if tks else None
    except Exception:
        return None


def sync_issues(run_id=None):
    conn = db.init_db()
    _sync_issues(conn, run_id)
    _print_issues(conn)


def _sync_issues(conn, run_id):
    """Open issues for failing patterns, close ones that now pass."""
    s = db.summary(conn, run_id)
    per_cat = s["per_category"]
    # Recompute failure stats per category for the latest run
    rows = conn.execute(
        "SELECT category, COUNT(*) n, SUM(CASE WHEN passed=0 THEN 1 ELSE 0 END) bad "
        "FROM evaluations WHERE run_id=? GROUP BY category", (run_id,)
    ).fetchall()
    seen_cats = set()
    for r in rows:
        cat = r["category"]
        seen_cats.add(cat)
        n = r["n"] or 0
        bad = r["bad"] or 0
        if bad == 0:
            # close any open issue for this category
            conn.execute("UPDATE issues SET status='fixed', fixed_at=? "
                         "WHERE category=? AND status='open'",
                         (db._now(), cat))
            conn.commit()
            continue
        frac = bad / max(n, 1)
        ex_row = conn.execute(
            "SELECT query FROM evaluations WHERE run_id=? AND category=? AND passed=0 "
            "ORDER BY id LIMIT 1", (run_id, cat)).fetchone()
        desc = f"{bad}/{n} queries failed in category '{cat}' ({frac:.0%})"
        db.insert_issue(conn, run_id=run_id, category=cat, description=desc,
                        severity=_severity_for(frac, n),
                        example_query=ex_row["query"] if ex_row else "",
                        affected_count=bad)
    # Close issues for categories no longer failing
    conn.execute("UPDATE issues SET status='fixed', fixed_at=? "
                 "WHERE category NOT IN (%s) AND status='open'"
                 % ",".join("?" * len(seen_cats)),
                 [db._now()] + list(seen_cats))
    conn.commit()


def _print_issues(conn):
    rows = conn.execute(
        "SELECT id, category, severity, description, example_query, affected_count, "
        "status, fix_description FROM issues ORDER BY id DESC LIMIT 40").fetchall()
    print("=" * 78)
    print("ISSUE LOG")
    for r in rows:
        status = r["status"]
        mark = ("FIXED" if status == "fixed" else "OPEN ")
        print(f"  [{mark}] #{r['id']} {r['category']:<28} sev={r['severity']:<8} "
              f"affected={r['affected_count']}")
        print(f"         {r['description']}")
        if r["example_query"]:
            print(f"         e.g. \"{r['example_query'][:100]}\"")
        if r["fix_description"]:
            print(f"         fix: {r['fix_description']}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description="Octavian chatbot evaluation harness")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--categories", type=str, default=None,
                    help="comma-separated category filter")
    ap.add_argument("--bucket", type=str, default=None,
                    choices=["small", "medium", "huge", "mega"],
                    help="run a sized prompt bucket (small/medium/huge/mega)")
    ap.add_argument("--bucket-size", type=int, default=10000,
                    help="number of prompts for a --bucket run (default 10000)")
    ap.add_argument("--no-visuals", action="store_true",
                    help="skip chart-generation checks (much faster; for large stress runs)")
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rerun-failed", action="store_true")
    ap.add_argument("--sync-issues", action="store_true")
    args = ap.parse_args()

    cats = args.categories.split(",") if args.categories else None
    if args.report:
        report(run_id=args.run_id)
    elif args.rerun_failed:
        rerun_failed(workers=args.workers, run_id=args.run_id, categories=cats)
    elif args.sync_issues:
        sync_issues(args.run_id)
    else:
        run(limit=args.limit, workers=args.workers, categories=cats,
            run_id=args.run_id, seed=args.seed, bucket=args.bucket,
            bucket_size=args.bucket_size, no_visuals=args.no_visuals)
        report(run_id=args.run_id)
        _print_issues(db.init_db())


if __name__ == "__main__":
    main()
