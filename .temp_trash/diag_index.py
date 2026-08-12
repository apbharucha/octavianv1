import json
import sqlite3

conn = sqlite3.connect("chatbot_eval/octavian_chatbot_eval.db")
conn.row_factory = sqlite3.Row
for cat in ("index_outlook", "current_events_earnings"):
    print("=" * 70)
    rows = conn.execute(
        "SELECT query, extracted_tickers, criteria_json FROM evaluations "
        "WHERE run_id='full-v4' AND category=? AND passed=0 LIMIT 6", (cat,)
    ).fetchall()
    for r in rows:
        crit = json.loads(r["criteria_json"])
        low = [(c, s) for c, (s, _) in crit.items() if s < 8]
        print("Q:", r["query"][:95])
        print("  t:", r["extracted_tickers"], "| low:", low)
        print("  tf:", crit.get("task_fulfillment", ["", ""])[1][:110])
