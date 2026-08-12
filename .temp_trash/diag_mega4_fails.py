import json
import sqlite3

conn = sqlite3.connect("chatbot_eval/octavian_chatbot_eval.db")
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT query, criteria_json FROM evaluations "
    "WHERE run_id='mega4' AND passed=0 ORDER BY overall_score LIMIT 12"
).fetchall()
for r in rows:
    crit = json.loads(r["criteria_json"])
    low = [(c, s) for c, (s, _) in crit.items() if s < 8]
    print("-" * 70)
    print("Q:", r["query"][:120])
    print("  low:", low)
    print("  tf:", crit.get("task_fulfillment", ["", ""])[1][:120])
