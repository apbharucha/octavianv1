import json
import sqlite3

conn = sqlite3.connect("chatbot_eval/octavian_chatbot_eval.db")
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT id, query, extracted_tickers, criteria_json FROM evaluations "
    "WHERE run_id='mega3' AND passed=0 ORDER BY overall_score LIMIT 12"
).fetchall()
print("failures:", len(rows))

# Low criteria across the run
import collections
low = collections.Counter()
for r in conn.execute("SELECT criteria_json FROM evaluations WHERE run_id='mega3'").fetchall():
    crit = json.loads(r["criteria_json"])
    for c, (s, reason) in crit.items():
        if s < 8:
            low[(c, s < 5)] += 1
for (c, severe), n in low.most_common(12):
    print(f"  {c:<26} {'SEVERE' if severe else 'mild'}: {n}")

print("\nExample failures:")
for r in rows:
    crit = json.loads(r["criteria_json"])
    print("=" * 80)
    print("Q:", r["query"][:120])
    for c in ("task_fulfillment", "risk_content", "honesty_no_fabrication", "relevance"):
        s, reason = crit.get(c, ("", ""))
        print(f"  {c}: {s} — {reason[:130]}")
