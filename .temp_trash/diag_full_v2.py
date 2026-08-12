import json
import sqlite3

conn = sqlite3.connect("chatbot_eval/octavian_chatbot_eval.db")
conn.row_factory = sqlite3.Row

print("=== macro_outlook failures ===")
rows = conn.execute(
    "SELECT id, query, extracted_tickers, criteria_json FROM evaluations "
    "WHERE run_id='full-v2' AND category='macro_outlook' AND passed=0 LIMIT 6"
).fetchall()
for r in rows:
    crit = json.loads(r["criteria_json"])
    low = [(c, s) for c, (s, _) in crit.items() if s < 8]
    print("-" * 60)
    print("Q:", r["query"][:90])
    print("  t:", r["extracted_tickers"], "| low:", low)
    print("  tf:", crit.get("task_fulfillment", ["", ""])[1][:110])

print()
print("=== visual over-generation sample (expected=0 but charts generated) ===")
rows2 = conn.execute(
    "SELECT query, category, visuals_json FROM evaluations WHERE run_id='full-v2' "
    "AND visuals_json IS NOT NULL AND json_extract(visuals_json, '$.expected')=0 "
    "AND json_extract(visuals_json, '$.generated[0]') IS NOT NULL LIMIT 8"
).fetchall()
for r in rows2:
    v = json.loads(r["visuals_json"])
    print(f"  [{r['category']:<22}] {r['query'][:60]} -> charts {v['generated']}")
