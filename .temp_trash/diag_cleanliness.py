import json
import sqlite3

conn = sqlite3.connect("chatbot_eval/octavian_chatbot_eval.db")
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT query, extracted_tickers, response, criteria_json FROM evaluations "
    "WHERE run_id='mega4' AND passed=0 AND criteria_json LIKE '%ticker_cleanliness%' "
    "ORDER BY overall_score LIMIT 4"
).fetchall()
for r in rows:
    crit = json.loads(r["criteria_json"])
    print("=" * 70)
    print("Q:", r["query"][:100])
    print("extracted:", r["extracted_tickers"])
    s, reason = crit["ticker_cleanliness"]
    print(f"cleanliness {s}: {reason[:220]}")
