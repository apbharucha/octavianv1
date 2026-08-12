import sqlite3
import sys

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
rid = sys.argv[1] if len(sys.argv) > 1 else 'bucket-medium-20260808-'
rows = c.execute(
    "SELECT run_id, COUNT(*), COALESCE(SUM(passed),0) FROM evaluations WHERE run_id LIKE ? GROUP BY run_id ORDER BY MAX(id) DESC LIMIT 3",
    (rid + '%',),
).fetchall()
for r in rows:
    print(r[0], 'n=', r[1], 'ok=', r[2])
