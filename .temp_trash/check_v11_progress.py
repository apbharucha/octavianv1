import sqlite3, sys

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row
prefix = sys.argv[1]
rows = c.execute(
    "SELECT run_id, COUNT(*) n, SUM(passed) ok FROM evaluations "
    "WHERE run_id LIKE ? GROUP BY run_id ORDER BY MAX(id) DESC LIMIT 2",
    (prefix + '%',)
).fetchall()
for r in rows:
    n, ok = r['n'], r['ok']
    print(f"{r['run_id']} n={n} ok={ok} pct={100.0*ok/n:.2f}%")
