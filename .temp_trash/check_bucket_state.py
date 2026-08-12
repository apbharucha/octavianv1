import sqlite3

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row
rows = c.execute(
    """SELECT run_id, COUNT(*) n, SUM(passed) ok, ROUND(AVG(overall_score),2) avg
       FROM evaluations
       WHERE run_id LIKE 'bucket-%'
       GROUP BY run_id ORDER BY MAX(id) DESC LIMIT 14"""
).fetchall()
for r in rows:
    pct = 100.0 * (r["ok"] or 0) / max(r["n"], 1)
    print(f'{r["run_id"]}: n={r["n"]} passed={r["ok"]} ({pct:.2f}%) avg={r["avg"]}')
