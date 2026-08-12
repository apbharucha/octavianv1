import sqlite3

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row
rows = c.execute(
    "SELECT run_id, COUNT(*) n, SUM(passed) ok FROM evaluations "
    "WHERE run_id LIKE '%huge%' OR run_id LIKE 'bucket-2026%' OR run_id LIKE '%303%' "
    "GROUP BY run_id ORDER BY MAX(id) DESC LIMIT 15"
).fetchall()
for r in rows:
    print(f"  {r['run_id']}  n={r['n']} ok={r['ok']} pct={100.0*r['ok']/r['n']:.2f}%")
print('--- all distinct run_id patterns (last 20) ---')
rows2 = c.execute("SELECT DISTINCT run_id FROM evaluations ORDER BY MAX(id) DESC LIMIT 20").fetchall()
for r in rows2:
    print(' ', r['run_id'])
