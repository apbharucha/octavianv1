import sqlite3

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row
for b in ['small', 'medium', 'huge', 'mega']:
    rows = c.execute(
        "SELECT run_id, COUNT(*) n, SUM(passed) ok FROM evaluations "
        "WHERE run_id LIKE ? GROUP BY run_id ORDER BY MAX(id) DESC LIMIT 3",
        (f'bucket-{b}-2026%',)
    ).fetchall()
    print(f'--- {b} ---')
    for r in rows:
        print(f"  {r['run_id']}  n={r['n']} ok={r['ok']} pct={100.0*r['ok']/r['n']:.2f}%")
