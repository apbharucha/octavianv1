import sqlite3

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row

buckets = ['small', 'medium', 'huge', 'mega']
print('=== V11 FINAL TALLY (newest non-archived run per bucket) ===')
tot_n = tot_ok = 0
for b in buckets:
    rows = c.execute(
        "SELECT run_id, COUNT(*) n, SUM(passed) ok FROM evaluations "
        "WHERE run_id LIKE ? AND run_id NOT LIKE ? "
        "GROUP BY run_id ORDER BY MAX(id) DESC LIMIT 1",
        (f'bucket-{b}-2026%', 'bucket-%-V%-%')
    ).fetchall()
    for r in rows:
        n, ok = r['n'], r['ok']
        tot_n += n
        tot_ok += ok
        print(f"{b:6s} {r['run_id']}  n={n} ok={ok} pct={100.0*ok/n:.2f}%")
print(f"TOTAL n={tot_n} ok={tot_ok} pct={100.0*tot_ok/tot_n:.2f}%")
