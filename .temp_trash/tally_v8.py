import sqlite3

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row

# Fresh v8 runs: latest non-archived bucket runs (today 20260809-19xx/20xx)
rows = c.execute("""
    SELECT run_id, COUNT(*) n, SUM(passed) ok, ROUND(AVG(overall_score),2) avg
    FROM evaluations
    WHERE run_id LIKE 'bucket-%20260809-19%' OR run_id LIKE 'bucket-%20260809-20%'
    GROUP BY run_id ORDER BY run_id
""").fetchall()

print('=== FRESH v8 RUNS (2026-08-09) ===')
tot_n = tot_ok = 0
for r in rows:
    pct = 100.0 * r['ok'] / r['n']
    tot_n += r['n']; tot_ok += r['ok']
    print(f"  {r['run_id']}: n={r['n']} passed={r['ok']} ({pct:.2f}%) avg={r['avg']}")
print(f'  TOTAL: n={tot_n} passed={tot_ok} ({100.0*tot_ok/tot_n:.2f}%)')
