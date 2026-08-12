import sqlite3

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row

rows = c.execute("""
    SELECT run_id, COUNT(*) n, SUM(passed) ok, ROUND(AVG(overall_score),2) avg
    FROM evaluations
    WHERE run_id IN ('bucket-small-20260809-203208','bucket-medium-20260809-203510',
                     'bucket-huge-20260809-204013','bucket-mega-20260809-204423')
    GROUP BY run_id ORDER BY run_id
""").fetchall()

print('=== FRESH v9 RUNS (2026-08-09) ===')
tot_n = tot_ok = 0
for r in rows:
    pct = 100.0 * r['ok'] / r['n']
    tot_n += r['n']; tot_ok += r['ok']
    print(f"  {r['run_id']}: n={r['n']} passed={r['ok']} ({pct:.2f}%) avg={r['avg']}")
print(f'  TOTAL: n={tot_n} passed={tot_ok} ({100.0*tot_ok/tot_n:.2f}%)')

print()
print('=== PREVIOUS (archived) ===')
prev = c.execute("""
    SELECT run_id, COUNT(*) n, SUM(passed) ok FROM evaluations
    WHERE run_id LIKE '%-V8-%' GROUP BY run_id ORDER BY run_id
""").fetchall()
tot_n = tot_ok = 0
for r in prev:
    pct = 100.0 * r['ok'] / r['n']
    tot_n += r['n']; tot_ok += r['ok']
    print(f"  {r['run_id']}: n={r['n']} passed={r['ok']} ({pct:.2f}%)")
print(f'  V8 TOTAL: n={tot_n} passed={tot_ok} ({100.0*tot_ok/tot_n:.2f}%)')
