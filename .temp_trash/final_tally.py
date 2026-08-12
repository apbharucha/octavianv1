import sqlite3
from collections import Counter

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row

v7 = {
    'SMALL': 'bucket-small-20260809-141719',
    'MEDIUM': 'bucket-medium-20260809-145207',
    'HUGE': 'bucket-huge-20260809-145646',
    'MEGA': 'bucket-mega-20260809-145812',
}

total_n = total_ok = 0
print('=== FRESH v7 RUNS ===')
for label, rid in v7.items():
    r = c.execute('SELECT COUNT(*) n, SUM(passed) ok, ROUND(AVG(overall_score),2) avg FROM evaluations WHERE run_id=?', (rid,)).fetchone()
    n, ok, avg = r['n'], r['ok'] or 0, r['avg']
    total_n += n
    total_ok += ok
    print(f'  {label}: n={n} passed={ok} ({100.0*ok/n:.2f}%) avg={avg}')
print(f'  TOTAL: {total_n} prompts, {100.0*total_ok/total_n:.2f}% pass')

print()
for label, rid in v7.items():
    fails = c.execute('SELECT query, response, issue_tags FROM evaluations WHERE run_id=? AND passed=0', (rid,)).fetchall()
    if not fails:
        continue
    tagc = Counter()
    for f in fails:
        for tg in (f['issue_tags'] or '').split(','):
            tagc[tg.strip()] += 1
    print(f'=== {label}: {len(fails)} failures ===')
    for tg, n in tagc.most_common(6):
        print(f'   {tg}: {n}')
    f = fails[0]
    print('   e.g. Q:', f['query'][:90])
    print('   R:', (f['response'] or '')[:140].replace('\n', ' | '))
    print()
