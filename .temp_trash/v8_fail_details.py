import sqlite3
from collections import Counter

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row

runs = {
    'SMALL': 'bucket-small-20260809-195938',
    'MEDIUM': 'bucket-medium-20260809-200231',
    'HUGE': 'bucket-huge-20260809-200745',
    'MEGA': 'bucket-mega-20260809-200903',
}

for label, rid in runs.items():
    fails = c.execute('SELECT query, response, issue_tags, overall_score FROM evaluations WHERE run_id=? AND passed=0 ORDER BY id', (rid,)).fetchall()
    if not fails:
        print(f'=== {label}: 0 failures ===\n')
        continue
    print(f'=== {label}: {len(fails)} failures ===')
    tagc = Counter()
    for f in fails:
        for tg in (f['issue_tags'] or '').split(','):
            tagc[tg.strip()] += 1
    for tg, n in tagc.most_common(8):
        print(f'   {tg}: {n}')
    print()
    # Show 2 samples with scores
    for f in fails[:2]:
        print('  Q:', f['query'][:120])
        print('  score:', f['overall_score'], 'tags:', f['issue_tags'])
        print('  R:', (f['response'] or '')[:250].replace('\n', ' | '))
        print()
