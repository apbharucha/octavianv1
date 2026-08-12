import sqlite3
from collections import Counter

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row

runs = {
    'SMALL': 'bucket-small-20260809-203208',
    'MEDIUM': 'bucket-medium-20260809-203510',
    'HUGE': 'bucket-huge-20260809-204013',
    'MEGA': 'bucket-mega-20260809-204423',
}

for label, rid in runs.items():
    fails = c.execute('SELECT query, response, issue_tags, overall_score FROM evaluations WHERE run_id=? AND passed=0 ORDER BY id', (rid,)).fetchall()
    if not fails:
        print(f'=== {label}: 0 failures ===\n')
        continue
    print(f'=== {label}: {len(fails)} failures ===')
    for f in fails:
        print('  Q:', f['query'][:130])
        print('    score:', f['overall_score'], 'tags:', f['issue_tags'])
        print('    R:', (f['response'] or '')[:180].replace('\n', ' | '))
        print()
