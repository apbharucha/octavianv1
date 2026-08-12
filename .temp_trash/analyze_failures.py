import sqlite3
from collections import Counter

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row

runs = {
    'SMALL': 'bucket-small-20260808-134514',
    'MEDIUM': 'bucket-medium-20260809-134208',
    'HUGE': 'bucket-huge-20260808-131839',
    'MEGA': 'bucket-mega-20260808-132008',
}

for label, rid in runs.items():
    fails = c.execute(
        'SELECT query, response, issue_tags FROM evaluations WHERE run_id=? AND passed=0',
        (rid,),
    ).fetchall()
    tagc = Counter()
    for f in fails:
        for tg in (f['issue_tags'] or '').split(','):
            tagc[tg.strip()] += 1
    print(f'=== {label}: {len(fails)} failures ===')
    for tg, n in tagc.most_common(10):
        print(f'   {tg}: {n}')
    # Sample 2 failure queries
    for f in fails[:2]:
        print('   Q:', f['query'][:100])
        print('   T:', f['issue_tags'])
        print('   R:', (f['response'] or '')[:160].replace('\n', ' | '))
        print()
