import sqlite3
import json

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row

for q, rid in [
    ('how does the price of coffee affect', 'bucket-mega-20260808-132008'),
    ('Compare PEP and ABNB', 'bucket-mega-20260808-132008'),
]:
    r = c.execute(
        'SELECT query, response, criteria_json, intents_json, issue_tags FROM evaluations WHERE run_id=? AND query LIKE ? ORDER BY id DESC LIMIT 1',
        (rid, q + '%'),
    ).fetchone()
    if not r:
        print('NOT FOUND:', q)
        continue
    print('Q:', r['query'][:130])
    print('tags:', r['issue_tags'])
    try:
        crit = json.loads(r['criteria_json'])
        for k, v in crit.items():
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                print(f'   {k}: {v[0]:.2f}  {str(v[1])[:95]}')
            else:
                print(f'   {k}: {v}')
    except Exception as e:
        print('crit err', e, r['criteria_json'][:200])
    print()
