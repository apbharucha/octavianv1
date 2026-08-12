import sqlite3, json

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row
rows = c.execute(
    "SELECT query, criteria_json, overall_score FROM evaluations "
    "WHERE run_id='bucket-mega-20260809-232046' AND passed=0 ORDER BY id"
).fetchall()
print(f'{len(rows)} mega failures')
for r in rows:
    print('Q:', r['query'][:200])
    print('score:', r['overall_score'])
    try:
        crit = json.loads(r['criteria_json'])
        for k, v in crit.items():
            if isinstance(v, list) and len(v) >= 2:
                print(f'   {k}: {v[0]}  {str(v[1])[:90]}')
    except Exception:
        pass
    print()
