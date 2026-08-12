import sqlite3, json

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row
runs = {
    'small': 'bucket-small-20260809-205902',
    'mega': 'bucket-mega-20260809-210737',
}
for name, run in runs.items():
    rows = c.execute(
        "SELECT query, criteria_json, overall_score, response FROM evaluations "
        "WHERE run_id=? AND passed=0 ORDER BY id", (run,)
    ).fetchall()
    print(f'===== {name} ({len(rows)} failures) =====')
    for r in rows:
        print('Q:', r['query'][:180])
        print('score:', r['overall_score'])
        try:
            crit = json.loads(r['criteria_json'])
            for k, v in crit.items():
                if isinstance(v, list) and len(v) >= 2:
                    print(f'   {k}: {v[0]}  {str(v[1])[:90]}')
        except Exception:
            pass
        resp = (r['response'] or '')
        print('RESP:', resp[:260].replace('\n', ' | '))
        print()
