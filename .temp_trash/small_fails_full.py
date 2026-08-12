import sqlite3

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row
rows = c.execute(
    "SELECT query, response FROM evaluations "
    "WHERE run_id='bucket-small-20260809-205902' AND passed=0 ORDER BY id"
).fetchall()
print(f'{len(rows)} small failures')
for i, r in enumerate(rows, 1):
    print(f'--- {i} ---')
    print('Q:', repr(r['query']))
    print('R:', (r['response'] or '')[:400].replace('\n', ' | '))
    print()
