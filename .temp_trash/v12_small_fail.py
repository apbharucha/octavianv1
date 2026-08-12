import sqlite3
c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row
rows = c.execute("""
    SELECT query, issue_tags, response
    FROM evaluations
    WHERE run_id='bucket-small-20260810-122742' AND passed=0
    LIMIT 3
""").fetchall()
for r in rows:
    print('Q:', r['query'][:200])
    print('TAGS:', r['issue_tags'])
    print('R:', (r['response'] or '')[:300])
    print('---')
