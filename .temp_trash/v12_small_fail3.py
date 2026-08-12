import sqlite3
c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row
cols = [x[1] for x in c.execute("PRAGMA table_info(evaluations)").fetchall()]
print('COLS:', cols)
r = c.execute("""
    SELECT query, issue_tags
    FROM evaluations
    WHERE run_id='bucket-small-20260810-122742' AND passed=0
    LIMIT 1
""").fetchone()
print('Q:', repr(r['query']))
print('TAGS:', r['issue_tags'])
