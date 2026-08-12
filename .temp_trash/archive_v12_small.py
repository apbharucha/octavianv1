import sqlite3
c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
old = 'bucket-small-20260810-122742'
cur = c.execute("UPDATE evaluations SET run_id=? WHERE run_id=?", ('archived-' + old, old))
print('archived rows:', cur.rowcount)
c.commit()
