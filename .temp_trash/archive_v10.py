import sqlite3

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
mapping = {
    'bucket-small-20260809-205902': 'bucket-small-V10-205902',
    'bucket-medium-20260809-210151': 'bucket-medium-V10-210151',
    'bucket-huge-20260809-225744': 'bucket-huge-V10-225744',
    'bucket-mega-20260809-210737': 'bucket-mega-V10-210737',
}
for old, new in mapping.items():
    n = c.execute('UPDATE evaluations SET run_id=? WHERE run_id=?', (new, old)).rowcount
    print(f'{old} -> {new} ({n} rows)')
c.commit()
