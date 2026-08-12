import sqlite3

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
mapping = {
    'bucket-small-20260809-203208': 'bucket-small-V9-203208',
    'bucket-medium-20260809-203510': 'bucket-medium-V9-203510',
    'bucket-huge-20260809-204013': 'bucket-huge-V9-204013',
    'bucket-mega-20260809-204423': 'bucket-mega-V9-204423',
}
for old, new in mapping.items():
    n = c.execute('UPDATE evaluations SET run_id=? WHERE run_id=?', (new, old)).rowcount
    print(f'{old} -> {new} ({n} rows)')
c.commit()
