import sqlite3

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
mapping = {
    'bucket-small-20260808-134514': 'bucket-small-V6-134514',
    'bucket-medium-20260809-134208': 'bucket-medium-V6-134208',
    'bucket-huge-20260808-131839': 'bucket-huge-V6-131839',
    'bucket-mega-20260808-132008': 'bucket-mega-V6-132008',
}
for old, new in mapping.items():
    n = c.execute('UPDATE evaluations SET run_id=? WHERE run_id=?', (new, old)).rowcount
    print(f'{old} -> {new} ({n} rows)')
c.commit()
