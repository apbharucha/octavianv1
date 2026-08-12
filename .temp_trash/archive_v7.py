import sqlite3

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
mapping = {
    'bucket-small-20260809-141719': 'bucket-small-V7-141719',
    'bucket-medium-20260809-145207': 'bucket-medium-V7-145207',
    'bucket-huge-20260809-145646': 'bucket-huge-V7-145646',
    'bucket-mega-20260809-145812': 'bucket-mega-V7-145812',
}
for old, new in mapping.items():
    n = c.execute('UPDATE evaluations SET run_id=? WHERE run_id=?', (new, old)).rowcount
    print(f'{old} -> {new} ({n} rows)')
c.commit()
