import sqlite3

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
mapping = {
    'bucket-small-20260809-195938': 'bucket-small-V8-195938',
    'bucket-medium-20260809-200231': 'bucket-medium-V8-200231',
    'bucket-huge-20260809-200745': 'bucket-huge-V8-200745',
    'bucket-mega-20260809-200903': 'bucket-mega-V8-200903',
}
for old, new in mapping.items():
    n = c.execute('UPDATE evaluations SET run_id=? WHERE run_id=?', (new, old)).rowcount
    print(f'{old} -> {new} ({n} rows)')
c.commit()
