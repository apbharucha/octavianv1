import sqlite3

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row

# Score a specific mega response with the rubric to see per-criterion scores
import sys
sys.path.insert(0, '.')
from chatbot_eval.rubric import score_response

print('=== MEGA failing cases: rubric detail ===')
for q, rid in [
    ('how does the price of coffee affect', 'bucket-mega-20260808-132008'),
    ('Compare PEP and ABNB', 'bucket-mega-20260808-132008'),
]:
    r = c.execute(
        'SELECT query, response, expectations_json FROM evaluations WHERE run_id=? AND query LIKE ? ORDER BY id DESC LIMIT 1',
        (rid, q + '%'),
    ).fetchone()
    if not r:
        print('NOT FOUND:', q)
        continue
    import json
    try:
        exp = json.loads(r['expectations_json']) if r['expectations_json'] else {}
    except Exception:
        exp = {}
    scores = score_response(r['query'], r['response'] or '', exp, [])
    print('Q:', r['query'][:120])
    for k, (v, reason) in scores.items():
        print(f'   {k}: {v:.2f}  {reason[:90]}')
    print()
