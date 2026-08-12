import sqlite3

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row

runs = {
    'SMALL': 'bucket-small-20260809-141719',
    'MEDIUM': 'bucket-medium-20260809-145207',
    'MEGA': 'bucket-mega-20260809-145812',
}

print('=== SMALL: sector_scan + options + comparison examples ===')
for rid in ('bucket-small-20260809-141719',):
    rows = c.execute(
        "SELECT query, response, issue_tags FROM evaluations WHERE run_id=? AND passed=0 "
        "AND (issue_tags LIKE '%sector_scan%' OR issue_tags LIKE '%options_strategy%' OR issue_tags LIKE '%comparison_two%') ORDER BY id LIMIT 5",
        (rid,)).fetchall()
    for r in rows:
        print('Q:', r['query'][:100])
        print('T:', r['issue_tags'])
        print('R:', (r['response'] or '')[:200].replace('\n', ' | '))
        print()

print('=== MEDIUM examples ===')
for r in c.execute(
        "SELECT query, response, issue_tags FROM evaluations WHERE run_id='bucket-medium-20260809-145207' AND passed=0 ORDER BY id LIMIT 3").fetchall():
    print('Q:', r['query'][:110])
    print('T:', r['issue_tags'])
    print('R:', (r['response'] or '')[:250].replace('\n', ' | '))
    print()

print('=== MEGA examples ===')
for r in c.execute(
        "SELECT query, response, issue_tags FROM evaluations WHERE run_id='bucket-mega-20260809-145812' AND passed=0 ORDER BY id LIMIT 3").fetchall():
    print('Q:', r['query'][:120])
    print('T:', r['issue_tags'])
    print('R:', (r['response'] or '')[:250].replace('\n', ' | '))
    print()
