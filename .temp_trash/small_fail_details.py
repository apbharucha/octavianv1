import sqlite3

c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row

groups = {
    'sector_scan': [],
    'probability_downside': [],
    'options_strategy': [],
    'comparison_two': [],
    'company_facts': [],
    'fx_outlook': [],
    'index_outlook': [],
}

rows = c.execute(
    "SELECT query, response, issue_tags FROM evaluations WHERE run_id='bucket-small-20260808-134514' AND passed=0 ORDER BY id"
).fetchall()
for r in rows:
    tags = r['issue_tags'] or ''
    for g in groups:
        if g in tags:
            groups[g].append(r)
            break

for g, items in groups.items():
    print(f'=== {g}: {len(items)} ===')
    for r in items[:2]:
        print('  Q:', r['query'][:100])
        print('  R:', (r['response'] or '')[:170].replace('\n', ' | '))
        print()
