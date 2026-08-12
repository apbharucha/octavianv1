import sqlite3
c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
c.row_factory = sqlite3.Row
for run in ['bucket-small-20260810-122742', 'bucket-small-20260810-001355', 'bucket-small-20260809-202418']:
    try:
        r = c.execute(
            "SELECT run_id, passed, overall_score, issue_tags, substr(response,1,150) r "
            "FROM evaluations WHERE run_id=? AND query LIKE 'How will USD/BRL%' LIMIT 1",
            (run,)).fetchone()
        if r:
            print(r['run_id'], 'passed=', r['passed'], 'score=', r['overall_score'], 'tags=', r['issue_tags'])
            print('   R:', (r['r'] or '').replace('\n', ' ')[:130])
        else:
            print(run, '-> not found')
    except Exception as e:
        print(run, 'ERR', e)
