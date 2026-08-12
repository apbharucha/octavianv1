import sqlite3
c = sqlite3.connect('chatbot_eval/octavian_chatbot_eval.db', timeout=60)
runs = {
    'small': 'bucket-small-20260810-123645',
    'medium': 'bucket-medium-20260810-124021',
    'huge': 'bucket-huge-20260810-124659',
    'mega': 'bucket-mega-20260810-125308',
}
tot_n = tot_ok = 0
for name, run in runs.items():
    n, ok = c.execute(
        "SELECT COUNT(*), COALESCE(SUM(passed),0) FROM evaluations WHERE run_id=?",
        (run,)).fetchone()
    tot_n += n; tot_ok += ok
    print(f"{name}: {n} / {ok} ({ok/n*100:.2f}%)")
print(f"TOTAL v12: {tot_n} / {tot_ok} ({tot_ok/tot_n*100:.2f}%)")
