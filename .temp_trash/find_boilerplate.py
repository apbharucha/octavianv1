import sqlite3

conn = sqlite3.connect("chatbot_eval/octavian_chatbot_eval.db")
conn.row_factory = sqlite3.Row
markers = [
    "landscape is evolving", "flow-of-funds suggests",
    "here's what the data shows for", "reflects broad thematic rotation",
    "opec+ supply discipline is keeping oil prices elevated",
    "stay patient and selective", "position sizing should dominate decision quality",
    "monitor regime",
]
for m in markers:
    n = conn.execute(
        "SELECT COUNT(*) c FROM evaluations WHERE run_id='mega3' AND response LIKE ?",
        (f"%{m}%",),
    ).fetchone()["c"]
    print(f"{m:<60} {n}")
