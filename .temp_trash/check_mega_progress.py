import sqlite3
import sys

sys.path.insert(0, ".")

conn = sqlite3.connect("chatbot_eval/octavian_chatbot_eval.db")
conn.row_factory = sqlite3.Row
n = conn.execute("SELECT COUNT(*) c FROM evaluations WHERE run_id='mega1'").fetchone()["c"]
print("mega1 rows persisted so far:", n)
# avg elapsed for mega1 rows
avg = conn.execute(
    "SELECT AVG(elapsed_ms) a, MAX(elapsed_ms) m, COUNT(*) c FROM evaluations "
    "WHERE run_id='mega1'").fetchone()
print(f"avg elapsed {avg['a']:.0f}ms, max {avg['m']:.0f}ms over {avg['c']} rows")
