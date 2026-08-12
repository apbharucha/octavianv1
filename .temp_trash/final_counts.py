import sqlite3

conn = sqlite3.connect("chatbot_eval/octavian_chatbot_eval.db")
conn.row_factory = sqlite3.Row

r = conn.execute(
    "SELECT COUNT(*) n, SUM(passed) ok, ROUND(AVG(overall_score),2) avg "
    "FROM evaluations WHERE run_id='full-v5'"
).fetchone()
print(f"full-v5: n={r['n']} passed={r['ok']} ({100.0*r['ok']/max(r['n'],1):.1f}%) avg={r['avg']}")

print("\nCategories below 97%:")
for row in conn.execute(
    "SELECT category, COUNT(*) n, SUM(passed) ok, ROUND(AVG(overall_score),2) avg "
    "FROM evaluations WHERE run_id='full-v5' GROUP BY category HAVING ok*1.0/n < 0.97 "
    "ORDER BY ok*1.0/n"
).fetchall():
    print(f"  {row['category']:<24} n={row['n']:<4} passed={row['ok']:<4} "
          f"({100.0*row['ok']/row['n']:.1f}%) avg={row['avg']}")
