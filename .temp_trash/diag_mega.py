import json
import sqlite3

conn = sqlite3.connect("chatbot_eval/octavian_chatbot_eval.db")
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT id, query, response, criteria_json FROM evaluations "
    "WHERE run_id='mega2' AND passed=0 ORDER BY overall_score LIMIT 10"
).fetchall()
n_fail = conn.execute(
    "SELECT COUNT(*) c FROM evaluations WHERE run_id='mega2' AND passed=0"
).fetchone()['c']
print('total failures:', n_fail)

# Per sub-task label miss rates
label_misses = {}
label_total = {}
for r in conn.execute("SELECT criteria_json FROM evaluations WHERE run_id='mega2'").fetchall():
    crit = json.loads(r["criteria_json"])
    reason = crit.get("task_fulfillment", ["", ""])[1]
    if "sub-tasks addressed" in reason:
        part = reason.split("(", 1)[1].rstrip(")")
        for item in part.split(","):
            item = item.strip()
            if ":" not in item:
                continue
            label, status = item.split(":", 1)
            label_total[label] = label_total.get(label, 0) + 1
            if status.strip() != "OK":
                label_misses[label] = label_misses.get(label, 0) + 1

print("\nSub-task label miss rates (label: misses/total):")
for label, tot in sorted(label_total.items(), key=lambda x: -(x[1])):
    miss = label_misses.get(label, 0)
    print(f"  {label:<14} {miss:>3}/{tot:<3} ({100*miss/tot:.0f}% missed)")

print("\nExample failures:")
for r in rows[:5]:
    crit = json.loads(r["criteria_json"])
    print("=" * 80)
    print("Q:", r["query"][:130])
    print("  tf:", crit.get("task_fulfillment", ["", ""])[1][:160])
    print("  rel:", crit.get("relevance", ["", ""])[1][:100])
    resp = (r["response"] or "")[:200].replace("\n", " | ")
    print("  R:", resp)
