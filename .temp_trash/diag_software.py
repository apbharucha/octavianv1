import sys

sys.path.insert(0, ".")

from chatbot_eval.pipeline import mocked_pipeline, run_query

q = "give me a trade setup for CSCO with entry, stop and target. what are the best software stocks to buy now?"
with mocked_pipeline():
    intents, tickers, sectors, text, el = run_query(q)
print("tickers:", tickers, "sectors:", sectors)
print("parts:", text.count("### Part "))
import re
for m in re.finditer(r"### Part \d+:[^\n]*\n(.*?)(?=\n\n---|\n\n### Part|\Z)", text, re.S):
    seg = m.group(0)
    print("PART HEAD:", seg[:6], "|", seg.split("\n")[1][:100] if len(seg.split("\n")) > 1 else "")
print("software mentioned:", "software" in text.lower() or "technology" in text.lower())
print("--- head ---")
print(text[:700])
