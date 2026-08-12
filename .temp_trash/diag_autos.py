import re
import sys

sys.path.insert(0, ".")

from chatbot_eval.pipeline import mocked_pipeline, run_query

q = ("Give me a sector view on autos, then the probability MS reaches $435 in 3 months, "
     "then a trade setup for MS with stops, and finally where USD/SGD is heading.")
with mocked_pipeline():
    intents, tickers, sectors, text, el = run_query(q)
print("tickers:", tickers)
print("sectors:", sectors)
unf = re.findall(r"[^\n]*[Uu]nable to fetch[^\n]*", text)
print("unable-fetch lines:", unf)
print("n parts:", text.count("### Part "))
# print the sector part specifically
for m in re.finditer(r"### Part \d+:[^\n]*\n(.*?)(?=\n\n---|\n\n### Part|\Z)", text, re.S):
    print("PART:", m.group(0)[:200].replace("\n", " | "))
