import sys
sys.path.insert(0, ".")

from chatbot_eval import prompt_factory

corpus = prompt_factory.generate_corpus(seed=42, max_per_category=400)
mega = [p for p in corpus if p["category"] == "mega_prompt"]
print("total corpus:", len(corpus), "| mega prompts:", len(mega))

# Show a sample of mega prompts with task expectations
import collections
counts = collections.Counter()
for p in mega:
    counts[len(p["expectations"].get("mega_tasks", []))] += 1
print("task-count distribution:", dict(sorted(counts.items())))

for p in mega[:6]:
    print("=" * 90)
    print("Q:", p["query"][:170])
    for t in p["expectations"]["mega_tasks"]:
        req = t.get("requires_any") or t.get("requires") or []
        flags = " ".join(k for k, v in t.items() if v is True and k in
                         ("setup_query", "risk_query", "transmission_query"))
        print(f"   [{t.get('label','?')}] req={req[:3]} {flags}")

# Exercise the visual check + one mega query end to end
from chatbot_eval.pipeline import mocked_pipeline, run_query, visual_check

q = mega[0]["query"]
with mocked_pipeline():
    intents, tickers, sectors, text, el = run_query(q)
    vis = visual_check(q, tickers, "mega_prompt")
print("=" * 90)
print("VISUAL CHECK:", vis)
print("RESPONSE head:", text[:240].replace("\n", " | "))
