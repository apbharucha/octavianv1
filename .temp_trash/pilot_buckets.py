import sys
sys.path.insert(0, ".")
from chatbot_eval import prompt_factory

for b in ("small", "medium", "huge", "mega"):
    c = prompt_factory.generate_sized_corpus(b, 250, seed=7)
    wc = [len(x["query"].split()) for x in c]
    uniq = len({x["query"] for x in c})
    print(f"{b}: n={len(c)} words min={min(wc)} max={max(wc)} avg={sum(wc)/len(wc):.0f} unique={uniq}")
    if b == "huge":
        q = c[0]["query"]
        print("  HUGE SAMPLE (first 200 chars):", q[:200])
        print("  expectations:", c[0]["expectations"])
    if b == "small":
        print("  SMALL SAMPLE:", c[0]["query"])
    if b == "medium":
        print("  MEDIUM SAMPLE:", c[0]["query"][:150])
    if b == "mega":
        print("  MEGA SAMPLE:", c[0]["query"][:150])
