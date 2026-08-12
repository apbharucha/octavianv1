import sys

sys.path.insert(0, ".")

import financial_llm_engine as fle

q = "give me a trade setup for CSCO with entry, stop and target. what are the best software stocks to buy now?"
print("sent split:", fle._MEGA_SENT_SPLIT_RE.split(q))
groups = fle._decompose_mega_query(q)
print("groups:", groups)
for g in groups or []:
    i, t, s = fle.expand_query_intents(g)
    print("  ", g[:60], "| intents:", sorted(k for k, v in i.items() if v), "| t:", t, "| s:", s)
