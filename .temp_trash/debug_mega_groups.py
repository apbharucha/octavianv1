import sys

sys.path.insert(0, ".")

import financial_llm_engine as fle

q6 = "Analyze the technicals for META, then if the trend is constructive give me a trade setup with entry, stop and target, and finally recommend how to hedge the position."
q1 = "earnings preview for F Now: what is the outlook for the 10-year treasury yield?"

for q in (q6, q1):
    print("=" * 80)
    print("Q:", q)
    print("raw parts:", fle._MEGA_SENT_SPLIT_RE.split(q))
    groups = fle._decompose_mega_query(q)
    print("groups:", groups)
    if groups:
        for g in groups:
            i, t, s = fle.expand_query_intents(g)
            print("   group:", g[:70], "| intents:", {k: v for k, v in i.items() if v},
                  "| t:", t, "| s:", s)
