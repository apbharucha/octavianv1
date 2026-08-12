import sys

sys.path.insert(0, ".")

import financial_llm_engine as fle

parts = [
    "Analyze the technicals for META",
    "if the trend is constructive give me a trade setup with entry, stop and target",
    "recommend how to hedge the position.",
    "if the trend is constructive give me a trade setup",
    "give me a trade setup with entry, stop and target",
]
for p in parts:
    i, t, s = fle.expand_query_intents(p)
    print(f"{p[:65]:<67} -> {sorted(k for k, v in i.items() if v)} | t={t}")
