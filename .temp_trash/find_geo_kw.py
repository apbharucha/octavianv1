import sys

sys.path.insert(0, ".")

import financial_llm_engine as fle

q = "what are the best software stocks to buy now?"
for kw in fle._GEOPOLITICS_KEYWORDS:
    if kw in q:
        print("MATCHED:", repr(kw))
