import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
import inspect
from financial_llm_engine import expand_query_intents, _is_equity_reference

q = ("You are the lead investment strategist at a multi-strategy institutional asset manager. "
     "Conduct a full fundamental, quantitative, macroeconomic, market-structure, and sentiment-driven "
     "investment assessment of NVIDIA (NVDA) over the next 12-24 months. ... "
     "strongest arguments. Give me probability-weighted fair value and the probability of a >30% drawdown.")
i, t, s = expand_query_intents(q)
print('tickers:', t)

# Find which phrase matched for E — instrument _is_equity_reference
src = inspect.getsource(_is_equity_reference)
import re
m = re.search(r'stock_context = \[(.*?)\]', src, re.S)
phrases = re.findall(r'f"([^"]+)"', m.group(1))

# Reconstruct the phrase list by exec
ns = {}
exec("def _f(tl, phrases):\n    ql='you are the lead investment strategist at a multi-strategy institutional asset manager. conduct a full fundamental, quantitative, macroeconomic, market-structure, and sentiment-driven investment assessment of nvidia (nvda) over the next 12-24 months. ... strongest arguments. give me probability-weighted fair value and the probability of a >30% drawdown.'\n    hits=[]\n    for p in phrases:\n        ctx = p.replace('{tl}', tl)\n        if ctx in ql:\n            hits.append(p)\n    return hits", ns)
for p in phrases:
    ctx = p.replace('{tl}', 'e')
    if ctx in q.lower():
        print('MATCH:', p)
