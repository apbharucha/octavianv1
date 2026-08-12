import sys, re, inspect
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import _is_equity_reference, _SEMANTIC_CONCEPTS, _COMMON_ENGLISH_WORDS

q = ("Conduct a full fundamental, quantitative, macroeconomic and sentiment-driven "
     "investment assessment of NVIDIA (NVDA). Reconstruct revenue and earnings "
     "drivers by segment. Evaluate competitive position against AMD, custom ASICs "
     "and hyperscaler silicon. Determine whether CUDA represents a durable moat. "
     "Build a DCF using at least three explicit scenarios. Perform a reverse DCF. "
     "Compare using EV/Revenue, EV/EBITDA, P/E and FCF yield. Analyze sensitivity "
     "to Fed policy, real interest rates, USD strength and the semiconductor cycle. "
     "Construct at least five scenarios and assign probabilities. Identify the 10 "
     "strongest arguments. Give me probability-weighted fair value and the "
     "probability of a >30% drawdown.")

ql = q.lower()
print('E equity-ref:', _is_equity_reference(q, 'E'))
print('FCF in SEMANTIC:', 'FCF' in _SEMANTIC_CONCEPTS, '| EV in SEMANTIC:', 'EV' in _SEMANTIC_CONCEPTS)

# Check which phrases in the stock_context list match 'e'
src = inspect.getsource(_is_equity_reference)
m = re.search(r'stock_context = \[(.*?)\]', src, re.S)
block = m.group(1)
for pm in re.finditer(r'f"((?:[^"\\]|\\.)*)"', block):
    p = pm.group(1).replace('{tl}', 'e')
    try:
        p = bytes(p, 'utf-8').decode('unicode_escape')
    except Exception:
        pass
    if p in ql:
        print('PHRASE MATCH:', repr(p))

# Check ratio-shorthand and coord rules for E
print('E before slash:', bool(re.search(r'\bE\s*/', q.upper())))
print('coord patterns:')
for pat in [r'\b([A-Z][A-Z0-9.]{1,6})\s+(?:and|or|vs|versus)\s+E\b',
            r'\bE\s+(?:and|or|vs|versus)\s+([A-Z][A-Z0-9.]{1,6})\b']:
    for m2 in re.finditer(pat, q, re.IGNORECASE):
        print('  ', pat, '->', m2.group(0))
