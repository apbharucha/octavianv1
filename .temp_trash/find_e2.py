import sys, re
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import expand_query_intents, _is_equity_reference

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
i, t, s = expand_query_intents(q)
print('tickers:', t)
print('E equity-ref:', _is_equity_reference(q, 'E'))

# where is standalone E
for m in re.finditer(r'\bE\b', q):
    print('E at', m.start(), repr(q[max(0,m.start()-25):m.start()+25]))
