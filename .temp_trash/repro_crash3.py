import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import generate_financial_analysis, _decompose_mega_query

q = "Analyze the technicals for UAL, then if the trend is constructive give me a trade setup with entry, stop and target, and finally recommend how to hedge the position."
parts = _decompose_mega_query(q)
print('parts:', parts)
try:
    r = generate_financial_analysis(q)
    print('OK len:', len(r) if isinstance(r, str) else len(r.get('text', '')))
except Exception:
    import traceback
    traceback.print_exc()
