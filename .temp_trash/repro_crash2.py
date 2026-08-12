import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import generate_financial_analysis

q = ("Analyze the technicals for UAL, then if the trend is constructive give me "
     "a trade setup with entry, stop and target, and finally rank UAL vs DAL vs "
     "LUV for the next quarter")
try:
    r = generate_financial_analysis(q)
    print('TYPE:', type(r))
    if isinstance(r, dict):
        print('OK len:', len(r.get('text', '')))
    else:
        print('STR:', str(r)[:300])
except Exception:
    import traceback
    traceback.print_exc()
