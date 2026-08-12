import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from chatbot_eval.pipeline import mocked_pipeline
from financial_llm_engine import generate_financial_analysis, _decompose_mega_query

q = ("Analyze the technicals for UAL, then if the trend is constructive give me "
     "a trade setup with entry, stop and target, and finally rank UAL vs DAL vs "
     "LUV for the next quarter")
parts = _decompose_mega_query(q)
print('parts:', parts)
try:
    r = generate_financial_analysis(q)
    print('OK, len:', len(r.get('text', '')))
except Exception as e:
    import traceback
    traceback.print_exc()
