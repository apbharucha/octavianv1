import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from chatbot_eval.pipeline import mocked_pipeline
from financial_llm_engine import generate_financial_analysis

q = "Analyze the technicals for UAL, then if the trend is constructive give me a trade setup with entry, stop and target, and finally recommend how to hedge the position."
with mocked_pipeline():
    try:
        r = generate_financial_analysis(q)
        print('OK', (r if isinstance(r, str) else r.get('text', ''))[:120])
    except Exception:
        import traceback
        traceback.print_exc()
