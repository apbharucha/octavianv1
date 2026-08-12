import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from chatbot_eval.pipeline import mocked_pipeline

q = "Analyze the technicals for UAL, then if the trend is constructive give me a trade setup with entry, stop and target, and finally recommend how to hedge the position."
try:
    r = mocked_pipeline(q)
    print('OK', r[:120])
except Exception:
    import traceback
    traceback.print_exc()
