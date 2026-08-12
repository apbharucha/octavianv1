import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from chatbot_eval.pipeline import mocked_pipeline
from financial_llm_engine import generate_financial_analysis, _decompose_mega_query

qs = [
    'what is the probability UBER reaches $41 in 24 months?. Meanwhile, give me a trade setup for BIIB with entry, stop and target compare JNJ and GOOG give me the key support and resistance',
    'should i own long-duration bonds right now?. On a different note, what is the outlook for quantitative tightening?',
]
for q in qs:
    with mocked_pipeline():
        try:
            resp = generate_financial_analysis(q)
        except Exception as e:
            print('Q:', q[:70])
            print('  EXC:', type(e).__name__, str(e)[:120])
            continue
    text = (resp or {}).get('text', '') if isinstance(resp, dict) else str(resp)
    parts = text.count('### Part ')
    labels = [ln.strip() for ln in text.splitlines() if ln.strip().startswith('### Part ')]
    print('Q:', q[:70])
    print(f'  parts={parts}  labels={labels}')
    print()
