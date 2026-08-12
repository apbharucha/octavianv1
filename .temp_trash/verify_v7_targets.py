import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')

from chatbot_eval.pipeline import mocked_pipeline
from financial_llm_engine import generate_financial_analysis, expand_query_intents

qs = [
    'Hedge my LOW position against a market correction: what put structures or collars make sense',
    'Hedge my TGT position against a market correction: what put structures or collars make sense',
    'Which technology stocks are undervalued?',
    'How likely is LOW to drop 25% in the next 3 months?',
    'What options strategy makes sense for LOW before earnings?',
    'How does EWZ compare to TGT?',
    'Should I own TGT or JPM?',
    "Who are LOW's main competitors?",
    'What does TGT do?',
    'Which currency is the best buy right now?',
    'Is the dollar strengthening?',
    'Where will the dollar index be in 12 months?',
]

with mocked_pipeline():
    for q in qs:
        try:
            intents, tickers, sectors = expand_query_intents(q)
            r = generate_financial_analysis(q) or ''
            head = r[:150].replace('\n', ' | ')
            print('Q:', q[:70])
            print('  T:', [t for t in tickers if t], 'S:', sectors)
            print('  R:', head)
            print()
        except Exception as e:
            print('Q:', q[:70], 'ERR:', e)
            print()
