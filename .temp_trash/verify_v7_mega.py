import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')

from chatbot_eval.pipeline import mocked_pipeline
from financial_llm_engine import generate_financial_analysis, _is_equity_reference

qs = [
    'Compare PEP and ABNB, then tell me which has better momentum, then give me a covered call strategy on the winner.',
    'Hedge my XOM position against a market correction',
    'What is the best put structure to protect TGT?',
]

with mocked_pipeline():
    for q in qs:
        try:
            r = generate_financial_analysis(q) or ''
            has_literal = '{primary}' in r
            print('Q:', q[:65])
            print('  {primary} literal present:', has_literal)
            print('  R:', r[:220].replace('\n', ' | '))
            print()
        except Exception as e:
            print('Q:', q[:65], 'ERR:', e)
            print()

# Security context sanity checks
print('--- security context ---')
for q, t in [('Hedge my LOW position', 'LOW'), ('Hedge my TGT position', 'TGT'),
             ('How likely is LOW to drop 25%', 'LOW'), ('protect TGT', 'TGT'),
             ('compare to TGT', 'TGT'), ('own TGT or JPM', 'TGT'),
             ("LOW's main competitors", 'LOW'), ('What does TGT do?', 'TGT'),
             ('AI infrastructure spending', 'AI'), ('low probability', 'LOW'),
             ('AI is a concept', 'AI')]:
    print(f'  {q[:38]:40s} {t}: {_is_equity_reference(q, t)}')
