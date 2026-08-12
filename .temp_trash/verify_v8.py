import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')

from chatbot_eval.pipeline import mocked_pipeline
from financial_llm_engine import generate_financial_analysis, expand_query_intents, _sector_display

qs = [
    'Scan the technology sector for opportunities',
    'What is the outlook for the technology sector?',
    'Scan the healthcare sector for opportunities',
    'What happens to TGT and gold if the Fed cuts rates aggressively next year while inflation stays sticky?',
    'Is TGT overvalued or undervalued at current levels, and what multiple should it trade at versus XLK?',
    'how likely is COST to fall to $837 in 18 months?',
    'is RIVN overvalued or undervalued?. In the same vein, should i own long-duration bonds right now?',
    'what are the best semiconductors stocks to buy now?. Additionally, should i own long-duration bonds right now?',
]

with mocked_pipeline():
    for q in qs:
        try:
            i, t, s = expand_query_intents(q)
            r = generate_financial_analysis(q) or ''
            print('Q:', q[:70])
            print('  T:', [x for x in t], 'S:', s)
            print('  R:', r[:170].replace('\n', ' | '))
            print()
        except Exception as e:
            print('Q:', q[:70], 'ERR:', e)
            print()

print('--- sector display ---')
from ticker_universe import get_ticker_universe
u = get_ticker_universe()
for q, key in [('Scan the technology sector', 'technology'),
               ('Give me technology stocks with the best momentum', 'technology'),
               ('Scan the healthcare sector', 'healthcare'),
               ('Top banks stocks', 'financials')]:
    print(f'  {q[:35]:38s} -> {_sector_display(q, key)}')
