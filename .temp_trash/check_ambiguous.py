import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')

from financial_llm_engine import _AMBIGUOUS_TICKER_CONCEPTS, _is_equity_reference, expand_query_intents

for t in ['TGT', 'LOW', 'AI', 'COST', 'RIVN', 'SUM']:
    print(t, 'ambiguous:', t in _AMBIGUOUS_TICKER_CONCEPTS)

print()
for q, t in [
    ('What happens to TGT and gold if the Fed cuts rates aggressively', 'TGT'),
    ('Is TGT overvalued or undervalued at current levels, and what multiple should it trade at versus XLK?', 'TGT'),
    ('How likely is COST to fall to $837 in 18 months?', 'COST'),
    ('is RIVN overvalued or undervalued', 'RIVN'),
]:
    print(f'{t}: {_is_equity_reference(q, t)}  |  {q[:60]}')
    i, tk, s = expand_query_intents(q)
    print('   tickers:', [x for x in tk])
    print()
