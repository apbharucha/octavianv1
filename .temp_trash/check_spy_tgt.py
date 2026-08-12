import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import expand_query_intents, _is_equity_reference, _SEMANTIC_CONCEPTS, _COMMON_ENGLISH_WORDS
from ticker_universe import get_ticker_universe

ks = get_ticker_universe().get_known_ticker_set()
print('SPY in known:', 'SPY' in ks)
print('TGT in known:', 'TGT' in ks)
print('SPY in SEMANTIC:', 'SPY' in _SEMANTIC_CONCEPTS)
print('SPY in COMMON:', 'SPY' in _COMMON_ENGLISH_WORDS)
print('TGT in COMMON:', 'TGT' in _COMMON_ENGLISH_WORDS)

q = 'which is the better investment, SPY or TGT?'
print('TGT equity-ref:', _is_equity_reference(q, 'TGT'))
i, t, s = expand_query_intents(q)
print('tickers:', t)
