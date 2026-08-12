import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import expand_query_intents, _is_equity_reference

for q in [
    "What is the outlook for ARE?",
    "What is the outlook for ARE stock?",
    "ARE earnings this week",
    "Is ARE overvalued?",
    "buy ARE",
    "What sector is ARE in?",
]:
    _, t, _ = expand_query_intents(q)
    print(f"{q[:52]!r:54s} -> {t}")

print()
print('ARE in COMMON_ENGLISH_WORDS:', 'ARE' in __import__('financial_llm_engine', fromlist=['_COMMON_ENGLISH_WORDS'])._COMMON_ENGLISH_WORDS)
