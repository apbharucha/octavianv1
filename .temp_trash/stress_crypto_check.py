import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import expand_query_intents

for q in [
    "What is the outlook for AGIX-USD?",
    "What is the outlook for ALGO-USD?",
    "What is the outlook for ALL?",
    "What is the outlook for AM?",
    "What is the outlook for CAT?",
    "What is the outlook for AIR?",
    "What is the outlook for GOLD?",
]:
    _, t, _ = expand_query_intents(q)
    print(f"{q[:48]!r:50s} -> {t}")
