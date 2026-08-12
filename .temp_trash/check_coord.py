import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import _is_equity_reference

cases = [
    ("Analyze TGT's competitive moat versus AAPL: pricing power", 'TGT'),
    ("Analyze TGT's competitive moat versus AAPL: pricing power", 'MOAT'),
    ('Should I own BA or TGT?', 'TGT'),
    ('compare XLF and TGT what is the outlook for MSFT?', 'TGT'),
    ('Compare TGT and JPM', 'TGT'),
    ('What is the dividend yield of LOW?', 'LOW'),
    ('What is LOW yield?', 'LOW'),
    ('Does LOW pay a dividend?', 'LOW'),
]
for q, tk in cases:
    print(f"{tk}: {_is_equity_reference(q, tk)!s:5s}  | {q[:60]}")
