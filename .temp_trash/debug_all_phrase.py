import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
import financial_llm_engine as E

# Find which stock_context phrase matches "all stocks are down" for token ALL
import re
src = open('financial_llm_engine.py').read()
fn_src = src[src.index('def _is_equity_reference'):src.index('def _is_equity_reference')+15000]
# extract the stock_context list literally
m = re.search(r'stock_context = \[(.*?)\n    \]', fn_src, re.S)
body = m.group(1)
phrases = re.findall(r'f"([^"]*)"', body)
ql = "all stocks are down".lower()
tl = "all"
for p in phrases:
    rendered = p.replace("{tl}", tl)
    if rendered in ql:
        print("MATCH:", repr(p), "->", repr(rendered))
