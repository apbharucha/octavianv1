import sys, re
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')

src = open('financial_llm_engine.py').read()
fn = src[src.index('def _is_equity_reference'):]
fn = fn[:fn.index('\n\n\n') if '\n\n\n' in fn else len(fn)]
m = re.search(r'stock_context = \[(.*?)\n    \]', fn, re.S)
body = m.group(1)
phrases = re.findall(r'f"([^"]*)"', body)

for q, tok in [("the setup for AI infrastructure spend", "ai"), ("setup for SUM of the parts", "sum")]:
    ql = q.lower()
    print('===', q)
    for p in phrases:
        rendered = p.replace("{tl}", tok)
        if rendered in ql:
            print('  PHRASE MATCH:', repr(p), '->', repr(rendered))
