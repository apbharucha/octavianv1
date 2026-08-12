import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
import financial_llm_engine as fe

q = 'compare XLF and TGT what is the outlook for MSFT?'

# Test the boundary insertion helper directly
from ticker_universe import get_ticker_universe
known = get_ticker_universe().get_known_ticker_set()
print('TGT known:', 'TGT' in known, 'XLF known:', 'XLF' in known)

# Reproduce _insert_unpunctuated_boundaries logic
import re
_QW = r"(?:what|how|which|where|when|why|should I|should we|should)"
out = []
pos = 0
for m in re.finditer(rf"\b([A-Z][A-Z0-9.]{{0,7}})\s+(?={_QW}\b)", q):
    toks = m.group(1)
    print('match at', m.start(), 'token=', toks, 'known=', toks in known)
    if toks in known and m.start() > pos:
        out.append(q[pos:m.start()].rstrip())
        out.append(" ")
        pos = m.start()
out.append(q[pos:])
q2 = "".join(out)
print('Q2:', repr(q2))

parts = fe._decompose_mega_query(q)
print('decompose:', parts)
if parts:
    for p in parts:
        print('  part:', repr(p), fe.expand_query_intents(p)[1])
