import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from chatbot_eval.pipeline import mocked_pipeline
from financial_llm_engine import _decompose_mega_query, _build_mega_response

def nparts(q):
    parts = _decompose_mega_query(q) or []
    with mocked_pipeline():
        out = _build_mega_response(q, parts, None) or ''
    return out.count('### Part '), parts

print('=== keep distinct instruments ===')
for q in [
    'what is the outlook for the VIX?. Next, what is the outlook for PYPL?',
    'what is the outlook for LI?. And since we are on the topic, what is the outlook for the Nasdaq?',
    'what is the outlook for GILD?. Meanwhile, what is the outlook for the VIX?',
]:
    n, parts = nparts(q)
    print(f"{'PASS' if n >= 2 else 'FAIL'}  parts={n}  {q[:70]}")

print()
print('=== keep distinct same-label macro (no instruments) ===')
q = 'should i own long-duration bonds right now?. On a different note, what is the outlook for quantitative tightening?'
n, _ = nparts(q)
print(f"{'PASS' if n >= 2 else 'FAIL'}  parts={n}  {q[:70]}")

print()
print('=== dedup near-duplicate (same instrument / none) ===')
parts = ['what is the outlook for the economy right now', 'what is the outlook for the economy']
with mocked_pipeline():
    out = _build_mega_response(' '.join(parts), parts, None) or ''
print(f"{'PASS' if out.count('### Part ') == 1 else 'FAIL'}  parts={out.count('### Part ')}")

print()
print('=== UBER 4 parts ===')
q = 'what is the probability UBER reaches $41 in 24 months?. Meanwhile, give me a trade setup for BIIB with entry, stop and target compare JNJ and GOOG give me the key support and resistance'
n, parts = nparts(q)
print(f"{'PASS' if n >= 4 else 'FAIL'}  parts={n}")

print()
print('=== same instrument twice = dedup ===')
parts2 = ['what is the outlook for AAPL', 'what is the outlook for AAPL next quarter']
with mocked_pipeline():
    out2 = _build_mega_response(' '.join(parts2), parts2, None) or ''
print(f"{'PASS' if out2.count('### Part ') == 1 else 'FAIL'}  parts={out2.count('### Part ')}")
