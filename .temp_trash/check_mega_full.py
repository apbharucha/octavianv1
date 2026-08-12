import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')

from financial_llm_engine import generate_financial_analysis

q = 'Compare PEP and ABNB, then tell me which has better momentum, then give me a covered call strategy on the winner.'
r = generate_financial_analysis(q) or ''
print(r[:3500])
