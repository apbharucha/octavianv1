import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
import financial_llm_engine as fe
import inspect

src = inspect.getsource(fe.expand_query_intents)

# Trace within the cleanup loop: which condition fires for COST?
trace_block = '''
        if t_clean == "COST" and t_clean.upper() == "COST":
            import sys as _s
            _s.stderr.write(f"COST check: stop={t_clean in _STOPWORDS} len={len(t_clean)} num={t_clean in _NUMBER_WORDS} "
                             f"sem={t_clean in _SEMANTIC_CONCEPTS} amb={t_clean in _AMBIGUOUS_TICKER_CONCEPTS} "
                             f"six={len(t_clean)==6 and t_clean.isalpha()}\\n")
'''
src2 = src.replace(
    "    for t in tickers:\n        t_clean = t.strip().upper()",
    "    for t in tickers:\n        t_clean = t.strip().upper()\n" + trace_block,
)
ns = {}
exec(src2, fe.__dict__, ns)
fn = ns['expand_query_intents']
q = 'What happened to COST this week?'
i, t, s = fn(q)
print('RESULT:', t)
