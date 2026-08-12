import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
import financial_llm_engine as fe
import inspect

src = inspect.getsource(fe.expand_query_intents)

src2 = src.replace(
    "    results = list(dict.fromkeys(",
    "    import sys as _s; _s.stderr.write('TRACE cleaned: ' + str(cleaned_tickers) + '\\n')\n    results = list(dict.fromkeys(",
)
src2 = src2.replace(
    "    except Exception:\n        pass\n\n    # \u2500\u2500 Drop bare CME/NYMEX contract codes",
    "    except Exception:\n        pass\n    import sys as _s; _s.stderr.write('TRACE after universe: ' + str(results) + '\\n')\n\n    # \u2500\u2500 Drop bare CME/NYMEX contract codes",
)
src2 = src2.replace(
    "    return intents, list(dict.fromkeys(results)), detected_sectors",
    "    import sys as _s; _s.stderr.write('TRACE final: ' + str(list(dict.fromkeys(results))) + '\\n')\n    return intents, list(dict.fromkeys(results)), detected_sectors",
)

ns = {}
exec(src2, fe.__dict__, ns)
fn = ns['expand_query_intents']

for q in ['What happened to COST this week?', 'latest news on COST']:
    print(f'=== {q!r}')
    i, t, s = fn(q)
    print('RESULT:', t)
