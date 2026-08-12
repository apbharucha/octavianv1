import sys, random, re
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
import importlib.util
spec = importlib.util.spec_from_file_location("s", ".temp_trash/stress_100k.py")
src = open('.temp_trash/stress_100k.py').read()
src = src.replace('if __name__ == "__main__":', 'if False:')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from financial_llm_engine import expand_query_intents, _decompose_mega_query

EQUITY = mod.EQUITY
fails = []
for tpl in mod.templates:
    if tpl == mod.MEGA_UAL or tpl == mod.MEGA_BONDS_QT or "outlook for the VIX" in tpl:
        continue
    for t in EQUITY:
        for t1 in [t]:
            for t2 in [t]:
                q = tpl.format(t=t, t1=t1, t2=t2, p=500, pct=25)
                if tpl in ("What is the best way to play {t} this week?",):
                    _, tk, _ = expand_query_intents(q)
                    if t not in [str(x) for x in tk]:
                        fails.append((t, tpl))
                elif tpl == "What is the best way to protect gains in {t}?":
                    _, tk, _ = expand_query_intents(q)
                    if t not in [str(x) for x in tk]:
                        fails.append((t, tpl))
                else:
                    _, tk, _ = expand_query_intents(q)
                    if t not in [str(x) for x in tk]:
                        fails.append((t, tpl))
seen = {}
for t, tpl in fails:
    seen.setdefault(t, []).append(tpl)
for t in sorted(seen):
    print(t, '->', len(seen[t]), 'templates')
    for tpl in seen[t][:3]:
        print('   ', tpl)
print('TOTAL unique failing tickers:', len(seen))
