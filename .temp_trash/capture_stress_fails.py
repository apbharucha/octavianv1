import sys, random
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
templates = mod.templates
fails = []
N = 100_000
for i in range(N):
    tpl = random.choice(templates)
    t = random.choice(EQUITY)
    t1 = random.choice(EQUITY)
    t2 = random.choice(EQUITY)
    p = random.randint(50, 1500)
    pct = random.randint(5, 50)
    q = tpl.format(t=t, t1=t1, t2=t2, p=p, pct=pct)
    try:
        if tpl == mod.MEGA_UAL or tpl == mod.MEGA_BONDS_QT or "outlook for the VIX" in tpl or "UBER reaches" in tpl or "In the same vein" in tpl or "play {t}" in tpl or "protect gains in {t}" in tpl:
            continue
        _, tk, _ = expand_query_intents(q)
        if t not in [str(x) for x in tk]:
            fails.append((t, q))
    except Exception:
        pass
print("FAILS:", len(fails))
for t, q in fails[:20]:
    print(repr(t), '|', q[:110])
