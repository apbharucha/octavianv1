import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
import importlib.util
spec = importlib.util.spec_from_file_location("s", ".temp_trash/stress_100k.py")
# Don't run main; just import the module data by executing with a guard
import re
src = open('.temp_trash/stress_100k.py').read()
src = src.replace('if __name__ == "__main__":', 'if False:')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from financial_llm_engine import expand_query_intents

fails = []
for t in mod.EQUITY:
    q = f"outlook for {t}"
    _, tk, _ = expand_query_intents(q)
    if t not in [str(x) for x in tk]:
        fails.append(t)
print("FAILING:", fails)
