import sys, os
sys.path.insert(0, os.path.abspath('.'))
import financial_llm_engine as f
import inspect

# Find the function that extracts tickers
src = inspect.getsource(f)
lines = src.split('\n')

# Locate stopword / extraction logic
import re
for i, l in enumerate(lines, 1):
    if re.search(r'stopword|STOPWORD|stop_word|STOP_WORD', l, re.I):
        print(f'{i}: {l.strip()[:110]}')

print('=== expand_query_intents head ===')
print(inspect.getsource(f.expand_query_intents)[:3000])
