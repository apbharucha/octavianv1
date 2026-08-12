import sys, os, re
sys.path.insert(0, os.path.abspath('.'))
src = open('financial_llm_engine.py', encoding='utf-8').read()
lines = src.split('\n')
for i, l in enumerate(lines, 1):
    if 'def _call_llm' in l:
        for j in range(i, min(i+25, len(lines))):
            print(f'{j}: {lines[j-1]}')
        break

print()
print('=== root test_lm_studio_interpreter.py ===')
print(open('test_lm_studio_interpreter.py', encoding='utf-8').read()[:4500])
