import sys, os, re
sys.path.insert(0, os.path.abspath('.'))
src = open('financial_llm_engine.py', encoding='utf-8').read()
lines = src.split('\n')

for i, l in enumerate(lines, 1):
    if 'def generate_financial_analysis' in l:
        for j in range(i, min(i+60, len(lines))):
            print(f'{j}: {lines[j-1]}')
        break
