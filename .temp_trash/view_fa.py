import sys, os
sys.path.insert(0, os.path.abspath('.'))
lines = open('financial_llm_engine.py', encoding='utf-8').read().split('\n')
for i in range(586, 650):
    print(f'{i}: {lines[i-1]}')
