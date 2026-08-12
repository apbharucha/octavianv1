import sys, os
sys.path.insert(0, os.path.abspath('.'))
lines = open('main.py', encoding='utf-8').read().split('\n')
for i in range(37, 107):
    print(f'{i}: {lines[i-1]}')
