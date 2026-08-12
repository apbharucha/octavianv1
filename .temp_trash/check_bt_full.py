import sys, os
sys.path.insert(0, os.path.abspath('.'))
ml = open('main.py', encoding='utf-8').read().split('\n')
for i in range(318, 365):
    print(f'{i}: {ml[i-1]}')
