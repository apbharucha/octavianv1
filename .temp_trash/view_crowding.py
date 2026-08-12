import sys, os
sys.path.insert(0, os.path.abspath('.'))
ql = open('quant_portal.py', encoding='utf-8').read().split('\n')
for i in range(753, 800):
    print(f'{i}: {ql[i-1]}')
