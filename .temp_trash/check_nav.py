import sys, os
sys.path.insert(0, os.path.abspath('.'))
src = open('main.py', encoding='utf-8').read()
lines = src.split('\n')

# Find sidebar selectbox / navigation
for i, l in enumerate(lines, 1):
    s = l.strip()
    if ('sidebar' in s and ('selectbox' in s or 'radio' in s)) or 'NAV_' in s or 'navigation' in s.lower():
        print(f'{i}: {s[:130]}')
