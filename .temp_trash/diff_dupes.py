import sys, os, re, difflib
sys.path.insert(0, os.path.abspath('.'))
src = open('ai_chatbot.py', encoding='utf-8').read()
lines = src.split('\n')

def extract(start_line):
    """Extract full method body starting at 1-based line."""
    i = start_line - 1
    body = []
    # find indentation of def
    indent = len(lines[i]) - len(lines[i].lstrip())
    body.append(lines[i])
    i += 1
    while i < len(lines):
        l = lines[i]
        if l.strip() == '':
            body.append(l)
            i += 1
            continue
        this_indent = len(l) - len(l.lstrip())
        if this_indent <= indent and l.strip():
            break
        body.append(l)
        i += 1
    return '\n'.join(body)

pairs = [
    ('_get_real_time_data', [2333, 5419]),
    ('_create_advanced_price_chart', [2410, 5496]),
    ('_create_prediction_chart', [2774, 5666]),
]
for name, (a, b) in pairs:
    A = extract(a)
    B = extract(b)
    same = A == B
    print(f'=== {name}: identical={same} | lenA={len(A)} lenB={len(B)} ===')
    if not same:
        diff = list(difflib.unified_diff(A.split('\n'), B.split('\n'), lineterm='', n=1))
        print('\n'.join(diff[:25]))
    print()
