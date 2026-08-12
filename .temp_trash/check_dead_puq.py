import sys, os
sys.path.insert(0, os.path.abspath('.'))
lines = open('ai_chatbot.py', encoding='utf-8').read().split('\n')
# Find both process_unbiased_query definitions now
for i, l in enumerate(lines, 1):
    if 'def process_unbiased_query' in l:
        print(f'{i}: {l.strip()}')
print()
# Show what's right after the first one - find next def after 500
for i in range(500, min(600, len(lines))):
    s = lines[i-1].strip()
    if s.startswith('def ') or s.startswith('async def ') or s.startswith('@property'):
        print(f'{i}: {s[:110]}')
