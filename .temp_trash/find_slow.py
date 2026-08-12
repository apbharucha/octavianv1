import re

src = open('ai_chatbot.py', encoding='utf-8').read()
lines = src.split('\n')

print('=== "Processed" message ===')
for i, l in enumerate(lines):
    if 'Processed' in l or 'processed' in l:
        print(f'{i+1}: {l.strip()[:120]}')

print('\n=== analysis complete message ===')
for i, l in enumerate(lines):
    if 'analysis complete' in l:
        print(f'{i+1}: {l.strip()[:120]}')

print('\n=== where _analyze_symbol loops happen (sequential for loops calling _analyze) ===')
for i, l in enumerate(lines):
    if ('for symbol' in l or 'for sym' in l) and ('_analyze' in src[max(0,i-5):i+5] or 'analyze' in l):
        print(f'{i+1}: {l.strip()[:120]}')

print('\n=== asyncio.gather usage ===')
for i, l in enumerate(lines):
    if 'asyncio.gather' in l or 'ThreadPoolExecutor' in l:
        print(f'{i+1}: {l.strip()[:120]}')

print('\n=== total lines ===')
print(len(lines))
