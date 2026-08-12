import sys, os, re
sys.path.insert(0, os.path.abspath('.'))
src = open('ai_chatbot.py', encoding='utf-8').read()
lines = src.split('\n')

# Find process_enhanced_query definition
for i, l in enumerate(lines, 1):
    if 'def process_enhanced_query' in l:
        print(f'process_enhanced_query at {i}')
        # print following 90 lines
        for j in range(i, min(i+90, len(lines))):
            print(f'{j}: {lines[j-1]}')
        print('---')
        break
