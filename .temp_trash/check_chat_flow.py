import re

src = open('ai_chatbot.py', encoding='utf-8').read()
lines = src.split('\n')

print('=== show_octavian_chatbot invocation of processing ===')
for i, l in enumerate(lines):
    if 'process_enhanced_query' in l or 'process_unbiased_query' in l or 'run_analysis' in l:
        print(f'{i+1}: {l.strip()[:130]}')

print('\n=== lines 5760-5840 (chat processing entry) ===')
for i in range(5759, min(5840, len(lines))):
    print(f'{i+1}: {lines[i][:130]}')
