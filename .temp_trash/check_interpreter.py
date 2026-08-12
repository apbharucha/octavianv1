import sys, os, subprocess
sys.path.insert(0, os.path.abspath('.'))
src = open('ai_chatbot.py', encoding='utf-8').read()
lines = src.split('\n')
for i, l in enumerate(lines, 1):
    s = l.strip()
    if 'interpreter' in s.lower() and ('def ' in s or 'lm_studio' in s.lower() or '_call_llm' in s):
        print(f'{i}: {s[:130]}')

print()
print('=== test file full ===')
print(open('tests/test_ai_chatbot.py', encoding='utf-8').read()[:3000])
