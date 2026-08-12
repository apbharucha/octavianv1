import sys, os, re
sys.path.insert(0, os.path.abspath('.'))
src = open('ai_chatbot.py', encoding='utf-8').read()
lines = src.split('\n')

# List all methods in OctavianEnhancedChatbot
in_class = False
methods = []
for i, l in enumerate(lines, 1):
    s = l.strip()
    if s.startswith('class OctavianEnhancedChatbot'):
        in_class = True
        continue
    if in_class:
        if s.startswith('class '):
            break
        if s.startswith('def '):
            methods.append((i, s[:110]))
print('=== OctavianEnhancedChatbot methods ===')
for i, m in methods:
    print(f'{i}: {m}')

print()
print('=== _call_llm refs in ai_chatbot ===')
for i, l in enumerate(lines, 1):
    if '_call_llm' in l or 'lm_studio' in l.lower() or 'financial_llm_engine' in l:
        print(f'{i}: {l.strip()[:120]}')
