import re

src = open('ai_chatbot.py', encoding='utf-8').read()
lines = src.split('\n')

print('=== chart-related imports ===')
for i, l in enumerate(lines[:60]):
    if 'plotly' in l or 'go as' in l or 'make_subplots' in l or 'import go' in l:
        print(f'{i+1}: {l.strip()[:110]}')

print('\n=== _generate_enhanced_charts / _generate_intent_aware_charts definitions ===')
for i, l in enumerate(lines):
    if 'def _generate_enhanced_charts' in l or 'def _generate_intent_aware_charts' in l or 'def _generate_symbol_charts' in l:
        print(f'{i+1}: {l.strip()[:110]}')

print('\n=== how _generate_enhanced_charts builds figs (sample) ===')
idx = None
for i, l in enumerate(lines):
    if 'def _generate_enhanced_charts' in l:
        idx = i
        break
if idx:
    for j in range(idx, min(idx+40, len(lines))):
        print(f'{j+1}: {lines[j][:110]}')
