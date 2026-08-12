import sys, os
sys.path.insert(0, os.path.abspath('.'))
src = open('tests/test_audit_regressions.py', encoding='utf-8').read()
lines = src.split('\n')

targets = ['test_ai_chatbot_uses_dynamic_examples', 'test_main_breaking_trades_is_dynamic', 'test_quant_portal_crowding_is_not_random']
for i, l in enumerate(lines, 1):
    for t in targets:
        if t in l:
            print(f'--- {t} at line {i} ---')
            for j in range(i, min(i+25, len(lines))):
                print(f'{j}: {lines[j-1]}')
            print()
