import sys, os
sys.path.insert(0, os.path.abspath('.'))

lines = open('ai_chatbot.py', encoding='utf-8').read().split('\n')
for i, l in enumerate(lines, 1):
    if 'def _generate_octavian_guidance' in l:
        print(f'--- ai_chatbot guidance at {i} ---')
        for j in range(i, min(i+25, len(lines))):
            print(f'{j}: {lines[j-1]}')
        print()

print('='*70)
ml = open('main.py', encoding='utf-8').read().split('\n')
for i in range(320, 365):
    print(f'{i}: {ml[i-1]}')
