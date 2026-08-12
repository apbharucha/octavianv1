import sys, os, subprocess
sys.path.insert(0, os.path.abspath('.'))
r = subprocess.run(['grep', '-n', '_generate_octavian_guidance', 'ai_chatbot.py'], capture_output=True, text=True)
print(r.stdout)
print('='*70)
lines = open('ai_chatbot.py', encoding='utf-8').read().split('\n')
for i in range(5110, 5185):
    print(f'{i}: {lines[i-1]}')
