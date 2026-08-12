import sys, os
sys.path.insert(0, os.path.abspath('.'))
lines = open('ai_chatbot.py', encoding='utf-8').read().split('\n')
for i in range(78, 170):
    print(f'{i}: {lines[i-1]}')
