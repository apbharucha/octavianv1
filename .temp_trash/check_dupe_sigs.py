import sys, os
sys.path.insert(0, os.path.abspath('.'))
lines = open('ai_chatbot.py', encoding='utf-8').read().split('\n')
pairs = [
    ('_get_real_time_data', [2333, 5419]),
    ('_create_advanced_price_chart', [2410, 5496]),
    ('_create_prediction_chart', [2774, 5666]),
]
for name, locs in pairs:
    for loc in locs:
        print(f'--- {name} at {loc} ---')
        for j in range(loc, loc+6):
            print(f'   {j}: {lines[j-1].strip()[:110]}')
        print()
