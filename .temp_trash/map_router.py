import sys, os, re
sys.path.insert(0, os.path.abspath('.'))
lines = open('main.py', encoding='utf-8').read().split('\n')
nav = ["Dashboard","Watchlist","Market Scanner","Symbol Analysis","Chart Analysis","Intelligence Center","Market Heartbeat","Financial Model Generator","Daily Briefing","Quant Portal","Strategy Research Lab","Paper Trading","Simulation Hub","Spreadsheet Generator","Trader Profile","Settings & Analytics"]
for i, l in enumerate(lines, 1):
    s = l.strip()
    for n in nav:
        if s == f'if selection == "{n}":':
            print(f'{i}: {s}')
            # print next 3 lines
            for j in range(i, min(i+4, len(lines))):
                print(f'    {j}: {lines[j-1].strip()[:100]}')
            print()
