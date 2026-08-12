import sys, os, subprocess
sys.path.insert(0, os.path.abspath('.'))

print('=== 1. get_full_universe_sample in ticker_universe ===')
r = subprocess.run(['grep', '-n', 'def get_full_universe_sample\\|def get_full_universe\\|def get_all_tickers\\|def sample', 'ticker_universe.py'], capture_output=True, text=True)
print(r.stdout or '  NONE')

print('=== 2. get_crowding_engine in factor_crowding_engine ===')
r = subprocess.run(['grep', '-n', 'def get_crowding_engine\\|class FactorCrowdingEngine\\|def build_dashboard', 'factor_crowding_engine.py'], capture_output=True, text=True)
print(r.stdout or '  NONE')

print('=== 3. _generate_octavian_guidance count in ai_chatbot ===')
src = open('ai_chatbot.py', encoding='utf-8').read()
print('  count:', src.count('def _generate_octavian_guidance'))
print('  "Popular Stocks" present:', '"Popular Stocks"' in src)

print('=== 4. breaking_trades usage in main.py ===')
r = subprocess.run(['grep', '-n', 'breaking_trades\\|BreakingTrades\\|get_breaking_trades', 'main.py'], capture_output=True, text=True)
print(r.stdout[:1500] or '  NONE')

print('=== 5. ticker_universe sample helpers full ===')
r = subprocess.run(['grep', '-n', 'def ', 'ticker_universe.py'], capture_output=True, text=True)
print(r.stdout or '  NONE')
