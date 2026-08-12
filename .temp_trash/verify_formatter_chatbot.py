import sys, os, time, asyncio, io
sys.path.insert(0, os.path.abspath('.'))

# 1) Formatter singleton import check
try:
    from response_formatter import get_response_formatter, ResponseFormatter
    f = get_response_formatter()
    print('FORMATTER IMPORT OK:', type(f).__name__)
except Exception as e:
    print('FORMATTER IMPORT FAIL:', repr(e))
    sys.exit(1)

# 2) ai_chatbot import check
try:
    from ai_chatbot import OctavianEnhancedChatbot
    print('AI_CHATBOT IMPORT OK')
except Exception as e:
    print('AI_CHATBOT IMPORT FAIL:', repr(e))
    sys.exit(1)

# 3) Live speed + charts test
async def main():
    bot = OctavianEnhancedChatbot()
    q = ("do an intensive market scan and pick 5 new and up and coming biotech stocks, "
         "present your reasoning why you think these stocks will rise")
    t0 = time.time()
    r = await bot.process_enhanced_query(q, user_id='test')
    dt = time.time() - t0
    print(f'RESPONSE TIME: {dt:.1f}s')
    print(f'TEXT chars: {len(r.get("text", ""))}')
    print(f'CHARTS: {len(r.get("charts", []))}')
    print(f'SYMBOLS: {r.get("symbols")}')
    if r.get('charts'):
        print(f'CHART KEYS: {list(r["charts"][0].keys())}')
        for c in r['charts'][:3]:
            print('  chart:', c.get('type'), '| title:', str(c.get('title'))[:60])
    else:
        print('NO CHARTS GENERATED')

asyncio.run(main())
