import sys, os, time, asyncio
sys.path.insert(0, os.path.abspath('.'))

from ai_chatbot import OctavianEnhancedChatbot

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
    for c in r.get('charts', [])[:5]:
        print(f'  chart: {c.get("type")} | {c.get("symbol")} | {str(c.get("title"))[:50]}')
    print()
    print('=== TEXT SAMPLE ===')
    t = r.get('text', '')
    print(t[:1200])

asyncio.run(main())
