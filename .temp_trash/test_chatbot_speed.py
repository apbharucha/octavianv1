import asyncio, sys, time
sys.path.insert(0, '.')
from ai_chatbot import OctavianEnhancedChatbot

async def main():
    bot = OctavianEnhancedChatbot()
    q = "do an intensive market scan and pick 5 new and up and coming biotech stocks, present your reasoning why you think these stocks will rise"
    t0 = time.time()
    r = await asyncio.wait_for(bot.process_enhanced_query(q, user_id='test'), timeout=110)
    dt = time.time() - t0
    print(f'RESPONSE TIME: {dt:.1f}s')
    print(f'TEXT chars: {len(r.get("text", ""))}')
    print(f'CHARTS: {len(r.get("charts", []))}')
    print(f'SYMBOLS: {r.get("symbols")}')
    charts = r.get('charts', [])
    print(f'FIRST CHART TYPE: {charts[0].get("type") if charts else "none"}')
    print('TEXT PREVIEW:', r.get('text', '')[:300].replace('\n', ' | '))

asyncio.run(main())
