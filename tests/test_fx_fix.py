
import sys
import os
import asyncio
import pandas as pd

# Add the project root to sys.path
sys.path.append('/Users/aavibharucha/Documents/market_ai')

from financial_llm_engine import expand_query_intents, _fetch_live_data_for_tickers
from ai_chatbot import OctavianEnhancedChatbot

async def verify_fx_fix():
    print("--- Verifying FX Symbol Extraction ---")
    query = "what are some FX pairs with the highest potential?"
    intents, tickers, sectors = expand_query_intents(query)
    print(f"Extracted tickers: {tickers}")
    
    expected = ["EURUSD=X", "USDJPY=X", "GBPUSD=X"]
    all_found = all(any(e in t for t in tickers) for e in expected)
    if all_found:
        print(" SUCCESS: All FX pairs recognized and normalized.")
    else:
        print(f" FAILURE: Missing some FX pairs. Found: {tickers}")

    print("\n--- Verifying FX Data Fetching ---")
    live_data = _fetch_live_data_for_tickers(tickers)
    for t in tickers:
        if t in live_data or t.replace("=X", "") in live_data:
            print(f" SUCCESS: Fetched data for {t}: {live_data.get(t) or live_data.get(t.replace('=X', ''))}")
        else:
            print(f" FAILURE: Could not fetch data for {t}")

    print("\n--- Verifying Chart Generation ---")
    bot = OctavianEnhancedChatbot()
    # Mock the streamlit parts or just call the method
    # Since process_enhanced_query is async, we call it
    response = await bot.process_enhanced_query(query)
    charts = response.get('charts', [])
    print(f"Generated {len(charts)} charts.")
    if len(charts) > 0:
        print(" SUCCESS: Charts generated.")
        for c in charts:
            print(f"  Chart for: {c.get('symbol')} ({c.get('type')})")
    else:
        print(" FAILURE: No charts generated.")

if __name__ == "__main__":
    asyncio.run(verify_fx_fix())
