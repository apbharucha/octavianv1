
import sys
import os
import asyncio
import pandas as pd

# Add the project root to sys.path
sys.path.append('/Users/aavibharucha/Documents/market_ai')

from financial_llm_engine import expand_query_intents, generate_financial_analysis
from ai_chatbot import OctavianEnhancedChatbot, TimeframeScope

async def verify_dynamic_system():
    bot = OctavianEnhancedChatbot()
    
    print("--- Verifying Stopwords & False Positives ---")
    query1 = "what are some FX pairs with the highest potential?"
    intents1, tickers1, sectors1 = expand_query_intents(query1)
    print(f"Query: {query1}")
    print(f"Extracted tickers: {tickers1}")
    if "WITH" not in tickers1:
        print(" SUCCESS: 'WITH' is no longer treated as a ticker.")
    else:
        print(" FAILURE: 'WITH' was found as a ticker.")

    print("\n--- Verifying Dynamic Sector Discovery ---")
    query2 = "what is the outlook for the uranium sector and nuclear power?"
    intents2, tickers2, sectors2 = expand_query_intents(query2)
    print(f"Query: {query2}")
    print(f"Detected sectors: {sectors2}")
    print(f"Injected tickers: {tickers2}")
    if "uranium" in sectors2 and any(t in tickers2 for t in ["CCJ", "UEC", "UUUU"]):
        print(" SUCCESS: Dynamic sector 'uranium' detected and tickers injected.")
    else:
        print(" FAILURE: Could not detect 'uranium' or find common uranium tickers.")

    print("\n--- Verifying Dynamic Reasoning ---")
    # Ticker not in old hardcoded list, e.g., PLTR
    analysis = generate_financial_analysis("tell me about PLTR")
    print("Analysis Snippet:")
    print(analysis[:300] + "...")
    if "PLTR" in analysis and "relative strength" in analysis.lower() or "momentum" in analysis.lower() or "price action" in analysis.lower():
        print(" SUCCESS: Analysis generated for PLTR with dynamic technical commentary.")
    else:
        print(" FAILURE: Analysis did not look dynamic or failed.")

    print("\n--- Verifying Unbiased Market Briefing ---")
    brief = bot._generate_market_wide_brief(TimeframeScope.SWING)
    print("Market Brief:")
    print(brief)
    if "Unbiased Intelligence" in brief and "assets" in brief:
        print(" SUCCESS: Unbiased market briefing generated dynamically.")
    else:
        print(" FAILURE: Briefing failed or looked hardcoded.")

if __name__ == "__main__":
    asyncio.run(verify_dynamic_system())
