
import asyncio
import pandas as pd
from data_sources import get_stock, get_fx

async def test_vix():
    print("Testing VIX correction...")
    # Mocking today's date as 2026-02-09 for the test if possible
    # But since the code checks df.index[-1], we can check what it returns
    df = get_stock("^VIX", period="5d")
    if not df.empty:
        last_date = df.index[-1].strftime('%Y-%m-%d')
        last_price = df['Close'].iloc[-1]
        print(f"VIX Last Date: {last_date}, Last Price: {last_price}")
        
        if last_date == '2026-02-09':
            prev_price = df['Close'].iloc[-2]
            change = (last_price / prev_price - 1) * 100
            print(f"VIX Change: {change:.2f}%")
            if abs(change + 2.25) < 0.1:
                print("[OK] VIX correction verified!")
            else:
                print(f"[X] VIX correction failed. Expected ~-2.25%, got {change:.2f}%")
        else:
            print("Skipping VIX verification (date mismatch).")

async def test_fx_inversion():
    print("\nTesting FX inversion (USD/EUR)...")
    # USD/EUR usually doesn't exist on Yahoo, EUR/USD does.
    df = get_fx("USD/EUR", period="5d")
    if not df.empty:
        print("[OK] USD/EUR data found via inversion!")
        print(df.tail(2))
    else:
        print("[X] USD/EUR data NOT found.")

    print("\nTesting USD/JPY...")
    df = get_fx("USD/JPY", period="5d")
    if not df.empty:
        print("[OK] USD/JPY data found!")
        print(df.tail(2))
    else:
        print("[X] USD/JPY data NOT found.")

if __name__ == "__main__":
    asyncio.run(test_vix())
    asyncio.run(test_fx_inversion())
