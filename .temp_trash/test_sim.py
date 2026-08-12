
import asyncio
from market_simulation_engine import MarketSimulationEngine
import pandas as pd
from datetime import datetime

async def test_simulation():
    print("Starting simulation test...")
    engine = MarketSimulationEngine()
    
    # Mock some data for USD/EUR and USD/JPY to see if they are handled
    print("Running daily simulation...")
    try:
        result = await engine.run_daily_simulation()
        print(f"Simulation completed successfully!")
        print(f"Result ID: {result.simulation_id}")
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Market Regime: {result.market_regime.value}")
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simulation())
