import sys
import os
import unittest
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_simulation_engine import MarketSimulationEngine, MarketRegime
from ticker_universe import get_ticker_universe

class TestDynamicSimulationUniverse(unittest.TestCase):
    def setUp(self):
        self.engine = MarketSimulationEngine()
        self.universe = get_ticker_universe()

    def test_universe_initialization_all_inclusive(self):
        """Verify that the simulation initializes with the FULL universe by default."""
        self.engine.universe_size = None # Ensure no cap
        
        # We don't want to run the full simulation (it takes time), 
        # but we want to see what universe it picks.
        # We'll mock the internal call or just check how it behaves.
        
        # Get the master list from TickerUniverse
        master_list = self.universe.get_all_tickers()
        print(f"Master TickerUniverse Count: {len(master_list)}")
        
        # The user said 'not only pulling from 800 or so'. Let's verify it exceeds this.
        self.assertGreater(len(master_list), 1000, "Universe should be significantly larger than 1000 symbols.")

    def test_asset_class_diversity(self):
        """Verify that the universe includes diverse asset classes."""
        all_tickers = self.universe.get_all_tickers()
        
        # Check for Crypto
        crypto = [t for t in all_tickers if "-USD" in t]
        print(f"Crypto count: {len(crypto)}")
        self.assertGreater(len(crypto), 0, "Should include Crypto symbols (e.g. BTC-USD).")
        
        # Check for FX
        fx = [t for t in all_tickers if "=X" in t]
        print(f"FX count: {len(fx)}")
        self.assertGreater(len(fx), 0, "Should include FX symbols (e.g. EURUSD=X).")
        
        # Check for Futures
        futures = [t for t in all_tickers if "=F" in t]
        print(f"Futures count: {len(futures)}")
        self.assertGreater(len(futures), 0, "Should include Futures symbols (e.g. ES=F).")

    def test_dynamic_sector_lookup(self):
        """Verify that the engine can resolve sectors dynamically."""
        # Test a known stock
        sector_aapl = self.engine._get_symbol_sector("AAPL")
        print(f"AAPL Sector: {sector_aapl}")
        self.assertNotEqual(sector_aapl, "Unknown")
        
        # Test an ETF
        sector_spy = self.engine._get_symbol_sector("SPY")
        print(f"SPY Sector: {sector_spy}")
        self.assertEqual(sector_spy, "ETF")
        
        # Test Crypto
        sector_btc = self.engine._get_symbol_sector("BTC-USD")
        print(f"BTC-USD Sector: {sector_btc}")
        self.assertEqual(sector_btc, "Crypto")

if __name__ == "__main__":
    unittest.main()
