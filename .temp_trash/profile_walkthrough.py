import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/Users/aavibharucha/Documents/octavian")
os.chdir("/Users/aavibharucha/Documents/octavian")

from unittest.mock import patch
from streamlit.testing.v1 import AppTest

def _mock_df(n=90):
    import numpy as np, pandas as pd
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.8, n))
    return pd.DataFrame({
        "Open": close*0.99, "High": close*1.02, "Low": close*0.98,
        "Close": close, "Adj Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n),
    }, index=idx)

_DF = None
def _mock_get_stock(symbol, *a, **k):
    global _DF
    if _DF is None:
        _DF = _mock_df()
    return _DF.copy()

class _FakeUniverse:
    def get_all_stocks(self): return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD"]
    def get_etfs(self): return ["SPY", "QQQ", "IWM"]
    def get_crypto(self): return ["BTC-USD", "ETH-USD"]
    def get_forex(self): return ["EURUSD=X", "GBPUSD=X"]
    def get_futures(self): return ["ES=F", "CL=F"]
    def get_all_sectors(self): return {"technology": ["AAPL"], "energy": ["XOM"]}
    def get_sector_tickers(self, s): return self.get_all_sectors().get(s, [])
    def get_sector_aliases(self): return {"tech": "technology"}
    def get_all_tickers_flat(self): return self.get_all_stocks()+self.get_etfs()+self.get_crypto()
    def get_all_tickers(self): return self.get_all_tickers_flat()
    def get_full_universe_sample(self, n=40): return self.get_all_tickers_flat()[:n]
    def get_random_sample(self, n=40): return self.get_full_universe_sample(n)
    def get_known_ticker_set(self): return set(self.get_all_tickers_flat())
    def get_universe_stats(self): return {"total": 10, "assets": {}}
    def get_symbol_sector(self, s): return ""
    def add_ticker(self, s, sector=None): pass
    def refresh_if_needed(self): return False

t0 = time.time()
at = AppTest.from_file("/Users/aavibharucha/Documents/octavian/main.py", default_timeout=240)
print(f"AppTest.from_file: {time.time()-t0:.1f}s")

t0 = time.time()
import pandas as _pd
import yfinance as _yf
with patch("data_sources.get_stock", side_effect=_mock_get_stock), \
     patch("data_sources.get_fx", side_effect=_mock_get_stock), \
     patch("data_sources.get_futures_proxy", side_effect=_mock_get_stock), \
     patch("data_sources.get_fresh_quote", side_effect=_mock_get_stock), \
     patch("financial_llm_engine.check_llm_connectivity", return_value=False), \
     patch("financial_llm_engine._call_llm", return_value=""), \
     patch("unbiased_market_analyzer.UnbiasedMarketAnalyzer.scan_entire_market", return_value=[]), \
     patch("news_analysis_engine.NewsAnalysisEngine.fetch_and_process_news", return_value=[]), \
     patch("news_analysis_engine.NewsAnalysisEngine.get_market_sentiment", return_value={"avg_sentiment": 0.1}), \
     patch("news_analysis_engine.NewsAnalysisEngine.get_market_whispers", return_value=[]), \
     patch("advanced_news_processor.AdvancedNewsProcessor.process_news_comprehensive", return_value=[]), \
     patch("ticker_universe.get_ticker_universe", return_value=_FakeUniverse()), \
     patch.object(_yf, "download", return_value=_pd.DataFrame()):
    t0 = time.time()
    at.run()
    print(f"at.run() initial (Dashboard render): {time.time()-t0:.1f}s")
    t0 = time.time()
    for r in at.sidebar.radio:
        if r.label == "Navigation":
            r.set_value("Paper Trading")
            break
    at.run()
    print(f"at.run() Paper Trading: {time.time()-t0:.1f}s")
    print("exceptions:", [str(e.value) for e in at.exception])
