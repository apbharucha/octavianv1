import os, sys, time
sys.path.insert(0, "/Users/aavibharucha/Documents/octavian")
os.chdir("/Users/aavibharucha/Documents/octavian")

from unittest.mock import patch
from streamlit.testing.v1 import AppTest

def _mock_df(n=90):
    import numpy as np, pandas as pd
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.8, n))
    return pd.DataFrame({"Open": close*0.99, "High": close*1.02, "Low": close*0.98,
        "Close": close, "Adj Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n)}, index=idx)
_DF = None
def _mock_get_stock(symbol, *a, **k):
    global _DF
    if _DF is None: _DF = _mock_df()
    return _DF.copy()

class _FU:
    def get_all_stocks(self): return ["AAPL"]
    def get_etfs(self): return []
    def get_crypto(self): return []
    def get_forex(self): return []
    def get_futures(self): return []
    def get_all_sectors(self): return {"technology": ["AAPL"]}
    def get_sector_tickers(self, s): return self.get_all_sectors().get(s, [])
    def get_sector_aliases(self): return {}
    def get_all_tickers_flat(self): return ["AAPL"]
    def get_all_tickers(self): return ["AAPL"]
    def get_full_universe_sample(self, n=40): return ["AAPL"]
    def get_random_sample(self, n=40): return ["AAPL"]
    def get_known_ticker_set(self): return {"AAPL"}
    def get_universe_stats(self): return {"total": 1, "assets": {}}
    def get_symbol_sector(self, s): return ""
    def add_ticker(self, s, sector=None): pass
    def refresh_if_needed(self): return False

t0 = time.time()
at = AppTest.from_file("main.py", default_timeout=240)
print(f"from_file: {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
import pandas as _pd
import yfinance as _yf
with patch("data_sources.get_stock", side_effect=_mock_get_stock), \
     patch("data_sources.get_fx", side_effect=_mock_get_stock), \
     patch("data_sources.get_futures_proxy", side_effect=_mock_get_stock), \
     patch("data_sources.get_fresh_quote", side_effect=_mock_get_stock), \
     patch("data_sources.get_realtime_prices_batch", return_value={"^GSPC": (5000.0, 4950.0)}), \
     patch("data_sources.get_realtime_price", return_value=(5000.0, 4950.0)), \
     patch("financial_llm_engine.check_llm_connectivity", return_value=False), \
     patch("financial_llm_engine._call_llm", return_value=""), \
     patch("unbiased_market_analyzer.UnbiasedMarketAnalyzer.scan_entire_market", return_value=[]), \
     patch("news_analysis_engine.NewsAnalysisEngine.fetch_and_process_news", return_value=[]), \
     patch("news_analysis_engine.NewsAnalysisEngine.get_market_sentiment", return_value={"avg_sentiment": 0.1}), \
     patch("news_analysis_engine.NewsAnalysisEngine.get_market_whispers", return_value=[]), \
     patch("advanced_news_processor.AdvancedNewsProcessor.process_news_comprehensive", return_value=[]), \
     patch("ticker_universe.get_ticker_universe", return_value=_FU()), \
     patch("market_movers.fetch_market_movers", return_value=([], [])), \
     patch.object(_yf, "download", return_value=_pd.DataFrame()):
    at.run()
    print(f"initial at.run(): {time.time()-t0:.1f}s, exceptions={[str(e.value) for e in at.exception]}", flush=True)
    # Now switch tab
    t1 = time.time()
    for r in at.sidebar.radio:
        if r.label == "Navigation":
            r.set_value("Trader Profile")
            break
    at.run()
    print(f"tab-switch run: {time.time()-t1:.1f}s, exceptions={[str(e.value) for e in at.exception]}", flush=True)
