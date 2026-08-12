import os, sys, time, cProfile, pstats, io
sys.path.insert(0, "/Users/aavibharucha/Documents/octavian")
os.chdir("/Users/aavibharucha/Documents/octavian")

TAB = sys.argv[1] if len(sys.argv) > 1 else "Paper Trading"

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

import yfinance as _yf
import pandas as _pd

at = AppTest.from_file("main.py", default_timeout=240)
mocks = [
    patch("data_sources.get_stock", side_effect=_mock_get_stock),
    patch("data_sources.get_fx", side_effect=_mock_get_stock),
    patch("data_sources.get_futures_proxy", side_effect=_mock_get_stock),
    patch("data_sources.get_fresh_quote", side_effect=_mock_get_stock),
    patch("data_sources.get_realtime_prices_batch", return_value={"^GSPC": (5000.0, 4950.0)}),
    patch("data_sources.get_realtime_price", return_value=(5000.0, 4950.0)),
    patch("financial_llm_engine.check_llm_connectivity", return_value=False),
    patch("financial_llm_engine._call_llm", return_value=""),
    patch("unbiased_market_analyzer.UnbiasedMarketAnalyzer.scan_entire_market", return_value=[]),
    patch("news_analysis_engine.NewsAnalysisEngine.fetch_and_process_news", return_value=[]),
    patch("news_analysis_engine.NewsAnalysisEngine.get_market_sentiment", return_value={"avg_sentiment": 0.1}),
    patch("news_analysis_engine.NewsAnalysisEngine.get_market_whispers", return_value=[]),
    patch("advanced_news_processor.AdvancedNewsProcessor.process_news_comprehensive", return_value=[]),
    patch("ticker_universe.get_ticker_universe", return_value=_FU()),
    patch("market_movers.fetch_market_movers", return_value=([], [])),
    patch("advanced_ml_engine.AdvancedEnsembleEngine.analyze_symbol_ensemble",
          return_value={"signal": "NEUTRAL", "confidence": 0.5, "bullish_prob": 0.5,
                        "bearish_prob": 0.5, "metrics": {}, "factors": []}),
    patch("macro_cross_asset_engine.MacroCrossAssetEngine.build_dashboard", return_value=None),
    patch("macro_cross_asset_engine.MacroDataLayer.get_close", return_value=None),
    patch("macro_cross_asset_engine.MacroDataLayer.get_momentum", return_value=None),
    patch("sec_13f_engine.SEC13FEngine.get_global_smart_money_flow",
          return_value={"flow": "neutral", "signal": "NEUTRAL", "summary": ""}),
    patch("market_scanner._scan_universe", return_value=_pd.DataFrame(columns=["symbol", "label", "score", "1D%"])),
    patch("market_scanner._scan_list", return_value=_pd.DataFrame(columns=["symbol", "label", "score", "1D%"])),
    patch.object(_yf, "download", return_value=_pd.DataFrame()),
]

for m in mocks:
    m.start()
try:
    at.run()
    for r in at.sidebar.radio:
        if r.label == "Navigation":
            r.set_value(TAB)
            break
    pr = cProfile.Profile()
    pr.enable()
    at.run()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(22)
    print(s.getvalue())
finally:
    for m in mocks:
        m.stop()
