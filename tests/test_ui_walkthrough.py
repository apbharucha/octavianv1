"""
End-to-end Streamlit UI walkthrough of every main.py tab.

External providers (price data, news, LLM) are mocked so the walkthrough
runs offline, deterministically, and quickly. Each tab must render without
raising an exception — this catches broken imports, stale API calls, and
wiring regressions across the whole app.
"""
import os
import sys

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contextlib import ExitStack
from unittest.mock import patch  # noqa: E402

NAV = [
    "Dashboard", "Watchlist", "Market Scanner", "Symbol Analysis",
    "Chart Analysis", "Intelligence Center", "Market Heartbeat",
    "Dark Pool Intelligence",
    "Institutional 13F & SEC Filings",
    "Financial Model Generator", "Target Probability",
    "Presentation Generator", "Document Analyzer", "Comparative Analysis",
    "Daily Briefing", "Quant Portal", "Quant Modeling Lab",
    "Strategy Research Lab", "Portfolio Analyzer", "Position Optimizer",
    "Paper Trading", "Simulation Hub", "Spreadsheet Generator",
    "Trader Profile", "Notification Settings", "Settings & Analytics",
    "Terms of Service",
]


def _mock_df(n=90):
    import numpy as np
    import pandas as pd
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.8, n))
    return pd.DataFrame({
        "Open": close * 0.99, "High": close * 1.02, "Low": close * 0.98,
        "Close": close, "Adj Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n),
    }, index=idx)


_DF = None


def _mock_get_stock(symbol, *a, **k):
    global _DF
    if _DF is None:
        _DF = _mock_df()
    return _DF.copy()


class _FakeUniverse:
    """Deterministic, offline-safe stand-in for the live ticker universe."""

    def get_all_stocks(self):
        return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD"]

    def get_etfs(self):
        return ["SPY", "QQQ", "IWM"]

    def get_crypto(self):
        return ["BTC-USD", "ETH-USD"]

    def get_forex(self):
        return ["EURUSD=X", "GBPUSD=X"]

    def get_futures(self):
        return ["ES=F", "CL=F"]

    def get_all_sectors(self):
        return {"technology": ["AAPL"], "energy": ["XOM"]}

    def get_sector_tickers(self, sector):
        return self.get_all_sectors().get(sector, [])

    def get_sector_aliases(self):
        return {"tech": "technology", "energy": "energy"}

    def get_all_tickers_flat(self):
        return self.get_all_stocks() + self.get_etfs() + self.get_crypto()

    def get_all_tickers(self):
        return self.get_all_tickers_flat()

    def get_full_universe_sample(self, n=40):
        flat = self.get_all_tickers_flat()
        return flat[:n]

    def get_random_sample(self, n=40):
        return self.get_full_universe_sample(n)

    def get_known_ticker_set(self):
        return set(self.get_all_tickers_flat())

    def get_universe_stats(self):
        return {"total": len(self.get_all_tickers_flat()), "assets": {}}

    def get_symbol_sector(self, symbol):
        for sec, syms in self.get_all_sectors().items():
            if symbol in syms:
                return sec
        return ""

    def add_ticker(self, symbol, sector=None):
        pass

    def refresh_if_needed(self):
        return False


def _build_mocks():
    """Return a list of patch objects that keep the UI walkthrough offline
    and fast (heavy ML / network engines stubbed at the source)."""
    import yfinance as _yf
    import pandas as _pd
    return [
        patch("data_sources.get_stock", side_effect=_mock_get_stock),
        patch("data_sources.get_fx", side_effect=_mock_get_stock),
        patch("data_sources.get_futures_proxy", side_effect=_mock_get_stock),
        patch("data_sources.get_fresh_quote", side_effect=_mock_get_stock),
        patch("data_sources.get_realtime_prices_batch",
              return_value={"^GSPC": (5000.0, 4950.0)}),
        patch("data_sources.get_realtime_price", return_value=(5000.0, 4950.0)),
        patch("financial_llm_engine.check_llm_connectivity", return_value=False),
        patch("financial_llm_engine._call_llm", return_value=""),
        patch("unbiased_market_analyzer.UnbiasedMarketAnalyzer.scan_entire_market",
              return_value=[]),
        patch("news_analysis_engine.NewsAnalysisEngine.fetch_and_process_news",
              return_value=[]),
        patch("news_analysis_engine.NewsAnalysisEngine.get_market_sentiment",
              return_value={"avg_sentiment": 0.1, "article_count": 0, "sources": {}}),
        patch("news_analysis_engine.NewsAnalysisEngine.get_market_whispers",
              return_value=[]),
        patch("advanced_news_processor.AdvancedNewsProcessor.process_news_comprehensive",
              return_value=[]),
        patch("ticker_universe.get_ticker_universe", return_value=_FakeUniverse()),
        patch("market_movers.fetch_market_movers", return_value=([], [])),
        patch("market_scanner._scan_universe",
              return_value=_pd.DataFrame(columns=["symbol", "label", "score", "1D%"])),
        patch("market_scanner._scan_list",
              return_value=_pd.DataFrame(columns=["symbol", "label", "score", "1D%"])),
        patch("advanced_ml_engine.AdvancedEnsembleEngine.analyze_symbol_ensemble",
              return_value={"signal": "NEUTRAL", "confidence": 0.5, "bullish_prob": 0.5,
                            "bearish_prob": 0.5, "metrics": {}, "factors": []}),
        # Target Probability page fetches historical data at render via
        # quant_ensemble + options engine; keep it offline-fast.
        patch("quant_ensemble_model.QuantEnsembleModel.predict",
              return_value=type("Sig", (), {
                  "direction": "NEUTRAL", "probability": 0.5, "confidence": 0.0,
                  "expected_return": 0.0})()),
        patch("advanced_ml_engine.AdvancedEnsembleEngine.analyze_symbol_ensemble",
              return_value={"signal": "NEUTRAL", "confidence": 0.5,
                            "final_predicted_price": 0.0, "metrics": {}, "factors": []}),
        # Quant Modeling Lab — factor crowding + narrative dislocation engines
        # short-circuit so the walkthrough stays offline and deterministic.
        patch("factor_crowding_engine.CrowdingDataLayer.get_close", return_value=None),
        patch("factor_crowding_engine.CrowdingDataLayer.get_momentum", return_value=None),
        patch("narrative_dislocation_engine.PriceDataLayer.get_close", return_value=None),
        patch("macro_cross_asset_engine.MacroCrossAssetEngine.build_dashboard",
              return_value=None),
        patch("macro_cross_asset_engine.MacroDataLayer.get_close",
              return_value=None),
        patch("macro_cross_asset_engine.MacroDataLayer.get_momentum",
              return_value=None),
        patch("sec_13f_engine.SEC13FEngine.get_global_smart_money_flow",
              return_value={
                  "top_inflows": [("AAPL", 1.5e9), ("MSFT", 1.2e9)],
                  "top_outflows": [("TSLA", -0.8e9)],
                  "net_flow_map": {"AAPL": 1.5e9, "MSFT": 1.2e9, "TSLA": -0.8e9},
                  "flow": "neutral", "signal": "NEUTRAL", "summary": "",
              }),
        # 13F tab calls fetch_latest_filings at render (SEC EDGAR network)
        patch("sec_13f_engine.SEC13FEngine.fetch_latest_filings",
              return_value=[]),
        # scan_sectors / futures_rank hit yf.download directly
        patch.object(_yf, "download", return_value=_pd.DataFrame()),
        # Dark Pool Intelligence binds get_stock/get_vix/get_realtime_price at
        # module import time, so patching data_sources.* never reaches it.
        # Mock its own bindings + force the no-key (modeled) FINRA path so the
        # whole 15-tab terminal renders offline.
        patch("dark_pool_engine.get_stock", side_effect=_mock_get_stock),
        patch("dark_pool_engine.get_vix", side_effect=_mock_get_stock),
        patch("dark_pool_engine.get_realtime_price", return_value=(225.0, 223.0)),
        patch("dark_pool_engine.DarkPoolEngine.get_finra_api_key", return_value=""),
    ]


@pytest.mark.parametrize("tab", NAV)
def test_tab_renders_without_exception(tab):
    at = AppTest.from_file(os.path.join(ROOT, "main.py"), default_timeout=240)
    with ExitStack() as stack:
        for m in _build_mocks():
            stack.enter_context(m)
        at.run()
        radios = list(at.sidebar.radio)
        for r in radios:
            if r.label == "Navigation":
                r.set_value(tab)
                at.run()
                break
        exc = [str(e.value) for e in at.exception]
        assert not exc, f"Tab {tab!r} raised: {exc[:2]}"
