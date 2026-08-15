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
    "Strategy Research Lab", "Algorithm Builder",
    "Portfolio Analyzer", "Position Optimizer",
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


def test_dark_pool_modes_are_actually_different():
    """Basic / Advanced / Institutional must render visibly different content.

    Guards the tiered-viewing design: Basic is a plain-language digest (no
    z-scores / evidence / raw tables), Advanced adds the statistics, and
    Institutional adds provenance + raw outputs. If this test fails the modes
    have collapsed back into looking identical.
    """
    at = AppTest.from_file(os.path.join(ROOT, "main.py"), default_timeout=240)
    with ExitStack() as stack:
        for m in _build_mocks():
            stack.enter_context(m)
        at.run()
        for r in at.sidebar.radio:
            if r.label == "Navigation":
                r.set_value("Dark Pool Intelligence")
                at.run()
                break

        mode = at.radio(key="dp_mode")
        assert mode is not None, "Viewing Mode radio not found"
        assert mode.value == "Basic"

        def _text():
            md = " ".join(m.value for m in at.markdown)
            cap = " ".join(c.value for c in at.caption)
            return md + " " + cap

        # Basic: plain-language digest present; statistics hidden
        assert "Plain-language summary" in _text()
        assert "Z-scores (std dev" not in _text()
        assert "Evidence for inference" not in _text()
        assert "Sector / Pressure Matrix" not in _text()
        assert "**Provenance**" not in _text()

        # Advanced: statistics appear; raw provenance still hidden
        mode.set_value("Advanced")
        at.run()
        mode = at.radio(key="dp_mode")
        assert mode.value == "Advanced"
        t = _text()
        assert "Plain-language summary" not in t
        assert "Z-scores (std dev" in t
        assert "Evidence for inference" in t
        assert "Sector / Pressure Matrix" in t
        assert "**Provenance**" not in t

        # Institutional: provenance + raw outputs appear
        mode.set_value("Institutional")
        at.run()
        t = _text()
        assert "Z-scores (std dev" in t
        assert "Evidence for inference" in t
        assert "Sector / Pressure Matrix" in t
        assert "**Provenance**" in t


def _mock_get_stock_long(symbol, *a, **k):
    """300-bar deterministic OHLCV so Algorithm Builder's train/test split
    (needs >= 60 train + >= 30 test bars) actually produces healthy results."""
    import numpy as np
    import pandas as pd
    n = 300
    idx = pd.bdate_range(end=pd.Timestamp("2026-08-01"), periods=n)
    rng = np.random.default_rng(7)
    close = 100 * np.cumprod(1 + rng.normal(0.0006, 0.01, n))
    return pd.DataFrame({
        "Open": close * 0.995, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Adj Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n),
    }, index=idx)


def test_algorithm_builder_build_click_does_not_raise():
    """Regression: clicking 'Build algorithms' crashed with

        TypeError: build_request_signature() got an unexpected keyword
        argument 'locked_params'

    because the UI passed the execution-lock dict under `locked_params` while
    build_request_signature names that parameter `locked`. The click must
    complete and render results with no exception."""
    import algorithm_builder_engine as abe

    at = AppTest.from_file(os.path.join(ROOT, "main.py"), default_timeout=240)
    with ExitStack() as stack:
        for m in _build_mocks():
            stack.enter_context(m)
        # build_algorithms binds data_fn as a *default argument value*, so
        # swapping the module-level binding is not enough — patch __defaults__.
        _defaults = list(abe.build_algorithms.__defaults__)
        _defaults[11] = _mock_get_stock_long  # data_fn
        stack.enter_context(patch.object(abe.build_algorithms, "__defaults__",
                                         tuple(_defaults)))
        at.run()
        for r in at.sidebar.radio:
            if r.label == "Navigation":
                r.set_value("Algorithm Builder")
                at.run()
                break
        btns = [b for b in at.button if b.label == "Build algorithms"]
        assert btns, "Build algorithms button not found"
        btns[0].click()
        at.run()
        exc = [str(e.value) for e in at.exception]
        assert not exc, f"Algorithm Builder build raised: {exc[:2]}"
        # Results rendered: summary line mentions the built algorithms
        md = " ".join(m.value for m in at.markdown)
        assert "Built" in md and "algorithm" in md, (
            f"expected build summary, got markdown: {md[:200]}")


def test_algorithm_builder_deploy_button_opens_panel():
    """Regression: the 'Test in paper trading' button on a built algorithm
    must open the deploy panel (new account / existing portfolio) with no
    exception, and 'Create account & deploy' must deploy the strategy."""
    import algorithm_builder_engine as abe

    class _FakeAccount:
        def __init__(self, aid, name, balance):
            self.account_id = aid
            self.account_name = name
            self.current_balance = balance

    fake_system = type("FakePT", (), {})()
    # the existing account already runs a strategy -> deploying must ask to override
    fake_system.deployed = {"PT_U_1": {"name": "Old Strategy", "metrics": {"sharpe": 0.7}}}
    fake_system.accounts = [_FakeAccount("PT_U_1", "My Portfolio", 100000.0)]

    def _create_account(user_id, account_name, initial_balance=100000.0):
        acc = _FakeAccount(f"PT_{user_id}_NEW", account_name, initial_balance)
        fake_system.accounts.append(acc)
        return acc

    def _deploy(account_id, spec):
        fake_system.deployed[account_id] = spec
        return True

    def _get_deployed(account_id):
        return fake_system.deployed.get(account_id)

    def _list_accounts(user_id):
        return list(fake_system.accounts)

    fake_system.create_account = _create_account
    fake_system.deploy_strategy = _deploy
    fake_system.get_deployed_strategy = _get_deployed
    fake_system.list_accounts = _list_accounts

    at = AppTest.from_file(os.path.join(ROOT, "main.py"), default_timeout=240)
    with ExitStack() as stack:
        for m in _build_mocks():
            stack.enter_context(m)
        _defaults = list(abe.build_algorithms.__defaults__)
        _defaults[11] = _mock_get_stock_long  # data_fn
        stack.enter_context(patch.object(abe.build_algorithms, "__defaults__",
                                         tuple(_defaults)))
        # The deploy panel imports get_paper_trading_system inside the function
        stack.enter_context(patch("paper_trading_system.get_paper_trading_system",
                                  return_value=fake_system))
        at.run()
        for r in at.sidebar.radio:
            if r.label == "Navigation":
                r.set_value("Algorithm Builder")
                at.run()
                break
        btns = [b for b in at.button if b.label == "Build algorithms"]
        assert btns, "Build algorithms button not found"
        btns[0].click()
        at.run()
        exc = [str(e.value) for e in at.exception]
        assert not exc, f"Algorithm Builder build raised: {exc[:2]}"
        # Click 'Test in paper trading' on the first algorithm
        deploy_btns = [b for b in at.button if b.label == "Test in paper trading"]
        assert deploy_btns, "'Test in paper trading' button not found"
        deploy_btns[0].click()
        at.run()
        exc = [str(e.value) for e in at.exception]
        assert not exc, f"Deploy panel raised: {exc[:2]}"
        md = " ".join(m.value for m in at.markdown)
        assert "Test in paper trading" in md, (
            f"deploy panel header missing: {md[:200]}")

        # -- New paper trading account path --
        radios = [r for r in at.radio if r.label == "Deploy to"]
        assert radios, "deploy target radio missing"
        create_btns = [b for b in at.button if b.label == "Create account & deploy"]
        assert create_btns, "create-account button missing"
        create_btns[0].click()
        at.run()
        exc = [str(e.value) for e in at.exception]
        assert not exc, f"Create & deploy raised: {exc[:2]}"
        assert len(fake_system.accounts) == 2, "new account not created"
        assert len(fake_system.deployed) == 2, "strategy not deployed to new account"
        assert any(aid != "PT_U_1" for aid in fake_system.deployed), (
            "deployed strategy should be on the new account")

        # -- Existing portfolio with a strategy already in place --
        # open the panel again for a fresh algorithm result id
        deploy_btns = [b for b in at.button if b.label == "Test in paper trading"]
        assert deploy_btns, "deploy button missing after first deploy"
        deploy_btns[0].click()
        at.run()
        radios = [r for r in at.radio if r.label == "Deploy to"]
        assert radios, "deploy target radio missing"
        radios[0].set_value("Existing portfolio")
        at.run()
        exc = [str(e.value) for e in at.exception]
        assert not exc, f"Existing-portfolio panel raised: {exc[:2]}"
        # a strategy is already deployed -> override confirmation is required
        warns = [w.value for w in at.warning]
        assert any("currently deployed" in w for w in warns), (
            f"expected override warning, got warnings: {warns[:3]}")
        # deploy button is disabled until the override checkbox is checked
        deploy_existing = [b for b in at.button
                           if b.label == "Deploy to this account"]
        assert deploy_existing, "deploy-to-existing button missing"
        assert deploy_existing[0].disabled, (
            "deploy button must be disabled before override confirmation")
        boxes = [c for c in at.checkbox if "replace" in c.label]
        assert boxes, "override checkbox missing"
        boxes[0].set_value(True)
        at.run()
        deploy_existing = [b for b in at.button
                           if b.label == "Deploy to this account"]
        assert deploy_existing and not deploy_existing[0].disabled, (
            "deploy button must enable after override confirmation")


def test_quant_portal_signal_history_is_capped_and_completes():
    """Regression: 'Generate Quant Signal' ran one full ensemble training per
    history window (~100 windows x ~10s each), hanging the tab for minutes.

    The history loop is now capped at 24 windows. Assert the click completes
    with no exception and predict() is called <= 25 times (1 current + 24
    history windows)."""
    calls = {"n": 0}

    def _fake_predict(self, prices, *a, **k):
        calls["n"] += 1
        return type("Sig", (), {
            "direction": "NEUTRAL", "probability": 0.5, "confidence": 0.3,
            "expected_return": 0.0, "sub_model_signals": {},
        })()

    at = AppTest.from_file(os.path.join(ROOT, "main.py"), default_timeout=240)
    with ExitStack() as stack:
        for m in _build_mocks():
            stack.enter_context(m)
        stack.enter_context(patch("quant_ensemble_model.QuantEnsembleModel.predict",
                                  new=_fake_predict))
        # quant_portal imports get_stock at module import time; the walkthrough's
        # data_sources.* patches never reach it, so bind its own.
        stack.enter_context(patch("quant_portal.get_stock",
                                  side_effect=_mock_get_stock))
        at.run()
        for r in at.sidebar.radio:
            if r.label == "Navigation":
                r.set_value("Quant Portal")
                at.run()
                break
        btns = [b for b in at.button if b.label == "Generate Quant Signal"]
        assert btns, "Generate Quant Signal button not found"
        btns[0].click()
        at.run()
        exc = [str(e.value) for e in at.exception]
        assert not exc, f"Quant Portal signal raised: {exc[:2]}"
        assert 2 <= calls["n"] <= 25, (
            f"signal history loop not capped: {calls['n']} ensemble predicts")


def test_quant_portal_evolution_progress_callback_arity():
    """Regression: the Strategy Evolution tab crashed with
    'render_quant_portal.<locals>._progress() takes 1 positional argument but 3
    were given' because genetic_strategy_engine.evolve calls the callback with
    (gen, total, fitness). The portal callback must accept all three."""
    import genetic_strategy_engine as gse
    from types import SimpleNamespace

    def _fake_evolve(self, close, capital, progress_callback):
        # self is bound because the patch replaces a class method
        progress_callback(1, 20, 0.5)   # the real 3-arg call site
        progress_callback(20, 20, 1.2)
        return SimpleNamespace(
            generations=[SimpleNamespace(best_fitness=0.5),
                         SimpleNamespace(best_fitness=1.2)],
            best_strategy=SimpleNamespace(params={"fast": 5}),
            best_sharpe=1.2,
        )

    at = AppTest.from_file(os.path.join(ROOT, "main.py"), default_timeout=240)
    with ExitStack() as stack:
        for m in _build_mocks():
            stack.enter_context(m)
        stack.enter_context(patch("genetic_strategy_engine.GeneticStrategyEngine.evolve",
                                  new=_fake_evolve))
        stack.enter_context(patch("quant_portal.get_stock", side_effect=_mock_get_stock_long))
        at.run()
        for r in at.sidebar.radio:
            if r.label == "Navigation":
                r.set_value("Quant Portal")
                at.run()
                break
        btns = [b for b in at.button if b.label == "Evolve Strategies"]
        assert btns, "Evolve Strategies button not found"
        btns[0].click()
        at.run()
        exc = [str(e.value) for e in at.exception]
        assert not exc, f"Evolve Strategies raised: {exc[:2]}"
        ok = " ".join(s.value for s in at.success)
        assert "Evolution complete" in ok


def test_quant_portal_backtest_uses_advanced_backtester():
    """Regression: 'Run Backtest' crashed with
    'AdvancedBacktester.__init__() got an unexpected keyword argument symbol'.
    The tab must construct the backtester with its real signature, run it, and
    render the resulting metrics without raising."""
    at = AppTest.from_file(os.path.join(ROOT, "main.py"), default_timeout=240)
    with ExitStack() as stack:
        for m in _build_mocks():
            stack.enter_context(m)
        stack.enter_context(patch("quant_portal.get_stock", side_effect=_mock_get_stock_long))
        at.run()
        for r in at.sidebar.radio:
            if r.label == "Navigation":
                r.set_value("Quant Portal")
                at.run()
                break
        btns = [b for b in at.button if b.label == "Run Backtest"]
        assert btns, "Run Backtest button not found"
        btns[0].click()
        at.run()
        exc = [str(e.value) for e in at.exception]
        assert not exc, f"Run Backtest raised: {exc[:2]}"


def test_quant_portal_alt_data_uses_valid_signal_fields():
    """Regression: the Alternative Data tab crashed with
    'AltDataSignal object has no attribute signal_type'. The display must use
    the dataclass's real fields (name/category), not signal_type."""
    import alternative_data_engine as ade
    from datetime import datetime

    def _fake_signals(ticker):
        return [ade.AltDataSignal(
            name="Satellite Footfall", category="satellite", ticker=ticker,
            direction="BULLISH", strength=72.0, confidence=60.0, decay_days=14,
            description="Foot traffic at retail locations rising.",
            generated_at=datetime.utcnow().isoformat())]

    at = AppTest.from_file(os.path.join(ROOT, "main.py"), default_timeout=240)
    with ExitStack() as stack:
        for m in _build_mocks():
            stack.enter_context(m)
        stack.enter_context(patch("alternative_data_engine.AlternativeDataEngine.get_all_signals",
                                  side_effect=_fake_signals))
        at.run()
        for r in at.sidebar.radio:
            if r.label == "Navigation":
                r.set_value("Quant Portal")
                at.run()
                break
        btns = [b for b in at.button if b.label == "Fetch Alternative Data"]
        assert btns, "Fetch Alternative Data button not found"
        btns[0].click()
        at.run()
        exc = [str(e.value) for e in at.exception]
        assert not exc, f"Fetch Alternative Data raised: {exc[:2]}"
        md = " ".join(m.value for m in at.markdown)
        assert "Satellite Footfall" in md and "satellite" in md
