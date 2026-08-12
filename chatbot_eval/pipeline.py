"""Fast deterministic pipeline runner.

Runs the real `financial_llm_engine` response generator against a *mocked*
data layer so thousands of queries execute in milliseconds each with zero
network dependency. Every symbol gets a deterministic synthetic price series
(seeded by the symbol name), so responses contain real numbers and the
transmission / setup engines operate on plausible data — exactly like
production, just fast and reproducible.

Mocks installed (restored on exit):
  * financial_llm_engine._fetch_live_data_for_tickers
  * data_sources.get_stock
  * macro_cross_asset_engine.MacroDataLayer  (used by cross_asset_transmission)
  * financial_llm_engine._CACHE_DIR           (isolated temp cache)
"""

import contextlib
import functools
import hashlib
import tempfile

import numpy as np
import pandas as pd

# Heavy engines imported ONCE here (main thread, module load) so the per-query
# mock context never re-imports torch/sklearn/streamlit stacks (11s+ each).
import advanced_ml_engine as _aml  # noqa: E402
import advanced_news_processor as _anp  # noqa: E402
import quant_ensemble_model as _qem  # noqa: E402
import target_probability_engine as _tpe  # noqa: E402

# The real chart builder used by ai_chatbot.process_enhanced_query, bound here
# ONCE so the per-query visual check exercises the genuine generation path
# (candlestick + indicators via plotly) against mocked price data.
import ai_chatbot as _acb  # noqa: E402

# Pin the class that actually owns the live chart builder (process_enhanced_query
# lives on OctavianEnhancedChatbot). Fall back to a scan only if renamed.
_VISUAL_CHART_FN = None
for _cls_name in ("OctavianEnhancedChatbot",):
    _obj = getattr(_acb, _cls_name, None)
    if isinstance(_obj, type) and hasattr(_obj, "_create_advanced_price_chart"):
        _VISUAL_CHART_FN = _obj._create_advanced_price_chart
        break
if _VISUAL_CHART_FN is None:
    for _name in dir(_acb):
        _obj = getattr(_acb, _name)
        if isinstance(_obj, type) and hasattr(_obj, "_create_advanced_price_chart"):
            _VISUAL_CHART_FN = _obj._create_advanced_price_chart
            break

# Query classes where a price/technical visual materially helps the answer.
_VISUAL_NEEDED_CATEGORIES = {
    "equity_outlook", "equity_technical", "sector_scan", "sector_etf",
    "transmission_oil", "transmission_gold", "transmission_rates",
    "transmission_dollar", "transmission_crypto", "commodity_outlook",
    "setup_futures", "setup_equity", "setup_crypto",
    "hedging_portfolio", "hedging_position", "risk_management",
    "options_strategy", "fx_outlook", "crypto_outlook",
    "probability_target", "probability_downside",
    "index_outlook", "bonds_outlook", "comparison_two",
    "equity_earnings", "macro_outlook", "mega_prompt",
}

_VISUAL_REQUEST_WORDS = (
    "chart", "graph", "plot", "visual", "show me", "display",
    "candlestick", "heatmap", "picture", "image",
)


_FX_BASE_PRICE = {
    "EURUSD=X": 1.08, "GBPUSD=X": 1.27, "USDJPY=X": 150.0, "AUDUSD=X": 0.66,
    "USDCAD=X": 1.36, "USDCHF=X": 0.88, "NZDUSD=X": 0.61, "EURGBP=X": 0.85,
    "EURJPY=X": 162.0, "GBPJPY=X": 190.0, "EURCHF=X": 0.95, "EURCAD=X": 1.47,
    "EURAUD=X": 1.64, "EURNZD=X": 1.77, "GBPAUD=X": 1.92, "GBPCAD=X": 1.73,
    "GBPCHF=X": 1.12, "GBPNZD=X": 2.08, "AUDCAD=X": 0.90, "AUDCHF=X": 0.58,
    "AUDNZD=X": 1.08, "CADJPY=X": 110.0, "CADCHF=X": 0.65, "CHFJPY=X": 170.0,
    "NZDJPY=X": 91.0, "USDKRW=X": 1350.0, "USDINR=X": 83.0, "USDSGD=X": 1.34,
    "USDHKD=X": 7.80, "USDCNY=X": 7.20, "USDMXN=X": 17.5, "USDZAR=X": 18.5,
    "USDBRL=X": 5.0, "USDTRY=X": 32.0, "USDPHP=X": 56.0, "USDMYR=X": 4.7,
    "USDTHB=X": 36.0, "USDTWD=X": 31.5, "USDIDR=X": 15600.0, "USDCLP=X": 950.0,
}


@functools.lru_cache(maxsize=4096)
def _series_for(sym: str, n: int) -> pd.DataFrame:
    """Memoized deterministic daily OHLCV for a symbol."""
    seed = int(hashlib.sha1(str(sym).encode("utf-8")).hexdigest(), 16) % (2 ** 31)
    rng = np.random.default_rng(seed)
    # FX pairs get realistic price levels so "EURUSD=X" reads ~1.08 rather
    # than an arbitrary 200+ hash-derived number. Everything else keeps the
    # equity-style range (15-315).
    if str(sym).endswith("=X"):
        base = _FX_BASE_PRICE.get(str(sym).upper(), 1.10)
        base *= (1.0 + rng.uniform(-0.02, 0.02))
        vol = rng.uniform(0.004, 0.010)
        rets = rng.normal(rng.normal(0.0001, 0.0002), vol, n)
    else:
        base = 15.0 + (seed % 3000) / 10.0
        drift = rng.normal(0.00025, 0.00035)
        vol = rng.uniform(0.010, 0.030)
        rets = rng.normal(drift, vol, n)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    close = base * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0.0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0.0, 0.006, n)))
    open_ = close * (1 + rng.normal(0.0, 0.004, n))
    volume = rng.integers(100_000, 8_000_000, n)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=idx)


def synthetic_series(sym: str, n: int = 520) -> pd.DataFrame:
    """Deterministic daily OHLCV for a symbol (seeded by its name)."""
    return _series_for(sym, n)


def _fake_live_data(tickers_list):
    out = {}
    for t in tickers_list or []:
        try:
            df = synthetic_series(t, n=40)
            close = df["Close"]
            price = float(close.iloc[-1])
            chg = (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100
            out[t] = {"price": price, "change_5d": chg}
        except Exception:
            continue
    return out


def _fake_get_stock(symbol, period="3y", interval="1d"):
    n = {"5d": 10, "1mo": 25, "3mo": 66, "6mo": 130, "1y": 260, "2y": 520,
         "3y": 780}.get(str(period), 520)
    return synthetic_series(symbol, n=n)


class FakeMacroDataLayer:
    """Drop-in for macro_cross_asset_engine.MacroDataLayer."""

    def __init__(self, *args, **kwargs):
        pass

    def get_returns(self, symbol, period="2y"):
        return synthetic_series(symbol)["Close"].pct_change().dropna()

    def get_close(self, symbol, period="2y"):
        return synthetic_series(symbol)["Close"]

    def get_fx_returns(self, *a, **k):
        return self.get_returns("EURUSD=X", "2y")


@contextlib.contextmanager
def mocked_pipeline():
    """Context manager installing the deterministic mocks."""
    import data_sources
    import financial_llm_engine as fle
    import macro_cross_asset_engine as mce

    old_cache_dir = fle._CACHE_DIR
    fle._CACHE_DIR = tempfile.mkdtemp(prefix="octavian_eval_cache_")
    old_live = fle._fetch_live_data_for_tickers
    old_stock = data_sources.get_stock
    old_layer = mce.MacroDataLayer
    old_print = fle.__dict__.get("print", None)

    # ── Probability engine network deps → deterministic fast stubs ──
    # NOTE: target_probability_engine binds `get_stock`/`get_realtime_price` at
    # module level (`from data_sources import ...`), so patching
    # `data_sources.get_stock` alone does NOT redirect it — patch the bound refs.
    old_tpe_iv = _tpe.TargetProbabilityEngine._get_implied_vol
    old_tpe_stock = _tpe.get_stock
    old_tpe_rt = _tpe.get_realtime_price
    old_anp = _anp.AdvancedNewsProcessor.analyze_symbol_sentiment
    old_aml = _aml.get_ensemble_engine
    old_qem = _qem.QuantEnsembleModel.predict
    try:
        import financial_model_generator as _fmg
        old_fmg_fund = _fmg.fetch_ticker_fundamentals
    except Exception:
        _fmg = None
        old_fmg_fund = None

    def _fake_iv(self, symbol, fallback_iv):
        return (fallback_iv, None, "historical")

    def _fake_tpe_realtime(symbol):
        df = _fake_get_stock(symbol, period="3y")
        closes = df["Close"].dropna()
        cur = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) > 1 else cur
        return (cur, prev)

    def _fake_sentiment(self, symbol, timeout=8):
        return {"score": 0.0, "top_headlines": []}

    class _FakeEnsemble:
        def analyze_symbol_ensemble(self, **kw):
            df = kw.get("data")
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            cur = float(close.dropna().iloc[-1])
            return {"final_predicted_price": cur * 1.02, "signal": "NEUTRAL",
                    "confidence": 0.55}

    class _FakeQuantSignal:
        direction = "NEUTRAL"
        probability = 0.55
        confidence = 0.5
        expected_return = 0.02
        score = 0.0
        factor_contributions = {}
        signal = "NEUTRAL"

    def _fake_quant_predict(self, prices, volumes=None, symbol="UNKNOWN",
                            asset_type="STOCK", options_context=None):
        return _FakeQuantSignal()

    def _fake_fundamentals(ticker, *a, **k):
        """Deterministic fundamentals so the valuation/reverse-DCF handler runs
        fast and consistently in eval instead of hitting live yfinance."""
        sym = str(ticker or "").strip().upper()
        df = _fake_get_stock(sym, period="3y")
        closes = df["Close"]
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        closes = closes.dropna().astype(float)
        px = float(closes.iloc[-1]) if len(closes) else 50.0
        shares_m = 1000.0 + (hashlib.sha1(sym.encode()).digest()[0] % 9000)
        rev_m = shares_m * px / 2.0
        return {
            "price": px,
            "revenue_m": rev_m,
            "revenue_growth": 8.0 + (hashlib.sha1(sym.encode()).digest()[1] % 18),
            "ebit_margin_pct": 12.0 + (hashlib.sha1(sym.encode()).digest()[2] % 25),
            "market_cap_m": shares_m * px,
            "shares_m": shares_m,
            "debt_m": shares_m * px * 0.25,
            "cash_m": shares_m * px * 0.15,
        }

    _tpe.TargetProbabilityEngine._get_implied_vol = _fake_iv
    _tpe.get_stock = _fake_get_stock
    _tpe.get_realtime_price = _fake_tpe_realtime
    _anp.AdvancedNewsProcessor.analyze_symbol_sentiment = _fake_sentiment
    _aml.get_ensemble_engine = lambda: _FakeEnsemble()
    _qem.QuantEnsembleModel.predict = _fake_quant_predict
    if _fmg is not None:
        _fmg.fetch_ticker_fundamentals = _fake_fundamentals

    fle._fetch_live_data_for_tickers = _fake_live_data
    data_sources.get_stock = _fake_get_stock
    mce.MacroDataLayer = FakeMacroDataLayer
    fle.print = lambda *a, **k: None  # silence [PERF]/DEBUG chatter during eval

    # Reset the transmission analyzer singleton so it builds with the fake layer
    try:
        import cross_asset_transmission as cat
        if hasattr(cat, "_analyzer"):
            cat._analyzer = None
    except Exception:
        pass

    try:
        yield
    finally:
        fle._CACHE_DIR = old_cache_dir
        fle._fetch_live_data_for_tickers = old_live
        data_sources.get_stock = old_stock
        mce.MacroDataLayer = old_layer
        _tpe.TargetProbabilityEngine._get_implied_vol = old_tpe_iv
        _tpe.get_stock = old_tpe_stock
        _tpe.get_realtime_price = old_tpe_rt
        _anp.AdvancedNewsProcessor.analyze_symbol_sentiment = old_anp
        _aml.get_ensemble_engine = old_aml
        _qem.QuantEnsembleModel.predict = old_qem
        if _fmg is not None and old_fmg_fund is not None:
            _fmg.fetch_ticker_fundamentals = old_fmg_fund
        if old_print is None:
            fle.__dict__.pop("print", None)
        else:
            fle.print = old_print


def visual_check(query: str, tickers, category: str = None) -> dict:
    """Exercise the real chart-generation path (mocked data) and report what
    would actually be produced, plus whether the query class warrants visuals.

    Returns
    -------
        expected: bool   — this query class warrants price/technical visuals
        requested: bool  — user explicitly asked for a visual
        generated: [sym] — charts the pipeline would actually produce
        candidates: [sym]— tickers available to chart (capped at 3)
    """
    expected = bool(category and (category in _VISUAL_NEEDED_CATEGORIES
                                  or category.startswith("current_events_")))
    requested = any(w in (query or "").lower() for w in _VISUAL_REQUEST_WORDS)
    candidates = list(tickers or [])[:3]
    # Production gating: ai_chatbot only builds charts when the query warrants
    # them (no charts for knowledge/definitional asks). Mirror that here so the
    # eval measures real behavior.
    if not requested and not _acb._query_warrants_charts(query):
        return {"expected": expected, "requested": requested,
                "generated": [], "candidates": candidates}
    generated = []
    for sym in candidates:
        try:
            df = _fake_get_stock(sym, period="6mo")
            if df is None or df.empty or "Close" not in df.columns:
                continue
            if _VISUAL_CHART_FN is not None:
                fig = _VISUAL_CHART_FN(None, df, sym, True, "standard")
                if fig is not None:
                    generated.append(sym)
            else:
                generated.append(sym)
        except Exception:
            continue
    return {"expected": expected, "requested": requested,
            "generated": generated, "candidates": candidates}


def run_query(query: str, with_intents: bool = True):
    """Run one query through the real pipeline; return (intents, tickers, sectors, text)."""
    import time
    import financial_llm_engine as fle

    t0 = time.time()
    intents, tickers, sectors = fle.expand_query_intents(query)
    text = fle.generate_financial_analysis(query)
    elapsed = (time.time() - t0) * 1000.0
    if with_intents:
        return intents, tickers, sectors, text, elapsed
    return text, elapsed
