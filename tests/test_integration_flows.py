"""
Per-feature integration tests with mocked external providers.

Exercises real end-to-end flows (SEC 13F EDGAR fetch & parse, market scanner,
breaking-trades pipeline, paper-trading lifecycle) with network and data
providers mocked so the tests are deterministic and offline-safe.
"""
import asyncio
import concurrent.futures
import logging
import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _side_val(p) -> str:
    """Normalize a position side (enum or string) to an uppercase string."""
    v = p.side
    return v.value if hasattr(v, "value") else str(v).upper()


# ─────────────────────────────────────────────────────────────────────────────
# 1. SEC 13F — real EDGAR parse + no fabrication
# ─────────────────────────────────────────────────────────────────────────────
def _fake_edgar():
    import sec_13f_engine as m

    class FakeResp:
        def __init__(self, payload, status=200):
            self._payload = payload
            self.status_code = status

        def json(self):
            return self._payload

    submissions = {"filings": {"recent": {
        "form": ["13F-HR", "13F-HR"],
        "accessionNumber": ["0000123-26-000001", "0000123-25-000001"],
        "filingDate": ["2026-05-15", "2026-02-14"],
        "reportDate": ["2026-03-31", "2025-12-31"],
        "primaryDocument": ["InfoTable.xml", "InfoTable.xml"],
    }}}
    infotable = {
        "header": ["cusip", "nameOfIssuer", "value", "sshPrnamt", "putCall", "tickers"],
        "data": [
            ["037833100", "APPLE INC", 1250000, 1000, "", "AAPL"],
            ["594918104", "MICROSOFT CORP", 900000, 800, "", "MSFT"],
        ],
    }

    def fake_get(url, **kwargs):
        if "submissions" in url:
            return FakeResp(submissions)
        return FakeResp(infotable)

    return m, fake_get


def test_13f_fetch_parses_real_edgar_data(monkeypatch):
    m, fake_get = _fake_edgar()
    m._EDGAR_CACHE.clear()
    monkeypatch.setattr(m.requests, "get", fake_get)
    engine = m.SEC13FEngine()

    data = engine._fetch_from_edgar("1067983")
    assert data and data.get("rows"), "EDGAR infotable rows should parse"
    assert len(data["rows"]) == 2

    filing = engine._build_filing_from_edgar("Berkshire Hathaway", data)
    assert filing is not None
    assert filing.total_aum > 0
    assert filing.data_source == m.PROVENANCE_REAL
    assert not filing.is_simulated
    assert filing.data_available


def test_13f_never_fabricates_when_edgar_unavailable(monkeypatch):
    m, _ = _fake_edgar()
    m._EDGAR_CACHE.clear()

    def fail(url, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(m.requests, "get", fail)
    engine = m.SEC13FEngine()
    filings = engine.fetch_latest_filings(limit=2)  # allow_simulated defaults to False
    assert len(filings) == 2
    for f in filings:
        assert not f.is_simulated
        assert not f.data_available
        assert f.data_source == m.PROVENANCE_UNAVAILABLE
        assert f.total_aum == 0.0


def test_13f_simulation_only_when_opted_in(monkeypatch):
    m, _ = _fake_edgar()
    m._EDGAR_CACHE.clear()

    def fail(url, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(m.requests, "get", fail)
    engine = m.SEC13FEngine()
    filings = engine.fetch_latest_filings(limit=1, allow_simulated=True)
    assert len(filings) == 1
    assert filings[0].is_simulated
    assert filings[0].data_source == m.PROVENANCE_SIMULATED


def test_global_smart_money_flow_is_cached_within_ttl():
    """Regression: get_global_smart_money_flow() rebuilt the aggregate from
    raw filings on every call (~3s each), and the ML ensemble called it twice
    per predict — a 100-window signal-history loop burned minutes re-computing
    identical data. The aggregate must be computed once per TTL window."""
    import sec_13f_engine as m

    engine = m.SEC13FEngine()
    m.SEC13FEngine._FLOW_CACHE = None      # clear any session state
    m.SEC13FEngine._FLOW_CACHE_TS = 0.0
    with patch.object(engine, "fetch_latest_filings", return_value=[]) as fetcher:
        r1 = engine.get_global_smart_money_flow()
        r2 = engine.get_global_smart_money_flow()
        r3 = engine.get_global_smart_money_flow()
    assert fetcher.call_count == 1, "aggregate must be computed once per TTL window"
    assert r1 == r2 == r3


# ─────────────────────────────────────────────────────────────────────────────
# 2. Discovery engine — real realized volatility (no placeholder)
# ─────────────────────────────────────────────────────────────────────────────
def test_discovery_technical_card_computes_real_volatility():
    from octavian_discovery_engine import OctavianDiscoveryEngine

    rng = np.random.default_rng(7)
    closes = 100 + np.cumsum(rng.normal(0, 1.0, 80))
    df = pd.DataFrame({"Close": closes})
    # __new__ skips the heavy __init__ (quant ensemble) — the card is pure math.
    engine = OctavianDiscoveryEngine.__new__(OctavianDiscoveryEngine)
    card = engine._calculate_technical_card(df)

    assert "volatility_20d" in card
    rets = np.diff(closes[-21:]) / closes[-21:-1]
    expected = float(np.std(rets, ddof=1) * np.sqrt(252) * 100)
    assert abs(card["volatility_20d"] - round(expected, 2)) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 3. Market scanner — Vol% is real, never the old 25.0 placeholder
# ─────────────────────────────────────────────────────────────────────────────
def test_scanner_pulse_uses_real_volatility(monkeypatch):
    import market_scanner as ms

    class FakeDiscovery:
        async def scan_market_pulse(self, symbols, deep_scan=False):
            return [{
                "symbol": sym,
                "score": 20.0,
                "price": 100.0,
                "change_1d": 1.5,
                "indicators": {
                    "momentum_10d": 3.0,
                    "rsi": 55.0,
                    "above_sma50": True,
                    "volatility_20d": 31.5,
                },
            } for sym in symbols]

    monkeypatch.setattr("octavian_discovery_engine.get_discovery_engine", lambda: FakeDiscovery())
    df = asyncio.run(ms._scan_universe_async({"TEST": "TEST"}))
    assert not df.empty
    assert df.iloc[0]["Vol%"] == 31.5
    assert df.iloc[0]["Vol%"] != 25.0  # placeholder must be gone


# ─────────────────────────────────────────────────────────────────────────────
# 4. Breaking-trades pipeline (mocked discovery + data + quant)
# ─────────────────────────────────────────────────────────────────────────────
def test_breaking_trades_pipeline_with_mocked_providers(monkeypatch):
    import breaking_trades_generator as btg

    rng = np.random.default_rng(3)
    n = 150
    closes = 100 + np.cumsum(rng.normal(0.15, 0.8, n))  # gently trending
    df = pd.DataFrame({"Close": closes, "Volume": np.linspace(1e6, 2.5e6, n)})
    monkeypatch.setattr(btg, "get_stock", lambda sym, **kw: df)

    class FakeDiscovery:
        async def scan_market_pulse(self, symbols, deep_scan=False):
            return [{"symbol": sym, "score": 30.0} for sym in symbols]

    monkeypatch.setattr("octavian_discovery_engine.get_discovery_engine", lambda: FakeDiscovery())

    class FakeSignal:
        direction = "BULLISH"
        confidence = 0.75
        expected_return = 0.06

    # Construct without heavy __init__ (quant ensemble + VIX fetch), then inject mocks.
    gen = btg.BreakingTradesGenerator.__new__(btg.BreakingTradesGenerator)
    gen.min_confidence = 35.0
    gen._effective_min_confidence = 0.0  # let any setup through in the test
    gen.quant = MagicMock()
    gen.quant.predict = lambda prices: FakeSignal()
    gen._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    gen._cache = {}
    gen.logger = logging.getLogger("test_bt")
    gen._vix_level = None

    setups = gen.generate_breaking_trades(["TEST"])
    assert isinstance(setups, list)
    assert len(setups) >= 1, "trending data + confident signal should produce a setup"
    s = setups[0]
    assert s.symbol == "TEST"
    assert s.confidence_score > 0
    assert s.current_price > 0
    assert s.stop_loss < s.current_price < s.take_profit_1


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Quant ensemble — price-only inputs + correct trend direction
# Regression: _prepare_data required Open/High/Low columns, so any caller that
# passed a bare Close array (breaking-trades generator, scanners) raised KeyError
# inside the ensemble, was swallowed by predict(), and always returned a neutral
# signal with 0.0 confidence — which is why breaking trades produced zero setups.
# Regression 2: undertrained regressors regressed toward the window mean, so a
# strong uptrend printed a false SELL.
# ─────────────────────────────────────────────────────────────────────────────
def test_quant_ensemble_predict_accepts_price_only_array():
    import quant_ensemble_model as qem

    rng = np.random.default_rng(11)
    n = 220
    arr = (100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))).astype(float)
    sig = qem.get_quant_ensemble().predict(arr)  # ndarray, no OHLC

    assert sig.direction in ("BULLISH", "BEARISH", "NEUTRAL")
    assert sig.confidence > 0.0, "price-only input must not silently return the neutral fallback"
    assert isinstance(sig.expected_return, float)
    assert sig.probability > 0.0


def test_quant_model_trend_direction_is_not_inverted():
    import quant_ensemble_model as qem

    n = 220
    q = qem.get_quant_ensemble()

    up = q.predict(100 * np.linspace(1.0, 1.5, n))
    assert up.direction == "BULLISH", f"strong uptrend must be BULLISH, got {up.direction}"

    down = q.predict(100 * np.linspace(1.5, 1.0, n))
    assert down.direction == "BEARISH", f"strong downtrend must be BEARISH, got {down.direction}"


# ─────────────────────────────────────────────────────────────────────────────
# 4c. Target probability — first-hitting-time recalibration
# Regression 1: the old blend gave distance-agnostic "bullishness" scores heavy
# weight, so an impossible target (CAT to $1,000,000 in 5 days) could print
# ~30%. Every factor must be distance-aware (measured in standard deviations
# over the horizon).
# Regression 2: endpoint probabilities P(S_T >= K) — finishing above the target
# at expiry — are structurally capped and barely ever exceeded 50%. The correct
# quantity for "probability of REACHING the target within the timeframe" is the
# first-hitting-time (touch) probability, which is ~2x the endpoint probability
# under zero drift and legitimately clears 50% for realistic targets.
# ─────────────────────────────────────────────────────────────────────────────
def test_target_probability_absurd_target_is_near_zero():
    from target_probability_engine import _first_hitting_prob

    # Direct math check: 350 -> 1,000,000 in 5 days is ~2,850x; must be ~0
    p = _first_hitting_prob(current=350.0, target=1_000_000.0,
                            mu_annual=0.3, sigma_annual=0.35, t_years=5 / 365.0)
    assert p < 0.001, f"absurd upside target must be ~0%, got {p}%"

    # Same for an absurd downside target
    p_down = _first_hitting_prob(current=350.0, target=0.01,
                                 mu_annual=-0.3, sigma_annual=0.35, t_years=5 / 365.0)
    assert p_down < 0.001, f"absurd downside target must be ~0%, got {p_down}%"

    # Touch probabilities legitimately clear 50% for a realistic target with
    # constructive drift (+10% in 90d at 25% vol) — the old endpoint quantity
    # never got there.
    p_mid = _first_hitting_prob(current=100.0, target=110.0,
                                mu_annual=0.30, sigma_annual=0.25, t_years=90 / 365.0)
    assert 50.0 < p_mid < 90.0, f"realistic target with bullish drift must clear 50%, got {p_mid}%"

    # The zero-drift anchor sits between the old endpoint (~35%) and the
    # bullish-drift case.
    p_neutral = _first_hitting_prob(current=100.0, target=110.0,
                                    mu_annual=0.0, sigma_annual=0.25, t_years=90 / 365.0)
    assert 35.0 < p_neutral < p_mid, f"neutral anchor must sit between, got {p_neutral}%"

    # Exact symmetry holds in LOG space: touching K = S0 * 1.1 with log drift
    # +theta equals touching K = S0 / 1.1 with log drift -theta (the barrier
    # must be log-symmetric: |ln(1.1)| == |ln(1/1.1)|). Because the convexity
    # term applies on both sides, the expected-return drift for the mirrored
    # downside case is -mu + sigma^2.
    sigma = 0.3
    p_up = _first_hitting_prob(100.0, 110.0, 0.2, sigma, 90 / 365.0)
    p_dn = _first_hitting_prob(100.0, 100.0 / 1.1, -0.2 + sigma ** 2, sigma, 90 / 365.0)
    assert abs(p_up - p_dn) < 1e-9, f"touch symmetry broken: {p_up:.4f} vs {p_dn:.4f}"

    # Monotonicity: a stronger drift must raise the touch probability
    p_high = _first_hitting_prob(100.0, 110.0, 0.40, sigma, 90 / 365.0)
    p_zero = _first_hitting_prob(100.0, 110.0, 0.0, sigma, 90 / 365.0)
    p_neg = _first_hitting_prob(100.0, 110.0, -0.20, sigma, 90 / 365.0)
    assert p_high > p_zero > p_neg, f"drift monotonicity broken: {p_high:.2f} > {p_zero:.2f} > {p_neg:.2f}"


def test_first_hitting_prob_matches_monte_carlo_simulation():
    """Independent cross-validation of the closed-form barrier formula.

    A seeded brute-force Monte Carlo touch frequency (fraction of GBM paths
    that reach the target at ANY point) must agree with the closed form within
    a tight tolerance. This guards against subtle sign/convention errors that
    the analytic-only checks (symmetry, monotonicity) cannot catch.
    """
    from target_probability_engine import _first_hitting_prob

    rng = np.random.default_rng(1234)
    current = 100.0
    sigma_annual = 0.30
    t_years = 60 / 252.0  # 60 trading days
    n_paths = 150_000
    # Fine-grained (0.1-day) steps so the MC approximates CONTINUOUS barrier
    # monitoring: discrete daily sampling understates touches that occur
    # mid-session and would bias the comparison low by several pp.
    n_steps = 600
    dt = t_years / n_steps
    step_vol = sigma_annual * np.sqrt(dt)

    def mc_touch(target, mu_annual):
        mu_t = mu_annual - 0.5 * sigma_annual ** 2
        eps = rng.normal(mu_t * dt, step_vol, (n_paths, n_steps))
        paths = current * np.exp(np.cumsum(eps, axis=1))
        if target >= current:
            return float(np.mean(np.max(paths, axis=1) >= target) * 100.0)
        return float(np.mean(np.min(paths, axis=1) <= target) * 100.0)

    for target, mu in [(110.0, 0.10), (108.0, -0.05), (92.0, 0.12), (105.0, 0.30)]:
        closed = _first_hitting_prob(current, target, mu, sigma_annual, t_years)
        sim = mc_touch(target, mu)
        assert abs(closed - sim) < 2.5, (
            f"closed form {closed:.2f}% diverges from MC touch {sim:.2f}% "
            f"(target={target}, mu={mu})"
        )


def test_target_probability_full_pipeline_absurd_case(monkeypatch):
    import target_probability_engine as tpe

    n = 500
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    rng = np.random.default_rng(7)
    close = pd.Series(300 * np.exp(np.cumsum(rng.normal(0.0004, 0.008, n))), index=idx)
    close.iloc[-1] = 350.0
    df = pd.DataFrame({"Close": close}, index=idx)

    monkeypatch.setattr(tpe, "get_stock", lambda sym, **kw: df.copy())
    monkeypatch.setattr(tpe, "get_realtime_price", lambda *a, **k: (350.0, 349.0))
    # Offline-safe + fast: no real torch ensemble, no network news fetch
    monkeypatch.setattr("advanced_ml_engine.AdvancedEnsembleEngine.analyze_symbol_ensemble",
                        lambda self, **kw: {"final_predicted_price": 360.0,
                                            "decision": "BULLISH", "confidence": 0.6})
    monkeypatch.setattr("advanced_news_processor.AdvancedNewsProcessor.analyze_symbol_sentiment",
                        lambda *a, **k: {"score": 0.3, "top_headlines": ["test headline"]})

    class FakeSig:
        direction = "BULLISH"
        probability = 0.7
        confidence = 0.6
        expected_return = 0.05

        def predict(self, *a, **k):
            return self

    # Even with strongly bullish signals, an impossible target must be ~0
    monkeypatch.setattr(tpe.TargetProbabilityEngine, "_build_technical_signal_stack",
                        lambda self, c, p: {"overall": 0.9, "sma_trend": 0.9, "rsi": 0.5,
                                            "macd": 0.8, "momentum": 0.9, "bollinger": 0.5})
    # No network options-chain fetch in tests: fall back to historical vol
    monkeypatch.setattr(tpe.TargetProbabilityEngine, "_get_implied_vol",
                        lambda self, sym, fallback: (fallback, None, "historical"))
    eng = tpe.TargetProbabilityEngine.__new__(tpe.TargetProbabilityEngine)
    eng.quant = FakeSig()
    eng.options = None
    eng.llm = None
    eng.news = None

    res = eng.analyze_target("CAT", 1_000_000.0, False, 5)
    assert res.final_probability < 1.0, (
        f"CAT->$1M in 5 days must be <1%, got {res.final_probability:.2f}%"
    )
    assert res.statistical_probability < 1.0
    assert res.technical_probability < 1.0
    assert res.quant_probability < 1.0
    assert res.news_probability < 1.0

    # Sanity: a moderate target over a longer horizon stays in a sane band
    res_mid = eng.analyze_target("CAT", 385.0, False, 90)  # +10% in 90 days
    assert 1.0 < res_mid.final_probability < 90.0, res_mid.final_probability

    # Touch calibration: with constructive signals, a realistic target must not
    # be crushed below the old endpoint ceiling — the touch probability should
    # land meaningfully higher than the old ~35% band for this case.
    assert res_mid.final_probability > 35.0, (
        f"+10% in 90d with bullish signals must clear the old endpoint ceiling, "
        f"got {res_mid.final_probability:.2f}%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4d. Target probability — implied vol + calibration audit
# ─────────────────────────────────────────────────────────────────────────────
def test_target_probability_implied_vol_fallback(monkeypatch):
    import target_probability_engine as tpe

    eng = tpe.TargetProbabilityEngine.__new__(tpe.TargetProbabilityEngine)

    # No usable chain -> historical fallback, never a crash
    monkeypatch.setattr("data_sources.get_options_chain",
                        lambda *a, **k: {"error": "no chain"})
    iv, expiry, source = eng._get_implied_vol("AAPL", 0.30)
    assert iv == 0.30
    assert expiry is None
    assert source == "historical"


def test_target_probability_implied_vol_from_chain(monkeypatch):
    import target_probability_engine as tpe

    eng = tpe.TargetProbabilityEngine.__new__(tpe.TargetProbabilityEngine)
    tpe.TargetProbabilityEngine._IV_CACHE.clear()

    def fake_chain(symbol, expiration=None):
        calls = pd.DataFrame({"strike": [98.0, 100.0, 102.0],
                              "impliedVolatility": [0.24, 0.26, 0.28]})
        puts = pd.DataFrame({"strike": [98.0, 100.0, 102.0],
                             "impliedVolatility": [0.22, 0.24, 0.26]})
        return {"calls": calls, "puts": puts, "underlying_price": 100.0,
                "expiration": "2026-08-21", "error": None}

    monkeypatch.setattr("data_sources.get_options_chain", fake_chain)
    iv, expiry, source = eng._get_implied_vol("AAPL", 0.30)
    assert source == "options-implied"
    assert expiry == "2026-08-21"
    # median of 0.22..0.28 = 0.25, shrunk 70/30 toward 0.30 -> 0.265, within
    # the [0.15, 0.75] winsorization band
    assert 0.20 < iv < 0.30, iv

    # Cache hit: the second call must NOT re-fetch the chain
    def boom(*a, **k):
        raise AssertionError("cache should prevent a second chain fetch")

    monkeypatch.setattr("data_sources.get_options_chain", boom)
    iv2, expiry2, source2 = eng._get_implied_vol("AAPL", 0.30)
    assert iv2 == iv and source2 == "options-implied"
    tpe.TargetProbabilityEngine._IV_CACHE.clear()


def test_target_probability_rejects_absurd_implied_vol(monkeypatch):
    """Regression: a garbage chain IV (e.g. 237% for a blue chip whose
    historical vol is ~44%) must be REJECTED and fall back to historical,
    never fed into the probability math."""
    import target_probability_engine as tpe

    eng = tpe.TargetProbabilityEngine.__new__(tpe.TargetProbabilityEngine)
    tpe.TargetProbabilityEngine._IV_CACHE.clear()

    def absurd_chain(symbol, expiration=None):
        # IVs that pass the raw 0.02-1.5 plausibility filter but are still far
        # above the 0.437 historical (3.0x band cap = 1.31) — the reported bug
        # class (implied ~1.4x-3x that of realized for a blue chip).
        calls = pd.DataFrame({"strike": [300.0, 308.0, 316.0],
                              "impliedVolatility": [1.38, 1.42, 1.45]})
        puts = pd.DataFrame({"strike": [300.0, 308.0, 316.0],
                             "impliedVolatility": [1.35, 1.40, 1.44]})
        return {"calls": calls, "puts": puts, "underlying_price": 308.0,
                "expiration": "2026-08-05", "error": None}

    monkeypatch.setattr("data_sources.get_options_chain", absurd_chain)
    iv, expiry, source = eng._get_implied_vol("AAPL", 0.437)
    assert source == "historical", "absurd implied vol must be rejected"
    assert iv == 0.437
    tpe.TargetProbabilityEngine._IV_CACHE.clear()


def test_target_probability_prefers_mid_dated_expiry(monkeypatch):
    """Regression: the nearest listed expiry (often expiring within days) has
    unreliable IVs; the engine must prefer a mid-dated (30-180 DTE) contract.

    Expirations are generated relative to "today" so the test is date-agnostic.
    """
    import target_probability_engine as tpe
    from datetime import datetime, timedelta, timezone

    eng = tpe.TargetProbabilityEngine.__new__(tpe.TargetProbabilityEngine)
    tpe.TargetProbabilityEngine._IV_CACHE.clear()

    today = datetime.now(timezone.utc).date()
    d_0 = (today + timedelta(days=0)).isoformat()     # nearest (garbage-adjacent)
    d_44 = (today + timedelta(days=44)).isoformat()   # mid-dated pick (~44 DTE)
    d_107 = (today + timedelta(days=107)).isoformat()
    d_163 = (today + timedelta(days=163)).isoformat()

    requested = []

    def chain_with_expiries(symbol, expiration=None):
        requested.append(expiration)
        iv_vals = [0.25, 0.26, 0.27]  # sane ATM IVs
        calls = pd.DataFrame({"strike": [98.0, 100.0, 102.0],
                              "impliedVolatility": iv_vals})
        puts = pd.DataFrame({"strike": [98.0, 100.0, 102.0],
                             "impliedVolatility": iv_vals})
        return {"calls": calls, "puts": puts, "underlying_price": 100.0,
                "expiration": expiration or d_0,
                "expirations": [d_0, d_44, d_107, d_163],
                "error": None}

    monkeypatch.setattr("data_sources.get_options_chain", chain_with_expiries)
    iv, expiry, source = eng._get_implied_vol("AAPL", 0.30)
    assert source == "options-implied"
    # ~44 DTE is the closest-to-60 mid-dated pick, not the same-day expiry
    assert expiry == d_44, expiry
    assert requested[1] == d_44  # second fetch asked for the chosen expiry
    tpe.TargetProbabilityEngine._IV_CACHE.clear()


def test_target_probability_down_target_not_absurd(monkeypatch):
    """Regression (the exact reported bug): AAPL ~$308, target $200 (-35%)
    over 365d must NOT read 97.7%. With a sane vol (~0.437) the touch
    probability for a 0.99-sigma downside move over a year sits far below
    that, even with mild bearish signals."""
    import target_probability_engine as tpe

    n = 500
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    rng = np.random.default_rng(5)
    close = pd.Series(300 * np.exp(np.cumsum(rng.normal(0.0004, 0.028, n))), index=idx)
    close.iloc[-1] = 308.73
    df = pd.DataFrame({"Close": close}, index=idx)

    monkeypatch.setattr(tpe, "get_stock", lambda sym, **kw: df.copy())
    monkeypatch.setattr(tpe, "get_realtime_price", lambda *a, **k: (308.73, 308.0))
    monkeypatch.setattr("advanced_ml_engine.AdvancedEnsembleEngine.analyze_symbol_ensemble",
                        lambda self, **kw: {"final_predicted_price": 315.0,
                                            "decision": "BULLISH", "confidence": 0.5})
    monkeypatch.setattr("advanced_news_processor.AdvancedNewsProcessor.analyze_symbol_sentiment",
                        lambda *a, **k: {"score": -0.1, "top_headlines": ["h"]})

    class FakeSig:
        direction = "NEUTRAL"
        probability = 0.5
        confidence = 0.2
        expected_return = 0.01

        def predict(self, *a, **k):
            return self

    monkeypatch.setattr(tpe.TargetProbabilityEngine, "_build_technical_signal_stack",
                        lambda self, c, p: {"overall": -0.1, "sma_trend": 0.0, "rsi": -0.2,
                                            "macd": 0.1, "momentum": -0.2, "bollinger": -0.3})
    # Force the SANE historical vol (44% band); no options-implied garbage
    monkeypatch.setattr(tpe.TargetProbabilityEngine, "_get_implied_vol",
                        lambda self, sym, fallback: (fallback, None, "historical"))
    eng = tpe.TargetProbabilityEngine.__new__(tpe.TargetProbabilityEngine)
    eng.quant = FakeSig()
    eng.options = None
    eng.llm = None
    eng.news = None

    res = eng.analyze_target("AAPL", 200.0, False, 365)
    # The model must NOT claim a ~35% decline is near-certain
    assert 0.0 < res.final_probability < 60.0, (
        f"down-target must not read absurdly high, got {res.final_probability:.1f}%"
    )
    # A 35% drop over 1y at 44% vol is a ~0.9-1.0 sigma move -> ~30% touch prob
    assert 5.0 < res.final_probability < 55.0, res.final_probability
    # Every factor must agree the scenario is NOT near-certain
    for p in (res.statistical_probability, res.options_implied_probability,
              res.technical_probability, res.quant_probability, res.news_probability):
        assert 0.0 < p < 70.0, f"factor out of sane band: {p}"

    # Forecast must be internally consistent with the drift that drove the prob
    expected_forecast = res.current_price * np.exp(res.signal_drift * (365 / 365.0))
    assert abs(res.model_predicted_price - expected_forecast) < 0.01
    assert abs(res.model_predicted_pct - (expected_forecast / res.current_price - 1) * 100) < 0.01


def test_target_probability_calibration_audit(monkeypatch, tmp_path):
    """record -> load round-trips, and the reliability audit buckets stated
    probabilities against realized touch outcomes (Brier score verified)."""
    import target_probability_engine as tpe
    from datetime import datetime, timedelta, timezone

    log = tmp_path / "tp_log.json"
    monkeypatch.setattr(tpe, "_log_path", lambda: str(log))

    now_utc = datetime.now(timezone.utc)
    # Matches what record_target_analysis actually writes (aware, no legacy Z)
    old = (now_utc - timedelta(days=40)).isoformat(timespec="seconds")
    recent = now_utc.isoformat(timespec="seconds")

    # Deterministic realized outcomes keyed by symbol
    def fake_touch(entry):
        return entry["symbol"] == "AAA"

    monkeypatch.setattr(tpe, "_realized_touch", fake_touch)

    entries = [
        {"ts": old, "symbol": "AAA", "target_price": 110.0, "current_price": 100.0,
         "target_pct": 10.0, "timeframe_days": 30, "probability": 70.0, "direction": "up"},
        {"ts": old, "symbol": "BBB", "target_price": 90.0, "current_price": 100.0,
         "target_pct": -10.0, "timeframe_days": 30, "probability": 30.0, "direction": "down"},
        {"ts": recent, "symbol": "CCC", "target_price": 120.0, "current_price": 100.0,
         "target_pct": 20.0, "timeframe_days": 90, "probability": 50.0, "direction": "up"},
    ]
    cal = tpe.evaluate_calibration(entries=entries)
    assert cal["sample"] == 2, "the 90-day pending entry must be excluded"
    bins = {b["bucket"]: b for b in cal["bins"]}
    assert bins["60-80%"]["n"] == 1 and bins["60-80%"]["realized_rate"] == 100.0
    assert bins["20-40%"]["n"] == 1 and bins["20-40%"]["realized_rate"] == 0.0
    # Brier = ((0.7-1)^2 + (0.3-0)^2) / 2 = 0.09
    assert abs(cal["brier"] - 0.09) < 1e-6

    # record -> load round trip persists to the log file
    rec = tpe.TargetAnalysisResult(
        symbol="TEST", current_price=100.0, target_price=110.0, target_pct=10.0,
        timeframe_days=30, statistical_probability=60.0, options_implied_probability=55.0,
        ml_probability=58.0, final_probability=62.0, model_predicted_price=105.0,
        model_predicted_pct=5.0, market_status="", what_it_takes_to_happen="",
        what_it_takes_to_fail="", final_conclusion="", volatility=0.25,
        sentiment_score=50.0, news_summary="")
    tpe.record_target_analysis(rec)
    loaded = tpe.load_target_analyses()
    assert loaded and loaded[-1]["symbol"] == "TEST"
    assert loaded[-1]["probability"] == 62.0
    assert loaded[-1]["direction"] == "up"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Paper trading — full BUY/SELL/SHORT/COVER lifecycle
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def pts():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    mock_db = MagicMock()
    mock_db._get_connection.return_value.__enter__.return_value = conn
    mock_db._get_connection.return_value.__exit__.return_value = None
    with patch("paper_trading_system.get_database_manager", return_value=mock_db):
        from paper_trading_system import PaperTradingSystem
        system = PaperTradingSystem(db_manager=mock_db)
        system._initialize_tables()
        yield system
    conn.close()


def test_paper_trading_full_flow(pts):
    from paper_trading_system import TradeAction

    acc = pts.create_account("flow_user", "Main", initial_balance=10000.0)
    assert acc is not None
    aid = acc.account_id

    # BUY 10 @ 100 → -$1000 cash
    assert pts.execute_trade(aid, "TEST", TradeAction.BUY, 10, price=100.0) is not None
    acct = pts.get_account(aid)
    assert acct.current_balance == pytest.approx(9000.0)
    positions = pts.get_positions(aid)
    long_pos = next(p for p in positions if p.symbol == "TEST")
    assert _side_val(long_pos) == "LONG"
    assert long_pos.quantity == 10

    # SELL 4 @ 110 → realize profit, position shrinks to 6
    assert pts.execute_trade(aid, "TEST", TradeAction.SELL, 4, price=110.0) is not None
    positions = pts.get_positions(aid)
    long_pos = next(p for p in positions if p.symbol == "TEST" and _side_val(p) == "LONG")
    assert long_pos.quantity == 6

    # SHORT 5 @ 120
    assert pts.execute_trade(aid, "TEST", TradeAction.SHORT, 5, price=120.0) is not None
    positions = pts.get_positions(aid)
    shorts = [p for p in positions if p.symbol == "TEST" and _side_val(p) == "SHORT"]
    assert shorts and shorts[0].quantity == 5

    # COVER 2 @ 118 → short position shrinks to 3
    assert pts.execute_trade(aid, "TEST", TradeAction.COVER, 2, price=118.0) is not None
    positions = pts.get_positions(aid)
    shorts = [p for p in positions if p.symbol == "TEST" and _side_val(p) == "SHORT"]
    assert shorts and shorts[0].quantity == 3

    # Trade history records all four action types
    history = pts.get_trade_history(aid)
    actions = [t.action.value for t in history]
    assert actions.count("BUY") == 1
    assert actions.count("SELL") == 1
    assert actions.count("SHORT") == 1
    assert actions.count("COVER") == 1

    # Insufficient balance is rejected (no fabricated fills)
    broke = pts.create_account("broke_user", "Broke", initial_balance=100.0)
    assert pts.execute_trade(broke.account_id, "TEST", TradeAction.BUY, 1000, price=10.0) is None


def test_paper_trading_accepts_int_user_id_like_auth_engine(pts):
    """auth_engine stores st.session_state['user_id'] as an int (SQLite row id).

    Regression: _normalize_user_id called .strip() on the int and raised
    AttributeError, so account listing/creation silently failed for every real
    logged-in user (the UI caught this during the AppTest walkthrough).
    """
    from paper_trading_system import TradeAction

    # Exact call pattern used by paper_trading_ui with a real authenticated user
    user_id = 42  # int, as set by auth_engine.authenticate_user
    acc = pts.create_account(user_id, "UI Account", initial_balance=10000.0)
    assert acc is not None, "create_account must accept an int user_id"
    assert acc.user_id == "42"

    accounts = pts.list_accounts(user_id)
    assert len(accounts) == 1
    assert accounts[0].account_id == acc.account_id

    # Same bucket resolved from the string form too
    accounts_str = pts.list_accounts("42")
    assert len(accounts_str) == 1
    assert accounts_str[0].account_id == acc.account_id

    # And trades flow through
    assert pts.execute_trade(acc.account_id, "TEST", TradeAction.BUY, 5, price=50.0) is not None
    assert len(pts.get_positions(acc.account_id)) == 1


def test_paper_trading_long_and_short_same_symbol_coexist(pts):
    """Regression: positions were keyed by (account_id, symbol) without side, so
    opening a SHORT on a symbol you are long merged into (and corrupted) the
    long position. They must be tracked independently.
    """
    from paper_trading_system import TradeAction

    acc = pts.create_account("side_user", "Side Test", initial_balance=50000.0)
    aid = acc.account_id

    assert pts.execute_trade(aid, "TEST", TradeAction.BUY, 10, price=100.0) is not None
    assert pts.execute_trade(aid, "TEST", TradeAction.SHORT, 5, price=110.0) is not None

    positions = pts.get_positions(aid)
    longs = [p for p in positions if p.symbol == "TEST" and _side_val(p) == "LONG"]
    shorts = [p for p in positions if p.symbol == "TEST" and _side_val(p) == "SHORT"]
    assert len(longs) == 1 and longs[0].quantity == 10
    assert len(shorts) == 1 and shorts[0].quantity == 5

    # Closing the SHORT must not touch the LONG
    assert pts.execute_trade(aid, "TEST", TradeAction.COVER, 5, price=105.0) is not None
    positions = pts.get_positions(aid)
    longs = [p for p in positions if p.symbol == "TEST" and _side_val(p) == "LONG"]
    assert len(longs) == 1 and longs[0].quantity == 10
    shorts = [p for p in positions if p.symbol == "TEST" and _side_val(p) == "SHORT"]
    assert shorts == []


# ─────────────────────────────────────────────────────────────────────────────
# 6. Options — robust execute_option_trade + side-aware closing
# ─────────────────────────────────────────────────────────────────────────────
def test_option_trade_minimal_and_full_calls(pts):
    """execute_option_trade must accept both the legacy minimal call pattern
    (account, symbol, action, qty, price — as portfolio_analyzer used it) and
    the full metadata call, and must reject invalid actions."""
    from paper_trading_system import TradeAction

    acc = pts.create_account("opt_user", "Opt", initial_balance=100000.0)
    aid = acc.account_id

    # Legacy minimal call with enum action (portfolio_analyzer pattern)
    r1 = pts.execute_option_trade(aid, "AAPL", None, TradeAction.SELL, 2, 5.0)
    assert r1 is not None

    # Full metadata call with string action
    r2 = pts.execute_option_trade(aid, "AAPL", "AAPL260515C00150000", "BUY", 3, 5.5,
                                  option_type="CALL", strike=150.0, expiration="2026-05-15")
    assert r2 is not None

    # Invalid action rejected (no fabricated fill)
    assert pts.execute_option_trade(aid, "AAPL", None, "HODL", 1, 1.0) is None

    # COVER (closing a short) normalizes to BUY and must not error
    assert pts.execute_option_trade(aid, "AAPL", None, TradeAction.COVER, 1, 5.0) is not None

    positions = pts.get_option_positions(aid)
    assert len(positions) == 2


def test_option_short_cover_closes_position(pts):
    """Regression: covering a short option with COVER previously ADDED quantity
    (position math ignored the existing side). A SHORT covered by BUY/COVER must
    reduce quantity and delete the position at zero."""
    from paper_trading_system import TradeAction

    acc = pts.create_account("opt_cov", "Cover", initial_balance=100000.0)
    aid = acc.account_id

    # Open SHORT 2 contracts
    assert pts.execute_option_trade(aid, "AAPL", None, TradeAction.SELL, 2, 5.0) is not None
    pos = pts.get_option_positions(aid)
    assert len(pos) == 1 and _side_val(pos[0]) == "SHORT" and pos[0].quantity == 2.0

    # Cover all 2 -> position deleted
    assert pts.execute_option_trade(aid, "AAPL", None, TradeAction.COVER, 2, 4.8) is not None
    assert pts.get_option_positions(aid) == []

    # Balance: +2*5*100 (premium received) - 2*4.8*100 (buyback) = +40
    acct = pts.get_account(aid)
    assert acct.current_balance == pytest.approx(100040.0)


def test_option_long_close_reduces_position(pts):
    """A LONG option position closed with SELL must reduce quantity and delete
    at zero, never grow it."""
    from paper_trading_system import TradeAction

    acc = pts.create_account("opt_long", "Long", initial_balance=100000.0)
    aid = acc.account_id

    assert pts.execute_option_trade(aid, "AAPL", None, TradeAction.BUY, 3, 5.0) is not None
    pos = pts.get_option_positions(aid)
    assert len(pos) == 1 and _side_val(pos[0]) == "LONG" and pos[0].quantity == 3.0

    assert pts.execute_option_trade(aid, "AAPL", None, TradeAction.SELL, 1, 5.2) is not None
    pos = pts.get_option_positions(aid)
    assert len(pos) == 1 and pos[0].quantity == 2.0

    assert pts.execute_option_trade(aid, "AAPL", None, TradeAction.SELL, 2, 5.4) is not None
    assert pts.get_option_positions(aid) == []
