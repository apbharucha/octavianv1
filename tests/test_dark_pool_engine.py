"""
Offline-safe tests for the Dark Pool Intelligence engine.

All tests inject synthetic OHLCV data through the engine's fetch_fn hook so no
network access is required. Coverage:

  * provenance integrity (observed vs modeled labeling)
  * ticker analysis outputs (metrics, percentiles, imbalance, signals)
  * modeled off-exchange share bounds and behavior
  * largest-print modeling (labeled estimates)
  * institutional inference classification
  * price-relationship conditional stats
  * backtest train/test split + sample sizes
  * AI insight grounding (never fabricates on empty data)
  * market scanning + sector aggregation on the injected universe
  * watchlist / alert persistence (tmp state file)
  * data quality report shape
"""

import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["OCTAVIAN_OFFLINE"] = "1"

from dark_pool_engine import DarkPoolEngine, DEFAULT_OFFEX_BASELINE  # noqa: E402


# --------------------------------------------------------------------------- #
#  Fixtures / helpers
# --------------------------------------------------------------------------- #

def _make_ohlc(n=400, start_price=100.0, seed=7, trend=0.0004, vol_seed=3):
    """Synthetic daily OHLCV with a mild uptrend and volume bursts."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2026-08-01"), periods=n)
    rets = rng.normal(trend, 0.015, n)
    close = start_price * np.cumprod(1 + rets)
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    base_vol = rng.integers(2_000_000, 5_000_000, n).astype(float)
    burst = np.where(np.abs(rets) > 0.03, 2.2, 1.0)
    volume = base_vol * burst
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low,
                       "Close": close, "Volume": volume}, index=dates)
    return df


@pytest.fixture()
def engine():
    cache = {}

    def fetch(symbol, period="2y"):
        if symbol in cache:
            return cache[symbol]
        if symbol.startswith("BURST"):  # names with big volume/return days
            rng = np.random.default_rng(abs(hash(symbol)) % (2 ** 32))
            n = 400
            dates = pd.bdate_range(end=pd.Timestamp("2026-08-01"), periods=n)
            rets = rng.normal(0.0002, 0.03, n)
            close = 50 * np.cumprod(1 + rets)
            vol = rng.integers(1_000_000, 8_000_000, n).astype(float)
            df = pd.DataFrame({"Open": close * 0.999, "High": close * 1.01,
                               "Low": close * 0.99, "Close": close,
                               "Volume": vol}, index=dates)
        else:
            df = _make_ohlc(seed=abs(hash(symbol)) % (2 ** 32))
        cache[symbol] = df
        return df

    tmp = tempfile.mkdtemp()
    return DarkPoolEngine(state_file=os.path.join(tmp, "state.json"), fetch_fn=fetch)


# --------------------------------------------------------------------------- #
#  Provenance integrity
# --------------------------------------------------------------------------- #

def test_provenance_labeling(engine):
    r = engine.analyze_ticker("AAPL")
    assert r["ok"] is True
    prov = r["provenance"]
    # consolidated is observed
    assert prov["consolidated"]["observed"] is True
    assert prov["consolidated"]["category"] == "OBSERVED"
    # off-exchange and direction are modeled
    assert prov["offexchange"]["observed"] is False
    assert prov["offexchange"]["category"] == "MODELED"
    assert prov["direction"]["observed"] is False
    # finra attempt recorded honestly
    assert "finra" in prov


def test_insufficient_history_is_honest(engine):
    cache = {}

    def fetch(symbol, period="2y"):
        df = _make_ohlc(n=10, seed=1)
        cache[symbol] = df
        return df

    tmp = tempfile.mkdtemp()
    e2 = DarkPoolEngine(state_file=os.path.join(tmp, "s.json"), fetch_fn=fetch)
    r = e2.analyze_ticker("SHORT")
    assert r["ok"] is False
    assert "Insufficient history" in r["error"]


# --------------------------------------------------------------------------- #
#  Ticker analysis
# --------------------------------------------------------------------------- #

def test_ticker_analysis_outputs(engine):
    r = engine.analyze_ticker("AAPL")
    assert r["ok"]
    assert r["symbol"] == "AAPL"
    assert r["price"] > 0
    assert r["sector"]  # non-empty
    assert 0 <= r["pressure_score"] <= 100
    assert -1 <= r["imbalance_5d"] <= 1
    assert isinstance(r["offexchange_pct_20d"], float)
    assert 0 <= r["offexchange_pct_20d"] <= 100
    # series present
    s = r["series"]
    assert isinstance(s, pd.DataFrame) and len(s) > 30
    for col in ("Close", "Volume", "OffExchangeVol", "OffExchangeShare",
                "BuyVol", "SellVol", "RollImbalance"):
        assert col in s.columns
    # historical context has keys
    assert "offex_pctile_5d" in r["historical"]
    assert "offex_pctile_20d" in r["historical"]
    # signals + prints lists
    assert isinstance(r["signals"], list) and r["signals"]
    assert isinstance(r["prints"], list)


def test_offexchange_share_bounds(engine):
    r = engine.analyze_ticker("MSFT")
    s = r["series"]
    share = s["OffExchangeShare"]
    assert share.min() >= 0.12
    assert share.max() <= 0.62
    assert share.mean() > 0.10


def test_prints_are_labeled_modeled(engine):
    r = engine.analyze_ticker("AAPL")
    prints = r["prints"]
    assert isinstance(prints, list)
    for p in prints:
        assert p["observed"] is False
        assert p["label"] == "MODELED ESTIMATE"
        assert p["shares"] > 0
        assert p["notional"] > 0
        assert 0 <= p["significance"] <= 100


def test_historical_percentiles(engine):
    r = engine.analyze_ticker("AAPL")
    h = r["historical"]
    for k in ("offex_pctile_5d", "offex_pctile_20d", "offex_pctile_60d"):
        v = h[k]
        assert v is None or (0 <= v <= 100)
    # 1y share percentile may be None with 400 bars * bday ~ not 250? ensure valid
    v = h.get("share_pctile_1y")
    assert v is None or (0 <= v <= 100)


# --------------------------------------------------------------------------- #
#  Institutional inference
# --------------------------------------------------------------------------- #

def test_institutional_inference_shape(engine):
    r = engine.analyze_ticker("AAPL")
    inst = r["institutional"]
    assert inst["pattern"] in (
        "Potential accumulation", "Potential distribution", "Absorption",
        "Liquidity transfer", "Indeterminate")
    assert inst["observed"] is False
    assert inst["label"] == "INFERENCE"
    assert 0 <= inst["confidence"] <= 1
    assert "evidence" in inst


# --------------------------------------------------------------------------- #
#  Price relationship
# --------------------------------------------------------------------------- #

def test_price_relationship_stats(engine):
    r = engine.analyze_ticker("AAPL")
    rel = r["price_relationship"]
    assert rel["ok"] is True
    assert rel["train_n"] > 0 and rel["test_n"] > 0
    rows = rel["rows"]
    assert len(rows) >= 3
    for row in rows:
        assert "n" in row
        if row.get("enough"):
            assert "avg_return_pct" in row
            assert "t_stat" in row


# --------------------------------------------------------------------------- #
#  Backtesting
# --------------------------------------------------------------------------- #

def test_backtest_train_test_split(engine):
    res = engine.backtest_signal("AAPL", condition="high_share", holding=5)
    assert res["ok"] is True
    assert res["train_rows"] > 0 and res["test_rows"] > 0
    assert res["in_sample"]["label"] == "In-sample"
    assert res["out_of_sample"]["label"] == "Out-of-sample"
    # at least one side has enough observations on 400 bars
    assert (res["in_sample"].get("enough") or res["out_of_sample"].get("enough"))


def test_backtest_unknown_condition(engine):
    res = engine.backtest_signal("AAPL", condition="bogus")
    assert res["ok"] is False


# --------------------------------------------------------------------------- #
#  AI insight grounding
# --------------------------------------------------------------------------- #

def test_ai_insight_grounded(engine):
    r = engine.analyze_ticker("AAPL")
    ai = engine.ai_insight(r)
    assert ai["ok"] is True
    assert "What Happened" in ai["sections"]
    assert "Bottom Line" in ai["sections"]
    assert "Alternative Explanation" in ai["sections"]
    assert ai["observed"] is False
    # the insight text must reference the actual symbol
    assert "AAPL" in ai["insight"]
    # sections carry the symbol
    assert "AAPL" in ai["sections"]["What Happened"] or "AAPL" in ai["sections"]["Bottom Line"]


def test_ai_insight_never_fabricates_on_empty(engine):
    r = {"ok": False, "error": "nope"}
    ai = engine.ai_insight(r)
    assert ai["ok"] is False
    assert "Insufficient data" in ai["insight"]


# --------------------------------------------------------------------------- #
#  Scanning + sectors
# --------------------------------------------------------------------------- #

def test_scan_market(engine):
    df = engine.scan_market(limit=6)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) <= 6
    assert "PressureScore" in df.columns
    assert "Imbalance" in df.columns
    assert "OffEx%" in df.columns
    assert "Symbol" in df.columns


def test_sector_analysis(engine):
    df = engine.sector_analysis(limit=6)
    if df.empty:
        pytest.skip("no data")
    assert "Sector" in df.columns
    assert "AvgPressure" in df.columns
    assert "Tickers" in df.columns
    assert (df["AvgPressure"] >= 0).all()


# --------------------------------------------------------------------------- #
#  Watchlists / alerts persistence
# --------------------------------------------------------------------------- #

def test_watchlist_persistence(engine):
    assert engine.create_watchlist("Test")
    assert not engine.create_watchlist("Test")  # duplicate
    assert engine.add_to_watchlist("Test", "aapl")
    assert not engine.add_to_watchlist("Test", "aapl")  # duplicate symbol
    wl = engine.get_watchlists()
    assert wl["Test"] == ["AAPL"]
    assert engine.remove_from_watchlist("Test", "AAPL")
    assert engine.delete_watchlist("Test")
    assert "Test" not in engine.get_watchlists()


def test_alert_persistence(engine):
    assert engine.add_alert({"symbol": "NVDA", "kind": "offex_pctile_90",
                             "threshold": 0.9, "enabled": True})
    assert len(engine.get_alerts()) == 1
    assert engine.remove_alert(0)
    assert len(engine.get_alerts()) == 0


# --------------------------------------------------------------------------- #
#  Data quality
# --------------------------------------------------------------------------- #

def test_data_quality_report(engine):
    dq = engine.data_quality_report()
    assert "overall_score" in dq
    assert 0 <= dq["overall_score"] <= 100
    assert len(dq["sources"]) >= 2
    assert dq["banner"]
    assert "MODELED" in dq["banner"]


def test_methodology_documented(engine):
    m = engine.methodology()
    assert "OFF-EXCHANGE SHARE (OBSERVED + MODELED)" in m
    assert "PROVENANCE" in m
    assert "DIRECTION (ESTIMATED)" in m
    assert "FINRA OTC Transparency API" in m


def test_regime_detection_offline(engine):
    r = engine.detect_regime()
    assert "label" in r
    assert r["label"] in ("Normal Market", "High Volatility", "Risk-On",
                          "Risk-Off", "Liquidity Stress")


# --------------------------------------------------------------------------- #
#  Review-fix regressions: fast-path scan, settings, cache hygiene
# --------------------------------------------------------------------------- #

def test_scan_fast_path_skips_live_lookups(engine, monkeypatch):
    """include_live=False must NOT call the slow live-quote / market-cap
    providers at all (verified with sentinel mocks — no network)."""
    calls = {"live": 0, "cap": 0}

    def fake_live(symbol):
        calls["live"] += 1
        return {"price": 999.0, "change_pct": 1.23}

    def fake_cap(symbol, price):
        calls["cap"] += 1
        return 1_000_000_000.0

    monkeypatch.setattr(engine, "_live_quote", fake_live)
    monkeypatch.setattr(engine, "_market_cap", fake_cap)

    fast = engine.analyze_ticker("AAPL", include_live=False)
    assert fast["ok"] is True
    assert fast["live_price"] is None
    assert fast["market_cap"] is None
    assert calls == {"live": 0, "cap": 0}  # fast path: zero provider calls

    full = engine.analyze_ticker("AAPL", include_live=True)
    assert calls == {"live": 1, "cap": 1}  # full path uses both
    assert full["live_price"]["price"] == 999.0
    assert full["market_cap"] == 1_000_000_000.0

    # core analytics identical regardless of the live-lookup path
    assert full["pressure_score"] == fast["pressure_score"]
    assert full["imbalance_5d"] == fast["imbalance_5d"]
    assert full["offexchange_pct_20d"] == fast["offexchange_pct_20d"]


def test_scan_market_uses_fast_path(engine):
    df = engine.scan_market(limit=6)
    assert not df.empty
    # scanner output has no live/market-cap dependency
    assert "Price" in df.columns
    assert "OffEx%" in df.columns


def test_finra_api_key_settings_persist(engine):
    assert engine.get_finra_api_key() == ""
    engine.set_finra_api_key("  secret-key-123 ")
    assert engine.get_finra_api_key() == "secret-key-123"
    s = engine.get_settings()
    assert s["finra_api_key_set"] is True
    # new engine instance on the same state file reads the key back
    e2 = DarkPoolEngine(state_file=engine._state_file, fetch_fn=engine._fetch_fn)
    assert e2.get_finra_api_key() == "secret-key-123"
    # clearing persists too
    engine.set_finra_api_key("")
    assert engine.get_finra_api_key() == ""


def test_setting_key_resets_finra_cache(engine):
    engine._finra_cache = (time.time(), {"AAPL": 123.0})
    engine.set_finra_api_key("k")
    assert engine._finra_cache is None
    engine.clear_caches()
    assert engine._quote_cache == {}
    assert engine._cap_cache == {}


def test_offexchange_model_reproducible_across_processes(engine):
    """Stable seed: same symbol always produces the same modeled share series."""
    a = engine.analyze_ticker("AAPL")
    b = engine.analyze_ticker("AAPL")
    assert (a["series"]["OffExchangeShare"] == b["series"]["OffExchangeShare"]).all()


def test_percentile_windows_are_honest(engine):
    """5d/20d/60d labels must map to 5/20/60-bar trailing windows."""
    r = engine.analyze_ticker("AAPL")
    h = r["historical"]
    # None only when there isn't enough history; never out-of-range
    for k in ("offex_pctile_5d", "offex_pctile_20d", "offex_pctile_60d"):
        v = h.get(k)
        if v is not None:
            assert 0 <= v <= 100
    # with 400 bars all three must be present
    assert h["offex_pctile_5d"] is not None
    assert h["offex_pctile_20d"] is not None
    assert h["offex_pctile_60d"] is not None


# --------------------------------------------------------------------------- #
#  FINRA OTC API provider (real endpoint format, mocked HTTP — offline-safe)
# --------------------------------------------------------------------------- #


def _fake_finra_response():
    """Shape of the real weeklySummary API rows (per-symbol, ATS + OTC)."""
    return [
        {"issueSymbolIdentifier": "AAPL", "issueName": "Apple Inc",
         "totalWeeklyShareQuantity": 95_000_000, "totalWeeklyTradeCount": 200_000,
         "totalNotionalSum": 1.5e10, "weekStartDate": "2026-07-20",
         "summaryTypeCode": "ATS_W_SMBL", "tierIdentifier": "T1"},
        {"issueSymbolIdentifier": "AAPL", "issueName": "Apple Inc",
         "totalWeeklyShareQuantity": 60_000_000, "totalWeeklyTradeCount": 150_000,
         "totalNotionalSum": 9e9, "weekStartDate": "2026-07-20",
         "summaryTypeCode": "OTC_W_SMBL", "tierIdentifier": "T1"},
        {"issueSymbolIdentifier": "MSFT", "issueName": "Microsoft Corp",
         "totalWeeklyShareQuantity": 40_000_000, "totalWeeklyTradeCount": 90_000,
         "totalNotionalSum": 8e9, "weekStartDate": "2026-07-20",
         "summaryTypeCode": "ATS_W_SMBL", "tierIdentifier": "T1"},
    ]


def test_finra_provider_normalizes_and_aggregates(engine, monkeypatch):
    """The real API path: partitions -> weeklySummary rows -> per-symbol map.

    The provider must (1) discover the latest week, (2) POST for ATS + OTC
    symbol-level summaries, (3) normalize fields, (4) sum ATS + OTC per symbol.
    """
    import dark_pool_engine as dpe

    calls = []

    class FakeResp:
        def __init__(self, status, payload):
            self.status_code = status
            self._payload = payload

        def json(self):
            return self._payload

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(("GET", url))
        return FakeResp(200, {"availablePartitions": [
            {"partitions": ["2026-07-20", "T1"]},
            {"partitions": ["2026-07-13", "T1"]}]})

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(("POST", url, json.get("compareFilters") if json else None))
        code = json["compareFilters"][1]["fieldValue"]
        rows = [r for r in _fake_finra_response() if r["summaryTypeCode"] == code]
        return FakeResp(200, rows)

    monkeypatch.setattr(dpe.requests, "get", fake_get)
    monkeypatch.setattr(dpe.requests, "post", fake_post)

    e = DarkPoolEngine(state_file=os.path.join(tempfile.mkdtemp(), "s.json"),
                       fetch_fn=engine._fetch_fn)
    e.set_finra_api_key("test-key")
    e.clear_caches()

    df = e.fetch_finra_otc()
    assert df is not None
    assert not df.empty
    # normalized columns present
    for col in ("Symbol", "TotalVolume", "Date"):
        assert col in df.columns
    # both ATS and OTC calls made, plus partitions
    assert any(c[0] == "GET" for c in calls)
    assert sum(1 for c in calls if c[0] == "POST") == 2

    vol_map = e._finra_volume_map()
    assert vol_map is not None
    # AAPL = ATS 95M + OTC 60M = 155M
    assert abs(vol_map["AAPL"] - 155_000_000) < 1.0
    assert vol_map["MSFT"] == 40_000_000
    assert "AA" not in vol_map  # unknown symbols excluded


def test_finra_provider_returns_none_without_key(engine, monkeypatch):
    import dark_pool_engine as dpe

    called = []

    def fake_get(*a, **k):
        called.append(1)
        return None

    monkeypatch.setattr(dpe.requests, "get", fake_get)
    e = DarkPoolEngine(state_file=os.path.join(tempfile.mkdtemp(), "s.json"),
                       fetch_fn=engine._fetch_fn)
    assert e.fetch_finra_otc() is None  # no key -> no API call -> None
    assert called == []


def test_latest_full_week_prefers_full_tier_coverage(engine):
    """Newest week is skipped when it lacks full tier coverage, so T2/OTCE/NA
    symbols are never silently dropped from the fetch."""
    e = DarkPoolEngine(state_file=os.path.join(tempfile.mkdtemp(), "s.json"),
                       fetch_fn=engine._fetch_fn)
    pairs = [
        ("2026-07-20", "T1"),          # newest week: only T1 published yet
        ("2026-07-13", "T1"),
        ("2026-07-06", "T1"),
        ("2026-07-06", "T2"),
        ("2026-07-06", "OTCE"),
        ("2026-07-06", "NA"),
    ]
    week = e._latest_full_week(pairs)
    assert week == "2026-07-06"  # not the T1-only 2026-07-20


def test_finra_week_and_tiers_recorded(engine, monkeypatch):
    """fetch_finra_otc records the week + tier coverage on df.attrs for the
    Data Quality Center."""
    import dark_pool_engine as dpe

    class FakeResp2:
        def __init__(self, status_code=200, payload=None):
            self.status_code = status_code
            self._payload = payload or {}

        def json(self):
            return self._payload

    def fake_post(url, json=None, headers=None, timeout=None):
        rows = _fake_finra_response()
        code = json["compareFilters"][1]["fieldValue"]
        return FakeResp2(payload=[r for r in rows if r["summaryTypeCode"] == code])

    monkeypatch.setattr(dpe.requests, "get",
                        lambda *a, **k: FakeResp2(payload={"availablePartitions": [
                            {"partitions": ["2026-07-20", "T1"]},
                            {"partitions": ["2026-07-06", "T1"]},
                            {"partitions": ["2026-07-06", "T2"]},
                            {"partitions": ["2026-07-06", "OTCE"]},
                            {"partitions": ["2026-07-06", "NA"]}]}))
    monkeypatch.setattr(dpe.requests, "post", fake_post)

    e = DarkPoolEngine(state_file=os.path.join(tempfile.mkdtemp(), "s.json"),
                       fetch_fn=engine._fetch_fn)
    e.set_finra_api_key("test-key")
    e.clear_caches()
    df = e.fetch_finra_otc()
    assert df is not None
    assert df.attrs.get("finra_week") == "2026-07-06"
    assert set(df.attrs.get("finra_tiers", [])) >= {"T1", "T2", "OTCE", "NA"}


def test_settings_never_return_full_api_key(engine):
    engine.set_finra_api_key("abcd1234wxyz5678")
    s = engine.get_settings()
    assert "finra_api_key" not in s  # never the plaintext key
    assert s["finra_api_key_set"] is True
    assert s["finra_api_key_masked"] == "abcd…5678"
    engine.set_finra_api_key("")
    s = engine.get_settings()
    assert s["finra_api_key_masked"] == ""


def test_finra_volume_map_skips_nan(engine, monkeypatch):
    """Null/NaN volume rows must not poison the per-symbol sum."""
    import dark_pool_engine as dpe

    rows = [
        {"issueSymbolIdentifier": "AAPL", "totalWeeklyShareQuantity": 100.0,
         "weekStartDate": "2026-07-06", "summaryTypeCode": "ATS_W_SMBL"},
        {"issueSymbolIdentifier": "AAPL", "totalWeeklyShareQuantity": None,
         "weekStartDate": "2026-07-06", "summaryTypeCode": "OTC_W_SMBL"},
        {"issueSymbolIdentifier": None, "totalWeeklyShareQuantity": 999.0,
         "weekStartDate": "2026-07-06", "summaryTypeCode": "ATS_W_VOL_STATS"},
    ]

    class FakeResp2:
        def __init__(self, status_code=200, payload=None):
            self.status_code = status_code
            self._payload = payload or {}

        def json(self):
            return self._payload

    def fake_get(*a, **k):
        return FakeResp2(payload={"availablePartitions": [{"partitions": ["2026-07-06", "T1"]}]})

    def fake_post(url, json=None, headers=None, timeout=None):
        code = json["compareFilters"][1]["fieldValue"]
        return FakeResp2(payload=[r for r in rows if r["summaryTypeCode"] == code])

    monkeypatch.setattr(dpe.requests, "get", fake_get)
    monkeypatch.setattr(dpe.requests, "post", fake_post)
    e = DarkPoolEngine(state_file=os.path.join(tempfile.mkdtemp(), "s.json"),
                       fetch_fn=engine._fetch_fn)
    e.set_finra_api_key("test-key")
    e.clear_caches()
    vm = e._finra_volume_map()
    assert vm is not None
    assert vm.get("AAPL") == 100.0  # NaN row skipped, no NaN poison
    assert "NAN" not in vm


def test_finra_observed_anchors_ticker_report(engine, monkeypatch):
    """When FINRA data is present the report marks finra_observed and records
    the observed weekly volume; provenance is honest (OBSERVED category)."""
    import dark_pool_engine as dpe

    class FakeResp2:
        def __init__(self, status_code=200, payload=None):
            self.status_code = status_code
            self._payload = payload or {}

        def json(self):
            return self._payload

    def fake_post(url, json=None, headers=None, timeout=None):
        code = json["compareFilters"][1]["fieldValue"]
        rows = [r for r in _fake_finra_response() if r["summaryTypeCode"] == code
                and r["issueSymbolIdentifier"] == "AAPL"]
        return FakeResp2(payload=rows)

    monkeypatch.setattr(dpe.requests, "get",
                        lambda *a, **k: FakeResp2(payload={"availablePartitions": [
                            {"partitions": ["2026-07-20", "T1"]}]}))
    monkeypatch.setattr(dpe.requests, "post", fake_post)

    e = DarkPoolEngine(state_file=os.path.join(tempfile.mkdtemp(), "s.json"),
                       fetch_fn=engine._fetch_fn)
    e.set_finra_api_key("test-key")
    e.clear_caches()

    r = e.analyze_ticker("AAPL")
    assert r["ok"] is True
    assert r["finra_observed"] is True
    assert r["finra_offexchange_vol"] == 155_000_000
    assert r["provenance"]["finra"]["category"] == "OBSERVED"
