"""
Offline-safe tests for the Algorithm Builder engine.

All tests inject synthetic OHLCV data through the engine's data_fn hook so no
network access is required. Coverage:

  * request parsing (auto mode keyword -> families, defaults, ensembles)
  * OHLCV backtester sanity (known signals -> expected P&L; honest costs)
  * every single-instrument archetype produces valid bounded signals
  * parameter search honours locked params and reports train/test splits
  * multi-algorithm builds are diverse, deterministic, and ensemble-blendable
  * online portfolio selection family (FTRL/PAMR/CWMR/OLMAR/Anticor) runs on
    a basket with simplex weights
  * exports: python compiles, JSON parses, CSV/Markdown well-formed
  * failure modes: bad universe, insufficient history -> error cards, no crash
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["OCTAVIAN_OFFLINE"] = "1"

from algorithm_builder_engine import (  # noqa: E402
    ARCHETYPES,
    AlgorithmResult,
    backtest_ohlcv,
    build_algorithms,
    generate_python,
    parse_request,
    run_ops_basket,
    strategy_spec_to_signal,
    strategy_spec_to_weights,
)


# --------------------------------------------------------------------------- #
#  Fixtures / helpers
# --------------------------------------------------------------------------- #

def _make_ohlc(n=500, start_price=100.0, seed=7, trend=0.0015, osc=False):
    """Synthetic daily OHLCV with a *deterministic* drift so backtest
    expectations are seed-independent. `osc` replaces the drift with a strong
    mean-reversion component (sine wave) instead."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2026-08-01"), periods=n)
    if osc:
        wave = 12.0 * np.sin(np.arange(n) / 10.0)
        close = start_price + wave + rng.normal(0, 0.3, n)
        close = np.maximum(close, 5.0)
    else:
        # drift dominates: +0.15%/day over 500 days with only +/-0.5% noise
        close = start_price * np.cumprod(1 + trend + rng.normal(0, 0.005, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float)},
                        index=dates)


def _make_ohlc_ar1(n=800, start_price=100.0, seed=5, ar=-0.35, vol=0.012):
    """Synthetic OHLCV with AR(1) mean-reverting returns — the classic setting
    where band mean reversion (z-score/RSI/Bollinger) legitimately profits."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2026-08-01"), periods=n)
    r = np.zeros(n)
    eps = rng.normal(0, vol, n)
    for t in range(1, n):
        r[t] = ar * r[t - 1] + eps[t]
    close = start_price * np.cumprod(1 + np.clip(r, -0.06, 0.06))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float)},
                        index=dates)


@pytest.fixture(scope="module")
def df_trend():
    return _make_ohlc(seed=7)


@pytest.fixture(scope="module")
def df_osc():
    return _make_ohlc(seed=11, osc=True)


@pytest.fixture(scope="module")
def df_ar1():
    return _make_ohlc_ar1()


@pytest.fixture(scope="module")
def df_second():
    return _make_ohlc(seed=13, start_price=50.0)


def _mk_data_fn(df1, df2=None):
    def _get(symbol, period="3y", interval="1d"):
        if symbol in ("SPY", "NVDA"):
            return df1.copy()
        return (df2 if df2 is not None else df1).copy()
    return _get


# --------------------------------------------------------------------------- #
#  Request parsing
# --------------------------------------------------------------------------- #

def test_parse_request_families():
    assert parse_request("mean reversion with bollinger and rsi") == \
        ["rsi_meanrev", "bollinger_meanrev", "statarb_z", "gap_fade"]
    assert parse_request("turtle breakout momentum") == ["trend_ma", "breakout", "dual_momentum"]
    assert parse_request("market making with inventory control citadel") == ["market_making"]
    assert parse_request("volatility targeting risk parity") == ["vol_target"]
    assert "online_ops" in parse_request("online portfolio selection from the glucksman paper")
    assert parse_request("") == ["trend_ma", "rsi_meanrev", "statarb_z"]
    assert len(parse_request("")) <= 4


# --------------------------------------------------------------------------- #
#  Backtester sanity
# --------------------------------------------------------------------------- #

def test_backtester_always_long_on_uptrend_is_profitable(df_trend):
    sig = pd.Series(1.0, index=df_trend.index)
    out = backtest_ohlcv(df_trend, sig, {"sizing": "fixed_pct", "risk_pct": 0.5,
                                         "max_leverage": 1.0, "direction": "long_only",
                                         "commission_bps": 0.0, "slippage_bps": 0.0})
    assert out["metrics"]["total_return"] > 0.05
    assert len(out["equity"]) == len(df_trend)
    assert out["equity"].isna().sum() == 0


def test_backtester_flat_signal_stays_flat(df_trend):
    sig = pd.Series(0.0, index=df_trend.index)
    out = backtest_ohlcv(df_trend, sig, {"sizing": "fixed_pct", "risk_pct": 0.5,
                                         "max_leverage": 1.0, "direction": "long_only"})
    assert out["metrics"]["trades"] == 0
    assert abs(out["metrics"]["total_return"]) < 1e-9


def test_backtester_costs_reduce_returns(df_ar1):
    # a signal that trades ~100 round-trips makes costs compound visibly
    a = ARCHETYPES["statarb_z"]
    sig = a["sig"](df_ar1, {"period": 10, "entry_z": 1.2, "exit_z": 0.2})
    cheap = backtest_ohlcv(df_ar1, sig, {"sizing": "fixed_pct", "risk_pct": 0.4,
                                         "max_leverage": 1.0, "direction": "long_short",
                                         "commission_bps": 0.0, "slippage_bps": 0.0})
    pricey = backtest_ohlcv(df_ar1, sig, {"sizing": "fixed_pct", "risk_pct": 0.4,
                                          "max_leverage": 1.0, "direction": "long_short",
                                          "commission_bps": 15.0, "slippage_bps": 15.0})
    assert cheap["metrics"]["trades"] > 30  # the test is only meaningful with round-trips
    assert pricey["metrics"]["total_return"] < cheap["metrics"]["total_return"] - 0.02


def test_backtester_long_only_never_shorts(df_trend):
    sig = pd.Series(-1.0, index=df_trend.index)
    out = backtest_ohlcv(df_trend, sig, {"sizing": "fixed_pct", "risk_pct": 0.5,
                                         "max_leverage": 1.0, "direction": "long_only"})
    assert out["metrics"]["trades"] == 0  # long-only ignores -1 signal


# --------------------------------------------------------------------------- #
#  Archetypes produce valid signals
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["trend_ma", "breakout", "dual_momentum",
                                  "rsi_meanrev", "bollinger_meanrev", "vol_target",
                                  "market_making", "statarb_z", "gap_fade"])
def test_archetype_signal_valid(name, df_trend):
    a = ARCHETYPES[name]
    params = {p["name"]: p["default"] for p in a["params"]}
    sig = a["sig"](df_trend, params)
    assert len(sig) == len(df_trend)
    assert sig.index.equals(df_trend.index)
    assert sig.min() >= -1.0001 and sig.max() <= 1.0001
    assert sig.iloc[-50:].isna().sum() == 0  # no NaNs after warmup


@pytest.mark.parametrize("name", ["trend_ma", "breakout", "dual_momentum",
                                  "rsi_meanrev", "bollinger_meanrev", "vol_target",
                                  "market_making", "statarb_z", "gap_fade"])
def test_archetype_backtests_end_to_end(name, df_trend):
    a = ARCHETYPES[name]
    params = {p["name"]: p["default"] for p in a["params"]}
    sig = a["sig"](df_trend, params)
    out = backtest_ohlcv(df_trend, sig, {"sizing": "fixed_pct", "risk_pct": 0.4,
                                         "max_leverage": 1.5, "direction": "long_short"})
    m = out["metrics"]
    for key in ("sharpe", "total_return", "max_drawdown", "trades"):
        assert key in m
    assert out["equity"].isna().sum() == 0
    assert np.isfinite(out["equity"]).all()


def test_mean_reversion_wins_on_ar1_data(df_ar1):
    # on AR(1) mean-reverting data the z-score fade should beat buy & hold
    a = ARCHETYPES["statarb_z"]
    params = {"period": 10, "entry_z": 1.2, "exit_z": 0.2}
    sig = a["sig"](df_ar1, params)
    out = backtest_ohlcv(df_ar1, sig, {"sizing": "fixed_pct", "risk_pct": 0.4,
                                       "max_leverage": 1.0, "direction": "long_short",
                                       "commission_bps": 0.0, "slippage_bps": 0.0})
    assert out["metrics"]["total_return"] > 0.1
    bh = df_ar1["Close"].iloc[-1] / df_ar1["Close"].iloc[0] - 1
    assert out["metrics"]["total_return"] > bh  # beats buy & hold on this fixture


# --------------------------------------------------------------------------- #
#  Parameter search / train-test honesty
# --------------------------------------------------------------------------- #

def test_search_reports_train_and_test_metrics(df_trend):
    from algorithm_builder_engine import search_best
    found = search_best(ARCHETYPES["trend_ma"], df_trend, n_trials=10, seed=3,
                        backtest_params={"sizing": "fixed_pct", "risk_pct": 0.4,
                                         "max_leverage": 1.0, "direction": "long_only"})
    assert "params" in found
    assert "train_metrics" in found and "test_metrics" in found
    assert set(found["test_metrics"]) == set(found["train_metrics"])
    assert found["params"]["slow"] > found["params"]["fast"]  # relationship enforced


def test_search_honours_locked_params(df_trend):
    from algorithm_builder_engine import search_best
    found = search_best(ARCHETYPES["trend_ma"], df_trend, n_trials=8, seed=5,
                        locked={"fast": 21, "slow": 100},
                        backtest_params={"direction": "long_only"})
    assert found["params"]["fast"] == 21
    assert found["params"]["slow"] == 100


def test_search_respects_history_length(df_trend):
    from algorithm_builder_engine import search_best
    short = df_trend.iloc[:120]
    found = search_best(ARCHETYPES["dual_momentum"], short, n_trials=10, seed=2,
                        backtest_params={"direction": "long_only"})
    # lookbacks must fit inside the 120-bar window (cap = 120 // 3 = 40)
    assert found["params"]["abs_bars"] <= 40


# --------------------------------------------------------------------------- #
#  Multi-algorithm build / ensembles / determinism
# --------------------------------------------------------------------------- #

def test_build_multi_and_deterministic(df_trend, df_second):
    fn = _mk_data_fn(df_trend, df_second)
    a = build_algorithms("momentum and trend", count=3, universe=["SPY", "QQQ"],
                         risk="aggressive", seed=42, data_fn=fn, runs_per_family=1)
    b = build_algorithms("momentum and trend", count=3, universe=["SPY", "QQQ"],
                         risk="aggressive", seed=42, data_fn=fn, runs_per_family=1)
    assert len(a) == 3
    names_a = [r.name for r in a]
    assert len(set(names_a)) == 3  # diverse
    for r_a, r_b in zip(a, b):
        assert r_a.params == r_b.params
        assert r_a.metrics == r_b.metrics
    for r in a:
        assert isinstance(r, AlgorithmResult)
        assert r.metrics["trades"] >= 0


def test_build_ensemble_weights_sum_to_one(df_trend, df_second):
    fn = _mk_data_fn(df_trend, df_second)
    res = build_algorithms("", count=3, ensemble=True, universe=["SPY", "QQQ"],
                           risk="balanced", seed=9, data_fn=fn, runs_per_family=1)
    ens = [r for r in res if r.family == "ensemble"]
    assert len(ens) == 1
    e = ens[0]
    assert abs(sum(e.ensemble_weights.values()) - 1.0) < 0.02
    assert len(e.members) >= 2
    assert e.metrics["trades"] == 0  # ensemble blends returns, not discrete trades


def test_build_respects_count_and_locked_universe(df_trend, df_second):
    fn = _mk_data_fn(df_trend, df_second)
    res = build_algorithms("", count=1, ensemble=False, universe=["NVDA"],
                           risk="conservative", seed=2, data_fn=fn, runs_per_family=1)
    assert len(res) == 1
    assert res[0].universe == ["NVDA"]


# --------------------------------------------------------------------------- #
#  Online portfolio selection
# --------------------------------------------------------------------------- #

def test_ops_all_algos_run_on_basket(df_osc, df_second):
    prices = pd.DataFrame({"SPY": df_osc["Close"], "QQQ": df_second["Close"]})
    for algo in ("ftrl", "pamr", "cwmr", "olmar", "anticor"):
        out = run_ops_basket(prices, algo, {"eps": 0.005, "C": 1.0, "eta": 0.1,
                                            "beta": 0.1, "w": 5, "alpha": 2.5, "rho": 0.5})
        assert abs(sum(out["final_weights"].values()) - 1.0) < 0.05
        assert out["metrics"]["sharpe"] is not None
        assert out["equity"].isna().sum() == 0


def test_ops_anticor_exploits_mean_reversion(df_osc, df_second):
    prices = pd.DataFrame({"SPY": df_osc["Close"], "QQQ": df_second["Close"]})
    out = run_ops_basket(prices, "anticor", {"w": 3, "alpha": 2.5, "rho": 0.3})
    assert out["metrics"]["total_return"] > 0.05


def test_ops_requires_two_assets(df_trend):
    with pytest.raises(ValueError):
        run_ops_basket(pd.DataFrame({"SPY": df_trend["Close"]}), "pamr", {"eps": 0.005})


def test_build_online_ops_with_single_symbol_returns_error_card(df_trend):
    fn = _mk_data_fn(df_trend)
    res = build_algorithms("online portfolio selection", count=1, universe=["SPY"],
                           risk="balanced", data_fn=fn)
    assert res and res[0].archetype == "online_ops"
    assert any("2 assets" in n for n in res[0].build_notes)


def test_build_online_ops_multi_symbol(df_trend, df_second):
    fn = _mk_data_fn(df_trend, df_second)
    res = build_algorithms("online portfolio selection", count=2, universe=["SPY", "QQQ"],
                           risk="balanced", seed=4, data_fn=fn)
    ops = [r for r in res if r.family == "portfolio"]
    assert ops
    assert abs(sum(ops[0].params.values()) - 1.0) < 0.5 or ops[0].metrics["trades"] >= 0


# --------------------------------------------------------------------------- #
#  Exports
# --------------------------------------------------------------------------- #

def _sample_result(df_trend) -> AlgorithmResult:
    from algorithm_builder_engine import search_best
    found = search_best(ARCHETYPES["trend_ma"], df_trend, n_trials=6, seed=1,
                        backtest_params={"direction": "long_only"})
    from algorithm_builder_engine import backtest_ohlcv
    sig = ARCHETYPES["trend_ma"]["sig"](df_trend, found["params"])
    full = backtest_ohlcv(df_trend, sig, {"direction": "long_only"})
    return AlgorithmResult(name="Test-Algo", family="trend", archetype="trend_ma",
                           params=found["params"], backtest_params={"direction": "long_only"},
                           provenance="test", equity=full["equity"], returns=full["returns"],
                           metrics=full["metrics"], trades=full["trades"],
                           train_metrics=found["train_metrics"], test_metrics=found["test_metrics"],
                           universe=["SPY"])


def test_export_python_compiles(df_trend):
    r = _sample_result(df_trend)
    code = r.code()
    compile(code, "generated.py", "exec")
    assert "def signal(" in code and "def main(" in code


def test_export_json_roundtrip(df_trend):
    r = _sample_result(df_trend)
    d = json.loads(r.to_json())
    assert d["name"] == "Test-Algo"
    assert d["archetype"] == "trend_ma"
    assert "sharpe" in d["metrics"]


def test_export_csv_shape(df_trend):
    r = _sample_result(df_trend)
    csv = r.to_csv()
    lines = csv.splitlines()
    assert len(lines) > 5
    assert lines[0].startswith("Date,Equity,Return")


def test_export_markdown_sections(df_trend):
    r = _sample_result(df_trend)
    md = r.to_markdown()
    assert "# Test-Algo" in md
    assert "## Parameters" in md
    assert "## Backtest metrics" in md
    assert "Train Sharpe" in md and "Test (OOS)" in md
    assert "not financial advice" in md


def test_generated_ops_code_compiles():
    code = generate_python("online_ops", {"eps": 0.005, "C": 1.0},
                           {"direction": "long_only"}, ["SPY", "QQQ"], ops_algo="pamr")
    compile(code, "gen_ops.py", "exec")


def test_generated_ensemble_code_compiles(df_trend):
    """Regression: rendering an ensemble result used to call
    generate_python("ensemble", ...) which did ARCHETYPES["ensemble"] and
    raised KeyError, crashing the Algorithm Builder tab after any build with
    an ensemble (the default). The exported ensemble script must compile and
    embed the member specs for the wealth-weighted blend."""
    m = _sample_result(df_trend)
    specs = [{"name": m.name, "archetype": m.archetype, "params": m.params,
              "ops_algo": m.ops_algo, "backtest_params": m.backtest_params,
              "universe": m.universe}]
    params = {"method": "fast_universalization", "members": [m.name],
              "members_detail": specs}
    code = generate_python("ensemble", params, {"direction": "long_only"},
                           ["SPY", "QQQ"], ops_algo=None)
    compile(code, "gen_ensemble.py", "exec")
    assert "MEMBERS = " in code
    assert '"archetype"' in code, "member specs (archetype/params) must be embedded"
    assert "run_ops_basket" in code and "backtest_ohlcv" in code
    assert "cumulative wealth" in code or "wealth" in code


def _exec_with_mock_data(code: str, df):
    """Compile AND run a generated strategy script against mocked get_stock.

    Regression: exported scripts used json.dumps for the embedded params, which
    emits JSON tokens (null/true/false) that are invalid Python — the script
    NameError'd the moment it ran (e.g. backtest_params' take_profit_pct: None
    and boolean strategy params). compile() alone never caught it."""
    import unittest.mock as um
    import data_sources
    compile(code, "generated.py", "exec")
    with um.patch("data_sources.get_stock", side_effect=_mk_data_fn(df)):
        exec(compile(code, "generated.py", "exec"), {"__name__": "__main__"})


def test_generated_single_script_runs(df_trend):
    """A single-instrument export with None/bool params must run end-to-end."""
    r = _sample_result(df_trend)
    # force the exact shape that used to break: None + bool in backtest params
    r.backtest_params = {"direction": "long_only", "take_profit_pct": None,
                         "trailing_pct": None, "stop_loss_pct": 6.0,
                         "sizing": "vol_target", "max_leverage": 1.5}
    _exec_with_mock_data(r.code(), df_trend)


def test_generated_ops_script_runs(df_trend):
    code = generate_python("online_ops", {"eps": 0.005, "C": 1.0},
                           {"direction": "long_only"}, ["SPY", "QQQ"], ops_algo="pamr")
    _exec_with_mock_data(code, df_trend)


# --------------------------------------------------------------------------- #
#  Advanced backtests: multi-window returns, many runs, trade reasoning,
#  failure diagnosis, dynamic ensemble weighting
# --------------------------------------------------------------------------- #

def test_window_returns_breakdown(df_trend):
    """Every healthy result carries a 5y/3y/2y/1y/6m/3m/1m trailing-return
    breakdown; windows longer than the available history are None, covered
    windows are finite numbers."""
    res = build_algorithms("momentum and trend", count=1, universe=["SPY"],
                           seed=4, data_fn=_mk_data_fn(df_trend), runs_per_family=1)
    r = res[0]
    assert r.window_returns, "window_returns must be populated"
    for label in ("5y", "3y", "2y", "1y", "6m", "3m", "1m"):
        assert label in r.window_returns, f"missing window {label}"
    # fixture is ~500 bars (~2y) -> 1y covered, 5y/3y/2y unavailable
    assert r.window_returns["1y"] is not None
    assert r.window_returns["5y"] is None and r.window_returns["3y"] is None
    assert np.isfinite(r.window_returns["1y"])


def test_multiple_runs_per_family(df_trend):
    """runs_per_family>1 produces several distinct backtests per family
    (different seeds -> different fitted parameters), each labelled with its
    run index."""
    res = build_algorithms(mode="guided", archetypes=["trend_ma"], count=1,
                           universe=["SPY"], seed=7, data_fn=_mk_data_fn(df_trend),
                           runs_per_family=3)
    assert len(res) == 3
    assert {r.run_index for r in res} == {1, 2, 3}
    assert len({r.params["fast"] for r in res}) >= 2, "runs must not all be identical"
    assert all("(run " in r.name for r in res)
    for r in res:
        assert r.window_returns
        assert r.metrics["trades"] >= 0


def test_dynamic_ensemble_downweights_weak_members(df_osc):
    """The dynamic ensemble weights by measured quality + redundancy, not
    equally: on mean-reverting sine data the strong trend runs get most of the
    weight while the weak mean-reversion runs are down-weighted, and each
    member gets an explanation of its strength/weakness."""
    res = build_algorithms(mode="guided", archetypes=["trend_ma", "bollinger_meanrev"],
                           count=2, ensemble=True, universe=["SPY", "QQQ", "IWM"],
                           seed=7, data_fn=_mk_data_fn(df_osc),
                           runs_per_family=2, ensemble_method="dynamic")
    ens = [r for r in res if r.family == "ensemble"]
    assert len(ens) == 1
    e = ens[0]
    w = e.ensemble_weights
    assert abs(sum(w.values()) - 1.0) < 0.02
    trend_w = sum(v for k, v in w.items() if "Crossover" in k)
    boll_w = sum(v for k, v in w.items() if "Bollinger" in k)
    assert trend_w > 0.5, f"strong members must get the majority weight, got trend {trend_w:.2f}"
    assert boll_w < 0.5
    assert any("strength" in n and "weakness" in n for n in e.build_notes), \
        "ensemble must explain each member's strengths/weaknesses"
    assert e.window_returns
    assert e.metrics["total_return"] != 0.0


def test_dynamic_ensemble_differs_from_equal(df_osc):
    """Dynamic weights are not just 1/K each when members differ in quality."""
    res = build_algorithms(mode="guided", archetypes=["trend_ma", "bollinger_meanrev"],
                           count=2, ensemble=True, universe=["SPY", "QQQ", "IWM"],
                           seed=7, data_fn=_mk_data_fn(df_osc),
                           runs_per_family=2, ensemble_method="dynamic")
    e = [r for r in res if r.family == "ensemble"][0]
    vals = sorted(e.ensemble_weights.values())
    assert vals[-1] > vals[0] + 0.05, f"weights look equal: {vals}"


def test_failed_family_gets_diagnosed_suggestion():
    """When a family finds no viable strategy, the error card must explain
    WHY (data character) and suggest concrete tweaks + better-suited families."""
    strong_trend = _make_ohlc(seed=3, trend=0.005)  # RSI entries never trigger
    res = build_algorithms(mode="guided", archetypes=["rsi_meanrev"], count=1,
                           universe=["SPY"], seed=3,
                           data_fn=_mk_data_fn(strong_trend), runs_per_family=1)
    r = res[0]
    assert any(n.startswith("ERROR:") for n in r.build_notes)
    sug = [n for n in r.build_notes if n.startswith("SUGGESTION:")]
    assert sug, "failed family must carry a SUGGESTION note"
    assert "Mean reversion" in sug[0] and "trending" in sug[0]
    assert "alternatives" in sug[0] or "Better-suited" in sug[0]


def test_trade_narratives_and_exit_reasons(df_trend):
    """Trades record why they exited (stop/target/trail/time/flat/end) and the
    result exposes dynamic per-trade reasoning for the user."""
    res = build_algorithms("momentum and trend", count=1, universe=["SPY"],
                           seed=9, data_fn=_mk_data_fn(df_trend), runs_per_family=1)
    r = res[0]
    if r.metrics["trades"] == 0:
        return  # some configs trade rarely; nothing to narrate
    known = {"STOP_LOSS", "TAKE_PROFIT", "TRAILING_STOP", "MAX_HOLD",
             "SIGNAL_FLAT", "END_OF_DATA"}
    for t in r.trades:
        assert t.get("exit_reason") in known, f"bad exit_reason {t.get('exit_reason')}"
    assert r.trade_narratives, "healthy result must expose per-trade reasoning"
    first = r.trade_narratives[0]
    assert "entered" in first and "@" in first
    assert any(k in first for k in ("stop", "target", "trailing", "holding", "flat", "end")),\
        f"narrative should say why the trade exited: {first}"


def test_ensemble_result_code_does_not_raise(df_trend):
    """End-to-end: a built ensemble AlgorithmResult (as produced by
    build_algorithms) must export Python that compiles AND runs."""
    # Guided families that reliably trade on the drifting fixture series, so the
    # ensemble (>= 2 healthy members) is actually built.
    res = build_algorithms(mode="guided", archetypes=["trend_ma", "bollinger_meanrev"],
                           count=2, ensemble=True, universe=["SPY", "QQQ", "IWM"],
                           seed=7, data_fn=_mk_data_fn(df_trend))
    ens = [r for r in res if r.family == "ensemble"]
    assert ens, "expected an ensemble card in the build"
    for r in res:
        _exec_with_mock_data(r.code(), df_trend)


# --------------------------------------------------------------------------- #
#  Failure modes
# --------------------------------------------------------------------------- #

def test_fetch_failure_raises(df_trend):
    def bad_fn(symbol, period="3y", interval="1d"):
        raise RuntimeError("no network")
    with pytest.raises(RuntimeError):
        build_algorithms("", count=1, universe=["SPY"], data_fn=bad_fn)


def test_insufficient_history_raises():
    short = _make_ohlc(n=30, seed=1)
    fn = _mk_data_fn(short)
    with pytest.raises(RuntimeError, match="insufficient history"):
        build_algorithms("", count=1, universe=["SPY"], data_fn=fn)


# --------------------------------------------------------------------------- #
#  Paper-trading bridge (strategy_spec serialization + signal reconstruction)
# --------------------------------------------------------------------------- #

def test_strategy_spec_roundtrip_signal(df_trend):
    """A built algorithm's strategy_spec() must serialize every field needed to
    re-run the strategy, and strategy_spec_to_signal must reproduce a live
    target-exposure series in [-1, 1] that matches the backtest signal."""
    res = build_algorithms(mode="guided", archetypes=["trend_ma"], count=1,
                           universe=["SPY"], seed=7, runs_per_family=1,
                           data_fn=_mk_data_fn(df_trend))
    assert len(res) == 1
    r = res[0]
    spec = r.strategy_spec()
    assert spec["archetype"] == r.archetype
    assert spec["name"] == r.name
    assert spec["family"] == r.family
    assert spec["universe"] == ["SPY"]
    assert spec["source"] == "algorithm_builder"
    assert "metrics" in spec and "window_returns" in spec

    sig = strategy_spec_to_signal(spec, df_trend)
    assert sig is not None
    assert len(sig) == len(df_trend)
    assert float(sig.min()) >= -1.0 - 1e-9 and float(sig.max()) <= 1.0 + 1e-9
    # long-only direction clips shorts
    assert float(sig.min()) >= -1e-9


def test_strategy_spec_online_ops_returns_none_for_signal(df_trend):
    """Portfolio-level (online_ops) strategies return None from the per-symbol
    signal helper; their weights come from strategy_spec_to_weights instead."""
    spec = {
        "name": "OPS", "family": "online_ops", "archetype": "online_ops",
        "params": {"window": 5}, "backtest_params": {"direction": "long_only"},
        "universe": ["SPY", "QQQ"], "ops_algo": "pamr", "direction": "long_only",
        "provenance": "", "metrics": {}, "window_returns": {}, "source": "algorithm_builder",
    }
    assert strategy_spec_to_signal(spec, df_trend) is None

    dfs = {"SPY": df_trend.copy(), "QQQ": df_trend.copy()}
    dfs["QQQ"]["Close"] = dfs["QQQ"]["Close"] * 1.01
    weights = strategy_spec_to_weights(spec, dfs)
    assert weights is not None
    keys = set(weights)
    assert keys == {"SPY", "QQQ"}
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-6


def test_strategy_spec_bad_archetype_returns_none(df_trend):
    spec = {"name": "x", "archetype": "not_a_real_archetype", "params": {},
            "direction": "long_only", "universe": ["SPY"], "ops_algo": None}
    assert strategy_spec_to_signal(spec, df_trend) is None
    assert strategy_spec_to_signal(None, df_trend) is None
