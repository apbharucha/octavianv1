"""Mock-based verification of fetch_ticker_fundamentals fixes.

Reproduces the two reported bugs without network:
  1. NoneType crash: yfinance `Ticker.info` returning None (the known
     `argument of type 'NoneType' is not iterable` failure mode).
  2. Revenue magnitude bug: AAPL-style company with `totalRevenue` missing
     but `revenuePerShare` present used to produce ~$100M instead of ~$390B.
"""
import os
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# ── Fake yfinance ────────────────────────────────────────────────────────────
fake_yf = types.ModuleType("yfinance")


class _FakeTicker:
    def __init__(self, info=None, rps=None, shares=15_500_000_000.0, financials=None):
        self._info = info
        self._rps = rps
        self._shares = shares
        if financials is None:
            idx = ["EBITDA", "Total Revenue", "Net Income", "Operating Income"]
            self._financials = pd.DataFrame(
                {"2024": [130_000_000_000.0, 390_000_000_000.0, 95_000_000_000.0, 100_000_000_000.0]},
                index=idx,
            )
        else:
            self._financials = financials

    @property
    def info(self):
        if self._info is None:
            raise TypeError("argument of type 'NoneType' is not iterable")
        return self._info

    @property
    def fast_info(self):
        return SimpleNamespace(last_price=223.5, shares=self._shares)

    @property
    def financials(self):
        return self._financials

    def history(self, period="1mo", interval="1d"):
        n = 25
        return pd.DataFrame(
            {"Close": np.linspace(210.0, 223.5, n)},
            index=pd.date_range("2026-07-01", periods=n, freq="D"),
        )


def _install_yf(info=None, rps=None, shares=15_500_000_000.0):
    fake_yf.Ticker = lambda sym: _FakeTicker(info=info, rps=rps, shares=shares)
    sys.modules["yfinance"] = fake_yf
    # ensure data_sources picks up the fake (it imports yfinance lazily inside fns)
    import data_sources
    sys.modules["data_sources"] = data_sources


def test_none_info_does_not_crash(monkeypatch_capture=None):
    import financial_model_generator as fmg

    _install_yf(info=None, rps=25.0)
    # _resolve_live_price calls get_realtime_price -> real yfinance path is
    # monkeypatched out below by making info raise; instead call the pieces.
    from financial_model_generator import _safe_info_fetch, _derive_revenue_millions

    t = _FakeTicker(info=None)
    info = _safe_info_fetch(t)
    assert isinstance(info, dict) and info == {}, f"expected empty dict got {info!r}"

    rev = _derive_revenue_millions({"revenuePerShare": 25.0}, shares_raw=15_500_000_000.0)
    assert 380_000 < rev < 400_000, f"expected ~$390B got ${rev}M"
    print(f"  revenue from per-share fallback: ${rev:,.0f}M  (AAPL-scale OK)")


def test_revenue_magnitude():
    from financial_model_generator import _derive_revenue_millions

    # totalRevenue present -> direct
    rev = _derive_revenue_millions({"totalRevenue": 390_000_000_000.0}, 1.0)
    assert 389_000 < rev < 391_000, f"got {rev}"
    # per-share fallback
    rev = _derive_revenue_millions({"revenuePerShare": 25.16}, shares_raw=15_500_000_000.0)
    assert 389_000 < rev < 391_000, f"got {rev}"
    # nothing at all -> 0 (no fabricated $100M)
    rev = _derive_revenue_millions({}, 0.0)
    assert rev == 0.0
    print("  revenue magnitude tests passed")


def test_extract_ebitda_no_fabrication():
    import financial_model_generator as fmg
    from financial_model_generator import _extract_ebitda_millions
    # company with no financials -> 0, not a made-up 500
    t = _FakeTicker(info={"totalRevenue": 1.0}, financials=pd.DataFrame())
    v = _extract_ebitda_millions(t, {"totalRevenue": 1.0})
    assert v == 0.0, f"expected 0 got {v}"
    # company with real EBITDA -> real number
    t2 = _FakeTicker(info={})
    v2 = _extract_ebitda_millions(t2, {})
    assert abs(v2 - 130_000) < 1000, f"got {v2}"
    print(f"  ebitda extraction OK: {v2:,.0f}M")


if __name__ == "__main__":
    print("Running auto-fill fix tests (mocked, no network)...")
    test_none_info_does_not_crash()
    test_revenue_magnitude()
    test_extract_ebitda_no_fabrication()
    print("ALL AUTO-FILL TESTS PASSED")
