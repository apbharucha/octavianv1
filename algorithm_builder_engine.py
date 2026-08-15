"""
Algorithm Builder Engine
========================

A research-grounded algorithmic strategy generator. Builds, backtests, and
exports quantitative trading algorithms from either vague natural-language
requests (auto mode) or explicit structured constraints (guided mode), within
the limits of what the user asks for while maximizing the quality of the
resulting algorithm.

Research provenance of the strategy library
-------------------------------------------
* Glucksman Fellowship paper (NYU Stern, "Online Quantitative Trading
  Strategies", Lahanis/Liu/Zhou): online portfolio selection algorithms —
  Follow-the-Regularized-Leader (FTRL, Sharpe ~1.04), Confidence-Weighted Mean
  Reversion (CWMR, Sharpe ~1.75), Passive-Aggressive Mean Reversion (PAMR,
  Sharpe ~1.63), Online Moving-Average Reversion (OLMAR), Robust Median
  Reversion (RMR), Anticor, pattern-matching, and Fast-Universalization /
  Online-Gradient-Update meta-ensembles which beat every single strategy.
* QuantConnect strategy library & community forum: RSI(2)-style mean
  reversion with a low-volatility filter, dual-momentum (Antonacci) with
  absolute-momentum gating, Donchian/Turtle breakouts, volatility targeting.
* Market-structure style notes (Citadel / Jane Street / Optiver): z-score
  statistical arbitrage and inventory-skewed two-sided market making. Daily
  OHLCV data can only *approximate* these; the labels stay honest about that.

Every generated algorithm carries a `provenance` string describing exactly
which source(s) inspired it. Nothing here is financial advice.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

try:  # allow the engine to be imported in tests without the data layer
    from data_sources import get_stock as _default_get_stock
except Exception:  # pragma: no cover - only hit in exotic import orders
    def _default_get_stock(symbol, period="3y", interval="1d"):  # type: ignore
        raise RuntimeError("data_sources unavailable")

BARS_PER_YEAR = 252  # daily bar convention used throughout
RF = 0.0             # risk-free rate for Sharpe (0 keeps comparisons honest)

# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

# Trailing-window breakdown shown for every backtest: (label, trading bars).
WINDOW_SPECS = [
    ("5y", 5 * BARS_PER_YEAR), ("3y", 3 * BARS_PER_YEAR), ("2y", 2 * BARS_PER_YEAR),
    ("1y", BARS_PER_YEAR), ("6m", 126), ("3m", 63), ("1m", 21),
]


def window_returns(equity: pd.Series) -> dict:
    """Trailing returns of the equity curve over 5y/3y/2y/1y/6m/3m/1m windows.

    Each window uses the most recent N trading bars of the strategy's own
    equity curve, so short backtests simply omit the windows they cannot cover.
    """
    out = {}
    if equity is None or len(equity) < 2:
        for label, _ in WINDOW_SPECS:
            out[label] = None
        return out
    eq = equity.dropna()
    for label, bars in WINDOW_SPECS:
        if len(eq) <= bars:
            out[label] = None
            continue
        start = eq.iloc[-1 - bars]
        end = eq.iloc[-1]
        out[label] = float(end / start - 1.0) if start and start > 0 else None
    return out


def window_metrics(equity: pd.Series, returns: pd.Series, trades: list) -> dict:
    """Per-window breakdown (5y/3y/2y/1y/6m/3m/1m) of return, Sharpe, max
    drawdown and trade count — the 'what happened in this time period' table.

    Each window is sliced from the most recent N trading bars of the strategy's
    own equity/returns series; trades are attributed to the window they EXITED
    in (the exit bar decides), so window trade counts sum to the full run.
    """
    out = {}
    if equity is None or len(equity) < 2 or returns is None:
        for label, _ in WINDOW_SPECS:
            out[label] = None
        return out
    eq = equity.dropna()
    ret = returns.dropna()
    for label, bars in WINDOW_SPECS:
        if len(eq) <= bars:
            out[label] = None
            continue
        eq_w = eq.iloc[-1 - bars:]
        ret_w = ret.iloc[-1 - bars:]
        total = float(eq_w.iloc[-1] / eq_w.iloc[0] - 1.0) if eq_w.iloc[0] > 0 else 0.0
        sd = float(ret_w.std())
        sharpe = float(ret_w.mean() / sd * math.sqrt(bars_per_year := 252)) if sd > 0 else 0.0
        peak = eq_w.cummax()
        dd = float((eq_w / peak - 1.0).min())
        if trades:
            exit_ts = pd.to_datetime([t.get("exit_date") for t in trades if t.get("exit_date")],
                                     errors="coerce")
            start_ts = eq_w.index[0]
            n_trades = int((exit_ts >= start_ts).sum())
        else:
            n_trades = 0
        out[label] = {"total_return": total, "sharpe": sharpe, "max_drawdown": dd,
                      "trades": n_trades}
    return out


def rank_factors(dfs: dict) -> pd.DataFrame:
    """Cross-sectional factor ranking of a symbol universe (WorldQuant retail
    smart-beta methodology: z-score each factor, weighted-average composite).

    Uses price-based factors computable from OHLCV (the engine has no
    fundamentals): 12-2 momentum (Asness et al.), a value proxy (depth below
    trailing 1y average), low-volatility, trend strength (distance above its
    200d average) and volume momentum. Returns a DataFrame of z-scores, the
    composite, and a rank (higher = more attractive), sorted by composite.
    """
    rows = []
    for sym, df in dfs.items():
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna().astype(float)
        if len(close) < 63:
            continue
        mom = float(close.iloc[-21] / close.iloc[-252] - 1.0) if len(close) > 252 else float("nan")
        value = float(close.iloc[-1] / close.iloc[-252:].mean() - 1.0)
        vol = float(close.pct_change().tail(126).std() * math.sqrt(252))
        trend = float(close.iloc[-1] / close.iloc[-200:].mean() - 1.0) if len(close) >= 200 else float("nan")
        vol_mom = float("nan")
        if "Volume" in df.columns:
            v = df["Volume"]
            if isinstance(v, pd.DataFrame):
                v = v.iloc[:, 0]
            v = v.dropna().astype(float)
            if len(v) >= 42:
                vol_mom = float(v.iloc[-21:].mean() / max(v.iloc[-252:-21].mean(), 1e-9) - 1.0)
        rows.append({"symbol": sym, "mom_12_2": mom, "value_proxy": value,
                     "low_vol": -vol, "trend": trend, "volume_mom": vol_mom})
    if not rows:
        return pd.DataFrame(columns=["symbol", "mom_12_2", "value_proxy", "low_vol",
                                     "trend", "volume_mom", "composite", "rank"])
    fdf = pd.DataFrame(rows).set_index("symbol")
    for col in ("mom_12_2", "value_proxy", "low_vol", "trend", "volume_mom"):
        mu, sd = fdf[col].mean(), fdf[col].std()
        fdf[col + "_z"] = (fdf[col] - mu) / sd if sd and sd > 0 else 0.0
        fdf[col + "_z"] = fdf[col + "_z"].fillna(0.0)
    # composite = equal-weight average of the available factor z-scores
    zcols = [c for c in ("mom_12_2_z", "value_proxy_z", "low_vol_z", "trend_z", "volume_mom_z")
             if c in fdf.columns]
    fdf["composite"] = fdf[zcols].mean(axis=1)
    fdf = fdf.sort_values("composite", ascending=False)
    fdf["rank"] = range(1, len(fdf) + 1)
    fdf = fdf.reset_index()  # keep symbol as a first-class column
    cols = ["symbol", "mom_12_2", "value_proxy", "low_vol", "trend", "volume_mom",
            "composite", "rank"]
    return fdf[cols].round(4)


def trade_narratives(trades: list, df: pd.DataFrame, signal: pd.Series,
                     params: dict) -> list:
    """Dynamic, per-trade reasoning: why this trade was entered, what the
    market context was, why it was exited, and what it contributed."""
    if not trades:
        return []
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    rets = close.pct_change()
    vol20 = rets.rolling(20).std() * math.sqrt(BARS_PER_YEAR)
    notes = []
    for t in trades:
        try:
            entry_ts = pd.Timestamp(t["entry_date"])
            loc = df.index.searchsorted(entry_ts, side="left")
            loc = min(max(loc, 0), len(df) - 1)
            px = float(close.iloc[loc])
            prev_px = float(close.iloc[max(loc - 5, 0)])
            drift_5d = (px / prev_px - 1.0) * 100.0 if prev_px > 0 else 0.0
            v = float(vol20.iloc[loc]) if loc < len(vol20) and not pd.isna(vol20.iloc[loc]) else None
            vol_txt = f"{v * 100:.0f}% annualized" if v else "n/a"
            pnl = t.get("exit_pnl_pct")
            pnl_txt = f"{pnl:+.2f}%" if pnl is not None else "open"
            reason = t.get("exit_reason") or "EXIT_RULE"
            reason_txt = {
                "STOP_LOSS": "hit the stop loss — the adverse move exceeded the risk budget",
                "TAKE_PROFIT": "reached the take-profit target",
                "TRAILING_STOP": "trailing stop triggered as the trend gave back gains",
                "MAX_HOLD": "hit the maximum holding period",
                "SIGNAL_FLAT": "the signal flipped flat, so the position was closed at the close",
                "END_OF_DATA": "still open at the end of the backtest window",
            }.get(reason, "the exit rule fired")
            entry_px = t["entry_price"]
            context = (f"5-day drift before entry {drift_5d:+.1f}%"
                       + (f", {vol_txt} vol" if v else ""))
            notes.append(
                f"{t['entry_date']} {t.get('direction', 'LONG')} @ {entry_px:.2f}: entered "
                f"({context}); {reason_txt} on {t.get('exit_date', '?')} "
                f"after {t.get('hold_bars', 0)} bar(s) for {pnl_txt}."
            )
        except Exception:  # never let a narrative crash the build
            continue
    return notes


_IMPACT_REASONS = {
    "STOP_LOSS": "risk control", "TAKE_PROFIT": "profit taking",
    "TRAILING_STOP": "trend protection", "MAX_HOLD": "time stop",
    "SIGNAL_FLAT": "signal reversal", "END_OF_DATA": "end of window",
}


def compute_metrics(returns: pd.Series, equity: pd.Series,
                    trades: Optional[list] = None, bars_per_year: int = BARS_PER_YEAR,
                    capital: float = 100_000.0) -> dict:
    """Standard risk/return metrics from a daily returns series."""
    returns = returns.dropna()
    n = len(returns)
    if n < 2:
        return {"total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "max_drawdown": 0.0, "calmar": 0.0, "volatility": 0.0, "win_rate": 0.0,
                "profit_factor": 0.0, "trades": 0, "exposure": 0.0, "best_trade": 0.0,
                "worst_trade": 0.0, "avg_hold_bars": 0.0, "years": 0.0}
    total = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0
    years = n / bars_per_year
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else 0.0
    vol = float(returns.std() * math.sqrt(bars_per_year)) if returns.std() > 0 else 0.0
    sharpe = float(returns.mean() / returns.std() * math.sqrt(bars_per_year)) if returns.std() > 0 else 0.0
    downside = returns[returns < 0]
    sortino = float(returns.mean() / downside.std() * math.sqrt(bars_per_year)) if len(downside) > 1 and downside.std() > 0 else 0.0
    peak = equity.cummax()
    dd = equity / peak - 1
    max_dd = float(dd.min()) if len(dd) else 0.0
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0
    trades = trades or []
    closed = [t for t in trades if t.get("exit_pnl_pct") is not None]
    wins = [t for t in closed if t["exit_pnl_pct"] > 0]
    losses = [t for t in closed if t["exit_pnl_pct"] < 0]
    win_rate = len(wins) / len(closed) if closed else 0.0
    gross_win = sum(t["exit_pnl_pct"] for t in wins)
    gross_loss = abs(sum(t["exit_pnl_pct"] for t in losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)
    exposure = float((returns != 0).mean()) if len(returns) else 0.0
    residual_autocorr = _lag1_autocorr(returns)
    return {
        "total_return": total, "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
        "max_drawdown": max_dd, "calmar": calmar, "volatility": vol, "win_rate": win_rate,
        "profit_factor": profit_factor, "trades": len(closed), "exposure": exposure,
        "best_trade": max((t["exit_pnl_pct"] for t in closed), default=0.0),
        "worst_trade": min((t["exit_pnl_pct"] for t in closed), default=0.0),
        "avg_hold_bars": float(np.mean([t["hold_bars"] for t in closed])) if closed else 0.0,
        "years": years,
        "residual_autocorr": residual_autocorr,
    }


def _lag1_autocorr(s: pd.Series) -> float:
    """Lag-1 autocorrelation of a returns series — the Ljung-Box-style residual
    check from Bergmeir & Hyndman (2018). Large |autocorr| on strategy returns
    means the model leaves predictable structure on the table (underfits); near
    zero means the signal already captured the linear dependence."""
    s = s.dropna()
    if len(s) < 5:
        return 0.0
    x = s.to_numpy()
    mu = float(x.mean())
    num = float(np.sum((x[1:] - mu) * (x[:-1] - mu)))
    den = float(np.sum((x - mu) ** 2))
    return num / den if den != 0.0 else 0.0


# --------------------------------------------------------------------------- #
#  Indicator helpers
# --------------------------------------------------------------------------- #

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(max(int(n), 2)).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=max(int(n), 2), adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    ag = gain.ewm(alpha=1 / max(int(period), 1), adjust=False).mean()
    al = loss.ewm(alpha=1 / max(int(period), 1), adjust=False).mean()
    rs = ag / al.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / max(int(n), 1), adjust=False).mean()


def _rolling_z(close: pd.Series, n: int) -> pd.Series:
    mu = close.rolling(max(int(n), 2)).mean()
    sd = close.rolling(max(int(n), 2)).std()
    return (close - mu) / sd.replace(0.0, np.nan)


def _simplex_project(w: np.ndarray) -> np.ndarray:
    """Project weights onto the probability simplex (iterative clip+renormalise)."""
    w = np.asarray(w, dtype=float)
    for _ in range(20):
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if s <= 0:
            w = np.full_like(w, 1.0 / len(w))
            break
        w = w / s
        if (w >= 0).all():
            break
    return w


# --------------------------------------------------------------------------- #
#  Archetype registry
# --------------------------------------------------------------------------- #

# Every archetype: name, family, inspiration/provenance, param space, and a
# signal generator `sig(df, params) -> pd.Series in [-1, 1]` (warmup -> 0).
# Params: {name, type: float|int|bool|categorical, min, max, default, choices}

ARCHETYPES: dict = {}


def _register(fn):
    a = fn()
    ARCHETYPES[a["name"]] = a
    return a


@_register
def _trend_ma():
    def sig(df, p):
        close = df["Close"]
        fast = _sma(close, p["fast"])
        slow = _sma(close, p["slow"])
        norm = (fast / slow - 1.0) * 100.0
        out = np.tanh(norm / max(p["hysteresis_pct"], 1e-6)).rename("sig")
        return out.fillna(0.0).clip(-1.0, 1.0)
    return {
        "name": "trend_ma", "family": "trend",
        "label": "Moving-Average Crossover (Trend Following)",
        "provenance": ("Classic trend-following (dual moving-average crossover with "
                       "hysteresis dead-band). Standard in the QuantConnect strategy "
                       "library; momentum family of the Glucksman paper (Follow-the-Winner)."),
        "params": [
            {"name": "fast", "type": "int", "min": 2, "max": 60, "default": 10},
            {"name": "slow", "type": "int", "min": 20, "max": 250, "default": 50},
            {"name": "hysteresis_pct", "type": "float", "min": 0.05, "max": 2.0, "default": 0.3},
        ],
        "sig": sig, "ops": False,
    }


@_register
def _breakout():
    def sig(df, p):
        close = df["Close"]
        entry_n, exit_n = int(p["entry_n"]), int(p["exit_n"])
        hi = df["High"].rolling(entry_n).max().shift(1)
        lo = df["Low"].rolling(exit_n).min().shift(1)
        state = 0
        out = np.zeros(len(df))
        for i in range(len(df)):
            c = close.iloc[i]
            if state == 0:
                if not pd.isna(hi.iloc[i]) and c > hi.iloc[i]:
                    state = 1
                elif not pd.isna(lo.iloc[i]) and c < lo.iloc[i]:
                    state = -1
            elif state == 1:
                if not pd.isna(lo.iloc[i]) and c < lo.iloc[i]:
                    state = -1
            elif state == -1:
                if not pd.isna(hi.iloc[i]) and c > hi.iloc[i]:
                    state = 1
            out[i] = state
        return pd.Series(out, index=df.index, name="sig")
    return {
        "name": "breakout", "family": "trend",
        "label": "Donchian / Turtle Breakout",
        "provenance": ("Donchian-channel breakout (Turtle trading system): enter "
                       "on an N-bar high breakout, exit on an M-bar low break. "
                       "Published on the QuantConnect forum and in the QC strategy library."),
        "params": [
            {"name": "entry_n", "type": "int", "min": 10, "max": 120, "default": 55},
            {"name": "exit_n", "type": "int", "min": 5, "max": 60, "default": 20},
        ],
        "sig": sig, "ops": False,
    }


@_register
def _dual_momentum():
    def sig(df, p):
        close = df["Close"]
        rel = close / close.shift(int(p["rel_bars"])) - 1.0
        abs_gate = close / close.shift(int(p["abs_bars"])) - 1.0
        out = np.tanh(rel * 100.0 / max(p["mom_scale"], 1e-6))
        out = out.where(abs_gate > 0, 0.0)
        return out.fillna(0.0).clip(-1.0, 1.0).rename("sig")
    return {
        "name": "dual_momentum", "family": "momentum",
        "label": "Dual Momentum (Absolute + Relative)",
        "provenance": ("Dual momentum (Antonacci): relative momentum drives sizing "
                       "but an absolute-momentum gate over the longer lookback keeps "
                       "you out of downtrends. Popularized on the QuantConnect forum "
                       "(Dual Momentum Sector Rotation) and in the QC strategy library."),
        "params": [
            {"name": "rel_bars", "type": "int", "min": 10, "max": 63, "default": 21},
            {"name": "abs_bars", "type": "int", "min": 126, "max": 504, "default": 252},
            {"name": "mom_scale", "type": "float", "min": 2.0, "max": 10.0, "default": 5.0},
        ],
        "sig": sig, "ops": False,
    }


@_register
def _rsi_meanrev():
    def sig(df, p):
        close = df["Close"]
        r = _rsi(close, int(p["period"]))
        vol = close.pct_change().rolling(20).std() * math.sqrt(BARS_PER_YEAR)
        vol_cap = vol.rolling(250).quantile(p["vol_cap_pct"] / 100.0)
        out = pd.Series(0.0, index=df.index, name="sig")
        long_cond = (r < p["oversold"]) & (vol <= vol_cap)
        short_cond = (r > p["overbought"]) & (vol <= vol_cap)
        if p.get("require_turn", False):
            long_cond &= close > close.shift(1)
            short_cond &= close < close.shift(1)
        out[long_cond] = 1.0
        out[short_cond] = -1.0
        return out
    return {
        "name": "rsi_meanrev", "family": "meanrev",
        "label": "RSI Mean Reversion + Low-Vol Filter",
        "provenance": ("RSI(2)-style mean reversion (Connors) with a volatility "
                       "cap, as championed in QuantConnect community algorithms "
                       "(e.g. 'The Alpha Formula': buy low RSI, low volatility) and "
                       "the Follow-the-Loser family of the Glucksman paper."),
        "params": [
            {"name": "period", "type": "int", "min": 2, "max": 14, "default": 2},
            {"name": "oversold", "type": "float", "min": 5.0, "max": 35.0, "default": 10.0},
            {"name": "overbought", "type": "float", "min": 65.0, "max": 95.0, "default": 90.0},
            {"name": "vol_cap_pct", "type": "float", "min": 50.0, "max": 99.0, "default": 80.0},
            {"name": "require_turn", "type": "bool", "default": True},
        ],
        "sig": sig, "ops": False,
    }


@_register
def _bollinger_meanrev():
    def sig(df, p):
        close = df["Close"]
        mid = _sma(close, int(p["period"]))
        sd = close.rolling(int(p["period"])).std()
        upper, lower = mid + p["n_std"] * sd, mid - p["n_std"] * sd
        state = 0
        out = np.zeros(len(df))
        for i in range(len(df)):
            c = close.iloc[i]
            if state == 0:
                if c < lower.iloc[i]:
                    state = 1
                elif c > upper.iloc[i]:
                    state = -1
            elif state == 1 and p.get("exit_at_mid", False):
                if c >= mid.iloc[i]:
                    state = 0
            elif state == -1 and p.get("exit_at_mid", False):
                if c <= mid.iloc[i]:
                    state = 0
            out[i] = state
        return pd.Series(out, index=df.index, name="sig")
    return {
        "name": "bollinger_meanrev", "family": "meanrev",
        "label": "Bollinger-Band Mean Reversion",
        "provenance": ("Mean reversion to Bollinger bands: fade band-touch extremes, "
                       "exit at the moving midpoint. A staple of the QuantConnect "
                       "strategy library and the Follow-the-Loser family."),
        "params": [
            {"name": "period", "type": "int", "min": 10, "max": 60, "default": 20},
            {"name": "n_std", "type": "float", "min": 1.5, "max": 3.0, "default": 2.0},
            {"name": "exit_at_mid", "type": "bool", "default": True},
        ],
        "sig": sig, "ops": False,
    }


@_register
def _vol_target():
    def sig(df, p):
        close = df["Close"]
        rets = close.pct_change()
        rv = rets.rolling(int(p["lookback"])).std() * math.sqrt(BARS_PER_YEAR)
        scale = (p["target_vol"] / 100.0) / rv.replace(0.0, np.nan)
        scale = scale.clip(upper=p["cap"])
        if p["direction"] == "long_only":
            out = scale.fillna(0.0).clip(lower=0.0, upper=1.0)
        else:
            trend = np.sign(close - _sma(close, int(p["lookback"])))
            out = (scale * trend).fillna(0.0).clip(-1.0, 1.0)
        return out.rename("sig")
    return {
        "name": "vol_target", "family": "volatility",
        "label": "Volatility Targeting / Risk Scaling",
        "provenance": ("Volatility targeting: scale exposure so realized vol "
                       "converges toward a target — the core risk engine of "
                       "risk-parity and hedge-fund vol-control mandates (long-only "
                       "variant) or trend-gated long/short variant."),
        "params": [
            {"name": "target_vol", "type": "float", "min": 5.0, "max": 30.0, "default": 15.0},
            {"name": "lookback", "type": "int", "min": 10, "max": 120, "default": 30},
            {"name": "cap", "type": "float", "min": 1.0, "max": 3.0, "default": 1.5},
            {"name": "direction", "type": "categorical", "choices": ["long_only", "long_short"], "default": "long_only"},
        ],
        "sig": sig, "ops": False,
    }


@_register
def _market_making():
    def sig(df, p):
        close = df["Close"]
        fair = _ema(close, int(p["fair_period"]))
        dev = (close / fair - 1.0) * 100.0
        spread = p["half_spread_pct"]
        # Below fair value -> we captured the bid; fade the deviation. This is the
        # inventory-skew intuition (quote below/above fair pulls inventory back to
        # target) approximated on daily bars.
        out = -np.tanh(dev / max(spread, 1e-6))
        out = out.clip(-p["max_inv"], p["max_inv"])
        return out.fillna(0.0).rename("sig")
    return {
        "name": "market_making", "family": "liquidity",
        "label": "Inventory-Skewed Market Making (approximation)",
        "provenance": ("Market making: quote around fair value and skew toward the "
                       "inventory target — the Optiver / Jane Street playbook. Daily "
                       "OHLCV only approximates spread capture; production use needs "
                       "L1/L2 order-book data. Sizing is capped by max_inv to model "
                       "inventory limits."),
        "params": [
            {"name": "fair_period", "type": "int", "min": 10, "max": 100, "default": 30},
            {"name": "half_spread_pct", "type": "float", "min": 0.05, "max": 1.0, "default": 0.25},
            {"name": "max_inv", "type": "float", "min": 0.1, "max": 0.9, "default": 0.5},
        ],
        "sig": sig, "ops": False,
    }


@_register
def _statarb_z():
    def sig(df, p):
        close = df["Close"]
        z = _rolling_z(close, int(p["period"]))
        state = 0
        out = np.zeros(len(df))
        for i in range(len(df)):
            zi = z.iloc[i]
            if pd.isna(zi):
                out[i] = 0.0
                continue
            if state == 0:
                if zi > p["entry_z"]:
                    state = -1
                elif zi < -p["entry_z"]:
                    state = 1
            elif state == 1:
                if zi >= -p["exit_z"]:
                    state = 0
            elif state == -1:
                if zi <= p["exit_z"]:
                    state = 0
            out[i] = state
        return pd.Series(out, index=df.index, name="sig")
    return {
        "name": "statarb_z", "family": "statarb",
        "label": "Z-Score Statistical Arbitrage (single-instrument)",
        "provenance": ("Statistical arbitrage: fade deviations beyond an entry z-score, "
                       "close when the z-score mean-reverts. The cross-sectional stat-arb "
                       "playbook of firms like Citadel, approximated on a single series "
                       "via rolling z-score of price around its own mean."),
        "params": [
            {"name": "period", "type": "int", "min": 10, "max": 120, "default": 30},
            {"name": "entry_z", "type": "float", "min": 1.0, "max": 3.0, "default": 2.0},
            {"name": "exit_z", "type": "float", "min": 0.1, "max": 1.0, "default": 0.5},
        ],
        "sig": sig, "ops": False,
    }


@_register
def _gap_fade():
    def sig(df, p):
        close, opn = df["Close"], df["Open"]
        gap = (opn / close.shift(1) - 1.0) * 100.0
        thresh = p["gap_thresh_pct"]
        hold = int(p["hold_bars"])
        out = np.zeros(len(df))
        active = 0
        sign = 0
        for i in range(len(df)):
            if active > 0:
                active -= 1
                if active == 0:
                    sign = 0
            g = gap.iloc[i]
            if not pd.isna(g) and abs(g) > thresh and sign == 0:
                sign = -1 if g > 0 else 1  # fade the gap
                active = hold
            out[i] = sign
        return pd.Series(out, index=df.index, name="sig")
    return {
        "name": "gap_fade", "family": "meanrev",
        "label": "Overnight Gap Fade",
        "provenance": ("Fade large overnight gaps (buy big gaps down, sell big gaps "
                       "up) for a fixed holding window — the classic gap-mean-reversion "
                       "event strategy studied in retail and institutional event desks."),
        "params": [
            {"name": "gap_thresh_pct", "type": "float", "min": 0.3, "max": 3.0, "default": 1.0},
            {"name": "hold_bars", "type": "int", "min": 1, "max": 10, "default": 3},
        ],
        "sig": sig, "ops": False,
    }


# --------------------------------------------------------------------------- #
#  Research-grounded families (peer-reviewed / NBER-adjacent sources)
# --------------------------------------------------------------------------- #

def _resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a daily OHLCV frame to weekly (W-FRI) bars."""
    out = df.resample("W-FRI").agg({"Open": "first", "High": "max",
                                     "Low": "min", "Close": "last", "Volume": "sum"})
    return out.dropna()


@_register
def _stoch_williams():
    """Paik, Choi & Vaquero (JRFM 2024) — low-frequency market-timing with a
    Stochastic oscillator, Williams %R, and a volume-surge position scaler.

    Weekly bars; buy when the stochastic (%K) is oversold (< 30) AND Williams
    %R is deeply oversold (< -75); sell when %K is overbought (> 80) AND %R is
    overbought (> -20). When a buy fires, a volume surge (weekly volume >= 20%
    above the 52-week mean, or more than 1 standard deviation above it) scales
    the position to 2x; meeting BOTH conditions scales to 3x. A weekly loss-cut
    (~10% below entry) flattens the position. The paper reports ~90% hit rate
    with <1% max drawdown on SPY 2010-2023 (1.5 trades/yr)."""
    def sig(df, p):
        w = _resample_weekly(df)
        if len(w) < 30:
            return pd.Series(0.0, index=df.index, name="sig")
        hi, lo, cl, vol = w["High"], w["Low"], w["Close"], w["Volume"]
        k_per = max(int(p["k_period"]), 2)
        rng_hi = hi.rolling(k_per).max()
        rng_lo = lo.rolling(k_per).min()
        denom = (rng_hi - rng_lo).replace(0.0, np.nan)
        stoch_k = ((cl - rng_lo) / denom * 100.0).fillna(50.0)
        stoch_d = stoch_k.rolling(max(int(p["d_period"]), 2)).mean()
        r_per = max(int(p["r_period"]), 2)
        rng_hi2 = hi.rolling(r_per).max()
        rng_lo2 = lo.rolling(r_per).min()
        denom2 = (rng_hi2 - rng_lo2).replace(0.0, np.nan)
        williams = ((cl - rng_hi2) / denom2 * 100.0).fillna(-50.0)
        vol_avg = vol.rolling(max(int(p["vol_period"]), 5)).mean()
        vol_std = vol.rolling(max(int(p["vol_period"]), 5)).std()
        vol_pct = p["vol_surge_pct"]
        buy_k, buy_r = p["buy_k"], p["buy_r"]
        sell_k, sell_r = p["sell_k"], p["sell_r"]
        loss_cut = p["loss_cut_pct"]
        out = np.zeros(len(w))
        state = 0
        entry_px = 0.0
        for i in range(len(w)):
            if state != 0 and entry_px > 0 and cl.iloc[i] / entry_px - 1.0 < -loss_cut / 100.0:
                state = 0  # weekly loss-cut flattens
            if state == 0:
                if stoch_d.iloc[i] < buy_k and williams.iloc[i] < buy_r:
                    vol_i = vol.iloc[i]
                    surge20 = not pd.isna(vol_avg.iloc[i]) and vol_i >= vol_avg.iloc[i] * (1 + vol_pct / 100.0)
                    surge1s = not pd.isna(vol_std.iloc[i]) and vol_std.iloc[i] > 0 and vol_i >= vol_avg.iloc[i] + vol_std.iloc[i]
                    if surge20 and surge1s:
                        state = 3.0
                    elif surge20 or surge1s:
                        state = 2.0
                    else:
                        state = 1.0
                    entry_px = cl.iloc[i]
                else:
                    state = 0.0
            elif stoch_d.iloc[i] > sell_k and williams.iloc[i] > sell_r:
                state = 0.0
                entry_px = 0.0
            out[i] = state
        # forward-fill the weekly target onto daily bars (hold until the weekly
        # signal changes; keeps the low-frequency character of the paper)
        sig_w = pd.Series(out, index=w.index, name="sig")
        sig_d = sig_w.reindex(df.index, method="ffill").fillna(0.0)
        return sig_d.clip(lower=0.0)  # long-only by construction
    return {
        "name": "stoch_williams", "family": "meanrev",
        "label": "Stochastic + Williams %R + Volume Surge (low-freq timing)",
        "provenance": ("Paik, Choi & Vaquero, 'Algorithm-Based Low-Frequency Trading "
                       "Using a Stochastic Oscillator, Williams%R, and Trading Volume "
                       "for the S&P 500', JRFM 17:501 (2024). Weekly Stochastic %K/%D + "
                       "Williams %R overbought/oversold timing with a 52-week volume-"
                       "surge position scaler (2x/3x) and a weekly loss-cut; reported "
                       "~90% hit rate, <1% max drawdown, 1.5 trades/yr on SPY."),
        "params": [
            {"name": "k_period", "type": "int", "min": 5, "max": 26, "default": 10},
            {"name": "d_period", "type": "int", "min": 2, "max": 13, "default": 6},
            {"name": "r_period", "type": "int", "min": 5, "max": 26, "default": 10},
            {"name": "vol_period", "type": "int", "min": 26, "max": 78, "default": 52},
            {"name": "vol_surge_pct", "type": "float", "min": 5.0, "max": 60.0, "default": 20.0},
            {"name": "buy_k", "type": "float", "min": 10.0, "max": 45.0, "default": 30.0},
            {"name": "buy_r", "type": "float", "min": -95.0, "max": -60.0, "default": -75.0},
            {"name": "sell_k", "type": "float", "min": 55.0, "max": 95.0, "default": 80.0},
            {"name": "sell_r", "type": "float", "min": -40.0, "max": -5.0, "default": -20.0},
            {"name": "loss_cut_pct", "type": "float", "min": 3.0, "max": 25.0, "default": 10.0},
        ],
        "sig": sig, "ops": False,
    }


@_register
def _value_momentum():
    """Asness, Moskowitz & Pedersen, 'Value and Momentum Everywhere' (J. Finance
    2013) — combine a value proxy with 12-2 month momentum, which are negatively
    correlated, so the blend smooths the equity curve.

    Value proxy (price-based, since the engine has no fundamentals): distance of
    price below its trailing 1y average, z-scored — cheap = far below the mean.
    Momentum: the standard MOM2-12 (past 12 months of returns skipping the most
    recent month) to dodge 1-month reversals. The composite z-score drives a
    tanh target; the paper finds value and momentum premia across 8 markets and
    asset classes with a strong common factor structure."""
    def sig(df, p):
        close = df["Close"]
        mom_n = max(int(p["mom_lookback"]), 42)      # ~12 months of bars
        skip = max(int(p["mom_skip"]), 5)            # skip most-recent month
        mom = (close.shift(skip) / close.shift(skip + mom_n) - 1.0)
        # value proxy: how far price sits below its trailing-1y average
        avg1y = close.rolling(max(int(p["value_lookback"]), 42)).mean()
        value = (close / avg1y - 1.0)
        z_mom = (mom - mom.rolling(max(int(p["z_span"]), 63)).mean()) / \
            mom.rolling(max(int(p["z_span"]), 63)).std().replace(0.0, np.nan)
        z_val = (value - value.rolling(max(int(p["z_span"]), 63)).mean()) / \
            value.rolling(max(int(p["z_span"]), 63)).std().replace(0.0, np.nan)
        composite = (p["w_mom"] * z_mom.fillna(0.0) - p["w_val"] * z_val.fillna(0.0))
        out = np.tanh(composite / max(p["scale"], 1e-6))
        # absolute-momentum gate (Antonacci-style): no longs in a 1y downtrend
        gate = close / close.shift(max(int(p["gate_lookback"]), 126)) - 1.0
        out = out.where(gate > p["gate_min"], 0.0)
        return out.fillna(0.0).clip(-1.0, 1.0).rename("sig")
    return {
        "name": "value_momentum", "family": "momentum",
        "label": "Value + Momentum Composite (MOM2-12)",
        "provenance": ("Asness, Moskowitz & Pedersen, 'Value and Momentum Everywhere', "
                       "Journal of Finance 68(3) (2013): value and momentum premia across "
                       "8 markets/asset classes, negatively correlated with each other, so "
                       "combining them smooths returns. Value here is a price proxy (depth "
                       "below trailing 1y average); momentum is the standard MOM2-12 with "
                       "the most recent month skipped, gated by absolute trend."),
        "params": [
            {"name": "mom_lookback", "type": "int", "min": 42, "max": 378, "default": 252},
            {"name": "mom_skip", "type": "int", "min": 5, "max": 42, "default": 21},
            {"name": "value_lookback", "type": "int", "min": 42, "max": 378, "default": 252},
            {"name": "z_span", "type": "int", "min": 21, "max": 252, "default": 126},
            {"name": "w_mom", "type": "float", "min": 0.0, "max": 2.0, "default": 1.0},
            {"name": "w_val", "type": "float", "min": 0.0, "max": 2.0, "default": 1.0},
            {"name": "scale", "type": "float", "min": 0.5, "max": 5.0, "default": 1.5},
            {"name": "gate_lookback", "type": "int", "min": 63, "max": 378, "default": 252},
            {"name": "gate_min", "type": "float", "min": -0.2, "max": 0.2, "default": 0.0},
        ],
        "sig": sig, "ops": False,
    }


@_register
def _flag_breakout():
    """Velay & Daniel (2018) — hard-coded bull/bear flag pattern recognition.

    The paper's key findings: chart patterns carry only a ~50-60% correlation
    with future trends (barely above random) and must be combined with other
    indicators; hard-coded detection with STRICT bounds keeps false positives
    near zero. This archetype implements a strict flag (flagpole -> tight
    consolidation -> continuation breakout) that only fires on a clean pattern
    plus a trend confirmation, so it complements trend/breakout families rather
    than trading the pattern alone."""
    def sig(df, p):
        close = df["Close"]
        hi, lo = df["High"], df["Low"]
        pole = max(int(p["pole_bars"]), 5)
        flag = max(int(p["flag_bars"]), 3)
        tight = p["flag_tight_pct"]
        out = np.zeros(len(df))
        # bull-flag: strong pole up, then a tight/shrinking range (the flag),
        # then a close above the flag high = continuation
        for i in range(pole + flag, len(df)):
            if out[i - 1] != 0:
                continue
            pole_hi = hi.iloc[i - flag - pole:i - flag].max()
            pole_lo = lo.iloc[i - flag - pole:i - flag].min()
            pole_move = (pole_hi / pole_lo - 1.0) * 100.0
            flag_hi = hi.iloc[i - flag:i].max()
            flag_lo = lo.iloc[i - flag:i].min()
            flag_range = (flag_hi / flag_lo - 1.0) * 100.0
            ret = close.iloc[i] / close.iloc[i - 1] - 1.0
            if pole_move >= p["pole_min_pct"] and flag_range <= tight and \
                    close.iloc[i] > flag_hi and ret > 0:
                out[i] = 1.0
            # bear-flag mirror (only if shorts allowed by direction gate)
            if pole_move <= -p["pole_min_pct"] and flag_range <= tight and \
                    close.iloc[i] < flag_lo and ret < 0:
                out[i] = -1.0
        return pd.Series(out, index=df.index, name="sig")
    return {
        "name": "flag_breakout", "family": "trend",
        "label": "Chart-Pattern Flag Breakout (strict bounds)",
        "provenance": ("Velay & Daniel, 'Stock Chart Pattern Recognition with Deep "
                       "Learning' (2018): patterns alone carry only ~50-60% predictive "
                       "correlation, so hard-coded detection with STRICT bounds keeps "
                       "false positives near zero and the pattern must be combined with "
                       "other signals. This strict flagpole->flag->continuation detector "
                       "only fires on clean patterns, complementing trend families."),
        "params": [
            {"name": "pole_bars", "type": "int", "min": 5, "max": 30, "default": 10},
            {"name": "flag_bars", "type": "int", "min": 3, "max": 15, "default": 5},
            {"name": "pole_min_pct", "type": "float", "min": 2.0, "max": 25.0, "default": 8.0},
            {"name": "flag_tight_pct", "type": "float", "min": 1.0, "max": 15.0, "default": 5.0},
        ],
        "sig": sig, "ops": False,
    }


# --------------------------------------------------------------------------- #
#  Online Portfolio Selection (Glucksman paper families)
# --------------------------------------------------------------------------- #

def _ftrl_step(b: np.ndarray, x: np.ndarray, st: dict, params: dict) -> np.ndarray:
    """Follow-the-Regularized-Leader: gradient ascent on log(b·x) - beta/2||b||^2."""
    eta, beta = params["eta"], params["beta"]
    g = x / max(float(b @ x), 1e-12) - beta * b
    return _simplex_project(b + eta * g)


def _pamr_step(b: np.ndarray, x: np.ndarray, st: dict, params: dict) -> np.ndarray:
    """Passive-Aggressive Mean Reversion (Li et al. 2012)."""
    eps, C = params["eps"], params["C"]
    mean = float(b @ x)
    if mean <= eps:
        denom = float(((x - mean) ** 2).sum()) + 1e-12
        tau = min((mean - eps) / denom, C)
        return _simplex_project(b - tau * (x - mean))
    return b


def _olmar_step(b: np.ndarray, x: np.ndarray, st: dict, params: dict) -> np.ndarray:
    """Online Moving-Average Reversion (Li & Hoi 2012): predict via window mean."""
    w, eps = params["w"], params["eps"]
    st["hist"].append(x)
    if len(st["hist"]) >= w:
        xhat = np.mean(st["hist"][-w:], axis=0)
        mean = float(b @ xhat)
        if mean <= eps:
            denom = float(((xhat - mean) ** 2).sum()) + 1e-12
            tau = (mean - eps) / denom
            return _simplex_project(b - tau * (xhat - mean))
    return b


def _cwmr_step(b: np.ndarray, x: np.ndarray, st: dict, params: dict) -> np.ndarray:
    """Confidence-Weighted Mean Reversion, diagonal-covariance variant (CWMR-2)."""
    eps, C = params["eps"], params["C"]
    sigma = st["sigma"]
    mean = float(b @ x)
    var = float(((sigma * (x - mean)) ** 2).sum()) + 1e-12
    if mean <= eps:
        lam = min(max((mean - eps) / var, 0.0), C)
        b = _simplex_project(b - lam * sigma * (x - mean))
        st["sigma"] = sigma * np.sqrt(np.clip(1.0 - 2.0 * lam * (x - mean) ** 2, 0.05, 1.0))
    return b


def _anticor_step(b: np.ndarray, x: np.ndarray, st: dict, params: dict) -> np.ndarray:
    """Anticor (Borodin et al. 2003): transfer wealth from winners to losers."""
    w, alpha, rho = params["w"], params["alpha"], params["rho"]
    st["hist"].append(np.log(np.clip(x, 1e-12, None)))
    L = len(st["hist"])
    m = len(b)
    if L >= 2 * w:
        y1 = np.array(st["hist"][L - 2 * w:L - w])
        y2 = np.array(st["hist"][L - w:])
        mu1, mu2 = y1.mean(axis=0), y2.mean(axis=0)
        var2 = y2.var(axis=0)
        with np.errstate(all="ignore"):
            c = np.corrcoef(y1.T, y2.T)[:m, m:]
        c = np.nan_to_num(c, nan=0.0)
        for i in range(m):
            for j in range(m):
                if i == j or c[i, j] <= rho:
                    continue
                if mu1[i] > mu2[i]:
                    transfer = alpha * c[i, j] * (mu1[i] - mu2[i]) / max(var2[i], 1e-12)
                    transfer = min(transfer, b[i] / max(m - 1, 1))
                    b[i] -= transfer
                    b[j] += transfer
    return _simplex_project(b)


_OPS_STEPS = {
    "ftrl": (_ftrl_step, lambda: {}),
    "pamr": (_pamr_step, lambda: {}),
    "olmar": (_olmar_step, lambda: {"hist": []}),
    "cwmr": (_cwmr_step, lambda: {"sigma": np.full(1, 0.5)}),
    "anticor": (_anticor_step, lambda: {"hist": []}),
}

OPS_ALGOS = {
    "ftrl": {"label": "FTRL (Follow-the-Regularized-Leader)", "params": [
        {"name": "eta", "type": "float", "min": 0.01, "max": 0.5, "default": 0.1},
        {"name": "beta", "type": "float", "min": 0.0, "max": 1.0, "default": 0.1},
    ]},
    "pamr": {"label": "PAMR (Passive-Aggressive Mean Reversion)", "params": [
        {"name": "eps", "type": "float", "min": 0.0005, "max": 0.02, "default": 0.005},
        {"name": "C", "type": "float", "min": 0.1, "max": 10.0, "default": 1.0},
    ]},
    "cwmr": {"label": "CWMR (Confidence-Weighted Mean Reversion)", "params": [
        {"name": "eps", "type": "float", "min": 0.0005, "max": 0.02, "default": 0.005},
        {"name": "C", "type": "float", "min": 0.1, "max": 10.0, "default": 2.0},
    ]},
    "olmar": {"label": "OLMAR (Online Moving-Average Reversion)", "params": [
        {"name": "w", "type": "int", "min": 2, "max": 30, "default": 5},
        {"name": "eps", "type": "float", "min": 0.0005, "max": 0.02, "default": 0.005},
    ]},
    "anticor": {"label": "Anticor (Anti-Correlation Transfer)", "params": [
        {"name": "w", "type": "int", "min": 2, "max": 20, "default": 3},
        {"name": "alpha", "type": "float", "min": 0.5, "max": 4.0, "default": 2.5},
        {"name": "rho", "type": "float", "min": 0.1, "max": 0.9, "default": 0.5},
    ]},
}


def run_ops_basket(prices: pd.DataFrame, algo: str, params: dict,
                   capital: float = 100_000.0) -> dict:
    """Run an online portfolio selection algorithm over a price matrix.

    Returns equity + wealth + final weights + metrics. `prices` columns are
    tickers, rows are bars. Needs >= 2 assets.
    """
    prices = prices.dropna(how="all").ffill().dropna()
    if prices.shape[1] < 2 or len(prices) < 20:
        raise ValueError("online portfolio selection needs >= 2 assets and >= 20 bars")
    rel = prices.to_numpy()
    rel = rel[1:] / rel[:-1]
    rel = np.clip(rel, 1e-8, None)
    n, m = rel.shape
    if algo not in _OPS_STEPS:
        raise ValueError(f"unknown OPS algorithm: {algo}")
    step, state0 = _OPS_STEPS[algo]
    b = np.full(m, 1.0 / m)
    st = state0()
    if algo == "cwmr":
        st["sigma"] = np.full(m, 0.5)
    wealths: list = []
    for t in range(n):
        x = rel[t]
        wealths.append(max(float(b @ x), 1e-12))  # wealth from weights set BEFORE x_t
        b = step(b, x, st, params)
    wealth = np.cumprod(wealths)
    equity = pd.Series(wealth * capital, index=prices.index[1:])
    returns = equity.pct_change().fillna(0.0)
    metrics = compute_metrics(returns, equity, bars_per_year=BARS_PER_YEAR, capital=capital)
    metrics["trades"] = m  # rebalanced weights, not discrete trades
    metrics["exposure"] = 1.0
    return {
        "equity": equity, "returns": returns, "metrics": metrics,
        "final_weights": {prices.columns[i]: round(float(b[i]), 4) for i in range(m)},
        "weights_history": None,
    }


def run_fast_universalization(prices: pd.DataFrame, base_algos: list,
                              base_params: dict, capital: float = 100_000.0) -> dict:
    """Fast Universalization (Glucksman eq. 21): reweight experts by cumulative wealth."""
    results = {}
    for algo in base_algos:
        results[algo] = run_ops_basket(prices, algo, base_params.get(algo, {}), capital)
    wealths = {a: res["equity"] for a, res in results.items()}
    idx = list(wealths.values())[0].index
    W = pd.DataFrame({a: wealths[a].reindex(idx).ffill() for a in base_algos}).fillna(1.0)
    weights = W.div(W.sum(axis=1), axis=0)
    blended = (W.diff().fillna(0.0) * weights).sum(axis=1) + 1.0
    blended.iloc[0] = 1.0
    equity = blended.cumprod() * capital
    returns = equity.pct_change().fillna(0.0)
    metrics = compute_metrics(returns, equity, bars_per_year=BARS_PER_YEAR, capital=capital)
    final_w = {a: round(float(weights[a].iloc[-1]), 4) for a in base_algos}
    return {"equity": equity, "returns": returns, "metrics": metrics,
            "final_weights": final_w, "weights_history": weights}


# --------------------------------------------------------------------------- #
#  Single-instrument backtester (OHLCV, honest fills & costs)
# --------------------------------------------------------------------------- #

def backtest_ohlcv(df: pd.DataFrame, signal: pd.Series, params: dict,
                   capital: float = 100_000.0, bars_per_year: int = BARS_PER_YEAR) -> dict:
    """Bar-by-bar backtest with cash accounting, costs, and gap-aware stops."""
    o, h, l, c = (df["Open"].to_numpy(), df["High"].to_numpy(),
                  df["Low"].to_numpy(), df["Close"].to_numpy())
    sig = signal.reindex(df.index).fillna(0.0).to_numpy()
    n = len(df)
    sizing = params.get("sizing", "vol_target")
    risk_pct = float(params.get("risk_pct", 0.15))
    max_lev = float(params.get("max_leverage", 2.0))
    atr_n = int(params.get("atr_period", 14))
    atr = _atr(df, atr_n).to_numpy()
    stop_pct = params.get("stop_loss_pct", None)
    tp_pct = params.get("take_profit_pct", None)
    trail_pct = params.get("trailing_pct", None)
    max_hold = int(params.get("max_hold_bars", 0)) or None
    commission = float(params.get("commission_bps", 3.0)) / 1e4
    slippage = float(params.get("slippage_bps", 5.0)) / 1e4
    direction = params.get("direction", "long_only")

    cash = capital
    shares = 0.0
    entry_price = 0.0
    entry_bar = 0
    peak_price = 0.0
    trades: list = []
    equity = np.zeros(n)
    open_pos = False
    cur_dir = 0
    open_trade = None

    for i in range(n):
        price = c[i]
        opn = o[i]
        # ---- exits (gap-aware: fill at open if it gaps through the stop) ----
        if open_pos:
            gapped = False
            exit_price = None
            exit_reason = None
            if stop_pct is not None:
                stop_px = entry_price * (1 - stop_pct / 100.0) if cur_dir > 0 else entry_price * (1 + stop_pct / 100.0)
                if (cur_dir > 0 and opn <= stop_px) or (cur_dir < 0 and opn >= stop_px):
                    exit_price, gapped, exit_reason = opn, True, "STOP_LOSS"
                elif (cur_dir > 0 and price <= stop_px) or (cur_dir < 0 and price >= stop_px):
                    exit_price, exit_reason = stop_px, "STOP_LOSS"
            if tp_pct is not None and exit_price is None:
                tp_px = entry_price * (1 + tp_pct / 100.0 * cur_dir)
                if (cur_dir > 0 and opn >= tp_px) or (cur_dir < 0 and opn <= tp_px):
                    exit_price, gapped, exit_reason = opn, True, "TAKE_PROFIT"
                elif (cur_dir > 0 and price >= tp_px) or (cur_dir < 0 and price <= tp_px):
                    exit_price, exit_reason = tp_px, "TAKE_PROFIT"
            if trail_pct is not None and exit_price is None:
                if cur_dir > 0:
                    peak_price = max(peak_price, price)
                    trail_stop = peak_price * (1 - trail_pct / 100.0)
                    if opn <= trail_stop:
                        exit_price, gapped, exit_reason = opn, True, "TRAILING_STOP"
                    elif price <= trail_stop:
                        exit_price, exit_reason = trail_stop, "TRAILING_STOP"
                else:
                    peak_price = min(peak_price, price) if peak_price != 0 else price
                    trail_stop = peak_price * (1 + trail_pct / 100.0)
                    if opn >= trail_stop:
                        exit_price, gapped, exit_reason = opn, True, "TRAILING_STOP"
                    elif price >= trail_stop:
                        exit_price, exit_reason = trail_stop, "TRAILING_STOP"
            if max_hold is not None and exit_price is None and (i - entry_bar) >= max_hold:
                exit_price = opn
                gapped = True
                exit_reason = "MAX_HOLD"
            if exit_price is not None:
                fill = exit_price if gapped else price
                cost = commission + slippage
                cash += shares * fill * (1 - cost)
                exit_pnl_pct = (fill / entry_price - 1.0) * cur_dir * 100.0
                open_trade["exit_pnl_pct"] = round(exit_pnl_pct, 4)
                open_trade["exit_date"] = str(df.index[i].date())
                open_trade["hold_bars"] = i - entry_bar
                open_trade["exit_reason"] = exit_reason or "EXIT_RULE"
                trades.append(open_trade)
                open_trade = None
                shares = 0.0
                open_pos = False
                cur_dir = 0

        # ---- target position ----
        target = float(sig[i])
        if direction == "long_only":
            target = max(target, 0.0)
        target = float(np.clip(target, -max_lev, max_lev))
        if target != 0 and not open_pos:
            px = opn if not pd.isna(opn) else price
            atr_i = atr[i] if not pd.isna(atr[i]) and atr[i] > 0 else px * 0.02
            atr_pct = max(atr_i / px, 1e-4)
            if sizing == "vol_target":
                # risk budget: one ATR adverse move costs `risk_pct` of capital
                notional = capital * risk_pct / atr_pct
            else:  # fixed_pct
                notional = capital * risk_pct
            notional = min(notional, capital * max_lev)
            notional = min(notional, cash / max((1 + commission + slippage), 1e-9))
            if notional > 0 and notional / max(capital, 1e-9) >= 0.002:
                buy_shares = notional / px
                cost = commission + slippage
                if target > 0:
                    cash -= buy_shares * px * (1 + cost)
                    shares = buy_shares
                else:
                    cash += buy_shares * px * (1 - cost)  # short proceeds held in cash
                    shares = -buy_shares
                open_pos = True
                cur_dir = 1.0 if target > 0 else -1.0
                entry_price = px
                entry_bar = i
                peak_price = px
                open_trade = {"entry_date": str(df.index[i].date()), "entry_price": round(px, 4),
                              "direction": "LONG" if cur_dir > 0 else "SHORT",
                              "exit_pnl_pct": None, "exit_date": None, "hold_bars": None,
                              "exit_reason": None}
        elif open_pos and abs(target) < 0.01 and not gapped:
            # flat signal while holding: close at close (no gap info needed)
            fill = price
            cost = commission + slippage
            cash += shares * fill * (1 - cost)
            open_trade["exit_pnl_pct"] = round((fill / entry_price - 1.0) * cur_dir * 100.0, 4)
            open_trade["exit_date"] = str(df.index[i].date())
            open_trade["hold_bars"] = i - entry_bar
            open_trade["exit_reason"] = "SIGNAL_FLAT"
            trades.append(open_trade)
            open_trade = None
            shares = 0.0
            open_pos = False
            cur_dir = 0

        equity[i] = cash + shares * price

    if open_pos:
        fill = c[-1]
        cash += shares * fill * (1 - commission - slippage)
        open_trade["exit_pnl_pct"] = round((fill / entry_price - 1.0) * cur_dir * 100.0, 4)
        open_trade["exit_date"] = str(df.index[-1].date())
        open_trade["hold_bars"] = n - 1 - entry_bar
        open_trade["exit_reason"] = "END_OF_DATA"
        trades.append(open_trade)

    eq = pd.Series(equity, index=df.index)
    returns = eq.pct_change().fillna(0.0)
    metrics = compute_metrics(returns, eq, trades=trades, bars_per_year=bars_per_year, capital=capital)
    return {"equity": eq, "returns": returns, "metrics": metrics, "trades": trades}


# --------------------------------------------------------------------------- #
#  Parameter search with train/test splits (conditional-return, like the repo)
# --------------------------------------------------------------------------- #

def sample_params(archetype: dict, rng: np.random.Generator,
                  locked: Optional[dict] = None) -> dict:
    out = {}
    for p in archetype["params"]:
        name = p["name"]
        if locked and name in locked:
            out[name] = locked[name]
            continue
        if p["type"] == "bool":
            out[name] = bool(rng.integers(0, 2))
        elif p["type"] == "int":
            out[name] = int(rng.integers(p["min"], p["max"] + 1))
        elif p["type"] == "categorical":
            out[name] = str(rng.choice(p["choices"]))
        else:
            out[name] = float(rng.uniform(p["min"], p["max"]))
    return out


def _validate_params(archetype: dict, params: dict, n_bars: Optional[int] = None) -> dict:
    """Enforce hard relationships (slow > fast, exit < entry, etc.) and keep
    every lookback window inside the available history."""
    p = dict(params)
    if "fast" in p and "slow" in p and p["slow"] <= p["fast"]:
        p["slow"] = p["fast"] + max(10, p["fast"])
    if "entry_n" in p and "exit_n" in p and p["exit_n"] >= p["entry_n"]:
        p["exit_n"] = max(2, p["entry_n"] - 2)
    if n_bars is not None:
        cap = max(10, n_bars // 3)
        for k, v in p.items():
            if isinstance(v, int) and k not in ("hold_bars", "max_hold_bars") and v > cap:
                p[k] = int(cap)
    return p


def walk_forward_evaluate(archetype: dict, df: pd.DataFrame, params: dict,
                          backtest_params: Optional[dict] = None,
                          folds: int = 4, min_train: float = 0.35) -> dict:
    """Multi-fold walk-forward OOS evaluation (Bergmeir & Hyndman, 'A Note on
    the Validity of Cross-Validation for Evaluating Autoregressive Time Series
    Prediction', 2018).

    Instead of ONE held-out window (a single OOS draw), the series is split into
    `folds` contiguous trailing folds of increasing size; the strategy runs on
    each fold and per-fold metrics are aggregated. The paper shows this kind of
    repeated evaluation controls overfitting far better than a single OOS split
    for autoregressive/ML strategies. Returns per-fold metrics plus aggregates
    (mean/median Sharpe, worst fold, consistency = share of folds with positive
    Sharpe)."""
    backtest_params = backtest_params or {}
    n = len(df)
    if n < 120:
        return {"folds": 0, "error": "insufficient history for walk-forward"}
    folds = max(int(folds), 2)
    start = max(int(n * min_train), 60)
    step = max((n - start) // folds, 1)
    fold_results = []
    sig_all = archetype["sig"](df, params)
    for k in range(folds):
        end = start + (k + 1) * step
        end = min(end, n)
        if end - start < 30:
            continue
        test_df = df.iloc[start:end]
        sig_test = sig_all.iloc[start:end]
        res = backtest_ohlcv(test_df, sig_test, backtest_params)
        fold_results.append({"fold": k + 1, "metrics": res["metrics"],
                             "equity": res["equity"], "trades": res["trades"]})
    if not fold_results:
        return {"folds": 0, "error": "no usable folds"}
    sharpes = [f["metrics"]["sharpe"] for f in fold_results]
    returns = [f["metrics"]["total_return"] for f in fold_results]
    dds = [f["metrics"]["max_drawdown"] for f in fold_results]
    autocorrs = [f["metrics"]["residual_autocorr"] for f in fold_results]
    return {
        "folds": len(fold_results),
        "per_fold": [{"fold": f["fold"], "sharpe": f["metrics"]["sharpe"],
                       "total_return": f["metrics"]["total_return"],
                       "max_drawdown": f["metrics"]["max_drawdown"],
                       "trades": f["metrics"]["trades"],
                       "residual_autocorr": f["metrics"]["residual_autocorr"]}
                      for f in fold_results],
        "mean_sharpe": float(np.mean(sharpes)),
        "median_sharpe": float(np.median(sharpes)),
        "worst_fold_sharpe": float(min(sharpes)),
        "mean_return": float(np.mean(returns)),
        "worst_drawdown": float(min(dds)),
        "consistency": float(np.mean([s > 0 for s in sharpes])),
        "mean_autocorr": float(np.mean(autocorrs)),
        "equities": [f["equity"] for f in fold_results],
        "trades": [t for f in fold_results for t in f["trades"]],
    }


def search_best(archetype: dict, df: pd.DataFrame, n_trials: int = 30,
                rng: Optional[np.random.Generator] = None, seed: int = 42,
                backtest_params: Optional[dict] = None, locked: Optional[dict] = None,
                split: float = 0.75, objective: str = "sharpe",
                wf_folds: int = 0) -> dict:
    """Random-search the archetype's parameter space on a train window, then
    report honest out-of-sample metrics on the held-out test window. When
    `wf_folds > 0`, ALSO run a walk-forward multi-fold OOS evaluation
    (Bergmeir & Hyndman 2018) of the chosen params and report it as
    `walk_forward` (mean/median/worst Sharpe across folds + consistency)."""
    rng = rng or np.random.default_rng(seed)
    backtest_params = backtest_params or {}
    n = len(df)
    cut = max(int(n * split), 60)
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]
    if len(train_df) < 60 or len(test_df) < 30:
        return {"error": "insufficient history for train/test split"}

    candidates = []
    for _ in range(max(n_trials, 5)):
        params = _validate_params(archetype, sample_params(archetype, rng, locked),
                                  n_bars=len(train_df))
        # compute the signal once over the FULL series: rolling indicators only
        # use past data, so the test slice sees exactly what live trading would
        sig_all = archetype["sig"](df, params)
        sig_train, sig_test = sig_all.iloc[:cut], sig_all.iloc[cut:]
        res = backtest_ohlcv(train_df, sig_train, backtest_params)
        m = res["metrics"]
        if m["trades"] < 1:  # at least one trade; trend systems can be 1-2 trades/window
            continue
        score = m.get(objective, m["sharpe"]) if objective in m else m["sharpe"]
        if objective == "sharpe" and math.isnan(score):
            score = -9.0
        candidates.append((score, params, res, sig_test))
    candidates.sort(key=lambda t: -t[0])
    if not candidates:
        return {"error": "no viable parameter sets found on training window"}

    # robust pick: from the top-5, prefer the one whose OOS drawdown is acceptable
    top = candidates[: min(5, len(candidates))]
    chosen = None
    for score, params, train_res, sig_test in top:
        test_res = backtest_ohlcv(test_df, sig_test, backtest_params)
        oos_m = test_res["metrics"]
        dd_cap = float(backtest_params.get("max_dd_cap", -0.60))
        if oos_m["trades"] >= 1 and oos_m["max_drawdown"] >= dd_cap:
            chosen = (params, train_res, test_res, oos_m)
            break
    if chosen is None:
        score, params, train_res, sig_test = candidates[0]
        test_res = backtest_ohlcv(test_df, sig_test, backtest_params)
        chosen = (params, train_res, test_res, test_res["metrics"])

    params, train_res, test_res, oos_m = chosen
    out = {
        "archetype": archetype["name"], "params": params,
        "train_metrics": train_res["metrics"], "test_metrics": oos_m,
        "train_equity": train_res["equity"], "test_equity": test_res["equity"],
        "train_trades": train_res["trades"], "test_trades": test_res["trades"],
    }
    if wf_folds and int(wf_folds) > 0:
        wf = walk_forward_evaluate(archetype, df, params, backtest_params,
                                   folds=int(wf_folds))
        out["walk_forward"] = wf if wf.get("folds", 0) > 0 else None
    return out


# --------------------------------------------------------------------------- #
#  Ensembles (single-instrument)
# --------------------------------------------------------------------------- #

def build_ensemble(members: list, df: pd.DataFrame, method: str = "fast_universalization",
                   capital: float = 100_000.0) -> dict:
    """Blend several strategies' daily returns into one ensemble.

    * equal_weight       : 1/K each (static)
    * sharpe_weight      : weight by train Sharpe (static, clipped)
    * fast_universalization: wealth-proportional reweighting (Glucksman eq. 21)
    * dynamic            : wealth-adaptive mixing tilted by each member's measured
      quality (Sharpe, sortino, drawdown, win rate, trade robustness) and
      penalized for redundancy (pairwise correlation) — maximizes strengths,
      down-weights weaknesses.
    """
    if not members:
        raise ValueError("ensemble needs at least one member")
    equities = {m.name: m.equity.reindex(df.index).ffill() for m in members}
    idx = df.index
    E = pd.DataFrame(equities)
    rets = E.pct_change().fillna(0.0)
    K = len(members)
    if method == "equal_weight":
        w = pd.Series({m.name: 1.0 / K for m in members})
        combined = (rets * w).sum(axis=1)
        final_w = w.to_dict()
    elif method == "sharpe_weight":
        sh = {m.name: max(m.metrics.get("sharpe", 0.0), 0.0) for m in members}
        tot = sum(sh.values())
        w = pd.Series({k: (v / tot if tot > 0 else 1.0 / K) for k, v in sh.items()})
        combined = (rets * w).sum(axis=1)
        final_w = w.round(4).to_dict()
    elif method == "dynamic":
        # ---- quality score per member (strengths) ----
        q = {}
        for m in members:
            mm = m.metrics
            sh = float(np.clip(mm.get("sharpe", 0.0), -1.0, 3.0))
            so = float(np.clip(mm.get("sortino", 0.0), -1.0, 3.0))
            dd = float(mm.get("max_drawdown", 0.0))  # negative
            dd_ok = float(np.clip(1.0 + dd, 0.1, 1.5))
            wr = float(np.clip(mm.get("win_rate", 0.0), 0.0, 1.0))
            rob = float(min(mm.get("trades", 0), 30) / 30.0)
            q[m.name] = ((sh + so) / 2.0) * dd_ok * (0.6 + 0.4 * wr) * (0.5 + 0.5 * rob)
        # ---- redundancy penalty (correlation with the other members) ----
        corr = rets.corr()
        div = {}
        for name in rets.columns:
            others = [c for c in rets.columns if c != name]
            if others:
                vals = [abs(corr.loc[name, o]) for o in others
                        if not pd.isna(corr.loc[name, o])]
                avg = float(np.mean(vals)) if vals else 0.5
            else:
                avg = 0.5
            div[name] = 1.0 / (1.0 + avg)
        wq = pd.Series({n: max(q[n], 0.0) * div[n] for n in q})
        if wq.sum() <= 0:
            wq = pd.Series({n: 1.0 / K for n in rets.columns})
        else:
            wq = wq / wq.sum()
        # ---- per-bar dynamic weights: wealth-adaptive, tilted by quality ----
        W = E.div(E.sum(axis=1), axis=0).fillna(1.0 / K)
        W = W.mul(wq, axis=1)
        W = W.div(W.sum(axis=1), axis=0).fillna(1.0 / K)
        combined = (rets * W.shift(1).fillna(wq)).sum(axis=1)
        final_w = {k: round(float(W[k].iloc[-1]), 4) for k in W.columns}
        rationale = _ensemble_rationale(members, q, div, wq, final_w)
    else:  # fast_universalization
        W = E.div(E.sum(axis=1), axis=0).fillna(1.0 / K)
        combined = (rets * W.shift(1).fillna(1.0 / K)).sum(axis=1)
        final_w = {k: round(float(W[k].iloc[-1]), 4) for k in W.columns}
        rationale = []
    equity = (1.0 + combined).cumprod() * capital
    returns = equity.pct_change().fillna(0.0)
    metrics = compute_metrics(returns, equity, bars_per_year=BARS_PER_YEAR, capital=capital)
    return {"equity": equity, "returns": returns, "metrics": metrics,
            "final_weights": final_w, "method": method, "members": [m.name for m in members],
            "rationale": rationale}


def _ensemble_rationale(members: list, quality: dict, diversity: dict,
                        quality_weights: pd.Series, final_weights: dict) -> list:
    """Per-member explanation of the dynamic ensemble weights: what strength the
    weight rewards, what weakness it down-weights, and the redundancy note."""
    notes = []
    for m in members:
        mm = m.metrics
        strength = "Sharpe", mm.get("sharpe", 0.0)
        if mm.get("sortino", 0.0) > mm.get("sharpe", 0.0):
            strength = "Sortino", mm.get("sortino", 0.0)
        weakness = "max drawdown", mm.get("max_drawdown", 0.0)
        if mm.get("trades", 0) < 5:
            weakness = "very few trades (thin evidence)", mm.get("trades", 0)
        elif mm.get("win_rate", 0.0) < 0.35:
            weakness = "low win rate", mm.get("win_rate", 0.0)
        qw = float(quality_weights.get(m.name, 0.0))
        fw = float(final_weights.get(m.name, 0.0))
        notes.append(
            f"• {m.name}: strength = {strength[0].lower()} {strength[1]:.2f}; "
            f"weakness = {weakness[0]} {weakness[1]:.2f}; "
            f"redundancy = {1 - diversity[m.name]:.2f} avg correlation with the other "
            f"members. Quality tilt {qw * 100:.0f}% -> final blend weight {fw * 100:.0f}%."
        )
    return notes


# --------------------------------------------------------------------------- #
#  Natural-language request parsing (auto mode)
# --------------------------------------------------------------------------- #

_KEYWORD_FAMILIES = [
    (["mean reversion", "mean-reversion", "mean reverting", "revert", "reversion",
      "fade", "buy the dip", "buy dip", "oversold", "bollinger", "rsi",
      "stat arb", "statarb", "pairs", "z-score", "zscore", "gap",
      "stochastic", "williams", "williams %r", "oversold oscillator", "volume surge"],
     ["stoch_williams", "rsi_meanrev", "bollinger_meanrev", "statarb_z", "gap_fade"]),
    (["trend", "momentum", "moving average", "crossover", "breakout", "turtle",
      "donchian", "follow the winner", "follow-the-winner", "follow the trend",
      "chart pattern", "flag", "flagpole", "value and momentum", "value momentum",
      "mom everywhere", "asness"],
     ["value_momentum", "flag_breakout", "trend_ma", "breakout", "dual_momentum"]),
    (["market mak", "liquidity", "spread", "optiver", "jane street", "citadel",
      "order book", "quoting"],
     ["market_making"]),
    (["volatility", "vol target", "vol targeting", "risk parity", "risk scaling",
      "hedge fund", "risk control"],
     ["vol_target"]),
    (["portfolio selection", "online portfolio", "pamr", "cwmr", "ftrl", "olmar",
      "anticor", "universal", "glucksman"],
     ["online_ops"]),
    (["ensemble", "meta", "combine", "combining", "multiple strategies", "blend",
      "all of them", "everything"],
     ["ensemble"]),
]

_DEFAULT_BY_RISK = {
    "conservative": ["rsi_meanrev", "vol_target", "bollinger_meanrev"],
    "balanced": ["trend_ma", "rsi_meanrev", "statarb_z"],
    "aggressive": ["breakout", "dual_momentum", "gap_fade"],
}


def parse_request(text: str, risk: str = "balanced") -> list:
    """Map a vague request to archetype families. Empty/unknown -> risk default.

    When the request mentions SEVERAL strategy ideas (e.g. "value and momentum
    plus stochastic oversold buys"), families are interleaved across the matching
    keyword groups (2 per group) instead of taking the first group's whole list,
    so a multi-topic request yields a genuinely diverse mix of families."""
    t = (text or "").lower()
    groups: list = []
    for keywords, families in _KEYWORD_FAMILIES:
        if any(k in t for k in keywords):
            groups.append([f for f in families if f != "ensemble"])
    if not groups:
        return list(_DEFAULT_BY_RISK.get(risk, _DEFAULT_BY_RISK["balanced"]))[:4]
    # interleave: take up to 2 families per matching group, round-robin, capped
    hits: list = []
    idx = 0
    while len(hits) < 4:
        added = 0
        for fams in groups:
            if idx < len(fams) and fams[idx] not in hits:
                hits.append(fams[idx])
                added += 1
                if len(hits) >= 4:
                    break
        idx += 1
        if added == 0:
            break
    return hits[:4]


# --------------------------------------------------------------------------- #
#  Code / spec / report generation
# --------------------------------------------------------------------------- #

def _py_lit(obj) -> str:
    """Serialize a params/backtest dict as a *valid Python literal*.

    json.dumps emits JSON tokens (null / true / false) that are not Python
    keywords, so a generated script embedding `BACKTEST = {...take_profit_pct:
    null...}` would NameError the moment it runs. Params are plain Python
    primitives (int/float/bool/str/None and nested lists/dicts), so repr()
    after normalizing numpy scalars is the safe, correct literal."""
    def _conv(o):
        if isinstance(o, dict):
            return {k: _conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_conv(v) for v in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    return repr(_conv(obj))


def generate_python(archetype_name: str, params: dict, backtest_params: dict,
                    universe: list, ops_algo: Optional[str] = None) -> str:
    """Generate a self-contained, runnable pandas strategy script."""
    sym = universe[0] if universe else "SPY"
    uni_literal = json.dumps(universe[:4])
    if archetype_name == "online_ops":
        ops = OPS_ALGOS.get(ops_algo or "pamr", OPS_ALGOS["pamr"])
        p_lit = _py_lit(params)
        return textwrap.dedent(f'''\
            """Self-contained {ops["label"]} strategy — generated by the Algorithm Builder.

            Research provenance: Glucksman Fellowship paper (NYU Stern), online
            portfolio selection family. Rebalances weights across the universe on
            every bar. Educational use only — not financial advice.
            """
            import numpy as np
            import pandas as pd
            from data_sources import get_stock

            UNIVERSE = {uni_literal}
            ALGO = {json.dumps(ops_algo or "pamr")!r}
            PARAMS = {p_lit}

            def main():
                prices = pd.DataFrame({{s: get_stock(s)["Close"] for s in UNIVERSE}})
                prices = prices.ffill().dropna()
                rel = np.clip(prices.to_numpy()[1:] / prices.to_numpy()[:-1], 1e-8, None)
                n, m = rel.shape
                w = np.full(m, 1.0 / m)
                weights = []
                for t in range(n):
                    x = rel[t]
                    mean = w @ x
                    if "eps" in PARAMS and mean <= PARAMS["eps"]:
                        denom = ((x - mean) ** 2).sum() + 1e-12
                        tau = min((mean - PARAMS["eps"]) / denom, PARAMS.get("C", 1.0))
                        w = w - tau * (x - mean)
                        w = np.clip(w, 0, None)
                        w /= max(w.sum(), 1e-12)
                    weights.append(w.copy())
                wealth = np.cumprod([w @ x for w, x in zip(weights, rel)])
                eq = pd.Series(wealth, index=prices.index[1:])
                print(eq.tail())
                return eq

            if __name__ == "__main__":
                main()
            ''')
    if archetype_name == "ensemble":
        # An ensemble card has no single signal function; rebuild each member
        # with its fitted parameters through the engine's own backtest path and
        # blend the equity curves by cumulative wealth (Fast Universalization).
        member_specs = params.get("members_detail") or [
            {"name": n, "archetype": "unknown", "params": {}} for n in params.get("members", [])
        ]
        specs_lit = _py_lit(member_specs)
        method = params.get("method", "fast_universalization")
        return textwrap.dedent(f'''\
            """Fast Universalization ensemble — generated by the Algorithm Builder.

            Rebuilds every member strategy with its fitted parameters (exactly the
            engine's backtest path) and blends their equity curves by cumulative
            wealth (Glucksman eq. 21). Educational use only — not financial advice.
            """
            import numpy as np
            import pandas as pd
            from data_sources import get_stock
            from algorithm_builder_engine import ARCHETYPES, run_ops_basket, backtest_ohlcv

            UNIVERSE = {json.dumps(universe[:4])}
            MEMBERS = {specs_lit}
            METHOD = {json.dumps(method)!r}

            def main():
                dfs = {{s: get_stock(s) for s in UNIVERSE}}
                dfs = {{s: df.iloc[:, 0].to_frame() if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1 else df for s, df in dfs.items()}}
                primary = UNIVERSE[0]
                prices = pd.DataFrame({{s: dfs[s]["Close"] for s in dfs}})
                member_results = {{}}
                for spec in MEMBERS:
                    name = spec["name"]
                    if spec.get("archetype") == "online_ops":
                        if prices.shape[1] < 2:
                            continue
                        res = run_ops_basket(prices, spec.get("ops_algo", "pamr"), spec.get("params", {{}}))
                    else:
                        arch = spec.get("archetype")
                        if arch not in ARCHETYPES:
                            continue
                        a = ARCHETYPES[arch]
                        sig = a["sig"](dfs[primary], spec.get("params", {{}}))
                        res = backtest_ohlcv(dfs[primary], sig, spec.get("backtest_params", {{}}))
                    member_results[name] = res["equity"]
                if len(member_results) < 2:
                    raise RuntimeError("need >= 2 healthy members to blend")
                idx = list(member_results.values())[0].index
                W = pd.DataFrame({{n: eq.reindex(idx).ffill() for n, eq in member_results.items()}}).fillna(1.0)
                weights = W.div(W.sum(axis=1), axis=0)
                blended = (W.diff().fillna(0.0) * weights).sum(axis=1) + 1.0
                blended.iloc[0] = 1.0
                eq = blended.cumprod() * 100_000.0
                print(eq.tail())
                return eq

            if __name__ == "__main__":
                main()
            ''')
    a = ARCHETYPES[archetype_name]
    p_lit = _py_lit(params)
    bp_lit = _py_lit(backtest_params)
    return textwrap.dedent(f'''\
        """{a["label"]} — generated by the Algorithm Builder.

        Research provenance: {a["provenance"]}
        Educational use only — not financial advice.
        """
        import numpy as np
        import pandas as pd
        from data_sources import get_stock

        SYMBOL = {json.dumps(sym)}
        PARAMS = {p_lit}
        BACKTEST = {bp_lit}

        def sma(s, n):
            return s.rolling(max(int(n), 2)).mean()

        def signal(df, p):
            """Returns target exposure in [-1, 1] per bar."""
            close = df["Close"]
    ''') + textwrap.indent(_signal_body(archetype_name), "    ") + textwrap.dedent(f'''\

        def main():
            df = get_stock(SYMBOL)
            sig = signal(df, PARAMS)
            eq = df["Close"].iloc[:0].copy()
            cash, shares, equity = 100_000.0, 0.0, []
            for i in range(len(df)):
                price = float(df["Close"].iloc[i])
                target = float(sig.iloc[i])
                if BACKTEST.get("direction") == "long_only":
                    target = max(target, 0.0)
                if shares == 0 and target != 0:
                    notional = min(100_000 * 0.5, cash * 0.95)
                    shares = notional / price * (1 if target > 0 else -1)
                    if target < 0:
                        cash += notional
                elif shares != 0 and abs(target) < 0.01:
                    cash += shares * price * (1 - 0.0008)
                    shares = 0.0
                equity.append(cash + shares * price)
            out = pd.Series(equity, index=df.index)
            print(out.tail())
            return out

        if __name__ == "__main__":
            main()
        ''')


def _signal_body(name: str) -> str:
    """Emit the signal function body for the generated script (mirrors engine)."""
    if name == "trend_ma":
        return textwrap.dedent('''\
                fast = sma(close, PARAMS["fast"])
                slow = sma(close, PARAMS["slow"])
                return np.tanh(((fast / slow - 1) * 100) / max(PARAMS["hysteresis_pct"], 1e-6)).fillna(0)

        ''')
    if name == "breakout":
        return textwrap.dedent('''\
                hi = df["High"].rolling(int(PARAMS["entry_n"])).max().shift(1)
                lo = df["Low"].rolling(int(PARAMS["exit_n"])).min().shift(1)
                out = pd.Series(0.0, index=df.index)
                state = 0
                for i in range(len(df)):
                    c = close.iloc[i]
                    if state == 0:
                        if c > hi.iloc[i]: state = 1
                        elif c < lo.iloc[i]: state = -1
                    elif state == 1 and c < lo.iloc[i]: state = -1
                    elif state == -1 and c > hi.iloc[i]: state = 1
                    out.iloc[i] = state
                return out

        ''')
    if name == "dual_momentum":
        return textwrap.dedent('''\
                rel = close / close.shift(int(PARAMS["rel_bars"])) - 1
                gate = close / close.shift(int(PARAMS["abs_bars"])) - 1
                return (np.tanh(rel * 100 / max(PARAMS["mom_scale"], 1e-6))).where(gate > 0, 0).fillna(0)

        ''')
    if name == "rsi_meanrev":
        return textwrap.dedent('''\
                delta = close.diff()
                gain = delta.clip(lower=0).ewm(alpha=1/int(PARAMS["period"]), adjust=False).mean()
                loss = (-delta.clip(upper=0)).ewm(alpha=1/int(PARAMS["period"]), adjust=False).mean()
                rsi = 100 - 100 / (1 + gain / loss.replace(0, float("nan")))
                vol = close.pct_change().rolling(20).std() * np.sqrt(252)
                cap = vol.rolling(250).quantile(PARAMS["vol_cap_pct"] / 100)
                out = pd.Series(0.0, index=df.index)
                out[(rsi < PARAMS["oversold"]) & (vol <= cap)] = 1.0
                out[(rsi > PARAMS["overbought"]) & (vol <= cap)] = -1.0
                return out.fillna(0)

        ''')
    if name == "bollinger_meanrev":
        return textwrap.dedent('''\
                mid = sma(close, int(PARAMS["period"]))
                sd = close.rolling(int(PARAMS["period"])).std()
                up, lo_ = mid + PARAMS["n_std"] * sd, mid - PARAMS["n_std"] * sd
                out = pd.Series(0.0, index=df.index)
                out[close < lo_] = 1.0
                out[close > up] = -1.0
                return out

        ''')
    if name == "vol_target":
        return textwrap.dedent('''\
                rv = close.pct_change().rolling(int(PARAMS["lookback"])).std() * np.sqrt(252)
                scale = (PARAMS["target_vol"] / 100) / rv.replace(0, float("nan"))
                scale = scale.clip(upper=PARAMS["cap"]).fillna(0)
                if PARAMS["direction"] == "long_only":
                    return scale.clip(lower=0, upper=1)
                trend = np.sign(close - sma(close, int(PARAMS["lookback"])))
                return (scale * trend).clip(-1, 1)

        ''')
    if name == "market_making":
        return textwrap.dedent('''\
                fair = close.ewm(span=int(PARAMS["fair_period"]), adjust=False).mean()
                dev = (close / fair - 1) * 100
                out = -np.tanh(dev / max(PARAMS["half_spread_pct"], 1e-6)).clip(-PARAMS["max_inv"], PARAMS["max_inv"])
                return out.fillna(0)

        ''')
    if name == "statarb_z":
        return textwrap.dedent('''\
                mu = close.rolling(int(PARAMS["period"])).mean()
                sd = close.rolling(int(PARAMS["period"])).std()
                z = (close - mu) / sd.replace(0, float("nan"))
                out = pd.Series(0.0, index=df.index)
                out[z > PARAMS["entry_z"]] = -1.0
                out[z < -PARAMS["entry_z"]] = 1.0
                return out

        ''')
    if name == "gap_fade":
        return textwrap.dedent('''\
                gap = (df["Open"] / close.shift(1) - 1) * 100
                out = pd.Series(0.0, index=df.index)
                out[gap.abs() > PARAMS["gap_thresh_pct"]] = -np.sign(gap[gap.abs() > PARAMS["gap_thresh_pct"]])
                return out

        ''')
    if name == "stoch_williams":
        return textwrap.dedent('''\
                w = df.resample("W-FRI").agg({"Open": "first", "High": "max",
                                                "Low": "min", "Close": "last",
                                                "Volume": "sum"}).dropna()
                hi, lo, cl, vol = w["High"], w["Low"], w["Close"], w["Volume"]
                denom = (hi.rolling(int(PARAMS["k_period"])).max() - lo.rolling(int(PARAMS["k_period"])).min()).replace(0, float("nan"))
                k = ((cl - lo.rolling(int(PARAMS["k_period"])).min()) / denom * 100).fillna(50)
                d = k.rolling(int(PARAMS["d_period"])).mean()
                denom2 = (hi.rolling(int(PARAMS["r_period"])).max() - lo.rolling(int(PARAMS["r_period"])).min()).replace(0, float("nan"))
                r = ((cl - hi.rolling(int(PARAMS["r_period"])).max()) / denom2 * 100).fillna(-50)
                va = vol.rolling(int(PARAMS["vol_period"])).mean()
                vs = vol.rolling(int(PARAMS["vol_period"])).std()
                out = pd.Series(0.0, index=w.index)
                state = 0.0
                for i in range(len(w)):
                    if state == 0 and d.iloc[i] < PARAMS["buy_k"] and r.iloc[i] < PARAMS["buy_r"]:
                        s20 = vol.iloc[i] >= va.iloc[i] * (1 + PARAMS["vol_surge_pct"] / 100)
                        s1 = vs.iloc[i] > 0 and vol.iloc[i] >= va.iloc[i] + vs.iloc[i]
                        state = 3.0 if (s20 and s1) else (2.0 if (s20 or s1) else 1.0)
                    elif state != 0 and d.iloc[i] > PARAMS["sell_k"] and r.iloc[i] > PARAMS["sell_r"]:
                        state = 0.0
                    out.iloc[i] = state
                return out.reindex(df.index, method="ffill").fillna(0).clip(lower=0)

        ''')
    if name == "value_momentum":
        return textwrap.dedent('''\
                skip = int(PARAMS["mom_skip"])
                mom = close.shift(skip) / close.shift(skip + int(PARAMS["mom_lookback"])) - 1
                val = close / close.rolling(int(PARAMS["value_lookback"])).mean() - 1
                span = int(PARAMS["z_span"])
                zm = (mom - mom.rolling(span).mean()) / mom.rolling(span).std().replace(0, float("nan"))
                zv = (val - val.rolling(span).mean()) / val.rolling(span).std().replace(0, float("nan"))
                comp = PARAMS["w_mom"] * zm.fillna(0) - PARAMS["w_val"] * zv.fillna(0)
                out = np.tanh(comp / max(PARAMS["scale"], 1e-6))
                gate = close / close.shift(int(PARAMS["gate_lookback"])) - 1
                return out.where(gate > PARAMS["gate_min"], 0).fillna(0).clip(-1, 1)

        ''')
    if name == "flag_breakout":
        return textwrap.dedent('''\
                hi, lo = df["High"], df["Low"]
                pole, flag = int(PARAMS["pole_bars"]), int(PARAMS["flag_bars"])
                out = pd.Series(0.0, index=df.index)
                for i in range(pole + flag, len(df)):
                    if out.iloc[i - 1] != 0:
                        continue
                    ph = hi.iloc[i - flag - pole:i - flag].max()
                    pl = lo.iloc[i - flag - pole:i - flag].min()
                    fh = hi.iloc[i - flag:i].max()
                    fl = lo.iloc[i - flag:i].min()
                    pm = (ph / pl - 1) * 100
                    fr = (fh / fl - 1) * 100
                    ret = close.iloc[i] / close.iloc[i - 1] - 1
                    if pm >= PARAMS["pole_min_pct"] and fr <= PARAMS["flag_tight_pct"] and close.iloc[i] > fh and ret > 0:
                        out.iloc[i] = 1.0
                    if pm <= -PARAMS["pole_min_pct"] and fr <= PARAMS["flag_tight_pct"] and close.iloc[i] < fl and ret < 0:
                        out.iloc[i] = -1.0
                return out

        ''')
    return "        return pd.Series(0.0, index=close.index)\n\n"


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #

@dataclass
class AlgorithmResult:
    name: str
    family: str
    archetype: str
    params: dict
    backtest_params: dict
    provenance: str
    equity: pd.Series
    returns: pd.Series
    metrics: dict
    trades: list
    train_metrics: Optional[dict] = None
    test_metrics: Optional[dict] = None
    universe: list = field(default_factory=list)
    ops_algo: Optional[str] = None
    ensemble_weights: Optional[dict] = None
    members: list = field(default_factory=list)
    build_notes: list = field(default_factory=list)
    window_returns: dict = field(default_factory=dict)
    window_metrics: dict = field(default_factory=dict)
    trade_narratives: list = field(default_factory=list)
    run_index: int = 1
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "family": self.family,
            "archetype": self.archetype, "params": self.params,
            "backtest_params": self.backtest_params, "provenance": self.provenance,
            "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in self.metrics.items()},
            "train_metrics": self.train_metrics, "test_metrics": self.test_metrics,
            "universe": self.universe, "ops_algo": self.ops_algo,
            "ensemble_weights": self.ensemble_weights, "members": self.members,
            "build_notes": self.build_notes, "window_returns": self.window_returns,
            "window_metrics": self.window_metrics, "run_index": self.run_index,
        }

    def strategy_spec(self) -> dict:
        """Serialize this algorithm into a deployable paper-trading strategy spec.

        Carries everything needed to re-run the strategy live: the archetype and
        its fitted parameters, the execution/risk parameters, the universe, and a
        summary of the backtest that justified it. Consumed by the paper-trading
        strategy runner (`strategy_spec_to_signal` / `strategy_spec_to_weights`)
        and shown back to the user when they deploy it."""
        return {
            "name": self.name,
            "family": self.family,
            "archetype": self.archetype,
            "params": dict(self.params),
            "backtest_params": dict(self.backtest_params),
            "universe": list(self.universe),
            "ops_algo": self.ops_algo,
            "direction": str(self.backtest_params.get("direction", "long_only")),
            "provenance": self.provenance or "",
            "metrics": {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in self.metrics.items()},
            "window_returns": dict(self.window_returns),
            "window_metrics": {k: (dict(v) if isinstance(v, dict) else v)
                               for k, v in self.window_metrics.items()},
            "source": "algorithm_builder",
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_csv(self) -> str:
        eq = self.equity.rename("Equity")
        ret = self.returns.rename("Return")
        sig = eq.index.to_series().rename("Date")
        out = pd.concat([sig, eq, ret], axis=1)
        out["Date"] = out["Date"].astype(str)
        return out.to_csv(index=False)

    def to_markdown(self, include_code: bool = True) -> str:
        m = self.metrics
        lines = [
            f"# {self.name}",
            "",
            f"**Family:** {self.family} · **Archetype:** {self.archetype}",
            f"**Provenance:** {self.provenance}",
            "",
            "## Backtest metrics (in-sample full window)",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Total return | {m['total_return'] * 100:.2f}% |",
            f"| CAGR | {m['cagr'] * 100:.2f}% |",
            f"| Sharpe | {m['sharpe']:.2f} |",
            f"| Sortino | {m['sortino']:.2f} |",
            f"| Volatility | {m['volatility'] * 100:.2f}% |",
            f"| Max drawdown | {m['max_drawdown'] * 100:.2f}% |",
            f"| Calmar | {m['calmar']:.2f} |",
            f"| Win rate | {m['win_rate'] * 100:.1f}% |",
            f"| Profit factor | {m['profit_factor']:.2f} |",
            f"| Trades | {m['trades']} |",
            "",
            "## Parameters",
            "",
            "```json",
            json.dumps(self.params, indent=2),
            "```",
            "",
            "## Backtest parameters",
            "",
            "```json",
            json.dumps(self.backtest_params, indent=2),
            "```",
        ]
        if self.train_metrics and self.test_metrics:
            lines += [
                "",
                "## Train / test split (honest out-of-sample)",
                "",
                f"- Train Sharpe: {self.train_metrics.get('sharpe', 0):.2f} "
                f"(drawdown {self.train_metrics.get('max_drawdown', 0) * 100:.1f}%)",
                f"- **Test (OOS) Sharpe: {self.test_metrics.get('sharpe', 0):.2f}** "
                f"(drawdown {self.test_metrics.get('max_drawdown', 0) * 100:.1f}%, trades {self.test_metrics.get('trades', 0)})",
            ]
        if self.window_metrics:
            lines += ["", "## Breakdown by time window", "",
                      "| Window | Return | Sharpe | Max DD | Trades |", "| --- | --- | --- | --- | --- |"]
            for k, v in self.window_metrics.items():
                if isinstance(v, dict):
                    lines.append(f"| {k} | {v.get('total_return', 0) * 100:+.2f}% | "
                                 f"{v.get('sharpe', 0):.2f} | {v.get('max_drawdown', 0) * 100:.2f}% | "
                                 f"{v.get('trades', 0)} |")
                else:
                    lines.append(f"| {k} | — | — | — | — |")
        elif self.window_returns:
            lines += ["", "## Return by window", "", "| Window | Return |", "| --- | --- |"]
            for k, v in self.window_returns.items():
                lines.append(f"| {k} | {v * 100:+.2f}% |" if v is not None else f"| {k} | — |")
        closed_t = [t for t in self.trades if t.get("exit_pnl_pct") is not None]
        if closed_t:
            lines += ["", "## Trade log", "",
                      "| Entry | Exit | Side | P&L % | Hold | Exit reason |",
                      "| --- | --- | --- | --- | --- | --- |"]
            for t in closed_t[:100]:
                lines.append(f"| {t.get('entry_date', '')} | {t.get('exit_date', '')} | "
                             f"{t.get('direction', '')} | "
                             f"{t.get('exit_pnl_pct', 0):+.2f}% | "
                             f"{t.get('hold_bars', 0)} | {t.get('exit_reason', '')} |")
            if len(closed_t) > 100:
                lines.append(f"| _...and {len(closed_t) - 100} more trades_ | | | | | |")
        if self.trade_narratives:
            lines += ["", "## Trade-by-trade reasoning", ""]
            for n in self.trade_narratives[:60]:
                lines.append(f"- {n}")
            if len(self.trade_narratives) > 60:
                lines.append(f"- _...and {len(self.trade_narratives) - 60} more trades_")
        if self.build_notes:
            lines += ["", "## Build notes", ""] + [f"- {n}" for n in self.build_notes]
        if self.ensemble_weights:
            lines += ["", "## Ensemble weights", ""]
            for k, v in self.ensemble_weights.items():
                lines.append(f"- {k}: {v * 100:.1f}%")
            lines.append("")
            lines.append(f"**Members:** {', '.join(self.members)}")
        if include_code:
            lines += ["", "## Generated code", "", "```python", self.code(), "```"]
        lines += ["", "---", "*Educational use only — not financial advice.*"]
        return "\n".join(lines)

    def code(self) -> str:
        return generate_python(self.archetype, self.params, self.backtest_params,
                               self.universe, self.ops_algo)


def _default_backtest_params(risk: str, direction: str) -> dict:
    if risk == "conservative":
        return {"sizing": "vol_target", "risk_pct": 0.08, "max_leverage": 1.0,
                "stop_loss_pct": 5.0, "take_profit_pct": None, "trailing_pct": None,
                "max_hold_bars": 0, "commission_bps": 3.0, "slippage_bps": 5.0,
                "direction": "long_only", "max_dd_cap": -0.25}
    if risk == "aggressive":
        return {"sizing": "vol_target", "risk_pct": 0.25, "max_leverage": 2.0,
                "stop_loss_pct": 8.0, "take_profit_pct": None, "trailing_pct": 12.0,
                "max_hold_bars": 0, "commission_bps": 3.0, "slippage_bps": 5.0,
                "direction": direction, "max_dd_cap": -0.70}
    return {"sizing": "vol_target", "risk_pct": 0.15, "max_leverage": 1.5,
            "stop_loss_pct": 6.0, "take_profit_pct": None, "trailing_pct": 8.0,
            "max_hold_bars": 0, "commission_bps": 3.0, "slippage_bps": 5.0,
            "direction": direction, "max_dd_cap": -0.45}


def _fmt_metric(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def strategy_spec_to_signal(spec: dict, df: pd.DataFrame) -> Optional[pd.Series]:
    """Recompute a deployed strategy's target-exposure signal on fresh OHLCV data.

    Returns a Series in [-1, 1] (per-bar target exposure), or None for
    portfolio-level (online_ops) strategies — those are rebalanced by weights via
    `strategy_spec_to_weights`. The direction is applied the same way the
    backtester did (long_only clips shorts)."""
    if not spec or df is None:
        return None
    archetype = spec.get("archetype", "")
    if archetype == "online_ops" or archetype not in ARCHETYPES:
        return None
    try:
        a = ARCHETYPES[archetype]
        params = {p["name"]: spec.get("params", {}).get(p["name"], p["default"])
                  for p in a["params"]}
        sig = a["sig"](df, params).fillna(0.0).clip(-1.0, 1.0)
        if spec.get("direction", "long_only") == "long_only":
            sig = sig.clip(lower=0.0)
        return sig
    except Exception:
        return None


def strategy_spec_to_weights(spec: dict, dfs: dict) -> Optional[dict]:
    """For portfolio-level (online_ops) strategies: recompute the current target
    weights across the universe from a price matrix, or None if not applicable."""
    if not spec or spec.get("archetype") != "online_ops" or len(dfs) < 2:
        return None
    try:
        prices = pd.DataFrame({s: dfs[s]["Close"] for s in dfs})
        ops_algo = spec.get("ops_algo") or "pamr"
        if ops_algo not in OPS_ALGOS:
            ops_algo = "pamr"
        params = {p["name"]: p["default"] for p in OPS_ALGOS[ops_algo]["params"]}
        for p in OPS_ALGOS[ops_algo]["params"]:
            if p["name"] in spec.get("params", {}):
                params[p["name"]] = spec["params"][p["name"]]
        res = run_ops_basket(prices, ops_algo, params)
        return res["final_weights"]
    except Exception:
        return None


def build_algorithms(
    request: str = "",
    mode: str = "auto",
    count: int = 3,
    ensemble: bool = False,
    risk: str = "balanced",
    direction: str = "long_only",
    universe: Optional[list] = None,
    archetypes: Optional[list] = None,
    locked_params: Optional[dict] = None,
    n_trials: int = 25,
    seed: int = 42,
    data_fn: Callable = _default_get_stock,
    period: str = "5y",
    backtest_params_override: Optional[dict] = None,
    runs_per_family: int = 3,
    ensemble_method: str = "dynamic",
    wf_folds: int = 0,
) -> list:
    """Build one or more algorithms from a request.

    Auto mode parses the natural-language request; guided mode uses the explicit
    `archetypes` list. Each family is searched `runs_per_family` times with
    distinct seeds ("many backtests per algorithm"); every result carries a
    5y/3y/2y/1y/6m/3m/1m window breakdown and per-trade reasoning. Ensemble
    mode appends a *dynamically weighted* ensemble card — weights are tilted by
    each member's measured strengths (Sharpe/drawdown/win rate/trade count) and
    penalized for redundancy (correlation) on top of wealth-adaptive mixing.
    """
    rng = np.random.default_rng(seed)
    universe = universe or ["SPY"]
    universe = [u.strip().upper() for u in universe if u and u.strip()]
    if not universe:
        universe = ["SPY"]

    if mode == "guided" and archetypes:
        families = [a for a in archetypes if a in ARCHETYPES or a == "online_ops"]
    else:
        families = parse_request(request, risk)
    if not families:
        families = _DEFAULT_BY_RISK.get(risk, _DEFAULT_BY_RISK["balanced"])

    # fetch data (single-symbol archetypes)
    dfs: dict = {}
    fetch_errors: list = []
    for sym in universe[:6]:
        try:
            df = data_fn(sym, period=period, interval="1d")
            if df is None or len(df) < 60:
                raise ValueError(f"insufficient history for {sym}")
            for col in ("Open", "High", "Low", "Close", "Volume"):
                if col not in df.columns:
                    df[col] = df["Close"] if col == "Close" else df.get(col)
            dfs[sym] = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception as e:  # noqa: BLE001
            fetch_errors.append(f"{sym}: {e}")
    if not dfs:
        raise RuntimeError("Could not fetch price data for any symbol in the universe: "
                           + "; ".join(fetch_errors))
    primary_sym = universe[0] if universe[0] in dfs else next(iter(dfs))
    primary = dfs[primary_sym]

    bp = backtest_params_override or _default_backtest_params(risk, direction)
    locked = locked_params or {}
    runs = max(int(runs_per_family), 1)
    multi = runs > 1

    results: list = []
    used_families = families[: max(count, 1)]
    for i, fam in enumerate(used_families):
        if fam == "online_ops":
            if len(dfs) < 2:
                results.append(_error_result("online_ops",
                    "Online portfolio selection needs ≥ 2 assets in the universe. "
                    "Add more tickers (e.g. SPY, QQQ, IWM) or pick another family."))
                continue
            prices = pd.DataFrame({s: dfs[s]["Close"] for s in dfs})
            ops_algo = locked.get("ops_algo", "pamr")
            params = {p["name"]: p["default"] for p in OPS_ALGOS[ops_algo]["params"]}
            for p in OPS_ALGOS[ops_algo]["params"]:
                if p["name"] in locked:
                    params[p["name"]] = locked[p["name"]]
            res = run_ops_basket(prices, ops_algo, params)
            prov = OPS_ALGOS[ops_algo]["label"] + " — " + (
                "Glucksman Fellowship paper (NYU Stern) online portfolio selection family."
                if ops_algo in ("pamr", "cwmr", "olmar", "anticor") else
                "Glucksman Fellowship paper FTRL (regularized Follow-the-Winner).")
            results.append(AlgorithmResult(
                name=f"OPS-{ops_algo.upper()}-{primary_sym}", family="portfolio",
                archetype="online_ops", params=params,
                backtest_params={"universe": list(dfs)},
                provenance=prov, equity=res["equity"], returns=res["returns"],
                metrics=res["metrics"], trades=[], universe=list(dfs),
                ops_algo=ops_algo,
                window_returns=window_returns(res["equity"]),
                window_metrics=window_metrics(res["equity"], res["returns"], []),
                build_notes=["Portfolio-level backtest over " + ", ".join(list(dfs)[:4])]))
            continue
        a = ARCHETYPES[fam]
        family_ok = False
        for run in range(runs):
            run_seed = seed + i * 1013 + run * 577
            trial_rng = np.random.default_rng(run_seed)
            found = search_best(a, primary, n_trials=n_trials, rng=trial_rng,
                                backtest_params=bp, locked=locked, seed=run_seed,
                                wf_folds=wf_folds)
            if "error" in found:
                continue  # this run failed; try the next seed before giving up
            family_ok = True
            params = found["params"]
            sig = a["sig"](primary, params)
            full = backtest_ohlcv(primary, sig, bp)
            suffix = f" (run {run + 1}/{runs})" if multi else ""
            notes = [f"Parameters optimized on train window "
                     f"(train Sharpe {found['train_metrics'].get('sharpe', 0):.2f}), "
                     f"reported OOS test Sharpe "
                     f"{found['test_metrics'].get('sharpe', 0):.2f}."]
            wf = found.get("walk_forward")
            if wf:
                notes.append(
                    f"Walk-forward CV across {wf['folds']} trailing folds "
                    f"(Bergmeir & Hyndman 2018): mean Sharpe {wf['mean_sharpe']:.2f}, "
                    f"median {wf['median_sharpe']:.2f}, worst fold "
                    f"{wf['worst_fold_sharpe']:.2f}, consistency "
                    f"{wf['consistency'] * 100:.0f}% of folds profitable, "
                    f"residual autocorrelation {wf['mean_autocorr']:.3f} "
                    f"({'(signal left structure on the table)' if abs(wf['mean_autocorr']) > 0.15 else '(signal is well-specified)'}).")
            results.append(AlgorithmResult(
                name=f"{a['label']}-{primary_sym}{suffix}", family=a["family"],
                archetype=fam, params=params, backtest_params=bp,
                provenance=a["provenance"],
                equity=full["equity"], returns=full["returns"], metrics=full["metrics"],
                trades=full["trades"], train_metrics=found["train_metrics"],
                test_metrics=found["test_metrics"], universe=[primary_sym],
                window_returns=window_returns(full["equity"]),
                window_metrics=window_metrics(full["equity"], full["returns"],
                                               full["trades"]),
                trade_narratives=trade_narratives(full["trades"], primary, sig, bp),
                run_index=run + 1, build_notes=notes))
        if not family_ok:
            results.append(_error_result(fam, "no viable parameter set found on the training window",
                                         suggestion=diagnose_failure(fam, primary)))

    # ensemble of the built members (skip error cards)
    if ensemble and len(results) >= 2:
        members = [r for r in results if r.metrics.get("trades", 0) > 0 or r.family == "portfolio"]
        # cap membership so the blend stays interpretable: top 6 by OOS/test Sharpe
        members.sort(key=lambda r: (r.test_metrics or {}).get("sharpe", r.metrics.get("sharpe", 0.0)),
                     reverse=True)
        members = members[:6]
        if len(members) >= 2:
            method = ensemble_method or "dynamic"
            ens = build_ensemble(members, primary, method=method)
            ens_notes = [f"Blend method: {method}. Per-bar weights combine wealth-adaptive "
                         f"mixing with a quality tilt so stronger, less-redundant members "
                         f"get more weight (weaknesses down-weighted)."]
            if ens.get("rationale"):
                ens_notes.extend(ens["rationale"])
            results.append(AlgorithmResult(
                name=f"ENSEMBLE-{len(members)}-strategies", family="ensemble",
                archetype="ensemble", params={"method": method,
                                              "members": [m.name for m in members],
                                              # full member specs so the exported
                                              # Python can rebuild the blend
                                              "members_detail": [
                                                  {"name": m.name,
                                                   "archetype": m.archetype,
                                                   "params": m.params,
                                                   "ops_algo": m.ops_algo,
                                                   "backtest_params": m.backtest_params,
                                                   "universe": m.universe}
                                                  for m in members
                                              ]},
                backtest_params=bp,
                provenance=("Dynamic ensemble: members are blended per-bar by cumulative "
                            "wealth tilted by measured quality (Sharpe, drawdown, win rate, "
                            "trade robustness) and penalized for cross-member correlation — "
                            "maximizing each member's strengths while down-weighting its "
                            "weaknesses."),
                equity=ens["equity"], returns=ens["returns"], metrics=ens["metrics"],
                trades=[], train_metrics=None, test_metrics=None,
                universe=[primary_sym], ensemble_weights=ens["final_weights"],
                members=[m.name for m in members],
                window_returns=window_returns(ens["equity"]),
                window_metrics=window_metrics(ens["equity"], ens["returns"], []),
                build_notes=ens_notes))
    return results


def _error_result(family: str, msg: str, suggestion: str = "") -> AlgorithmResult:
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    eq = pd.Series([100_000.0, 100_000.0], index=idx)
    notes = ["ERROR: " + msg]
    if suggestion:
        notes.append("SUGGESTION: " + suggestion)
    return AlgorithmResult(
        name=f"{family}-ERROR", family=family, archetype=family,
        params={}, backtest_params={}, provenance="",
        equity=eq, returns=pd.Series([0.0, 0.0], index=idx),
        metrics={"total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
                 "max_drawdown": 0.0, "calmar": 0.0, "volatility": 0.0, "win_rate": 0.0,
                 "profit_factor": 0.0, "trades": 0, "exposure": 0.0, "best_trade": 0.0,
                 "worst_trade": 0.0, "avg_hold_bars": 0.0, "years": 0.0},
        trades=[], build_notes=notes)


def _data_character(df: pd.DataFrame) -> dict:
    """Quick statistical profile of a price series: trend strength, mean-reversion
    potential, volatility — used to explain *why* a family failed and what to tweak."""
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    rets = close.pct_change().dropna()
    out = {"bars": int(len(df)), "total_return": None, "vol": None,
           "trend_strength": None, "autocorr1": None, "meanrev_index": None}
    if len(rets) < 30:
        return out
    out["total_return"] = float(close.iloc[-1] / close.iloc[0] - 1.0)
    vol = float(rets.std() * math.sqrt(BARS_PER_YEAR))
    out["vol"] = vol
    mean_d = float(rets.mean())
    # |mean|/std over a year — high means a strong, persistent drift
    out["trend_strength"] = float(abs(mean_d) / max(rets.std(), 1e-9) * math.sqrt(BARS_PER_YEAR))
    out["autocorr1"] = float(rets.autocorr(1)) if len(rets) > 5 else None
    ac = out["autocorr1"] or 0.0
    out["meanrev_index"] = float(max(-ac, 0.0))  # negative autocorr => mean-reverting
    return out


def diagnose_failure(family: str, df: pd.DataFrame) -> str:
    """Explain why a family produced no viable strategy on this data, and suggest
    concrete tweaks plus families that suit the data's character."""
    c = _data_character(df)
    base = (f"{c['bars']} bars, {c['total_return'] * 100:+.0f}% over the window"
            if c["total_return"] is not None else f"{c['bars']} bars")
    vol_txt = f"{c['vol'] * 100:.0f}% annualized vol" if c["vol"] else ""
    fam_label = ARCHETYPES.get(family, {}).get("label", family)
    lines = [f"{fam_label} found no viable parameter set on your data ({base}, {vol_txt})."]
    if family in ("rsi_meanrev", "bollinger_meanrev", "statarb_z", "gap_fade"):
        ac = c["autocorr1"]
        ts = c["trend_strength"]
        if (ac is not None and ac > 0.15) or (ts is not None and ts > 2.0):
            why = (f"lag-1 autocorrelation {ac:+.2f}" if ac is not None and ac > 0.15
                   else f"a persistent drift ({ts:.1f} annualized sigmas)")
            lines.append(f"Mean reversion needs oscillation, but your series is trending "
                         f"({why}): entries almost never trigger, so no trial produced a single "
                         f"trade.")
            lines.append("Tweaks that could unlock it: (a) a range-bound / mean-reverting asset, "
                         "(b) a longer backtest with more regime variety, (c) looser entry "
                         "thresholds (e.g. higher oversold RSI, lower z-score entry) via the "
                         "search trials.")
            lines.append("Better-suited alternatives: trend / momentum families "
                         "(Moving-Average Crossover, Dual Momentum, Donchian Breakout) or "
                         "volatility targeting.")
        else:
            lines.append("The series is not strongly trending, but entries still never fired — "
                         "try more search trials, a wider universe, or a volatility-targeting "
                         "family that works in any regime.")
    elif family in ("trend_ma", "breakout", "dual_momentum"):
        ac = c["autocorr1"]
        if ac is not None and ac < -0.1:
            lines.append(f"Trend following needs persistent direction, but your series mean-reverts "
                         f"(lag-1 autocorrelation {ac:+.2f}): breakouts and crossovers whipsaw.")
            lines.append("Tweaks: (a) a trending asset, (b) wider breakout windows / slower "
                         "averages, (c) an absolute-momentum gate.")
            lines.append("Better-suited alternatives: mean-reversion families (RSI, Bollinger, "
                         "z-score stat-arb) or gap fading.")
        else:
            lines.append("Breakouts never produced trades — try more search trials, a longer "
                         "period, or a mean-reversion family on this data.")
    else:
        lines.append("Try more search trials, a longer history, or a different family — the "
                     "search could not find parameters that trade at least once on the "
                     "training window.")
    return " ".join(lines)


def build_request_signature(request: str, mode: str, count: int, ensemble: bool,
                            risk: str, direction: str, universe: list,
                            archetypes: list, locked: dict, n_trials: int, seed: int,
                            runs_per_family: int = 3, ensemble_method: str = "dynamic",
                            wf_folds: int = 0) -> str:
    blob = json.dumps([request, mode, count, ensemble, risk, direction,
                       universe, archetypes, locked, n_trials, seed,
                       runs_per_family, ensemble_method, wf_folds], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# Re-export for UI convenience
ALL_ARCHETYPE_NAMES = list(ARCHETYPES.keys()) + ["online_ops"]
RISK_PROFILES = ["conservative", "balanced", "aggressive"]
EXPORT_FORMATS = ["python", "json", "csv", "markdown"]
