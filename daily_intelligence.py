"""
Daily Market Intelligence Engine - Professional Grade
Generates automated daily briefings with institutional-depth analysis:
  - Market regime detection via HMM
  - Macro indicators (yield curve, inflation, Fed signals)
  - Sector rotation analysis
  - Cross-asset correlation signals
  - Volatility surface insights
  - Earnings & catalyst calendar
  - Narrative dislocation detection
  - Technical analysis on major indices
  - Global macro overview
"""

import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DailyIntel")

#  Safe imports 
try:
    import yfinance as yf

    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    from hmm_engine import get_regime_detector

    HAS_HMM = True
except Exception:
    HAS_HMM = False

try:
    from data_sources import get_stock

    HAS_DATA = True
except Exception:
    HAS_DATA = False

try:
    from market_movers import fetch_market_movers

    HAS_MOVERS = True
except Exception:
    HAS_MOVERS = False


# 
# Utility helpers
# 


def _safe_yf_download(
    ticker: str, period: str = "1y", interval: str = "1d"
) -> pd.DataFrame:
    """Safely download yfinance data, returning empty DataFrame on failure."""
    if not HAS_YF:
        return pd.DataFrame()
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df is not None and isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.debug(f"yf download failed for {ticker}: {e}")
        return pd.DataFrame()


def _pct_change_1d(df: pd.DataFrame) -> float:
    """Return the most recent 1-day pct change for Close."""
    if df is None or df.empty or "Close" not in df.columns:
        return 0.0
    close = df["Close"].dropna()
    if len(close) < 2:
        return 0.0
    return float((close.iloc[-1] / close.iloc[-2] - 1) * 100)


def _pct_change_nd(df: pd.DataFrame, n: int = 5) -> float:
    if df is None or df.empty or "Close" not in df.columns:
        return 0.0
    close = df["Close"].dropna()
    if len(close) < n + 1:
        return 0.0
    return float((close.iloc[-1] / close.iloc[-(n + 1)] - 1) * 100)


def _last_close(df: pd.DataFrame) -> float:
    if df is None or df.empty or "Close" not in df.columns:
        return 0.0
    close = df["Close"].dropna()
    return float(close.iloc[-1]) if len(close) > 0 else 0.0


def _annualized_vol(df: pd.DataFrame, window: int = 20) -> float:
    if df is None or df.empty or "Close" not in df.columns:
        return 0.0
    close = df["Close"].dropna()
    if len(close) < window + 1:
        return 0.0
    rets = close.pct_change().dropna()
    return float(rets.tail(window).std() * np.sqrt(252) * 100)


def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.dropna()
    return float(val.iloc[-1]) if len(val) > 0 else 50.0


def _above_sma(df: pd.DataFrame, n: int = 200) -> bool:
    if df is None or df.empty or "Close" not in df.columns:
        return False
    close = df["Close"].dropna()
    if len(close) < n:
        return False
    return float(close.iloc[-1]) > float(close.rolling(n).mean().iloc[-1])


def _macd_signal(df: pd.DataFrame) -> str:
    if df is None or df.empty or "Close" not in df.columns:
        return "Neutral"
    close = df["Close"].dropna()
    if len(close) < 27:
        return "Neutral"
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    if len(hist) < 2:
        return "Neutral"
    h_now = float(hist.iloc[-1])
    h_prev = float(hist.iloc[-2])
    if h_now > 0 and h_now > h_prev:
        return "Bullish"
    if h_now < 0 and h_now < h_prev:
        return "Bearish"
    return "Neutral"


def _bollinger_position(df: pd.DataFrame, window: int = 20) -> str:
    if df is None or df.empty or "Close" not in df.columns:
        return "Middle"
    close = df["Close"].dropna()
    if len(close) < window:
        return "Middle"
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    price = float(close.iloc[-1])
    u = float(upper.iloc[-1])
    l = float(lower.iloc[-1])
    if price >= u * 0.98:
        return "Upper Band (Overbought)"
    if price <= l * 1.02:
        return "Lower Band (Oversold)"
    if price > float(ma.iloc[-1]):
        return "Upper Half"
    return "Lower Half"


# 
# Data collection modules
# 


def _fetch_equity_indices() -> dict:
    """Fetch major global equity indices."""
    tickers = {
        "S&P 500": "^GSPC",
        "NASDAQ 100": "^NDX",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
        "VIX": "^VIX",
        "FTSE 100": "^FTSE",
        "DAX": "^GDAXI",
        "Nikkei 225": "^N225",
        "Shanghai Comp.": "000001.SS",
        "Hang Seng": "^HSI",
        "Euro Stoxx 50": "^STOXX50E",
    }
    result = {}
    for name, tick in tickers.items():
        df = _safe_yf_download(tick, period="6mo")
        if not df.empty:
            result[name] = {
                "ticker": tick,
                "price": _last_close(df),
                "chg_1d": _pct_change_1d(df),
                "chg_5d": _pct_change_nd(df, 5),
                "chg_1m": _pct_change_nd(df, 21),
                "chg_3m": _pct_change_nd(df, 63),
                "vol_20d": _annualized_vol(df),
                "above_200": _above_sma(df, 200),
                "rsi": _rsi(df["Close"].dropna()) if "Close" in df.columns else 50,
                "macd": _macd_signal(df),
                "bb_pos": _bollinger_position(df),
            }
    return result


def _fetch_yield_curve() -> dict:
    """Fetch US Treasury yields and compute curve metrics."""
    yield_tickers = {
        "3M": "^IRX",
        "2Y": "^UST2Y",
        "5Y": "^FVX",
        "10Y": "^TNX",
        "30Y": "^TYX",
    }
    # Fallback tickers for FRED-style via yfinance
    fallback = {
        "2Y": "SHY",
        "10Y": "IEF",
        "30Y": "TLT",
    }
    yields = {}
    for tenor, tick in yield_tickers.items():
        df = _safe_yf_download(tick, period="3mo")
        if not df.empty:
            price = _last_close(df)
            if price > 0:
                yields[tenor] = {
                    "rate": round(price / 100, 4) if price > 1 else round(price, 4),
                    "chg_1d": _pct_change_1d(df),
                    "chg_1m": _pct_change_nd(df, 21),
                }

    # Compute key spreads
    r2 = yields.get("2Y", {}).get("rate", 0)
    r10 = yields.get("10Y", {}).get("rate", 0)
    r3m = yields.get("3M", {}).get("rate", 0)
    r30 = yields.get("30Y", {}).get("rate", 0)

    spread_2s10s = round((r10 - r2) * 100, 1)  # bps
    spread_3m10y = round((r10 - r3m) * 100, 1)  # bps
    spread_2s30s = round((r30 - r2) * 100, 1)  # bps

    if spread_2s10s < -25:
        curve_shape = "Deeply Inverted - Recession Signal"
        curve_signal = "BEARISH"
    elif spread_2s10s < 0:
        curve_shape = "Inverted - Contraction Warning"
        curve_signal = "CAUTION"
    elif spread_2s10s < 50:
        curve_shape = "Flat - Transition Phase"
        curve_signal = "NEUTRAL"
    elif spread_2s10s < 100:
        curve_shape = "Normal - Moderate Growth"
        curve_signal = "BULLISH"
    else:
        curve_shape = "Steep - Strong Growth / Reflation"
        curve_signal = "BULLISH"

    return {
        "yields": yields,
        "spread_2s10s_bps": spread_2s10s,
        "spread_3m10y_bps": spread_3m10y,
        "spread_2s30s_bps": spread_2s30s,
        "curve_shape": curve_shape,
        "curve_signal": curve_signal,
        "r10": r10,
        "r2": r2,
        "r3m": r3m,
    }


def _fetch_fx_rates() -> dict:
    """Fetch key FX pairs."""
    fx_pairs = {
        "EUR/USD": "EURUSD=X",
        "USD/JPY": "USDJPY=X",
        "GBP/USD": "GBPUSD=X",
        "USD/CHF": "USDCHF=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CNY": "USDCNY=X",
        "DXY (Dollar)": "DX-Y.NYB",
        "USD/MXN": "USDMXN=X",
        "USD/BRL": "USDBRL=X",
    }
    result = {}
    for name, tick in fx_pairs.items():
        df = _safe_yf_download(tick, period="3mo")
        if not df.empty:
            result[name] = {
                "rate": _last_close(df),
                "chg_1d": _pct_change_1d(df),
                "chg_5d": _pct_change_nd(df, 5),
                "chg_1m": _pct_change_nd(df, 21),
                "vol": _annualized_vol(df),
            }
    return result


def _fetch_commodities() -> dict:
    """Fetch key commodity prices."""
    commodities = {
        "WTI Crude": "CL=F",
        "Brent Crude": "BZ=F",
        "Natural Gas": "NG=F",
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Copper": "HG=F",
        "Corn": "ZC=F",
        "Wheat": "ZW=F",
        "Soybeans": "ZS=F",
        "Lumber": "LBR=F",
    }
    result = {}
    for name, tick in commodities.items():
        df = _safe_yf_download(tick, period="3mo")
        if not df.empty:
            result[name] = {
                "price": _last_close(df),
                "chg_1d": _pct_change_1d(df),
                "chg_5d": _pct_change_nd(df, 5),
                "chg_1m": _pct_change_nd(df, 21),
                "vol": _annualized_vol(df),
            }
    return result


def _fetch_crypto() -> dict:
    """Fetch key crypto assets."""
    cryptos = {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Solana": "SOL-USD",
        "XRP": "XRP-USD",
    }
    result = {}
    for name, tick in cryptos.items():
        df = _safe_yf_download(tick, period="3mo")
        if not df.empty:
            result[name] = {
                "price": _last_close(df),
                "chg_1d": _pct_change_1d(df),
                "chg_5d": _pct_change_nd(df, 5),
                "chg_1m": _pct_change_nd(df, 21),
                "vol": _annualized_vol(df),
            }
    return result


def _fetch_volatility_surface() -> dict:
    """Fetch VIX term structure and related volatility signals."""
    vix_tickers = {
        "VIX (30d)": "^VIX",
        "VIX3M (3m)": "^VIX3M",
        "VIX6M (6m)": "^VIX6M",
        "VVIX (VIX of VIX)": "^VVIX",
        "MOVE Index (Bond Vol)": "^MOVE",
        "SKEW Index": "^SKEW",
        "GVZ (Gold Vol)": "^GVZ",
        "OVX (Oil Vol)": "^OVX",
        "BVIX (BitVol proxy)": "BITO",  # ETF proxy
    }
    result = {}
    for name, tick in vix_tickers.items():
        df = _safe_yf_download(tick, period="6mo")
        if not df.empty:
            price = _last_close(df)
            chg = _pct_change_1d(df)
            chg_1m = _pct_change_nd(df, 21)
            result[name] = {
                "value": price,
                "chg_1d": chg,
                "chg_1m": chg_1m,
            }

    # VIX term structure analysis
    vix_spot = result.get("VIX (30d)", {}).get("value", 0)
    vix3m = result.get("VIX3M (3m)", {}).get("value", 0)

    if vix_spot > 0 and vix3m > 0:
        vix_ts_slope = vix3m - vix_spot
        if vix_ts_slope > 3:
            ts_regime = "Contango - Normal / Calm"
        elif vix_ts_slope > 0:
            ts_regime = "Mild Contango - Slight Caution"
        elif vix_ts_slope > -3:
            ts_regime = "Flat - Transition"
        else:
            ts_regime = "Backwardation - Elevated Fear / Potential Spike"
    else:
        vix_ts_slope = 0
        ts_regime = "Unavailable"

    # VIX regime classification
    if vix_spot < 15:
        vix_regime = "Complacent - Potential Tail Risk Build-Up"
        vix_color = "green"
    elif vix_spot < 20:
        vix_regime = "Low Volatility - Risk-On Environment"
        vix_color = "green"
    elif vix_spot < 25:
        vix_regime = "Moderate - Normal Market Conditions"
        vix_color = "yellow"
    elif vix_spot < 35:
        vix_regime = "Elevated - Caution / Potential Reversal"
        vix_color = "orange"
    elif vix_spot < 50:
        vix_regime = "High Stress - Fear Dominant"
        vix_color = "red"
    else:
        vix_regime = "Extreme Fear - Crisis Mode"
        vix_color = "darkred"

    skew = result.get("SKEW Index", {}).get("value", 100)
    if skew > 140:
        skew_signal = "Extreme Tail Risk Hedging - Smart Money Worried"
    elif skew > 125:
        skew_signal = "Elevated Skew - Downside Hedging Active"
    elif skew > 115:
        skew_signal = "Normal Skew Range"
    else:
        skew_signal = "Low Skew - Complacent / Under-Hedged"

    result["_meta"] = {
        "vix_spot": vix_spot,
        "vix3m": vix3m,
        "vix_ts_slope": vix_ts_slope,
        "ts_regime": ts_regime,
        "vix_regime": vix_regime,
        "vix_color": vix_color,
        "skew": skew,
        "skew_signal": skew_signal,
    }
    return result


def _fetch_sector_performance() -> dict:
    """Fetch sector ETF performance and rotation signals."""
    sectors = {
        "Technology": "XLK",
        "Healthcare": "XLV",
        "Financials": "XLF",
        "Energy": "XLE",
        "Consumer Disc.": "XLY",
        "Consumer Staples": "XLP",
        "Industrials": "XLI",
        "Materials": "XLB",
        "Real Estate": "XLRE",
        "Utilities": "XLU",
        "Communication": "XLC",
        "Semiconductors": "SOXX",
        "Biotech": "XBI",
        "Regional Banks": "KRE",
        "Clean Energy": "ICLN",
    }
    result = {}
    for name, tick in sectors.items():
        df = _safe_yf_download(tick, period="6mo")
        if not df.empty:
            close = df["Close"].dropna() if "Close" in df.columns else pd.Series()
            vol = _annualized_vol(df)
            rsi_val = _rsi(close) if len(close) > 14 else 50
            result[name] = {
                "ticker": tick,
                "price": _last_close(df),
                "chg_1d": _pct_change_1d(df),
                "chg_5d": _pct_change_nd(df, 5),
                "chg_1m": _pct_change_nd(df, 21),
                "chg_3m": _pct_change_nd(df, 63),
                "vol": vol,
                "rsi": rsi_val,
                "above_50": _above_sma(df, 50),
                "above_200": _above_sma(df, 200),
                "macd": _macd_signal(df),
            }
    return result


def _fetch_macro_indicators() -> dict:
    """Fetch macro-proxy indicators via market instruments."""
    macro = {}

    # Inflation proxies
    tips = _safe_yf_download("TIP", period="6mo")
    tlt = _safe_yf_download("TLT", period="6mo")
    inflation_exp = {}
    if not tips.empty:
        inflation_exp["TIPS ETF Trend"] = _pct_change_nd(tips, 21)
    if not tlt.empty:
        inflation_exp["Long Bond Trend (1m)"] = _pct_change_nd(tlt, 21)

    # Breakeven inflation rates (10Y TIPS spread proxy)
    try:
        spilz = _safe_yf_download("RINF", period="6mo")  # ProShares Inflation ETF
        if not spilz.empty:
            inflation_exp["Inflation Expectation ETF (1m)"] = _pct_change_nd(spilz, 21)
    except Exception:
        pass

    macro["inflation"] = inflation_exp

    # Credit markets
    hyg = _safe_yf_download("HYG", period="6mo")
    lqd = _safe_yf_download("LQD", period="6mo")
    emb = _safe_yf_download("EMB", period="6mo")
    credit = {}
    if not hyg.empty:
        credit["High Yield Bond (HYG 1m)"] = _pct_change_nd(hyg, 21)
        credit["High Yield RSI"] = (
            _rsi(hyg["Close"].dropna()) if "Close" in hyg.columns else 50
        )
    if not lqd.empty:
        credit["Invest. Grade Bond (LQD 1m)"] = _pct_change_nd(lqd, 21)
    if not emb.empty:
        credit["EM Bonds (EMB 1m)"] = _pct_change_nd(emb, 21)

    # HYG vs SPY relative
    spy = _safe_yf_download("SPY", period="3mo")
    if not hyg.empty and not spy.empty:
        hyg_ret = _pct_change_nd(hyg, 21)
        spy_ret = _pct_change_nd(spy, 21)
        credit["HYG vs SPY (Credit vs Equity)"] = round(hyg_ret - spy_ret, 2)
    macro["credit"] = credit

    # Risk appetite proxies
    risk = {}
    eem = _safe_yf_download("EEM", period="3mo")
    gld = _safe_yf_download("GLD", period="3mo")
    slv = _safe_yf_download("SLV", period="3mo")
    uup = _safe_yf_download("UUP", period="3mo")  # Dollar ETF
    xhb = _safe_yf_download("XHB", period="3mo")  # Homebuilders

    if not eem.empty:
        risk["EM Equities (EEM 1m)"] = _pct_change_nd(eem, 21)
    if not gld.empty:
        risk["Gold (GLD 1m)"] = _pct_change_nd(gld, 21)
    if not slv.empty:
        risk["Silver (SLV 1m)"] = _pct_change_nd(slv, 21)
    if not uup.empty:
        risk["US Dollar (UUP 1m)"] = _pct_change_nd(uup, 21)
    if not xhb.empty:
        risk["Homebuilders (XHB 1m)"] = _pct_change_nd(xhb, 21)

    macro["risk_appetite"] = risk

    # Breadth indicators
    breadth = {}
    spy_df = _safe_yf_download("SPY", period="1y")
    qqq_df = _safe_yf_download("QQQ", period="1y")
    iwm_df = _safe_yf_download("IWM", period="1y")
    mdl_df = _safe_yf_download("MDY", period="1y")

    # Small vs large (risk-on/off signal)
    if not iwm_df.empty and not spy_df.empty:
        iwm_ret = _pct_change_nd(iwm_df, 21)
        spy_ret = _pct_change_nd(spy_df, 21)
        breadth["Small vs Large Cap"] = round(iwm_ret - spy_ret, 2)
        breadth["Small Cap Status"] = "Risk-ON" if iwm_ret > spy_ret else "Risk-OFF"

    if not qqq_df.empty:
        breadth["NASDAQ vs S&P"] = round(
            _pct_change_nd(qqq_df, 21) - _pct_change_nd(spy_df, 21), 2
        )

    # Value vs Growth
    val = _safe_yf_download("VTV", period="3mo")
    grw = _safe_yf_download("VUG", period="3mo")
    if not val.empty and not grw.empty:
        breadth["Value vs Growth"] = round(
            _pct_change_nd(val, 21) - _pct_change_nd(grw, 21), 2
        )
        breadth["Rotation Signal"] = (
            "Value Favored" if breadth["Value vs Growth"] > 0 else "Growth Favored"
        )

    macro["breadth"] = breadth

    # Fed proxy: Fed Funds Futures (via SOFR / ZQ futures proxy)
    macro["fed"] = _fetch_fed_signals()

    return macro


def _fetch_fed_signals() -> dict:
    """Estimate Fed posture from market signals."""
    signals = {}
    try:
        # 2-year yield is the most sensitive Fed proxy
        df_2y = _safe_yf_download("^UST2Y", period="6mo")
        df_10y = _safe_yf_download("^TNX", period="6mo")
        df_tips = _safe_yf_download("^RINF", period="3mo")

        if not df_2y.empty:
            r2y_now = _last_close(df_2y)
            r2y_1m = _pct_change_nd(df_2y, 21)
            signals["2Y Yield (Fed Proxy)"] = (
                f"{r2y_now / 100:.2%}" if r2y_now > 1 else f"{r2y_now:.2%}"
            )
            signals["2Y Trend (1m)"] = f"{r2y_1m:+.2f}%"
            if r2y_1m > 2:
                signals["Fed Expectation"] = "Hawkish shift - market pricing MORE hikes"
            elif r2y_1m < -2:
                signals["Fed Expectation"] = "Dovish shift - market pricing CUTS"
            else:
                signals["Fed Expectation"] = "Stable - market expects status quo"

        if not df_10y.empty:
            r10y = _last_close(df_10y)
            signals["10Y Yield"] = f"{r10y / 100:.2%}" if r10y > 1 else f"{r10y:.2%}"

        # Real yield proxy (10Y nominal - inflation expectations)
        signals["Monetary Policy Bias"] = _assess_monetary_bias()
    except Exception as e:
        signals["Error"] = str(e)
    return signals


def _assess_monetary_bias() -> str:
    """Assess monetary policy bias from market signals."""
    try:
        df_2y = _safe_yf_download("^UST2Y", period="3mo")
        if df_2y.empty:
            return "Unknown"
        r2y = _last_close(df_2y)
        chg_1m = _pct_change_nd(df_2y, 21)
        if r2y > 5:
            return "Very Restrictive - High Real Rates"
        elif r2y > 4:
            if chg_1m < -3:
                return "Restrictive but Easing - Transition to Cuts"
            return "Restrictive - Tight Policy Regime"
        elif r2y > 3:
            return "Moderately Tight - Neutral to Restrictive"
        elif r2y > 2:
            return "Neutral - Near-Zero Real Rates"
        else:
            return "Accommodative - Low Rate Environment"
    except Exception:
        return "Unknown"


def _detect_cross_asset_signals(
    indices: dict, yields: dict, fx: dict, commodities: dict, sectors: dict
) -> list:
    """Detect cross-asset signals and divergences."""
    signals = []

    # Signal 1: Equities vs Bond Yields divergence
    sp500_1m = indices.get("S&P 500", {}).get("chg_1m", 0)
    yield_10y_chg = yields.get("yields", {}).get("10Y", {}).get("chg_1m", 0)
    if sp500_1m > 5 and yield_10y_chg > 5:
        signals.append(
            {
                "type": "DIVERGENCE",
                "severity": "HIGH",
                "title": "Stocks Rising WITH Rates",
                "detail": (
                    f"S&P 500 +{sp500_1m:.1f}% while 10Y yields also rising "
                    f"{yield_10y_chg:+.1f}%. Classic reflation/growth regime - "
                    "typically bullish but watch for valuation compression if yields break higher."
                ),
                "implication": "Favor cyclicals, financials, value over growth/duration",
            }
        )
    elif sp500_1m < -3 and yield_10y_chg > 5:
        signals.append(
            {
                "type": "DIVERGENCE",
                "severity": "HIGH",
                "title": "Stagflation Signal - Stocks Down, Rates Up",
                "detail": (
                    f"S&P 500 {sp500_1m:.1f}% while 10Y yields rising {yield_10y_chg:+.1f}%. "
                    "Stagflation risk elevated. This is the worst macro environment for equities."
                ),
                "implication": "Reduce equity exposure, favor commodities, TIPS, real assets",
            }
        )

    # Signal 2: Dollar vs Risk Assets
    dxy_chg = fx.get("DXY (Dollar)", {}).get("chg_1m", 0)
    eem_ret = 0.0
    if "Shanghai Comp." in indices:
        eem_ret = indices["Shanghai Comp."].get("chg_1m", 0)
    if dxy_chg > 3 and eem_ret < -2:
        signals.append(
            {
                "type": "MACRO",
                "severity": "MEDIUM",
                "title": "Dollar Strength Crushing EM / Commodities",
                "detail": (
                    f"DXY +{dxy_chg:.1f}% while EM equities {eem_ret:.1f}%. "
                    "Strong dollar historically headwind for EM, commodities, and multinationals."
                ),
                "implication": "Underweight EM, commodity exporters; favor USD-earners",
            }
        )
    elif dxy_chg < -3:
        signals.append(
            {
                "type": "MACRO",
                "severity": "MEDIUM",
                "title": "Dollar Weakness - Tailwind for Risk Assets",
                "detail": f"DXY {dxy_chg:.1f}% - dollar weakness historically bullish for gold, commodities, EM equities.",
                "implication": "Overweight commodities, gold, EM equities, international developed",
            }
        )

    # Signal 3: Oil vs Inflation signal
    oil_1m = commodities.get("WTI Crude", {}).get("chg_1m", 0)
    gold_1m = commodities.get("Gold", {}).get("chg_1m", 0)
    if oil_1m > 10:
        signals.append(
            {
                "type": "INFLATION",
                "severity": "MEDIUM",
                "title": "Oil Surge - Inflation Re-Acceleration Risk",
                "detail": f"WTI Crude +{oil_1m:.1f}% over past month. Energy inflation historically feeds into CPI with 1-2 month lag.",
                "implication": "TIPS, energy equities, commodity traders benefit; caution on high-multiple growth",
            }
        )
    if gold_1m > 5 and oil_1m > 5:
        signals.append(
            {
                "type": "INFLATION",
                "severity": "HIGH",
                "title": "Hard Assets Rally - Inflation Hedge Accumulation",
                "detail": f"Gold +{gold_1m:.1f}% and Oil +{oil_1m:.1f}%. Institutional rotation into inflation protection.",
                "implication": "Commodity supercycle indicators flashing; real assets over financial assets",
            }
        )

    # Signal 4: Yield curve vs equity sectors
    spread_2s10s = yields.get("spread_2s10s_bps", 0)
    financials_1m = sectors.get("Financials", {}).get("chg_1m", 0)
    if spread_2s10s > 50 and financials_1m > 3:
        signals.append(
            {
                "type": "SECTOR",
                "severity": "LOW",
                "title": "Steepening Curve Fueling Banks",
                "detail": f"2s10s at {spread_2s10s:.0f}bps while Financials +{financials_1m:.1f}%. Steeper curve expands bank NIM.",
                "implication": "Financials, regional banks structurally favored in steepening environment",
            }
        )
    elif spread_2s10s < -30 and financials_1m < 0:
        signals.append(
            {
                "type": "SECTOR",
                "severity": "HIGH",
                "title": "Inverted Curve Pressuring Banks",
                "detail": f"2s10s at {spread_2s10s:.0f}bps (inverted). Financial sector under pressure as NIM compresses.",
                "implication": "Avoid regional banks; prefer non-bank financials or asset managers",
            }
        )

    # Signal 5: Growth vs Defensive rotation
    tech_1m = sectors.get("Technology", {}).get("chg_1m", 0)
    utils_1m = sectors.get("Utilities", {}).get("chg_1m", 0)
    staples_1m = sectors.get("Consumer Staples", {}).get("chg_1m", 0)
    if utils_1m > 3 and staples_1m > 2 and tech_1m < 0:
        signals.append(
            {
                "type": "ROTATION",
                "severity": "MEDIUM",
                "title": "Defensive Rotation Underway",
                "detail": (
                    f"Utilities +{utils_1m:.1f}%, Staples +{staples_1m:.1f}% "
                    f"while Tech {tech_1m:.1f}%. Classic late-cycle/risk-off rotation."
                ),
                "implication": "Reduce beta, add defensives, consider tail hedges",
            }
        )
    elif tech_1m > 5 and utils_1m < 0:
        signals.append(
            {
                "type": "ROTATION",
                "severity": "LOW",
                "title": "Growth / Risk-On Rotation",
                "detail": f"Tech +{tech_1m:.1f}% leading while defensives lag ({utils_1m:.1f}%). Risk appetite elevated.",
                "implication": "Favor growth, momentum, high-beta; market in risk-on mode",
            }
        )

    # Signal 6: Crypto vs Risk-On
    try:
        btc_chg = 0  # will be filled if crypto data is available
        if sp500_1m > 5 and btc_chg > 15:
            signals.append(
                {
                    "type": "RISK",
                    "severity": "MEDIUM",
                    "title": "Risk-On Euphoria - Both Equities and Crypto Surging",
                    "detail": "Correlated gains across equities and crypto signal broad risk appetite. Watch for crowding.",
                    "implication": "Late-stage risk-on; consider partial profit-taking and vol hedges",
                }
            )
    except Exception:
        pass

    # Signal 7: VIX vs equity level
    vix = indices.get("VIX", {}).get("price", 20)
    if vix < 14 and sp500_1m > 5:
        signals.append(
            {
                "type": "COMPLACENCY",
                "severity": "HIGH",
                "title": "Extreme Complacency - VIX Near Historic Lows",
                "detail": (
                    f"VIX at {vix:.1f} while S&P up {sp500_1m:.1f}%. "
                    "Sub-14 VIX historically precedes volatility spikes. Options are cheap."
                ),
                "implication": "Consider long volatility, tail hedges; excellent time to buy cheap puts",
            }
        )

    return signals


def _get_regime_analysis() -> dict:
    """Get HMM regime analysis for SPY."""
    if not HAS_HMM:
        return {
            "regime": "Unknown",
            "confidence": 0,
            "description": "HMM not available",
        }
    try:
        if HAS_DATA:
            from data_sources import get_stock

            df = get_stock("SPY", period="1y", interval="1d")
        else:
            df = _safe_yf_download("SPY", period="1y")
        if df is None or df.empty:
            return {"regime": "No Data", "confidence": 0, "description": ""}
        detector = get_regime_detector()
        result = detector.predict_regime(df)
        return result
    except Exception as e:
        logger.error(f"Regime detection failed: {e}")
        return {"regime": "Error", "confidence": 0, "description": str(e)}


def _detect_narrative_dislocations(
    indices: dict, sectors: dict, yields: dict, vix_data: dict
) -> list:
    """Detect narrative vs. price dislocations - where the story and the data diverge."""
    dislocations = []

    sp500_3m = indices.get("S&P 500", {}).get("chg_3m", 0)
    sp500_rsi = indices.get("S&P 500", {}).get("rsi", 50)
    vix_val = vix_data.get("_meta", {}).get("vix_spot", 20)
    spread_2s10s = yields.get("spread_2s10s_bps", 0)
    r10 = yields.get("r10", 0)

    # Dislocation 1: Equities at highs, yield curve inverted
    if sp500_3m > 10 and spread_2s10s < -20:
        dislocations.append(
            {
                "title": "Stocks at Highs but Yield Curve Screaming Recession",
                "narrative": "Equity market pricing in continued growth and earnings expansion",
                "reality": f"Yield curve inverted at {spread_2s10s:.0f}bps - historically predicts recession within 12-18 months",
                "implication": "Either the yield curve is wrong (this time is different) or equities are priced for perfection",
                "probability": "Equity market risk is elevated. Historical base rate: 7/8 inversions precede recessions.",
                "severity": "HIGH",
            }
        )

    # Dislocation 2: Low VIX + high RSI = complacency at market highs
    if vix_val < 16 and sp500_rsi > 68:
        dislocations.append(
            {
                "title": "Euphoria Signal - Low Fear + Overbought Technicals",
                "narrative": f"VIX at {vix_val:.1f} suggests market participants unconcerned about risk",
                "reality": f"S&P RSI at {sp500_rsi:.0f} - historically overbought. When everyone is comfortable, markets become fragile.",
                "implication": "Mean reversion risk elevated. Consider trimming longs and adding vol protection.",
                "probability": "Sub-16 VIX at market highs has preceded 10%+ corrections historically.",
                "severity": "MEDIUM",
            }
        )

    # Dislocation 3: Tech outperforming despite rising rates
    tech_rsi = sectors.get("Technology", {}).get("rsi", 50)
    tech_3m = sectors.get("Technology", {}).get("chg_3m", 0)
    if tech_3m > 15 and r10 > 4.5 and tech_rsi > 65:
        dislocations.append(
            {
                "title": "Tech Rallying Despite High Rates - Valuation Divergence",
                "narrative": f"Tech sector +{tech_3m:.1f}% in 3 months amid 'AI-driven growth'",
                "reality": f"10Y yields at {r10 / 100:.2%} historically compress long-duration growth multiples",
                "implication": "Tech valuations stretched vs. discount rate reality. Either rates must fall or multiples must compress.",
                "probability": "DCF math says tech should be underperforming at these rate levels. Either rates decline soon or correction needed.",
                "severity": "HIGH",
            }
        )

    # Dislocation 4: Defensive outperformance + bullish narrative
    utils_3m = sectors.get("Utilities", {}).get("chg_3m", 0)
    staples_3m = sectors.get("Consumer Staples", {}).get("chg_3m", 0)
    if utils_3m > tech_3m + 5 and sp500_3m > 5:
        dislocations.append(
            {
                "title": "Smart Money Hiding in Defensives While Indices Rise",
                "narrative": "Broad market indices showing gains - headline bullish",
                "reality": f"Utilities outperforming Tech by {(utils_3m - tech_3m):.1f}%. Institutional rotation into defensives beneath the surface.",
                "implication": "Leadership rotation is bearish. When defensives outperform at market highs, often a late-cycle signal.",
                "probability": "Sector rotation analysis suggests distribution of risk, not accumulation.",
                "severity": "MEDIUM",
            }
        )

    # Dislocation 5: Small caps diverging from large caps
    russell_3m = indices.get("Russell 2000", {}).get("chg_3m", 0)
    if sp500_3m > 10 and russell_3m < 0:
        dislocations.append(
            {
                "title": "Narrow Market Rally - Only Mega-Caps Working",
                "narrative": "S&P 500 showing strong gains - 'broad bull market'",
                "reality": f"Russell 2000 {russell_3m:.1f}% while S&P +{sp500_3m:.1f}%. Market breadth deteriorating.",
                "implication": "Narrow leadership is bearish. When gains concentrate in few mega-caps, the rally is fragile.",
                "probability": "Historically, narrow breadth precedes corrections or breadth catch-up. Either scenario means risk.",
                "severity": "MEDIUM",
            }
        )

    return dislocations


def _generate_trade_ideas(
    regime: dict,
    cross_asset_signals: list,
    sectors: dict,
    indices: dict,
    yield_data: dict,
    vix_meta: dict,
) -> list:
    """Generate high-conviction trade ideas from cross-asset analysis."""
    ideas = []
    regime_label = regime.get("regime", "Unknown")
    vix_spot = vix_meta.get("vix_spot", 20)
    spread_2s10s = yield_data.get("spread_2s10s_bps", 0)

    # Idea 1: Based on regime
    if "Bull" in regime_label:
        ideas.append(
            {
                "type": "LONG",
                "instrument": "SPY / QQQ",
                "thesis": f"HMM Regime confirms {regime_label}. Trend-following longs in leading indices.",
                "timeframe": "2-6 weeks",
                "risk": "Regime reversal or external shock",
                "sizing": "Full position (regime-confirmed)",
            }
        )
    elif "Bear" in regime_label or "Crash" in regime_label:
        ideas.append(
            {
                "type": "SHORT / HEDGE",
                "instrument": "SPY puts or SH (inverse ETF)",
                "thesis": f"HMM Regime signals {regime_label}. Downside protection warranted.",
                "timeframe": "2-8 weeks",
                "risk": "Policy surprise causing bear market rally",
                "sizing": "Defensive hedge 15-25% of portfolio",
            }
        )

    # Idea 2: Yield curve steepener
    if spread_2s10s < 0:
        ideas.append(
            {
                "type": "MACRO TRADE",
                "instrument": "Long TLT + Short SHY (Steepener)",
                "thesis": f"Curve inverted at {spread_2s10s:.0f}bps. Historically curves un-invert - steepener trade.",
                "timeframe": "3-12 months",
                "risk": "Curve remains inverted longer than expected",
                "sizing": "3-8% of portfolio",
            }
        )

    # Idea 3: VIX-based ideas
    if vix_spot < 15:
        ideas.append(
            {
                "type": "VOLATILITY LONG",
                "instrument": "VIX calls or UVXY partial",
                "thesis": f"VIX at {vix_spot:.1f} - historically cheap. Long vol as tail hedge when VIX sub-15.",
                "timeframe": "4-12 weeks",
                "risk": "Vol remains suppressed; theta decay on options",
                "sizing": "2-4% of portfolio (tail hedge only)",
            }
        )
    elif vix_spot > 35:
        ideas.append(
            {
                "type": "VOLATILITY SHORT / FADE",
                "instrument": "SPY long + short SVIX",
                "thesis": f"VIX at {vix_spot:.1f} - elevated fear historically mean-reverts. Buy the panic.",
                "timeframe": "2-6 weeks",
                "risk": "Crisis deepens; vol spikes further",
                "sizing": "Partial position 5-10% - scale in tranches",
            }
        )

    # Idea 4: Sector rotation trades from signals
    for sig in cross_asset_signals[:3]:
        impl = sig.get("implication", "")
        if "energy" in impl.lower() or "Energy" in impl:
            ideas.append(
                {
                    "type": "SECTOR LONG",
                    "instrument": "XLE (Energy ETF)",
                    "thesis": f"Cross-asset signal: {sig['title']}. {impl}",
                    "timeframe": "4-8 weeks",
                    "risk": "Oil demand shock or demand-side recession",
                    "sizing": "3-6%",
                }
            )
        elif "gold" in impl.lower() or "Gold" in impl:
            ideas.append(
                {
                    "type": "COMMODITIES LONG",
                    "instrument": "GLD or GDX (Gold Miners)",
                    "thesis": f"Cross-asset signal: {sig['title']}. {impl}",
                    "timeframe": "4-12 weeks",
                    "risk": "Dollar strength, rising real yields",
                    "sizing": "3-7%",
                }
            )

    return ideas[:8]  # Cap at 8 ideas


def _recommend_posture(regime: str, vix: float, spread: float) -> dict:
    """Recommend portfolio posture based on regime + macro."""
    if "Bull" in regime and vix < 20 and spread > 0:
        return {
            "label": "Aggressive Long",
            "equity_pct": "70-85%",
            "bond_pct": "5-10%",
            "alternatives": "5-15%",
            "cash": "5-10%",
            "notes": "Full deployment appropriate. Trend-following, momentum strategies favored.",
        }
    elif "Bear" in regime or spread < -30:
        return {
            "label": "Defensive / Risk-Off",
            "equity_pct": "20-35%",
            "bond_pct": "30-40%",
            "alternatives": "10-20%",
            "cash": "20-30%",
            "notes": "Capital preservation mode. Quality, dividend, defensive sectors only. Active hedging.",
        }
    elif vix > 30:
        return {
            "label": "Crisis Mode - Reduce + Hedge",
            "equity_pct": "15-30%",
            "bond_pct": "20-30%",
            "alternatives": "10-20%",
            "cash": "30-50%",
            "notes": "Elevated tail risk. Reduce gross exposure. Add puts, inverse ETFs. Wait for capitulation.",
        }
    elif "Volatile" in regime or vix > 25:
        return {
            "label": "Cautious - Reduced Sizing",
            "equity_pct": "40-55%",
            "bond_pct": "20-30%",
            "alternatives": "10-15%",
            "cash": "15-25%",
            "notes": "Volatility regime active. Smaller position sizes, wider stops. Avoid illiquid positions.",
        }
    else:
        return {
            "label": "Neutral / Hedged",
            "equity_pct": "50-65%",
            "bond_pct": "15-25%",
            "alternatives": "10-15%",
            "cash": "10-15%",
            "notes": "Balanced posture. Active management, sector-selective. Watch macro regime shift signals.",
        }


# 
# Main Engine
# 


class DailyIntelligenceEngine:
    """
    Orchestrates the generation of comprehensive daily market intelligence briefings.
    Combines regime analysis, macro indicators, cross-asset signals, and trade ideas
    into a structured, institutional-grade report.
    """

    def generate_briefing(self, asset_class: str = "US Markets") -> dict:
        """
        Generate a full daily briefing.
        Returns structured data dict with all analysis components.
        """
        report_date = datetime.now().strftime("%A, %B %d, %Y")
        ts = datetime.now().strftime("%H:%M:%S ET")

        #  1. Regime Detection 
        regime_data = _get_regime_analysis()

        #  2. Market Indices 
        indices = _fetch_equity_indices()

        #  3. Yield Curve 
        yield_data = _fetch_yield_curve()

        #  4. Volatility Surface 
        vix_data = _fetch_volatility_surface()
        vix_meta = vix_data.get("_meta", {})

        #  5. FX 
        fx_data = _fetch_fx_rates()

        #  6. Commodities 
        commodities = _fetch_commodities()

        #  7. Crypto 
        crypto = _fetch_crypto()

        #  8. Sector Rotation 
        sectors = _fetch_sector_performance()

        #  9. Macro Indicators 
        macro = _fetch_macro_indicators()

        #  10. Market Movers 
        gainers, losers = [], []
        if HAS_MOVERS:
            try:
                gainers, losers = fetch_market_movers()
            except Exception:
                pass

        #  11. Cross-Asset Signals 
        cross_signals = _detect_cross_asset_signals(
            indices, yield_data, fx_data, commodities, sectors
        )

        #  12. Narrative Dislocations 
        dislocations = _detect_narrative_dislocations(
            indices, sectors, yield_data, vix_data
        )

        #  13. Trade Ideas 
        trade_ideas = _generate_trade_ideas(
            regime_data, cross_signals, sectors, indices, yield_data, vix_meta
        )

        #  14. Portfolio Posture 
        posture = _recommend_posture(
            regime_data.get("regime", "Unknown"),
            vix_meta.get("vix_spot", 20),
            yield_data.get("spread_2s10s_bps", 0),
        )

        #  15. Executive Summary 
        regime_label = regime_data.get("regime", "Unknown")
        regime_conf = regime_data.get("confidence", 0)
        vix_spot = vix_meta.get("vix_spot", 0)
        spread = yield_data.get("spread_2s10s_bps", 0)
        sp500_1d = indices.get("S&P 500", {}).get("chg_1d", 0)
        sp500_1m = indices.get("S&P 500", {}).get("chg_1m", 0)
        curve_shape = yield_data.get("curve_shape", "Unknown")

        # Synthesize overall market tone
        bullish_signals = sum(
            [
                "Bull" in regime_label,
                vix_spot < 20,
                sp500_1m > 3,
                spread > 0,
                macro.get("breadth", {}).get("Small Cap Status", "") == "Risk-ON",
            ]
        )
        bearish_signals = sum(
            [
                "Bear" in regime_label or "Crash" in regime_label,
                vix_spot > 25,
                sp500_1m < -3,
                spread < -30,
                len(dislocations) >= 3,
            ]
        )

        if bullish_signals >= 4:
            market_tone = "RISK-ON | BULLISH"
            tone_color = "green"
        elif bearish_signals >= 3:
            market_tone = "RISK-OFF | BEARISH"
            tone_color = "red"
        elif bullish_signals > bearish_signals:
            market_tone = "CAUTIOUSLY BULLISH"
            tone_color = "lightgreen"
        elif bearish_signals > bullish_signals:
            market_tone = "CAUTIOUSLY BEARISH"
            tone_color = "orange"
        else:
            market_tone = "NEUTRAL | MIXED SIGNALS"
            tone_color = "gray"

        executive_summary = {
            "date": report_date,
            "timestamp": ts,
            "market_tone": market_tone,
            "tone_color": tone_color,
            "regime": regime_label,
            "regime_confidence": regime_conf,
            "regime_desc": regime_data.get("desc", regime_data.get("description", "")),
            "vix": vix_spot,
            "vix_regime": vix_meta.get("vix_regime", ""),
            "yield_curve": curve_shape,
            "spread_2s10s": spread,
            "sp500_1d": sp500_1d,
            "sp500_1m": sp500_1m,
            "posture": posture,
            "key_risks": _compile_key_risks(
                dislocations, cross_signals, vix_meta, yield_data
            ),
            "key_opportunities": _compile_key_opportunities(trade_ideas, cross_signals),
        }

        return {
            "executive_summary": executive_summary,
            "regime": regime_data,
            "indices": indices,
            "yield_curve": yield_data,
            "volatility": vix_data,
            "fx": fx_data,
            "commodities": commodities,
            "crypto": crypto,
            "sectors": sectors,
            "macro": macro,
            "movers": {"gainers": gainers, "losers": losers},
            "cross_asset_signals": cross_signals,
            "narrative_dislocations": dislocations,
            "trade_ideas": trade_ideas,
            "posture": posture,
        }

    #  Legacy interface 

    def _get_market_regime(self):
        return _get_regime_analysis()

    def _recommend_posture(self, regime: str) -> str:
        p = _recommend_posture(regime, 20, 0)
        return p["label"]


def _compile_key_risks(
    dislocations: list, signals: list, vix_meta: dict, yields: dict
) -> list:
    """Compile top 5 key risks for executive summary."""
    risks = []
    for d in dislocations[:3]:
        risks.append(d["title"])
    for s in [s for s in signals if s.get("severity") == "HIGH"][:2]:
        risks.append(s["title"])
    if vix_meta.get("vix_spot", 20) > 30:
        risks.append(f"Elevated VIX at {vix_meta['vix_spot']:.1f} - Fear regime active")
    if yields.get("spread_2s10s_bps", 0) < -30:
        risks.append(
            f"Yield curve inverted at {yields['spread_2s10s_bps']:.0f}bps - Recession signal"
        )
    return risks[:6]


def _compile_key_opportunities(ideas: list, signals: list) -> list:
    """Compile top opportunities."""
    opps = []
    for idea in ideas[:4]:
        opps.append(f"{idea['type']}: {idea['instrument']} - {idea['thesis'][:80]}...")
    for s in [s for s in signals if s.get("severity") in ("LOW", "MEDIUM")][:2]:
        opps.append(s.get("implication", ""))
    return opps[:5]


#  Singleton 
_daily_engine = None


def get_daily_engine() -> DailyIntelligenceEngine:
    global _daily_engine
    if _daily_engine is None:
        _daily_engine = DailyIntelligenceEngine()
    return _daily_engine
