import io
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from config import ALPHA_VANTAGE_KEY, OANDA_API_KEY, POLYGON_API_KEY


# ---------------------------------------------------------------------------
# DATA CORRECTIONS REGISTRY
#
# Known-bad data points are managed as data (data_corrections.json), not as
# hardcoded values in source code, so every correction is attributable,
# reviewable, and removable without a code change.
# ---------------------------------------------------------------------------

_CORRECTIONS_PATH = Path(__file__).resolve().parent / "data_corrections.json"
_CORRECTIONS: dict = {}


def _load_corrections() -> dict:
    """Load the corrections registry (cached after first read)."""
    global _CORRECTIONS
    if _CORRECTIONS:
        return _CORRECTIONS
    try:
        with open(_CORRECTIONS_PATH, "r") as f:
            data = json.load(f)
        corr: dict = {}
        for entry in data.get("corrections", []):
            sym = (entry.get("symbol") or "").upper()
            dt = entry.get("date")
            if not sym or not dt:
                continue
            corr.setdefault(sym, {})[dt] = entry
        _CORRECTIONS = corr
    except Exception as e:
        logging.getLogger(__name__).warning("Could not load data_corrections.json: %s", e)
        _CORRECTIONS = {}
    return _CORRECTIONS

OANDA_BASE_URL = "https://api-fxpractice.oanda.com"

# Suppress noisy yfinance logging when Yahoo is flaky
for _logger_name in ("yfinance", "yfinance.shared", "yfinance.utils"):
    logging.getLogger(_logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(_logger_name).propagate = False
    logging.getLogger(_logger_name).disabled = True

# If Yahoo starts returning invalid JSON, back off for a short cooldown
_YF_SKIP_UNTIL_TS: float = 0.0

# yfinance's module-level `download()` writes into shared module globals
# (e.g. shared._DFS), so concurrent calls from multiple threads can
# cross-contaminate results (symbol A receiving symbol B's data).
# Serialize all network fetches behind a lock so parallel callers
# (watchlist ThreadPoolExecutor, scanners, etc.) never mix instruments.
_YF_FETCH_LOCK = threading.RLock()  # RLock: reentrant-safe against nested fetch paths


_CACHE: dict = {}


def _apply_data_corrections(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Apply registered corrections for known bad data points (data-driven)."""
    if df.empty:
        return df

    corrections = _load_corrections()
    sym_corrections = corrections.get((ticker or "").upper())
    if not sym_corrections:
        return df

    date_strs = df.index.strftime("%Y-%m-%d")
    for bad_date, entry in sym_corrections.items():
        if bad_date not in date_strs:
            continue
        ohlc = entry.get("ohlc", {})
        corrected_close = ohlc.get("Close")
        if corrected_close is None:
            continue
        idx_list = df.index[date_strs == bad_date]
        for idx in idx_list:
            for col, val in ohlc.items():
                if col in df.columns:
                    df.loc[idx, col] = val
            if "Adj Close" in df.columns:
                df.loc[idx, "Adj Close"] = corrected_close

            # Adjust previous close so the daily change computes correctly.
            prev_close = entry.get("prev_close")
            if prev_close is not None:
                pos = df.index.get_loc(idx)
                if pos > 0:
                    prev_idx = df.index[pos - 1]
                    df.loc[prev_idx, "Close"] = prev_close
                    if "Adj Close" in df.columns:
                        df.loc[prev_idx, "Adj Close"] = prev_close

    return df


def _normalize_ohlc(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        # Detect which level has the OHLC price names
        ohlc_names = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
        level_0_names = set(str(x) for x in df.columns.get_level_values(0).unique())
        level_1_names = set(str(x) for x in df.columns.get_level_values(1).unique())
        
        if level_0_names & ohlc_names:
            # Level 0 has price names (e.g., ('Close', 'AAPL')) - drop level 1 (ticker)
            df.columns = df.columns.droplevel(1)
        elif level_1_names & ohlc_names:
            # Level 1 has price names (e.g., ('AAPL', 'Close')) - drop level 0 (ticker)
            df.columns = df.columns.droplevel(0)
        else:
            # Fallback: try dropping the last level
            df.columns = df.columns.droplevel(-1)

    df = df.rename(columns=str.title)
    required_cols = ["Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()

    out = df.copy()
    if "Adj Close" in out.columns:
        out["Close"] = out["Adj Close"]

    out = out.dropna()

    # Apply specific data corrections
    if ticker:
        out = _apply_data_corrections(out, ticker)

    return out


def _cache_key(ticker: str, period: str, interval: str) -> str:
    return f"{ticker}::{period}::{interval}"


def _get_dynamic_ttl(ticker: str, interval: str) -> int:
    """Intelligent Response Caching: Dynamic TTL based on asset type and interval."""
    sym = ticker.upper()
    
    # Intraday data needs fresher cache than daily
    if interval in ["1m", "5m", "15m", "30m", "1h"]:
        if sym.endswith("-USD"): return 15  # Crypto is 24/7 and highly volatile
        if sym.endswith("=F"): return 30    # Futures
        if "^VIX" in sym: return 10         # Volatility index needs ultra-low latency
        if "/" in sym or sym.endswith("=X"): return 30 # Forex
        return 60  # Standard intraday equities
    
    # Daily or longer intervals
    if sym.endswith("-USD"): return 120
    if "^VIX" in sym: return 120
    return 300  # Default 5 min for daily standard equities

def _get_cached(
    ticker: str, period: str, interval: str, ttl_seconds: int = None
) -> pd.DataFrame:
    key = _cache_key(ticker, period, interval)
    hit = _CACHE.get(key)
    if not hit:
        return pd.DataFrame()

    ts, df = hit
    
    # Apply context-aware dynamic TTL if none specifically requested
    if ttl_seconds is None:
        ttl_seconds = _get_dynamic_ttl(ticker, interval)
        
    if (time.time() - ts) > ttl_seconds:
        return pd.DataFrame()
    return df.copy()

def _set_cached(ticker: str, period: str, interval: str, df: pd.DataFrame) -> None:
    key = _cache_key(ticker, period, interval)
    _CACHE[key] = (time.time(), df.copy())


def _period_to_compact(period: str) -> str:
    # AlphaVantage is daily-only for the endpoints we use here, so we just fetch the full daily history
    return period


def _period_to_days(period: str) -> int:
    try:
        p = (period or "").strip().lower()
        if p == "max":
            return 36500  # 100 years
        if p.endswith("y"):
            return int(p[:-1]) * 365
        if p.endswith("mo"):
            return int(p[:-2]) * 30
        if p.endswith("d"):
            return int(p[:-1])
    except Exception:
        return 365
    return 365

def _slice_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df is None or df.empty or not period or period.lower() == "max":
        return df
    days = _period_to_days(period)
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - timedelta(days=days)
    df_idx = df.index
    if df_idx.tz is not None:
        df_idx = df_idx.tz_localize(None)
    
    sliced = df[df_idx >= cutoff].copy()
    if sliced.empty:
        return df.tail(10)  # fallback if period too short
    return sliced


def _polygon_ticker(symbol: str) -> str:
    s = symbol.strip().upper()
    index_map = {
        "^GSPC": "I:SPX",
        "^IXIC": "I:NDX",
        "^DJI": "I:DJI",
        "^VIX": "I:VIX",
    }
    return index_map.get(s, s)


def _fetch_polygon_daily(symbol: str, period: str) -> pd.DataFrame:
    api_key = os.environ.get("POLYGON_API_KEY") or POLYGON_API_KEY
    if not api_key:
        return pd.DataFrame()

    try:
        days = _period_to_days(period)
        end = datetime.utcnow().date()
        start = end - timedelta(days=days)

        poly_symbol = _polygon_ticker(symbol)
        url = f"https://api.polygon.io/v2/aggs/ticker/{poly_symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": api_key,
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()

        results = data.get("results")
        if not isinstance(results, list) or not results:
            return pd.DataFrame()

        rows = []
        for bar in results:
            ts = bar.get("t")
            if ts is None:
                continue
            dt = datetime.utcfromtimestamp(ts / 1000.0)
            rows.append(
                {
                    "Date": dt,
                    "Open": bar.get("o"),
                    "High": bar.get("h"),
                    "Low": bar.get("l"),
                    "Close": bar.get("c"),
                    "Volume": bar.get("v"),
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
        df = df.set_index("Date").sort_index()
        return _normalize_ohlc(df, symbol)
    except Exception:
        return pd.DataFrame()


def _fetch_alpha_vantage_daily(symbol: str, period: str = "max") -> pd.DataFrame:
    api_key = os.environ.get("ALPHA_VANTAGE_KEY") or ALPHA_VANTAGE_KEY
    if not api_key:
        return pd.DataFrame()

    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "compact",
            "apikey": api_key,
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()

        series = data.get("Time Series (Daily)")
        if not isinstance(series, dict) or not series:
            return pd.DataFrame()

        rows = []
        for dt, values in series.items():
            rows.append(
                {
                    "Date": dt,
                    "Open": float(values.get("1. open"))
                    if values.get("1. open") is not None
                    else None,
                    "High": float(values.get("2. high"))
                    if values.get("2. high") is not None
                    else None,
                    "Low": float(values.get("3. low"))
                    if values.get("3. low") is not None
                    else None,
                    "Close": float(values.get("4. close"))
                    if values.get("4. close") is not None
                    else None,
                    "Volume": float(values.get("6. volume"))
                    if values.get("6. volume") is not None
                    else None,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        return _slice_by_period(_normalize_ohlc(df, symbol), period)
    except Exception:
        return pd.DataFrame()


def _stooq_symbol(symbol: str) -> str:
    s = symbol.strip().lower()

    # Indices / common symbols mapping
    stooq_map = {
        "^gspc": "^spx",
        "^ixic": "^ndq",
        "^dji": "^dji",
        "^vix": "^vix",
        "spx": "^spx",
    }

    # Common futures mappings (best-effort)
    futures_map = {
        "es=f": "es.f",
        "nq=f": "nq.f",
        "ym=f": "ym.f",
        "rty=f": "rty.f",
        "cl=f": "cl.f",
        "ng=f": "ng.f",
        "gc=f": "gc.f",
        "si=f": "si.f",
        "hg=f": "hg.f",
        "pa=f": "pa.f",
        "pl=f": "pl.f",
        "zc=f": "zc.f",
        "zs=f": "zs.f",
        "zw=f": "zw.f",
        "zn=f": "zn.f",
        "zb=f": "zb.f",
        "zf=f": "zf.f",
        "zt=f": "zt.f",
    }

    # FX pairs: Yahoo format "EURUSD=X" → stooq format "eurusd"
    # Also handle slash-separated "EUR/USD" → "eurusd"
    if s.endswith("=x") and len(s) == 8:
        # e.g. "eurusd=x" → "eurusd"
        return s[:-2]
    if "/" in s:
        # e.g. "eur/usd" → "eurusd"
        clean = s.replace("/", "").replace("_", "")
        if len(clean) == 6 and clean.isalpha():
            return clean

    if s in stooq_map:
        return stooq_map[s]
    if s in futures_map:
        return futures_map[s]

    # US equities/ETFs
    if s.isalpha() and len(s) <= 5:
        return f"{s}.us"

    return s


def _fetch_stooq_daily(symbol: str, period: str = "max") -> pd.DataFrame:
    try:
        stooq_s = _stooq_symbol(symbol)
        url = "https://stooq.com/q/d/l/"
        params = {"s": stooq_s, "i": "d"}
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if df is None or df.empty:
            return pd.DataFrame()

        # Stooq uses: Date,Open,High,Low,Close,Volume
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        return _slice_by_period(_normalize_ohlc(df, symbol), period)
    except Exception:
        return pd.DataFrame()


def _safe_yf_download(ticker, period="3y", interval="1d"):
    cached = _get_cached(ticker, period, interval)
    if not cached.empty:
        return cached

    # Stooq/AlphaVantage/Polygon fallbacks are daily-only
    daily_only_fallbacks = interval in ("1d", "1wk", "1mo")

    ticker_str = str(ticker) if ticker is not None else ""
    equity_like = ticker_str.isalpha() and len(ticker_str) <= 5
    index_like = ticker_str.startswith("^")
    futures_like = ticker_str.lower().endswith("=f")
    # FX pairs formatted as "EURUSD=X" (Yahoo Finance format) or slash-separated
    fx_like = ticker_str.upper().endswith("=X") or (
        "/" in ticker_str and len(ticker_str.replace("/", "").replace("_", "")) == 6
    )

    # For equities/indices/futures/FX, prefer resilient providers before touching Yahoo
    if daily_only_fallbacks and (equity_like or index_like or futures_like or fx_like):
        df = _fetch_stooq_daily(ticker_str, period)
        if not df.empty:
            _set_cached(ticker_str, period, interval, df)
            return df

        df = _fetch_polygon_daily(ticker_str, period)
        if not df.empty:
            _set_cached(ticker_str, period, interval, df)
            return df

        df = _fetch_alpha_vantage_daily(ticker_str, period)
        if not df.empty:
            _set_cached(ticker_str, period, interval, df)
            return df

    global _YF_SKIP_UNTIL_TS
    now_ts = time.time()

    # If Yahoo is in cooldown and caller asked for intraday data, return best-effort daily data instead
    if now_ts < _YF_SKIP_UNTIL_TS and not daily_only_fallbacks:
        df = _fetch_stooq_daily(ticker_str, period)
        if not df.empty:
            _set_cached(ticker_str, period, interval, df)
            return df

        df = _fetch_polygon_daily(ticker_str, period)
        if not df.empty:
            _set_cached(ticker_str, period, interval, df)
            return df

        df = _fetch_alpha_vantage_daily(ticker_str, period)
        if not df.empty:
            _set_cached(ticker_str, period, interval, df)
            return df

    # For FX pairs: always try stooq first even in cooldown to avoid Yahoo dependency
    if fx_like and now_ts < _YF_SKIP_UNTIL_TS:
        df = _fetch_stooq_daily(ticker_str, period)
        if not df.empty:
            _set_cached(ticker_str, period, interval, df)
            return df

    # Retry Yahoo a couple times (often transient JSONDecodeError).
    # The entire download attempt is serialized: yfinance's `download()`
    # mutates shared module state and is not thread-safe, so we hold the
    # lock for the full retry cycle.
    last_exc: Exception | None = None
    if now_ts >= _YF_SKIP_UNTIL_TS:
        with _YF_FETCH_LOCK:
            for attempt in range(3):
                try:
                    df = yf.download(
                        ticker,
                        period=period,
                        interval=interval,
                        progress=False,
                        threads=False,
                    )
                    df = _normalize_ohlc(df, ticker)
                    if not df.empty:
                        _set_cached(ticker, period, interval, df)
                        return df

                    # If Yahoo returns empty for a likely-valid ticker, treat as outage and switch providers
                    if (
                        attempt == 0
                        and isinstance(ticker, str)
                        and ticker
                        and len(ticker) <= 10
                    ):
                        _YF_SKIP_UNTIL_TS = time.time() + 600
                        break
                except Exception as e:
                    last_exc = e
                    msg = str(e)
                    if "Expecting value" in msg or "JSONDecodeError" in msg:
                        _YF_SKIP_UNTIL_TS = time.time() + 600
                time.sleep(0.6 * (attempt + 1))

    # If Yahoo was skipped/failed and interval is intraday, fall back to daily providers
    if not daily_only_fallbacks:
        df = _fetch_stooq_daily(ticker_str, period)
        if not df.empty:
            _set_cached(ticker_str, period, interval, df)
            return df

        df = _fetch_polygon_daily(ticker_str, period)
        if not df.empty:
            _set_cached(ticker_str, period, interval, df)
            return df

        df = _fetch_alpha_vantage_daily(ticker_str)
        if not df.empty:
            _set_cached(ticker_str, period, interval, df)
            return df

    if daily_only_fallbacks:
        # Fallback providers for when Yahoo is failing
        df = _fetch_polygon_daily(ticker_str, period)
        if not df.empty:
            _set_cached(ticker_str, period, interval, df)
            return df

        # Prefer Stooq (no key, generally stable) before Alpha Vantage (rate-limited)
        df = _fetch_stooq_daily(ticker_str, period)
        if not df.empty:
            _set_cached(ticker_str, period, interval, df)
            return df

        df = _fetch_alpha_vantage_daily(ticker_str)
        if not df.empty:
            _set_cached(ticker_str, period, interval, df)
            return df

    if last_exc is not None:
        print(f"Yahoo download failed for {ticker}: {last_exc}")
    return pd.DataFrame()


def get_stock(symbol, period="3y", interval="1d"):
    return _safe_yf_download(symbol, period=period, interval=interval)


def get_stock_data(symbol, period="3y", interval="1d"):
    return get_stock(symbol, period, interval)


def get_futures_proxy(ticker, period="3y", interval="1d"):
    """Get futures data with ETF fallbacks for robustness."""
    df = _safe_yf_download(ticker, period=period, interval=interval)
    if not df.empty:
        return df

    # Fallback to ETF proxies if direct futures data fails
    # This ensures "free" access via ETF equivalents when futures data is restricted/unavailable
    proxies = {
        "ES=F": "SPY",  # S&P 500
        "NQ=F": "QQQ",  # Nasdaq 100
        "RTY=F": "IWM",  # Russell 2000
        "YM=F": "DIA",  # Dow Jones
        "CL=F": "USO",  # Crude Oil
        "GC=F": "GLD",  # Gold
        "SI=F": "SLV",  # Silver
        "NG=F": "UNG",  # Natural Gas
        "ZB=F": "TLT",  # 30Y Treasury
        "ZN=F": "IEF",  # 10Y Treasury
        "ZF=F": "IEI",  # 5Y Treasury
        "ZT=F": "SHY",  # 2Y Treasury
        "HG=F": "CPER",  # Copper
        "BTC=F": "BTC-USD",  # Bitcoin
        "ETH=F": "ETH-USD",  # Ethereum
    }

    t_upper = ticker.upper()
    if t_upper in proxies:
        proxy = proxies[t_upper]
        print(
            f"Futures data unavailable for {ticker}, falling back to ETF proxy {proxy}"
        )
        return get_stock(proxy, period=period, interval=interval)

    return pd.DataFrame()


def get_vix(period="3y"):
    return _safe_yf_download("^VIX", period=period)


def get_fx(pair, count=500, period="2y", interval="1d"):
    """
    Fetch FX data with multi-provider fallback chain.
    Provider order: OANDA → Stooq → Yahoo Finance (direct) → Yahoo Finance (inverted)
    Handles formats: "EUR/USD", "EURUSD", "EURUSD=X", "EUR_USD"
    """
    # Normalise to 6-char code, e.g. "EURUSD"
    clean_pair = (
        pair.replace("/", "")
        .replace("_", "")
        .replace("=X", "")
        .replace("=x", "")
        .upper()
    )

    # Try OANDA first if configured
    if OANDA_API_KEY and OANDA_API_KEY != "your_oanda_key_here":
        try:
            # OANDA expects format "EUR_USD"
            oanda_pair = clean_pair[:3] + "_" + clean_pair[3:]

            url = f"{OANDA_BASE_URL}/v3/instruments/{oanda_pair}/candles"
            headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
            params = {"count": count, "granularity": "D", "price": "M"}

            r = requests.get(url, headers=headers, params=params, timeout=5)
            r.raise_for_status()
            data = r.json()["candles"]

            rows = []
            for c in data:
                if c["complete"]:
                    rows.append(
                        {
                            "Date": c["time"][:10],
                            "Open": float(c["mid"]["o"]),
                            "High": float(c["mid"]["h"]),
                            "Low": float(c["mid"]["l"]),
                            "Close": float(c["mid"]["c"]),
                        }
                    )

            df = pd.DataFrame(rows)
            if not df.empty:
                df["Date"] = pd.to_datetime(df["Date"])
                return _normalize_ohlc(df.set_index("Date"), pair)

        except Exception as e:
            print(f"OANDA fetch failed for {pair}: {e}")
            # Fall through to next provider

    # Provider 2: Stooq (free, no API key, very reliable for major FX)
    # Stooq uses lowercase 6-char code, e.g. "eurusd"
    try:
        stooq_sym = clean_pair.lower()
        url = "https://stooq.com/q/d/l/"
        params = {"s": stooq_sym, "i": "d"}
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        df_stooq = pd.read_csv(io.StringIO(r.text))
        if df_stooq is not None and not df_stooq.empty and "Date" in df_stooq.columns:
            df_stooq["Date"] = pd.to_datetime(df_stooq["Date"], errors="coerce")
            df_stooq = df_stooq.dropna(subset=["Date"]).set_index("Date").sort_index()
            df_stooq = _slice_by_period(_normalize_ohlc(df_stooq, pair), period)
            if not df_stooq.empty:
                return df_stooq
    except Exception as e:
        print(f"Stooq FX fetch failed for {pair}: {e}")

    # Provider 3: Yahoo Finance (direct) — "EURUSD=X"
    yf_ticker = clean_pair + "=X"
    df = _safe_yf_download(yf_ticker, period=period, interval=interval)
    if not df.empty:
        return df

    # Provider 4: Yahoo Finance (inverted pair fallback)
    # e.g. if USDJPY=X fails, try JPYUSD=X and invert
    if len(clean_pair) == 6:
        inv_pair = clean_pair[3:] + clean_pair[:3]
        inv_ticker = inv_pair + "=X"
        df_inv = _safe_yf_download(inv_ticker, period=period, interval=interval)

        if not df_inv.empty:
            df_out = df_inv.copy()
            df_out["Open"] = 1 / df_inv["Open"]
            df_out["Close"] = 1 / df_inv["Close"]
            df_out["High"] = 1 / df_inv["Low"]  # Swap High/Low when inverting
            df_out["Low"] = 1 / df_inv["High"]

            if "Adj Close" in df_out.columns:
                df_out["Adj Close"] = 1 / df_inv["Adj Close"]

            return df_out

    print(f"[get_fx] All providers failed for {pair}")
    return pd.DataFrame()


def _flatten_yf_columns(df, symbol=None):
    """Flatten yfinance multi-level columns to simple column names."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        if symbol:
            try:
                tickers_in_cols = df.columns.get_level_values(1).unique()
                sym_upper = symbol.upper()
                if sym_upper in tickers_in_cols:
                    df = df.xs(sym_upper, level=1, axis=1)
                    return df
            except Exception:
                pass
        df.columns = df.columns.get_level_values(0)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def get_fresh_quote(symbol: str, period: str = "3mo"):
    """Fetch clean OHLCV data for a single symbol, bypassing cache.
    Handles forex (USD/JPY, USDJPY=X), futures (ES=F), crypto (BTC-USD), indices (^GSPC).

    Provider order for FX pairs: Stooq (free, reliable) → Yahoo Finance
    Provider order for all others: Yahoo Finance → Stooq fallback
    """
    sym = symbol.strip()

    # Detect FX pair: "EUR/USD", "EURUSD", "EURUSD=X", "EUR_USD"
    clean6 = (
        sym.replace("/", "")
        .replace("_", "")
        .replace("=X", "")
        .replace("=x", "")
        .upper()
    )
    is_fx = (
        (sym.upper().endswith("=X") and len(sym) == 8)
        or ("/" in sym and len(clean6) == 6 and clean6.isalpha())
        or (len(clean6) == 6 and clean6.isalpha() and not sym.startswith("^"))
    )

    #  For FX pairs: try Stooq FIRST (no API key, very reliable) 
    if is_fx:
        try:
            df = _fetch_stooq_daily(clean6 + "=X", period)  # _stooq_symbol strips "=X" → clean6
            if df is not None and not df.empty and "Close" in df.columns:
                return df
        except Exception:
            pass

    #  Yahoo Finance 
    try:
        import yfinance as yf

        yf_sym = sym
        if "/" in yf_sym and "=" not in yf_sym:
            yf_sym = yf_sym.replace("/", "") + "=X"

        tk = yf.Ticker(yf_sym)
        df = tk.history(period=period)
        if df is not None and not df.empty:
            df = _flatten_yf_columns(df, yf_sym)
            if "Close" in df.columns and len(df) > 0:
                return df
    except Exception:
        pass

    #  For non-FX: Stooq fallback if Yahoo failed 
    if not is_fx:
        try:
            df = _fetch_stooq_daily(sym, period)
            if df is not None and not df.empty and "Close" in df.columns:
                return df
        except Exception:
            pass

    return None


def _fetch_polygon_realtime(symbol: str):
    """Fetch latest price from Polygon.io daily bars (returns current, prev_close)."""
    api_key = os.environ.get("POLYGON_API_KEY") or POLYGON_API_KEY
    if not api_key:
        return None, None

    try:
        from datetime import timezone

        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=10)
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/range/1/day/{start}/{end}"
        params = {"adjusted": "true", "sort": "desc", "limit": 5, "apiKey": api_key}
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            if results and len(results) >= 1:
                current = results[0].get("c")
                prev = results[1].get("c") if len(results) >= 2 else None
                if current and current > 0:
                    return float(current), float(prev) if prev else None
    except Exception:
        pass
    return None, None


_REALTIME_CACHE: dict = {}
_REALTIME_CACHE_TTL = 5  # 5 seconds for live price cache


def get_realtime_price(symbol: str):
    """
    Get the most accurate real-time price with previous close.
    Returns (current_price, previous_close) tuple.
    Uses yfinance as primary source (intraday data), with fallbacks.
    Results cached for 15 seconds to avoid hammering APIs on rapid refreshes.
    """
    sym = symbol.strip().upper()

    # Check short-lived realtime cache
    cache_key = f"rt:{sym}"
    cached = _REALTIME_CACHE.get(cache_key)
    if cached:
        ts, result = cached
        if (time.time() - ts) < _REALTIME_CACHE_TTL:
            return result

    # Normalize forex symbols
    yf_sym = sym
    if "/" in sym and "=" not in sym:
        yf_sym = sym.replace("/", "") + "=X"

    current_price = None
    prev_close = None

    # Method 1: Try WebSocket Stream Engine (Ultra-low latency)
    try:
        from websocket_engine import get_engine, get_websocket_price
        # Initialize engine if not started
        get_engine()
        ws_price = get_websocket_price(sym)
        if ws_price is not None:
            # We have live price, just need previous close from yfinance fast_info cache
            current_price = ws_price
    except ImportError:
        pass

    # Method 2: Try yfinance (always attempt for prev_close, ignore cooldown for realtime)
    try:
        import yfinance as yf

        tk = yf.Ticker(yf_sym)

        # Always try fast_info first - it's the fastest and most reliable
        # for getting both current price AND previous close
        try:
            fi = tk.fast_info
            
            # Use yf price if websocket didn't provide one
            if current_price is None:
                price = getattr(fi, "last_price", None) or getattr(fi, "regularMarketPrice", None)
                if price and price > 0:
                    current_price = float(price)
            
            prev = getattr(fi, "previous_close", None) or getattr(fi, "regularMarketPreviousClose", None)
            if prev and prev > 0:
                prev_close = float(prev)
            print(f"DEBUG: YF fast_info for {yf_sym}: current={current_price}, prev={prev_close}")
        except Exception:
            pass

        # Try 1-minute data for most up-to-date price
        if current_price is None:
            try:
                df_1m = tk.history(period="1d", interval="1m")
                if df_1m is not None and not df_1m.empty:
                    df_1m = _flatten_yf_columns(df_1m, yf_sym)
                    if "Close" in df_1m.columns:
                        c = df_1m["Close"]
                        if isinstance(c, pd.DataFrame):
                            c = c.iloc[:, 0]
                        c = c.dropna()
                        if len(c) > 0:
                            current_price = float(c.iloc[-1])
            except Exception:
                pass

        # Try daily data only if we still need prev_close
        if prev_close is None:
            try:
                df_daily = tk.history(period="5d")
                if df_daily is not None and not df_daily.empty:
                    df_daily = _flatten_yf_columns(df_daily, yf_sym)
                    if "Close" in df_daily.columns:
                        c = df_daily["Close"]
                        if isinstance(c, pd.DataFrame):
                            c = c.iloc[:, 0]
                        c = c.dropna()
                        if len(c) >= 2:
                            if current_price is None:
                                current_price = float(c.iloc[-1])
                            prev_close = float(c.iloc[-2])
                        elif len(c) == 1 and current_price is None:
                            current_price = float(c.iloc[-1])
            except Exception:
                pass
    except Exception:
        pass

    # Method 2: Fallback to Polygon if yfinance failed entirely
    if current_price is None:
        poly_price, poly_prev = _fetch_polygon_realtime(sym)
        if poly_price:
            current_price = poly_price
            if prev_close is None and poly_prev:
                prev_close = poly_prev
        print(f"DEBUG: Polygon fallback for {sym}: current={current_price}, prev={prev_close}")

    # Method 3: Fallback to Stooq (end-of-day only)
    if current_price is None:
        try:
            df = _fetch_stooq_daily(sym, "5d")
            if not df.empty and "Close" in df.columns:
                c = df["Close"]
                if isinstance(c, pd.DataFrame):
                    c = c.iloc[:, 0]
                c = c.dropna()
                if len(c) >= 2:
                    current_price = float(c.iloc[-1])
                    if prev_close is None:
                        prev_close = float(c.iloc[-2])
                elif len(c) == 1:
                    current_price = float(c.iloc[-1])
        except Exception:
            pass

    # Method 4: Fallback to Alpha Vantage
    if current_price is None:
        try:
            df = _fetch_alpha_vantage_daily(sym, "5d")
            if not df.empty and "Close" in df.columns:
                c = df["Close"]
                if isinstance(c, pd.DataFrame):
                    c = c.iloc[:, 0]
                c = c.dropna()
                if len(c) >= 2:
                    current_price = float(c.iloc[-1])
                    if prev_close is None:
                        prev_close = float(c.iloc[-2])
                elif len(c) == 1:
                    current_price = float(c.iloc[-1])
        except Exception:
            pass

    result = (current_price, prev_close)
    _REALTIME_CACHE[cache_key] = (time.time(), result)
    return result


def get_latest_price(symbol: str):
    """
    Get the freshest possible price for a symbol, bypassing cache.

    Returns a positive float when a real price is available, otherwise None.
    A price of 0 is NEVER returned: presenting 0.0 as a market price would be
    fabricated data. Callers must treat None as "price unavailable".
    """
    price, _ = get_realtime_price(symbol)
    if price is not None and price > 0:
        return float(price)

    # Last resort: stock info snapshot (still a real provider value).
    try:
        import yfinance as yf
        tk = yf.Ticker(symbol)
        info = tk.info
        price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        if price and price > 0:
            return float(price)
    except Exception:
        pass
    return None


def get_realtime_prices_batch(symbols: list):
    """
    Fetch live prices for multiple symbols in a single API call.
    Returns dict of {symbol: (current_price, previous_close)}.
    Uses yfinance batch download to minimize API calls.
    """
    results = {}
    if not symbols:
        return results

    # Check cache first
    uncached = []
    for sym in symbols:
        cache_key = f"rt:{sym.strip().upper()}"
        cached = _REALTIME_CACHE.get(cache_key)
        if cached:
            ts, result = cached
            if (time.time() - ts) < _REALTIME_CACHE_TTL:
                results[sym.strip().upper()] = result
                continue
        uncached.append(sym.strip().upper())

    if not uncached:
        return results

    # Normalize symbols for yfinance
    yf_map = {}
    for sym in uncached:
        yf_sym = sym
        if "/" in sym and "=" not in sym:
            yf_sym = sym.replace("/", "") + "=X"
        yf_map[yf_sym] = sym

    # Method 1: yfinance fast_info batch (most reliable for live prices)
    try:
        import yfinance as yf

        for yf_sym, orig_sym in list(yf_map.items()):
            if orig_sym in results:
                continue
            try:
                tk = yf.Ticker(yf_sym)
                fi = tk.fast_info
                price = getattr(fi, "last_price", None) or getattr(
                    fi, "regularMarketPrice", None
                )
                prev = getattr(fi, "previous_close", None) or getattr(
                    fi, "regularMarketPreviousClose", None
                )
                if price and price > 0:
                    result = (float(price), float(prev) if prev and prev > 0 else None)
                    results[orig_sym] = result
                    _REALTIME_CACHE[f"rt:{orig_sym}"] = (time.time(), result)
                    print(f"DEBUG: Batch YF fast_info for {orig_sym}: current={price}, prev={prev}")
            except Exception as e:
                print(f"DEBUG: Batch YF fast_info failed for {orig_sym}: {e}")
    except Exception:
        pass

    # Method 2: Polygon fallback for remaining
    still_missing = [s for s in uncached if s not in results]
    for sym in still_missing:
        try:
            poly_price, poly_prev = _fetch_polygon_realtime(sym)
            if poly_price:
                result = (poly_price, poly_prev)
                results[sym] = result
                _REALTIME_CACHE[f"rt:{sym}"] = (time.time(), result)
        except Exception:
            pass

    return results


# Patch get_stock to always flatten columns
_original_get_stock = None


def _patched_get_stock(symbol, period="6mo", **kwargs):
    """Wrapper around original get_stock that ensures clean single-level columns."""
    global _original_get_stock
    try:
        df = _original_get_stock(symbol, period=period, **kwargs)
        if df is not None and not df.empty:
            df = _flatten_yf_columns(df, symbol)
        return df
    except Exception:
        # Direct yfinance fallback
        try:
            import yfinance as yf

            tk = yf.Ticker(symbol)
            df = tk.history(period=period)
            if df is not None and not df.empty:
                df = _flatten_yf_columns(df, symbol)
                return df
        except Exception:
            pass
        return pd.DataFrame()


# Auto-patch get_stock on module load if not already patched
try:
    if _original_get_stock is None:
        import sys

        _this = sys.modules[__name__]
        # Save original
        _original_get_stock = get_stock
        # Replace with patched version
        get_stock = _patched_get_stock
except Exception:
    pass


def _normalize_options_chain(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Normalize an options chain DataFrame to a consistent schema.

    Ensures the following columns are present, substituting 0 for any that are
    missing: vol, strike, lastPrice, bid, ask, impliedVolatility, openInterest.

    yfinance returns 'volume' rather than 'vol', so if 'vol' is absent but
    'volume' is present the column is renamed.  If neither exists a zero-filled
    'vol' column is inserted and a warning is logged.

    Args:
        df: Raw options chain DataFrame (calls or puts) from yfinance.
        symbol: Underlying ticker symbol — used in log messages.

    Returns:
        DataFrame with a guaranteed consistent schema.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # --- Handle vol / volume column ---
    if 'vol' not in df.columns:
        if 'volume' in df.columns:
            df = df.rename(columns={'volume': 'vol'})
        else:
            logging.warning(f"vol column missing for {symbol}; defaulting to 0")
            df['vol'] = 0

    # --- Ensure all required numeric columns exist ---
    required_numeric = {
        'strike': 0,
        'lastPrice': 0,
        'bid': 0,
        'ask': 0,
        'impliedVolatility': 0.0,
        'openInterest': 0,
    }
    for col, default in required_numeric.items():
        if col not in df.columns:
            logging.warning(f"{col} column missing for {symbol}; defaulting to {default}")
            df[col] = default

    return df


def get_options_chain(symbol: str, expiration: Optional[str] = None) -> dict:
    """
    Fetch live options chain data for a symbol via yfinance.

    Args:
        symbol: Ticker symbol (e.g., 'AAPL')
        expiration: Specific expiry date (e.g., '2026-05-15'). If None, returns nearest.

    Returns:
        Dict containing calls, puts, expirations, and underlying price information.
        On total fetch failure returns:
            {'calls': pd.DataFrame(), 'puts': pd.DataFrame(),
             'error': "Options chain unavailable for {symbol}. Please try again later."}
    """
    try:
        tk = yf.Ticker(symbol)
        expirations = tk.options

        if not expirations:
            logging.warning(f"No options found for {symbol}")
            return {
                'calls': pd.DataFrame(),
                'puts': pd.DataFrame(),
                'error': f"Options chain unavailable for {symbol}. Please try again later.",
            }

        # Select expiration
        target_exp = expiration if expiration in expirations else expirations[0]
        opt = tk.option_chain(target_exp)

        # Get underlying price
        price_data = get_realtime_price(symbol)
        underlying_price = price_data[0] if price_data else None

        return {
            'symbol': symbol,
            'expiration': target_exp,
            'expirations': expirations,
            'calls': _normalize_options_chain(opt.calls, symbol),
            'puts': _normalize_options_chain(opt.puts, symbol),
            'underlying_price': underlying_price,
            'timestamp': datetime.now().isoformat(),
        }
    except Exception as e:
        logging.error(f"Error fetching options chain for {symbol}: {e}")
        return {
            'calls': pd.DataFrame(),
            'puts': pd.DataFrame(),
            'error': f"Options chain unavailable for {symbol}. Please try again later.",
        }
