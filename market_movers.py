"""
Octavian Market Movers & Comprehensive Search — Top/Worst Performers & Deep Analysis
Author: APB - Octavian Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import concurrent.futures

from data_sources import get_stock
from ticker_universe import get_ticker_universe

try:
    from data_sources import get_realtime_price as _ds_get_realtime_price
    HAS_REALTIME_PRICE = True
except ImportError:
    HAS_REALTIME_PRICE = False
    _ds_get_realtime_price = None

try:
    from octavian_theme import COLORS, section_header
except ImportError:
    COLORS = {"gold": "#D4AF37", "lavender": "#C4B5E0", "navy": "#0d1117",
              "navy_light": "#161b22", "white_soft": "#e6e6e6",
              "text_primary": "#ffffff", "text_secondary": "#8b949e",
              "border": "#30363d", "danger": "#f85149", "success": "#3fb950"}
    def section_header(text): st.subheader(text)


# 
# ROBUST LIVE PRICE FETCHING
# 

def _get_accurate_live_price(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Get the most accurate live price for any symbol.
    Returns (current_price, previous_close) tuple.
    Uses centralized data_sources.get_realtime_price with multiple fallbacks.
    """
    # Use centralized function (has 15s cache, fast_info priority, multi-provider fallback)
    if HAS_REALTIME_PRICE and _ds_get_realtime_price:
        try:
            return _ds_get_realtime_price(symbol)
        except Exception:
            pass

    # Fallback to direct yfinance if centralized function unavailable
    import yfinance as yf

    sym = symbol.strip().upper()
    yf_sym = sym
    if '/' in sym and '=' not in sym:
        yf_sym = sym.replace('/', '') + '=X'

    current_price = None
    prev_close = None

    try:
        tk = yf.Ticker(yf_sym)

        # fast_info first — fastest and gives both price + previous_close
        try:
            fi = tk.fast_info
            price = getattr(fi, 'last_price', None) or getattr(fi, 'regularMarketPrice', None)
            if price and price > 0:
                current_price = float(price)
            prev = getattr(fi, 'previous_close', None) or getattr(fi, 'regularMarketPreviousClose', None)
            if prev and prev > 0:
                prev_close = float(prev)
        except Exception:
            pass

        # 1-minute intraday for most current price
        if current_price is None:
            try:
                df_1m = tk.history(period="1d", interval="1m")
                if df_1m is not None and not df_1m.empty:
                    if isinstance(df_1m.columns, pd.MultiIndex):
                        df_1m.columns = df_1m.columns.get_level_values(0)
                    if "Close" in df_1m.columns:
                        close_col = df_1m["Close"]
                        if isinstance(close_col, pd.DataFrame):
                            close_col = close_col.iloc[:, 0]
                        close_col = close_col.dropna()
                        if len(close_col) > 0:
                            current_price = float(close_col.iloc[-1])
            except Exception:
                pass

        # Daily data only if we still need prev_close
        if prev_close is None:
            try:
                df_daily = tk.history(period="5d")
                if df_daily is not None and not df_daily.empty:
                    if isinstance(df_daily.columns, pd.MultiIndex):
                        df_daily.columns = df_daily.columns.get_level_values(0)
                    if "Close" in df_daily.columns:
                        close_col = df_daily["Close"]
                        if isinstance(close_col, pd.DataFrame):
                            close_col = close_col.iloc[:, 0]
                        close_col = close_col.dropna()
                        if len(close_col) >= 2:
                            if current_price is None:
                                current_price = float(close_col.iloc[-1])
                            prev_close = float(close_col.iloc[-2])
            except Exception:
                pass

    except Exception:
        pass

    return current_price, prev_close


def _get_live_price(symbol: str) -> float:
    """Get live price for any symbol - wrapper for compatibility."""
    price, _ = _get_accurate_live_price(symbol)
    return price if price and price > 0 else 0.0


# 
# MARKET MOVERS - TOP/WORST PERFORMERS (MARKET-WIDE)
# 

_MIN_MOVER_PRICE = 3.00  # Minimum share price to qualify for top movers


def _parse_screener_quotes(quotes: list) -> List[Dict]:
    """Parse quote dicts from Yahoo screener response into our standard format."""
    results = []
    for q in quotes:
        symbol = q.get('symbol', '')
        if not symbol:
            continue
        price = q.get('regularMarketPrice', 0)
        if not price or price < _MIN_MOVER_PRICE:
            continue
        prev_close = q.get('regularMarketPreviousClose', 0)
        change_pct = q.get('regularMarketChangePercent', 0)
        change_abs = q.get('regularMarketChange', 0)
        name = (q.get('shortName') or q.get('longName') or symbol)[:30]

        results.append({
            "symbol": symbol,
            "name": name,
            "price": float(price),
            "prev_close": float(prev_close) if prev_close else float(price - change_abs),
            "change": float(change_pct),
            "change_abs": float(change_abs),
        })
    return results


def _fetch_via_yfinance_screen(screener_id: str) -> List[Dict]:
    """Use yfinance's built-in screen() function (handles auth/crumbs automatically)."""
    try:
        import yfinance as yf
        result = yf.screen(screener_id)
        if result and isinstance(result, dict):
            quotes = result.get('quotes', [])
            return _parse_screener_quotes(quotes)
    except Exception:
        pass
    return []


# Shared session for raw Yahoo Finance API fallback
import requests as _requests
_yahoo_session = None
_yahoo_crumb = None
_yahoo_session_ts = 0


def _get_yahoo_session():
    """Get or create a requests session with Yahoo Finance cookies and crumb."""
    global _yahoo_session, _yahoo_crumb, _yahoo_session_ts
    import time

    if _yahoo_session is None or (time.time() - _yahoo_session_ts) > 600:
        _yahoo_session = _requests.Session()
        _yahoo_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        })
        try:
            _yahoo_session.get('https://fc.yahoo.com', timeout=10, allow_redirects=True)
        except Exception:
            pass
        try:
            r = _yahoo_session.get('https://query2.finance.yahoo.com/v1/test/getcrumb', timeout=10)
            if r.status_code == 200 and r.text.strip():
                _yahoo_crumb = r.text.strip()
        except Exception:
            _yahoo_crumb = None
        _yahoo_session_ts = time.time()
    return _yahoo_session, _yahoo_crumb


def _fetch_via_raw_api(screener_id: str, count: int = 25) -> List[Dict]:
    """Fallback: fetch from Yahoo's raw screener API endpoint."""
    session, crumb = _get_yahoo_session()

    urls = [
        'https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved',
        'https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved',
    ]

    for url in urls:
        try:
            params = {'scrIds': screener_id, 'count': count}
            if crumb:
                params['crumb'] = crumb

            r = session.get(url, params=params, timeout=15)

            if r.status_code in (401, 403, 429):
                global _yahoo_session, _yahoo_crumb, _yahoo_session_ts
                _yahoo_session = None
                _yahoo_session_ts = 0
                session, crumb = _get_yahoo_session()
                params = {'scrIds': screener_id, 'count': count}
                if crumb:
                    params['crumb'] = crumb
                r = session.get(url, params=params, timeout=15)

            if r.status_code != 200:
                continue

            data = r.json()
            quotes = data.get('finance', {}).get('result', [{}])[0].get('quotes', [])
            results = _parse_screener_quotes(quotes)
            if results:
                return results

        except Exception:
            continue

    return []


def _fetch_yahoo_screener(screener_id: str, count: int = 25) -> List[Dict]:
    """
    Fetch market movers from Yahoo Finance matching the website exactly.
    Uses yfinance's built-in screen() first (best auth handling),
    falls back to raw API if that fails.
    """
    # Primary: yfinance built-in screen (handles cookies/crumb automatically)
    results = _fetch_via_yfinance_screen(screener_id)
    if results:
        return results[:count]

    # Fallback: raw HTTP API
    return _fetch_via_raw_api(screener_id, count)


def fetch_yahoo_gainers(max_items: int = 25) -> List[Dict]:
    """Fetch top gainers from Yahoo Finance (entire market)."""
    return _fetch_yahoo_screener('day_gainers', max_items)


def fetch_yahoo_losers(max_items: int = 25) -> List[Dict]:
    """Fetch top losers from Yahoo Finance (entire market)."""
    return _fetch_yahoo_screener('day_losers', max_items)


def fetch_yahoo_most_active(max_items: int = 25) -> List[Dict]:
    """Fetch most active stocks from Yahoo Finance (entire market)."""
    return _fetch_yahoo_screener('most_actives', max_items)


def fetch_market_movers_live() -> Tuple[List[Dict], List[Dict]]:
    """
    Fetch REAL market-wide top gainers and losers from Yahoo Finance screener API.
    Returns actual market movers across all listed stocks (>= $3/share), ranked
    by highest/lowest daily percentage change.
    Returns (top_gainers, top_losers).
    """
    top_gainers = fetch_yahoo_gainers(50)  # fetch extra to account for filtered-out penny stocks
    top_losers = fetch_yahoo_losers(50)

    # Ensure sorted by percentage change after $3 min-price filter
    top_gainers.sort(key=lambda x: x["change"], reverse=True)
    top_losers.sort(key=lambda x: x["change"], reverse=False)

    return top_gainers[:25], top_losers[:25]


# Fallback: pull a broad sample from ticker_universe.py (used only if Yahoo screener API is unavailable)
def _get_scan_universe() -> List[str]:
    """Get a broad, unbiased sample of tickers from the full ticker universe."""
    try:
        universe = get_ticker_universe()
        return universe.get_random_sample(200)
    except Exception:
        # Universe unavailable — return empty so the caller degrades gracefully
        # rather than presenting a fixed screening list as if it were the universe.
        return []


def _get_ticker_performance(symbol: str) -> Optional[Dict]:
    """Get daily performance for a single ticker with accurate live prices."""
    try:
        current_price, prev_close = _get_accurate_live_price(symbol)

        if current_price and current_price >= _MIN_MOVER_PRICE and prev_close and prev_close > 0:
            change = ((current_price - prev_close) / prev_close * 100)
            return {
                "symbol": symbol,
                "price": float(current_price),
                "prev_close": float(prev_close),
                "change": float(change),
                "change_abs": float(current_price - prev_close),
            }
    except Exception:
        pass
    return None


def _seed_movers_into_universe(movers: List[Dict]):
    """Add any newly discovered mover tickers into the ticker universe."""
    try:
        universe = get_ticker_universe()
        known = universe.get_known_ticker_set()
        for m in movers:
            sym = m.get("symbol", "").strip().upper()
            if sym and sym not in known:
                universe.add_ticker(sym)
    except Exception:
        pass


def fetch_market_movers(symbols: List[str] = None, max_workers: int = 10) -> Tuple[List[Dict], List[Dict]]:
    """
    Fetch top gainers and losers.
    Uses Yahoo Finance screener API for real market-wide data (matches yahoo.com).
    Falls back to scanning predefined list if API is unavailable.
    Returns (top_performers, worst_performers).
    """
    # Try to get real market-wide data from Yahoo Finance screener API
    try:
        top_gainers, top_losers = fetch_market_movers_live()
        if top_gainers and top_losers:
            # Add any new tickers to the universe for future scans
            _seed_movers_into_universe(top_gainers + top_losers)
            return top_gainers, top_losers
    except Exception:
        pass

    # Fallback to broad ticker universe scan
    if symbols is None:
        symbols = _get_scan_universe()

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_get_ticker_performance, s): s for s in symbols}
        done, _ = concurrent.futures.wait(futures, timeout=20)

        for f in done:
            try:
                r = f.result(timeout=2)
                if r:
                    results.append(r)
            except Exception:
                pass

    # Sort by change
    results.sort(key=lambda x: x["change"], reverse=True)

    top_performers = results[:10] if len(results) >= 10 else results
    worst_performers = list(reversed(results[-10:])) if len(results) >= 10 else list(reversed(results))

    return top_performers, worst_performers


def show_market_movers():
    """Display top/worst performers section."""
    section_header("Market Movers")

    @st.cache_data(ttl=120, show_spinner=False)
    def _cached_movers():
        return fetch_market_movers()

    top_performers, worst_performers = _cached_movers()

    if not top_performers and not worst_performers:
        st.warning("Unable to fetch market movers data. Market may be closed.")
        return

    st.caption(f"Market-wide data | Updated: {datetime.now().strftime('%H:%M:%S')} | Refreshes with page")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Top Gainers")
        if top_performers:
            for i, t in enumerate(top_performers[:10], 1):
                name = t.get("name", t["symbol"])
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:8px 12px;background:#1a1f2e;border-radius:6px;margin-bottom:4px;'
                    f'border-left:3px solid #3fb950;">'
                    f'<span style="color:white;font-weight:600;min-width:60px;">{i}. {t["symbol"]}</span>'
                    f'<span style="color:#8b949e;font-size:0.8rem;flex:1;margin:0 8px;'
                    f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{name}</span>'
                    f'<span style="color:white;min-width:80px;text-align:right;">${t["price"]:,.2f}</span>'
                    f'<span style="color:#3fb950;font-weight:600;min-width:75px;text-align:right;">'
                    f'{t["change"]:+.2f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    with col2:
        st.markdown("### Top Losers")
        if worst_performers:
            for i, t in enumerate(worst_performers[:10], 1):
                name = t.get("name", t["symbol"])
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:8px 12px;background:#1a1f2e;border-radius:6px;margin-bottom:4px;'
                    f'border-left:3px solid #f85149;">'
                    f'<span style="color:white;font-weight:600;min-width:60px;">{i}. {t["symbol"]}</span>'
                    f'<span style="color:#8b949e;font-size:0.8rem;flex:1;margin:0 8px;'
                    f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{name}</span>'
                    f'<span style="color:white;min-width:80px;text-align:right;">${t["price"]:,.2f}</span>'
                    f'<span style="color:#f85149;font-weight:600;min-width:75px;text-align:right;">'
                    f'{t["change"]:+.2f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )


# 
# COMPREHENSIVE SYMBOL SEARCH
# 

def _fetch_symbol_data(symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
    """Fetch historical data for a symbol."""
    try:
        sym = symbol.strip().upper()
        # Handle forex
        if '/' in sym and '=' not in sym:
            yf_sym = sym.replace('/', '') + '=X'
        else:
            yf_sym = sym
        
        df = get_stock(yf_sym, period=period)
        if df is not None and not df.empty:
            return df
        
        # Fallback
        import yfinance as yf
        df = yf.Ticker(yf_sym).history(period=period)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception:
        pass
    return None


def _safe_close(df):
    """Extract close series safely."""
    if df is None or df.empty or "Close" not in df.columns:
        return None
    c = df["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.dropna().astype(float)


def _calculate_technicals(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive technical indicators."""
    close = _safe_close(df)
    if close is None or len(close) < 20:
        return {}
    
    try:
        # SMAs
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean() if len(close) >= 50 else None
        sma200 = close.rolling(200).mean() if len(close) >= 200 else None
        
        # EMAs
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        
        # MACD
        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_signal
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        
        # ATR
        if "High" in df.columns and "Low" in df.columns:
            high = df["High"].iloc[:, 0] if isinstance(df["High"], pd.DataFrame) else df["High"]
            low = df["Low"].iloc[:, 0] if isinstance(df["Low"], pd.DataFrame) else df["Low"]
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
        else:
            atr = None
        
        # Stochastic
        if "High" in df.columns and "Low" in df.columns:
            high = df["High"].iloc[:, 0] if isinstance(df["High"], pd.DataFrame) else df["High"]
            low = df["Low"].iloc[:, 0] if isinstance(df["Low"], pd.DataFrame) else df["Low"]
            lowest_low = low.rolling(14).min()
            highest_high = high.rolling(14).max()
            stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
            stoch_d = stoch_k.rolling(3).mean()
        else:
            stoch_k, stoch_d = None, None
        
        # Volatility
        returns = close.pct_change().dropna()
        ann_vol = float(returns.std() * np.sqrt(252) * 100) if len(returns) > 10 else 0
        
        current = float(close.iloc[-1])
        
        return {
            "current_price": current,
            "sma20": float(sma20.iloc[-1]) if not sma20.empty else None,
            "sma50": float(sma50.iloc[-1]) if sma50 is not None and not sma50.empty else None,
            "sma200": float(sma200.iloc[-1]) if sma200 is not None and not sma200.empty else None,
            "ema12": float(ema12.iloc[-1]) if not ema12.empty else None,
            "ema26": float(ema26.iloc[-1]) if not ema26.empty else None,
            "macd_line": float(macd_line.iloc[-1]) if not macd_line.empty else None,
            "macd_signal": float(macd_signal.iloc[-1]) if not macd_signal.empty else None,
            "macd_hist": float(macd_hist.iloc[-1]) if not macd_hist.empty else None,
            "rsi": float(rsi.iloc[-1]) if not rsi.empty else None,
            "bb_upper": float(bb_upper.iloc[-1]) if not bb_upper.empty else None,
            "bb_mid": float(bb_mid.iloc[-1]) if not bb_mid.empty else None,
            "bb_lower": float(bb_lower.iloc[-1]) if not bb_lower.empty else None,
            "atr": float(atr.iloc[-1]) if atr is not None and not atr.empty else None,
            "stoch_k": float(stoch_k.iloc[-1]) if stoch_k is not None and not stoch_k.empty else None,
            "stoch_d": float(stoch_d.iloc[-1]) if stoch_d is not None and not stoch_d.empty else None,
            "volatility": ann_vol,
            "close_series": close,
            "sma20_series": sma20,
            "sma50_series": sma50,
            "bb_upper_series": bb_upper,
            "bb_lower_series": bb_lower,
            "rsi_series": rsi,
            "macd_line_series": macd_line,
            "macd_signal_series": macd_signal,
            "macd_hist_series": macd_hist,
        }
    except Exception as e:
        return {}


def _generate_ai_insights(symbol: str, technicals: Dict, df: pd.DataFrame, asset_type: str = "Stock") -> Dict:
    """Generate AI-powered insights with weighted multi-factor probability model."""
    if not technicals:
        return {"signal": "NEUTRAL", "insights": [], "trade_ideas": [],
                "outlook": "Insufficient data", "bullish_prob": 0.33,
                "bearish_prob": 0.33}

    insights = []
    trade_ideas = []

    current = technicals.get("current_price", 0)
    rsi = technicals.get("rsi")
    macd_hist = technicals.get("macd_hist")
    macd_line = technicals.get("macd_line")
    macd_signal = technicals.get("macd_signal")
    sma20 = technicals.get("sma20")
    sma50 = technicals.get("sma50")
    sma200 = technicals.get("sma200")
    bb_upper = technicals.get("bb_upper")
    bb_lower = technicals.get("bb_lower")
    bb_mid = technicals.get("bb_mid")
    stoch_k = technicals.get("stoch_k")
    stoch_d = technicals.get("stoch_d")
    volatility = technicals.get("volatility", 0)
    close_series = technicals.get("close_series")
    macd_hist_series = technicals.get("macd_hist_series")
    sma20_series = technicals.get("sma20_series")

    #  Weighted scoring: each factor contributes a continuous score 
    # Positive = bullish, negative = bearish, magnitude = conviction
    weighted_scores = []  # list of (score, weight) tuples

    # MULTI-ASSET INTEGRATION
    try:
        from multi_asset_analyzer import MultiAssetAnalyzer
        maa = MultiAssetAnalyzer()
        
        if asset_type.upper() == "FX":
            fx_data = maa.analyze_fx_pair(symbol)
            if 'carry_trade_analysis' in fx_data:
                carry = fx_data['carry_trade_analysis']
                if carry.get('recommendation') == 'FAVORABLE_CARRY_LONG':
                    weighted_scores.append((0.5, 1.5))
                    insights.append(f"Favorable carry trade ({carry.get('carry_differential', 0):.2f}% yield advantage) supporting longs")
                    trade_ideas.append("Carry trade setup: hold long positions to collect swap yield")
                elif carry.get('recommendation') == 'FAVORABLE_CARRY_SHORT':
                    weighted_scores.append((-0.5, 1.5))
                    insights.append(f"Unfavorable carry ({carry.get('carry_differential', 0):.2f}%) putting downward pressure")
                    trade_ideas.append("Carry trade setup: short positions benefit from positive swap")
            
            if 'central_bank_analysis' in fx_data:
                cb = fx_data['central_bank_analysis']
                if cb.get('divergence_interpretation') == 'BASE_FAVORED':
                    weighted_scores.append((0.4, 1.0))
                    insights.append("Central bank policy divergence favors base currency")
                elif cb.get('divergence_interpretation') == 'QUOTE_FAVORED':
                    weighted_scores.append((-0.4, 1.0))
                    insights.append("Central bank policy divergence favors quote currency")

        elif asset_type.upper() == "FUTURES":
            fut_data = maa.analyze_futures(symbol)
            if 'curve_analysis' in fut_data:
                curve = fut_data['curve_analysis']
                if curve.get('structure') == 'BACKWARDATION':
                    weighted_scores.append((0.4, 1.2))
                    insights.append(f"Curve in backwardation (slope {curve.get('curve_slope_percent',0):.2f}%) indicating tight near-term supply")
                    trade_ideas.append("Positive roll yield favors holding long positions")
                elif curve.get('structure') == 'CONTANGO':
                    weighted_scores.append((-0.3, 1.0))
                    insights.append(f"Curve in contango (slope {curve.get('curve_slope_percent',0):.2f}%) indicating oversupply or carrying costs")
                    trade_ideas.append("Negative roll yield creates headwind for long positions")
                    
            if 'seasonality' in fut_data:
                season = fut_data['seasonality']
                if season.get('seasonal_bias') == 'BULLISH':
                    weighted_scores.append((0.3, 1.0))
                    insights.append("Current month exhibits historically bullish seasonality")
                elif season.get('seasonal_bias') == 'BEARISH':
                    weighted_scores.append((-0.3, 1.0))
                    insights.append("Current month exhibits historically bearish seasonality")

        elif asset_type.upper() == "CRYPTO":
            crypto_data = maa.analyze_crypto(symbol)
            if 'volatility_regime' in crypto_data:
                regime = crypto_data['volatility_regime']
                vol = crypto_data.get('volatility_30d_annualized', 0)
                if regime == 'EXTREME':
                    insights.append(f"Extreme spot volatility regime ({vol:.0f}% annualized) — expect violent price action")
                    trade_ideas.append("Use significantly wider stops and reduce position sizing")
                elif regime == 'HIGH':
                    insights.append(f"High spot volatility regime ({vol:.0f}% annualized)")

    except Exception as e:
        print(f"Multi-asset context error: {e}")

    # --- RSI (weight 2.0) — most reliable mean-reversion signal ---
    if rsi is not None:
        if rsi > 80:
            weighted_scores.append((-0.9, 2.0))
            insights.append(f"RSI extremely overbought at {rsi:.1f} - high reversal probability")
            trade_ideas.append("Consider taking profits or waiting for RSI to cool below 70")
        elif rsi > 70:
            weighted_scores.append((-0.6, 2.0))
            insights.append(f"RSI overbought at {rsi:.1f} - potential pullback ahead")
        elif rsi < 20:
            weighted_scores.append((0.9, 2.0))
            insights.append(f"RSI extremely oversold at {rsi:.1f} - strong bounce opportunity")
            trade_ideas.append("Look for reversal confirmation for aggressive long entry")
        elif rsi < 30:
            weighted_scores.append((0.6, 2.0))
            insights.append(f"RSI oversold at {rsi:.1f} - potential bounce opportunity")
        elif rsi > 60:
            weighted_scores.append((0.3, 1.5))
            insights.append(f"RSI at {rsi:.1f} shows solid bullish momentum")
        elif rsi < 40:
            weighted_scores.append((-0.3, 1.5))
            insights.append(f"RSI at {rsi:.1f} shows bearish momentum")
        else:
            # 40-60 zone: weak/neutral signal
            bias = (rsi - 50) / 50  # -0.2 to +0.2
            weighted_scores.append((bias * 0.15, 0.5))
            insights.append(f"RSI at {rsi:.1f} — neutral zone, no strong momentum signal")

    # --- SMA Trend Alignment (weight varies by timeframe) ---
    sma_alignment = 0  # count aligned SMAs
    if sma20 and current:
        pct_above = (current - sma20) / sma20
        score = np.clip(pct_above * 8, -0.8, 0.8)
        weighted_scores.append((score, 1.5))
        if current > sma20:
            insights.append(f"Price {pct_above:.1%} above 20-day SMA — short-term uptrend")
            sma_alignment += 1
        else:
            insights.append(f"Price {abs(pct_above):.1%} below 20-day SMA — short-term downtrend")
            sma_alignment -= 1

    if sma50 and current:
        pct_above = (current - sma50) / sma50
        score = np.clip(pct_above * 5, -0.8, 0.8)
        weighted_scores.append((score, 2.0))
        if current > sma50:
            insights.append(f"Price above 50-day SMA (${sma50:.2f}) — medium-term uptrend")
            sma_alignment += 1
        else:
            insights.append(f"Price below 50-day SMA (${sma50:.2f}) — medium-term downtrend")
            sma_alignment -= 1

    if sma200 and current:
        pct_above = (current - sma200) / sma200
        score = np.clip(pct_above * 3, -0.8, 0.8)
        weighted_scores.append((score, 2.5))
        if current > sma200:
            insights.append(f"Price above 200-day SMA (${sma200:.2f}) — long-term uptrend")
            sma_alignment += 1
        else:
            insights.append(f"Price below 200-day SMA (${sma200:.2f}) — long-term downtrend")
            sma_alignment -= 1

    # Bonus for SMA alignment (multiple moving averages agree)
    if abs(sma_alignment) >= 2:
        bonus = 0.6 if sma_alignment > 0 else -0.6
        if abs(sma_alignment) == 3:
            bonus = 0.8 if sma_alignment > 0 else -0.8
        weighted_scores.append((bonus, 2.5))

    # Golden cross / death cross detection
    if sma20 and sma50 and sma20_series is not None:
        sma50_series = technicals.get("sma50_series")
        if sma50_series is not None and len(sma20_series) > 5 and len(sma50_series) > 5:
            try:
                prev_diff = float(sma20_series.iloc[-5]) - float(sma50_series.iloc[-5])
                curr_diff = float(sma20_series.iloc[-1]) - float(sma50_series.iloc[-1])
                if prev_diff < 0 and curr_diff > 0:
                    weighted_scores.append((0.7, 2.0))
                    insights.append("Golden Cross detected (SMA20 crossed above SMA50) — bullish")
                elif prev_diff > 0 and curr_diff < 0:
                    weighted_scores.append((-0.7, 2.0))
                    insights.append("Death Cross detected (SMA20 crossed below SMA50) — bearish")
            except Exception:
                pass

    # --- MACD (weight 1.8) — momentum and direction ---
    if macd_hist is not None:
        if macd_line is not None and macd_signal is not None:
            # Signal line crossover strength
            macd_spread = abs(macd_line - macd_signal)
            if current > 0:
                normalized_spread = macd_spread / current * 100  # as % of price
            else:
                normalized_spread = 0
            strength = min(normalized_spread * 2, 0.7)

            if macd_hist > 0:
                weighted_scores.append((strength, 1.8))
                insights.append("MACD histogram positive — bullish momentum")
            else:
                weighted_scores.append((-strength, 1.8))
                insights.append("MACD histogram negative — bearish momentum")

            # MACD momentum acceleration (histogram expanding or contracting)
            if macd_hist_series is not None and len(macd_hist_series) >= 5:
                try:
                    recent = macd_hist_series.iloc[-3:]
                    if all(recent.diff().dropna() > 0):
                        weighted_scores.append((0.3, 1.0))
                        insights.append("MACD histogram expanding — accelerating bullish momentum")
                    elif all(recent.diff().dropna() < 0):
                        weighted_scores.append((-0.3, 1.0))
                        insights.append("MACD histogram contracting — accelerating bearish momentum")
                except Exception:
                    pass
        else:
            if macd_hist > 0:
                weighted_scores.append((0.3, 1.5))
            else:
                weighted_scores.append((-0.3, 1.5))

    # --- Bollinger Bands (weight 1.5) — mean reversion ---
    if bb_upper and bb_lower and bb_mid and current:
        bb_width = bb_upper - bb_lower
        if bb_width > 0:
            bb_position = (current - bb_lower) / bb_width  # 0 = lower, 1 = upper
            if bb_position > 1.0:
                weighted_scores.append((-0.5, 1.5))
                insights.append("Price above upper Bollinger Band — potentially overextended")
                trade_ideas.append("Consider waiting for mean reversion before new longs")
            elif bb_position < 0.0:
                weighted_scores.append((0.5, 1.5))
                insights.append("Price below lower Bollinger Band — potentially oversold")
                trade_ideas.append("Watch for reversal patterns for potential long entry")
            elif bb_position > 0.8:
                weighted_scores.append((-0.2, 1.0))
            elif bb_position < 0.2:
                weighted_scores.append((0.2, 1.0))

    # --- Stochastic (weight 1.2) ---
    if stoch_k is not None:
        if stoch_k > 80:
            # Overbought, but check if in strong uptrend (stoch can stay high)
            penalty = -0.4 if (sma_alignment <= 0) else -0.15
            weighted_scores.append((penalty, 1.2))
            insights.append(f"Stochastic overbought at {stoch_k:.1f}")
        elif stoch_k < 20:
            bonus = 0.4 if (sma_alignment >= 0) else 0.15
            weighted_scores.append((bonus, 1.2))
            insights.append(f"Stochastic oversold at {stoch_k:.1f}")
        elif stoch_d is not None:
            if stoch_k > stoch_d:
                weighted_scores.append((0.1, 0.8))
            else:
                weighted_scores.append((-0.1, 0.8))

    # --- Price momentum (weight 1.5) — recent returns ---
    if close_series is not None and len(close_series) >= 10:
        try:
            ret_5d = (float(close_series.iloc[-1]) / float(close_series.iloc[-5]) - 1) if len(close_series) >= 5 else 0
            ret_10d = (float(close_series.iloc[-1]) / float(close_series.iloc[-10]) - 1) if len(close_series) >= 10 else 0
            # Scale returns to score (-1 to +1)
            mom_score = np.clip(ret_5d * 10, -0.6, 0.6)
            weighted_scores.append((mom_score, 1.5))
            if abs(ret_5d) > 0.03:
                direction = "up" if ret_5d > 0 else "down"
                insights.append(f"5-day momentum: {ret_5d:+.1%} {direction}")
        except Exception:
            pass

    # --- Volume analysis (weight 1.0) ---
    if df is not None and "Volume" in df.columns:
        try:
            vol = df["Volume"]
            if isinstance(vol, pd.DataFrame):
                vol = vol.iloc[:, 0]
            vol = vol.dropna().astype(float)
            if len(vol) >= 20 and vol.mean() > 0:
                vol_ratio = float(vol.iloc[-1]) / float(vol.rolling(20).mean().iloc[-1])
                if vol_ratio > 2.0:
                    # High volume confirms the current move
                    if close_series is not None and len(close_series) >= 2:
                        last_ret = float(close_series.iloc[-1]) - float(close_series.iloc[-2])
                        if last_ret > 0:
                            weighted_scores.append((0.3, 1.0))
                            insights.append(f"Volume surge ({vol_ratio:.1f}x avg) confirming buying pressure")
                        else:
                            weighted_scores.append((-0.3, 1.0))
                            insights.append(f"Volume surge ({vol_ratio:.1f}x avg) confirming selling pressure")
                elif vol_ratio < 0.5:
                    insights.append("Below-average volume — low conviction in current move")
        except Exception:
            pass

    # --- Volatility context (no directional bias, affects confidence) ---
    if volatility:
        if volatility > 40:
            insights.append(f"High volatility ({volatility:.1f}%) — use wider stops and smaller positions")
        elif volatility < 15:
            insights.append(f"Low volatility ({volatility:.1f}%) — potential for breakout, watch for range expansion")

    #  Calculate final probabilities using weighted average 
    if weighted_scores:
        total_weight = sum(abs(w) for _, w in weighted_scores)
        if total_weight > 0:
            # Weighted average score: range roughly -1 to +1
            raw_score = sum(s * w for s, w in weighted_scores) / total_weight
        else:
            raw_score = 0.0
    else:
        raw_score = 0.0

    # Convert raw_score to probabilities using a soft allocation model
    # raw_score range: roughly -1.0 (extreme bearish) to +1.0 (extreme bullish)
    # We allocate a neutral band that shrinks as conviction grows
    conviction = min(abs(raw_score) * 1.5, 1.0)  # 0 to 1
    neutral_prob = max(0.05, 0.40 * (1 - conviction))  # 5% to 40%

    remaining = 1.0 - neutral_prob
    if raw_score >= 0:
        bullish_prob = remaining * (0.5 + raw_score * 0.5)
        bearish_prob = remaining - bullish_prob
    else:
        bearish_prob = remaining * (0.5 + abs(raw_score) * 0.5)
        bullish_prob = remaining - bearish_prob

    # Clamp to sensible bounds
    bullish_prob = max(0.03, min(0.92, bullish_prob))
    bearish_prob = max(0.03, min(0.92, bearish_prob))
    neutral_prob = max(0.03, 1.0 - bullish_prob - bearish_prob)
    # Renormalize
    total_p = bullish_prob + bearish_prob + neutral_prob
    bullish_prob /= total_p
    bearish_prob /= total_p

    #  Determine signal label 
    if bullish_prob > 0.60:
        signal = "STRONG_BULLISH" if bullish_prob > 0.75 else "BULLISH"
        outlook = ("Strong bullish bias — consider long positions with appropriate risk management"
                   if "STRONG" in signal else
                   "Moderate bullish bias — look for pullbacks to enter long positions")
    elif bearish_prob > 0.60:
        signal = "STRONG_BEARISH" if bearish_prob > 0.75 else "BEARISH"
        outlook = ("Strong bearish bias — consider short positions or staying out"
                   if "STRONG" in signal else
                   "Moderate bearish bias — exercise caution with long positions")
    else:
        signal = "NEUTRAL"
        outlook = "Mixed signals — wait for clearer direction before taking positions"

    #  Generate trade ideas based on signal 
    if signal in ["STRONG_BULLISH", "BULLISH"]:
        if sma20:
            trade_ideas.append(f"Entry: Consider buying near SMA20 support at ${sma20:.2f}")
        if bb_lower:
            trade_ideas.append(f"Entry: Buy on dips to lower BB at ${bb_lower:.2f}")
        trade_ideas.append("Stop Loss: Place below recent swing low or 1-2 ATR below entry")
        trade_ideas.append("Take Profit: Target recent highs or 2:1 reward-to-risk ratio")
    elif signal in ["STRONG_BEARISH", "BEARISH"]:
        if sma20:
            trade_ideas.append(f"Entry: Consider shorting near SMA20 resistance at ${sma20:.2f}")
        if bb_upper:
            trade_ideas.append(f"Entry: Short on rallies to upper BB at ${bb_upper:.2f}")
        trade_ideas.append("Stop Loss: Place above recent swing high or 1-2 ATR above entry")
        trade_ideas.append("Take Profit: Target recent lows or 2:1 reward-to-risk ratio")

    return {
        "signal": signal,
        "bullish_prob": bullish_prob,
        "bearish_prob": bearish_prob,
        "insights": insights[:6],
        "trade_ideas": trade_ideas[:4],
        "outlook": outlook,
    }


def _create_comprehensive_chart(symbol: str, df: pd.DataFrame, technicals: Dict, 
                                 selected_indicators: List[str]) -> go.Figure:
    """Create a comprehensive chart with selected indicators."""
    close = technicals.get("close_series")
    if close is None:
        return None
    
    # Determine subplots based on selected indicators
    has_volume = "Volume" in selected_indicators and "Volume" in df.columns
    has_rsi = "RSI" in selected_indicators
    has_macd = "MACD" in selected_indicators
    
    panel_count = sum([has_volume, has_rsi, has_macd])
    total_rows = 1 + panel_count
    
    row_heights = [0.5]
    if panel_count > 0:
        panel_height = 0.5 / panel_count
        row_heights.extend([panel_height] * panel_count)
    
    fig = make_subplots(
        rows=total_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.03,
    )
    
    # Price chart
    if "Open" in df.columns and "High" in df.columns and "Low" in df.columns:
        open_s = df["Open"].iloc[:, 0] if isinstance(df["Open"], pd.DataFrame) else df["Open"]
        high_s = df["High"].iloc[:, 0] if isinstance(df["High"], pd.DataFrame) else df["High"]
        low_s = df["Low"].iloc[:, 0] if isinstance(df["Low"], pd.DataFrame) else df["Low"]
        
        fig.add_trace(go.Candlestick(
            x=df.index, open=open_s, high=high_s, low=low_s, close=close,
            name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=close.index, y=close.values, mode="lines", name="Price",
            line=dict(color="#26a69a", width=2),
        ), row=1, col=1)
    
    # Overlay indicators
    if "SMA 20" in selected_indicators:
        sma20 = technicals.get("sma20_series")
        if sma20 is not None:
            fig.add_trace(go.Scatter(
                x=sma20.index, y=sma20, mode="lines", name="SMA 20",
                line=dict(color="#42a5f5", width=1.5),
            ), row=1, col=1)
    
    if "SMA 50" in selected_indicators:
        sma50 = technicals.get("sma50_series")
        if sma50 is not None:
            fig.add_trace(go.Scatter(
                x=sma50.index, y=sma50, mode="lines", name="SMA 50",
                line=dict(color="#ef5350", width=1.5),
            ), row=1, col=1)
    
    if "Bollinger Bands" in selected_indicators:
        bb_upper = technicals.get("bb_upper_series")
        bb_lower = technicals.get("bb_lower_series")
        if bb_upper is not None and bb_lower is not None:
            fig.add_trace(go.Scatter(
                x=bb_upper.index, y=bb_upper, mode="lines", name="BB Upper",
                line=dict(color="#78909c", width=1, dash="dot"),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=bb_lower.index, y=bb_lower, mode="lines", name="BB Lower",
                fill="tonexty", fillcolor="rgba(120,144,156,0.1)",
                line=dict(color="#78909c", width=1, dash="dot"),
            ), row=1, col=1)
    
    # Panel indicators
    current_row = 2
    
    if has_volume:
        vol = df["Volume"]
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]
        fig.add_trace(go.Bar(
            x=vol.index, y=vol, name="Volume",
            marker_color="rgba(100,100,255,0.3)",
        ), row=current_row, col=1)
        fig.update_yaxes(title_text="Volume", row=current_row, col=1)
        current_row += 1
    
    if has_rsi:
        rsi = technicals.get("rsi_series")
        if rsi is not None:
            fig.add_trace(go.Scatter(
                x=rsi.index, y=rsi, mode="lines", name="RSI",
                line=dict(color="#ce93d8", width=1.5),
            ), row=current_row, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", line_width=0.8, row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", line_width=0.8, row=current_row, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)
        current_row += 1
    
    if has_macd:
        macd_line = technicals.get("macd_line_series")
        macd_signal = technicals.get("macd_signal_series")
        macd_hist = technicals.get("macd_hist_series")
        if macd_line is not None:
            fig.add_trace(go.Scatter(
                x=macd_line.index, y=macd_line, mode="lines", name="MACD",
                line=dict(color="#42a5f5", width=1.2),
            ), row=current_row, col=1)
            if macd_signal is not None:
                fig.add_trace(go.Scatter(
                    x=macd_signal.index, y=macd_signal, mode="lines", name="Signal",
                    line=dict(color="#ef5350", width=1.2),
                ), row=current_row, col=1)
            if macd_hist is not None:
                colors = ["#26a69a" if v >= 0 else "#ef5350" for v in macd_hist.values]
                fig.add_trace(go.Bar(
                    x=macd_hist.index, y=macd_hist, name="Histogram",
                    marker_color=colors, opacity=0.6,
                ), row=current_row, col=1)
            fig.update_yaxes(title_text="MACD", row=current_row, col=1)
    
    # Add live price line
    live_price = _get_live_price(symbol)
    if live_price > 0:
        fig.add_hline(
            y=live_price, line_dash="dash", line_color="#00ff88", line_width=1.5,
            annotation_text=f"LIVE: ${live_price:.2f}",
            annotation_position="right",
            row=1, col=1
        )
    
    # Layout
    chart_height = 400 + panel_count * 150
    fig.update_layout(
        height=chart_height,
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
        title=dict(text=f"{symbol} - Comprehensive Analysis", x=0.01, font=dict(size=14, color="white")),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    
    return fig


def show_symbol_search():
    """Display comprehensive symbol search with live chart and AI insights."""
    section_header("Symbol Search & Analysis")
    
    # Search input
    col_search, col_period = st.columns([3, 1])
    with col_search:
        search_symbol = st.text_input(
            "Search Symbol",
            placeholder="Enter symbol (e.g., AAPL, EUR/USD, BTC-USD, ES=F)",
            key="market_search_symbol"
        ).strip().upper()
    with col_period:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=2,
                              key="market_search_period",
                              format_func=lambda x: {"1mo": "1 Month", "3mo": "3 Months", "6mo": "6 Months",
                                                     "1y": "1 Year", "2y": "2 Years", "5y": "5 Years",
                                                     "max": "All Time"}.get(x, x))
    
    if not search_symbol:
        st.info("Enter a symbol above to see comprehensive analysis with live data, AI insights, and trade ideas.")
        return
    
    # Indicator selection
    with st.expander("Customize Indicators", expanded=False):
        available_indicators = ["SMA 20", "SMA 50", "Bollinger Bands", "Volume", "RSI", "MACD"]
        default_indicators = ["SMA 20", "SMA 50", "Volume", "RSI"]
        selected_indicators = st.multiselect(
            "Select Indicators",
            available_indicators,
            default=default_indicators,
            key="market_search_indicators"
        )
    
    # Fetch and analyze
    with st.spinner(f"Analyzing {search_symbol}..."):
        df = _fetch_symbol_data(search_symbol, period)
        live_price = _get_live_price(search_symbol)
    
    if df is None or df.empty:
        st.error(f"Could not fetch data for {search_symbol}. Check the symbol format.")
        st.caption("Supported formats: AAPL (stocks), EUR/USD (forex), BTC-USD (crypto), ES=F (futures)")
        return
    
    # Calculate technicals and AI insights
    technicals = _calculate_technicals(df)
    ai_insights = _generate_ai_insights(search_symbol, technicals, df)
    
    # Live price header
    @st.fragment(run_every=30)
    def _live_header():
        fresh_price, prev = _get_accurate_live_price(search_symbol)
        if not fresh_price or fresh_price <= 0:
            fresh_price = _get_live_price(search_symbol)
        if not prev or prev <= 0:
            close = _safe_close(df)
            prev = float(close.iloc[-2]) if close is not None and len(close) >= 2 else fresh_price
        change = ((fresh_price - prev) / prev * 100) if prev and prev > 0 and fresh_price else 0
        
        signal = ai_insights.get("signal", "NEUTRAL")
        signal_color = "#00ff88" if "BULLISH" in signal else "#ff5252" if "BEARISH" in signal else "#8b949e"
        
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1a1f2e,#0d1117);padding:20px;border-radius:12px;'
            f'border-left:4px solid {signal_color};margin-bottom:16px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div>'
            f'<h2 style="color:white;margin:0;font-size:1.8rem;">{search_symbol}</h2>'
            f'<span style="color:#8b949e;">Live Price</span>'
            f'</div>'
            f'<div style="text-align:right;">'
            f'<span style="color:white;font-size:2rem;font-weight:700;">${fresh_price:,.2f}</span>'
            f'<br><span style="color:{"#00ff88" if change >= 0 else "#ff5252"};font-size:1.2rem;">'
            f'{change:+.2f}%</span>'
            f'</div>'
            f'</div>'
            f'<div style="margin-top:12px;padding-top:12px;border-top:1px solid #30363d;">'
            f'<span style="color:{signal_color};font-weight:600;font-size:1.1rem;">Signal: {signal}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
    
    _live_header()
    
    # Chart
    fig = _create_comprehensive_chart(search_symbol, df, technicals, selected_indicators)
    if fig:
        st.plotly_chart(fig, width='stretch', key=f"search_chart_{search_symbol}_{period}")
    
    # AI Insights and Trade Ideas
    col_insights, col_trades = st.columns(2)
    
    with col_insights:
        st.markdown("### AI Insights")
        outlook = ai_insights.get("outlook", "")
        if outlook:
            st.info(outlook)
        
        insights = ai_insights.get("insights", [])
        for insight in insights:
            st.markdown(f"- {insight}")
    
    with col_trades:
        st.markdown("### Trade Ideas")
        trade_ideas = ai_insights.get("trade_ideas", [])
        if trade_ideas:
            for idea in trade_ideas:
                st.markdown(f"- {idea}")
        else:
            st.caption("No specific trade ideas at this time.")
    
    # Technical Summary
    st.markdown("### Technical Summary")
    tech_cols = st.columns(5)
    
    with tech_cols[0]:
        rsi = technicals.get("rsi")
        rsi_status = "Overbought" if rsi and rsi > 70 else "Oversold" if rsi and rsi < 30 else "Neutral"
        st.metric("RSI (14)", f"{rsi:.1f}" if rsi else "N/A", rsi_status)
    
    with tech_cols[1]:
        macd = technicals.get("macd_hist")
        macd_status = "Bullish" if macd and macd > 0 else "Bearish" if macd and macd < 0 else "Neutral"
        st.metric("MACD Hist", f"{macd:.4f}" if macd else "N/A", macd_status)
    
    with tech_cols[2]:
        sma20 = technicals.get("sma20")
        st.metric("SMA 20", f"${sma20:.2f}" if sma20 else "N/A")
    
    with tech_cols[3]:
        sma50 = technicals.get("sma50")
        st.metric("SMA 50", f"${sma50:.2f}" if sma50 else "N/A")
    
    with tech_cols[4]:
        vol = technicals.get("volatility")
        vol_status = "High" if vol and vol > 30 else "Low" if vol and vol < 15 else "Normal"
        st.metric("Volatility", f"{vol:.1f}%" if vol else "N/A", vol_status)
    
    # Probability breakdown
    st.markdown("### Signal Probability")
    bullish_prob = ai_insights.get("bullish_prob", 0.5)
    bearish_prob = ai_insights.get("bearish_prob", 0.5)
    neutral_prob = max(0, 1 - bullish_prob - bearish_prob)
    
    prob_cols = st.columns(3)
    with prob_cols[0]:
        st.markdown(
            f'<div style="background:#1a3d1a;padding:16px;border-radius:8px;text-align:center;">'
            f'<span style="color:#00ff88;font-size:1.5rem;font-weight:700;">{bullish_prob:.0%}</span>'
            f'<br><span style="color:#8b949e;">Bullish</span></div>',
            unsafe_allow_html=True
        )
    with prob_cols[1]:
        st.markdown(
            f'<div style="background:#2d2d2d;padding:16px;border-radius:8px;text-align:center;">'
            f'<span style="color:#8b949e;font-size:1.5rem;font-weight:700;">{neutral_prob:.0%}</span>'
            f'<br><span style="color:#8b949e;">Neutral</span></div>',
            unsafe_allow_html=True
        )
    with prob_cols[2]:
        st.markdown(
            f'<div style="background:#3d1a1a;padding:16px;border-radius:8px;text-align:center;">'
            f'<span style="color:#ff5252;font-size:1.5rem;font-weight:700;">{bearish_prob:.0%}</span>'
            f'<br><span style="color:#8b949e;">Bearish</span></div>',
            unsafe_allow_html=True
        )
