"""
Octavian Comprehensive Market Scanner
Multi-asset real-time scanning with heatmaps, signals, and momentum analysis.
Author: APB - Octavian Team
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from plotly.subplots import make_subplots

from data_sources import get_futures_proxy, get_fx, get_stock
from sector_scanner import scan_sectors

try:
    from data_sources import get_latest_price

    HAS_LATEST_PRICE = True
except ImportError:
    HAS_LATEST_PRICE = False

try:
    from data_sources import get_fresh_quote

    HAS_FRESH_QUOTE = True
except ImportError:
    HAS_FRESH_QUOTE = False


#
# ASSET UNIVERSES
#

def _get_watchlist_groups() -> Dict[str, List[str]]:
    """Stock scanner groups built dynamically from the ticker universe's sectors.

    Every group is populated live from the current universe (a deterministic
    spread of each sector's constituents) — no hardcoded ticker lists.
    """
    groups: Dict[str, List[str]] = {}
    try:
        from ticker_universe import get_ticker_universe
        u = get_ticker_universe()
        sectors = u.get_all_sectors()
        for sector, tickers in sorted(sectors.items()):
            chosen = sorted(tickers)[:8]
            if chosen:
                groups[sector.replace("_", " ").title()] = chosen
        if not groups:
            groups["Stocks"] = sorted(u.get_all_stocks())[:8]
    except Exception:
        pass
    return groups


def _get_crypto_list() -> List[str]:
    """Crypto scan list drawn live from the ticker universe."""
    try:
        from ticker_universe import get_ticker_universe
        return list(get_ticker_universe().get_crypto())
    except Exception:
        return []

_ETF_SECTORS = {
    "S&P 500": "SPY",
    "Nasdaq 100": "QQQ",
    "Russell 2000": "IWM",
    "Dow 30": "DIA",
    "Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Consumer Disc.": "XLY",
    "Consumer Stap.": "XLP",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Comms": "XLC",
}

_COMMODITIES = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil": "CL=F",
    "Natural Gas": "NG=F",
    "Copper": "HG=F",
    "Platinum": "PL=F",
    "Corn": "ZC=F",
    "Soybeans": "ZS=F",
    "Wheat": "ZW=F",
    "Coffee": "KC=F",
}

_BONDS = {
    "20Y Treasury": "TLT",
    "7-10Y Treasury": "IEF",
    "1-3Y Treasury": "SHY",
    "IG Corporate": "LQD",
    "High Yield": "HYG",
    "TIPS": "TIP",
    "EM Bonds": "EMB",
}

_FOREX_PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "USD/CHF": "USDCHF=X",
}

_FUTURES_INDEX = {
    "E-mini S&P": "ES=F",
    "E-mini Nasdaq": "NQ=F",
    "E-mini Dow": "YM=F",
    "Russell 2000": "RTY=F",
}


#
# OVERVIEW: Use actual indices/spot, not ETF proxies
#

_OVERVIEW_ITEMS = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "Bitcoin": "BTC-USD",
    "VIX": "^VIX",
}


#
# DATA FETCHING
#


def _safe_close(df):
    """Extract Close as a Series from potentially multi-level columns."""
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten multi-level columns
        df.columns = df.columns.get_level_values(0)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
    c = df["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.dropna().astype(float)


def _fetch_df_for_symbol(symbol: str, period: str = "3mo") -> Optional[pd.DataFrame]:
    """Fetch clean dataframe for a single symbol with multiple fallbacks.

    For FX pairs the provider order is:
      1. get_fx()   Stooq-backed, free, no API key required (most reliable)
      2. get_fresh_quote()  Yahoo Finance via yfinance
      3. yf.Ticker().history()  direct Yahoo Finance
    For all other asset classes:
      1. get_fresh_quote()  Yahoo Finance
      2. yf.Ticker().history()  direct Yahoo Finance
      3. get_stock() / get_futures_proxy()  data_sources fallback chain
    """

    # Detect FX: slash-separated "EUR/USD", Yahoo-format "EURUSD=X", bare 6-char "EURUSD"
    sym_upper = symbol.strip().upper()
    clean6 = sym_upper.replace("/", "").replace("_", "").replace("=X", "")
    is_fx = (
        ("/" in sym_upper and len(clean6) == 6 and clean6.isalpha())
        or (sym_upper.endswith("=X") and len(sym_upper) == 8)
        or (len(clean6) == 6 and clean6.isalpha() and not sym_upper.startswith("^"))
    )

    # Normalize to Yahoo format for non-Stooq providers
    yf_sym = sym_upper
    if "/" in sym_upper and "=" not in sym_upper:
        yf_sym = sym_upper.replace("/", "") + "=X"

    #  FX: try Stooq-backed get_fx FIRST 
    if is_fx:
        try:
            # get_fx accepts slash, bare-6, or =X formats
            fx_arg = (
                sym_upper
                if "=" in sym_upper
                else (sym_upper if "/" in sym_upper else clean6[:3] + "/" + clean6[3:])
            )
            df = get_fx(fx_arg, period=period)
            if (
                df is not None
                and not df.empty
                and "Close" in df.columns
                and len(df) >= 5
            ):
                return df
        except Exception:
            pass

    #  Yahoo Finance via get_fresh_quote 
    if HAS_FRESH_QUOTE:
        try:
            df = get_fresh_quote(yf_sym, period=period)
            if df is not None and not df.empty and "Close" in df.columns:
                return df
        except Exception:
            pass

    #  Direct yfinance Ticker 
    try:
        import yfinance as yf

        tk = yf.Ticker(yf_sym)
        df = tk.history(period=period)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                if df.columns.duplicated().any():
                    df = df.loc[:, ~df.columns.duplicated(keep="first")]
            if "Close" in df.columns and len(df) >= 5:
                return df
    except Exception:
        pass

    #  data_sources fallback chain 
    try:
        if is_fx:
            # Already tried get_fx above; try get_stock which also has Stooq fallback
            df = get_stock(yf_sym, period=period)
            if df is not None and not df.empty:
                return df
        elif "=F" in sym_upper:
            df = get_futures_proxy(sym_upper, period=period)
            if df is not None and not df.empty:
                return df
        else:
            df = get_stock(sym_upper, period=period)
            if df is not None and not df.empty:
                return df
    except Exception:
        pass

    return None


def _has_valid_volume(df) -> bool:
    """Check if volume data is meaningful (not all zeros/NaN)."""
    if "Volume" not in df.columns:
        return False
    v = df["Volume"]
    if isinstance(v, pd.DataFrame):
        v = v.iloc[:, 0]
    v = v.dropna()
    return len(v) > 0 and v.sum() > 0 and v.max() > 1


def _fetch_and_score(
    symbol: str, label: str = None, period: str = "3mo"
) -> Optional[Dict]:
    """Fetch price data and compute a comprehensive score card for one symbol."""
    try:
        df = _fetch_df_for_symbol(symbol, period)

        if df is None or df.empty:
            return None

        close = _safe_close(df)
        if close is None or len(close) < 5:
            return None

        price = float(close.iloc[-1])
        if price <= 0:
            return None

        # Try to get freshest price
        if HAS_LATEST_PRICE:
            fresh_sym = symbol
            if "/" in symbol and "=" not in symbol:
                fresh_sym = symbol.replace("/", "") + "=X"
            try:
                fresh = get_latest_price(fresh_sym)
                if fresh is not None and fresh > 0:
                    # Sanity: don't use if wildly different from historical
                    ratio = fresh / price if price > 0 else 999
                    if 0.5 < ratio < 2.0:
                        price = fresh
            except Exception:
                pass

        # --- Get timestamp of most recent bar ---
        last_bar_date = ""
        try:
            last_bar_date = (
                close.index[-1].strftime("%Y-%m-%d")
                if hasattr(close.index[-1], "strftime")
                else str(close.index[-1])[:10]
            )
        except Exception:
            pass

        # Previous close for accurate 1D return
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else price
        ret_1d = ((price / prev_close) - 1) * 100 if prev_close > 0 else 0.0

        def _ret(n):
            idx = min(n, len(close) - 1)
            if idx > 0 and float(close.iloc[-idx - 1]) > 0:
                return (price / float(close.iloc[-idx - 1]) - 1) * 100
            return 0.0

        ret_5d = _ret(5)
        ret_1m = _ret(21) if len(close) > 21 else _ret(len(close) - 1)
        ret_3m = _ret(63) if len(close) > 63 else _ret(len(close) - 1)

        # RSI 14
        rsi = 50.0
        if len(close) >= 15:
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            rsi_series = 100 - (100 / (1 + rs))
            rsi_val = rsi_series.iloc[-1]
            if pd.notna(rsi_val):
                rsi = float(np.clip(rsi_val, 0, 100))

        daily_rets = close.pct_change().dropna()
        vol = (
            float(daily_rets.std() * np.sqrt(252) * 100) if len(daily_rets) > 5 else 0.0
        )

        above_sma20 = None
        above_sma50 = None
        if len(close) >= 20:
            sma20_val = close.rolling(20).mean().iloc[-1]
            if pd.notna(sma20_val):
                above_sma20 = price > float(sma20_val)
        if len(close) >= 50:
            sma50_val = close.rolling(50).mean().iloc[-1]
            if pd.notna(sma50_val):
                above_sma50 = price > float(sma50_val)

        macd_rising = False
        if len(close) >= 27:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(
                span=9, adjust=False
            ).mean()
            if (
                len(macd_hist) >= 2
                and pd.notna(macd_hist.iloc[-1])
                and pd.notna(macd_hist.iloc[-2])
            ):
                macd_rising = float(macd_hist.iloc[-1]) > float(macd_hist.iloc[-2])

        score = 0.0
        score += np.clip(ret_5d * 5, -25, 25)
        score += np.clip(ret_1m * 2, -25, 25)
        score += (rsi - 50) * 0.5
        if above_sma20 is True:
            score += 10
        elif above_sma20 is False:
            score -= 10
        if macd_rising:
            score += 15
        else:
            score -= 15
        score = float(np.clip(score, -100, 100))

        if score > 40:
            signal = "[GOOD] Strong Buy"
        elif score > 15:
            signal = "[NOTE] Buy"
        elif score > -15:
            signal = " Neutral"
        elif score > -40:
            signal = "[SELL] Sell"
        else:
            signal = "[ALERT] Strong Sell"

        vol_spike = False
        if _has_valid_volume(df):
            v = df["Volume"]
            if isinstance(v, pd.DataFrame):
                v = v.iloc[:, 0]
            v = v.dropna().astype(float)
            if len(v) >= 20:
                avg_v = v.rolling(20).mean().iloc[-1]
                if pd.notna(avg_v) and avg_v > 0 and v.iloc[-1] > avg_v * 1.5:
                    vol_spike = True

        # --- Format price display based on magnitude ---
        # Large indices (S&P, Nasdaq, Dow, Russell) should show as integers
        # Small prices show more decimals
        price_fmt = price

        return {
            "symbol": symbol,
            "label": label or symbol,
            "price": price_fmt,
            "1D%": round(ret_1d, 2),
            "5D%": round(ret_5d, 2),
            "1M%": round(ret_1m, 2),
            "3M%": round(ret_3m, 2),
            "RSI": round(rsi, 1),
            "Vol%": round(vol, 1),
            "Score": round(score, 1),
            "Signal": signal,
            "Above SMA20": above_sma20,
            "Above SMA50": above_sma50,
            "MACD Rising": macd_rising,
            "Vol Spike": vol_spike,
            "last_bar": last_bar_date,
            "price_source": "live" if HAS_LATEST_PRICE else "daily",
        }
    except Exception:
        return None


async def _scan_universe_async(items: Dict[str, str], period: str = "3mo") -> pd.DataFrame:
    """High-velocity scan using OHVDE backbone."""
    from octavian_discovery_engine import get_discovery_engine
    discovery = get_discovery_engine()
    
    symbols = list(items.values())
    labels = list(items.keys())
    
    # Run market pulse scan (Statistical Tier)
    pulse_results = await discovery.scan_market_pulse(symbols, deep_scan=False)
    
    # Map results back to scanner format
    results = []
    label_map = {v: k for k, v in items.items()}
    
    for r in pulse_results:
        sym = r['symbol']
        label = label_map.get(sym, sym)
        
        # Scoring logic for scanner (matching original behavior)
        score = r['score']
        signal = "[GOOD] Strong Buy" if score > 40 else "[NOTE] Buy" if score > 15 else " Neutral" if score > -15 else "[SELL] Sell" if score > -40 else "[ALERT] Strong Sell"
        
        results.append({
            "symbol": sym,
            "label": label,
            "price": r['price'],
            "1D%": round(r['change_1d'], 2),
            "5D%": round(r['indicators'].get('momentum_10d', 0) / 2, 2), # Approximation
            "1M%": round(r['indicators'].get('momentum_10d', 0), 2),
            "RSI": round(r['indicators'].get('rsi', 50), 1),
            "Vol%": round(r['indicators'].get('volatility_20d', 0.0), 1),
            "Score": round(score, 1),
            "Signal": signal,
            "Above SMA20": r['indicators'].get('above_sma50'),
            "last_bar": datetime.now().strftime("%Y-%m-%d"),
            "price_source": "DiscoveryEngine"
        })
    
    if not results: return pd.DataFrame()
    df = pd.DataFrame(results)
    df.sort_values("Score", ascending=False, inplace=True)
    return df.reset_index(drop=True)

def _scan_universe(items: Dict[str, str], period: str = "3mo", max_workers: int = 8) -> pd.DataFrame:
    """Synchronous wrapper for OHVDE scan."""
    return asyncio.run(_scan_universe_async(items, period))


def _scan_list(
    symbols: List[str], period: str = "3mo", max_workers: int = 8
) -> pd.DataFrame:
    """Scan a list of symbols."""
    return _scan_universe(
        {s: s for s in symbols}, period=period, max_workers=max_workers
    )


#
# RENDERING HELPERS
#


def _fmt_ret(val):
    """Format a return value with color emoji prefix for plain dataframe display."""
    if pd.isna(val):
        return ""
    return f"{val:+.2f}%"


def _render_heatmap(
    df: pd.DataFrame, title: str, value_col: str = "1D%", label_col: str = "label"
):
    """Render a treemap-style heatmap from scanner results."""
    if df.empty:
        st.info("No data available.")
        return
    plot_df = df[[label_col, value_col, "Score"]].dropna().copy()
    if plot_df.empty:
        st.info("No data for heatmap.")
        return
    plot_df["abs_score"] = plot_df["Score"].abs() + 10  # size weighting
    plot_df["display_val"] = plot_df[value_col].astype(float)
    fig = go.Figure(
        go.Treemap(
            labels=plot_df[label_col],
            parents=[""] * len(plot_df),
            values=plot_df["abs_score"],
            marker=dict(
                colors=plot_df["display_val"],
                colorscale="RdYlGn",
                cmid=0,
                colorbar=dict(title=value_col),
            ),
            customdata=plot_df[["display_val"]].values,
            texttemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}%",
            hovertemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=400,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
        title=title,
    )
    st.plotly_chart(fig, width='stretch')


def _fmt_price(val):
    """Format price  always show exact cents, never round off decimals."""
    if pd.isna(val):
        return ""
    if val >= 1000:
        return f"{val:,.2f}"
    elif val >= 1:
        return f"{val:.2f}"
    else:
        return f"{val:.6f}"


def _render_signal_table(df: pd.DataFrame, show_cols: List[str] = None):
    """Render a styled signal table using plotly for proper coloring."""
    if df.empty:
        st.info("No data to display.")
        return
    display_cols = show_cols or [
        "label",
        "price",
        "1D%",
        "5D%",
        "1M%",
        "RSI",
        "Vol%",
        "Score",
        "Signal",
    ]
    available = [c for c in display_cols if c in df.columns]
    if not available:
        st.info("No columns to display.")
        return
    show_df = df[available].copy()

    # Format columns for display
    rename_map = {"label": "Asset", "price": "Price", "Vol%": "Ann.Vol%"}
    show_df = show_df.rename(
        columns={k: v for k, v in rename_map.items() if k in show_df.columns}
    )

    # Format numeric columns to strings for clean display
    fmt_df = show_df.copy()
    for c in fmt_df.columns:
        if c == "Price":
            fmt_df[c] = fmt_df[c].apply(lambda x: _fmt_price(x) if pd.notna(x) else "")
        elif c in ("1D%", "5D%", "1M%", "3M%", "Ann.Vol%"):
            fmt_df[c] = fmt_df[c].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "")
        elif c == "RSI":
            fmt_df[c] = fmt_df[c].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "")
        elif c == "Score":
            fmt_df[c] = fmt_df[c].apply(lambda x: f"{x:+.0f}" if pd.notna(x) else "")

    # Use column_config for colored display
    col_config = {}
    for c in ["1D%", "5D%", "1M%", "3M%"]:
        if c in show_df.columns:
            col_config[c] = st.column_config.TextColumn(c, help=f"{c} return")
    if "Score" in show_df.columns:
        col_config["Score"] = st.column_config.TextColumn(
            "Score", help="Momentum score -100 to +100"
        )

    st.dataframe(
        fmt_df,
        width='stretch',
        hide_index=True,
        height=min(35 * len(fmt_df) + 38, 600),
        column_config=col_config,
    )


def _render_momentum_bars(df: pd.DataFrame, title: str, col: str = "Score"):
    """Horizontal bar chart of momentum scores."""
    if df.empty:
        return
    top = df.head(20).copy()
    colors = ["#00e676" if s > 0 else "#ff5252" for s in top[col]]
    fig = go.Figure(
        go.Bar(
            x=top[col].values,
            y=top["label"].values,
            orientation="h",
            marker_color=colors,
            text=[f"{s:+.0f}" for s in top[col]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        height=max(250, len(top) * 28),
        template="plotly_dark",
        yaxis=dict(autorange="reversed"),
        xaxis_title="Momentum Score",
        margin=dict(l=0, r=0, t=35, b=0),
    )
    st.plotly_chart(fig, width='stretch')


def _render_rsi_chart(df: pd.DataFrame):
    """RSI bar chart with overbought/oversold zones."""
    if df.empty or len(df) < 3:
        return
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["label"],
            y=df["RSI"],
            marker_color=[
                "#ff5252" if r > 70 else "#00e676" if r < 30 else "#78909c"
                for r in df["RSI"]
            ],
            text=[f"{r:.0f}" for r in df["RSI"]],
            textposition="outside",
        )
    )
    fig.add_hline(
        y=70, line_dash="dash", line_color="#ff5252", annotation_text="Overbought (70)"
    )
    fig.add_hline(
        y=30, line_dash="dash", line_color="#00e676", annotation_text="Oversold (30)"
    )
    fig.add_hline(y=50, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="RSI Levels",
        height=300,
        template="plotly_dark",
        yaxis_range=[0, 100],
        margin=dict(l=0, r=0, t=35, b=0),
    )
    st.plotly_chart(fig, width='stretch')


#
# MAIN SCANNER VIEW
#


def show_market_scanner():
    """Render the comprehensive market scanner."""
    st.title("[SCAN] Market Scanner")
    st.caption("Real-time multi-asset momentum, signals & heatmaps across all markets")

    if HAS_LATEST_PRICE:
        st.caption(
            " **Data:** Intraday prices (1-min bars) + daily close fallback via yfinance"
        )
    else:
        st.caption(" **Data:** Daily close prices via yfinance")

    #  Controls
    col_c1, col_c2, col_c3 = st.columns([2, 1, 1])
    with col_c1:
        scan_period = st.selectbox(
            "Lookback Period",
            ["1mo", "3mo", "6mo", "1y"],
            index=1,
            key="scanner_period",
        )
    with col_c2:
        sort_by = st.selectbox(
            "Sort By",
            ["Score", "1D%", "5D%", "1M%", "RSI", "Vol%"],
            index=0,
            key="scanner_sort",
        )
    with col_c3:
        min_score = st.slider("Min Score", -100, 100, -100, 10, key="scanner_min_score")

    tabs = st.tabs(
        [
            " Overview",
            "[DATA] Sector ETFs",
            "[CHART] Stocks",
            "[COIN] Crypto",
            " Commodities",
            " Forex",
            "[DOWN] Bonds",
            " Top Movers",
        ]
    )

    #  TAB 0: Overview
    with tabs[0]:
        st.subheader("Market Pulse")

        with st.spinner("Scanning markets..."):
            overview_df = _scan_universe(
                _OVERVIEW_ITEMS, period=scan_period, max_workers=8
            )

        if not overview_df.empty:
            cols = st.columns(min(len(overview_df), 4))
            for i, (_, row) in enumerate(overview_df.iterrows()):
                with cols[i % 4]:
                    delta_str = f"{row['1D%']:+.2f}%"
                    st.metric(row["label"], f"{row['price']:,.2f}", delta_str)

            _render_heatmap(overview_df, "Market Overview  1D Change", "1D%")

        # Sector snapshot using ETF proxies (these ARE ETFs, labeled correctly)
        st.markdown("### Sector ETF Momentum")
        with st.spinner("Scanning sectors..."):
            sector_df = _scan_universe(_ETF_SECTORS, period=scan_period)
        if not sector_df.empty:
            _render_heatmap(sector_df, "Sector ETFs  1D Performance", "1D%")
            _render_signal_table(sector_df)

    #  TAB 1: Sector ETFs
    with tabs[1]:
        st.subheader("[DATA] Sector ETF Scanner")
        st.caption("Prices shown are ETF prices (SPY, XLK, etc.), not raw index values")
        with st.spinner("Scanning sector ETFs..."):
            sector_full = _scan_universe(_ETF_SECTORS, period=scan_period)
        if not sector_full.empty:
            sector_full = sector_full[sector_full["Score"] >= min_score].sort_values(
                sort_by, ascending=(sort_by in ("Vol%", "RSI"))
            )
            col_a, col_b = st.columns(2)
            with col_a:
                _render_heatmap(sector_full, "Sector Heatmap  1M Return", "1M%")
            with col_b:
                _render_momentum_bars(sector_full, "Sector Momentum")
            _render_signal_table(
                sector_full,
                [
                    "label",
                    "price",
                    "1D%",
                    "5D%",
                    "1M%",
                    "3M%",
                    "RSI",
                    "Vol%",
                    "Score",
                    "Signal",
                ],
            )

    #  TAB 2: Stocks
    with tabs[2]:
        st.subheader("[CHART] Stock Scanner")
        _watchlist_groups = _get_watchlist_groups() or {"All Stocks": []}
        stock_group = st.selectbox(
            "Stock Group", list(_watchlist_groups.keys()), key="stock_group_sel"
        )
        with st.spinner(f"Scanning {stock_group}..."):
            stock_df = _scan_list(_watchlist_groups[stock_group], period=scan_period)
        if not stock_df.empty:
            stock_df = stock_df[stock_df["Score"] >= min_score].sort_values(
                sort_by, ascending=(sort_by in ("Vol%", "RSI"))
            )
            _render_heatmap(stock_df, f"{stock_group}  1D Change", "1D%")

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                _render_momentum_bars(stock_df, f"{stock_group} Momentum")
            with col_s2:
                _render_rsi_chart(stock_df)

            _render_signal_table(
                stock_df,
                [
                    "label",
                    "price",
                    "1D%",
                    "5D%",
                    "1M%",
                    "RSI",
                    "Vol%",
                    "Score",
                    "Signal",
                ],
            )

            # Volume spike alerts
            spikes = stock_df[stock_df["Vol Spike"] == True] if "Vol Spike" in stock_df.columns else pd.DataFrame()
            if not spikes.empty:
                st.warning(
                    f" **Volume Spike Alert:** {', '.join(spikes['label'].tolist())}"
                )

    #  TAB 3: Crypto
    with tabs[3]:
        st.subheader("[COIN] Crypto Scanner")
        with st.spinner("Scanning crypto..."):
            crypto_df = _scan_list(_get_crypto_list(), period=scan_period)
        if not crypto_df.empty:
            crypto_df = crypto_df[crypto_df["Score"] >= min_score].sort_values(
                sort_by, ascending=(sort_by in ("Vol%", "RSI"))
            )

            crypto_cols = st.columns(min(len(crypto_df), 5))
            for i, (_, row) in enumerate(crypto_df.head(5).iterrows()):
                with crypto_cols[i]:
                    st.metric(
                        row["label"].replace("-USD", ""),
                        f"${row['price']:,.2f}",
                        f"{row['1D%']:+.2f}%",
                    )

            _render_heatmap(crypto_df, "Crypto Heatmap  1D", "1D%")
            _render_momentum_bars(crypto_df, "Crypto Momentum")
            _render_signal_table(crypto_df)
        else:
            st.info("Could not fetch crypto data.")

    #  TAB 4: Commodities
    with tabs[4]:
        st.subheader(" Commodities Scanner")
        with st.spinner("Scanning commodities..."):
            comm_df = _scan_universe(_COMMODITIES, period=scan_period)
        if not comm_df.empty:
            comm_df = comm_df[comm_df["Score"] >= min_score].sort_values(
                sort_by, ascending=(sort_by in ("Vol%", "RSI"))
            )
            col_cm1, col_cm2 = st.columns(2)
            with col_cm1:
                _render_heatmap(comm_df, "Commodities  1M Return", "1M%")
            with col_cm2:
                _render_momentum_bars(comm_df, "Commodity Momentum")
            _render_signal_table(comm_df)
        else:
            st.info("Could not fetch commodity data.")

    #  TAB 5: Forex
    with tabs[5]:
        st.subheader(" Forex Scanner")
        with st.spinner("Scanning FX pairs..."):
            fx_df = _scan_universe(_FOREX_PAIRS, period=scan_period)
        if not fx_df.empty:
            fx_df = fx_df[fx_df["Score"] >= min_score].sort_values(
                sort_by, ascending=(sort_by in ("Vol%", "RSI"))
            )
            _render_heatmap(fx_df, "Forex  1D Change", "1D%")
            _render_signal_table(fx_df)
        else:
            st.info("Could not fetch forex data.")

    #  TAB 6: Bonds
    with tabs[6]:
        st.subheader("[DOWN] Bond / Fixed Income Scanner")
        with st.spinner("Scanning bonds..."):
            bond_df = _scan_universe(_BONDS, period=scan_period)
        if not bond_df.empty:
            bond_df = bond_df[bond_df["Score"] >= min_score].sort_values(
                sort_by, ascending=(sort_by in ("Vol%", "RSI"))
            )
            _render_heatmap(bond_df, "Bonds  1M Return", "1M%")
            _render_signal_table(bond_df)
        else:
            st.info("Could not fetch bond data.")

    #  TAB 7: Top Movers
    with tabs[7]:
        st.subheader(" Top Movers & Losers (Cross-Asset)")
        with st.spinner("Aggregating all assets..."):
            all_frames = []
            for name, syms in (_get_watchlist_groups() or {"Stocks": []}).items():
                adf = _scan_list(syms, period=scan_period, max_workers=10)
                if not adf.empty:
                    adf["category"] = name
                    all_frames.append(adf)

            for cat_name, cat_items in [
                ("Crypto", {s: s for s in _get_crypto_list()}),
                ("Commodities", _COMMODITIES),
                ("Forex", _FOREX_PAIRS),
                ("Sector ETFs", _ETF_SECTORS),
                ("Bonds", _BONDS),
            ]:
                adf = _scan_universe(cat_items, period=scan_period, max_workers=8)
                if not adf.empty:
                    adf["category"] = cat_name
                    all_frames.append(adf)

        if all_frames:
            all_df = pd.concat(all_frames, ignore_index=True).drop_duplicates(
                subset="symbol"
            )

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("### [UP] Top Gainers (1D)")
                gainers = all_df.nlargest(15, "1D%")
                _render_signal_table(
                    gainers,
                    [
                        "label",
                        "category",
                        "price",
                        "1D%",
                        "5D%",
                        "RSI",
                        "Score",
                        "Signal",
                    ],
                )

            with col_t2:
                st.markdown("### [DOWN] Top Losers (1D)")
                losers = all_df.nsmallest(15, "1D%")
                _render_signal_table(
                    losers,
                    [
                        "label",
                        "category",
                        "price",
                        "1D%",
                        "5D%",
                        "RSI",
                        "Score",
                        "Signal",
                    ],
                )

            st.markdown("### [TARGET] Highest Momentum Score")
            top_momentum = all_df.nlargest(20, "Score")
            _render_momentum_bars(top_momentum, "Cross-Asset Momentum Leaders")
            _render_signal_table(
                top_momentum,
                [
                    "label",
                    "category",
                    "price",
                    "1D%",
                    "1M%",
                    "RSI",
                    "Vol%",
                    "Score",
                    "Signal",
                ],
            )

            st.markdown("###  Most Oversold (RSI < 35)")
            oversold = all_df[all_df["RSI"] < 35].sort_values("RSI")
            if not oversold.empty:
                _render_signal_table(
                    oversold.head(15),
                    ["label", "category", "price", "1D%", "RSI", "Score", "Signal"],
                )
            else:
                st.info("No assets currently oversold (RSI < 35).")

            st.markdown("###  Most Overbought (RSI > 70)")
            overbought = all_df[all_df["RSI"] > 70].sort_values("RSI", ascending=False)
            if not overbought.empty:
                _render_signal_table(
                    overbought.head(15),
                    ["label", "category", "price", "1D%", "RSI", "Score", "Signal"],
                )
            else:
                st.info("No assets currently overbought (RSI > 70).")

            # Volume spikes across all assets
            all_spikes = all_df[all_df["Vol Spike"] == True] if "Vol Spike" in all_df.columns else pd.DataFrame()
            if not all_spikes.empty:
                st.markdown("###  Volume Spike Alerts")
                _render_signal_table(
                    all_spikes,
                    ["label", "category", "price", "1D%", "Vol%", "Score", "Signal"],
                )
        else:
            st.info("No data available for aggregation.")
