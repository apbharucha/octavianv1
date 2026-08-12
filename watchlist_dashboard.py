"""
Octavian Watchlist Dashboard  Live monitoring and deep-dive for user's watchlist.
Author: APB - Octavian Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Optional
import concurrent.futures

from trader_profile import get_trader_profile, get_watchlist, save_profile
from data_sources import get_stock

try:
    from data_sources import get_realtime_price as _ds_get_realtime_price
    HAS_REALTIME = True
except ImportError:
    HAS_REALTIME = False
    _ds_get_realtime_price = None


# 
# ROBUST LIVE PRICE FETCHING
# 

def _get_accurate_live_price(symbol: str) -> float:
    """Get the most accurate live price for any symbol using centralized data_sources."""
    # Use centralized function with multiple fallbacks (yfinance + Polygon + Stooq + Alpha Vantage)
    if HAS_REALTIME and _ds_get_realtime_price:
        try:
            price, _ = _ds_get_realtime_price(symbol)
            if price and price > 0:
                return float(price)
        except Exception:
            pass
    
    # Fallback to direct yfinance
    import yfinance as yf
    
    sym = symbol.strip().upper()
    yf_sym = sym
    if '/' in sym and '=' not in sym:
        yf_sym = sym.replace('/', '') + '=X'
    
    try:
        tk = yf.Ticker(yf_sym)
        
        # Try 1-minute data
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
                        return float(close_col.iloc[-1])
        except Exception:
            pass
        
        # Try fast_info
        try:
            info = tk.fast_info
            price = getattr(info, 'last_price', None) or getattr(info, 'regularMarketPrice', None)
            if price and price > 0:
                return float(price)
        except Exception:
            pass
        
        # Try daily data
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
                    if len(close_col) > 0:
                        return float(close_col.iloc[-1])
        except Exception:
            pass
            
    except Exception:
        pass
    
    return 0.0


# 
# HELPER FUNCTIONS
# 

def _fetch_symbol_data(symbol: str, period: str = "6mo"):
    """Fetch data for a symbol and return (DataFrame, asset_type) using robust method."""
    try:
        # Import data sources efficiently
        from data_sources import get_stock, get_fx, get_futures_proxy, get_fresh_quote
        
        sym = symbol.strip().upper()
        
        #  Forex 
        if '/' in sym or sym.endswith('=X'):
            # Try raw cache first
            df = get_fresh_quote(sym, period=period)
            if df is not None and not df.empty:
                return df, 'forex'
            
            # Try get_fx
            fx_key = sym.replace('/', '_').replace('=X', '')
            df = get_fx(fx_key)
            if df is not None and not df.empty:
                return df, 'forex'
                
            # Try stock fallback
            df = get_stock(sym, period=period)
            if df is not None and not df.empty:
                return df, 'forex'
                
            return None, 'forex'

        #  Futures 
        if sym.endswith('=F'):
            df = get_futures_proxy(sym, period=period)
            if df is not None and not df.empty:
                return df, 'futures'
            return None, 'futures'

        #  Crypto 
        if '-' in sym:
            df = get_stock(sym, period=period)
            if df is not None and not df.empty:
                return df, 'crypto'
            return None, 'crypto'

        #  Stocks / Indices 
        df = get_stock(sym, period=period)
        if df is not None and not df.empty:
            return df, 'stock'
            
        return None, 'stock'
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None, None

def _safe_close(df):
    """Extract close price series safely."""
    if df is None or df.empty:
        return None
    if 'Close' in df.columns:
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            return close.iloc[:, 0]
        return close
    return None

def _compute_technicals(close, df):
    """Compute technical indicators and generate signal with confidence."""
    if close is None or len(close) < 20:
        return {
            'signal': 'NEUTRAL',
            'confidence': 0.0,
            'bullish_prob': 0.33,
            'bearish_prob': 0.33,
            'metrics': {},
            'factors': [],
        }
    
    try:
        import numpy as np
        factors = []
        bullish_score = 0
        bearish_score = 0
        total_factors = 0
        
        # Simple moving averages
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean() if len(close) >= 50 else None
        
        current_price = float(close.iloc[-1])
        sma20_val = float(sma20.iloc[-1]) if not sma20.empty else None
        sma50_val = float(sma50.iloc[-1]) if sma50 is not None and not sma50.empty else None
        
        # SMA signals
        if sma20_val:
            total_factors += 1
            if current_price > sma20_val:
                bullish_score += 1
                factors.append(f"Price above SMA20 (${sma20_val:.2f})")
            else:
                bearish_score += 1
                factors.append(f"Price below SMA20 (${sma20_val:.2f})")
        
        if sma50_val:
            total_factors += 1
            if current_price > sma50_val:
                bullish_score += 1
                factors.append(f"Price above SMA50 (${sma50_val:.2f})")
            else:
                bearish_score += 1
                factors.append(f"Price below SMA50 (${sma50_val:.2f})")
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_val = float(rsi.iloc[-1]) if not rsi.empty else 50
        
        total_factors += 1
        if rsi_val > 70:
            bearish_score += 1
            factors.append(f"RSI overbought ({rsi_val:.0f})")
        elif rsi_val < 30:
            bullish_score += 1
            factors.append(f"RSI oversold ({rsi_val:.0f})")
        elif rsi_val > 50:
            bullish_score += 0.5
            factors.append(f"RSI bullish ({rsi_val:.0f})")
        else:
            bearish_score += 0.5
            factors.append(f"RSI bearish ({rsi_val:.0f})")
        
        # Momentum (5-day return)
        if len(close) >= 5:
            total_factors += 1
            momentum = (current_price / float(close.iloc[-5]) - 1) * 100
            if momentum > 2:
                bullish_score += 1
                factors.append(f"Strong 5D momentum ({momentum:+.1f}%)")
            elif momentum < -2:
                bearish_score += 1
                factors.append(f"Weak 5D momentum ({momentum:+.1f}%)")
            elif momentum > 0:
                bullish_score += 0.5
            else:
                bearish_score += 0.5
        
        # Volatility
        returns = close.pct_change().dropna()
        ann_vol = float(returns.std() * np.sqrt(252) * 100) if len(returns) > 10 else 0
        
        # Calculate signal and confidence
        if total_factors > 0:
            bullish_prob = bullish_score / total_factors
            bearish_prob = bearish_score / total_factors
        else:
            bullish_prob = 0.33
            bearish_prob = 0.33
        
        if bullish_prob > 0.6:
            signal = 'BULLISH'
            confidence = bullish_prob
        elif bearish_prob > 0.6:
            signal = 'BEARISH'
            confidence = bearish_prob
        else:
            signal = 'NEUTRAL'
            confidence = max(bullish_prob, bearish_prob, 0.5)
        
        metrics = {
            'sma20': sma20_val,
            'sma50': sma50_val,
            'rsi': rsi_val,
            'ann_vol': ann_vol,
        }
        
        return {
            'signal': signal,
            'confidence': confidence,
            'bullish_prob': bullish_prob,
            'bearish_prob': bearish_prob,
            'metrics': metrics,
            'factors': factors,
        }
    except Exception:
        return {
            'signal': 'NEUTRAL',
            'confidence': 0.0,
            'bullish_prob': 0.33,
            'bearish_prob': 0.33,
            'metrics': {},
            'factors': [],
        }

def _flatten(df):
    """Flatten DataFrame if needed."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df


# 
# DATA FETCHING
# 

def _fetch_watchlist_data(symbols: List[str], period: str = "6mo") -> Dict:
    """Fetch data for all watchlist symbols in parallel."""
    results = {}
    if not symbols:
        return results

    def _fetch_one(sym):
        try:
            df, asset_type = _fetch_symbol_data(sym, period)
            if df is None or df.empty:
                return sym, None
            close = _safe_close(df)
            if close is None or len(close) < 5:
                return sym, None
            tech = _compute_technicals(close, df)
            price = float(close.iloc[-1])
            # Use true previous trading day close, not close.iloc[-2]
            prev = None
            if HAS_REALTIME and _ds_get_realtime_price:
                try:
                    _, _prev_close = _ds_get_realtime_price(sym)
                    if _prev_close and _prev_close > 0:
                        prev = float(_prev_close)
                except Exception:
                    pass
            if prev is None:
                prev = float(close.iloc[-2]) if len(close) >= 2 else price
            change_1d = (price / prev - 1) * 100

            # Period return
            period_ret = (price / float(close.iloc[0]) - 1) * 100

            # 5d return
            ret_5d = (price / float(close.iloc[-5]) - 1) * 100 if len(close) >= 5 else 0

            return sym, {
                "symbol": sym,
                "asset_type": asset_type,
                "df": df,
                "close": close,
                "price": price,
                "change_1d": change_1d,
                "ret_5d": ret_5d,
                "period_ret": period_ret,
                "signal": tech["signal"],
                "confidence": tech["confidence"],
                "bullish_prob": tech.get("bullish_prob", 0.33),
                "bearish_prob": tech.get("bearish_prob", 0.33),
                "metrics": tech["metrics"],
                "factors": tech["factors"],
            }
        except Exception:
            return sym, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(symbols), 8)) as pool:
        futures = {pool.submit(_fetch_one, s): s for s in symbols}
        done, _ = concurrent.futures.wait(futures, timeout=30)
        for f in done:
            try:
                sym, data = f.result(timeout=5)
                if data:
                    results[sym] = data
            except Exception:
                pass
    return results


# 
# CHARTS
# 

def _mini_sparkline(close: pd.Series, height: int = 120) -> go.Figure:
    """Create a tiny sparkline chart."""
    color = "#00d4aa" if float(close.iloc[-1]) >= float(close.iloc[0]) else "#ff5252"
    fig = go.Figure(go.Scatter(
        x=close.index, y=close.values,
        mode='lines', line=dict(color=color, width=1.5),
        fill='tozeroy', fillcolor=color.replace(")", ",0.08)").replace("rgb", "rgba") if "rgb" in color else f"rgba(0,212,170,0.08)" if color == "#00d4aa" else "rgba(255,82,82,0.08)",
    ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_dark", showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def _full_chart(sym: str, data: Dict, period_label: str) -> go.Figure:
    """Full price + volume + SMA chart for deep dive."""
    close = data["close"]
    df = data["df"]

    has_vol = "Volume" in df.columns
    if has_vol:
        vol = df["Volume"]
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]
        vol = vol.dropna().astype(float)
        has_vol = len(vol) > 0 and vol.sum() > 0 and vol.max() > 1

    # Determine rows: price always, volume only if valid, RSI always
    rows = 2 if has_vol else 1
    rsi_row = rows + 1
    total_rows = rsi_row
    heights = [0.55, 0.20, 0.25] if has_vol else [0.65, 0.35]

    fig = make_subplots(
        rows=total_rows, cols=1, shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=0.05,
    )

    # Price + SMAs
    fig.add_trace(go.Scatter(x=close.index, y=close.values,
        mode='lines', name=sym, line=dict(color='#00d4aa', width=2)), row=1, col=1)
    if len(close) >= 20:
        sma20 = close.rolling(20).mean()
        fig.add_trace(go.Scatter(x=sma20.index, y=sma20.values,
            mode='lines', name='SMA20', line=dict(color='orange', width=1, dash='dot')), row=1, col=1)
    if len(close) >= 50:
        sma50 = close.rolling(50).mean()
        fig.add_trace(go.Scatter(x=sma50.index, y=sma50.values,
            mode='lines', name='SMA50', line=dict(color='#ff6b6b', width=1, dash='dot')), row=1, col=1)

    # Bollinger Bands
    if len(close) >= 20:
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        fig.add_trace(go.Scatter(x=bb_upper.index, y=bb_upper.values,
            mode='lines', name='BB Upper', line=dict(color='rgba(150,150,150,0.3)', width=1),
            showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=bb_lower.index, y=bb_lower.values,
            mode='lines', name='BB Lower', line=dict(color='rgba(150,150,150,0.3)', width=1),
            fill='tonexty', fillcolor='rgba(150,150,150,0.05)', showlegend=False), row=1, col=1)

    # Volume (only if valid)
    if has_vol:
        fig.add_trace(go.Bar(x=vol.index, y=vol.values,
            name='Volume', marker_color='rgba(100,100,255,0.3)'), row=2, col=1)
        fig.update_yaxes(title_text="Vol", row=2, col=1)

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi.values,
        mode='lines', name='RSI', line=dict(color='#ab47bc', width=1.5)), row=rsi_row, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=rsi_row, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=rsi_row, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=rsi_row, col=1)

    fig.update_layout(
        height=480, template="plotly_dark",
        title=dict(text=f"{sym}  {period_label}", x=0.01, font=dict(size=13)),
        showlegend=True, margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=9)),
    )
    return fig


# 
# MAIN PAGE
# 

def show_watchlist_dashboard():
    st.title("[PIN] Watchlist Dashboard")

    profile = get_trader_profile()
    watchlist = list(profile.get("watchlist", []))

    if not watchlist:
        st.info("Your watchlist is empty. Go to ** Trader Profile** to add symbols.")
        if st.button("Go to Profile "):
            st.session_state["_nav_override"] = "Trader Profile"
            st.rerun()
        return

    #  Controls 
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
    with col_ctrl1:
        st.caption(f"Tracking **{len(watchlist)}** symbols")
    with col_ctrl2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2, key="wl_period")
    with col_ctrl3:
        if st.button(" Refresh", key="wl_refresh"):
            st.cache_data.clear()
            st.rerun()

    # Quick-add inline
    with st.expander(" Quick Add / Remove Symbols"):
        col_qa1, col_qa2 = st.columns([3, 1])
        with col_qa1:
            add_sym = st.text_input("Add symbol", placeholder="TSLA, EUR/USD", key="wl_quick_add")
        with col_qa2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Add", key="wl_qa_btn") and add_sym:
                for s in add_sym.split(","):
                    s = s.strip().upper()
                    if s and s not in watchlist:
                        watchlist.append(s)
                profile["watchlist"] = watchlist
                save_profile(profile)
                st.rerun()

        if watchlist:
            remove_sym = st.multiselect("Remove symbols", watchlist, key="wl_remove")
            if remove_sym and st.button("Remove Selected", key="wl_rm_btn"):
                profile["watchlist"] = [s for s in watchlist if s not in remove_sym]
                save_profile(profile)
                st.rerun()

    #  Fetch all data 
    with st.spinner(f"Loading {len(watchlist)} symbols..."):
        all_data = _fetch_watchlist_data(watchlist, period)

    failed = [s for s in watchlist if s not in all_data]
    if failed:
        st.warning(f"Could not load: {', '.join(failed)}")

    if not all_data:
        st.error("No data available for any watchlist symbol.")
        return

    #  Tab layout 
    wl_tabs = st.tabs([" Overview", " Charts", " Deep Dive", " Comparison"])

    #  Tab 1: Overview Grid 
    with wl_tabs[0]:
        st.subheader("Watchlist Overview")

        # Summary table
        table_rows = []
        for sym, d in all_data.items():
            sig_icon = "" if d["signal"] == "BULLISH" else "" if d["signal"] == "BEARISH" else ""
            m = d.get("metrics", {})
            table_rows.append({
                "Symbol": sym,
                "Price": f"${d['price']:,.2f}",
                "1D %": f"{d['change_1d']:+.2f}%",
                "5D %": f"{d['ret_5d']:+.2f}%",
                f"{period} %": f"{d['period_ret']:+.2f}%",
                "Signal": f"{sig_icon} {d['signal']}",
                "Conf": f"{d['confidence']:.0%}",
                "RSI": f"{m.get('rsi', 0):.0f}" if "rsi" in m else "",
                "Vol": f"{m.get('ann_vol', 0):.1f}%" if "ann_vol" in m else "",
            })

        df_table = pd.DataFrame(table_rows)
        st.dataframe(df_table, width='stretch', hide_index=True, height=min(400, 40 + len(table_rows) * 35))

        # Metric cards
        st.markdown("---")
        sorted_by_change = sorted(all_data.values(), key=lambda x: x["change_1d"], reverse=True)
        cards_per_row = min(len(sorted_by_change), 4)
        if cards_per_row > 0:
            rows = [sorted_by_change[i:i+cards_per_row] for i in range(0, len(sorted_by_change), cards_per_row)]
            for row in rows[:3]:  # Max 3 rows of cards
                cols = st.columns(len(row))
                for col, d in zip(cols, row):
                    with col:
                        sig_icon = "" if d["signal"] == "BULLISH" else "" if d["signal"] == "BEARISH" else ""
                        st.metric(
                            f"{d['symbol']} {sig_icon}",
                            f"${d['price']:,.2f}",
                            f"{d['change_1d']:+.2f}% today"
                        )
                        st.plotly_chart(_mini_sparkline(d["close"].tail(30), height=80),
                                       width='stretch', key=f"spark_{d['symbol']}")

    #  Tab 2: Charts 
    with wl_tabs[1]:
        st.subheader("Price Charts")
        chart_layout = st.radio("Layout", ["Single", "Grid (2-col)"], horizontal=True, key="wl_chart_layout")

        if chart_layout == "Single":
            selected_sym = st.selectbox("Symbol", list(all_data.keys()), key="wl_chart_sym")
            if selected_sym in all_data:
                fig = _full_chart(selected_sym, all_data[selected_sym], period)
                st.plotly_chart(fig, width='stretch')

                # Show factors
                d = all_data[selected_sym]
                sig_icon = "" if d["signal"] == "BULLISH" else "" if d["signal"] == "BEARISH" else ""
                st.markdown(f"**{sig_icon} {d['signal']}**  Confidence: {d['confidence']:.0%} | "
                            f"Bull: {d['bullish_prob']:.0%} | Bear: {d['bearish_prob']:.0%}")
                for f in d.get("factors", [])[:6]:
                    st.caption(f" {f}")
        else:
            syms = list(all_data.keys())
            for i in range(0, len(syms), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(syms):
                        sym = syms[idx]
                        with col:
                            fig = _full_chart(sym, all_data[sym], period)
                            fig.update_layout(height=350)
                            st.plotly_chart(fig, width='stretch', key=f"grid_{sym}")

    #  Tab 3: Deep Dive 
    with wl_tabs[2]:
        st.subheader(" Symbol Deep Dive")
        dd_sym = st.selectbox("Select Symbol", list(all_data.keys()), key="wl_dd_sym")
        if dd_sym in all_data:
            d = all_data[dd_sym]
            close = d["close"]
            df_raw = d["df"]
            m = d["metrics"]

            # Header
            sig_icon = "" if d["signal"] == "BULLISH" else "" if d["signal"] == "BEARISH" else ""
            st.markdown(f"### {dd_sym}  {sig_icon} {d['signal']}")

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            with mc1:
                st.metric("Price", f"${d['price']:,.2f}", f"{d['change_1d']:+.2f}%")
            with mc2:
                st.metric("5D Return", f"{d['ret_5d']:+.2f}%")
            with mc3:
                st.metric(f"{period} Return", f"{d['period_ret']:+.2f}%")
            with mc4:
                st.metric("Confidence", f"{d['confidence']:.0%}")
            with mc5:
                rsi_val = m.get("rsi", 50)
                st.metric("RSI", f"{rsi_val:.0f}",
                          "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral")

            # Full chart
            fig = _full_chart(dd_sym, d, period)
            fig.update_layout(height=550)
            st.plotly_chart(fig, width='stretch')

            # Technical details
            st.markdown("###  Technical Breakdown")
            tech_cols = st.columns(4)
            with tech_cols[0]:
                st.metric("SMA20", f"${m.get('sma20', 0):,.2f}" if "sma20" in m else "")
            with tech_cols[1]:
                st.metric("SMA50", f"${m.get('sma50', 0):,.2f}" if "sma50" in m else "")
            with tech_cols[2]:
                st.metric("MACD Hist", f"{m.get('macd_hist', 0):.4f}" if "macd_hist" in m else "")
            with tech_cols[3]:
                st.metric("Ann. Volatility", f"{m.get('ann_vol', 0):.1f}%" if "ann_vol" in m else "")

            # Signal factors
            st.markdown("###  Signal Factors")
            for f in d.get("factors", []):
                st.caption(f" {f}")

            # Probability breakdown
            st.markdown("###  Probability Breakdown")
            prob_fig = go.Figure(go.Bar(
                x=["Bullish", "Neutral", "Bearish"],
                y=[d["bullish_prob"], 1 - d["bullish_prob"] - d["bearish_prob"], d["bearish_prob"]],
                marker_color=["green", "gray", "red"],
                text=[f"{d['bullish_prob']:.0%}",
                      f"{1 - d['bullish_prob'] - d['bearish_prob']:.0%}",
                      f"{d['bearish_prob']:.0%}"],
                textposition="outside",
            ))
            prob_fig.update_layout(height=250, template="plotly_dark", yaxis_title="Probability")
            st.plotly_chart(prob_fig, width='stretch')

            # Return distribution
            daily_rets = close.pct_change().dropna()
            if len(daily_rets) > 20:
                st.markdown("###  Return Distribution")
                dist_fig = px.histogram(x=daily_rets.values * 100, nbins=40,
                                        title="Daily Returns (%)",
                                        color_discrete_sequence=["#636EFA"])
                dist_fig.update_layout(template="plotly_dark", height=250,
                                       xaxis_title="Daily Return %", yaxis_title="Frequency")
                st.plotly_chart(dist_fig, width='stretch')

            # Risk metrics
            if len(daily_rets) > 20:
                st.markdown("###  Risk Metrics")
                ann_vol = float(daily_rets.std() * np.sqrt(252) * 100)
                sharpe = float((daily_rets.mean() / (daily_rets.std() + 1e-10)) * np.sqrt(252))
                var_95 = float(daily_rets.quantile(0.05) * 100)
                cum = (1 + daily_rets).cumprod()
                max_dd = float(((cum - cum.expanding().max()) / cum.expanding().max()).min() * 100)

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("Volatility", f"{ann_vol:.1f}%")
                with rc2:
                    st.metric("Sharpe", f"{sharpe:.2f}")
                with rc3:
                    st.metric("VaR 95%", f"{var_95:+.2f}%")
                with rc4:
                    st.metric("Max Drawdown", f"{max_dd:.1f}%")

    #  Tab 4: Comparison 
    with wl_tabs[3]:
        st.subheader(" Performance Comparison")

        compare_syms = st.multiselect("Select symbols to compare",
                                       list(all_data.keys()),
                                       default=list(all_data.keys())[:5],
                                       key="wl_compare")
        if compare_syms:
            # Normalized cumulative returns
            fig_comp = go.Figure()
            colors = px.colors.qualitative.Set1
            for idx, sym in enumerate(compare_syms):
                if sym in all_data:
                    close = all_data[sym]["close"]
                    norm = (close / float(close.iloc[0]) - 1) * 100
                    fig_comp.add_trace(go.Scatter(
                        x=norm.index, y=norm.values,
                        mode='lines', name=sym,
                        line=dict(color=colors[idx % len(colors)], width=2),
                    ))
            fig_comp.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_comp.update_layout(
                title="Normalized Returns (%)", height=450, template="plotly_dark",
                hovermode="x unified", yaxis_title="Return %",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(fig_comp, width='stretch')

            # Correlation matrix
            if len(compare_syms) >= 2:
                st.markdown("### Correlation Matrix")
                rets_dict = {}
                for sym in compare_syms:
                    if sym in all_data:
                        s = all_data[sym]["close"].pct_change().dropna()
                        # Normalize timezone to avoid tz-naive/tz-aware mismatch
                        if hasattr(s.index, 'tz') and s.index.tz is not None:
                            s = s.tz_localize(None)
                        rets_dict[sym] = s
                rets_df = pd.DataFrame(rets_dict).dropna()
                if len(rets_df) > 10:
                    corr = rets_df.corr()
                    fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                                         color_continuous_scale="RdBu",
                                         color_continuous_midpoint=0)
                    fig_corr.update_layout(height=400, template="plotly_dark")
                    st.plotly_chart(fig_corr, width='stretch')

            # Summary comparison table
            st.markdown("### Comparative Metrics")
            comp_rows = []
            for sym in compare_syms:
                if sym in all_data:
                    d = all_data[sym]
                    m = d["metrics"]
                    daily_rets = d["close"].pct_change().dropna()
                    sharpe = float((daily_rets.mean() / (daily_rets.std() + 1e-10)) * np.sqrt(252)) if len(daily_rets) > 20 else 0
                    comp_rows.append({
                        "Symbol": sym,
                        "Price": f"${d['price']:,.2f}",
                        "1D": f"{d['change_1d']:+.2f}%",
                        "5D": f"{d['ret_5d']:+.2f}%",
                        f"{period}": f"{d['period_ret']:+.2f}%",
                        "Signal": d["signal"],
                        "RSI": f"{m.get('rsi', 0):.0f}",
                        "Vol%": f"{m.get('ann_vol', 0):.1f}" if "ann_vol" in m else "",
                        "Sharpe": f"{sharpe:.2f}",
                    })
            if comp_rows:
                st.dataframe(pd.DataFrame(comp_rows), width='stretch', hide_index=True)
