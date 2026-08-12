import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ml_analysis
from ml_analysis import get_analyzer
from indicators import add_indicators
from data_sources import get_stock, get_fx, get_futures_proxy

try:
    from data_sources import get_fresh_quote
    _HAS_FRESH = True
except ImportError:
    _HAS_FRESH = False

try:
    from data_sources import get_realtime_price as _ds_realtime
    _HAS_REALTIME = True
except ImportError:
    _HAS_REALTIME = False
    _ds_realtime = None

from regime import volatility_regime, risk_on_off
from trader_profile import show_trader_selection, get_trader_profile, get_recommendation_style
from database_manager import get_database_manager
import traceback
from trade_signal_overlay import generate_model_trades, add_trade_markers_to_fig, get_trade_summary
from quant_ensemble_model import get_quant_ensemble

# ML libraries loaded lazily on first use
_ml_initialized = False
ML_AVAILABLE = False

def _ensure_ml():
    global _ml_initialized, ML_AVAILABLE
    if not _ml_initialized:
        ml_analysis.ensure_ml_libraries()
        ML_AVAILABLE = ml_analysis.SKLEARN_AVAILABLE and ml_analysis.XGBOOST_AVAILABLE
        _ml_initialized = True

# Optional imports
try:
    import options_engine
    HAS_OPTIONS = True
except ImportError:
    HAS_OPTIONS = False

try:
    from graph_analysis import analyze_price_chart, analyze_rsi_chart, analyze_volume_chart
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI without external ta library."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _compute_macd(close: pd.Series) -> tuple:
    """Compute MACD line, signal, histogram without external ta library."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _safe_col(df, col):
    """Safely extract a column that might be multi-level."""
    c = df[col]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c


@st.cache_data(ttl=300)
def fetch_market_data(symbol, asset_type, period, interval):
    """Cached data fetching to improve performance"""
    try:
        if asset_type == "Stock":
            df = get_stock(symbol, period=period, interval=interval)
        elif asset_type == "FX":
            # Handle different FX symbol formats
            fx_symbol = symbol.replace("=X", "").replace("-", "_").replace("/", "_")
            if "_" not in fx_symbol and len(fx_symbol) == 6:
                fx_symbol = f"{fx_symbol[:3]}_{fx_symbol[3:]}"
            df = get_fx(fx_symbol)
        elif asset_type == "Futures":
            df = get_futures_proxy(symbol, period=period, interval=interval)
        elif asset_type == "Crypto":
            df = get_stock(symbol, period=period, interval=interval)
        else:
            df = get_stock(symbol, period=period, interval=interval)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def _robust_fetch(symbol: str, period: str = "6mo", interval: str = "1d"):
    """Robust multi-fallback data fetcher for any asset type.
    Returns (df, asset_type) or (None, asset_type).
    """
    sym = symbol.strip().upper()

    is_intraday = interval not in ("1d", "5d", "1wk", "1mo", "3mo")

    #  Forex: USD/JPY, EUR/USD, USDJPY=X 
    if '/' in sym and '=' not in sym:
        yf_sym = sym.replace('/', '') + '=X'

        # For daily intervals, try cached sources first
        if not is_intraday and _HAS_FRESH:
            df = get_fresh_quote(yf_sym, period=period)
            if df is not None and not df.empty:
                return df, 'fx'

        if not is_intraday:
            fx_key = sym.replace('/', '_')
            for key in [fx_key, fx_key.lower(), fx_key.upper()]:
                try:
                    df = get_fx(key)
                    if df is not None and not df.empty:
                        return df, 'fx'
                except Exception:
                    pass

        # Direct yfinance with interval
        try:
            import yfinance as yf
            df = yf.Ticker(yf_sym).history(period=period, interval=interval)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                    if df.columns.duplicated().any():
                        df = df.loc[:, ~df.columns.duplicated(keep='first')]
                if "Close" in df.columns:
                    return df, 'fx'
        except Exception:
            pass

        return None, 'fx'

    #  Forex =X symbols 
    if sym.endswith('=X'):
        if not is_intraday and _HAS_FRESH:
            df = get_fresh_quote(sym, period=period)
            if df is not None and not df.empty:
                return df, 'fx'

        df = get_stock(sym, period=period, interval=interval)
        if df is not None and not df.empty:
            return df, 'fx'

        return None, 'fx'

    #  Futures 
    if '=F' in sym:
        if not is_intraday and _HAS_FRESH:
            df = get_fresh_quote(sym, period=period)
            if df is not None and not df.empty:
                return df, 'futures'

        df = get_futures_proxy(sym, period=period, interval=interval)
        if df is not None and not df.empty:
            return df, 'futures'

        df = get_stock(sym, period=period, interval=interval)
        if df is not None and not df.empty:
            return df, 'futures'

        return None, 'futures'

    #  Indices 
    if sym.startswith('^'):
        if not is_intraday and _HAS_FRESH:
            df = get_fresh_quote(sym, period=period)
            if df is not None and not df.empty:
                return df, 'index'

        df = get_stock(sym, period=period, interval=interval)
        if df is not None and not df.empty:
            return df, 'index'

        return None, 'index'

    #  Crypto / Stocks 
    if sym.endswith("-USD"):
        asset_type = 'crypto'
    else:
        asset_type = 'stock'

    if not is_intraday and _HAS_FRESH:
        df = get_fresh_quote(sym, period=period)
        if df is not None and not df.empty:
            return df, asset_type

    df = get_stock(sym, period=period, interval=interval)
    if df is not None and not df.empty:
        return df, asset_type

    # Last resort direct yfinance
    try:
        import yfinance as yf
        df = yf.Ticker(sym).history(period=period, interval=interval)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                if df.columns.duplicated().any():
                    df = df.loc[:, ~df.columns.duplicated(keep='first')]
            if "Close" in df.columns:
                return df, asset_type
    except Exception:
        pass

    return None, asset_type

def _has_valid_volume(df) -> bool:
    """Check if a dataframe has meaningful volume data (not all zero/NaN)."""
    if "Volume" not in df.columns:
        return False
    vol = df["Volume"]
    if isinstance(vol, pd.DataFrame):
        vol = vol.iloc[:, 0]
    vol = vol.dropna()
    if len(vol) == 0:
        return False
    # Check if all zeros or near-zero
    if vol.sum() == 0 or vol.max() < 1:
        return False
    return True

def show_custom_dashboard():
    _ensure_ml()
    st.title("Symbol Analysis")
    st.caption("Comprehensive multi-asset analysis with technicals, ML predictions, and risk metrics.")

    # --- Full-screen inline configuration panel ---
    _CD_TIMEFRAMES = {
        "1 Day (1m)": {"period": "1d", "interval": "1m"},
        "5 Days (5m)": {"period": "5d", "interval": "5m"},
        "1 Month (30m)": {"period": "1mo", "interval": "30m"},
        "3 Months (Daily)": {"period": "3mo", "interval": "1d"},
        "6 Months (Daily)": {"period": "6mo", "interval": "1d"},
        "1 Year (Daily)": {"period": "1y", "interval": "1d"},
        "2 Years (Daily)": {"period": "2y", "interval": "1d"},
        "5 Years (Weekly)": {"period": "5y", "interval": "1wk"},
        "All Time (Weekly)": {"period": "max", "interval": "1wk"},
    }

    cfg1, cfg2, cfg3, cfg4, cfg5 = st.columns([2, 1.5, 1.5, 1, 1])
    with cfg1:
        search_query = st.text_input(
            "Ticker / Symbol",
            placeholder="AAPL, USD/JPY, ES=F, BTC-USD",
            help="Enter stock ticker, FX pair (USD/JPY), futures (ES=F), or crypto (BTC-USD)",
            key="cd_symbol_input",
        )
    with cfg2:
        asset_type = st.selectbox("Asset Class", ["Auto-detect", "Stock", "FX", "Futures", "Crypto"], key="cd_asset")
    with cfg3:
        tf_label = st.selectbox("Timeframe", list(_CD_TIMEFRAMES.keys()), index=5, key="cd_tf")
        tf_cfg = _CD_TIMEFRAMES[tf_label]
        period = tf_cfg["period"]
        interval = tf_cfg["interval"]
    with cfg4:
        show_model_trades = st.toggle("Model Trades", value=False, key="cd_show_trades")
    with cfg5:
        analyze_btn = st.button("Analyze", type="primary", key="cd_analyze_btn", use_container_width=True)

    trade_strategy = "model_choice"
    if show_model_trades:
        strat_col1, strat_col2 = st.columns([2, 1])
        with strat_col1:
            trade_strategy = st.selectbox("Trade Strategy", ["model_choice", "combined", "rsi", "sma_cross", "macd"],
                                           index=0, key="cd_trade_strat",
                                           format_func=lambda x: {"model_choice": "Model's Choice (Full Freedom)",
                                                                     "combined": "Combined (RSI+SMA+MACD)",
                                                                     "rsi": "RSI Crossover",
                                                                     "sma_cross": "SMA 20/50 Cross",
                                                                     "macd": "MACD Cross"}[x])
        with strat_col2:
            sim_timeframe = st.selectbox("Simulation Period", 
                                          ["3 Months", "6 Months", "1 Year", "2 Years", "5 Years"],
                                          index=2, key="cd_sim_tf",
                                          help="Historical period for the model to simulate trades on")
            _sim_tf_map = {"3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
            st.session_state['sim_period'] = _sim_tf_map.get(sim_timeframe, "1y")
        
        if trade_strategy == "model_choice":
            st.caption("The model has complete freedom to decide entries, exits, sizing, and strategy based on its own analysis of the asset.")

    st.markdown("---")

    # Info banner if ML missing
    if not ML_AVAILABLE:
        st.warning("ML libraries missing. Running in technical-analysis-only mode.")

    # Handle Analysis Trigger
    if analyze_btn and search_query:
        st.session_state['analyze_symbol'] = search_query.strip().upper()
        st.session_state['analyze_asset_type'] = asset_type
        st.session_state['analyze_period'] = period
        st.session_state['analyze_interval'] = interval
        st.rerun()

    # Main Analysis View
    if 'analyze_symbol' in st.session_state and st.session_state['analyze_symbol']:
        symbol = st.session_state['analyze_symbol']
        asset_type = st.session_state.get('analyze_asset_type', 'Auto-detect')
        period = st.session_state.get('analyze_period', '1y')
        interval = st.session_state.get('analyze_interval', '1d')
        
        # Auto-detect logic
        if asset_type == "Auto-detect":
            if "_" in symbol or symbol.endswith("=X") or '/' in symbol:
                asset_type = "FX"
            elif "=F" in symbol:
                asset_type = "Futures"
            elif symbol.endswith("-USD"):
                asset_type = "Crypto"
            else:
                asset_type = "Stock"  # Default to stock if no other conditions match
        
        # Fetch data
        with st.spinner(f"Fetching data for {symbol}..."):
            df, fetched_asset_type = _robust_fetch(symbol, period=period, interval=interval)
        
        if df is None or df.empty:
            st.error(f"No data found for {symbol}")
            st.info("Tips:\n"
                    "- Forex: `USD/JPY`, `EUR/USD`, `GBPUSD=X`\n"
                    "- Futures: `ES=F`, `GC=F`, `CL=F`\n"
                    "- Crypto: `BTC-USD`, `ETH-USD`\n"
                    "- Stocks: `AAPL`, `MSFT`, `NVDA`\n"
                    "- Indices: `^GSPC`, `^IXIC`")
            return

        # Flatten multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated(keep='first')]

        # Extract close safely
        close = _safe_col(df, 'Close').dropna().astype(float)
        if len(close) < 5:
            st.error("Insufficient data (less than 5 bars).")
            return

        current_price = float(close.iloc[-1])
        # Use true previous trading day close from fast_info, not close.iloc[-2]
        prev_price = None
        if _HAS_REALTIME and _ds_realtime:
            try:
                _, _prev = _ds_realtime(symbol)
                if _prev and _prev > 0:
                    prev_price = float(_prev)
            except Exception:
                pass
        if prev_price is None:
            prev_price = float(close.iloc[-2]) if len(close) >= 2 else current_price
        pct_change = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0

        has_vol = _has_valid_volume(df)

        #  Live Price Ticker (auto-refresh without full reload) 
        @st.fragment(run_every=30)
        def _live_price_ticker():
            try:
                live_p = None
                if _HAS_FRESH:
                    ldf = get_fresh_quote(symbol, period="1d")
                    if ldf is not None and not ldf.empty and "Close" in ldf.columns:
                        lc = ldf["Close"]
                        if isinstance(lc, pd.DataFrame):
                            lc = lc.iloc[:, 0]
                        lc = lc.dropna().astype(float)
                        if len(lc) > 0:
                            live_p = float(lc.iloc[-1])
                if live_p is None:
                    live_p = current_price
                live_chg = ((live_p - prev_price) / prev_price * 100) if prev_price > 0 else 0
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Price", f"${live_p:,.4f}" if live_p < 10 else f"${live_p:,.2f}", f"{live_chg:+.2f}%")
                with mc2:
                    hi = float(close.max())
                    st.metric("Period High", f"${hi:,.4f}" if hi < 10 else f"${hi:,.2f}")
                with mc3:
                    lo = float(close.min())
                    st.metric("Period Low", f"${lo:,.4f}" if lo < 10 else f"${lo:,.2f}")
                with mc4:
                    total_ret = ((live_p / float(close.iloc[0])) - 1) * 100
                    st.metric(f"Period Return ({period})", f"{total_ret:+.2f}%")
            except Exception:
                pass

        st.markdown(f"### {symbol}  {asset_type}")
        _live_price_ticker()

        #  Key Statistics Row 
        try:
            import yfinance as _yf_stats
            _tk_info = _yf_stats.Ticker(symbol).info or {}
        except Exception:
            _tk_info = {}

        _52w_hi = _tk_info.get('fiftyTwoWeekHigh') or float(close.max())
        _52w_lo = _tk_info.get('fiftyTwoWeekLow') or float(close.min())
        _avg_vol = _tk_info.get('averageVolume') or (int(_safe_col(df, 'Volume').mean()) if has_vol else None)
        _ann_vol_quick = float(close.pct_change().dropna().std() * np.sqrt(252) * 100) if len(close) > 20 else None
        _mkt_cap = _tk_info.get('marketCap')
        _beta = _tk_info.get('beta')

        ks1, ks2, ks3, ks4, ks5, ks6 = st.columns(6)
        with ks1:
            st.metric("52-Wk High", f"${_52w_hi:,.2f}" if _52w_hi else "")
        with ks2:
            st.metric("52-Wk Low", f"${_52w_lo:,.2f}" if _52w_lo else "")
        with ks3:
            st.metric("Avg Volume", f"{_avg_vol:,.0f}" if _avg_vol else "N/A")
        with ks4:
            st.metric("Ann. Volatility", f"{_ann_vol_quick:.1f}%" if _ann_vol_quick else "")
        with ks5:
            st.metric("Market Cap", f"${_mkt_cap/1e9:.1f}B" if _mkt_cap and _mkt_cap > 0 else "")
        with ks6:
            st.metric("Beta", f"{_beta:.2f}" if _beta else "")

        # --- Volatility Indicator Panel ---
        try:
            _daily_rets = close.pct_change().dropna()
            if len(_daily_rets) >= 20:
                _rv_20 = float(_daily_rets.tail(20).std() * np.sqrt(252) * 100)
                _rv_60 = float(_daily_rets.tail(min(60, len(_daily_rets))).std() * np.sqrt(252) * 100) if len(_daily_rets) >= 60 else _rv_20
                _rv_full = float(_daily_rets.std() * np.sqrt(252) * 100)
                
                # Vol Z-Score: how current 20d vol compares to historical
                _vol_roll = _daily_rets.rolling(20).std() * np.sqrt(252) * 100
                _vol_mean = float(_vol_roll.dropna().mean())
                _vol_std = float(_vol_roll.dropna().std())
                _vol_zscore = (_rv_20 - _vol_mean) / max(_vol_std, 0.01)
                
                # Regime classification
                if _rv_20 < 12:
                    _vol_regime = "LOW"
                    _vol_color = "#00ff88"
                    _vol_desc = "Compressed — breakout potential"
                elif _rv_20 < 20:
                    _vol_regime = "NORMAL"
                    _vol_color = "#a2ffb3"
                    _vol_desc = "Standard market conditions"
                elif _rv_20 < 30:
                    _vol_regime = "ELEVATED"
                    _vol_color = "#ffaa00"
                    _vol_desc = "Heightened risk — size down"
                elif _rv_20 < 50:
                    _vol_regime = "HIGH"
                    _vol_color = "#ff6644"
                    _vol_desc = "Active risk management required"
                else:
                    _vol_regime = "EXTREME"
                    _vol_color = "#ff2222"
                    _vol_desc = "Crisis-level — capital preservation"
                
                st.markdown(
                    f'<div style="background:rgba(13,17,23,0.7);border:1px solid {_vol_color};border-radius:8px;'
                    f'padding:12px 16px;margin-bottom:15px;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">'
                    f'<div>'
                    f'<span style="color:#8b949e;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">Volatility Regime</span>'
                    f'<div style="color:{_vol_color};font-size:1.3rem;font-weight:800;margin-top:2px;">{_vol_regime}</div>'
                    f'<div style="color:#8b949e;font-size:0.75rem;">{_vol_desc}</div>'
                    f'</div>'
                    f'<div style="display:flex;gap:20px;">'
                    f'<div style="text-align:center;">'
                    f'<div style="color:#aaa;font-size:0.65rem;">20D RV</div>'
                    f'<div style="color:white;font-size:1.1rem;font-weight:700;">{_rv_20:.1f}%</div>'
                    f'</div>'
                    f'<div style="text-align:center;">'
                    f'<div style="color:#aaa;font-size:0.65rem;">60D RV</div>'
                    f'<div style="color:white;font-size:1.1rem;font-weight:700;">{_rv_60:.1f}%</div>'
                    f'</div>'
                    f'<div style="text-align:center;">'
                    f'<div style="color:#aaa;font-size:0.65rem;">Vol Z-Score</div>'
                    f'<div style="color:{"#ff4444" if _vol_zscore > 1.5 else "#00ff88" if _vol_zscore < -1 else "white"};font-size:1.1rem;font-weight:700;">{_vol_zscore:+.2f}</div>'
                    f'</div>'
                    f'<div style="text-align:center;">'
                    f'<div style="color:#aaa;font-size:0.65rem;">Hist Avg</div>'
                    f'<div style="color:white;font-size:1.1rem;font-weight:700;">{_rv_full:.1f}%</div>'
                    f'</div>'
                    f'</div></div></div>',
                    unsafe_allow_html=True
                )
        except Exception:
            pass

        st.markdown("---")

        #  Integrated AI Insights (from Dashboard) 
        try:
            from market_movers import _generate_ai_insights, _calculate_technicals
            # Calculate technicals for AI
            ai_tech = _calculate_technicals(df)
            ai_data = _generate_ai_insights(symbol, ai_tech, df, asset_type)
            
            # Display colored signal banner
            sig = ai_data.get("signal", "NEUTRAL")
            sig_color = "#00ff88" if "BULLISH" in sig else "#ff5252" if "BEARISH" in sig else "#8b949e"
            
            st.markdown(
                f'<div style="border-left: 4px solid {sig_color}; padding: 10px 15px; background: rgba(255,255,255,0.05); margin-bottom: 20px; border-radius: 4px;">'
                f'<strong style="color: {sig_color}; font-size: 1.1em;">AI SIGNAL: {sig}</strong>'
                f'<span style="margin-left: 15px; color: #ccc;">Probability: Bull {ai_data.get("bullish_prob",0):.0%} | Bear {ai_data.get("bearish_prob",0):.0%}</span>'
                f'<div style="margin-top: 5px; font-size: 0.9em; color: #ddd;">{ai_data.get("outlook", "")}</div>'
                f'</div>', 
                unsafe_allow_html=True
            )
            
            with st.expander("AI Analysis & Trade Ideas", expanded=True):
                col_ai1, col_ai2 = st.columns(2)
                with col_ai1:
                    st.markdown("**Analysis Insights**")
                    for insight in ai_data.get("insights", [])[:5]:
                        st.markdown(f"- {insight}")
                with col_ai2:
                    st.markdown("**Trade Setup Ideas**")
                    ideas = ai_data.get("trade_ideas", [])
                    if ideas:
                        for idea in ideas:
                            st.markdown(f"- {idea}")
                    else:
                        st.caption("No clear trade setup identified.")

        except ImportError:
            pass
        except Exception as e:
            st.error(f"AI Analysis Error: {e}")

        # --- FULL MODEL OUTLOOK (Global Integration) ---
        try:
            from quant_ensemble_model import get_quant_ensemble
            from market_movers import _generate_ai_insights, _calculate_technicals
            
            # 1. Get Fundamental/AI Consensus Score
            ai_tech = _calculate_technicals(df)
            ai_data = _generate_ai_insights(symbol, ai_tech, df, asset_type)
            bull_p = ai_data.get("bullish_prob", 0.5)
            bear_p = ai_data.get("bearish_prob", 0.5)
            # Normalize to 0-100 scale
            f_score = ((bull_p - bear_p) + 1) / 2 * 100
            
            # 2. Get Quant Signal
            qe = get_quant_ensemble()
            q_signal = qe.predict(prices=df, symbol=symbol, asset_type=asset_type)
            q_prob = q_signal.probability * 100
            
            # 3. Unified Conviction (Full Model)
            unified_score = (f_score + q_prob) / 2
            
            # Determine Outlook Label and Color
            if unified_score >= 80:
                outlook_label = "STRONG OVERWEIGHT"
                outlook_color = "#00ff88"  # Neon Green
            elif unified_score >= 60:
                outlook_label = "ACCUMULATE"
                outlook_color = "#a2ffb3"  # Soft Green
            elif unified_score >= 40:
                outlook_label = "NEUTRAL"
                outlook_color = "#8b949e"  # Gray
            elif unified_score >= 20:
                outlook_label = "REDUCE"
                outlook_color = "#ff9b9b"  # Soft Red
            else:
                outlook_label = "STRONG UNDERWEIGHT"
                outlook_color = "#ff5252"  # Bright Red

            st.write("")
            st.markdown(f"""
            <div style="background: rgba(13, 17, 23, 0.8); border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 25px;">
                <h4 style="margin-top: 0; color: #8b949e; font-size: 0.9rem; letter-spacing: 1px; text-transform: uppercase;">Full Model Outlook (Unified Conviction)</h4>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 style="margin: 0; color: {outlook_color}; font-size: 2.2rem; font-weight: 800;">{outlook_label}</h2>
                        <p style="margin: 5px 0 0 0; color: #8b949e; font-size: 0.95rem;">
                            Synthesized Signal: {q_signal.direction} ({q_signal.probability:.0%}) | Fundamental Bias: {f_score:.0f}/100
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2.8rem; font-weight: 800; color: white;">{unified_score:.1f}</div>
                        <div style="font-size: 0.75rem; color: #8b949e; text-transform: uppercase;">Conviction Index</div>
                    </div>
                </div>
                <div style="margin-top: 20px; border-top: 1px solid #30363d; padding-top: 15px;">
                    <p style="font-size: 0.9rem; color: #ddd; line-height: 1.5;">
                        <b>Model Reasoning:</b> {ai_data.get('outlook', 'Consensus bias aligned with quantitative direction.')} 
                        Quant model detects {q_signal.direction.lower()} momentum with {q_signal.confidence:.0%} engine confidence.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as ex:
            st.caption(f"Predictive engine offline for {symbol}")

        # Generate model trades if enabled
        trade_markers = []
        if show_model_trades and len(close) >= 30:
            trade_markers = generate_model_trades(close, df, trade_strategy)

        #  Support & Resistance Levels 
        def _pivot_levels(series, window=10):
            """Detect support/resistance from local highs/lows."""
            highs, lows = [], []
            arr = series.values
            for i in range(window, len(arr) - window):
                if arr[i] == max(arr[i - window:i + window + 1]):
                    highs.append(float(arr[i]))
                if arr[i] == min(arr[i - window:i + window + 1]):
                    lows.append(float(arr[i]))
            return sorted(set(highs), reverse=True)[:4], sorted(set(lows))[:4]

        res_levels, sup_levels = _pivot_levels(close)
        if res_levels or sup_levels:
            st.markdown("#### Key Support & Resistance Levels")
            sr1, sr2 = st.columns(2)
            with sr1:
                st.markdown("**Resistance Levels**")
                for i, lvl in enumerate(res_levels[:4], 1):
                    pct_away = ((lvl - current_price) / current_price) * 100
                    st.markdown(f"R{i}: **${lvl:,.2f}** ({pct_away:+.1f}% away)")
            with sr2:
                st.markdown("**Support Levels**")
                for i, lvl in enumerate(sup_levels[:4], 1):
                    pct_away = ((lvl - current_price) / current_price) * 100
                    st.markdown(f"S{i}: **${lvl:,.2f}** ({pct_away:+.1f}% away)")
            st.markdown("---")

        #  Tabs 
        tabs = st.tabs(["Price & Technicals", "Volume & Momentum", "Risk", "ML Predictions", "Options Intelligence", "Correlation"])

        #  Tab 1: Price & Technicals 
        with tabs[0]:
            # Moving Averages
            sma20 = close.rolling(20).mean()
            sma50 = close.rolling(min(50, len(close))).mean()
            ema20 = close.ewm(span=20, adjust=False).mean()

            # Only add volume subplot if volume data exists
            fig_price = make_subplots(rows=2 if has_vol else 1, cols=1, shared_xaxes=True,
                                    row_heights=[0.75, 0.25] if has_vol else [1.0],
                                    vertical_spacing=0.08)
            fig_price.add_trace(go.Scatter(x=close.index, y=close.values,
                mode='lines', name='Close', line=dict(color='white', width=2)), row=1, col=1)
            fig_price.add_trace(go.Scatter(x=sma20.index, y=sma20.values,
                mode='lines', name='SMA 20', line=dict(color='#42a5f5', width=1, dash='dot')), row=1, col=1)
            fig_price.add_trace(go.Scatter(x=sma50.index, y=sma50.values,
                mode='lines', name='SMA 50', line=dict(color='#ef5350', width=1, dash='dot')), row=1, col=1)
            fig_price.add_trace(go.Scatter(x=ema20.index, y=ema20.values,
                mode='lines', name='EMA 20', line=dict(color='#66bb6a', width=1, dash='dash')), row=1, col=1)

            # Bollinger Bands
            bb_mid = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            fig_price.add_trace(go.Scatter(x=bb_upper.index, y=bb_upper.values,
                mode='lines', name='BB Upper', line=dict(color='rgba(255,215,0,0.4)', width=1, dash='dot')), row=1, col=1)
            fig_price.add_trace(go.Scatter(x=bb_lower.index, y=bb_lower.values,
                mode='lines', name='BB Lower', line=dict(color='rgba(255,215,0,0.4)', width=1, dash='dot'),
                fill='tonexty', fillcolor='rgba(255,215,0,0.05)'), row=1, col=1)

            if has_vol:
                vol = _safe_col(df, 'Volume').dropna().astype(float)
                fig_price.add_trace(go.Bar(x=vol.index, y=vol.values,
                    name='Volume', marker_color='rgba(100,100,255,0.3)'), row=2, col=1)

            #  Add model trade markers 
            if show_model_trades and trade_markers:
                add_trade_markers_to_fig(fig_price, trade_markers, row=1, col=1)

            fig_price.update_layout(
                title=dict(text=f"{symbol} Price & Moving Averages", x=0.01, font=dict(size=14)),
                height=550 if show_model_trades else 480, template="plotly_dark", showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig_price, width='stretch')

            #  Trade summary if enabled 
            if show_model_trades and trade_markers:
                summary = get_trade_summary(trade_markers)
                if summary["total_trades"] > 0:
                    st.markdown("#### Model Trade Summary")
                    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
                    with tc1: st.metric("Trades", summary["total_trades"])
                    with tc2: st.metric("Win Rate", f"{summary['win_rate']:.0%}")
                    with tc3: st.metric("Avg Win", f"{summary['avg_win']:+.2%}")
                    with tc4: st.metric("Avg Loss", f"{summary['avg_loss']:+.2%}")
                    with tc5: st.metric("Total Return", f"{summary['total_return']:+.2%}")
                    tc6, tc7, tc8 = st.columns(3)
                    with tc6: st.metric("Best Trade", f"{summary['best_trade']:+.2%}")
                    with tc7: st.metric("Worst Trade", f"{summary['worst_trade']:+.2%}")
                    with tc8:
                        pf = summary['profit_factor']
                        st.metric("Profit Factor", f"{pf:.2f}" if pf < 100 else "Inf")
                    with st.expander("Trade Log"):
                        log_rows = []
                        for m in trade_markers:
                            row_data = {"Time": str(m.timestamp)[:10], "Action": m.side,
                                        "Price": f"${m.price:,.2f}", "Reason": m.reason}
                            row_data["P&L"] = f"{m.pnl_pct:+.2%} (${m.pnl_dollar:+,.2f})" if m.pnl_pct is not None else ""
                            log_rows.append(row_data)
                        st.dataframe(pd.DataFrame(log_rows), width='stretch', hide_index=True)

            # RSI
            rsi = _compute_rsi(close, 14)
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=rsi.index, y=rsi.values,
                mode='lines', name='RSI', line=dict(color='#ab47bc', width=2)))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",
                              annotation=dict(text="OB", showarrow=False, font=dict(size=10)))
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                              annotation=dict(text="OS", showarrow=False, font=dict(size=10)))
            fig_rsi.update_layout(
                title=dict(text=f"{symbol} RSI (14)", x=0.01, font=dict(size=13)),
                height=220, template="plotly_dark", yaxis_range=[0, 100],
                margin=dict(l=10, r=10, t=35, b=10), showlegend=False,
            )
            st.plotly_chart(fig_rsi, width='stretch')

            # MACD
            macd_line, signal_line, macd_hist = _compute_macd(close)
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=macd_line.index, y=macd_line.values,
                mode='lines', name='MACD', line=dict(color='cyan', width=2)))
            fig_macd.add_trace(go.Scatter(x=signal_line.index, y=signal_line.values,
                mode='lines', name='Signal', line=dict(color='magenta', width=1.5)))
            fig_macd.add_trace(go.Bar(x=macd_hist.index, y=macd_hist.values,
                name='Histogram', marker_color=['green' if v >= 0 else 'red' for v in macd_hist.values]))
            fig_macd.update_layout(
                title=dict(text=f"{symbol} MACD", x=0.01, font=dict(size=13)),
                height=220, template="plotly_dark",
                margin=dict(l=10, r=10, t=35, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
            )
            st.plotly_chart(fig_macd, width='stretch')

            # Key levels
            current_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50
            sma20_val = float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else 0
            sma50_val = float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else 0
            macd_val = float(macd_hist.iloc[-1]) if pd.notna(macd_hist.iloc[-1]) else 0
            lc1, lc2, lc3, lc4 = st.columns(4)
            with lc1: st.metric("RSI (14)", f"{current_rsi:.1f}", "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral")
            with lc2: st.metric("vs SMA20", f"{'Above' if current_price > sma20_val else 'Below'}", f"${sma20_val:,.2f}")
            with lc3: st.metric("vs SMA50", f"{'Above' if current_price > sma50_val else 'Below'}", f"${sma50_val:,.2f}")
            with lc4: st.metric("MACD Hist", f"{macd_val:.4f}", "Rising" if macd_val > 0 else "Falling")

        #  Tab 2: Volume & Momentum 
        with tabs[1]:
            if has_vol:
                vol = _safe_col(df, 'Volume').dropna().astype(float)
                vol_sma20 = vol.rolling(20).mean()
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(x=vol.index, y=vol.values, name='Volume', marker_color='rgba(100,149,237,0.5)'))
                fig_vol.add_trace(go.Scatter(x=vol_sma20.index, y=vol_sma20.values, mode='lines', name='Vol SMA 20', line=dict(color='orange', width=2)))
                fig_vol.update_layout(
                    title=dict(text=f"{symbol} Volume", x=0.01, font=dict(size=13)),
                    height=280, template="plotly_dark",
                    margin=dict(l=10, r=10, t=35, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
                )
                st.plotly_chart(fig_vol, width='stretch')
            else:
                st.info(f"Volume data is not available for {symbol}. "
                        f"This is normal for forex pairs (OTC), certain indices, and some futures contracts. "
                        f"Price-based indicators (RSI, MACD, SMAs) remain fully functional.")

            st.markdown("### Return Distribution")
            daily_rets = close.pct_change().dropna()
            if len(daily_rets) > 10:
                fig_dist = px.histogram(x=daily_rets.values * 100, nbins=50, title="Daily Return Distribution (%)", color_discrete_sequence=['#636EFA'])
                fig_dist.update_layout(template="plotly_dark", height=280, xaxis_title="Daily Return %", yaxis_title="Frequency",
                                       margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_dist, width='stretch')

        #  Tab 3: Risk 
        with tabs[2]:
            daily_rets = close.pct_change().dropna()
            if len(daily_rets) >= 20:
                ann_vol = float(daily_rets.std() * np.sqrt(252) * 100)
                var_95 = float(daily_rets.quantile(0.05) * 100)
                var_99 = float(daily_rets.quantile(0.01) * 100)
                cum_rets = (1 + daily_rets).cumprod()
                running_max = cum_rets.expanding().max()
                drawdown = (cum_rets - running_max) / running_max
                max_dd = float(drawdown.min() * 100)
                sharpe = float((daily_rets.mean() / daily_rets.std()) * np.sqrt(252)) if daily_rets.std() > 0 else 0
                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1: st.metric("Ann. Volatility", f"{ann_vol:.2f}%")
                with rc2: st.metric("VaR (95%)", f"{var_95:+.2f}%")
                with rc3: st.metric("Max Drawdown", f"{max_dd:.2f}%")
                with rc4: st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values * 100, mode='lines', fill='tozeroy', fillcolor='rgba(255,0,0,0.2)', line=dict(color='red', width=1), name='Drawdown'))
                fig_dd.update_layout(title=dict(text="Drawdown (%)", x=0.01, font=dict(size=13)), height=230, template="plotly_dark", yaxis_title="Drawdown %", margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_dd, width='stretch')
                rolling_vol = daily_rets.rolling(21).std() * np.sqrt(252) * 100
                fig_rv = go.Figure()
                fig_rv.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, mode='lines', name='21d Rolling Vol', line=dict(color='#42a5f5', width=2)))
                fig_rv.update_layout(title=dict(text="Rolling 21-Day Annualized Volatility (%)", x=0.01, font=dict(size=13)), height=230, template="plotly_dark", margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_rv, width='stretch')
            else:
                st.info("Insufficient data for risk analysis (need 20+ bars).")

        #  Tab 4: ML Predictions 
        with tabs[3]:
            if not ML_AVAILABLE:
                st.warning("ML libraries (scikit-learn, xgboost) not available.")
            elif len(close) < 60:
                st.warning("Need at least 60 bars for ML prediction.")
            else:
                try:
                    from sklearn.ensemble import RandomForestClassifier
                    df_ml = pd.DataFrame({'Close': close})
                    df_ml['Return'] = df_ml['Close'].pct_change()
                    df_ml['Direction'] = (df_ml['Return'] > 0).astype(int)
                    df_ml['Lag1'] = df_ml['Close'].shift(1)
                    df_ml['Lag2'] = df_ml['Close'].shift(2)
                    df_ml['Lag3'] = df_ml['Close'].shift(3)
                    df_ml['SMA5'] = df_ml['Close'].rolling(5).mean()
                    df_ml['SMA20'] = df_ml['Close'].rolling(20).mean()
                    df_ml['Vol5'] = df_ml['Return'].rolling(5).std()
                    df_ml.dropna(inplace=True)
                    features = ['Lag1', 'Lag2', 'Lag3', 'SMA5', 'SMA20', 'Vol5']
                    split = int(len(df_ml) * 0.8)
                    train = df_ml[:split]
                    test = df_ml[split:]
                    if len(train) > 20 and len(test) > 5:
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(train[features], train['Direction'])
                        test = test.copy()
                        test['Predicted'] = model.predict(test[features])
                        test['Strategy_Return'] = test['Predicted'].shift(1) * test['Return']
                        test['Cum_Strategy'] = (1 + test['Strategy_Return'].fillna(0)).cumprod()
                        test['Cum_Market'] = (1 + test['Return']).cumprod()
                        fig_ml = go.Figure()
                        fig_ml.add_trace(go.Scatter(x=test.index, y=test['Cum_Strategy'], mode='lines', name='ML Strategy', line=dict(color='#00e676', width=2)))
                        fig_ml.add_trace(go.Scatter(x=test.index, y=test['Cum_Market'], mode='lines', name='Buy & Hold', line=dict(color='#ff5252', width=2)))
                        fig_ml.update_layout(title=dict(text=f"{symbol} ML Strategy vs Buy & Hold", x=0.01, font=dict(size=13)), height=320, template="plotly_dark", yaxis_title="Cumulative Return", margin=dict(l=10, r=10, t=35, b=10))
                        st.plotly_chart(fig_ml, width='stretch')
                        accuracy = (test['Predicted'] == test['Direction']).mean()
                        strat_ret = float(test['Cum_Strategy'].iloc[-1] - 1) * 100
                        market_ret = float(test['Cum_Market'].iloc[-1] - 1) * 100
                        mlc1, mlc2, mlc3 = st.columns(3)
                        with mlc1: st.metric("Model Accuracy", f"{accuracy:.1%}")
                        with mlc2: st.metric("Strategy Return", f"{strat_ret:+.2f}%")
                        with mlc3: st.metric("Market Return", f"{market_ret:+.2f}%")
                    else:
                        st.warning("Insufficient data for train/test split.")
                except ImportError:
                    st.warning("scikit-learn not installed.")
                except Exception as e:
                    st.error(f"ML prediction error: {e}")

        #  Tab 5: Options Intelligence
        with tabs[4]:
            if not HAS_OPTIONS:
                st.warning("Options Analysis Module (options_engine.py) not found.")
            else:
                try:
                    oa = options_engine.get_options_engine()
                    
                    # Generate options analysis
                    opt_data = oa.analyze_options_opportunity(
                        symbol, df, ai_data, current_price, 
                        trader_type=get_trader_profile().get("trader_type", "Swing Trader")
                    )
                    
                    st.markdown(f"###  Options Strategy Optimizer for {symbol}")
                    st.caption("Derived from IV-HV parity, Expected Move (1 SD), and model directionality.")
                    
                    # Metrics Row
                    ocol1, ocol2, ocol3, ocol4 = st.columns(4)
                    ocol1.metric("Est. Implied Vol", f"{opt_data['iv_annual']:.1%}", help="Estimated using HV and volatility premium")
                    ocol2.metric("14D Expected Move", f"${opt_data['expected_move_14d']:.2f}", f"{opt_data['expected_move_14d']/current_price:.1%}")
                    ocol3.metric("Rec. Strategy", opt_data['recommended_strategy'])
                    ocol4.metric("Prob. ITM (Target)", f"{opt_data['prob_itm_target']:.1f}%")

                    # Strategy Card
                    with st.container(border=True):
                        st.markdown(f"#### Recommended Implementation: {opt_data['recommended_strategy']}")
                        st.markdown(f"**Playbook:** {opt_data['details']}")
                        st.markdown(f"**Confidence:** {opt_data['confidence']}/100")
                        
                        # Comparison Chart (Payoff)
                        try:
                            import options_engine
                            engine = options_engine.get_options_engine()
                            
                            # Simple payoff for recommended tactic
                            # (Mapping basic strategies to engine-readable legs)
                            legs = []
                            if opt_data['recommended_strategy'] == "Long Call":
                                legs = [{'strike': current_price * 1.02, 'type': 'call', 'side': 1, 'cost': current_price * 0.03}]
                            elif opt_data['recommended_strategy'] == "Long Put":
                                legs = [{'strike': current_price * 0.98, 'type': 'put', 'side': 1, 'cost': current_price *0.03}]
                            elif opt_data['recommended_strategy'] == "Bull Call Spread":
                                legs = [
                                    {'strike': current_price, 'type': 'call', 'side': 1, 'cost': current_price * 0.05},
                                    {'strike': current_price * 1.05, 'type': 'call', 'side': -1, 'cost': current_price * 0.02}
                                ]
                            elif opt_data['recommended_strategy'] == "Bear Put Spread":
                                legs = [
                                    {'strike': current_price, 'type': 'put', 'side': 1, 'cost': current_price * 0.05},
                                    {'strike': current_price * 0.95, 'type': 'put', 'side': -1, 'cost': current_price * 0.02}
                                ]
                            
                            if legs:
                                payoff_fig = engine.generate_strategy_payoff_diagram(legs, current_price)
                                st.plotly_chart(payoff_fig, use_container_width=True)
                        except Exception:
                            st.info("Interactive payoff diagram unavailable for this complexity level.")

                    # Volatility Regime
                    st.markdown("---")
                    v_regime = volatility_regime(df)
                    st.markdown(f"**Volatility Regime:** {v_regime}")
                    st.caption("Low IV regimes favor Long Gamma (Buying options), High IV favors Short Gamma (Selling premium).")
                    
                    # Options Chain Viewer (NEW)
                    st.markdown("---")
                    with st.expander(" View Real-time Options Chain", expanded=False):
                        try:
                            from data_sources import get_options_chain
                            chain_data = get_options_chain(symbol)
                            
                            if chain_data:
                                exp_list = chain_data.get('expirations', [])
                                selected_exp = st.selectbox("Expiration Date", exp_list)
                                
                                if selected_exp != chain_data['expiration']:
                                    chain_data = get_options_chain(symbol, expiration=selected_exp)
                                
                                c_chain, p_chain = st.columns(2)
                                
                                with c_chain:
                                    vol_col_c = 'volume' if 'volume' in chain_data['calls'].columns else 'vol' if 'vol' in chain_data['calls'].columns else None
                                    st.markdown("**Calls**")
                                    call_cols = ['strike', 'lastPrice', 'change', 'bid', 'ask']
                                    if vol_col_c: call_cols.append(vol_col_c)
                                    if 'openInterest' in chain_data['calls'].columns: call_cols.append('openInterest')
                                    df_calls = chain_data['calls'][[c for c in call_cols if c in chain_data['calls'].columns]]
                                    st.dataframe(df_calls, use_container_width=True, hide_index=True)
                                    
                                with p_chain:
                                    vol_col_p = 'volume' if 'volume' in chain_data['puts'].columns else 'vol' if 'vol' in chain_data['puts'].columns else None
                                    st.markdown("**Puts**")
                                    put_cols = ['strike', 'lastPrice', 'change', 'bid', 'ask']
                                    if vol_col_p: put_cols.append(vol_col_p)
                                    if 'openInterest' in chain_data['puts'].columns: put_cols.append('openInterest')
                                    df_puts = chain_data['puts'][[c for c in put_cols if c in chain_data['puts'].columns]]
                                    st.dataframe(df_puts, use_container_width=True, hide_index=True)
                            else:
                                st.info("No options chain available for this symbol.")
                        except Exception as e:
                            st.error(f"Failed to fetch options chain: {e}")
                except Exception as e:
                    st.error(f"Options analysis error: {e}")

        #  Tab 6: Correlation 
        with tabs[5]:
            st.markdown("### Correlation with Major Indices")
            benchmarks = {'SPY': 'S&P 500', 'QQQ': 'NASDAQ 100', 'DIA': 'Dow Jones'}
            corr_data = {}
            try:
                import yfinance as _yf_corr
                for bm_sym, bm_name in benchmarks.items():
                    if bm_sym.upper() == symbol.upper():
                        continue
                    bm_df = _yf_corr.Ticker(bm_sym).history(period=period)
                    if bm_df is not None and not bm_df.empty:
                        if isinstance(bm_df.columns, pd.MultiIndex):
                            bm_df.columns = bm_df.columns.get_level_values(0)
                        bm_close = bm_df['Close'].dropna().astype(float)
                        # Align on common dates
                        combined = pd.DataFrame({'Asset': close, bm_name: bm_close}).dropna()
                        if len(combined) > 20:
                            corr_val = combined['Asset'].pct_change().corr(combined[bm_name].pct_change())
                            corr_data[bm_name] = corr_val
            except Exception:
                pass

            if corr_data:
                cc_cols = st.columns(len(corr_data))
                for i, (bm_name, corr_val) in enumerate(corr_data.items()):
                    with cc_cols[i]:
                        color = '#00ff88' if corr_val > 0.5 else '#ff5252' if corr_val < -0.3 else '#ffff00'
                        st.metric(f"vs {bm_name}", f"{corr_val:.3f}")
                        st.caption("Strong +" if corr_val > 0.7 else "Moderate +" if corr_val > 0.3 else "Weak/Negative")
            else:
                st.info("Could not compute correlations for this asset.")

        #  Data Download (inside analysis block) 
        with st.expander("Download Data"):
            try:
                csv = df.to_csv().encode('utf-8')
                st.download_button("Download CSV", csv, f"{symbol}_data.csv", "text/csv", key="dl_csv")
            except Exception:
                st.info("No data to download.")

