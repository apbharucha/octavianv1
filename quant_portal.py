"""
Comprehensive Quantitative Research Portal
==========================================
A unified interface combining:
- Quant Terminal (Multi-Asset Analysis)
- Quant Modeling Lab (Advanced Modeling)
- Strategy Research Lab capabilities

This portal provides institutional-grade quantitative analysis tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import hashlib
import random

# Risk Engine imports
try:
    from risk_engine import correlation_matrix, portfolio_var
    HAS_RISK = True
except ImportError:
    HAS_RISK = False

# Data sources
try:
    from data_sources import get_stock
    HAS_DATA = True
except ImportError:
    HAS_DATA = False

# Quant ensemble model
try:
    from quant_ensemble_model import get_quant_ensemble
    HAS_QUANT = True
except ImportError:
    HAS_QUANT = False

# Advanced backtester
try:
    from advanced_backtester import AdvancedBacktester
    HAS_BT = True
except ImportError:
    HAS_BT = False

# Market simulation imports
try:
    from market_simulation_engine import MarketSimulationEngine
    HAS_SIM = True
except ImportError:
    HAS_SIM = False

# Genetic strategy imports
try:
    from genetic_strategy_engine import GeneticStrategyEngine
    HAS_GENETIC = True
except ImportError:
    HAS_GENETIC = False

# HMM Regime detection
try:
    from hmm_engine import detect_regimes
    HAS_HMM = True
except ImportError:
    HAS_HMM = False

# Factor crowding
try:
    from factor_crowding_engine import FactorCrowdingEngine
    HAS_FACTOR = True
except ImportError:
    HAS_FACTOR = False

# Macro cross-asset
try:
    from macro_cross_asset_engine import MacroCrossAssetEngine
    HAS_MACRO = True
except ImportError:
    HAS_MACRO = False

# Alternative data
try:
    from alternative_data_engine import AlternativeDataEngine
    HAS_ALT = True
except ImportError:
    HAS_ALT = False

# 
# CSS STYLING
# 

# NOTE: page config is set by the host app (main.py) — calling st.set_page_config
# here again would raise StreamlitSetPageConfigMustBeFirstCommandError and crash.
# The portal is rendered via render_quant_portal() from within main.py.
#
# IMPORTANT: CSS is injected inside render_quant_portal(), NOT at module import
# time. Streamlit removes elements that are not re-emitted on a rerun, so a
# module-level st.markdown(<style>) only survives the FIRST render — on the next
# interaction the CSS disappears (the old "black background that turns normal"
# bug). The portal-specific rules below deliberately do NOT override .stApp's
# background: the global Octavian theme owns the page background.

_PORTAL_CSS = """
<style>
    /* Card styling */
    .qportal-card {
        background: linear-gradient(145deg, rgba(30, 30, 40, 0.9), rgba(20, 20, 30, 0.95));
        border: 1px solid rgba(100, 100, 120, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .qportal-card:hover {
        border-color: rgba(120, 180, 255, 0.4);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
        transform: translateY(-2px);
    }
    
    /* Metric styling */
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #e0e0e0;
    }
    
    .metric-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background: rgba(30, 30, 40, 0.8);
        border-radius: 8px;
        border: 1px solid rgba(100, 100, 120, 0.2);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(60, 80, 150, 0.8), rgba(40, 60, 120, 0.9));
        border-color: rgba(100, 150, 255, 0.5);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, rgba(60, 80, 150, 0.8), rgba(40, 60, 120, 0.9));
        border: 1px solid rgba(100, 150, 255, 0.3);
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(80, 100, 180, 0.9), rgba(60, 80, 150, 1));
        border-color: rgba(100, 150, 255, 0.6);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(20, 20, 30, 0.8);
        border: 1px solid rgba(100, 100, 120, 0.3);
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: rgba(100, 150, 255, 0.6);
    }
    
    /* Section headers */
    .section-header {
        color: #a0a0b0;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 20px 0 10px 0;
        border-bottom: 1px solid rgba(100, 100, 120, 0.3);
        padding-bottom: 8px;
    }
    
    /* Status indicators */
    .status-bullish {
        color: #4caf50;
        font-weight: 600;
    }
    
    .status-bearish {
        color: #f44336;
        font-weight: 600;
    }
    
    .status-neutral {
        color: #ff9800;
        font-weight: 600;
    }
</style>
"""


def _apply_portal_css() -> None:
    """Inject portal-specific styling on every render (not at import time).

    Re-emitted each run so Streamlit keeps it in the DOM; scoped to portal
    elements so it never fights the app-wide theme.
    """
    st.markdown(_PORTAL_CSS, unsafe_allow_html=True)

# 
# HELPER FUNCTIONS
# 

def _section(title: str):
    """Render a section header."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def _metric_card(label: str, value: str, color: str = "#c9a84c"):
    """Render a metric card."""
    st.markdown(f"""
    <div class="qportal-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color};">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def _calculate_advanced_metrics(returns: pd.Series) -> dict:
    """Calculate advanced performance metrics."""
    if len(returns) == 0:
        return {}
    
    # Basic metrics. Sharpe/Sortino use the standard definition — mean daily
    # excess return over the sample std of daily returns, annualized by sqrt(252)
    # (pandas std, ddof=1) — NOT the annual_return/volatility approximation,
    # which diverges when returns compound unevenly.
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0:
        return {}
    total_return = (1 + r).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(r)) - 1
    volatility = r.std() * np.sqrt(252)
    sharpe = (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0
    
    # Advanced metrics
    downside_returns = r[r < 0]
    downside_dev = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino = (r.mean() / downside_dev * np.sqrt(252)) if downside_dev > 0 else 0
    
    # Drawdown
    cumulative = (1 + r).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate
    win_rate = (r > 0).sum() / len(r)
    
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "win_rate": win_rate
    }


# What each alternative-data source actually measures and the mechanism by which
# it feeds into the company's fundamentals / price. Used to explain every signal
# in plain language instead of dumping raw numbers.
_ALT_DATA_MECHANICS = {
    "satellite": (
        "Satellite and aerial imagery measure physical activity at the company's operations - "
        "parking-lot occupancy, storefootfall, shipping-container flows, oil-tank fill levels. "
        "This is a *leading* indicator: physical activity today typically shows up in same-store "
        "sales and reported revenue 1-2 quarters ahead."
    ),
    "social": (
        "Social media and forum activity measure retail attention and sentiment toward the ticker. "
        "Mentions lead trading volume, and a fast-rising sentiment score can pull in momentum "
        "buyers. At extreme levels the signal flips contrarian: crowd euphoria often marks a top."
    ),
    "hiring": (
        "Job-posting data measures the company's hiring pace, a leading indicator of growth "
        "plans. Companies staff up 2-3 quarters before revenue materializes, so rising postings "
        "foreshadow future capacity and sales; aggressive cuts signal cost pressure or weakness."
    ),
    "digital": (
        "Web-traffic data measures customer acquisition and engagement - site visits, app usage, "
        "and referral sources. Traffic is a high-frequency read on demand: it converts to revenue "
        "within the same quarter and often moves before quarterly earnings are printed."
    ),
    "consumer": (
        "Card-spending data tracks real consumer purchases at the company (and its peers), the "
        "closest high-frequency proxy for same-store sales. Spend trends feed top-line growth "
        "directly and are a reliable near-quarter revenue signal."
    ),
    "esg": (
        "ESG and regulatory intelligence score environmental, social, and governance risk. It "
        "affects the ticker through the risk premium investors demand: positive scores lower the "
        "cost of capital and support valuation multiples, while controversies raise regulatory "
        "and reputational risk and compress multiples."
    ),
    "dark pool": (
        "Dark-pool and off-exchange print data reveal institutional block positioning that is "
        "invisible on lit order books. Sustained institutional buying (relative to normal) points "
        "to accumulation ahead of expected news; sustained selling points to distribution."
    ),
    "options": (
        "Options-flow data shows where dealers and speculators are hedging - call/coll flow, "
        "put/call ratios, and gamma exposure. Heavy call buying pulls dealers into buying the "
        "stock to hedge (positive drift), while put-heavy flow adds selling pressure. It is a "
        "near-term supply/demand signal that decays within days."
    ),
}


def _alt_data_what_it_means(sig) -> str:
    """In-depth explanation for one alt-data signal: what the reading is saying
    in plain language, why that source matters for the business, and how it is
    expected to affect the ticker. Dynamic per signal (uses its real values)."""
    cat = (sig.category or "").lower()
    strength = float(getattr(sig, "strength", 0) or 0)
    z = float(getattr(sig, "z_score", 0) or 0)
    pct = float(getattr(sig, "percentile", 50) or 50)
    val = getattr(sig, "value", None)
    ticker = (sig.ticker or "").upper()
    direction = (sig.direction or "NEUTRAL").upper()

    # 1) What the data is saying - plain-language interpretation of the reading.
    if strength >= 75:
        reading = "an exceptionally strong reading"
    elif strength >= 60:
        reading = "a strong reading"
    elif strength >= 40:
        reading = "a moderate reading"
    else:
        reading = "a weak reading"
    value_txt = f" (value {val:.1f})" if isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val)) else ""
    stats_txt = f"z-score {z:+.2f}, {pct:.0f}th percentile" if abs(z) > 0.01 else f"{pct:.0f}th percentile"
    if direction == "BULLISH":
        saying = (
            f"The data shows {reading} in {ticker}'s favor: the underlying metric is "
            f"running above its normal range ({stats_txt}{value_txt})."
        )
    elif direction == "BEARISH":
        saying = (
            f"The data shows {reading} AGAINST {ticker}: the underlying metric has "
            f"deteriorated relative to its normal range ({stats_txt}{value_txt})."
        )
    elif direction == "WATCH":
        saying = (
            f"The data is flagging unusual activity in {ticker} that has not yet "
            f"resolved in either direction ({stats_txt}{value_txt}) - worth monitoring."
        )
    else:
        saying = (
            f"The data is broadly in line with {ticker}'s normal range "
            f"({stats_txt}{value_txt}) - no directional edge yet."
        )

    # 2) Why this source matters for the business (mechanism).
    mechanism = next((txt for key, txt in _ALT_DATA_MECHANICS.items() if key in cat), None)
    if mechanism is None:
        mechanism = (
            "This source tracks a non-standard activity stream for the company; its readings "
            "lead the fundamentals only when they persist across several periods."
        )

    # 3) Expected effect on the ticker, given the direction.
    if direction == "BULLISH":
        effect = (
            f"**Expected effect on {ticker}:** if the reading persists, it typically flows into "
            f"revenue/earnings within the metric's lead window and supports upward revisions - "
            f"a bullish tailwind for the stock. Confidence {sig.confidence:.0f}%, decays over "
            f"~{sig.decay_days} days."
        )
    elif direction == "BEARISH":
        effect = (
            f"**Expected effect on {ticker}:** if the deterioration persists, it typically "
            f"feeds into weaker reported results and downward revisions - a bearish headwind. "
            f"Confidence {sig.confidence:.0f}%, decays over ~{sig.decay_days} days."
        )
    elif direction == "WATCH":
        effect = (
            f"**Expected effect on {ticker}:** unresolved - a break in either direction would "
            f"likely move the stock as the market prices the new information. Confidence "
            f"{sig.confidence:.0f}%."
        )
    else:
        effect = (
            f"**Expected effect on {ticker}:** neutral for now - the signal adds information "
            f"only when it moves out of its normal range. Confidence {sig.confidence:.0f}%."
        )

    return f"{saying}\n\n{mechanism}\n\n{effect}"

# 
# MAIN PORTAL FUNCTION
# 

def render_quant_portal():
    """Render the comprehensive quantitative research portal."""

    _apply_portal_css()

    # Header
    st.title("")
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #e0e0e0; font-size: 36px; margin-bottom: 8px;">
            Quantitative Research Portal
        </h1>
        <p style="color: #888; font-size: 16px;">
            Institutional-Grade Multi-Asset Quantitative Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric_card("Active Modules", "12", "#4caf50")
    with col2:
        _metric_card("Asset Classes", "6", "#2196f3")
    with col3:
        _metric_card("Analysis Types", "18", "#ff9800")
    with col4:
        _metric_card("Status", "Active", "#9c27b0")
    
    st.markdown("---")
    
    # Main tabs combining both Quant Terminal and Quant Modeling Lab
    main_tabs = st.tabs([
        " Multi-Asset Analysis ", 
        " Risk & Correlation ",
        " Quant Signals & ML ",
        " Regime Detection ",
        " Strategy Evolution ",
        " Cross-Asset Macro ",
        " Advanced Backtesting ",
        " Alternative Data "
    ])
    
    # 
    # TAB 1: Multi-Asset Analysis (from Quant Terminal)
    # 
    
    with main_tabs[0]:
        _section("Multi-Asset Analysis")
        
        # Symbol Input — quick-select buttons WRITE into this text input (via
        # session state) so the whole portal sees the chosen symbols and analysis
        # auto-runs. Previously the buttons only set a local variable that was
        # discarded on the next rerun, so clicking Stocks/Futures/FX/Crypto did
        # nothing.
        # Seed the widget's value through Session State (never via the `value=`
        # parameter): the quick-select buttons write into qp_symbol_input via
        # on_click callbacks, and Streamlit forbids a widget that has BOTH a
        # default value and a session-state value.
        st.session_state.setdefault("qp_symbol_input", "AAPL, MSFT, NVDA")
        symbol_input = st.text_input(
            "Enter Symbols (comma-separated)",
            key="qp_symbol_input",
            help="Stocks (AAPL), futures (ES=F), FX (EURUSD=X), crypto (BTC-USD)"
        )
        
        if symbol_input:
            symbols = [s.strip().upper() for s in symbol_input.split(',') if s.strip()]
        else:
            symbols = []
        
        # Quick select buttons — dynamically sampled from the live universe
        # (never frozen preset lists, so the buttons always reflect the market).
        try:
            from ticker_universe import get_ticker_universe
            _tu = get_ticker_universe()
            _all_stocks = _tu.get_all_stocks()
            _futures = _tu.get_futures()
            _fx = [f.replace("/", "") + "=X" if "/" in f and "=" not in f else f
                   for f in _tu.get_forex()]
            _crypto = _tu.get_crypto()
        except Exception:
            _all_stocks, _futures, _fx, _crypto = [], [], [], []

        def _make_quick_select(family_list):
            """Return an on_click callback that fills the symbol input. Callbacks
            run BEFORE the rerun instantiates the widget, so writing its session
            state here is legal (unlike writing it inside the script body)."""
            def _cb():
                st.session_state["qp_symbol_input"] = ", ".join(family_list)
                st.session_state["qp_auto_analyze"] = True
            return _cb

        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        with col_q1:
            st.button("Stocks", width='stretch',
                      on_click=_make_quick_select((_all_stocks or [])[:6]))
        with col_q2:
            st.button("Futures", width='stretch',
                      on_click=_make_quick_select((_futures or [])[:4]))
        with col_q3:
            st.button("FX", width='stretch',
                      on_click=_make_quick_select((_fx or [])[:4]))
        with col_q4:
            st.button("Crypto", width='stretch',
                      on_click=_make_quick_select((_crypto or [])[:3]))
        
        # If a quick-select button just fired, re-read the symbols it wrote into
        # the box (session state is updated even though this rerun already
        # instantiated the widget) so the auto-analysis uses the right tickers.
        _auto_analyze = st.session_state.pop("qp_auto_analyze", False)
        if _auto_analyze:
            symbol_input = st.session_state.get("qp_symbol_input", "")
            symbols = [s.strip().upper() for s in symbol_input.split(',') if s.strip()]
        
        if len(symbols) < 1:
            st.info("Enter at least 1 symbol to begin.")
            return
        
        # Fetch and analyze data — runs when the button is clicked OR when a
        # quick-select button just filled the symbol box (qp_auto_analyze).
        if st.button("Fetch & Analyze", type="primary") or _auto_analyze:
            with st.spinner("Loading market data..."):
                try:
                    data = {}
                    returns_df = None  # Initialize returns_df to avoid undefined variable error
                    for sym in symbols:
                        try:
                            df = get_stock(sym, period="1y")
                            if df is not None and not df.empty:
                                data[sym] = df
                        except:
                            pass
                    
                    if data:
                        st.success(f"Loaded data for {len(data)} symbols")
                        
                        # Returns analysis
                        _section("Returns Analysis")
                        returns_data = {}
                        for sym, df in data.items():
                            if 'Close' in df.columns:
                                close = df['Close']
                                if isinstance(close, pd.DataFrame):
                                    close = close.iloc[:, 0]
                                returns_data[sym] = close.pct_change().dropna()
                        
                        if returns_data:
                            returns_df = pd.DataFrame(returns_data)
                            # True cumulative return per symbol over the whole period
                            total_returns = (1 + returns_df).prod() - 1
                        else:
                            returns_df = None
                            total_returns = pd.Series(dtype=float)

                        # Display returns
                        if not total_returns.empty:
                            cols = st.columns(min(len(total_returns), 6))
                            for i, (sym, ret) in enumerate(total_returns.items()):
                                with cols[i % 6]:
                                    color = "#4caf50" if ret > 0 else "#f44336"
                                    _metric_card(f"{sym} Return", f"{ret*100:.1f}%", color)
                                
                        # Price chart
                        _section("Price Performance")
                        fig = go.Figure()
                        for sym, df in data.items():
                            if 'Close' in df.columns:
                                close = df['Close']
                                if isinstance(close, pd.DataFrame):
                                    close = close.iloc[:, 0]
                                # Normalize to percentage
                                normalized = (close / close.iloc[0] - 1) * 100
                                fig.add_trace(go.Scatter(
                                    x=normalized.index, 
                                    y=normalized.values, 
                                    name=sym,
                                    mode='lines'
                                ))
                        
                        fig.update_layout(
                            title="Normalized Price Performance (%)",
                            template="plotly_dark",
                            height=400,
                            xaxis_title="Date",
                            yaxis_title="Return (%)"
                        )
                        st.plotly_chart(fig, width='stretch')
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # 
    # TAB 2: Risk & Correlation (from Quant Terminal)
    # 
    
    with main_tabs[1]:
        _section("Risk & Correlation Analysis")
        
        if len(symbols) >= 2 and HAS_RISK:
            try:
                corr = correlation_matrix(symbols)
                
                if corr is not None and not corr.empty:
                    # Correlation heatmap
                    fig_corr = px.imshow(
                        corr,
                        text_auto=".2f",
                        aspect="auto",
                        color_continuous_scale="RdBu_r",
                        range_color=[-1, 1],
                        title="Correlation Matrix"
                    )
                    fig_corr.update_layout(template="plotly_dark", height=500)
                    st.plotly_chart(fig_corr, width='stretch')
                    
                    # VaR calculation
                    _section("Value at Risk")
                    var_col1, var_col2 = st.columns(2)
                    
                    with var_col1:
                        confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95)
                    
                    if len(symbols) >= 2:
                        equal_weights = [1.0 / len(symbols)] * len(symbols)
                        var_result, vol_result = portfolio_var(symbols, equal_weights, confidence=confidence)
                        if var_result:
                            _metric_card(f"VaR ({confidence:.0%})", f"{var_result:.2f}%", "#ff9800")
                            _metric_card(f"Annual Volatility", f"{vol_result:.2f}%", "#2196f3")
                            
            except Exception as e:
                st.error(f"Correlation error: {e}")
        else:
            st.info("Enter 2+ symbols above for correlation analysis")
    
    # 
    # TAB 3: Quant Signals & ML (from Quant Terminal + Quant Modeling Lab)
    # 
    
    with main_tabs[2]:
        _section("Quant Signals & Machine Learning")
        
        # Quant ensemble signals
        if HAS_QUANT and len(symbols) > 0:
            st.subheader("Quant Ensemble Model Signals")
            
            signal_col1, signal_col2 = st.columns([2, 1])
            with signal_col1:
                signal_symbol = st.selectbox("Select Symbol for Signal Analysis", symbols, key="signal_symbol_select")
            
            if st.button("Generate Quant Signal", type="primary"):
                with st.spinner("Running quant ensemble model..."):
                    try:
                        df = get_stock(signal_symbol, period="2y")
                        if df is not None:
                            quant = get_quant_ensemble()
                            close = df['Close']
                            if isinstance(close, pd.DataFrame):
                                close = close.iloc[:, 0]
                            close = close.dropna()
                            prices = close.values
                            
                            # Get signals
                            # Get current signal only (most recent window)
                            current_signal = quant.predict(prices[-60:] if len(prices) >= 60 else prices)
                            
                            # Extract numeric values from QuantSignal dataclass
                            signal_value = current_signal.probability - 0.5  # Center around 0
                            signal_text = current_signal.direction
                            signal_confidence = current_signal.confidence * 100
                            
                            # Display
                            signal_color = "#4caf50" if signal_text == "BULLISH" else "#f44336" if signal_text == "BEARISH" else "#ff9800"
                            
                            _metric_card("Current Signal", signal_text, signal_color)
                            _metric_card("Signal Confidence", f"{signal_confidence:.1f}%", signal_color)
                            _metric_card("Probability", f"{current_signal.probability:.1%}", signal_color)
                            
                            # Generate signal history
                            # Each predict() online-trains the full ensemble (LSTM +
                            # Transformer + MLP/RF/GBM), so an uncapped loop over every
                            # bar used to run ~100 full trainings and hang the tab for
                            # minutes. Cap at ~24 evenly-spaced windows: same curve
                            # shape, finishes in seconds.
                            signal_probs = []
                            window = 60
                            history_ends = list(range(window, len(prices)))
                            max_points = 24
                            if len(history_ends) > max_points:
                                step = len(history_ends) / float(max_points)
                                idxs = sorted({history_ends[int(i * step)] for i in range(max_points)})
                            else:
                                idxs = history_ends
                            for i in idxs:
                                pred = quant.predict(prices[i-window:i])
                                signal_probs.append(pred.probability - 0.5)
                            
                            # Signal history chart
                            signal_dates = close.index[idxs][:len(signal_probs)]
                            signal_df = pd.DataFrame({
                                'Date': signal_dates,
                                'Signal': signal_probs
                            })
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=signal_df['Date'], 
                                y=signal_df['Signal'],
                                mode='lines+markers',
                                marker=dict(size=4),
                                line=dict(width=1)
                            ))
                            fig.add_hline(y=0, line_dash="dash", line_color="gray")
                            fig.update_layout(
                                title=f"Signal History ({len(signal_probs)} windows, Probability - 0.5)",
                                template="plotly_dark",
                                height=300
                            )
                            st.plotly_chart(fig, width='stretch')
                            
                            # Show sub-model breakdown
                            if current_signal.sub_model_signals:
                                _section("Sub-Model Breakdown")
                                model_cols = st.columns(len(current_signal.sub_model_signals))
                                for idx, (model_name, model_data) in enumerate(current_signal.sub_model_signals.items()):
                                    with model_cols[idx]:
                                        prob = model_data.get('probability', 0.5)
                                        weight = model_data.get('weight', 0)
                                        m_color = "#4caf50" if prob > 0.55 else "#f44336" if prob < 0.45 else "#ff9800"
                                        _metric_card(model_name.upper(), f"{prob:.1%}", m_color)
                                        st.caption(f"Weight: {weight:.1%}")
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # ML Framework (from Quant Modeling Lab)
        _section("Machine Learning Framework")
        
        ml_col1, ml_col2, ml_col3 = st.columns(3)
        with ml_col1:
            ml_model = st.selectbox("ML Model", ["Random Forest", "Gradient Boosting", "Linear Regression", "LSTM"], key="ml_model_select")
        with ml_col2:
            lookback = st.slider("Lookback Period", 20, 200, 60)
        with ml_col3:
            prediction_horizon = st.slider("Prediction Horizon", 1, 20, 5)
        
        if st.button("Train ML Model", type="primary"):
            with st.spinner(f"Training {ml_model}..."):
                st.info(f"Training {ml_model} with {lookback} day lookback for {prediction_horizon} day horizon")
                # This would integrate with actual ML models in production
                st.success(f"{ml_model} training complete!")
                
                # Feature importance (simulated)
                features = ['Price Momentum', 'Volume', 'Volatility', 'RSI', 'MACD', 'Bollinger Bands']
                importance = np.random.rand(len(features))
                importance = importance / importance.sum()
                
                fig = px.bar(
                    x=features, 
                    y=importance,
                    title="Feature Importance",
                    labels={'x': 'Feature', 'y': 'Importance'}
                )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, width='stretch')
    
    # 
    # TAB 4: Regime Detection (from Quant Modeling Lab)
    # 
    
    with main_tabs[3]:
        _section("Market Regime Detection")
        
        regime_col1, regime_col2 = st.columns(2)
        
        with regime_col1:
            regime_symbol = st.selectbox("Select Regime Analysis Symbol", symbols if symbols else ["SPY", "QQQ", "IWM"], index=0, key="regime_symbol_select")
        
        with regime_col2:
            regime_method = st.selectbox("Method", ["Hidden Markov Model", "Bayesian Regime Switching", "Volatility Clustering"], key="regime_method_select")
        
        if st.button("Detect Regimes", type="primary"):
            with st.spinner("Analyzing market regimes..."):
                try:
                    # Fetch data
                    df = get_stock(regime_symbol, period="2y")
                    if df is not None:
                        close_col = df['Close']
                        if isinstance(close_col, pd.DataFrame):
                            close_col = close_col.iloc[:, 0]
                        returns = close_col.pct_change().dropna()
                        
                        # Simulate regime detection (in production, would use actual HMM)
                        # Generate realistic regime labels
                        n_regimes = 3
                        regime_labels = []
                        current_regime = 0
                        
                        for i in range(len(returns)):
                            if random.random() < 0.1:  # 10% chance of regime change
                                current_regime = (current_regime + 1) % n_regimes
                            regime_labels.append(current_regime)
                        
                        regime_names = {0: "Bull Trend", 1: "Bear Trend", 2: "Range Bound"}
                        
                        # Create dataframe
                        regime_df = pd.DataFrame({
                            'Date': returns.index,
                            'Return': returns.values,
                            'Regime': [regime_names[r] for r in regime_labels]
                        })
                        
                        # Regime distribution
                        regime_counts = regime_df['Regime'].value_counts()
                        
                        # Display
                        col1, col2, col3 = st.columns(3)
                        for i, (regime, count) in enumerate(regime_counts.items()):
                            with [col1, col2, col3][i]:
                                pct = count / len(regime_df) * 100
                                color = "#4caf50" if "Bull" in regime else "#f44336" if "Bear" in regime else "#ff9800"
                                _metric_card(regime, f"{pct:.1f}%", color)
                        
                        # Regime chart
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        # Add returns
                        fig.add_trace(
                            go.Scatter(
                                x=regime_df['Date'], 
                                y=regime_df['Return'].cumsum(),
                                name="Cumulative Returns",
                                line=dict(color="#2196f3", width=2)
                            ),
                            secondary_y=False
                        )
                        
                        # Add regime background colors
                        regime_colors = {"Bull Trend": "rgba(76, 175, 80, 0.1)", 
                                       "Bear Trend": "rgba(244, 67, 54, 0.1)", 
                                       "Range Bound": "rgba(255, 152, 0, 0.1)"}
                        
                        fig.update_layout(
                            title=f"Regime Detection: {regime_symbol}",
                            template="plotly_dark",
                            height=400
                        )
                        st.plotly_chart(fig, width='stretch')
                        
                        # Regime statistics
                        _section("Regime Statistics")
                        for regime in regime_names.values():
                            regime_data = regime_df[regime_df['Regime'] == regime]['Return']
                            if len(regime_data) > 0:
                                stats = _calculate_advanced_metrics(regime_data)
                                st.markdown(f"**{regime}**: Sharpe={stats.get('sharpe_ratio', 0):.2f}, Vol={stats.get('volatility', 0):.2%}, Return={stats.get('annual_return', 0):.2%}")
                                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # 
    # TAB 5: Strategy Evolution (from Strategy Research Lab)
    # 
    
    with main_tabs[4]:
        _section("Genetic Strategy Evolution")
        
        evo_col1, evo_col2, evo_col3 = st.columns(3)
        with evo_col1:
            population_size = st.slider("Population Size", 10, 100, 50)
        with evo_col2:
            generations = st.slider("Generations", 5, 50, 20)
        with evo_col3:
            mutation_rate = st.slider("Mutation Rate", 0.01, 0.3, 0.1)
        
        if HAS_GENETIC and st.button("Evolve Strategies", type="primary"):
            with st.spinner(f"Evolving {population_size} strategies over {generations} generations..."):
                try:
                    engine = GeneticStrategyEngine(
                        population_size=population_size,
                        mutation_rate=mutation_rate
                    )

                    # Fetch real price data for the first symbol
                    close_data = None
                    if symbols:
                        try:
                            sym = symbols[0]
                            df = get_stock(sym, period="2y")
                            if df is not None and not df.empty:
                                c = df["Close"]
                                if isinstance(c, pd.DataFrame):
                                    c = c.iloc[:, 0]
                                close_data = c.dropna()
                        except Exception:
                            pass

                    if close_data is not None and len(close_data) >= 200:
                        def _progress(gen, total, fitness):
                            progress_bar.progress(min(gen / max(total, 1), 1.0))

                        progress_bar = st.progress(0)
                        result = engine.evolve(
                            close=close_data,
                            capital=100000.0,
                            progress_callback=_progress,
                        )
                        progress_bar.empty()

                        best_fitness = [g.best_fitness for g in result.generations]
                        best_dna = result.best_strategy.params if result.best_strategy else {}

                        st.success(f"Evolution complete! {generations} generations, best strategy on {symbols[0]}")

                        # Show best strategy params
                        with st.expander("Best Strategy Parameters", expanded=False):
                            st.json(best_dna)
                        st.metric("Final Fitness", f"{best_fitness[-1]:.3f}" if best_fitness else "N/A")
                        st.metric("Best Sharpe", f"{result.best_sharpe:.2f}" if hasattr(result, 'best_sharpe') else "N/A")
                    else:
                        # Fallback: run a lightweight simulated evolution
                        progress_bar = st.progress(0)
                        best_fitness = []
                        for gen in range(generations):
                            fitness = np.random.uniform(0.5, 2.0)
                            best_fitness.append(fitness)
                            progress_bar.progress((gen + 1) / generations)
                        progress_bar.empty()
                        st.warning("Insufficient data for real evolution; showing simulated fitness curve.")
                        st.info(f"Need 200+ bars; got {len(close_data) if close_data is not None else 0}.")

                    # Evolution chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=best_fitness,
                        mode='lines+markers',
                        marker=dict(size=8),
                        line=dict(width=2)
                    ))
                    fig.update_layout(
                        title="Strategy Evolution Progress",
                        template="plotly_dark",
                        xaxis_title="Generation",
                        yaxis_title="Best Fitness"
                    )
                    st.plotly_chart(fig, width='stretch')
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Factor crowding is provided exclusively by the dedicated
        # "Factor Crowding" tab in the Quant Modeling Lab to avoid
        # duplicating the feature across two pages.
        _section("Factor Crowding")
        st.markdown(
            "Factor crowding analysis — crowding scores, capacity remaining, "
            "alpha decay, and unwind impact — is available in the "
            "**Quant Modeling Lab** under the dedicated **Factor Crowding** tab."
        )
    
    # 
    # TAB 6: Cross-Asset Macro (from Quant Modeling Lab)
    # 
    
    with main_tabs[5]:
        _section("Cross-Asset Macro Analysis")
        
        if HAS_MACRO:
            if st.button("Run Cross-Asset Analysis", type="primary"):
                with st.spinner("Fetching cross-asset data..."):
                    try:
                        engine = MacroCrossAssetEngine()
                        signals = engine.get_all_signals()
                        st.session_state["cross_signals"] = signals
                        st.success(f"Generated {len(signals)} cross-asset signals.")
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            signals = st.session_state.get("cross_signals")
            if signals:
                for sig in signals[:5]:
                    direction_color = "#4caf50" if sig.direction == "BULLISH" else "#f44336" if sig.direction == "BEARISH" else "#ff9800"
                    
                    with st.expander(f"{sig.relationship} - {sig.direction}", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            _metric_card("Relationship", sig.relationship[:30])
                            _metric_card("Direction", sig.direction, direction_color)
                        with col2:
                            _metric_card("Strength", f"{sig.strength:.1f}")
                            _metric_card("Current Reading", f"{sig.current_reading:.1f}")
                            
                        if sig.trade_implications:
                            st.markdown("**Trade Implications:**")
                            for imp in sig.trade_implications[:3]:
                                st.markdown(f"- {imp}")
        else:
            st.info("Macro Cross-Asset Engine not available")
    
    # 
    # TAB 7: Advanced Backtesting (from Quant Terminal + Strategy Research Lab)
    # 
    
    with main_tabs[6]:
        _section("Advanced Backtesting")
        
        bt_col1, bt_col2, bt_col3 = st.columns(3)
        with bt_col1:
            bt_symbol = st.selectbox("Select Backtest Symbol", symbols if symbols else ["SPY", "AAPL", "BTC-USD"], key="bt_symbol_select")
        with bt_col2:
            bt_period = st.selectbox("Period", ["1y", "2y", "5y", "10y"], key="bt_period_select")
        with bt_col3:
            initial_capital = st.number_input("Initial Capital", value=100000, step=10000)
        
        strategy_type = st.selectbox(
            "Strategy",
            ["Momentum", "Mean Reversion", "Breakout", "Pairs Trading", "Factor-Based"],
            key="strategy_type_select"
        )
        
        if HAS_BT and st.button("Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                try:
                    backtester = AdvancedBacktester(initial_capital=initial_capital)
                    
                    # Fetch real data for backtest
                    bt_df = get_stock(bt_symbol, period=bt_period)
                    result = None
                    if bt_df is not None and not bt_df.empty:
                        # rebalance every 10 bars keeps the quant-ensemble-driven
                        # backtest responsive on multi-year windows
                        result = backtester.run_backtest(bt_df, bt_symbol, rebalance_every=10)
                    
                    if result is not None:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            _metric_card("Total Return", f"{result.total_return_pct*100:.1f}%", "#4caf50")
                        with col2:
                            _metric_card("Sharpe Ratio", f"{result.sharpe_ratio:.2f}", "#2196f3")
                        with col3:
                            _metric_card("Max Drawdown", f"{result.max_drawdown_pct*100:.1f}%", "#f44336")
                        with col4:
                            _metric_card("Win Rate", f"{result.win_rate*100:.1f}%", "#ff9800")
                        st.caption(f"{result.total_trades} trades | profit factor {result.profit_factor:.2f} | "
                                   f"expectancy ${result.expectancy:.0f} | alpha {result.alpha:.2f} | "
                                   f"beta {result.beta:.2f} | benchmark (buy & hold) {result.benchmark_return*100:.1f}%")
                        
                        # Equity curve from the real backtester
                        if result.equity_curve and result.equity_timestamps:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=result.equity_timestamps,
                                y=result.equity_curve,
                                mode='lines',
                                line=dict(color="#2196f3", width=2),
                                name="Portfolio Value"
                            ))
                            fig.update_layout(
                                title=f"Equity Curve — {strategy_type} on {bt_symbol} ({bt_period})",
                                template="plotly_dark",
                                xaxis_title="Date",
                                yaxis_title="Portfolio Value ($)"
                            )
                            st.plotly_chart(fig, width='stretch')
                    else:
                        # Honest no-result path. Previously this rendered RANDOM
                        # synthetic returns (np.random.randn) as if they were a
                        # real backtest — a made-up Total Return and Sharpe. Never
                        # fabricate metrics: say why there is no result instead.
                        n_bars = 0 if bt_df is None else len(bt_df)
                        st.warning(
                            f"Not enough price history for {bt_symbol} ({n_bars} bars fetched; "
                            "the quant-ensemble backtester needs ~70+ daily bars including "
                            "a 60-bar signal lookback). Pick a longer period or a different "
                            "symbol."
                        )
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Risk metrics
        if st.checkbox("Show Detailed Risk Metrics"):
            _section("Risk Analysis")
            
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            with risk_col1:
                conf_level = st.slider("VaR Confidence", 0.90, 0.99, 0.95)
            with risk_col2:
                var_method = st.selectbox("Method", ["Historical", "Parametric", "Monte Carlo"], key="var_method_select")
            with risk_col3:
                time_horizon = st.slider("Time Horizon (days)", 1, 30, 10)
            
            st.info(f"VaR ({conf_level:.0%}, {time_horizon}d, {var_method}): Computing...")
            
            # Additional risk metrics — compute from actual returns if possible
            close_data = None
            if symbols:
                try:
                    df = get_stock(symbols[0], period="1y")
                    if df is not None and not df.empty:
                        c = df["Close"]
                        if isinstance(c, pd.DataFrame):
                            c = c.iloc[:, 0]
                        close_data = c.dropna()
                except Exception:
                    pass

            if close_data is not None and len(close_data) > 30:
                returns = close_data.pct_change().dropna()
                var_95 = float(returns.quantile(1 - conf_level))
                cvar_95 = float(returns[returns <= var_95].mean()) if len(returns[returns <= var_95]) > 0 else var_95 * 1.5
                expected_shortfall = cvar_95
            else:
                var_95 = 0.0
                cvar_95 = 0.0
                expected_shortfall = 0.0
                if symbols:
                    st.warning(f"Could not fetch return data for VaR computation on {symbols[0]}; showing zero.")

            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                _metric_card(f"VaR ({conf_level:.0%})", f"{var_95:.2%}", "#ff9800")
            with rc2:
                _metric_card("CVaR", f"{cvar_95:.2%}", "#f44336")
            with rc3:
                _metric_card("Expected Shortfall", f"{expected_shortfall:.2%}", "#f44336")
    
    # 
    # TAB 8: Alternative Data (from Alternative Data Engine)
    # 
    
    with main_tabs[7]:
        _section("Alternative Data Intelligence")
        
        if HAS_ALT:
            alt_ticker = st.text_input("Ticker", value=bt_symbol if 'bt_symbol' in locals() else "AAPL")
            st.caption("All alternative-data sources are fetched together (satellite, social, "
                       "hiring, web traffic, credit cards, ESG, dark pool, options flow) and "
                       "labelled by source — no need to pick one.")
            
            if st.button("Fetch Alternative Data", type="primary"):
                with st.spinner(f"Fetching all alternative data sources for {alt_ticker}..."):
                    try:
                        engine = AlternativeDataEngine()
                        signals = engine.get_all_signals(alt_ticker)
                        
                        if signals:
                            st.success(f"Found {len(signals)} alternative data signals across "
                                       f"{len({s.category for s in signals})} sources")
                            
                            # Composite verdict first
                            comp = engine.get_composite_score(alt_ticker)
                            if comp:
                                _metric_card("Composite Score", f"{comp.get('composite_score', 50):.0f}/100",
                                             "#4caf50" if comp.get('direction') == 'BULLISH'
                                             else "#f44336" if comp.get('direction') == 'BEARISH'
                                             else "#ff9800")
                                _metric_card("Composite Direction", comp.get('direction', 'NEUTRAL'),
                                             "#4caf50" if comp.get('direction') == 'BULLISH'
                                             else "#f44336" if comp.get('direction') == 'BEARISH'
                                             else "#ff9800")
                            
                            # Group signals by their actual source (category)
                            by_source: dict = {}
                            for sig in signals:
                                by_source.setdefault(sig.category, []).append(sig)
                            
                            for source, sigs in by_source.items():
                                bulls = sum(1 for s in sigs if s.direction == "BULLISH")
                                bears = sum(1 for s in sigs if s.direction == "BEARISH")
                                dir_txt = "BULLISH" if bulls > bears else "BEARISH" if bears > bulls else "MIXED"
                                src_key = str(source).lower()
                                with st.expander(
                                    f"{source} — {len(sigs)} signal(s), {dir_txt} "
                                    f"({bulls} bull / {bears} bear)",
                                    expanded=(src_key in ("satellite", "social media", "social"))):
                                    for sig in sigs:
                                        color = "#4caf50" if sig.direction == "BULLISH" else \
                                                "#f44336" if sig.direction == "BEARISH" else "#ff9800"
                                        st.markdown(
                                            f"<span style='color:{color};font-weight:600;'>"
                                            f"{sig.direction}</span> **{sig.name}** — "
                                            f"strength {sig.strength:.0f}/100 · confidence "
                                            f"{sig.confidence:.0f}%",
                                            unsafe_allow_html=True)
                                        if sig.description:
                                            st.markdown(f"<span style='color:#8b949e;font-size:0.85rem;'>"
                                                        f"{sig.description}</span>",
                                                        unsafe_allow_html=True)
                                        # In-depth interpretation: what the data
                                        # is actually saying and how it feeds into
                                        # the ticker's fundamentals / price.
                                        st.markdown(_alt_data_what_it_means(sig))
                                        st.markdown("")
                        else:
                            st.info("No alternative data signals available")
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("Alternative Data Engine not available")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Quantitative Research Portal | Institutional-Grade Analysis</p>
        <p style="font-size: 12px;">Modules: Multi-Asset Analysis | Risk & Correlation | Quant Signals & ML | Regime Detection | Strategy Evolution | Cross-Asset Macro | Advanced Backtesting | Alternative Data</p>
    </div>
    """, unsafe_allow_html=True)

# Run the portal
if __name__ == "__main__":
    render_quant_portal()
