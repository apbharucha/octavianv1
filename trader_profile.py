"""
Octavian Trader Profile System  Comprehensive Personalization Engine
Author: APB - Octavian Team

Collects detailed trader preferences and tailors the entire app experience.
"""

import streamlit as st
import json
import os
import html as _html
from datetime import datetime
from typing import Dict, List, Optional, Any

from timeframe_analysis_engine import TimeframeScope, TimeframeAnalysisEngine

# --- Profile data management ---

PROFILE_PATH = os.path.join(os.path.dirname(__file__), "trader_profiles")
os.makedirs(PROFILE_PATH, exist_ok=True)

# 
# PROFILE SCHEMA & DEFAULTS
# 

EXPERIENCE_LEVELS = ["Beginner", "Intermediate", "Advanced", "Professional", "Institutional"]

TRADING_STYLES = {
    "Scalper": {"timeframe": "1m-15m", "hold": "seconds to minutes", "icon": ""},
    "Day Trader": {"timeframe": "5m-1h", "hold": "minutes to hours", "icon": ""},
    "Swing Trader": {"timeframe": "1h-1d", "hold": "days to weeks", "icon": ""},
    "Position Trader": {"timeframe": "1d-1w", "hold": "weeks to months", "icon": ""},
    "Long-Term Investor": {"timeframe": "1w-1mo", "hold": "months to years", "icon": ""},
}

RISK_PROFILES = {
    "Conservative": {"max_risk_pct": 1.0, "max_positions": 3, "color": "", "desc": "Capital preservation first"},
    "Moderate": {"max_risk_pct": 2.0, "max_positions": 5, "color": "[NOTE]", "desc": "Balanced risk/reward"},
    "Aggressive": {"max_risk_pct": 5.0, "max_positions": 10, "color": "#ff9800", "desc": "Growth-focused, higher drawdowns OK"},
    "Very Aggressive": {"max_risk_pct": 10.0, "max_positions": 20, "color": "", "desc": "Maximum returns, high volatility tolerance"},
}

ASSET_CLASSES = {
    "US Stocks": {"symbols": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"], "icon": ""},
    "International Stocks": {"symbols": ["BABA", "TSM", "ASML", "NVO", "SHOP", "SE"], "icon": ""},
    "ETFs": {"symbols": ["SPY", "QQQ", "IWM", "VTI", "ARKK", "XLK", "XLF"], "icon": ""},
    "Forex": {"symbols": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"], "icon": ""},
    "Crypto": {"symbols": ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD"], "icon": ""},
    "Futures": {"symbols": ["ES=F", "NQ=F", "CL=F", "GC=F", "SI=F"], "icon": ""},
    "Options": {"symbols": ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"], "icon": ""},
    "Commodities": {"symbols": ["GLD", "SLV", "USO", "UNG", "DBA"], "icon": "[COIN]"},
    "Bonds/Fixed Income": {"symbols": ["TLT", "IEF", "LQD", "HYG", "TIP"], "icon": ""},
}

SECTORS_OF_INTEREST = [
    "Technology", "Healthcare", "Financials", "Energy", "Consumer Discretionary",
    "Consumer Staples", "Industrials", "Materials", "Real Estate", "Utilities",
    "Communication Services", "AI & Robotics", "Clean Energy", "Biotech",
    "Semiconductors", "Cybersecurity", "EV & Autonomous", "Space & Defense",
]

PREFERRED_INDICATORS = [
    "RSI", "MACD", "Bollinger Bands", "Moving Averages (SMA/EMA)",
    "Volume Profile", "VWAP", "Fibonacci Retracements", "Ichimoku Cloud",
    "Stochastic Oscillator", "ATR", "OBV", "ADX", "Pivot Points",
    "Keltner Channels", "Parabolic SAR", "Williams %R",
]

TRADING_GOALS = [
    "Consistent income (weekly/monthly)",
    "Long-term wealth building",
    "Quick profits (short-term trades)",
    "Portfolio hedging",
    "Learning & skill development",
    "Retirement savings",
    "Beat the S&P 500",
    "Generate alpha with options",
    "Diversify across asset classes",
    "Build a dividend portfolio",
]

NEWS_PREFERENCES = [
    "Earnings reports", "Fed/Central bank decisions", "Macro economics",
    "Geopolitical events", "Sector rotation", "Insider trading",
    "Options flow", "Short interest", "IPOs & SPACs", "Crypto regulation",
    "AI/Tech developments", "Commodity supply/demand",
]

DEFAULT_PROFILE = {
    "name": "",
    "experience_level": "Intermediate",
    "trading_style": "Swing Trader",
    "risk_profile": "Moderate",
    "asset_classes": ["US Stocks", "ETFs"],
    "sectors_of_interest": ["Technology"],
    "watchlist": ["SPY", "QQQ", "AAPL", "NVDA"],
    "preferred_indicators": ["RSI", "MACD", "Moving Averages (SMA/EMA)"],
    "trading_goals": ["Long-term wealth building"],
    "news_preferences": ["Earnings reports", "Fed/Central bank decisions"],
    # Expanded Fields
    "portfolio_size": 10000.0,
    "time_horizon": "1-3 Years",
    "current_holdings": "", # JSON string or text
    "model_sensitivity": "Medium", # Low, Medium, High
    "notification_rules": {
        "price_alerts": True,
        "regime_change": True,
        "model_signals": False,
        "news_impact": False
    },
    # Existing
    "capital_range": "$10,000 - $50,000",
    "max_loss_per_trade_pct": 2.0,
    "profit_target_pct": 5.0,
    "preferred_session": "US Market Hours",
    "alerts_enabled": True,
    "show_advanced_metrics": False,
    "dark_pool_alerts": False,
    "options_greeks": False,
    "ai_commentary_style": "balanced",
    "adaptive_mode": False, # New: Model learns and adapts site
    "learned_persona": {},   # Stored learned data
    "created_at": None,
    "updated_at": None,
    "interaction_count": 0,
    "queries_history": [],
}

CAPITAL_RANGES = [
    "Under $1,000",
    "$1,000 - $5,000",
    "$5,000 - $10,000",
    "$10,000 - $50,000",
    "$50,000 - $100,000",
    "$100,000 - $500,000",
    "$500,000 - $1,000,000",
    "$1,000,000+",
]

TIME_HORIZONS = [
    "Scalp (Minutes)",
    "Intraday (Hours)",
    "Swing (Days-Weeks)",
    "Position (Weeks-Months)",
    "1-3 Years",
    "5+ Years",
    "Retirement"
]

MODEL_SENSITIVITIES = ["Low (High Confidence Only)", "Medium (Balanced)", "High (More Signals)"]

TRADING_SESSIONS = [
    "Pre-Market (4:00-9:30 ET)",
    "US Market Hours (9:30-16:00 ET)",
    "After-Hours (16:00-20:00 ET)",
    "Asian Session (19:00-4:00 ET)",
    "European Session (3:00-12:00 ET)",
    "24/7 (Crypto)",
    "All Sessions",
]

AI_STYLES = {
    "conservative": "Focus on risk management, downside protection, and high-probability setups only.",
    "balanced": "Balanced view of risk and reward. Present both bull and bear cases.",
    "aggressive": "Focus on highest-return opportunities. Emphasize momentum and breakout trades.",
    "educational": "Explain reasoning in detail. Great for learning.",
    "institutional": "Concise, data-dense. Assume advanced knowledge.",
}

WATCHLIST_PRESETS = {
    " Top Tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA"],
    " Major ETFs": ["SPY", "QQQ", "IWM", "VTI", "DIA", "GLD", "TLT"],
    " Crypto Majors": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD"],
    " Forex Majors": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"],
    " Futures": ["ES=F", "NQ=F", "CL=F", "GC=F", "SI=F"],
    " Biotech": ["LLY", "MRNA", "PFE", "ABBV", "GILD", "REGN"],
    " Energy": ["XOM", "CVX", "COP", "SLB", "OXY"],
    " Financials": ["JPM", "BAC", "GS", "MS", "BLK", "SCHW"],
    " AI & Semis": ["NVDA", "AMD", "AVGO", "SMCI", "AI", "PLTR", "CRWD"],
}


_BOND_ETF_SET = {
    "TLT", "IEF", "LQD", "HYG", "TIP", "SHY", "AGG", "EMB",
    "BND", "GOVT", "MBB", "VMBS", "MUB", "BSV", "VCSH",
}

# Watchlist presets map onto live universe sectors / asset classes.
_PRESET_SOURCES = {
    " Top Tech": ("sector", "technology"),
    " Major ETFs": ("etfs",),
    " Crypto Majors": ("crypto",),
    " Forex Majors": ("forex",),
    " Futures": ("futures",),
    " Biotech": ("sector", "healthcare"),
    " Energy": ("sector", "energy"),
    " Financials": ("sector", "financials"),
    " AI & Semis": ("sector", "semiconductors"),
}


def get_asset_classes() -> Dict[str, Dict[str, Any]]:
    """Asset-class catalog with example symbols drawn from the dynamic universe.

    Falls back to the static catalog above only when the universe is unavailable.
    """
    try:
        from ticker_universe import get_ticker_universe
        u = get_ticker_universe()
        stocks = sorted(u.get_all_stocks())
        etfs = sorted(u.get_etfs())
        forex = sorted(u.get_forex())
        crypto = sorted(u.get_crypto())
        futures = sorted(u.get_futures())
        bond_etfs = [e for e in etfs if e in _BOND_ETF_SET]
        symbols = {
            "US Stocks": stocks[:7],
            "International Stocks": [s for s in stocks if len(s) >= 4][:6],
            "ETFs": etfs[:7],
            "Forex": forex[:5],
            "Crypto": crypto[:5],
            "Futures": futures[:5],
            "Options": stocks[:5],
            "Commodities": futures[:5],
            "Bonds/Fixed Income": bond_etfs[:5] or etfs[:5],
        }
        return {
            name: {"symbols": symbols.get(name, []), "icon": meta["icon"]}
            for name, meta in ASSET_CLASSES.items()
        }
    except Exception:
        return dict(ASSET_CLASSES)


def get_watchlist_presets() -> Dict[str, List[str]]:
    """Watchlist presets populated from the dynamic universe (sectors/asset classes)."""
    try:
        from ticker_universe import get_ticker_universe
        u = get_ticker_universe()
        presets: Dict[str, List[str]] = {}
        for name, src in _PRESET_SOURCES.items():
            if src[0] == "sector":
                tickers = sorted(u.get_sector_tickers(src[1]))
            elif src[0] == "etfs":
                tickers = sorted(u.get_etfs())
            elif src[0] == "crypto":
                tickers = sorted(u.get_crypto())
            elif src[0] == "forex":
                tickers = sorted(u.get_forex())
            else:
                tickers = sorted(u.get_futures())
            presets[name] = tickers[:7]
        return presets or dict(WATCHLIST_PRESETS)
    except Exception:
        return dict(WATCHLIST_PRESETS)


# 
# PROFILE PERSISTENCE
# 

def _profile_file(profile_id: str = "default") -> str:
    return os.path.join(PROFILE_PATH, f"{profile_id}.json")


def save_profile(profile: Dict, profile_id: str = None):
    if profile_id is None:
        profile_id = st.session_state.get("username", "default")
    profile["updated_at"] = datetime.now().isoformat()
    if not profile.get("created_at"):
        profile["created_at"] = profile["updated_at"]
    with open(_profile_file(profile_id), "w") as f:
        json.dump(profile, f, indent=2)
    st.session_state["trader_profile"] = profile


def load_profile(profile_id: str = "default") -> Dict:
    path = _profile_file(profile_id)
    if os.path.exists(path):
        try:
            with open(path) as f:
                loaded = json.load(f)
            # Merge with defaults for any new fields
            merged = {**DEFAULT_PROFILE, **loaded}
            return merged
        except Exception:
            pass
    return {**DEFAULT_PROFILE}


def get_trader_profile() -> Dict:
    username = st.session_state.get("username", "default")
    
    # Reload if the profile memory is missing or belongs to a different logged-in user
    if "trader_profile" not in st.session_state or st.session_state.get("_active_profile_username") != username:
        st.session_state["trader_profile"] = load_profile(username)
        st.session_state["_active_profile_username"] = username
        
    return st.session_state["trader_profile"]


def list_profiles() -> List[str]:
    profiles = []
    for f in os.listdir(PROFILE_PATH):
        if f.endswith(".json"):
            profiles.append(f.replace(".json", ""))
    return profiles or ["default"]


# 
# INTERACTION LEARNING
# 

def learn_from_user_interaction(query: str, response: Any = None):
    """Track user queries to learn preferences over time."""
    profile = get_trader_profile()
    profile["interaction_count"] = profile.get("interaction_count", 0) + 1
    history = profile.get("queries_history", [])
    history.append({
        "query": query[:200],
        "timestamp": datetime.now().isoformat(),
    })
    # Keep last 100 queries
    profile["queries_history"] = history[-100:]
    save_profile(profile)


def get_timeframe_context_for_analysis() -> Dict:
    """Get timeframe context based on trader profile."""
    profile = get_trader_profile()
    style = profile.get("trading_style", "Swing Trader")
    style_info = TRADING_STYLES.get(style, TRADING_STYLES["Swing Trader"])

    from timeframe_analysis_engine import TimeframeScope
    style_to_scope = {
        "Scalper": TimeframeScope.SCALPING,
        "Day Trader": TimeframeScope.INTRADAY,
        "Swing Trader": TimeframeScope.SWING,
        "Position Trader": TimeframeScope.POSITION,
        "Long-Term Investor": TimeframeScope.INVESTMENT,
    }
    return {
        "primary_timeframe": style_to_scope.get(style, TimeframeScope.SWING),
        "style": style,
        "hold_period": style_info["hold"],
    }


def get_recommendation_style() -> str:
    """Get the AI commentary style preference."""
    profile = get_trader_profile()
    return profile.get("ai_commentary_style", "balanced")


def get_watchlist() -> List[str]:
    """Get user's watchlist symbols."""
    profile = get_trader_profile()
    return profile.get("watchlist", ["SPY", "QQQ", "AAPL"])


def get_risk_params() -> Dict:
    """Get risk parameters for position sizing and alerts."""
    profile = get_trader_profile()
    risk_name = profile.get("risk_profile", "Moderate")
    risk_info = RISK_PROFILES.get(risk_name, RISK_PROFILES["Moderate"])
    return {
        "profile": risk_name,
        "max_risk_pct": risk_info["max_risk_pct"],
        "max_positions": risk_info["max_positions"],
        "max_loss_per_trade_pct": profile.get("max_loss_per_trade_pct", 2.0),
        "profit_target_pct": profile.get("profit_target_pct", 5.0),
        "capital_range": profile.get("capital_range", "$10,000 - $50,000"),
    }


def get_preferred_assets() -> List[str]:
    """Get flat list of preferred symbols based on selected asset classes."""
    profile = get_trader_profile()
    classes = profile.get("asset_classes", ["US Stocks"])
    symbols = []
    _asset_classes = get_asset_classes()
    for cls in classes:
        if cls in _asset_classes:
            symbols.extend(_asset_classes[cls]["symbols"])
    # Add watchlist
    symbols.extend(profile.get("watchlist", []))
    return list(dict.fromkeys(symbols))  # Dedupe preserving order


def should_show_advanced() -> bool:
    profile = get_trader_profile()
    return profile.get("show_advanced_metrics", False) or \
           profile.get("experience_level", "Intermediate") in ["Advanced", "Professional", "Institutional"]


# 
# SIDEBAR WIDGET (compact, for every page)
# 

def show_trader_selection(key_suffix: str = ""):
    """Compact sidebar widget showing current profile summary."""
    profile = get_trader_profile()
    style = profile.get("trading_style", "Swing Trader")
    style_info = TRADING_STYLES.get(style, TRADING_STYLES["Swing Trader"])
    risk = profile.get("risk_profile", "Moderate")
    risk_info = RISK_PROFILES.get(risk, RISK_PROFILES["Moderate"])
    exp = profile.get("experience_level", "Intermediate")

    st.sidebar.markdown(
        f"**{style_info['icon']} {style}** | {risk_info['color']} {risk} | {exp}"
    )
    wl = profile.get("watchlist", [])
    if wl:
        st.sidebar.caption(f"Watchlist: {', '.join(wl[:6])}")


# 
# FULL PROFILE SETTINGS PAGE
# 

def _safe_index(lst: list, value, default: int = 0) -> int:
    """Safely get index of value in list, returning default if not found."""
    try:
        return lst.index(value)
    except ValueError:
        # Try partial match (e.g. "US Market Hours" matches "US Market Hours (9:30-16:00 ET)")
        val_lower = str(value).lower()
        for i, item in enumerate(lst):
            if val_lower in str(item).lower() or str(item).lower() in val_lower:
                return i
        return default


def _render_watchlist_editor(profile: Dict) -> List[str]:
    """Rich watchlist editor with categories, presets, and validation."""
    st.subheader("[PIN] Watchlist Management")

    current_wl = list(profile.get("watchlist", ["SPY", "QQQ", "AAPL"]))

    # Show current watchlist with remove buttons
    if current_wl:
        st.markdown(f"**Current Watchlist** ({len(current_wl)} symbols)")
        # Display in rows of 6
        rows = [current_wl[i:i+6] for i in range(0, len(current_wl), 6)]
        remove_syms = []
        for row_idx, row in enumerate(rows):
            cols = st.columns(len(row))
            for col, sym in zip(cols, row):
                with col:
                    # Color code by type
                    if sym.endswith("-USD"):
                        icon = ""
                    elif "/" in sym:
                        icon = ""
                    elif "=F" in sym:
                        icon = ""
                    elif sym.startswith("^"):
                        icon = ""
                    else:
                        icon = ""
                    if st.button(f" {icon} {sym}", key=f"wl_rm_{sym}_{row_idx}",
                                 width='stretch'):
                        remove_syms.append(sym)
        if remove_syms:
            for sym in remove_syms:
                if sym in current_wl:
                    current_wl.remove(sym)
            profile["watchlist"] = current_wl
            save_profile(profile)
            st.rerun()
    else:
        st.info("Your watchlist is empty. Add symbols below.")

    st.markdown("---")

    # Add symbols
    col_add1, col_add2 = st.columns([3, 1])
    with col_add1:
        new_syms = st.text_input(
            "Add Symbols (comma-separated)",
            placeholder="AAPL, USD/JPY, BTC-USD, ES=F",
            key="wl_add_input",
            help="Stocks (AAPL), Forex (USD/JPY or EURUSD=X), Crypto (BTC-USD), Futures (ES=F)"
        )
    with col_add2:
        st.markdown("<br>", unsafe_allow_html=True)
        add_clicked = st.button(" Add", key="wl_add_btn", type="primary", width='stretch')

    if add_clicked and new_syms:
        added = False
        for s in new_syms.split(","):
            s = s.strip().upper()
            if s and s not in current_wl:
                current_wl.append(s)
                added = True
        if added:
            profile["watchlist"] = current_wl
            save_profile(profile)
            st.rerun()

    # Preset quick-add
    st.markdown("**Quick Add Presets:**")
    preset_cols = st.columns(3)
    for idx, (preset_name, preset_syms) in enumerate(get_watchlist_presets().items()):
        with preset_cols[idx % 3]:
            if st.button(preset_name, key=f"wl_preset_{idx}", width='stretch'):
                added = False
                for s in preset_syms:
                    if s not in current_wl:
                        current_wl.append(s)
                        added = True
                if added:
                    profile["watchlist"] = current_wl
                    save_profile(profile)
                    st.rerun()

    # Clear all
    if current_wl and st.button(" Clear Entire Watchlist", key="wl_clear"):
        current_wl.clear()
        profile["watchlist"] = current_wl
        save_profile(profile)
        st.rerun()

    return current_wl


def show_profile_settings():
    """Full profile configuration page."""
    st.title(" Trader Profile & Preferences")
    st.markdown("Configure your trading profile to personalize your Octavian experience.")

    profile = get_trader_profile()

    # Profile selector
    profiles = list_profiles()
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        active = st.selectbox("Active Profile", profiles, index=0, key="profile_select")
    with col_p2:
        new_name = st.text_input("New Profile", key="new_profile_name", placeholder="my_profile")
        if st.button(" Create", key="create_profile") and new_name:
            save_profile({**DEFAULT_PROFILE, "name": new_name}, new_name.strip().lower().replace(" ", "_"))
            st.success(f"Profile '{new_name}' created!")
            st.rerun()

    if active != "default":
        profile = load_profile(active)
        st.session_state["trader_profile"] = profile

    st.markdown("---")

    #  Section 1: Identity & Experience 
    st.subheader(" Experience & Style")
    col1, col2, col3 = st.columns(3)
    with col1:
        profile["name"] = st.text_input("Display Name", value=profile.get("name", ""), key="prof_name")
    with col2:
        profile["experience_level"] = st.selectbox(
            "Experience Level", EXPERIENCE_LEVELS,
            index=_safe_index(EXPERIENCE_LEVELS, profile.get("experience_level", "Intermediate"), 1),
            key="prof_exp"
        )
    with col3:
        styles_list = list(TRADING_STYLES.keys())
        profile["trading_style"] = st.selectbox(
            "Trading Style", styles_list,
            index=_safe_index(styles_list, profile.get("trading_style", "Swing Trader"), 2),
            key="prof_style",
            format_func=lambda x: f"{TRADING_STYLES[x]['icon']} {x} ({TRADING_STYLES[x]['hold']})"
        )

    col4, col5 = st.columns(2)
    with col4:
        profile["time_horizon"] = st.selectbox(
            "Time Horizon", TIME_HORIZONS,
            index=_safe_index(TIME_HORIZONS, profile.get("time_horizon", "1-3 Years"), 4),
            key="prof_horizon"
        )
    with col5:
        # Portfolio Size (Numeric Input)
        profile["portfolio_size"] = st.number_input(
            "Portfolio Size ($)", min_value=0.0, step=1000.0,
            value=float(profile.get("portfolio_size", 10000.0)),
            format="%.2f", key="prof_size_val"
        )

    # Show style details
    style_info = TRADING_STYLES[profile["trading_style"]]
    st.info(f"**{style_info['icon']} {profile['trading_style']}**  "
            f"Typical timeframe: {style_info['timeframe']} | Hold period: {style_info['hold']}")

    #  Section 2: Risk Management 
    st.markdown("---")
    st.subheader(" Risk Management")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        risk_list = list(RISK_PROFILES.keys())
        profile["risk_profile"] = st.selectbox(
            "Risk Tolerance", risk_list,
            index=_safe_index(risk_list, profile.get("risk_profile", "Moderate"), 1),
            key="prof_risk",
            format_func=lambda x: f"{RISK_PROFILES[x]['color']} {x}  {RISK_PROFILES[x]['desc']}"
        )
    with col_r2:
        profile["capital_range"] = st.selectbox(
            "Trading Capital", CAPITAL_RANGES,
            index=_safe_index(CAPITAL_RANGES, profile.get("capital_range", "$10,000 - $50,000"), 3),
            key="prof_capital"
        )
    with col_r3:
        profile["preferred_session"] = st.selectbox(
            "Trading Session", TRADING_SESSIONS,
            index=_safe_index(TRADING_SESSIONS, profile.get("preferred_session", "US Market Hours"), 1),
            key="prof_session"
        )

    col_r4, col_r5 = st.columns(2)
    with col_r4:
        profile["max_loss_per_trade_pct"] = st.slider(
            "Max Loss Per Trade (%)", 0.5, 20.0,
            value=float(profile.get("max_loss_per_trade_pct", 2.0)),
            step=0.5, key="prof_max_loss"
        )
    with col_r5:
        profile["profit_target_pct"] = st.slider(
            "Profit Target Per Trade (%)", 1.0, 50.0,
            value=float(profile.get("profit_target_pct", 5.0)),
            step=1.0, key="prof_target"
        )

    risk_info = RISK_PROFILES[profile["risk_profile"]]
    rr_ratio = profile["profit_target_pct"] / max(profile["max_loss_per_trade_pct"], 0.1)
    st.caption(f"Risk/Reward Ratio: **{rr_ratio:.1f}:1** | "
               f"Max concurrent positions: **{risk_info['max_positions']}** | "
               f"Max risk per trade: **{risk_info['max_risk_pct']}%**")

    #  Section 3: Watchlist (NEW ENHANCED) 
    st.markdown("---")
    profile["watchlist"] = _render_watchlist_editor(profile)

    #  Section 4: Asset Preferences 
    st.markdown("---")
    st.subheader(" Asset Classes & Sectors")

    _asset_classes = get_asset_classes()
    profile["asset_classes"] = st.multiselect(
        "Preferred Asset Classes",
        list(_asset_classes.keys()),
        default=profile.get("asset_classes", ["US Stocks", "ETFs"]),
        format_func=lambda x: f"{_asset_classes[x]['icon']} {x}",
        key="prof_assets"
    )

    profile["sectors_of_interest"] = st.multiselect(
        "Sectors of Interest",
        SECTORS_OF_INTEREST,
        default=profile.get("sectors_of_interest", ["Technology"]),
        key="prof_sectors"
    )

    #  Section 5: Analysis Preferences 
    st.markdown("---")
    st.subheader(" Analysis & Indicator Preferences")

    profile["preferred_indicators"] = st.multiselect(
        "Preferred Technical Indicators",
        PREFERRED_INDICATORS,
        default=profile.get("preferred_indicators", ["RSI", "MACD", "Moving Averages (SMA/EMA)"]),
        key="prof_indicators"
    )

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        profile["show_advanced_metrics"] = st.toggle(
            "Show Advanced Metrics (Greeks, Sharpe, Sortino, etc.)",
            value=profile.get("show_advanced_metrics", False),
            key="prof_advanced"
        )
        profile["dark_pool_alerts"] = st.toggle(
            "Dark Pool / Institutional Flow Alerts",
            value=profile.get("dark_pool_alerts", False),
            key="prof_darkpool"
        )
    with col_a2:
        profile["options_greeks"] = st.toggle(
            "Show Options Greeks & Chain",
            value=profile.get("options_greeks", False),
            key="prof_greeks"
        )
        profile["alerts_enabled"] = st.toggle(
            "Enable Price & Signal Alerts",
            value=profile.get("alerts_enabled", True),
            key="prof_alerts"
        )
        profile["adaptive_mode"] = st.toggle(
            "Adaptive Terminal Mode (Beta)",
            value=profile.get("adaptive_mode", False),
            help="Enable AI to learn from your behavior and tailor the terminal's focus, tone, and recommendations.",
            key="prof_adaptive"
        )

    #  Section 6: Goals & AI Style 
    st.markdown("---")
    st.subheader(" Goals, Sensitivity & Holdings")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        profile["model_sensitivity"] = st.selectbox(
            "Model Sensitivity", MODEL_SENSITIVITIES,
            index=_safe_index(MODEL_SENSITIVITIES, profile.get("model_sensitivity", "Medium"), 1),
            key="prof_sensitivity"
        )
    with col_g2:
        profile["trading_goals"] = st.multiselect(
            "Trading Goals", TRADING_GOALS,
            default=profile.get("trading_goals", ["Long-term wealth building"]),
            key="prof_goals"
        )

    # Current Holdings (Simple Text Area for now)
    profile["current_holdings"] = st.text_area(
        "Current Holdings (Optional - for Context)",
        value=profile.get("current_holdings", ""),
        placeholder="e.g. 100 AAPL @ 150, 50 MSFT @ 300...",
        help="Paste a summary of your positions here so the AI can give context-aware advice."
    )

    st.markdown("---")
    st.subheader(" Notification Rules")
    
    notif_rules = profile.get("notification_rules", {})
    c_n1, c_n2, c_n3, c_n4 = st.columns(4)
    with c_n1:
        notif_rules["price_alerts"] = st.checkbox("Price Alerts", value=notif_rules.get("price_alerts", True), key="n_price")
    with c_n2:
        notif_rules["regime_change"] = st.checkbox("Regime Change", value=notif_rules.get("regime_change", True), key="n_regime")
    with c_n3:
        notif_rules["model_signals"] = st.checkbox("Model Signals", value=notif_rules.get("model_signals", False), key="n_model")
    with c_n4:
        notif_rules["news_impact"] = st.checkbox("High Impact News", value=notif_rules.get("news_impact", False), key="n_news")
    profile["notification_rules"] = notif_rules

    profile["news_preferences"] = st.multiselect(
        "News & Event Preferences",
        NEWS_PREFERENCES,
        default=profile.get("news_preferences", ["Earnings reports", "Fed/Central bank decisions"]),
        key="prof_news"
    )

    ai_styles_list = list(AI_STYLES.keys())
    profile["ai_commentary_style"] = st.selectbox(
        "AI Commentary Style",
        ai_styles_list,
        index=_safe_index(ai_styles_list, profile.get("ai_commentary_style", "balanced"), 1),
        key="prof_ai_style",
        format_func=lambda x: f"{x.title()}  {AI_STYLES[x]}"
    )

    #  Save 
    st.markdown("---")
    col_s1, col_s2 = st.columns([1, 4])
    with col_s1:
        if st.button(" Save Profile", type="primary", key="save_profile", width='stretch'):
            save_profile(profile, active)
            st.success(" Profile saved!")
            st.rerun()
    with col_s2:
        if profile.get("updated_at"):
            st.caption(f"Last saved: {profile['updated_at'][:19]} | "
                       f"Interactions: {profile.get('interaction_count', 0)}")

    #  Profile Summary Card 
    st.markdown("---")
    st.subheader(" Profile Summary")
    _render_profile_card(profile)


def _render_profile_card(profile: Dict):
    """Render a visual profile summary card."""
    style = profile.get("trading_style", "Swing Trader")
    style_info = TRADING_STYLES.get(style, TRADING_STYLES["Swing Trader"])
    risk = profile.get("risk_profile", "Moderate")
    risk_info = RISK_PROFILES.get(risk, RISK_PROFILES["Moderate"])

    name = profile.get("name") or "Trader"
    wl = profile.get("watchlist", [])
    goals = profile.get("trading_goals", [])

    st.markdown(f"""
| Field | Value |
|-------|-------|
| **Name** | {name} |
| **Experience** | {profile.get('experience_level', 'Intermediate')} |
| **Style** | {style_info['icon']} {style} ({style_info['hold']}) |
| **Risk** | {risk_info['color']} {risk}  {risk_info['desc']} |
| **Capital** | {profile.get('capital_range', 'N/A')} |
| **Session** | {profile.get('preferred_session', 'US Market Hours')} |
| **Max Loss/Trade** | {profile.get('max_loss_per_trade_pct', 2.0)}% |
| **Profit Target** | {profile.get('profit_target_pct', 5.0)}% |
| **Watchlist** | {', '.join(wl[:8]) if wl else 'None'} |
| **Goals** | {', '.join(goals[:3]) if goals else 'None set'} |
| **AI Style** | {profile.get('ai_commentary_style', 'balanced').title()} |
""")


# 
# MARKET-AWARE PERSONALIZED INSIGHTS
# 

def _fetch_watchlist_signal_snapshot(watchlist: List[str], max_symbols: int = 6) -> List[Dict]:
    """Lightweight RSI/momentum snapshot of the watchlist for insight generation."""
    import concurrent.futures
    from watchlist_dashboard import _fetch_symbol_data, _safe_close, _compute_technicals

    def _snap(sym: str) -> Optional[Dict]:
        try:
            df, _at = _fetch_symbol_data(sym, "1mo")
            if df is None or df.empty:
                return None
            close = _safe_close(df)
            if close is None or len(close) < 5:
                return None
            tech = _compute_technicals(close, df)
            price = float(close.iloc[-1])
            prev = float(close.iloc[-2]) if len(close) >= 2 else price
            return {"symbol": sym, "price": price, "change": (price / prev - 1) * 100, **tech}
        except Exception:
            return None

    results: List[Dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_snap, s): s for s in (watchlist or [])[:max_symbols]}
        done, _ = concurrent.futures.wait(futures, timeout=12)
        for f in done:
            try:
                r = f.result(timeout=1)
                if r:
                    results.append(r)
            except Exception:
                pass
    return results


@st.cache_data(ttl=300, show_spinner=False)
def _cached_market_state() -> tuple:
    """Live market state used by the insight cards (bias, regime, conviction,
    vol forecast, VIX). Cached 5 minutes so the dashboard never re-fetches
    VIX or re-runs the master engine on every rerun — this was the single
    largest per-rerun load-time cost.
    """
    market_bias = "NEUTRAL"
    market_regime = "Unknown"
    conviction = 0.5
    vol_forecast = 0.20
    try:
        from master_strategy_engine import get_master_engine
        import concurrent.futures as _cf

        def _fetch_outlook():
            return get_master_engine().get_dominant_outlook()

        with _cf.ThreadPoolExecutor(1) as pool:
            _fut = pool.submit(_fetch_outlook)
            _outlook = _fut.result(timeout=6)  # never block the dashboard
        market_bias = getattr(_outlook, "bias", "NEUTRAL") or "NEUTRAL"
        market_regime = getattr(_outlook, "regime", "Unknown") or "Unknown"
        conviction = float(getattr(_outlook, "conviction", 0.5) or 0.5)
        vol_forecast = float(getattr(_outlook, "volatility_forecast", 0.20) or 0.20)
    except Exception:
        # Fast fallback: adaptive engine's cached regime (no extra network work)
        try:
            from adaptive_reasoning_engine import get_adaptive_engine
            _regime = getattr(get_adaptive_engine(), "market_regime", "NEUTRAL")
            if _regime == "BULL_TREND":
                market_bias = "BULLISH"
            elif _regime == "BEAR_TREND":
                market_bias = "BEARISH"
        except Exception:
            pass

    vix: Optional[float] = None
    try:
        from data_sources import get_vix
        vix_df = get_vix(period="6mo")
        if vix_df is not None and not vix_df.empty:
            close = vix_df["Close"]
            if hasattr(close, "columns"):
                close = close.iloc[:, 0]
            vix = float(close.dropna().iloc[-1])
    except Exception:
        vix = None

    return market_bias, market_regime, conviction, vol_forecast, vix


def generate_market_aware_insights(watchlist_signals: Optional[List[Dict]] = None) -> List[Dict]:
    """Generate 4-6 market-aware personalized insight cards.

    Each card is a dict: {'title', 'body', 'urgency'} with urgency in
    high/medium/low. Built from: live VIX level, master-engine market
    bias/regime, the user's trading style and risk profile, and current
    watchlist signals (RSI, momentum). Replaces the old static fallback tips.
    """
    insights: List[Dict] = []
    profile = get_trader_profile()
    style = profile.get("trading_style", "Swing Trader")
    risk = profile.get("risk_profile", "Moderate")
    watchlist = profile.get("watchlist", [])

    # ── 1. Live market state (master strategy engine + VIX) — cached 5 min
    market_bias, market_regime, conviction, vol_forecast, vix = _cached_market_state()

    bias_lower = (market_bias or "NEUTRAL").upper()
    regime_lower = (market_regime or "").lower()
    # Only claim consolidation when the regime is explicitly range-bound/choppy;
    # an unknown regime (engine unavailable) should not be assumed ranging.
    is_ranging = ("range" in regime_lower or "choppy" in regime_lower)

    # ── 2. Volatility / position-sizing card (always) ──────────────────────
    max_loss = float(profile.get("max_loss_per_trade_pct", 2.0))
    if vix is not None and vix > 25:
        insights.append({
            "title": f"VIX Elevated ({vix:.1f}) — Cut Position Sizes",
            "body": (f"Elevated implied volatility means wider daily ranges. With a {risk} risk profile, "
                     f"reduce per-trade risk toward {max(0.5, max_loss * 0.5):.1f}% to keep portfolio "
                     "volatility constant."),
            "urgency": "high",
        })
    elif vix is not None and vix < 15:
        insights.append({
            "title": f"Low-Volatility Regime (VIX {vix:.1f})",
            "body": ("Options are historically cheap when VIX compresses below 15. Defined-risk traders can "
                     "sell premium or buy long straddles into catalysts; trend traders should expect lower "
                     "realized ranges."),
            "urgency": "medium",
        })
    else:
        vix_str = f"{vix:.1f}" if vix is not None else "n/a"
        insights.append({
            "title": "Moderate Volatility Environment",
            "body": (f"VIX {vix_str} / model vol forecast {vol_forecast*100:.0f}% "
                     "keeps a balanced risk posture — standard sizing applies, tighten stops on fade setups."),
            "urgency": "low",
        })

    # ── 3. Market bias / regime card ───────────────────────────────────────
    if bias_lower == "BULLISH":
        insights.append({
            "title": f"Bullish Regime — {market_regime}",
            "body": (f"Trend-following and breakout buying have favorable expectancy (conviction "
                     f"{conviction*100:.0f}%). Ride winners; on pullback entries, require higher-timeframe support."),
            "urgency": "low",
        })
    elif bias_lower == "BEARISH":
        insights.append({
            "title": f"Bearish Regime — {market_regime}",
            "body": ("Momentum is negative. Favor short setups, relative-strength pairs, or cash. If trading "
                     "long, require higher-quality entries and tighter stops."),
            "urgency": "medium",
        })
    else:
        insights.append({
            "title": "Neutral Market Outlook",
            "body": ("Without a clear directional tilt, breakouts are prone to failure. Favor mean-reversion "
                     "and range-trading setups over momentum entries until the market resolves."),
            "urgency": "low",
        })

    # ── 4. Style-specific card (always present) ───────────────────────────
    if style == "Swing Trader" and is_ranging:
        insights.append({
            "title": "Consolidation Alert for Swing Traders",
            "body": ("Markets are in consolidation — avoid breakout plays, favor mean-reversion setups at "
                     "range extremes with defined invalidation levels."),
            "urgency": "high",
        })
    elif style == "Swing Trader":
        insights.append({
            "title": "Swing Trader Playbook",
            "body": ("Current conditions favor daily-chart setups — require weekly-level support/resistance "
                     "confluence before entries and set invalidation below the swing low."),
            "urgency": "low",
        })
    elif style == "Day Trader" and bias_lower == "BULLISH":
        insights.append({
            "title": "Bullish Momentum — Intraday Longs Favored",
            "body": "Trade with the intraday trend: longs on pullbacks to VWAP with volume confirmation.",
            "urgency": "low",
        })
    elif style == "Day Trader":
        insights.append({
            "title": "Day Trader Playbook",
            "body": "Focus on the opening range and VWAP levels; avoid holding through low-liquidity midday chop.",
            "urgency": "low",
        })
    elif style in ("Position Trader", "Long-Term Investor") and bias_lower == "BEARISH":
        insights.append({
            "title": "Long-Horizon Caution",
            "body": "A bearish tilt argues for patience: build positions gradually, keep dry powder, and add only at strong support.",
            "urgency": "medium",
        })
    elif style in ("Position Trader", "Long-Term Investor"):
        insights.append({
            "title": "Position Trader Playbook",
            "body": "Keep position building incremental and scale in on weakness to defined support rather than chasing strength.",
            "urgency": "low",
        })
    else:
        insights.append({
            "title": f"{style} Playbook",
            "body": ("Match your entry triggers to your holding horizon — confirm with volume and a defined "
                     "invalidation before committing capital."),
            "urgency": "low",
        })

    # ── 5. Watchlist signal card ───────────────────────────────────────────
    signals = watchlist_signals
    if signals is None:
        signals = _fetch_watchlist_signal_snapshot(watchlist)
    if signals:
        overbought = [s for s in signals if (s.get("metrics") or {}).get("rsi", 50) > 70]
        oversold = [s for s in signals if (s.get("metrics") or {}).get("rsi", 50) < 30]
        strong_mom = [s for s in signals if abs(s.get("change", 0) or 0) > 4]
        if overbought:
            syms = ", ".join(s["symbol"] for s in overbought[:3])
            insights.append({
                "title": f"Overbought Watch: {syms}",
                "body": f"RSI above 70 on {syms} — tighten stops or trim into strength rather than adding.",
                "urgency": "medium",
            })
        elif oversold:
            syms = ", ".join(s["symbol"] for s in oversold[:3])
            insights.append({
                "title": f"Oversold Watch: {syms}",
                "body": f"RSI below 30 on {syms} — watch for mean-reversion bounce setups with defined invalidation.",
                "urgency": "medium",
            })
        elif strong_mom:
            syms = ", ".join(s["symbol"] for s in strong_mom[:3])
            insights.append({
                "title": f"Momentum Leaders: {syms}",
                "body": f"{syms} moved >4% recently — momentum strategies should watch for continuation or exhaustion.",
                "urgency": "low",
            })
        else:
            insights.append({
                "title": "Watchlist: No Extreme Setups",
                "body": "No watchlist name is overbought/oversold right now — patience beats forced entries.",
                "urgency": "low",
            })
    else:
        insights.append({
            "title": "Watchlist Signals Unavailable",
            "body": "Could not compute watchlist RSI/momentum — regime and volatility insights above remain valid.",
            "urgency": "low",
        })

    # ── 6. Risk-management card (always present) ───────────────────────────
    if risk == "Conservative" and (vix is not None and vix > 25):
        insights.append({
            "title": "Conservative Risk Lockdown",
            "body": ("Elevated volatility + conservative profile: halve concurrent positions and consider "
                     "hedging beta with index puts."),
            "urgency": "high",
        })
    elif risk in ("Aggressive", "Very Aggressive") and bias_lower == "BULLISH":
        insights.append({
            "title": "Aggressive Upside Appetite",
            "body": "Bullish regime + aggressive profile: allocate toward the strongest momentum names, with a hard max-loss rule per trade.",
            "urgency": "low",
        })
    else:
        insights.append({
            "title": f"{risk} Risk Discipline",
            "body": (f"Keep per-trade risk at {max_loss:.1f}% of capital and cap concurrent positions per your "
                     f"profile. Re-check sizing whenever the volatility regime shifts."),
            "urgency": "low",
        })

    # Prioritize urgency and cap at 6 cards
    insights.sort(key=lambda c: {"high": 0, "medium": 1, "low": 2}.get(c.get("urgency", "low"), 3))
    return insights[:6]


# 
# PERSONALIZED DASHBOARD (used by main.py)
# 

def show_personalized_dashboard():
    """Show a personalized 'My Dashboard' based on trader profile."""
    import concurrent.futures

    profile = get_trader_profile()
    style = profile.get("trading_style", "Swing Trader")
    style_info = TRADING_STYLES.get(style, TRADING_STYLES["Swing Trader"])
    risk = profile.get("risk_profile", "Moderate")
    risk_info = RISK_PROFILES.get(risk, RISK_PROFILES["Moderate"])
    name = profile.get("name") or "Trader"

    st.markdown(f"### {style_info['icon']} Welcome back, **{name}**")
    st.caption(f"{style} | {risk_info['color']} {risk} | {profile.get('experience_level', '')} | "
               f"Capital: {profile.get('capital_range', 'N/A')}")

    #  Watchlist Quick View 
    watchlist = profile.get("watchlist", ["SPY", "QQQ", "AAPL"])
    if watchlist:
        st.subheader("[PIN] Your Watchlist")
        from watchlist_dashboard import _fetch_symbol_data, _safe_close, _compute_technicals

        @st.cache_data(ttl=180, show_spinner=False)
        def _cached_watchlist_analysis(symbols: tuple) -> list:
            """Fetch watchlist quotes + technicals once per 3 minutes instead of
            on every rerun (this was a top dashboard load-time cost)."""
            import concurrent.futures as _cf

            def _quick_analyze(sym):
                try:
                    df, at = _fetch_symbol_data(sym, "1mo")
                    if df is None or df.empty:
                        return None
                    close = _safe_close(df)
                    if close is None or len(close) < 5:
                        return None
                    tech = _compute_technicals(close, df)
                    price = float(close.iloc[-1])
                    prev = float(close.iloc[-2]) if len(close) >= 2 else price
                    change = (price / prev - 1) * 100
                    return {"symbol": sym, "price": price, "change": change, **tech}
                except Exception:
                    return None

            results = []
            with _cf.ThreadPoolExecutor(max_workers=6) as pool:
                futures = {pool.submit(_quick_analyze, s): s for s in symbols}
                done, _ = _cf.wait(futures, timeout=15)
                for f in done:
                    try:
                        r = f.result(timeout=1)
                        if r:
                            results.append(r)
                    except Exception:
                        pass
            return results

        wl_results = _cached_watchlist_analysis(tuple(watchlist[:12]))

        if wl_results:
            cols_per_row = min(len(wl_results), 4)
            rows = [wl_results[i:i + cols_per_row] for i in range(0, len(wl_results), cols_per_row)]
            for row in rows:
                cols = st.columns(len(row))
                for col, r in zip(cols, row):
                    with col:
                        sig_icon = "" if r.get("signal") == "BULLISH" else "" if r.get("signal") == "BEARISH" else ""
                        chg = r.get("change", 0)
                        st.metric(
                            f"{r['symbol']} {sig_icon}",
                            f"${r['price']:,.2f}",
                            f"{chg:+.2f}%"
                        )
                        rsi = r.get("metrics", {}).get("rsi")
                        if rsi:
                            st.caption(f"RSI: {rsi:.0f} | Conf: {r.get('confidence', 0):.0%}")

    #  Tailored Tips Based on Profile & Adaptive Engine
    st.markdown("---")
    st.subheader(" Personalized Insights")

    # Market-aware insight cards (VIX, market bias, trading style, watchlist signals)
    try:
        wl_signal_list = wl_results if 'wl_results' in locals() else None
        cards = generate_market_aware_insights(wl_signal_list)

        if cards:
            urgency_colors = {"high": "#ef5350", "medium": "#ff9800", "low": "#4caf50"}
            for card in cards:
                urg = card.get("urgency", "low")
                color = urgency_colors.get(urg, "#ff9800")
                _title = _html.escape(str(card.get('title', 'Insight')))
                _body = _html.escape(str(card.get('body', '')))
                st.markdown(
                    f"""<div style="background: rgba(10,22,40,0.7); border-left: 3px solid {color};
                                border-radius: 0 8px 8px 0; padding: 12px 16px; margin-bottom: 10px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="color:{color}; font-size:0.72rem; text-transform:uppercase; letter-spacing:1px; font-weight:700;">{_title}</span>
                            <span style="color:{color}; font-size:0.62rem; border:1px solid {color}; border-radius:4px; padding:1px 6px; font-weight:700;">{urg.upper()}</span>
                        </div>
                        <div style="color:#e8eaf0; margin-top:6px; font-size:0.9rem; line-height:1.5;">{_body}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No personalized insights available right now — check back after market data refreshes.")
    except Exception as e:
        st.error(f"Could not load dynamic insights: {e}")
        st.info("The market-aware insight engine is temporarily unavailable. Static tips may not reflect current conditions.")

    #  Recommended Actions (Dynamic)
    st.markdown("---")
    st.subheader(" Recommended Actions")

    goals = profile.get("trading_goals", [])
    actions = []
    
    # 1. Generate dynamic actions based on live watchlist data
    try:
        if 'wl_results' in locals() and wl_results:
            for r in wl_results:
                sym = r.get("symbol", "")
                rsi = r.get("metrics", {}).get("rsi")
                sig = r.get("signal")
                
                if rsi:
                    if rsi < 30:
                        actions.append(("[RSI]", f"**{sym}** is oversold (RSI: {rsi:.0f})", "Look for mean reversion setups"))
                    elif rsi > 70:
                        actions.append(("[RSI]", f"**{sym}** is overbought (RSI: {rsi:.0f})", "Consider tightening stops or taking profits"))
                
                if sig == "BULLISH":
                    actions.append(("[BULL]", f"**{sym}** flashed a Bullish signal", "Review for potential long entry"))
                elif sig == "BEARISH":
                    actions.append(("[BEAR]", f"**{sym}** flashed a Bearish signal", "Review for potential short entry or hedge"))
    except Exception:
        pass

    # 2. Add goal-based contextual actions if we need more
    if len(actions) < 4:
        if "Consistent income (weekly/monthly)" in goals:
            actions.append(("[YIELD]", "Review covered call opportunities on your watchlist", "Navigate to Options Engine"))
        if "Beat the S&P 500" in goals:
            actions.append(("[ALPHA]", "Compare your watchlist performance vs SPY", "Go to Symbol Analysis -> SPY"))
        if "Learning & skill development" in goals:
            actions.append(("[LEARN]", "Run a simulation to practice without risk", "Go to Simulation Hub"))
        if "Generate alpha with options" in goals:
            actions.append(("[OPT]", "Check unusual options activity on your watchlist", "Use Options Scanner"))

    # Fallback
    if not actions:
        actions.append(("[SCAN]", "Scan the market for opportunities matching your style", "Go to Market Scanner"))

    # Limit to top 4 most relevant actions
    import random
    if len(actions) > 4:
        # Keep the first 2 dynamic ones, then sample the rest
        display_actions = actions[:2] + random.sample(actions[2:], 2)
    else:
        display_actions = actions

    for icon, action, how in display_actions:
        st.markdown(f"{icon} **{action}**")
        st.caption(f" {how}")

    #  Quick Stats 
    st.markdown("---")
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    with col_q1:
        st.metric("Interactions", profile.get("interaction_count", 0))
    with col_q2:
        st.metric("Watchlist Size", len(profile.get("watchlist", [])))
    with col_q3:
        st.metric("Asset Classes", len(profile.get("asset_classes", [])))
    with col_q4:
        updated = profile.get("updated_at", "Never")
        if updated and updated != "Never":
            updated = updated[:10]
        st.metric("Last Updated", updated)

