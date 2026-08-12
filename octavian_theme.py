"""
Octavian Professional Theme
Navy blue, white, black, gold, lavender color scheme with micro-animations.
Author: APB - Octavian Team
"""

import streamlit as st

COLORS = {
    "navy": "#0a1628",
    "navy_light": "#132240",
    "navy_mid": "#1a2d4a",
    "gold": "#c9a84c",
    "gold_light": "#d4b86a",
    "gold_dark": "#a08030",
    "white": "#f0f2f6",
    "white_soft": "#e0e4ec",
    "black": "#060d18",
    "lavender": "#9b8ec4",
    "lavender_light": "#b8aed6",
    "lavender_dark": "#7b6ea4",
    "success": "#4caf50",
    "danger": "#ef5350",
    "neutral": "#78909c",
    "text_primary": "#e8eaf0",
    "text_secondary": "#a0a8b8",
    "border": "#1e3050",
    "glass_bg": "rgba(19, 34, 64, 0.65)",
    "glass_border": "rgba(201, 168, 76, 0.12)",
}

PLOTLY_TEMPLATE = {
    "template": "plotly_dark",
    "paper_bgcolor": COLORS["navy"],
    "plot_bgcolor": COLORS["navy_light"],
    "font_color": COLORS["text_primary"],
}


def apply_theme():
    """Apply the full Octavian professional theme."""
    st.markdown(_get_theme_css(), unsafe_allow_html=True)


def _get_theme_css() -> str:
    return f"""
    <style>
    /* ========================================
       OCTAVIAN PROFESSIONAL THEME v4.0
       Micro-animations | Glassmorphism | Polish
       ======================================== */

    /* --- TYPOGRAPHY: Google Fonts --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* --- GLOBAL --- */
    .stApp {{
        background: linear-gradient(160deg, {COLORS['black']} 0%, {COLORS['navy']} 40%, {COLORS['navy_light']} 100%);
        color: {COLORS['text_primary']};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* --- ANIMATED BACKGROUND GLOW --- */
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
        background:
            radial-gradient(ellipse at 20% 50%, rgba(155,142,196,0.05) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(201,168,76,0.04) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(26,45,74,0.06) 0%, transparent 50%);
        animation: octavian-glow 12s ease-in-out infinite alternate;
    }}

    @keyframes octavian-glow {{
        0%   {{ opacity: 0.6; transform: scale(1.0); }}
        50%  {{ opacity: 1.0; transform: scale(1.02); }}
        100% {{ opacity: 0.7; transform: scale(1.0); }}
    }}

    /* Subtle grid overlay */
    .stApp::after {{
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
        background-image:
            linear-gradient(rgba(201,168,76,0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(201,168,76,0.015) 1px, transparent 1px);
        background-size: 60px 60px;
        animation: grid-drift 40s linear infinite;
    }}

    @keyframes grid-drift {{
        0%   {{ transform: translate(0, 0); }}
        100% {{ transform: translate(60px, 60px); }}
    }}

    /* --- MICRO-ANIMATION KEYFRAMES --- */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes pulse-dot {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50%       {{ opacity: 0.4; transform: scale(0.85); }}
    }}

    @keyframes status-blink {{
        0%, 100% {{ opacity: 1; }}
        50%       {{ opacity: 0.5; }}
    }}

    @keyframes badge-glow {{
        0%, 100% {{ box-shadow: 0 0 0 0 rgba(201,168,76,0); }}
        50%       {{ box-shadow: 0 0 6px 2px rgba(201,168,76,0.3); }}
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to   {{ opacity: 1; }}
    }}

    @keyframes shimmer {{
        0%   {{ background-position: -200% center; }}
        100% {{ background-position: 200% center; }}
    }}

    @keyframes pulse-border {{
        0%, 100% {{ border-color: {COLORS['border']}; }}
        50%      {{ border-color: {COLORS['gold_dark']}; }}
    }}

    @keyframes slideInLeft {{
        from {{ opacity: 0; transform: translateX(-8px); }}
        to   {{ opacity: 1; transform: translateX(0); }}
    }}

    /* --- MICROANIMATION UTILITY CLASSES (emoji replacements) --- */

    /* Status indicator dot — replaces colored circle emojis */
    .oct-dot {{
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
        flex-shrink: 0;
    }}
    .oct-dot-red   {{ background: #ef5350; animation: pulse-dot 1.8s ease-in-out infinite; }}
    .oct-dot-green {{ background: #4caf50; animation: pulse-dot 1.8s ease-in-out infinite; }}
    .oct-dot-gold  {{ background: #c9a84c; animation: pulse-dot 2.2s ease-in-out infinite; }}
    .oct-dot-gray  {{ background: #78909c; }}
    .oct-dot-orange {{ background: #ff9800; animation: pulse-dot 2s ease-in-out infinite; }}

    /* Status badge — replaces emoji status labels */
    .oct-badge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        border-radius: 4px;
        padding: 2px 9px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        animation: fadeInUp 0.3s ease-out;
    }}
    .oct-badge-danger {{
        background: rgba(239,83,80,0.12);
        border: 1px solid rgba(239,83,80,0.35);
        color: #ef5350;
    }}
    .oct-badge-success {{
        background: rgba(76,175,80,0.12);
        border: 1px solid rgba(76,175,80,0.35);
        color: #4caf50;
    }}
    .oct-badge-neutral {{
        background: rgba(120,144,156,0.12);
        border: 1px solid rgba(120,144,156,0.35);
        color: #78909c;
    }}
    .oct-badge-gold {{
        background: rgba(201,168,76,0.12);
        border: 1px solid rgba(201,168,76,0.35);
        color: #c9a84c;
        animation: badge-glow 2.5s ease-in-out infinite;
    }}
    .oct-badge-warning {{
        background: rgba(255,152,0,0.12);
        border: 1px solid rgba(255,152,0,0.35);
        color: #ff9800;
    }}

    /* Alert icon — replaces  warning emoji */
    .oct-alert-icon {{
        display: inline-block;
        width: 14px;
        height: 14px;
        background: #ff9800;
        clip-path: polygon(50% 0%, 0% 100%, 100% 100%);
        margin-right: 5px;
        vertical-align: middle;
        animation: status-blink 2s ease-in-out infinite;
        flex-shrink: 0;
    }}

    /* Check icon — replaces  checkmark */
    .oct-check-icon {{
        display: inline-block;
        width: 12px;
        height: 12px;
        border-right: 2px solid #4caf50;
        border-bottom: 2px solid #4caf50;
        transform: rotate(45deg) translateY(-2px);
        margin-right: 5px;
        vertical-align: middle;
        flex-shrink: 0;
    }}

    /* Cross icon — replaces  */
    .oct-cross-icon {{
        display: inline-block;
        width: 10px;
        height: 10px;
        position: relative;
        margin-right: 5px;
        vertical-align: middle;
        flex-shrink: 0;
    }}
    .oct-cross-icon::before,
    .oct-cross-icon::after {{
        content: '';
        position: absolute;
        top: 50%; left: 0;
        width: 100%; height: 2px;
        background: #ef5350;
        border-radius: 1px;
    }}
    .oct-cross-icon::before {{ transform: translateY(-50%) rotate(45deg); }}
    .oct-cross-icon::after  {{ transform: translateY(-50%) rotate(-45deg); }}

    /* Arrow icon — replaces → */
    .oct-arrow-icon {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-right: 2px solid #64b5f6;
        border-bottom: 2px solid #64b5f6;
        transform: rotate(-45deg) translateX(1px);
        margin: 0 3px;
        vertical-align: middle;
        flex-shrink: 0;
    }}

    /* Plus icon — replaces  opportunity marker */
    .oct-plus-icon {{
        display: inline-block;
        width: 12px;
        height: 12px;
        position: relative;
        margin-right: 5px;
        vertical-align: middle;
        flex-shrink: 0;
    }}
    .oct-plus-icon::before,
    .oct-plus-icon::after {{
        content: '';
        position: absolute;
        background: #4caf50;
        border-radius: 1px;
    }}
    .oct-plus-icon::before {{ top: 50%; left: 0; width: 100%; height: 2px; transform: translateY(-50%); }}
    .oct-plus-icon::after  {{ top: 0; left: 50%; width: 2px; height: 100%; transform: translateX(-50%); }}

    /* Severity indicator bar — replaces HIGH/MEDIUM/LOW text */
    .oct-severity {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }}
    .oct-severity-high  {{ color: #ef5350; }}
    .oct-severity-med   {{ color: #ff9800; }}
    .oct-severity-low   {{ color: #c9a84c; }}

    /* Animated severity bar */
    .oct-sev-bar {{
        display: inline-block;
        height: 6px;
        border-radius: 3px;
        vertical-align: middle;
        margin-right: 4px;
    }}
    .oct-sev-bar-high  {{ width: 24px; background: #ef5350; animation: status-blink 1.5s ease-in-out infinite; }}
    .oct-sev-bar-med   {{ width: 16px; background: #ff9800; animation: status-blink 2s ease-in-out infinite; }}
    .oct-sev-bar-low   {{ width: 8px;  background: #c9a84c; }}

    /* Live indicator — animated green dot for live data */
    .oct-live {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.7rem;
        color: #4caf50;
        font-weight: 600;
        letter-spacing: 0.06em;
    }}
    .oct-live-dot {{
        width: 7px;
        height: 7px;
        background: #4caf50;
        border-radius: 50%;
        animation: pulse-dot 1.2s ease-in-out infinite;
        flex-shrink: 0;
    }}

    /* Direction arrow — replaces up/down arrows */
    .oct-arrow-up {{
        display: inline-block;
        width: 0; height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-bottom: 8px solid #4caf50;
        margin-right: 4px;
        vertical-align: middle;
    }}
    .oct-arrow-down {{
        display: inline-block;
        width: 0; height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 8px solid #ef5350;
        margin-right: 4px;
        vertical-align: middle;
    }}

    /* --- SIDEBAR --- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['black']} 0%, {COLORS['navy']} 100%);
        border-right: 1px solid {COLORS['border']};
    }}

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {COLORS['gold']};
    }}

    /* --- HEADERS --- */
    h1 {{
        color: {COLORS['gold']} !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        border-bottom: 1px solid {COLORS['gold_dark']};
        padding-bottom: 8px;
        animation: fadeInUp 0.5s ease-out;
    }}

    h2 {{
        color: {COLORS['white']} !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        animation: fadeInUp 0.4s ease-out;
    }}

    h3 {{
        color: {COLORS['lavender_light']} !important;
        font-weight: 500 !important;
        animation: fadeInUp 0.35s ease-out;
    }}

    h4 {{
        color: {COLORS['gold_light']} !important;
        font-weight: 500 !important;
    }}

    /* --- METRICS with micro-animation --- */
    [data-testid="stMetric"] {{
        background: linear-gradient(135deg, {COLORS['glass_bg']}, {COLORS['navy_mid']});
        border: 1px solid {COLORS['glass_border']};
        border-radius: 10px;
        padding: 14px 18px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.4s ease-out both;
        min-width: 0;
        overflow: visible !important;
        width: auto !important;
        max-width: none !important;
    }}

    [data-testid="stMetric"]:hover {{
        border-color: {COLORS['gold']};
        box-shadow: 0 4px 20px rgba(201,168,76,0.15), 0 0 0 1px rgba(201,168,76,0.1);
        transform: translateY(-3px) scale(1.01);
    }}

    [data-testid="stMetricLabel"] {{
        color: {COLORS['text_secondary']} !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 500;
        overflow: visible !important;
        word-wrap: break-word !important;
        white-space: normal !important;
        min-width: 0;
        line-height: 1.2;
    }}

    [data-testid="stMetricValue"] {{
        color: {COLORS['white']} !important;
        font-family: 'JetBrains Mono', 'Consolas', monospace !important;
        font-weight: 600;
        font-size: 0.95rem !important;
        overflow: visible !important;
        word-wrap: break-word !important;
        white-space: normal !important;
        min-width: 0;
    }}

    [data-testid="stMetricDelta"] > div {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem;
    }}

    /* --- BUTTONS with micro-animations --- */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['navy_mid']}, {COLORS['navy_light']});
        color: {COLORS['gold_light']};
        border: 1px solid {COLORS['gold_dark']};
        border-radius: 8px;
        font-weight: 500;
        letter-spacing: 0.3px;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}

    .stButton > button::after {{
        content: '';
        position: absolute;
        top: 50%; left: 50%;
        width: 0; height: 0;
        background: rgba(201,168,76,0.15);
        border-radius: 50%;
        transition: width 0.4s ease, height 0.4s ease;
        transform: translate(-50%, -50%);
    }}

    .stButton > button:hover::after {{
        width: 300px;
        height: 300px;
    }}

    .stButton > button:hover {{
        background: linear-gradient(135deg, {COLORS['gold_dark']}, {COLORS['gold']});
        color: {COLORS['black']};
        border-color: {COLORS['gold']};
        box-shadow: 0 4px 18px rgba(201,168,76,0.3);
        transform: translateY(-1px);
    }}

    .stButton > button:active {{
        transform: translateY(0px) scale(0.98);
        transition: transform 0.1s;
    }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {COLORS['gold_dark']}, {COLORS['gold']});
        color: {COLORS['black']};
        border: none;
        font-weight: 600;
        box-shadow: 0 2px 12px rgba(201,168,76,0.2);
    }}

    .stButton > button[kind="primary"]:hover {{
        box-shadow: 0 6px 24px rgba(201,168,76,0.4);
        transform: translateY(-2px);
    }}

    /* --- TABS with slide animation --- */
    .stTabs [data-baseweb="tab-list"] {{
        background: {COLORS['navy']};
        border-bottom: 1px solid {COLORS['border']};
        gap: 0;
    }}

    .stTabs [data-baseweb="tab"] {{
        color: {COLORS['text_secondary']};
        border-bottom: 2px solid transparent;
        padding: 10px 22px;
        font-size: 0.85rem;
        letter-spacing: 0.3px;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        color: {COLORS['gold_light']};
        background: rgba(201,168,76,0.04);
    }}

    .stTabs [aria-selected="true"] {{
        color: {COLORS['gold']} !important;
        border-bottom-color: {COLORS['gold']} !important;
        background: transparent !important;
        font-weight: 600;
    }}

    /* --- DATAFRAMES --- */
    .stDataFrame {{
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        overflow: hidden;
        animation: fadeIn 0.3s ease-out;
    }}

    /* --- INPUTS with focus animation --- */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {{
        background: {COLORS['navy_light']} !important;
        color: {COLORS['text_primary']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }}

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus {{
        border-color: {COLORS['gold']} !important;
        box-shadow: 0 0 0 2px rgba(201,168,76,0.15) !important;
        outline: none;
    }}

    .stSelectbox > div > div {{
        background: {COLORS['navy_light']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 8px;
        transition: border-color 0.3s ease;
    }}

    .stSelectbox > div > div:hover {{
        border-color: {COLORS['gold_dark']} !important;
    }}

    /* --- EXPANDERS with smooth open --- */
    .streamlit-expanderHeader {{
        background: {COLORS['navy_light']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        color: {COLORS['gold_light']} !important;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }}

    .streamlit-expanderHeader:hover {{
        border-color: {COLORS['gold_dark']};
        background: {COLORS['navy_mid']};
    }}

    details[open] .streamlit-expanderContent {{
        animation: fadeInUp 0.3s ease-out;
    }}

    /* --- CHAT --- */
    .stChatMessage {{
        background: {COLORS['navy_light']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        animation: fadeInUp 0.3s ease-out;
        transition: border-color 0.3s ease;
    }}

    .stChatMessage:hover {{
        border-color: {COLORS['glass_border']};
    }}

    /* --- RADIO SIDEBAR NAV with slide animation --- */
    .stRadio > div {{
        gap: 2px;
    }}

    .stRadio > div > label {{
        padding: 8px 14px;
        border-radius: 6px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 0.88rem;
        font-family: 'Inter', sans-serif;
        border-left: 2px solid transparent;
    }}

    .stRadio > div > label:hover {{
        background: {COLORS['navy_mid']};
        border-left-color: {COLORS['gold_dark']};
        padding-left: 18px;
    }}

    /* --- DIVIDERS --- */
    hr {{
        border-color: {COLORS['border']};
        opacity: 0.5;
    }}

    /* --- CAPTIONS --- */
    .stCaption {{
        color: {COLORS['text_secondary']} !important;
        font-size: 0.75rem;
    }}

    /* --- SCROLLBAR --- */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: {COLORS['navy']};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['gold_dark']};
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['gold']};
    }}

    /* --- PLOTLY CHART CONTAINERS --- */
    .js-plotly-plot {{
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        animation: fadeIn 0.4s ease-out;
    }}

    /* --- LOADING SPINNER --- */
    .stSpinner > div {{
        border-color: {COLORS['gold']} transparent transparent transparent !important;
    }}

    /* --- FILE UPLOADER --- */
    [data-testid="stFileUploader"] {{
        border: 1px dashed {COLORS['border']};
        border-radius: 10px;
        padding: 14px;
        transition: all 0.3s ease;
    }}

    [data-testid="stFileUploader"]:hover {{
        border-color: {COLORS['gold_dark']};
        background: rgba(201,168,76,0.02);
    }}

    /* --- ALERTS --- */
    .stAlert {{
        border-radius: 8px;
        animation: fadeInUp 0.3s ease-out;
        backdrop-filter: blur(4px);
    }}

    /* --- COLUMNS: stagger animation --- */
    [data-testid="stHorizontalBlock"] > div {{
        animation: fadeInUp 0.4s ease-out both;
    }}

    [data-testid="stHorizontalBlock"] > div:nth-child(1) {{ animation-delay: 0.02s; }}
    [data-testid="stHorizontalBlock"] > div:nth-child(2) {{ animation-delay: 0.06s; }}
    [data-testid="stHorizontalBlock"] > div:nth-child(3) {{ animation-delay: 0.10s; }}
    [data-testid="stHorizontalBlock"] > div:nth-child(4) {{ animation-delay: 0.14s; }}
    [data-testid="stHorizontalBlock"] > div:nth-child(5) {{ animation-delay: 0.18s; }}

    /* --- DOWNLOAD BUTTON --- */
    .stDownloadButton > button {{
        background: linear-gradient(135deg, {COLORS['navy_mid']}, {COLORS['lavender_dark']}) !important;
        color: {COLORS['white']} !important;
        border: 1px solid {COLORS['lavender_dark']} !important;
        border-radius: 8px;
        transition: all 0.3s ease;
    }}

    .stDownloadButton > button:hover {{
        background: linear-gradient(135deg, {COLORS['lavender_dark']}, {COLORS['lavender']}) !important;
        box-shadow: 0 4px 16px rgba(155,142,196,0.25);
        transform: translateY(-1px);
    }}

    /* --- SLIDER --- */
    .stSlider [data-testid="stThumbValue"] {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
    }}

    /* --- MARKDOWN LINKS --- */
    a {{
        color: {COLORS['gold_light']} !important;
        text-decoration: none;
        transition: color 0.2s ease;
    }}
    a:hover {{
        color: {COLORS['gold']} !important;
        text-decoration: underline;
    }}

    /* --- TOOLTIPS & POPOVERS --- */
    [data-baseweb="tooltip"] {{
        background: {COLORS['navy_mid']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 6px;
    }}

    /* --- PROGRESS BAR --- */
    .stProgress > div > div {{
        background: linear-gradient(90deg, {COLORS['gold_dark']}, {COLORS['gold']}, {COLORS['gold_light']});
        background-size: 200% 100%;
        animation: shimmer 2s linear infinite;
        border-radius: 4px;
    }}

    /* --- EMPTY STATE TEXT --- */
    .stMarkdown p {{
        line-height: 1.6;
    }}

    /* --- PHASE 6: MOBILE RESPONSIVE TERMINAL (PWA Support) --- */
    @media (max-width: 768px) {{
        .stApp {{
            font-size: 0.9rem;
        }}
        [data-testid="stSidebar"] {{
            min-width: 100vw;
        }}
        .stApp::before {{
            display: none; /* Disable heavy animations on mobile for performance */
        }}
        h1 {{ font-size: 1.4rem !important; }}
        h2 {{ font-size: 1.2rem !important; }}
        h3 {{ font-size: 1.1rem !important; }}
        
        [data-testid="stMetric"] {{
            padding: 10px 12px;
            margin-bottom: 8px;
        }}
        
        /* Force single column for metrics horizontally */
        [data-testid="stHorizontalBlock"] {{
            flex-direction: column !important;
            gap: 8px !important;
        }}
        [data-testid="stHorizontalBlock"] > div {{
            width: 100% !important;
        }}
        
        .stButton > button {{
            width: 100%;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            padding: 8px 12px;
            font-size: 0.75rem;
        }}
    }}
    </style>
    """


def apply_glass_card(content_html: str, accent: str = "gold") -> str:
    """Return HTML for a glassmorphism card with optional accent color."""
    accent_color = COLORS.get(accent, COLORS["gold"])
    return (
        f'<div style="'
        f'background:{COLORS["glass_bg"]};'
        f'border:1px solid {COLORS["glass_border"]};'
        f'border-top:2px solid {accent_color};'
        f'border-radius:12px;'
        f'padding:20px;'
        f'backdrop-filter:blur(12px);'
        f'-webkit-backdrop-filter:blur(12px);'
        f'animation:fadeInUp 0.4s ease-out;'
        f'">{content_html}</div>'
    )


def render_header(title: str, subtitle: str = ""):
    """Render a professional header bar with shimmer accent."""
    sub_html = (
        f'<p style="color:{COLORS["lavender"]};font-size:0.85rem;'
        f'margin:6px 0 0 0;font-weight:300;letter-spacing:1px;">{subtitle}</p>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="text-align:center;padding:28px 24px 22px;'
        f'background:linear-gradient(135deg,{COLORS["black"]},{COLORS["navy"]},{COLORS["navy_mid"]});'
        f'border:1px solid {COLORS["border"]};border-radius:12px;margin-bottom:24px;'
        f'box-shadow:0 8px 32px rgba(0,0,0,0.4);'
        f'animation:fadeInUp 0.5s ease-out;">'
        f'<h1 style="color:{COLORS["gold"]};margin:0;font-family:Inter,sans-serif;'
        f'letter-spacing:3px;font-size:1.8rem;border:none;padding:0;'
        f'font-weight:700;">OCTAVIAN</h1>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


def section_header(text: str):
    """Render a section header with gold accent and fade animation."""
    st.markdown(
        f'<h3 style="color:{COLORS["gold_light"]};border-left:3px solid {COLORS["gold"]};'
        f'padding-left:14px;margin:24px 0 12px 0;font-weight:600;'
        f'animation:slideInLeft 0.4s ease-out;">{text}</h3>',
        unsafe_allow_html=True,
    )


def status_badge(text: str, variant: str = "neutral") -> str:
    """Return inline HTML for a colored status badge. variant: success|danger|neutral|gold|lavender"""
    color_map = {
        "success": (COLORS["success"], "rgba(76,175,80,0.12)"),
        "danger": (COLORS["danger"], "rgba(239,83,80,0.12)"),
        "neutral": (COLORS["neutral"], "rgba(120,144,156,0.12)"),
        "gold": (COLORS["gold"], "rgba(201,168,76,0.12)"),
        "lavender": (COLORS["lavender"], "rgba(155,142,196,0.12)"),
    }
    fg, bg = color_map.get(variant, color_map["neutral"])
    return (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;'
        f'font-size:0.72rem;font-weight:600;letter-spacing:0.5px;'
        f'color:{fg};background:{bg};border:1px solid {fg};">{text}</span>'
    )


# 
# ENHANCED MICROANIMATIONS - Sleek & Professional
# 

def get_extended_animations() -> str:
    """Return extended CSS animations for enhanced UI."""
    return """
    <style>
    /* Pulse animation for live indicators */
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 5px currentColor; }
        50% { box-shadow: 0 0 20px currentColor, 0 0 30px currentColor; }
        100% { box-shadow: 0 0 5px currentColor; }
    }
    
    /* Shimmer effect for loading states */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    /* Smooth fade in for cards */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Ripple effect for buttons */
    @keyframes ripple {
        0% {
            transform: scale(0);
            opacity: 0.5;
        }
        100% {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    /* Smooth slide in from right */
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Bounce subtle for notifications */
    @keyframes bounce-subtle {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    
    /* Rotate for loading spinners */
    @keyframes spin-smooth {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Scale up animation */
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Gradient flow animation */
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Hover glow effect */
    .hover-glow:hover {
        box-shadow: 0 0 15px rgba(201, 168, 76, 0.5);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    /* Live pulse class */
    .live-pulse {
        animation: pulse-glow 2s ease-in-out infinite;
    }
    
    /* Card animations */
    .animate-card {
        animation: fadeInUp 0.5s ease-out forwards;
    }
    
    /* Slide in elements */
    .animate-slide-right {
        animation: slideInRight 0.4s ease-out forwards;
    }
    
    /* Scale animations */
    .animate-scale {
        animation: scaleIn 0.3s ease-out forwards;
    }
    
    /* Gradient backgrounds */
    .gradient-animate {
        background: linear-gradient(-45deg, #0a1628, #132240, #1a2d4a, #0a1628);
        background-size: 400% 400%;
        animation: gradientFlow 15s ease infinite;
    }
    
    /* Notification bounce */
    .notify-bounce {
        animation: bounce-subtle 0.5s ease-in-out;
    }
    
    /* Loading shimmer */
    .shimmer-load {
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
    }
    
    /* Smooth button transitions */
    .smooth-btn {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .smooth-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Status indicator animations */
    .status-active {
        animation: pulse-glow 2s ease-in-out infinite;
        border-radius: 50%;
    }
    
    /* Progress bar animation */
    .progress-animate {
        position: relative;
        overflow: hidden;
    }
    
    .progress-animate::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 2s infinite;
    }
    
    /* Number counter animation */
    @keyframes countUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .count-animate {
        animation: countUp 0.5s ease-out forwards;
    }
    
    /* Attention grabber */
    .attention-pulse {
        animation: pulse-glow 1.5s ease-in-out infinite;
    }
    
    /* Table row hover */
    .table-row-hover:hover {
        background: rgba(201, 168, 76, 0.1);
        transition: background 0.2s ease;
    }
    
    /* Icon animations */
    .icon-spin {
        animation: spin-smooth 1s linear infinite;
    }
    
    /* Card hover lift */
    .card-lift {
        transition: all 0.3s ease;
    }
    
    .card-lift:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    /* Focus ring animation */
    .focus-ring:focus {
        outline: none;
        box-shadow: 0 0 0 3px rgba(201, 168, 76, 0.5);
        transition: box-shadow 0.2s ease;
    }
    
    /* Stagger animation delays */
    .stagger-1 { animation-delay: 0.1s; }
    .stagger-2 { animation-delay: 0.2s; }
    .stagger-3 { animation-delay: 0.3s; }
    .stagger-4 { animation-delay: 0.4s; }
    .stagger-5 { animation-delay: 0.5s; }
    
    /* Success checkmark animation */
    @keyframes checkmark {
        0% { stroke-dashoffset: 100; }
        100% { stroke-dashoffset: 0; }
    }
    
    .checkmark-animate {
        stroke-dasharray: 100;
        stroke-dashoffset: 100;
        animation: checkmark 0.5s ease-out forwards;
    }
    
    /* Warning shake */
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    
    .shake-animate {
        animation: shake 0.3s ease-in-out;
    }
    
    /* Smooth scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a1628;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #1a2d4a;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #c9a84c;
    }
    
    /* Tooltip animation */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        opacity: 0;
        background-color: #132240;
        color: #c9a84c;
        text-align: center;
        padding: 8px 12px;
        border-radius: 6px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        transition: opacity 0.3s, visibility 0.3s;
        border: 1px solid #c9a84c;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Skeleton loading */
    .skeleton {
        background: linear-gradient(90deg, #132240 25%, #1a2d4a 50%, #132240 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
    }
    
    /* Breadcrumb animation */
    .breadcrumb-item + .breadcrumb-item::before {
        content: '>';
        padding: 0 8px;
        color: #78909c;
        animation: fadeInUp 0.3s ease-out;
    }
    
    /* Tab indicator animation */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #132240, #1a2d4a) !important;
        border-color: #c9a84c !important;
        transition: all 0.3s ease !important;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(201, 168, 76, 0.1) !important;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        border: 1px solid #1e3050 !important;
        border-radius: 8px !important;
        overflow: hidden;
    }
    
    /* Metrics animation */
    .stMetric {
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        background: rgba(201, 168, 76, 0.05);
        border-radius: 8px;
    }
    
    /* Slider thumb animation */
    .stSlider [role="slider"] {
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stSlider [role="slider"]:hover {
        transform: scale(1.2);
        box-shadow: 0 0 10px rgba(201, 168, 76, 0.5);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        transition: width 0.5s ease;
    }
    
    /* Spinner animation */
    .stSpinner > div {
        animation: spin-smooth 1s linear infinite;
    }
    
    /* Success/Error/Info toast animations */
    .stToast {
        animation: fadeInUp 0.3s ease-out;
    }
    
    /* Selectbox dropdown */
    .stSelectbox > div > div > div:first-child {
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div > div > div:first-child:hover {
        border-color: #c9a84c;
    }
    
    /* Text input focus */
    .stTextInput > div > div > input:focus {
        border-color: #c9a84c !important;
        box-shadow: 0 0 0 2px rgba(201, 168, 76, 0.2) !important;
        transition: all 0.2s ease;
    }
    
    /* Button hover effects */
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: all 0.2s ease;
    }
    
    /* Radio button animation */
    .stRadio > label > div:first-child {
        transition: all 0.2s ease;
    }
    
    /* Checkbox animation */
    .stCheckbox > label > div:first-child {
        transition: all 0.2s ease;
    }
    
    /* Divider animation */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #1e3050, transparent);
        transition: background 0.3s ease;
    }
    
    hr:hover {
        background: linear-gradient(90deg, transparent, #c9a84c, transparent);
    }
    
    /* Image hover zoom */
    .img-zoom {
        overflow: hidden;
    }
    
    .img-zoom img {
        transition: transform 0.5s ease;
    }
    
    .img-zoom:hover img {
        transform: scale(1.1);
    }
    
    /* Custom scrollable areas */
    .scroll-smooth {
        scroll-behavior: smooth;
    }
    
    /* Paragraph text reveal */
    @keyframes textReveal {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .text-reveal {
        animation: textReveal 0.5s ease-out forwards;
    }
    
    /* Gradient text */
    .gradient-text {
        background: linear-gradient(90deg, #c9a84c, #d4b86a, #c9a84c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-size: 200% auto;
        animation: gradientFlow 3s ease infinite;
    }
    
    /* Border animation */
    .border-animate {
        position: relative;
    }
    
    .border-animate::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #c9a84c, transparent);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .border-animate:hover::before {
        transform: scaleX(1);
    }
    
    /* Floating animation */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .float-animate {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Typing effect */
    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }
    
    .typing-effect {
        overflow: hidden;
        white-space: nowrap;
        animation: typing 2s steps(40, end);
    }
    
    /* Blinking cursor */
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }
    
    .cursor-blink {
        animation: blink 1s step-end infinite;
    }
    
    /* Matrix rain effect (subtle) */
    .matrix-bg {
        position: relative;
        overflow: hidden;
    }
    
    .matrix-bg::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(201, 168, 76, 0.03) 2px,
            rgba(201, 168, 76, 0.03) 4px
        );
        pointer-events: none;
    }
    </style>
    """


def apply_extended_animations():
    """Apply extended animations to the Streamlit app."""
    st.markdown(get_extended_animations(), unsafe_allow_html=True)


def animated_metric(label: str, value: str, delta: str = None, style: str = "default"):
    """Render an animated metric card."""
    delta_color = "#4caf50" if delta and delta.startswith("+") else "#ef5350" if delta else "#78909c"
    
    html = f"""
    <div class="animate-card" style="
        background: linear-gradient(145deg, {COLORS['navy_light']}, {COLORS['navy_mid']});
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        transition: all 0.3s ease;
    ">
        <div style="
            color: {COLORS['text_secondary']};
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        ">{label}</div>
        <div style="
            color: {COLORS['text_primary']};
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 4px;
        ">{value}</div>
        {f'<div style="color: {delta_color}; font-size: 14px; font-weight: 600;">{delta}</div>' if delta else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def animated_progress_bar(value: float, max_value: float = 100, label: str = ""):
    """Render an animated progress bar."""
    percentage = (value / max_value) * 100 if max_value > 0 else 0
    
    html = f"""
    <div class="progress-animate" style="
        background: {COLORS['navy_mid']};
        border-radius: 8px;
        height: 24px;
        overflow: hidden;
        position: relative;
    ">
        <div style="
            background: linear-gradient(90deg, {COLORS['gold']}, {COLORS['gold_light']});
            height: 100%;
            width: {percentage}%;
            border-radius: 8px;
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            {f'<span style="color: {COLORS["navy"]}; font-size: 12px; font-weight: 600;">{label}</span>' if label else ''}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def animated_status_indicator(status: str, label: str = ""):
    """Render an animated status indicator."""
    status_colors = {
        "active": ("#4caf50", "Active"),
        "inactive": ("#78909c", "Inactive"), 
        "warning": ("#ff9800", "Warning"),
        "error": ("#ef5350", "Error"),
        "success": ("#4caf50", "Success")
    }
    
    color, status_text = status_colors.get(status.lower(), status_colors["inactive"])
    
    html = f"""
    <div style="display: flex; align-items: center; gap: 8px;" class="animate-slide-right">
        <div class="status-active" style="
            width: 12px;
            height: 12px;
            background: {color};
            border-radius: 50%;
            color: {color};
        "></div>
        <span style="color: {COLORS['text_secondary']}; font-size: 14px;">{label if label else status_text}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def animated_container(content: str, animation: str = "fadeInUp"):
    """Wrap content in an animated container."""
    animations = {
        "fadeInUp": "fadeInUp 0.5s ease-out forwards",
        "slideInRight": "slideInRight 0.4s ease-out forwards",
        "scaleIn": "scaleIn 0.3s ease-out forwards"
    }
    
    anim = animations.get(animation, animations["fadeInUp"])
    
    html = f"""
    <div style="
        animation: {anim};
        background: linear-gradient(145deg, {COLORS['navy_light']}, {COLORS['navy_mid']});
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    ">
        {content}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def loading_skeleton(width: str = "100%", height: str = "20px"):
    """Render a loading skeleton placeholder."""
    html = f"""
    <div class="skeleton" style="
        width: {width};
        height: {height};
        border-radius: 4px;
    "></div>
    """
    st.markdown(html, unsafe_allow_html=True)
