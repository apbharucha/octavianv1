import pandas as pd
import streamlit as st
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_sources import get_stock

# No hardcoded SECTOR_MAP: sector composition is sourced from the dynamic
# self-expanding ticker universe (seed + comprehensive-universe bridge + daily
# S&P 500 / NASDAQ-100 / ETF-holdings expansion), so the scan always reflects
# the currently known constituents instead of a frozen preset list.

# Human-readable display names for the universe's sector keys.
_SECTOR_DISPLAY = {
    "technology": "Technology",
    "semiconductors": "Semiconductors",
    "cybersecurity": "Cybersecurity",
    "healthcare": "Healthcare",
    "biotech": "Biotech",
    "financials": "Financials",
    "banks": "Banks",
    "consumer": "Consumer",
    "consumer_staples": "Consumer Staples",
    "energy": "Energy",
    "clean_energy": "Clean Energy",
    "uranium": "Uranium",
    "mining": "Mining",
    "minerals": "Minerals",
    "rare_earth": "Rare Earth",
    "gold": "Gold",
    "silver": "Silver",
    "copper": "Copper",
    "lithium": "Lithium",
    "steel": "Steel",
    "materials": "Materials",
    "industrials": "Industrials",
    "defense": "Defense",
    "real_estate": "Real Estate",
    "utilities": "Utilities",
    "communication": "Communication",
    "agriculture": "Agriculture",
    "shipping": "Shipping",
    "cannabis": "Cannabis",
    "solar": "Solar",
    "ev": "EV",
    "space": "Space",
    "international": "International",
    "crypto_equities": "Crypto Equities",
    "special": "Special Situations",
}


def get_dynamic_sector_map() -> dict:
    """Build the sector map from the live dynamic universe.

    Returns {display_name: [tickers]} sourced from TickerUniverse.get_all_sectors(),
    which self-expands daily. Never a frozen preset list. Falls back to an empty
    map (never a fake preset) if the universe is unavailable.
    """
    try:
        from ticker_universe import get_ticker_universe

        universe = get_ticker_universe()
        raw = universe.get_all_sectors()
        if not raw:
            return {}
        out = {}
        for key, tickers in raw.items():
            if not tickers:
                continue
            display = _SECTOR_DISPLAY.get(str(key).lower(), str(key).replace("_", " ").title())
            out[display] = list(tickers)
        return out
    except Exception:
        return {}


def _fetch_symbol_score(sym, lookback):
    """Helper to fetch score for a single symbol."""
    try:
        df = get_stock(sym)
        if df is None or df.empty or len(df) < lookback:
            return None

        close_col = df["Close"]
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]

        current = float(close_col.iloc[-1])
        prev = float(close_col.iloc[-lookback])

        # Avoid division by zero
        if prev <= 0:
            return None

        return ((current / prev) - 1) * 100
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def scan_sectors(lookback=21):
    rows = []

    sector_map = get_dynamic_sector_map()
    if not sector_map:
        return pd.DataFrame(columns=["Sector", "TrendScore", "AssetsUsed"])

    # 1. Collect all unique symbols (dynamic, market-driven)
    all_symbols = []
    for symbols in sector_map.values():
        all_symbols.extend(symbols)
    all_symbols = list(set(all_symbols))

    # 2. Bulk download data (much faster than loop) — chunked so large
    #    dynamic universes don't fail on one giant yfinance request.
    data = None
    try:
        _CHUNK = 150
        frames = []
        for i in range(0, len(all_symbols), _CHUNK):
            chunk = all_symbols[i:i + _CHUNK]
            try:
                d = yf.download(chunk, period="3mo", group_by='ticker',
                                progress=False, threads=True)
                if d is not None and not d.empty:
                    frames.append(d)
            except Exception:
                continue
        if frames:
            data = pd.concat(frames, axis=1)
    except Exception:
        data = None

    if data is None or data.empty:
        return pd.DataFrame(columns=["Sector", "TrendScore", "AssetsUsed"])

    # 3. Process each sector
    for sector, symbols in sector_map.items():
        scores = []
        for sym in symbols:
            try:
                # Handle yfinance multi-index columns
                if isinstance(data.columns, pd.MultiIndex):
                    if sym not in data.columns.levels[0]:
                        continue
                    df = data[sym].dropna()
                else:
                    if sym != all_symbols[0]:
                        continue
                    df = data.dropna()

                if df.empty or len(df) < lookback:
                    continue

                # Get close prices
                close_col = df["Close"]
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]

                current = float(close_col.iloc[-1])
                prev = float(close_col.iloc[-lookback])

                if prev > 0:
                    ret = ((current / prev) - 1) * 100
                    scores.append(ret)
            except Exception:
                continue

        if len(scores) > 0:
            rows.append({
                "Sector": sector,
                "TrendScore": round(sum(scores) / len(scores), 2),
                "AssetsUsed": len(scores)
            })

    if len(rows) == 0:
        return pd.DataFrame(columns=["Sector", "TrendScore", "AssetsUsed"])

    return pd.DataFrame(rows).sort_values("TrendScore", ascending=False)
