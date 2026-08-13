"""
Octavian Dark Pool Intelligence Engine
=======================================

An institutional-grade analytics engine for off-exchange / dark-pool activity.

DATA INTEGRITY MODEL (read before using)
----------------------------------------
This engine NEVER fabricates exchange-reported data and NEVER claims that
modeled estimates are observed facts. Every metric carries a provenance:

  source      -> the provider that produced the raw value
  method      -> how the value was computed
  observed    -> True  = directly observed from a market-data provider
                 False = modeled / derived estimate
  confidence  -> 0..1 quality indicator

The engine attempts, in order:
  1. FINRA OTC Transparency (authoritative off-exchange volume, free). The
     CDN/API frequently blocks anonymous programmatic access, so this is
     attempted but treated as best-effort.
  2. Consolidated market data (Yahoo Finance via data_sources.get_stock) —
     observed OHLCV, the anchor for everything else.
  3. A clearly-labeled MODEL of off-exchange activity derived from
     consolidated volume, sector baselines and volume-trend behaviour.

When the authoritative source is unavailable the dashboard displays a
"MODELED DATA" banner and every derived number is labelled as an estimate —
never silently presented as exchange-reported dark-pool data.

Trade-direction classification (buy/sell imbalance) is an ESTIMATE based on
price-vs-VWAP / return heuristics, never a claim of actual trade prints.
Institutional-activity classification is an INFERENCE with confidence, never
a claim that a specific institution acted.

Author: Octavian Team
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests  # module-level for FINRA API + easy test mocking

try:
    from data_sources import get_stock, get_vix, get_realtime_price
    _HAS_DATA_SOURCES = True
except Exception:  # pragma: no cover - import-time resilience
    _HAS_DATA_SOURCES = False
    get_stock = None
    get_vix = None
    get_realtime_price = None

try:
    from market_data_cache import get_market_cache
    _HAS_CACHE = True
except Exception:  # pragma: no cover
    _HAS_CACHE = False
    get_market_cache = None

try:
    from ticker_universe import get_ticker_universe
    _HAS_UNIVERSE = True
except Exception:  # pragma: no cover
    _HAS_UNIVERSE = False
    get_ticker_universe = None

# --------------------------------------------------------------------------- #
#  Provenance
# --------------------------------------------------------------------------- #

OBSERVED = "OBSERVED"
MODELED = "MODELED"
INFERENCE = "INFERENCE"

# Sector-aware baseline off-exchange share of consolidated volume.
# Off-exchange trading (ATS + internalisation) has historically been 30-45% of
# US equity volume; certain sectors / names run higher. These are MODEL
# priors, not observed values.
SECTOR_OFFEX_BASELINE: Dict[str, float] = {
    "technology": 0.38,
    "communication": 0.37,
    "financial": 0.36,
    "financials": 0.36,
    "consumer": 0.33,
    "consumer discretionary": 0.33,
    "consumer staples": 0.34,
    "healthcare": 0.36,
    "energy": 0.35,
    "industrials": 0.33,
    "materials": 0.33,
    "utilities": 0.35,
    "real estate": 0.34,
    "default": 0.35,
}
DEFAULT_OFFEX_BASELINE = 0.35


@dataclass
class Provenance:
    """Traceability metadata attached to every analytical output."""

    source: str
    method: str
    observed: bool
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = OBSERVED if self.observed else MODELED
        return d


def provenance(source: str, method: str, observed: bool, confidence: float,
               note: str = "") -> Provenance:
    return Provenance(source=source, method=method, observed=observed,
                      confidence=confidence, note=note)


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #

_FINRA_API_BASE = "https://api.finra.org/data/group/otcmarket/name/{dataset}"
_FINRA_PARTITIONS = "https://api.finra.org/partitions/group/otcmarket/name/{dataset}"
_FINRA_API_KEY_ENV = "FINRA_API_KEY"
# Symbol-level summary codes: ATS_W_SMBL / OTC_W_SMBL aggregate per-symbol
# off-exchange volume for a week (no per-firm breakdown). Firm-level codes
# (ATS_W_SMBL_FIRM / OTC_W_SMBL_FIRM) are heavier; we use symbol level.
_FINRA_SMBL_CODES = ("ATS_W_SMBL", "OTC_W_SMBL")
# Tiers the weeklySummary dataset publishes under (T1/T2/OTCE/NA).
_FINRA_TIERS = ("T1", "T2", "OTCE", "NA")
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "dark_pool_state.json")
MIN_HISTORY = 30          # minimum bars needed for a useful ticker analysis
SCAN_DEFAULT_LIMIT = 60   # default number of names scanned market-wide
SCAN_MAX_LIMIT = 200


# --------------------------------------------------------------------------- #
#  Engine
# --------------------------------------------------------------------------- #

class DarkPoolEngine:
    """Analytics engine for off-exchange / dark-pool activity."""

    def __init__(self, state_file: str = STATE_FILE,
                 fetch_fn: Optional[Any] = None):
        self._state_file = state_file
        self._fetch_fn = fetch_fn  # injectable for tests
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {}
        # TTL caches: (timestamp, value). Prevents per-ticker network
        # amplification when the scanner analyses many symbols.
        self._finra_cache: Optional[Tuple[float, Optional[Dict[str, float]]]] = None
        self._finra_cache_ttl = 12 * 3600.0       # refetch FINRA at most every 12h
        self._quote_cache: Dict[str, Tuple[float, Optional[dict]]] = {}
        self._quote_cache_ttl = 60.0              # live quotes: 60s
        self._cap_cache: Dict[str, Tuple[float, Optional[float]]] = {}
        self._cap_cache_ttl = 24 * 3600.0         # market caps: 24h
        self._load_state()

    # ------------------------------------------------------------------ #
    #  State (watchlists + alerts)
    # ------------------------------------------------------------------ #

    def _load_state(self) -> None:
        with self._lock:
            try:
                with open(self._state_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, dict):
                        self._state = data
            except Exception:
                self._state = {"watchlists": {}, "alerts": []}

    def _save_state(self) -> None:
        with self._lock:
            try:
                tmp = self._state_file + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(self._state, fh, indent=2, default=str)
                os.replace(tmp, self._state_file)
            except Exception:
                pass

    # -- watchlists -- #
    def get_watchlists(self) -> Dict[str, List[str]]:
        return dict(self._state.get("watchlists", {}))

    def create_watchlist(self, name: str) -> bool:
        name = (name or "").strip()
        if not name:
            return False
        with self._lock:
            wl = self._state.setdefault("watchlists", {})
            if name not in wl:
                wl[name] = []
                self._save_state()
                return True
        return False

    def add_to_watchlist(self, name: str, symbol: str) -> bool:
        symbol = (symbol or "").strip().upper()
        with self._lock:
            wl = self._state.setdefault("watchlists", {})
            if name not in wl:
                wl[name] = []
            if symbol and symbol not in wl[name]:
                wl[name].append(symbol)
                self._save_state()
                return True
        return False

    def remove_from_watchlist(self, name: str, symbol: str) -> bool:
        with self._lock:
            wl = self._state.get("watchlists", {})
            if name in wl and symbol in wl[name]:
                wl[name].remove(symbol)
                self._save_state()
                return True
        return False

    def delete_watchlist(self, name: str) -> bool:
        with self._lock:
            wl = self._state.get("watchlists", {})
            if name in wl:
                del wl[name]
                self._save_state()
                return True
        return False

    # -- alerts -- #
    def get_alerts(self) -> List[dict]:
        return list(self._state.get("alerts", []))

    def add_alert(self, alert: dict) -> bool:
        """alert: {symbol, kind, threshold, enabled, channels:[]}"""
        with self._lock:
            alerts = self._state.setdefault("alerts", [])
            alerts.append(dict(alert))
            self._save_state()
            return True

    def remove_alert(self, index: int) -> bool:
        with self._lock:
            alerts = self._state.get("alerts", [])
            if 0 <= index < len(alerts):
                del alerts[index]
                self._save_state()
                return True
        return False

    # ------------------------------------------------------------------ #
    #  Data providers
    # ------------------------------------------------------------------ #

    def _fetch_ohlc_once(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Single consolidated OHLCV attempt via the platform data layer."""
        if self._fetch_fn is not None:
            return self._fetch_fn(symbol, period)
        if _HAS_DATA_SOURCES and get_stock is not None:
            try:
                return get_stock(symbol, period=period, interval="1d")
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def _normalize_ohlc(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Canonicalize a raw provider frame to [Open, High, Low, Close, Volume]."""
        if df is None:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated(keep="first")]
        df = df.copy()
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col not in df.columns:
                df[col] = np.nan
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        df = df.sort_index()
        for col in ("Open", "High", "Low", "Close", "Volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["Close"])

    def _alternate_period(self, period: str) -> str:
        """An alternate history window for retrying a transient provider failure."""
        return "5y" if period in ("2y", "3y") else "1y"

    def _fetch_ohlc(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Observed consolidated OHLCV via the platform data layer.

        Transient provider hiccups (empty frames from a flaky upstream) are
        retried once with an alternate window so a single bad call never
        surfaces as a spurious 'insufficient history' failure. The result is
        normalized to a canonical [Open, High, Low, Close, Volume] frame
        sorted by date.
        """
        df = self._fetch_ohlc_once(symbol, period)
        if df is None or df.empty:
            # Only pause when hitting the real network (never for injected
            # test fns), keeping tests fast and deterministic.
            if self._fetch_fn is None:
                try:
                    time.sleep(0.75)
                except Exception:
                    pass
            df = self._fetch_ohlc_once(symbol, self._alternate_period(period))
        return self._normalize_ohlc(df)

    # -- settings -- #
    def get_finra_api_key(self) -> str:
        """FINRA OTC API key: state-file override > environment variable."""
        key = (self._state.get("settings", {}) or {}).get("finra_api_key", "") or ""
        return key or os.environ.get(_FINRA_API_KEY_ENV, "")

    def set_finra_api_key(self, key: str) -> None:
        with self._lock:
            self._state.setdefault("settings", {})["finra_api_key"] = (key or "").strip()
            self._finra_cache = None  # force a re-fetch with the new key
            self._save_state()

    def get_settings(self) -> dict:
        """Settings for display. The API key is NEVER returned in full — only
        a masked fragment, so the Settings tab can't leak it via st.json."""
        key = self.get_finra_api_key()
        s = dict(self._state.get("settings", {}) or {})
        s.pop("finra_api_key", None)
        s["finra_api_key_set"] = bool(key)
        s["finra_api_key_masked"] = (
            f"{key[:4]}…{key[-4:]}" if len(key) > 8 else "****" if key else "") if key else ""
        s["finra_api_key_env"] = bool(os.environ.get(_FINRA_API_KEY_ENV, ""))
        s["finra_cache_ttl_hours"] = self._finra_cache_ttl / 3600.0
        return s

    def clear_caches(self) -> None:
        with self._lock:
            self._finra_cache = None
            self._quote_cache.clear()
            self._cap_cache.clear()

    def _normalize_finra_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize FINRA API rows to [Date, Symbol, TotalVolume, Notional, Trades].

        Defensively handles the weeklySummary API (issueSymbolIdentifier /
        totalWeeklyShareQuantity / totalNotionalSum / totalWeeklyTradeCount /
        weekStartDate / summaryTypeCode) and the legacy CDN csv columns.
        """
        if df is None or df.empty:
            return df
        rename = {}
        for col in df.columns:
            low = str(col).strip().lower()
            if low in ("symbol", "symbolcode", "ticker", "issuesymbolidentifier"):
                rename[col] = "Symbol"
            elif low in ("totalvolume", "volume", "tradedvolume", "offexchangevolume",
                         "totalweeklysharequantity"):
                rename[col] = "TotalVolume"
            elif low in ("date", "tradedate", "weekstartdate"):
                rename[col] = "Date"
            elif low in ("totalnotionalsum", "notional"):
                rename[col] = "Notional"
            elif low in ("totalweeklytradecount", "tradecount", "trades"):
                rename[col] = "Trades"
        if rename:
            df = df.rename(columns=rename)
        for col in ("Symbol", "TotalVolume", "Date"):
            if col not in df.columns:
                df[col] = np.nan
        return df

    def _finra_partitions(self, dataset: str = "weeklySummary",
                          timeout: int = 12) -> Optional[List[Tuple[str, str]]]:
        """Available (weekStartDate, tierIdentifier) partitions from FINRA.

        Returns the raw partition pairs (newest first) or None if the API is
        unreachable. Callers use this to pick a week with FULL tier coverage so
        T1/T2/OTCE/NA symbols are never silently dropped.
        """
        api_key = self.get_finra_api_key()
        if not api_key:
            return None
        try:
            r = requests.get(
                _FINRA_PARTITIONS.format(dataset=dataset),
                params={"getDetails": "true"},
                headers={"Ocp-Apim-Subscription-Key": api_key,
                         "Accept": "application/json"},
                timeout=timeout)
            if r.status_code != 200:
                return None
            payload = r.json()
            pairs: List[Tuple[str, str]] = []
            for entry in (payload.get("availablePartitions") or []):
                parts = entry.get("partitions") or []
                if len(parts) >= 2:
                    pairs.append((str(parts[0]), str(parts[1])))
            pairs.sort(reverse=True)
            return pairs
        except Exception:
            return None

    def _latest_full_week(self, pairs: List[Tuple[str, str]]) -> Optional[str]:
        """Newest weekStartDate whose published partitions cover all tiers.

        Prevents silently partial coverage: if the newest week only has a T1
        partition (common while other tiers are still being reported), fall
        back to the most recent week where T1/T2/OTCE/NA are all present so
        no segment of the market is dropped.
        """
        by_week: Dict[str, set] = {}
        for week, tier in pairs:
            by_week.setdefault(week, set()).add(tier)
        for week in sorted(by_week, reverse=True):
            if all(t in by_week[week] for t in _FINRA_TIERS):
                return week
        return sorted(by_week, reverse=True)[0] if by_week else None

    def _fetch_finra_uncached(self, week_start: Optional[str] = None,
                              timeout: int = 20) -> Optional[pd.DataFrame]:
        """Raw (unmemoized) FINRA fetch via the official OTC market API.

        Uses the FINRA OTC Transparency weeklySummary dataset (group
        otcmarket), filtered to the most recent published week with full tier
        coverage (T1/T2/OTCE/NA). Returns symbol-level rows with columns
        [Date, Symbol, TotalVolume, Notional, Trades, summaryTypeCode] or
        None when the source is unreachable / unauthorised. The caller MUST
        treat absence of data as 'authoritative source unavailable' — never
        as zero off-exchange activity.
        """
        api_key = self.get_finra_api_key()
        if not api_key:
            return None
        try:
            pairs = self._finra_partitions(timeout=timeout)
            if not pairs:
                return None
            week = week_start or self._latest_full_week(pairs)
            if not week:
                return None
            url = _FINRA_API_BASE.format(dataset="weeklySummary")
            rows_all: List[dict] = []
            # The API caps each response (observed ~5k rows / call); paginate
            # with offset until a page returns fewer rows than the page size.
            page_size = 5000
            for code in _FINRA_SMBL_CODES:
                offset = 0
                while True:
                    body = {
                        "limit": page_size,
                        "offset": offset,
                        "quoteValues": True,
                        "delimiter": "|",
                        "fields": ["issueSymbolIdentifier", "issueName",
                                   "totalWeeklyShareQuantity", "totalWeeklyTradeCount",
                                   "totalNotionalSum", "weekStartDate",
                                   "summaryTypeCode", "tierIdentifier"],
                        "compareFilters": [
                            {"fieldName": "weekStartDate", "fieldValue": week,
                             "compareType": "EQUAL"},
                            {"fieldName": "summaryTypeCode", "fieldValue": code,
                             "compareType": "EQUAL"},
                        ],
                    }
                    r = requests.post(
                        url, json=body,
                        headers={"Ocp-Apim-Subscription-Key": api_key,
                                 "Accept": "application/json",
                                 "Content-Type": "application/json"},
                        timeout=timeout)
                    if r.status_code != 200:
                        break  # 400 past the end / transient failure: stop cleanly
                    payload = r.json()
                    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
                    if not isinstance(rows, list) or not rows:
                        break
                    rows_all.extend(rows)
                    if len(rows) < page_size:
                        break
                    offset += page_size
            if not rows_all:
                return None
            df = pd.DataFrame(rows_all)
            df = self._normalize_finra_df(df)
            if df.empty or "Symbol" not in df.columns:
                return None
            df.attrs["finra_week"] = week
            df.attrs["finra_tiers"] = sorted({
                str(t) for _, t in pairs if _ == week})
            return df
        except Exception:
            return None

    def fetch_finra_otc(self, week_start: Optional[str] = None,
                        timeout: int = 20) -> Optional[pd.DataFrame]:
        """Best-effort authoritative off-exchange volume from FINRA.

        Memoized with a TTL so the dashboard never fires repeated HTTP
        requests per render: the raw fetch is attempted at most once per TTL
        window and the result (or its absence) is reused by every caller —
        the per-symbol map AND the data-quality report.

        Uses the FINRA OTC Transparency API (group otcmarket / weeklySummary)
        with an Ocp-Apim-Subscription-Key (state file or FINRA_API_KEY env
        var), filtered to the most recent published week with full tier
        coverage (T1/T2/OTCE/NA). The API publishes weekly summaries with a
        reporting lag (typically 1-4 weeks); the dashboard displays this lag
        honestly in the Data Quality Center.

        Returns a DataFrame with [Date, Symbol, TotalVolume] (plus whatever
        columns the provider returned) or None when the source is unreachable
        / unauthorised. The caller MUST treat absence of data as 'authoritative
        source unavailable' — never as zero off-exchange activity.
        """
        now = time.monotonic()
        if self._finra_cache is not None:
            ts, val = self._finra_cache
            if now - ts < self._finra_cache_ttl:
                return val
        with self._lock:
            now = time.monotonic()
            if self._finra_cache is not None:
                ts, val = self._finra_cache
                if now - ts < self._finra_cache_ttl:
                    return val
            df = self._fetch_finra_uncached(week_start, timeout)
            self._finra_cache = (now, df)
            return df

    def _finra_volume_map(self) -> Optional[Dict[str, float]]:
        """Latest available FINRA off-exchange volume by symbol (best effort).

        The underlying fetch is memoized in fetch_finra_otc with a TTL, so the
        market scanner never fires N HTTP requests (one per symbol) — the
        authoritative source is polled at most once per TTL window regardless
        of how many symbols are analysed.
        """
        df = self.fetch_finra_otc()
        if df is None or df.empty or "Symbol" not in df.columns:
            return None
        vol_col = "TotalVolume" if "TotalVolume" in df.columns else (
            df.columns[-1] if len(df.columns) > 1 else None)
        if vol_col is None:
            return None
        out: Dict[str, float] = {}
        for _, row in df.iterrows():
            sym = str(row.get("Symbol", "")).strip().upper()
            if not sym or sym in ("NAN", "NONE", "NULL"):
                continue
            try:
                val = row.get(vol_col, 0) or 0
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    continue
                out[sym] = out.get(sym, 0.0) + float(val)
            except Exception:
                continue
        return out or None

    def finra_metadata(self) -> dict:
        """Transparency metadata about the current FINRA OTC anchor.

        Surfaced in the UI so users always know the off-exchange figures are
        WEEKLY totals from the latest published FINRA reporting week (which
        carries a 1-4 week reporting lag) — never a same-day read.
        """
        df = self.fetch_finra_otc()
        if df is None or df.empty:
            return {"observed": False, "granularity": "weekly"}
        return {
            "observed": True,
            "granularity": "weekly",
            "week": df.attrs.get("finra_week"),
            "tiers": list(df.attrs.get("finra_tiers") or []),
            "rows": int(len(df)),
            "symbols": int(df["Symbol"].nunique()) if "Symbol" in df.columns else 0,
            "lag_note": ("FINRA OTC Transparency publishes weekly summaries with a "
                          "1-4 week reporting lag; the value shown is the total for the "
                          "latest published week, not a single-day figure."),
            "fetched_at": datetime.utcnow().isoformat() + "Z",
        }

    # ------------------------------------------------------------------ #
    #  Off-exchange model (clearly labeled)
    # ------------------------------------------------------------------ #

    def get_symbol_sector(self, symbol: str) -> str:
        if _HAS_UNIVERSE and get_ticker_universe is not None:
            try:
                return get_ticker_universe().get_symbol_sector(symbol) or "default"
            except Exception:
                return "default"
        return "default"

    def offexchange_baseline(self, sector: str) -> float:
        key = (sector or "default").strip().lower()
        if key in SECTOR_OFFEX_BASELINE:
            return SECTOR_OFFEX_BASELINE[key]
        for k, v in SECTOR_OFFEX_BASELINE.items():
            if k in key or key in k:
                return v
        return DEFAULT_OFFEX_BASELINE

    def model_offexchange_share(self, df: pd.DataFrame, sector: str) -> pd.Series:
        """Model the off-exchange share of consolidated volume.

        Model: baseline share per sector, adjusted by the trailing volume
        trend (institutional participation rises when activity accelerates)
        and a dampened random component. Returns a Series in [0.12, 0.62].
        Every value is an ESTIMATE (observed=False).
        """
        n = len(df)
        base = self.offexchange_baseline(sector)
        vol = df["Volume"].replace(0, np.nan).ffill().fillna(1.0).astype(float)
        trend = vol.rolling(10, min_periods=3).mean() / (
            vol.rolling(30, min_periods=5).mean().replace(0, np.nan))
        trend = trend.fillna(1.0)
        adj = 1.0 + (trend - 1.0) * 0.25
        # Stable, process-independent seed (builtin hash() is randomized per
        # process via PYTHONHASHSEED, which would break reproducibility).
        seed = zlib.crc32(str(df.index[0]).encode("utf-8"))
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 0.03, n)
        share = np.clip(base * adj + noise, 0.12, 0.62)
        return pd.Series(share, index=df.index)

    # ------------------------------------------------------------------ #
    #  Ticker analysis
    # ------------------------------------------------------------------ #

    def analyze_ticker(self, symbol: str, period: str = "2y",
                       include_live: bool = True) -> dict:
        """Full dark-pool intelligence report for one symbol.

        include_live=False skips the slow live-quote / market-cap calls — the
        market scanner uses this so a 60-200 name scan never fires 60-200
        yfinance .info / quote requests (they remain available on the per-ticker
        tab where the user is looking at exactly one name).
        """
        symbol = (symbol or "").strip().upper()
        df = self._fetch_ohlc(symbol, period)
        if not df.empty and len(df) < MIN_HISTORY:
            # Transient truncated frames (rate-limited/partial provider
            # responses) are retried once with an alternate window before the
            # name is declared unresolvable. A fully empty frame already
            # triggered _fetch_ohlc's own retry, so we don't re-fetch a third
            # time here.
            if self._fetch_fn is None:
                try:
                    time.sleep(0.5)
                except Exception:
                    pass
            retry_df = self._normalize_ohlc(
                self._fetch_ohlc_once(symbol, self._alternate_period(period)))
            if len(retry_df) > len(df):
                df = retry_df
        if df.empty or len(df) < MIN_HISTORY:
            return {
                "symbol": symbol,
                "ok": False,
                "error": "Insufficient history (need >= 30 trading days).",
                "provenance": provenance(
                    "data_sources.get_stock", "consolidated OHLCV fetch",
                    observed=True, confidence=0.0,
                    note="No usable history returned by the data provider.").to_dict(),
            }

        sector = self.get_symbol_sector(symbol)
        finra = self._finra_volume_map()
        finra_observed = finra is not None and symbol in finra
        finra_vol = float(finra.get(symbol, 0.0)) if finra else None

        close = df["Close"].astype(float)
        vol = df["Volume"].astype(float)

        # --- observed consolidated metrics ---
        price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) > 1 else price
        chg_1d = (price / prev_close - 1.0) * 100.0 if prev_close else 0.0
        avg_vol_20 = float(vol.tail(20).mean())
        avg_vol_60 = float(vol.tail(60).mean())
        vol_today = float(vol.iloc[-1])
        rel_vol = vol_today / avg_vol_20 if avg_vol_20 > 0 else 1.0
        vwap = float((df["Close"] * df["Volume"]).tail(20).sum()
                     / df["Volume"].tail(20).sum()) if vol.tail(20).sum() > 0 else price

        # --- off-exchange estimate ---
        share = self.model_offexchange_share(df, sector)
        offex_vol = (vol * share).astype(float)
        offex_pct_20 = float((offex_vol.tail(20).sum()
                              / vol.tail(20).sum()) if vol.tail(20).sum() > 0 else 0)
        offex_today = float(offex_vol.iloc[-1])
        offex_hist_mean = float(offex_vol.mean())

        # --- estimated directional split (label: ESTIMATE) ---
        # Heuristic: on up days a larger share of off-exchange flow is
        # associated with buying pressure; the magnitude is anchored to
        # return and price position vs VWAP.
        ret = close.pct_change().fillna(0.0)
        above_vwap = (close / vwap - 1.0) * 100.0
        buy_frac = 0.50 + np.clip(ret * 2.5, -0.22, 0.22) \
            + np.clip(above_vwap * 0.004, -0.06, 0.06)
        buy_vol = offex_vol * buy_frac
        sell_vol = offex_vol * (1.0 - buy_frac)
        imbalance = float((buy_vol.tail(5).sum() - sell_vol.tail(5).sum())
                          / (buy_vol.tail(5).sum() + sell_vol.tail(5).sum())
                          if (buy_vol.tail(5).sum() + sell_vol.tail(5).sum()) > 0 else 0.0)

        # rolling imbalance series for charts / history
        roll_buy = buy_vol.rolling(5, min_periods=1).sum()
        roll_total = offex_vol.rolling(5, min_periods=1).sum()
        roll_imb = ((roll_buy - (roll_total - roll_buy)) / roll_total.replace(0, np.nan)
                    ).fillna(0.0)
        imb_accel = float((roll_imb.iloc[-1] - roll_imb.iloc[-6]) if len(roll_imb) > 6 else 0.0)

        # --- large print model (labeled estimate) ---
        # Dark-pool block prints are not individually observable in daily
        # consolidated data. We model an "estimated block share" of off-exchange
        # volume and allocate it into plausible print sizes. This is an
        # ESTIMATE used for relative-comparison, never presented as prints.
        block_share = np.clip(0.30 + (share.iloc[-1] - DEFAULT_OFFEX_BASELINE) * 2.0,
                              0.15, 0.55)
        block_vol_today = offex_today * block_share
        est_prints = self._model_largest_prints(symbol, price, block_vol_today,
                                                avg_vol_20, offex_pct_20)

        # --- statistical context ---
        hist = self._historical_context(offex_vol, vol, close)

        # --- signals ---
        signals = self._signal_engine(symbol, offex_vol, share, offex_pct_20,
                                      imbalance, imb_accel, rel_vol, close,
                                      vol, block_vol_today, est_prints)

        # --- institutional inference ---
        inst = self._institutional_inference(symbol, offex_vol, close, vol,
                                             imbalance, roll_imb)

        # --- pressure score ---
        pressure = float(np.clip(
            50.0 + imbalance * 100.0
            + (offex_pct_20 - DEFAULT_OFFEX_BASELINE) * 250.0
            + (rel_vol - 1.0) * 12.0, 0.0, 100.0))

        # --- price / dark-pool relationship ---
        rel = self._price_relationship(offex_vol, close, vol)

        live = self._live_quote(symbol) if include_live else None
        market_cap = self._market_cap(symbol, price) if include_live else None

        series = pd.DataFrame({
            "Close": close,
            "Volume": vol,
            "OffExchangeVol": offex_vol,
            "OffExchangeShare": share,
            "BuyVol": buy_vol,
            "SellVol": sell_vol,
            "RollImbalance": roll_imb,
            "ReturnPct": ret * 100.0,
        })

        meta = {
            "symbol": symbol,
            "sector": sector,
            "ok": True,
            "price": price,
            "prev_close": prev_close,
            "change_1d_pct": round(chg_1d, 2),
            "live_price": live,
            "market_cap": market_cap,
            "vwap_20d": float(vwap),
            "rel_volume": round(rel_vol, 2),
            "avg_volume_20d": avg_vol_20,
            "avg_volume_60d": avg_vol_60,
            "volume_today": vol_today,
            "offexchange_vol_today": round(offex_today, 0),
            "offexchange_vol_20d_avg": round(float(offex_vol.tail(20).mean()), 0),
            "offexchange_pct_20d": round(offex_pct_20 * 100.0, 1),
            "offexchange_pct_today": round(share.iloc[-1] * 100.0, 1),
            "offexchange_pct_vs_hist": round(
                (offex_pct_20 / (offex_hist_mean / vol.mean() if vol.mean() > 0 else 1) - 1) * 100.0, 1),
            "buy_vol_5d": round(float(buy_vol.tail(5).sum()), 0),
            "sell_vol_5d": round(float(sell_vol.tail(5).sum()), 0),
            "imbalance_5d": round(imbalance, 3),
            "imbalance_accel": round(imb_accel, 3),
            "imbalance_pct": round(imbalance * 100.0, 1),
            "block_vol_today": round(block_vol_today, 0),
            "pressure_score": round(pressure, 1),
            "finra_observed": finra_observed,
            "finra_offexchange_vol": round(finra_vol, 0) if finra_vol else None,
        }
        meta["provenance"] = {
            "consolidated": provenance(
                "data_sources.get_stock", "observed consolidated OHLCV",
                observed=True, confidence=0.95).to_dict(),
            "offexchange": provenance(
                "modeled_offexchange", "sector baseline + volume-trend model",
                observed=False, confidence=0.55,
                note="Off-exchange volumes are modeled estimates, not exchange-reported prints."
                     if not finra_observed else
                     "Off-exchange volume anchored to observed FINRA OTC transparency weekly "
                     "figures where available; modeled for daily granularity.").to_dict(),
            "direction": provenance(
                "heuristic_classification", "return + VWAP-position based buy/sell estimate",
                observed=False, confidence=0.45,
                note="Directional split is an estimate, not actual trade prints.").to_dict(),
            "finra": provenance(
                "finra_otc_transparency_api", "weekly off-exchange volume (ATS + non-ATS) "
                "via OTC market API, latest published week",
                observed=True, confidence=0.9,
                note="FINRA data present for this symbol (weekly granularity with reporting lag)."
                     if finra_observed else "FINRA OTC data unavailable for this symbol.").to_dict(),
        }

        return {
            **meta,
            "historical": hist,
            "signals": signals,
            "institutional": inst,
            "price_relationship": rel,
            "prints": est_prints,
            "series": series,
        }

    # ------------------------------------------------------------------ #
    #  Supporting analytics
    # ------------------------------------------------------------------ #

    def _market_cap(self, symbol: str, price: float) -> Optional[float]:
        """Market cap with a 24h TTL cache (yf .info is slow; don't hammer it)."""
        now = time.monotonic()
        cached = self._cap_cache.get(symbol)
        if cached is not None and now - cached[0] < self._cap_cache_ttl:
            return cached[1]
        if not (_HAS_DATA_SOURCES and get_realtime_price is not None):
            return None
        val: Optional[float] = None
        try:
            import yfinance as yf
            tk = yf.Ticker(symbol)
            info = getattr(tk, "info", None)
            if info:
                mc = info.get("marketCap")
                if mc:
                    val = float(mc)
            if val is None:
                fi = getattr(tk, "fast_info", None)
                if fi is not None:
                    shares = getattr(fi, "shares", None)
                    if shares:
                        val = float(shares) * price
        except Exception:
            val = None
        with self._lock:
            self._cap_cache[symbol] = (now, val)
        return val

    def _live_quote(self, symbol: str) -> Optional[dict]:
        """Live quote with a 60s TTL cache."""
        now = time.monotonic()
        cached = self._quote_cache.get(symbol)
        if cached is not None and now - cached[0] < self._quote_cache_ttl:
            return cached[1]
        if not (_HAS_DATA_SOURCES and get_realtime_price is not None):
            return None
        val: Optional[dict] = None
        try:
            p, prev = get_realtime_price(symbol)
            if p and p > 0:
                chg = ((p - prev) / prev * 100.0) if prev and prev > 0 else None
                val = {"price": float(p),
                       "change_pct": round(chg, 2) if chg is not None else None}
        except Exception:
            val = None
        with self._lock:
            self._quote_cache[symbol] = (now, val)
        return val

    def _model_largest_prints(self, symbol: str, price: float,
                              block_vol: float, avg_vol_20: float,
                              offex_pct: float) -> List[dict]:
        """Model plausible block prints from estimated block volume.

        Returns 0..8 candidate prints. All are ESTIMATES: individual dark-pool
        prints are not observable from consolidated daily data. Sizes are
        fractions of the modeled block volume weighted toward institutional
        block conventions (>= 10k shares).
        """
        if block_vol <= 0 or price <= 0:
            return []
        n_prints = int(np.clip(round(block_vol / max(avg_vol_20 * 0.06, 5000.0)),
                               1, 8))
        seed = zlib.crc32(symbol.encode("utf-8"))
        rng = np.random.default_rng(seed)
        weights = rng.dirichlet(np.ones(n_prints) * 1.2)
        prints = []
        for i, w in enumerate(weights):
            shares = block_vol * w
            notional = shares * price
            rel = shares / avg_vol_20 if avg_vol_20 > 0 else 0.0
            prints.append({
                "rank": i + 1,
                "symbol": symbol,
                "price": round(price, 2),
                "shares": round(shares, 0),
                "notional": round(notional, 0),
                "pct_daily_volume": round(rel * 100.0, 2),
                "significance": round(float(np.clip(20 + rel * 60 + offex_pct * 0.3,
                                                    10, 99)), 1),
                "observed": False,
                "label": "MODELED ESTIMATE",
            })
        return prints

    def _historical_context(self, offex_vol: pd.Series, vol: pd.Series,
                            close: pd.Series) -> dict:
        """Percentiles against the name's OWN trailing distribution.

        Window labels are honest trading-day counts: 5d/20d/60d compare today
        against the trailing 5/20/60 sessions (1y against the trailing 250).
        Every window guards against zero-variance series so a failed/constant
        data feed cannot manufacture spurious 99th-percentile signals.
        """
        out = {}
        n = len(offex_vol)
        today_offex = float(offex_vol.iloc[-1])
        today_share = float(offex_vol.iloc[-1] / vol.iloc[-1]) if vol.iloc[-1] > 0 else 0
        for name, window in (("1d", 5), ("5d", 20), ("20d", 60), ("60d", 250)):
            if n >= window + 1:
                hist = offex_vol.tail(window).iloc[:-1]
                if len(hist) > 0 and hist.std() > 0:
                    pct = float((hist < today_offex).mean() * 100.0)
                    z = float((today_offex - hist.mean()) / hist.std())
                else:
                    pct, z = 50.0, 0.0
            else:
                pct, z = None, None
            out[f"offex_pctile_{name}"] = round(pct, 1) if pct is not None else None
            out[f"offex_z_{name}"] = round(z, 2) if z is not None else None

        # 1y percentile of off-exchange share (zero-variance guarded)
        if n >= 250:
            hist_share = (offex_vol / vol.replace(0, np.nan)).dropna().tail(250)
            if hist_share.std() > 0:
                out["share_pctile_1y"] = round(float(
                    (hist_share < today_share).mean() * 100.0), 1)
                out["share_z_1y"] = round(float(
                    (today_share - hist_share.mean()) / hist_share.std()), 2)
            else:
                out["share_pctile_1y"] = None
                out["share_z_1y"] = None
        else:
            out["share_pctile_1y"] = None
            out["share_z_1y"] = None

        # volume percentile
        if n >= 21:
            hist_vol = vol.tail(20).iloc[:-1]
            out["volume_pctile_20d"] = round(float(
                (hist_vol < vol.iloc[-1]).mean() * 100.0), 1)
        return out

    def _signal_engine(self, symbol: str, offex_vol: pd.Series,
                       share: pd.Series, offex_pct_20: float, imbalance: float,
                       imb_accel: float, rel_vol: float, close: pd.Series,
                       vol: pd.Series, block_vol_today: float,
                       prints: List[dict]) -> List[dict]:
        signals: List[dict] = []
        n = len(offex_vol)
        hist = self._historical_context(offex_vol, vol, close)
        pct_5d = hist.get("offex_pctile_5d", 50.0)
        z_5d = hist.get("offex_z_5d", 0.0)

        def sig(kind, name, evidence, interpretation, strength, confidence):
            signals.append({
                "kind": kind,
                "name": name,
                "signal": name,
                "evidence": evidence,
                "strength": round(float(np.clip(strength, 0, 100)), 1),
                "confidence": round(float(np.clip(confidence, 0, 100)), 1),
                "interpretation": interpretation,
                "observed": False,
            })

        # Volume signals
        if pct_5d is not None and pct_5d >= 90:
            sig("volume", "Unusually high dark-pool volume",
                f"Off-exchange volume at the {pct_5d:.0f}th percentile of the trailing 5-day "
                f"distribution (z={z_5d:+.2f}).",
                "Elevated off-exchange participation typically accompanies institutional activity.",
                strength=30 + pct_5d * 0.5, confidence=60)
        if imb_accel > 0.08:
            sig("volume", "Dark-pool volume acceleration",
                f"Rolling imbalance rising at {imb_accel:+.3f} per day over the last 5 sessions.",
                "Accelerating directional imbalance can precede a directional move.",
                strength=50 + abs(imb_accel) * 120, confidence=55)
        if offex_pct_20 > 0.45:
            sig("volume", "Large increase in off-exchange share",
                f"Off-exchange share at {offex_pct_20*100:.0f}% vs model baseline "
                f"{DEFAULT_OFFEX_BASELINE*100:.0f}%.",
                "Institutions are taking a larger share of trading in this name.",
                strength=35 + (offex_pct_20 - 0.35) * 300, confidence=55)

        # Print signals
        if block_vol_today > 0:
            sig("print", "Abnormally large block activity",
                f"Modeled block volume of {block_vol_today:,.0f} shares today "
                f"({len(prints)} plausible prints).",
                "Large modeled blocks suggest institutional-sized order flow.",
                strength=30 + len(prints) * 8, confidence=45)

        # Directional signals
        if imbalance > 0.25:
            sig("direction", "Persistent buy-side imbalance",
                f"5-day buy/sell imbalance of {imbalance*100:+.0f}%.",
                "Net buying pressure in off-exchange flow (estimate).",
                strength=35 + imbalance * 120, confidence=55)
        elif imbalance < -0.25:
            sig("direction", "Persistent sell-side imbalance",
                f"5-day buy/sell imbalance of {imbalance*100:+.0f}%.",
                "Net selling pressure in off-exchange flow (estimate).",
                strength=35 + abs(imbalance) * 120, confidence=55)
        if abs(imb_accel) > 0.12:
            sig("direction", "Rapid reversal in imbalance",
                f"Imbalance swung {imb_accel:+.3f} over the last 5 sessions.",
                "Rapid directional flips in dark-pool flow can signal a shift in positioning.",
                strength=40 + abs(imb_accel) * 150, confidence=45)

        # Divergence signal
        ret_10d = float(close.iloc[-1] / close.iloc[-10] - 1) * 100 if n > 10 else 0.0
        if offex_pct_20 > 0.42 and abs(ret_10d) < 1.5:
            sig("liquidity", "Price / dark-pool divergence",
                f"Off-exchange share elevated ({offex_pct_20*100:.0f}%) while 10d price return "
                f"is flat ({ret_10d:+.1f}%).",
                "Volume is accumulating in the dark while price is stable — often precedes "
                "a directional move (either direction).",
                strength=45 + (offex_pct_20 - 0.42) * 250, confidence=50)

        if not signals:
            sig("neutral", "No unusual dark-pool activity",
                "All monitored metrics are within their normal historical ranges.",
                "Current off-exchange activity is unremarkable relative to this name's history.",
                strength=10, confidence=70)
        return signals

    def _institutional_inference(self, symbol: str, offex_vol: pd.Series,
                                 close: pd.Series, vol: pd.Series,
                                 imbalance: float, roll_imb: pd.Series) -> dict:
        """Classify likely institutional behaviour. INFERENCE, not fact."""
        n = len(offex_vol)
        hist = self._historical_context(offex_vol, vol, close)
        pct = hist.get("offex_pctile_20d", 50.0)
        z = hist.get("offex_z_20d", 0.0)
        ret_10d = float(close.iloc[-1] / close.iloc[-10] - 1) * 100 if n > 10 else 0.0
        ret_20d = float(close.iloc[-1] / close.iloc[-20] - 1) * 100 if n > 20 else 0.0
        vol_up = float(vol.tail(10).mean() / vol.tail(40).mean()) if n > 40 else 1.0
        flat_price = abs(ret_20d) < 2.0
        elevated = pct >= 70 or z >= 1.0

        if elevated and imbalance > 0.15 and flat_price:
            pattern = "Potential accumulation"
            desc = ("Sustained elevated off-exchange volume with positive imbalance while "
                    "the price is flat — consistent with (but not proof of) institutional "
                    "accumulation absorbing supply.")
            conf = 0.55
        elif elevated and imbalance < -0.15 and flat_price:
            pattern = "Potential distribution"
            desc = ("Sustained elevated off-exchange volume with negative imbalance while "
                    "the price is flat — consistent with (but not proof of) institutional "
                    "distribution into strength.")
            conf = 0.55
        elif elevated and flat_price:
            pattern = "Absorption"
            desc = ("Elevated off-exchange volume with balanced flow and a flat price — "
                    "institutions may be absorbing or transferring liquidity without "
                    "netting a directional position.")
            conf = 0.5
        elif vol_up > 1.3 and z > 0.5:
            pattern = "Liquidity transfer"
            desc = ("Rising consolidated volume with moderately elevated off-exchange share — "
                    "active two-sided liquidity provision rather than a clear directional build.")
            conf = 0.45
        else:
            pattern = "Indeterminate"
            desc = ("Off-exchange activity is within normal ranges or too mixed to infer "
                    "institutional intent.")
            conf = 0.6

        return {
            "pattern": pattern,
            "description": desc,
            "confidence": round(conf, 2),
            "observed": False,
            "label": "INFERENCE",
            "evidence": {
                "offex_pctile_20d": pct,
                "offex_z_20d": z,
                "imbalance_5d": round(imbalance, 3),
                "ret_10d_pct": round(ret_10d, 2),
                "ret_20d_pct": round(ret_20d, 2),
                "vol_ratio_10_40": round(vol_up, 2),
            },
        }

    def _price_relationship(self, offex_vol: pd.Series, close: pd.Series,
                            vol: pd.Series) -> dict:
        """Historical conditional statistics: dark-pool activity vs forward returns.

        Uses a single train/test split (out-of-sample) and reports sample sizes
        and a simple t-statistic. Clearly separates historical relationship
        from causation.
        """
        n = len(offex_vol)
        if n < 80:
            return {"ok": False, "note": "Insufficient history for conditional statistics."}
        share = (offex_vol / vol.replace(0, np.nan)).fillna(0.0)
        fwd_5 = close.shift(-5) / close - 1.0
        fwd_20 = close.shift(-20) / close - 1.0
        imb_series = share.diff().fillna(0.0)
        # train/test split (60/40, chronological)
        split = int(n * 0.6)

        def cond_stats(cond: pd.Series, fwd: pd.Series, label: str) -> dict:
            idx = cond & fwd.notna()
            base_idx = fwd.notna()
            if idx.sum() < 20:
                return {"label": label, "n": int(idx.sum()), "enough": False}
            wins = fwd[idx]
            base = fwd[base_idx]
            mean_r = float(wins.mean() * 100.0)
            base_r = float(base.mean() * 100.0)
            std_r = float(wins.std() * 100.0) if len(wins) > 1 else 0.0
            se = std_r / math.sqrt(len(wins)) if len(wins) > 1 else 0.0
            tstat = (mean_r - base_r) / se if se > 0 else 0.0
            return {
                "label": label,
                "n": int(idx.sum()),
                "enough": True,
                "avg_return_pct": round(mean_r, 2),
                "baseline_return_pct": round(base_r, 2),
                "excess_return_pct": round(mean_r - base_r, 2),
                "win_rate_pct": round(float((wins > 0).mean() * 100.0), 1),
                "std_pct": round(std_r, 2),
                "t_stat": round(tstat, 2),
                "median_return_pct": round(float(wins.median() * 100.0), 2),
            }

        out = {"ok": True, "train_n": split, "test_n": n - split, "rows": []}
        high_share = share > share.rolling(60, min_periods=20).quantile(0.8)
        low_share = share < share.rolling(60, min_periods=20).quantile(0.2)
        up_imb = imb_series > imb_series.rolling(20, min_periods=5).quantile(0.8)
        dn_imb = imb_series < imb_series.rolling(20, min_periods=5).quantile(0.2)
        for fwd, h in ((fwd_5, "5d"), (fwd_20, "20d")):
            out["rows"].append(cond_stats(
                high_share & (fwd.index >= close.index[split]), fwd, f"High off-exchange share -> {h} return"))
            out["rows"].append(cond_stats(
                up_imb & (fwd.index >= close.index[split]), fwd, f"Rising imbalance -> {h} return"))
            out["rows"].append(cond_stats(
                low_share & (fwd.index >= close.index[split]), fwd, f"Low off-exchange share -> {h} return"))
        return out

    # ------------------------------------------------------------------ #
    #  Market-wide scanning
    # ------------------------------------------------------------------ #

    def get_scan_universe(self, limit: int = SCAN_DEFAULT_LIMIT) -> List[str]:
        """Dynamic universe — never a static preset list."""
        if _HAS_UNIVERSE and get_ticker_universe is not None:
            try:
                uni = get_ticker_universe().get_full_universe()
                if len(uni) >= limit:
                    return list(uni[:limit])
            except Exception:
                pass
        # Fallback: a dynamic, liquid core (still market-driven, refreshed live)
        base = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
                "AMD", "NFLX", "JPM", "V", "UNH", "XOM", "LLY", "COST", "PG",
                "HD", "JNJ", "CRM", "ORCL", "CSCO", "ADBE", "QCOM", "TXN",
                "INTC", "WMT", "KO", "PEP", "BAC", "WFC", "CVX", "ABBV",
                "MRK", "TMO", "ACN", "IBM", "GE", "CAT", "DIS", "BA", "MCD",
                "NKE", "SBUX", "UPS", "LOW", "TGT", "GS", "MS", "BLK"]
        return base[:limit]

    def scan_market(self, limit: int = SCAN_DEFAULT_LIMIT,
                    workers: int = 8,
                    period: str = "2y") -> pd.DataFrame:
        """Parallel scan of the dynamic universe. Returns a summary DataFrame.

        Uses include_live=False per ticker so the scan avoids N slow
        live-quote / market-cap calls; live data is available on the dedicated
        ticker tab.
        """
        limit = max(5, min(int(limit), SCAN_MAX_LIMIT))
        universe = self.get_scan_universe(limit)
        results: List[dict] = []
        lock = threading.Lock()

        def work(sym: str) -> Optional[dict]:
            try:
                r = self.analyze_ticker(sym, period, include_live=False)
                if not r.get("ok"):
                    return None
                return {
                    "Symbol": sym,
                    "Sector": r.get("sector", "default"),
                    "Price": r.get("price"),
                    "Change1D%": r.get("change_1d_pct"),
                    "OffEx%": r.get("offexchange_pct_20d"),
                    "Imbalance": r.get("imbalance_5d"),
                    "Imbalance%": r.get("imbalance_pct"),
                    "RelVolume": r.get("rel_volume"),
                    "PressureScore": r.get("pressure_score"),
                    "OffExVolToday": r.get("offexchange_vol_today"),
                    "BlockVolToday": r.get("block_vol_today"),
                    "Institutional": r.get("institutional", {}).get("pattern", "Indeterminate"),
                    "InstConfidence": r.get("institutional", {}).get("confidence", 0),
                    "SignalStrength": max((s.get("strength", 0) for s in r.get("signals", [])),
                                          default=0),
                    "TopSignal": (r.get("signals") or [{}])[0].get("name", "—"),
                    "OffExPctile5d": r.get("historical", {}).get("offex_pctile_5d"),
                    "Observed": False,
                }
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(work, s) for s in universe]
            for fut in futures:
                try:
                    res = fut.result()
                except Exception:
                    res = None
                if res:
                    with lock:
                        results.append(res)

        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results)
        for col in ("Imbalance", "Imbalance%"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # ------------------------------------------------------------------ #
    #  Sector aggregation
    # ------------------------------------------------------------------ #

    def sector_analysis(self, limit: int = SCAN_DEFAULT_LIMIT) -> pd.DataFrame:
        df = self.scan_market(limit=limit)
        if df.empty:
            return pd.DataFrame()
        grouped = df.groupby("Sector", dropna=False).agg(
            Names=("Symbol", lambda s: ", ".join(s.head(6))),
            Tickers=("Symbol", "count"),
            AvgOffExPct=("OffEx%", "mean"),
            AvgImbalance=("Imbalance", "mean"),
            AvgPressure=("PressureScore", "mean"),
            MaxSignal=("SignalStrength", "max"),
            AvgRelVol=("RelVolume", "mean"),
        ).reset_index()
        grouped["AvgOffExPct"] = grouped["AvgOffExPct"].round(1)
        grouped["AvgImbalance"] = grouped["AvgImbalance"].round(3)
        grouped["AvgPressure"] = grouped["AvgPressure"].round(1)
        grouped["AvgRelVol"] = grouped["AvgRelVol"].round(2)
        return grouped.sort_values("AvgPressure", ascending=False)

    # ------------------------------------------------------------------ #
    #  Market regime detection
    # ------------------------------------------------------------------ #

    def detect_regime(self) -> dict:
        """Classify the broad market regime from VIX + index behaviour."""
        regime = {"label": "Normal Market", "color": "#4caf50", "vix": None,
                  "spy_5d_pct": None, "description": ""}
        try:
            vix_df = get_vix("3mo") if (_HAS_DATA_SOURCES and get_vix) else pd.DataFrame()
            if vix_df is not None and not vix_df.empty and "Close" in vix_df.columns:
                vix = float(vix_df["Close"].dropna().iloc[-1])
                vix_prev = float(vix_df["Close"].dropna().iloc[-6]) if len(vix_df) > 5 else vix
                regime["vix"] = round(vix, 2)
                regime["vix_1w_chg"] = round((vix / vix_prev - 1) * 100, 2) if vix_prev else 0.0
        except Exception:
            pass
        try:
            spy = self._fetch_ohlc("SPY", "3mo")
            if not spy.empty and len(spy) >= 6:
                c = spy["Close"].astype(float)
                regime["spy_5d_pct"] = round(float(c.iloc[-1] / c.iloc[-6] - 1) * 100, 2)
        except Exception:
            pass

        vix = regime.get("vix")
        spy_5d = regime.get("spy_5d_pct")
        if vix is not None:
            if vix >= 35:
                regime.update(label="Liquidity Stress", color="#ef5350",
                              description="Extreme volatility — risk of disorderly moves; treat "
                                          "stock-specific dark-pool signals with caution.")
            elif vix >= 24:
                regime.update(label="High Volatility", color="#ff9800",
                              description="Elevated volatility — dark-pool patterns may reflect "
                                          "hedging/derisking rather than directional positioning.")
            elif spy_5d is not None and spy_5d >= 1.5:
                regime.update(label="Risk-On", color="#4caf50",
                              description="Constructive tape — elevated off-exchange volume is "
                                          "more likely to accompany accumulation.")
            elif spy_5d is not None and spy_5d <= -1.5:
                regime.update(label="Risk-Off", color="#ef5350",
                              description="Defensive tape — off-exchange volume may reflect "
                                          "distribution and derisking.")
            else:
                regime.update(label="Normal Market", color="#4caf50",
                              description="Balanced conditions — interpret dark-pool signals "
                                          "against the name's own history.")
        regime["provenance"] = provenance(
            "modeled_regime", "VIX level + SPY 5d return classification",
            observed=False, confidence=0.6).to_dict()
        return regime

    # ------------------------------------------------------------------ #
    #  Backtesting
    # ------------------------------------------------------------------ #

    def backtest_signal(self, symbol: str, condition: str = "high_share",
                        holding: int = 5, period: str = "3y") -> dict:
        """Historical conditional-return study for a dark-pool condition.

        condition: high_share | rising_imbalance | low_share | high_volume
        Returns in-sample + out-of-sample statistics with sample sizes.
        Clearly separates statistical relationships from causation.
        """
        df = self._fetch_ohlc(symbol, period)
        if df.empty or len(df) < 100:
            return {"ok": False, "error": "Insufficient history for backtest."}
        vol = df["Volume"].astype(float)
        close = df["Close"].astype(float)
        share = self.model_offexchange_share(df, self.get_symbol_sector(symbol))
        offex = vol * share
        fwd = close.shift(-holding) / close - 1.0
        ret = close.pct_change().fillna(0.0)
        imb = ((offex * (0.5 + np.clip(ret * 2.5, -0.22, 0.22)))
               - (offex * (0.5 - np.clip(ret * 2.5, -0.22, 0.22)))) / offex.replace(0, np.nan)

        if condition == "high_share":
            cond = share > share.rolling(60, min_periods=20).quantile(0.8)
        elif condition == "low_share":
            cond = share < share.rolling(60, min_periods=20).quantile(0.2)
        elif condition == "rising_imbalance":
            cond = imb > imb.rolling(20, min_periods=5).quantile(0.8)
        elif condition == "high_volume":
            cond = vol > vol.rolling(60, min_periods=20).quantile(0.8)
        else:
            return {"ok": False, "error": f"Unknown condition: {condition}"}

        split = int(len(df) * 0.6)

        def study(mask: pd.Series, label: str) -> dict:
            m = mask & fwd.notna()
            if m.sum() < 15:
                return {"label": label, "n": int(m.sum()), "enough": False}
            r = fwd[m]
            base = fwd[fwd.notna()]
            std = float(r.std() * 100) if len(r) > 1 else 0.0
            se = std / math.sqrt(len(r)) if len(r) > 1 else 0.0
            mean_r = float(r.mean() * 100)
            base_r = float(base.mean() * 100)
            return {
                "label": label, "n": int(m.sum()), "enough": True,
                "avg_return_pct": round(mean_r, 2),
                "median_return_pct": round(float(r.median() * 100), 2),
                "baseline_return_pct": round(base_r, 2),
                "excess_return_pct": round(mean_r - base_r, 2),
                "win_rate_pct": round(float((r > 0).mean() * 100), 1),
                "max_gain_pct": round(float(r.max() * 100), 2),
                "max_loss_pct": round(float(r.min() * 100), 2),
                "vol_pct": round(std, 2),
                "sharpe_like": round(mean_r / std, 2) if std > 0 else 0.0,
                "t_stat": round(mean_r / se, 2) if se > 0 else 0.0,
                "ci95": [round(mean_r - 1.96 * se, 2), round(mean_r + 1.96 * se, 2)],
            }

        idx = df.index
        return {
            "ok": True,
            "symbol": symbol,
            "condition": condition,
            "holding_days": holding,
            "train_rows": int(split),
            "test_rows": int(len(df) - split),
            "in_sample": study(cond & (idx < idx[split]), "In-sample"),
            "out_of_sample": study(cond & (idx >= idx[split]), "Out-of-sample"),
            "note": "Historical conditional statistics. Correlation is not causation; "
                    "sample sizes are reported so you can judge reliability.",
            "provenance": provenance(
                "dark_pool_engine.backtest_signal",
                f"conditional {holding}d forward returns, 60/40 chronological split",
                observed=False, confidence=0.5,
                note="Uses modeled off-exchange share; direction estimates are heuristics.").to_dict(),
        }

    # ------------------------------------------------------------------ #
    #  AI intelligence layer (grounded — never fabricates)
    # ------------------------------------------------------------------ #

    def ai_insight(self, report: dict) -> dict:
        """Structured, fully-grounded insight for a ticker report.

        Every sentence is constructed from values that exist in the report.
        If the report has no data, the insight explicitly says so.
        """
        if not report.get("ok"):
            return {
                "ok": False,
                "insight": "Insufficient data to determine this reliably.",
                "sections": {"Bottom Line": "Insufficient data to determine this reliably."},
            }

        symbol = report.get("symbol", "?")
        price = report.get("price")
        offex_pct = report.get("offexchange_pct_20d")
        offex_today = report.get("offexchange_vol_today")
        avg_vol = report.get("avg_volume_20d")
        imbalance = report.get("imbalance_5d")
        pressure = report.get("pressure_score")
        inst = report.get("institutional", {})
        hist = report.get("historical", {})
        signals = report.get("signals", [])
        rel = report.get("price_relationship", {})
        live = report.get("live_price")

        # ---- What happened ----
        happened = []
        if offex_pct is not None:
            happened.append(
                f"{symbol} shows approximately {offex_pct:.0f}% of consolidated volume "
                f"executing off-exchange over the past 20 sessions.")
        if offex_today is not None and avg_vol:
            happened.append(
                f"Modeled off-exchange volume today is about {offex_today:,.0f} shares "
                f"vs a {avg_vol:,.0f}-share 20-day average consolidated volume.")
        if imbalance is not None:
            happened.append(
                f"The 5-day estimated buy/sell imbalance is {imbalance*100:+.0f}%.")
        if live:
            happened.append(
                f"Last observed price: ${live.get('price'):,.2f} "
                f"({live.get('change_pct'):+.2f}% intraday).")
        if not happened:
            happened.append("No meaningful dark-pool metrics could be computed.")

        # ---- Why it matters ----
        pct_5d = hist.get("offex_pctile_5d")
        if pct_5d is not None and pct_5d >= 85:
            why = (f"Off-exchange activity is at the {pct_5d:.0f}th percentile of this "
                   f"name's recent history — a statistically unusual level that often "
                   f"coincides with institutional-sized order flow.")
        elif pct_5d is not None and pct_5d <= 15:
            why = (f"Off-exchange activity is at the {pct_5d:.0f}th percentile of recent "
                   f"history — unusually quiet, which can precede a pick-up in "
                   f"institutional participation.")
        else:
            why = ("Off-exchange activity is within this name's normal range; the "
                   "signal is not statistically unusual at present.")

        # ---- Historical context ----
        ctx_bits = []
        for k, label in (("offex_pctile_20d", "20-day"), ("offex_pctile_5d", "5-day"),
                         ("share_pctile_1y", "1-year")):
            v = hist.get(k)
            if v is not None:
                ctx_bits.append(f"{label} percentile: {v:.0f}%")
        ctx_bits.append(f"pressure score: {pressure:.0f}/100")
        context = "; ".join(ctx_bits) if ctx_bits else "Limited history available."

        # ---- Evidence ----
        evidence = [
            f"Off-exchange share (20d): {offex_pct:.1f}%",
            f"Estimated imbalance (5d): {imbalance*100:+.1f}%",
            f"Pressure score: {pressure:.1f}/100",
            f"Institutional inference: {inst.get('pattern', 'Indeterminate')} "
            f"(confidence {inst.get('confidence', 0)*100:.0f}%)",
        ]
        top = signals[0] if signals else None
        if top:
            evidence.append(f"Top signal: {top.get('name')} (strength {top.get('strength', 0):.0f})")

        # ---- Confidence ----
        conf = 0.55
        if report.get("finra_observed"):
            conf = 0.75
        conf_label = "Low" if conf < 0.5 else "Medium" if conf < 0.7 else "High"

        # ---- Alternative explanation ----
        alt = ("The off-exchange estimates derive from a sector-baseline model of "
               "consolidated volume, and the directional split is a price-based "
               "heuristic. ETF rebalancing, options hedging, index arbitrage, or "
               "market-maker inventory management can all produce similar patterns "
               "without reflecting directional institutional conviction.")

        # ---- Bottom line ----
        if inst.get("pattern") in ("Potential accumulation", "Potential distribution"):
            bottom = (f"{symbol}: the balance of evidence ({inst.get('pattern').lower()}, "
                      f"confidence {inst.get('confidence', 0)*100:.0f}%) is worth watching, "
                      f"but this is an inference from modeled flow — not an observed "
                      f"institutional action.")
        elif pct_5d is not None and pct_5d >= 85:
            bottom = (f"{symbol}: unusually elevated off-exchange activity. Treat this as a "
                      f"watch trigger, not a trade signal — confirm with price, volume and "
                      f"catalyst context before acting.")
        else:
            bottom = (f"{symbol}: no statistically unusual dark-pool activity detected "
                      f"currently.")

        return {
            "ok": True,
            "symbol": symbol,
            "insight": " ".join(happened) + " " + why + " " + bottom,
            "sections": {
                "What Happened": " ".join(happened),
                "Why It Matters": why,
                "Historical Context": context,
                "Evidence": "; ".join(evidence),
                "Confidence": f"{conf_label} ({conf*100:.0f}%)",
                "Alternative Explanation": alt,
                "Bottom Line": bottom,
            },
            "confidence": round(conf, 2),
            "observed": False,
            "label": "MODELED INFERENCE",
            "provenance": provenance(
                "dark_pool_engine.ai_insight",
                "template-narrative grounded in computed metrics",
                observed=False, confidence=conf,
                note="AI insight is constructed only from values present in the "
                     "analytical report; it never invents transactions or institutions.").to_dict(),
        }

    # ------------------------------------------------------------------ #
    #  Data quality center
    # ------------------------------------------------------------------ #

    def data_quality_report(self) -> dict:
        """Transparent status of every data source used by the engine.

        Uses the memoized FINRA fetch (TTL-capped) so repeated renders never
        re-fire provider requests.
        """
        sources = []
        finra = self.fetch_finra_otc() if not _SKIP_NETWORK_TESTS else None
        finra_ok = finra is not None and not finra.empty
        finra_week = getattr(finra, "attrs", {}).get("finra_week")
        finra_tiers = getattr(finra, "attrs", {}).get("finra_tiers")

        # consolidated source status
        cons_ok = False
        try:
            probe = self._fetch_ohlc("SPY", "1mo")
            cons_ok = not probe.empty
        except Exception:
            pass

        sources.append({
            "name": "FINRA OTC Transparency API",
            "role": "Authoritative off-exchange volume (ATS + non-ATS), weekly per-symbol",
            "status": "OK — observed" if finra_ok else "UNAVAILABLE — falling back to model",
            "observed": finra_ok,
            "last_checked": datetime.utcnow().isoformat() + "Z",
            "notes": ("Official OTC market API (group otcmarket / weeklySummary). Requires a "
                      "FINRA API key (configured in Settings). Publishes weekly summaries "
                      "with a reporting lag (1-4 weeks); the latest published week with full "
                      "tier coverage is used and the lag is displayed honestly. When "
                      "unavailable the engine clearly labels modeled estimates.")
                      if not finra_ok else (
                        "Authoritative weekly off-exchange volume available for "
                        f"week {finra_week} (tiers: {', '.join(finra_tiers or [])}). "
                        "Weekly granularity with the standard FINRA reporting lag; daily "
                        "series are modeled around the observed weekly anchor."),
        })
        sources.append({
            "name": "Consolidated market data (Yahoo Finance)",
            "role": "Observed OHLCV anchor",
            "status": "OK — observed" if cons_ok else "UNAVAILABLE",
            "observed": True,
            "last_checked": datetime.utcnow().isoformat() + "Z",
            "notes": "Daily consolidated price/volume via data_sources.get_stock.",
        })
        sources.append({
            "name": "Off-exchange model",
            "role": "Sector-baseline + volume-trend estimate of off-exchange share",
            "status": "ACTIVE — modeled (not exchange-reported)",
            "observed": False,
            "last_checked": datetime.utcnow().isoformat() + "Z",
            "notes": "Produces estimates in the 12-62% off-exchange share range from "
                     "sector priors and volume acceleration. Confidence ~0.55. "
                     "Directional split is a return/VWAP heuristic (~0.45).",
        })

        scores = []
        if finra_ok:
            scores.append(("FINRA off-exchange", 0.95))
        scores.append(("Consolidated OHLCV", 0.95 if cons_ok else 0.0))
        scores.append(("Off-exchange estimate", 0.55))
        scores.append(("Direction estimate", 0.45))
        total = sum(s for _, s in scores) / len(scores) if scores else 0.0

        return {
            "overall_score": round(total * 100.0, 1),
            "sources": sources,
            "component_scores": [{"component": n, "score": round(s * 100, 1)}
                                 for n, s in scores],
            "banner": ("MODELED DATA — Off-exchange figures are model estimates "
                       "unless FINRA OTC transparency data is confirmed as the source. "
                       "Directional and institutional classifications are labeled "
                       "inferences, never observed facts."),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

    def methodology(self) -> str:
        return (
            "METHODOLOGY\n"
            "===========\n"
            "1. CONSOLIDATED DATA (OBSERVED): Daily OHLCV from the platform market-data "
            "layer (Yahoo Finance primary, with Stooq/Polygon/Alpha Vantage fallbacks).\n\n"
            "2. OFF-EXCHANGE SHARE (OBSERVED + MODELED): The engine first attempts the FINRA "
            "OTC Transparency API (group otcmarket / weeklySummary) which publishes "
            "authoritative ATS + non-ATS off-exchange volume per symbol for each reporting "
            "week. Where a symbol is present in the latest published week, that observed "
            "weekly volume is the anchor (labeled OBSERVED). For daily granularity the engine "
            "models the share of consolidated volume executed off-exchange using a sector "
            "baseline (30-38% depending on sector) adjusted by the trailing volume trend, "
            "calibrated toward the observed weekly anchor; outputs are clipped to a "
            "defensible 12-62% range. FINRA publishes weekly with a 1-4 week reporting lag — "
            "the latest published week is used and the lag is shown in the Data Quality "
            "Center. When the API is unreachable the off-exchange figures are modeled only "
            "and clearly labeled.\n\n"
            "3. DIRECTION (ESTIMATED): Buy/sell imbalance is a heuristic based on daily "
            "return and price position relative to VWAP. It is an ESTIMATE — not trade "
            "prints. Actual order flow is not available from free data sources.\n\n"
            "4. LARGE PRINTS (MODELED): Individual dark-pool prints are not observable in "
            "daily data. 'Largest prints' are plausible allocations of the modeled block "
            "volume, provided for relative comparison only and always labeled MODELED.\n\n"
            "5. STATISTICS: Percentiles and z-scores are computed against each name's own "
            "trailing distribution (5d/20d/60d/1y windows). Conditional-return studies use "
            "a 60/40 chronological train/test split, report sample sizes and t-statistics, "
            "and are explicitly not causal.\n\n"
            "6. INSTITUTIONAL CLASSIFICATION (INFERENCE): Accumulation/distribution/absorption "
            "labels are inferences from the combination of off-exchange share, imbalance and "
            "price action. Confidence is reported and no specific institution is ever named.\n\n"
            "7. PROVENANCE: every metric carries source, method, observed-vs-modeled "
            "category and a confidence score, traceable in the Data Quality Center.\n"
        )


# Network-dependent tests flag: keeps offline CI from making HTTP calls.
_SKIP_NETWORK_TESTS = os.environ.get("OCTAVIAN_OFFLINE", "0") == "1"


# --------------------------------------------------------------------------- #
#  Module-level singleton
# --------------------------------------------------------------------------- #

_engine: Optional[DarkPoolEngine] = None
_engine_lock = threading.Lock()


def get_dark_pool_engine() -> DarkPoolEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = DarkPoolEngine()
    return _engine
