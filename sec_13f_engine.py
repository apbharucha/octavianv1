"""
SEC 13F Engine — Comprehensive Institutional Holdings Analyzer
Tracks equities, options (calls/puts), bonds/fixed income, commodities, ETFs,
and all asset classes that institutions hold via SEC 13F filings.
Author: Octavian Research
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
import requests
import streamlit as st

try:
    from financial_llm_engine import _call_llm
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PROVENANCE CONSTANTS
# ---------------------------------------------------------------------------

# Every filing carries an explicit provenance. Simulated data is ONLY produced
# in opt-in demo mode and is never presented as real SEC data.
PROVENANCE_REAL = "SEC EDGAR"
PROVENANCE_SIMULATED = "SIMULATED ESTIMATE (not actual filing data)"
PROVENANCE_UNAVAILABLE = "UNAVAILABLE"

# EDGAR requests are cached for 24h to respect SEC rate limits and avoid
# re-fetching the same filings on every UI render (the UI calls the engine
# multiple times per page load).
_EDGAR_CACHE_TTL_S = 24 * 3600
_EDGAR_CACHE_PATH = Path(__file__).resolve().parent / ".edgar_filings_cache.json"
_EDGAR_CACHE: Dict[str, Any] = {}
_EDGAR_CACHE_LOCK = threading.Lock()

# SEC User-Agent policy requires a contact identity for data.sec.gov.
_EDGAR_HEADERS = {
    "User-Agent": "Octavian Research (contact: research@octavian.local)",
    "Accept-Encoding": "gzip, deflate",
}


def _load_edgar_cache():
    global _EDGAR_CACHE
    try:
        if _EDGAR_CACHE_PATH.exists():
            with open(_EDGAR_CACHE_PATH, "r") as f:
                _EDGAR_CACHE = json.load(f)
    except Exception:
        _EDGAR_CACHE = {}


def _save_edgar_cache():
    try:
        with open(_EDGAR_CACHE_PATH, "w") as f:
            json.dump(_EDGAR_CACHE, f)
    except Exception:
        pass


_load_edgar_cache()

# ---------------------------------------------------------------------------
# CIK MAP
# ---------------------------------------------------------------------------

FUND_CIKS = {
    "Renaissance Technologies LLC": "1037389",
    "Citadel Advisors LLC": "1423053",
    "Bridgewater Associates, LP": "1350694",
    "Point72 Asset Management, L.P.": "1603466",
    "Berkshire Hathaway Inc": "1067983",
    "Two Sigma Investments, LP": "1179392",
    "Millennium Management LLC": "1273931",
    "D.E. Shaw & Co., L.P.": "1009207",
    "Tiger Global Management LLC": "1167483",
    "Appaloosa Management LP": "1070154",
}

# ---------------------------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------------------------

@dataclass
class OptionPosition:
    symbol: str
    contract_type: str    # "CALL" or "PUT"
    strike: float
    expiry: str
    contracts: int
    notional_value: float
    action: str
    pct_portfolio: float


@dataclass
class BondPosition:
    issuer: str
    bond_type: str        # "Treasury", "Corporate IG", "Corporate HY", "Municipal", "Agency"
    maturity: str
    coupon: float
    face_value: float
    market_value: float
    action: str
    pct_portfolio: float


@dataclass
class CommodityPosition:
    commodity: str
    instrument: str       # "ETF", "Futures", "Physical"
    ticker: str
    notional_value: float
    action: str
    pct_portfolio: float


@dataclass
class PositionChange:
    symbol: str
    asset_class: str      # "Equity", "ETF", "Option", "Bond", "Commodity", "Other"
    action: str           # "NEW", "ADD", "REDUCE", "EXIT", "HOLD"
    shares_changed: int
    value_changed: float
    pct_portfolio: float
    rationale: str = ""
    # Provenance metadata for a single position. `is_resolved` is False when
    # the underlying symbol could not be mapped from the filing (13F tables do
    # not always carry tickers; positions are then shown by issuer name).
    cusip: str = ""
    issuer_name: str = ""
    is_resolved: bool = True


@dataclass
class InstitutionalFiling:
    fund_name: str
    filing_date: str
    report_period: str
    total_aum: float
    top_buys: List[PositionChange]
    top_sells: List[PositionChange]
    sector_allocation: Dict[str, float]
    asset_class_allocation: Dict[str, float]
    option_positions: List[OptionPosition]
    bond_positions: List[BondPosition]
    commodity_positions: List[CommodityPosition]
    put_call_ratio: float
    options_sentiment: str
    bond_duration: float
    ai_insight: str = ""
    # Provenance: ALWAYS populated so consumers/UI can distinguish real SEC
    # data from simulated estimates and unavailable states.
    data_source: str = PROVENANCE_REAL
    is_simulated: bool = False
    data_available: bool = True

    def provenance_label(self) -> str:
        """Human-readable provenance string shown in the UI."""
        if self.is_simulated:
            return PROVENANCE_SIMULATED
        if not self.data_available:
            return PROVENANCE_UNAVAILABLE
        return self.data_source


# ---------------------------------------------------------------------------
# MAIN ENGINE CLASS
# ---------------------------------------------------------------------------

class SEC13FEngine:
    """
    Comprehensive engine to track, analyze, and synthesize institutional 13F filings.
    Covers equities, options, bonds, commodities, and all asset classes.
    """

    def __init__(self):
        self.has_llm = HAS_LLM
        self.top_funds = list(FUND_CIKS.keys())

    # -----------------------------------------------------------------------
    # EDGAR FETCHING (real SEC data)
    # -----------------------------------------------------------------------

    _COMPANY_TICKERS_CACHE: Optional[Dict[str, str]] = None
    _COMPANY_TICKERS_TS: float = 0.0

    def _fetch_from_edgar(self, cik: str) -> Optional[dict]:
        """
        Fetch the two most recent 13F-HR filings from SEC EDGAR for a CIK.

        Returns real parsed holdings (with CUSIPs / issuer names), the current
        and prior filing dates, and the prior-period value map so actions
        (NEW / ADD / REDUCE / EXIT) can be computed from actual data. Returns
        None on any failure. Results are cached for 24h to respect SEC rate
        limits (data.sec.gov enforces ~10 requests/second).
        """
        cache_key = f"edgar:{cik}"
        cached = _EDGAR_CACHE.get(cache_key)
        if cached and (time.time() - cached.get("ts", 0)) < _EDGAR_CACHE_TTL_S:
            return cached.get("data")

        try:
            cik_int = int(cik)
            meta_url = f"https://data.sec.gov/submissions/CIK{cik_int:010d}.json"
            resp = requests.get(meta_url, headers=_EDGAR_HEADERS, timeout=15)
            if resp.status_code != 200:
                return None
            meta = resp.json()

            filings = meta.get("filings", {}).get("recent", {})
            forms = filings.get("form", [])
            accessions = filings.get("accessionNumber", [])
            dates = filings.get("filingDate", [])
            periods = filings.get("reportDate", [])
            docs = filings.get("primaryDocument", [])

            targets: List[Tuple[str, str, str, Optional[str]]] = []
            for form, acc, date, period, doc in zip(forms, accessions, dates, periods, docs):
                if form == "13F-HR":
                    targets.append((acc.replace("-", ""), date, period or date, doc))
                    if len(targets) == 2:
                        break

            if not targets:
                return None

            def _fetch_infotable(acc: str, doc: Optional[str]) -> Optional[dict]:
                doc = doc or "infotable.json"
                info_url = (
                    f"https://data.sec.gov/Archives/edgar/data/{cik_int}/"
                    f"{acc}/{doc}"
                )
                r = requests.get(info_url, headers=_EDGAR_HEADERS, timeout=20)
                if r.status_code != 200:
                    return None
                return r.json()

            current_acc, current_date, current_period, current_doc = targets[0]
            prior = targets[1] if len(targets) > 1 else None

            current_holdings = _fetch_infotable(current_acc, current_doc)
            if not current_holdings:
                return None

            prior_map: Dict[str, float] = {}
            if prior:
                prior_holdings = _fetch_infotable(prior[0], prior[3])
                if prior_holdings:
                    prior_map = self._index_holdings(prior_holdings)

            rows = self._parse_infotable(current_holdings, prior_map)
            if not rows:
                return None

            result = {
                "rows": rows,
                "filing_date": current_date,
                "report_period": current_period,
            }
            with _EDGAR_CACHE_LOCK:
                _EDGAR_CACHE[cache_key] = {"ts": time.time(), "data": result}
            _save_edgar_cache()
            return result

        except Exception as e:
            logger.warning(f"EDGAR fetch failed for CIK {cik}: {e}")
            return None

    @staticmethod
    def _index_holdings(holdings: dict) -> Dict[str, Tuple[float, float]]:
        """Index a prior infotable by CUSIP -> (value, shares)."""
        index: Dict[str, Tuple[float, float]] = {}
        rows = SEC13FEngine._parse_infotable(holdings, prior_map=None, with_actions=False)
        for r in rows:
            if r.get("cusip"):
                index[r["cusip"]] = (r.get("value", 0.0), r.get("shares", 0))
        return index

    @staticmethod
    def _parse_infotable(
        holdings: dict,
        prior_map: Optional[Dict[str, Tuple[float, float]]] = None,
        with_actions: bool = True,
    ) -> List[dict]:
        """Parse an EDGAR 13F infotable into normalized row dicts.

        Handles both the SEC "header"/"data" layout and a plain list-of-dicts
        layout. Returns rows with: cusip, issuer_name, value, shares, put_call,
        tickers, and (when with_actions and prior_map are provided) action and
        shares_changed computed from actual prior-period data.
        """
        rows: List[dict] = []
        try:
            if isinstance(holdings, dict) and isinstance(holdings.get("data"), list):
                header = [str(h).lower() for h in (holdings.get("header") or [])]
                records = holdings["data"]
                if header:
                    records = [dict(zip(header, raw)) for raw in records if isinstance(raw, (list, tuple))]
            elif isinstance(holdings, list):
                records = holdings
            else:
                return rows

            for rec in records:
                if not isinstance(rec, dict):
                    continue
                try:
                    cusip = str(rec.get("cusip", "") or "").strip().upper()
                    name = str(rec.get("nameOfIssuer", "") or "")
                    value_th = float(rec.get("value", 0) or 0)
                    shares = float(rec.get("sshPrnamt", 0) or 0)
                    put_call = str(rec.get("putCall", "") or "").upper()
                    tickers = rec.get("tickers") or None
                    if not name and not cusip:
                        continue
                    rows.append({
                        "cusip": cusip,
                        "issuer_name": name,
                        "value": value_th * 1000.0,
                        "shares": shares,
                        "put_call": put_call,
                        "tickers": tickers,
                    })
                except Exception:
                    continue
        except Exception:
            return rows

        if with_actions and prior_map is not None:
            for r in rows:
                prior = prior_map.get(r.get("cusip", ""))
                prev_value = prior[0] if prior else None
                prev_shares = prior[1] if prior else None
                value = r["value"]
                shares = r["shares"]
                if prev_value is None:
                    r["action"] = "NEW"
                    r["shares_changed"] = int(shares)
                elif prev_value > 0:
                    if value > prev_value * 1.05:
                        r["action"] = "ADD"
                    elif value < prev_value * 0.95:
                        r["action"] = "REDUCE"
                    else:
                        r["action"] = "HOLD"
                    r["shares_changed"] = int(shares - (prev_shares or 0))
                else:
                    r["action"] = "HOLD"
                    r["shares_changed"] = 0
        else:
            for r in rows:
                r["action"] = "HOLD"
                r["shares_changed"] = int(r.get("shares", 0))
        return rows

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize an issuer name for fuzzy ticker resolution."""
        n = (name or "").lower()
        for suffix in (" incorporated", " corporation", " inc", " corp", " ltd", " llc",
                       " l.p.", " lp", " plc", " co", " holdings", " holding", " group",
                       " technologies", " systems", " and co", " & co"):
            if n.endswith(suffix) and len(n) > len(suffix) + 2:
                n = n[:-len(suffix)]
        for ch in " .,&-#'":
            n = n.replace(ch, "")
        return n.strip()

    def _load_company_ticker_map(self) -> Optional[Dict[str, str]]:
        """
        Fetch SEC's company tickers file (name -> ticker) and cache for 24h.
        This is real reference data used to resolve issuer names to symbols.
        """
        now = time.time()
        if self._COMPANY_TICKERS_CACHE is not None and (now - self._COMPANY_TICKERS_TS) < 86400:
            return self._COMPANY_TICKERS_CACHE
        try:
            resp = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=_EDGAR_HEADERS,
                timeout=15,
            )
            if resp.status_code != 200:
                return self._COMPANY_TICKERS_CACHE
            data = resp.json()
            name_map: Dict[str, str] = {}
            for _cik, entry in data.items():
                title = entry.get("title", "")
                ticker = entry.get("ticker", "")
                if title and ticker:
                    name_map[self._normalize_name(title)] = str(ticker).upper()
            self._COMPANY_TICKERS_CACHE = name_map
            self._COMPANY_TICKERS_TS = now
            return name_map
        except Exception as e:
            logger.warning(f"company_tickers fetch failed: {e}")
            return self._COMPANY_TICKERS_CACHE

    def _resolve_symbol(self, issuer_name: str, cusip: str, tickers_field: Any) -> Tuple[str, bool]:
        """
        Resolve a 13F row to a ticker symbol.

        Priority: (1) the SEC-provided tickers field on the infotable row,
        (2) the SEC company-tickers name map, (3) issuer-name abbreviation
        (flagged as unresolved). Values are never fabricated — this only
        affects symbol display.
        """
        if tickers_field:
            if isinstance(tickers_field, list) and tickers_field:
                sym = str(tickers_field[0]).strip().upper()
                if 1 <= len(sym) <= 6:
                    return sym, True
            elif isinstance(tickers_field, str) and tickers_field.strip():
                sym = tickers_field.strip().upper()
                if 1 <= len(sym) <= 6:
                    return sym, True

        name_map = self._load_company_ticker_map()
        if name_map:
            hit = name_map.get(self._normalize_name(issuer_name))
            if hit:
                return hit, True

        abbr = (issuer_name or "UNKNOWN").replace(" ", "")[:10].upper() or "UNKNOWN"
        return abbr, False

    def _build_filing_from_edgar(self, fund_name: str, edgar_data: dict) -> Optional[InstitutionalFiling]:
        """Build an InstitutionalFiling entirely from real EDGAR data."""
        rows = edgar_data.get("rows", [])
        if not rows:
            return None

        sector_map: Dict[str, str] = {}
        try:
            from ticker_universe import get_ticker_universe
            universe = get_ticker_universe()
            for sector, tickers in universe.get_all_sectors().items():
                for t in tickers:
                    sector_map[str(t).upper()] = sector.replace("_", " ").title()
        except Exception:
            pass

        total_value = sum(r["value"] for r in rows) or 1.0

        equity_rows = []
        option_rows = []
        for r in rows:
            sym, resolved = self._resolve_symbol(r.get("issuer_name", ""), r.get("cusip", ""), r.get("tickers"))
            r["symbol"] = sym
            r["is_resolved"] = resolved
            if r.get("put_call") in ("PUT", "CALL"):
                option_rows.append(r)
            else:
                equity_rows.append(r)

        equity_positions = []
        option_positions = []
        for r in equity_rows:
            equity_positions.append(PositionChange(
                symbol=r["symbol"],
                asset_class="Equity",
                action=r.get("action", "HOLD"),
                shares_changed=r.get("shares_changed", 0),
                value_changed=float(r["value"]),
                pct_portfolio=float(r["value"]) / total_value * 100,
                cusip=r.get("cusip", ""),
                issuer_name=r.get("issuer_name", ""),
                is_resolved=r["is_resolved"],
            ))
        for r in option_rows:
            option_positions.append(OptionPosition(
                symbol=r["symbol"],
                contract_type=r["put_call"],
                strike=0.0,  # 13F does not disclose strike price
                expiry="Not disclosed in 13F",
                contracts=int(r.get("shares", 0)),
                notional_value=float(r["value"]),
                action=r.get("action", "HOLD"),
                pct_portfolio=float(r["value"]) / total_value * 100,
            ))

        top_buys = sorted(
            [p for p in equity_positions if p.action in ("NEW", "ADD")],
            key=lambda p: p.value_changed, reverse=True,
        )[:8]
        top_sells = sorted(
            [p for p in equity_positions if p.action in ("REDUCE", "EXIT")],
            key=lambda p: p.value_changed,
        )[:8]
        if not top_buys:
            top_buys = sorted(equity_positions, key=lambda p: p.value_changed, reverse=True)[:5]
        if not top_sells:
            top_sells = []

        # Sector allocation from resolved symbols (real holdings).
        sector_value: Dict[str, float] = {}
        for p in equity_positions:
            sector = sector_map.get(p.symbol.upper(), "Other")
            sector_value[sector] = sector_value.get(sector, 0.0) + p.value_changed
        sector_allocation = {k: v / total_value * 100 for k, v in sector_value.items()} if total_value > 0 else {}

        equity_value = sum(p.value_changed for p in equity_positions)
        options_value = sum(o.notional_value for o in option_positions)
        asset_class_allocation = {}
        if total_value > 0:
            asset_class_allocation["Equities"] = equity_value / total_value * 100
            if options_value > 0:
                asset_class_allocation["Options"] = options_value / total_value * 100

        call_value = sum(o.notional_value for o in option_positions if o.contract_type == "CALL")
        put_value = sum(o.notional_value for o in option_positions if o.contract_type == "PUT")
        if call_value > 0:
            put_call_ratio = put_value / call_value
            options_sentiment = (
                "Hedging" if put_call_ratio > 1.2
                else "Speculative Long" if put_call_ratio < 0.7
                else "Neutral"
            )
        else:
            put_call_ratio = 0.0
            options_sentiment = "No options disclosed"

        return InstitutionalFiling(
            fund_name=fund_name,
            filing_date=edgar_data.get("filing_date", ""),
            report_period=edgar_data.get("report_period", ""),
            total_aum=sum(r["value"] for r in rows),
            top_buys=top_buys,
            top_sells=top_sells,
            sector_allocation=sector_allocation,
            asset_class_allocation=asset_class_allocation,
            option_positions=option_positions,
            bond_positions=[],
            commodity_positions=[],
            put_call_ratio=put_call_ratio,
            options_sentiment=options_sentiment,
            bond_duration=0.0,
            data_source=PROVENANCE_REAL,
            is_simulated=False,
            data_available=True,
        )

    # -----------------------------------------------------------------------
    # MOCK DATA GENERATION
    # -----------------------------------------------------------------------

    def _generate_mock_filing(self, fund_name: str) -> InstitutionalFiling:
        """Generate comprehensive realistic proxy data for 13F filings."""
        now = datetime.now()
        report_date = (now.replace(day=1) - timedelta(days=1)).strftime("%Y-%m-%d")
        filing_date = now.strftime("%Y-%m-%d")

        if "Renaissance" in fund_name:
            return self._mock_renaissance(filing_date, report_date)
        elif "Citadel" in fund_name:
            return self._mock_citadel(filing_date, report_date)
        elif "Bridgewater" in fund_name:
            return self._mock_bridgewater(filing_date, report_date)
        elif "Berkshire" in fund_name:
            return self._mock_berkshire(filing_date, report_date)
        elif "Point72" in fund_name:
            return self._mock_point72(filing_date, report_date)
        elif "Two Sigma" in fund_name:
            return self._mock_two_sigma(filing_date, report_date)
        elif "Millennium" in fund_name:
            return self._mock_millennium(filing_date, report_date)
        elif "D.E. Shaw" in fund_name:
            return self._mock_de_shaw(filing_date, report_date)
        elif "Tiger Global" in fund_name:
            return self._mock_tiger_global(filing_date, report_date)
        elif "Appaloosa" in fund_name:
            return self._mock_appaloosa(filing_date, report_date)
        else:
            return self._mock_generic(fund_name, filing_date, report_date)


    def _mock_renaissance(self, filing_date: str, report_date: str) -> InstitutionalFiling:
        """Renaissance Technologies — quant, all asset classes, heavily diversified."""
        aum = 115_000_000_000
        buys = [
            PositionChange("NVDA", "Equity", "ADD", 1_500_000, 1.8e9, 2.1),
            PositionChange("LLY", "Equity", "NEW", 450_000, 3.5e8, 0.8),
            PositionChange("MSFT", "Equity", "ADD", 800_000, 3.0e8, 0.7),
            PositionChange("PLTR", "Equity", "ADD", 5_000_000, 1.2e8, 0.5),
            PositionChange("GLD", "ETF", "ADD", 2_000_000, 4.5e8, 1.5),
        ]
        sells = [
            PositionChange("TSLA", "Equity", "REDUCE", -2_000_000, -4.5e8, 0.4),
            PositionChange("AAPL", "Equity", "REDUCE", -5_000_000, -8.5e8, 1.2),
            PositionChange("META", "Equity", "EXIT", -1_200_000, -6.0e8, 0.0),
        ]
        options = [
            OptionPosition("SPY", "PUT", 480.0, "2026-03-21", 15_000, 7.2e8, "ADD", 1.8),
            OptionPosition("SPY", "PUT", 460.0, "2026-06-20", 8_000, 3.5e8, "NEW", 0.9),
            OptionPosition("NVDA", "CALL", 150.0, "2026-06-20", 5_000, 4.5e8, "ADD", 1.1),
            OptionPosition("MSFT", "CALL", 450.0, "2026-03-21", 3_000, 1.8e8, "NEW", 0.5),
            OptionPosition("QQQ", "PUT", 460.0, "2026-09-19", 6_000, 2.8e8, "ADD", 0.7),
        ]
        bonds = [
            BondPosition("US Treasury", "Treasury", "2026-03-31", 5.25, 8.0e9, 7.9e9, "ADD", 6.9),
            BondPosition("US Treasury", "Treasury", "2027-06-30", 4.75, 4.0e9, 3.9e9, "HOLD", 3.4),
            BondPosition("Apple Inc", "Corporate IG", "2030-02-08", 3.85, 1.5e9, 1.4e9, "NEW", 1.2),
            BondPosition("Microsoft Corp", "Corporate IG", "2032-08-08", 3.30, 1.0e9, 9.2e8, "HOLD", 0.8),
        ]
        commodities = [
            CommodityPosition("Gold", "ETF", "GLD", 4.5e8, "ADD", 1.5),
            CommodityPosition("Crude Oil", "ETF", "USO", 1.2e8, "NEW", 0.4),
            CommodityPosition("Silver", "ETF", "SLV", 8.0e7, "HOLD", 0.3),
        ]
        return InstitutionalFiling(
            fund_name="Renaissance Technologies LLC",
            filing_date=filing_date,
            report_period=report_date,
            total_aum=aum,
            top_buys=buys,
            top_sells=sells,
            sector_allocation={
                "Technology": 35.0, "Healthcare": 22.0,
                "Consumer Cyclical": 15.0, "Financials": 10.0,
                "Industrials": 8.0, "Other": 10.0,
            },
            asset_class_allocation={
                "Equities": 45.0, "Options": 20.0,
                "Fixed Income": 25.0, "Commodities": 5.0, "Other": 5.0,
            },
            option_positions=options,
            bond_positions=bonds,
            commodity_positions=commodities,
            put_call_ratio=1.4,
            options_sentiment="Hedging",
            bond_duration=2.8,
        )

    def _mock_citadel(self, filing_date: str, report_date: str) -> InstitutionalFiling:
        """Citadel — multi-strategy, heavy options book."""
        aum = 350_000_000_000
        buys = [
            PositionChange("MSFT", "Equity", "ADD", 2_500_000, 1.0e9, 1.5),
            PositionChange("AMZN", "Equity", "ADD", 3_000_000, 5.5e8, 1.2),
            PositionChange("SPY", "ETF", "NEW", 500_000, 2.5e8, 0.5),
            PositionChange("GOOGL", "Equity", "ADD", 1_200_000, 2.0e8, 0.6),
            PositionChange("NVDA", "Equity", "ADD", 900_000, 1.1e9, 1.8),
        ]
        sells = [
            PositionChange("META", "Equity", "REDUCE", -1_500_000, -7.5e8, 0.8),
            PositionChange("JPM", "Equity", "EXIT", -4_000_000, -8.0e8, 0.0),
            PositionChange("TSLA", "Equity", "REDUCE", -2_200_000, -5.0e8, 0.5),
        ]
        options = [
            OptionPosition("SPY", "CALL", 520.0, "2026-03-21", 25_000, 1.3e9, "ADD", 2.1),
            OptionPosition("QQQ", "CALL", 480.0, "2026-06-20", 18_000, 8.6e8, "ADD", 1.4),
            OptionPosition("AAPL", "PUT", 200.0, "2026-03-21", 8_000, 2.4e8, "NEW", 0.4),
            OptionPosition("TSLA", "PUT", 250.0, "2026-06-20", 6_000, 1.8e8, "ADD", 0.3),
            OptionPosition("NVDA", "CALL", 160.0, "2026-09-19", 10_000, 9.5e8, "NEW", 1.5),
            OptionPosition("IWM", "PUT", 200.0, "2026-03-21", 5_000, 1.5e8, "NEW", 0.2),
        ]
        bonds = [
            BondPosition("US Treasury", "Treasury", "2026-12-31", 5.00, 15.0e9, 14.8e9, "ADD", 4.3),
            BondPosition("US Treasury", "Treasury", "2028-06-30", 4.50, 8.0e9, 7.8e9, "HOLD", 2.3),
            BondPosition("JPMorgan Chase", "Corporate IG", "2031-04-23", 3.70, 3.0e9, 2.8e9, "NEW", 0.8),
            BondPosition("Goldman Sachs", "Corporate IG", "2030-10-28", 4.25, 2.0e9, 1.9e9, "HOLD", 0.5),
        ]
        commodities = [
            CommodityPosition("Gold", "ETF", "GLD", 1.5e8, "HOLD", 0.2),
            CommodityPosition("Natural Gas", "Futures", "NG=F", 8.0e7, "NEW", 0.1),
        ]
        return InstitutionalFiling(
            fund_name="Citadel Advisors LLC",
            filing_date=filing_date,
            report_period=report_date,
            total_aum=aum,
            top_buys=buys,
            top_sells=sells,
            sector_allocation={
                "Financial Services": 25.0, "Technology": 25.0,
                "Healthcare": 15.0, "Consumer Cyclical": 12.0,
                "Industrials": 8.0, "Other": 15.0,
            },
            asset_class_allocation={
                "Equities": 35.0, "Options": 35.0,
                "Fixed Income": 25.0, "Commodities": 3.0, "Other": 2.0,
            },
            option_positions=options,
            bond_positions=bonds,
            commodity_positions=commodities,
            put_call_ratio=0.8,
            options_sentiment="Speculative Long",
            bond_duration=3.2,
        )


    def _mock_bridgewater(self, filing_date: str, report_date: str) -> InstitutionalFiling:
        """Bridgewater — macro/risk parity, heavy bonds and commodities."""
        aum = 150_000_000_000
        buys = [
            PositionChange("IVV", "ETF", "ADD", 5_000_000, 2.5e9, 4.5),
            PositionChange("IEMG", "ETF", "ADD", 10_000_000, 5.0e8, 2.0),
            PositionChange("GLD", "ETF", "ADD", 4_000_000, 9.0e8, 3.0),
            PositionChange("TLT", "ETF", "ADD", 8_000_000, 7.2e8, 2.4),
            PositionChange("EMB", "ETF", "NEW", 3_000_000, 2.7e8, 0.9),
        ]
        sells = [
            PositionChange("PG", "Equity", "REDUCE", -3_000_000, -4.5e8, 0.5),
            PositionChange("JNJ", "Equity", "REDUCE", -2_000_000, -3.2e8, 0.4),
            PositionChange("VZ", "Equity", "EXIT", -5_000_000, -2.0e8, 0.0),
        ]
        options = [
            OptionPosition("TLT", "PUT", 85.0, "2026-06-20", 10_000, 2.5e8, "ADD", 0.8),
            OptionPosition("TLT", "PUT", 80.0, "2026-12-19", 6_000, 1.4e8, "NEW", 0.5),
            OptionPosition("SPY", "PUT", 450.0, "2026-12-19", 4_000, 1.8e8, "HOLD", 0.6),
        ]
        bonds = [
            BondPosition("US Treasury", "Treasury", "2026-06-30", 5.25, 20.0e9, 19.8e9, "ADD", 13.3),
            BondPosition("US Treasury", "Treasury", "2030-08-15", 4.00, 15.0e9, 14.5e9, "ADD", 10.0),
            BondPosition("US Treasury", "Treasury", "2053-02-15", 3.625, 8.0e9, 7.2e9, "HOLD", 5.3),
            BondPosition("iShares EM Bond", "Agency", "2035-01-01", 5.50, 4.0e9, 3.9e9, "NEW", 2.7),
            BondPosition("Vanguard Muni", "Municipal", "2032-01-01", 3.20, 2.0e9, 1.9e9, "HOLD", 1.3),
        ]
        commodities = [
            CommodityPosition("Gold", "ETF", "GLD", 9.0e8, "ADD", 3.0),
            CommodityPosition("Crude Oil", "ETF", "USO", 3.5e8, "ADD", 1.2),
            CommodityPosition("Broad Commodities", "ETF", "DJP", 2.5e8, "HOLD", 0.8),
            CommodityPosition("Copper", "Futures", "HG=F", 1.5e8, "NEW", 0.5),
        ]
        return InstitutionalFiling(
            fund_name="Bridgewater Associates, LP",
            filing_date=filing_date,
            report_period=report_date,
            total_aum=aum,
            top_buys=buys,
            top_sells=sells,
            sector_allocation={
                "ETFs / Macro": 65.0, "Consumer Defensive": 15.0,
                "Healthcare": 10.0, "Utilities": 5.0, "Other": 5.0,
            },
            asset_class_allocation={
                "Equities": 20.0, "Fixed Income": 45.0,
                "Commodities": 20.0, "Options": 5.0, "Other": 10.0,
            },
            option_positions=options,
            bond_positions=bonds,
            commodity_positions=commodities,
            put_call_ratio=1.8,
            options_sentiment="Hedging",
            bond_duration=7.4,
        )

    def _mock_berkshire(self, filing_date: str, report_date: str) -> InstitutionalFiling:
        """Berkshire Hathaway — value equity, minimal derivatives."""
        aum = 380_000_000_000
        buys = [
            PositionChange("OXY", "Equity", "ADD", 5_000_000, 2.5e8, 0.7),
            PositionChange("CVX", "Equity", "ADD", 2_000_000, 3.0e8, 0.8),
            PositionChange("KO", "Equity", "HOLD", 0, 0.0, 9.2),
            PositionChange("AXP", "Equity", "HOLD", 0, 0.0, 7.8),
        ]
        sells = [
            PositionChange("AAPL", "Equity", "REDUCE", -100_000_000, -1.8e10, 28.0),
            PositionChange("BAC", "Equity", "REDUCE", -50_000_000, -2.0e9, 3.5),
            PositionChange("HP", "Equity", "EXIT", -10_000_000, -3.5e8, 0.0),
        ]
        options = [
            OptionPosition("OXY", "CALL", 60.0, "2027-01-15", 2_000, 1.2e8, "HOLD", 0.3),
            OptionPosition("BAC", "CALL", 35.0, "2026-01-16", 1_500, 5.0e7, "HOLD", 0.1),
        ]
        bonds = [
            BondPosition("US Treasury", "Treasury", "2025-12-31", 5.40, 120.0e9, 119.5e9, "ADD", 31.5),
            BondPosition("US Treasury", "Treasury", "2026-06-30", 5.25, 40.0e9, 39.8e9, "HOLD", 10.5),
            BondPosition("Berkshire Hathaway", "Corporate IG", "2030-03-15", 3.125, 5.0e9, 4.7e9, "HOLD", 1.3),
        ]
        commodities = []
        return InstitutionalFiling(
            fund_name="Berkshire Hathaway Inc",
            filing_date=filing_date,
            report_period=report_date,
            total_aum=aum,
            top_buys=buys,
            top_sells=sells,
            sector_allocation={
                "Financial Services": 30.0, "Technology": 28.0,
                "Consumer Defensive": 18.0, "Energy": 12.0,
                "Industrials": 7.0, "Other": 5.0,
            },
            asset_class_allocation={
                "Equities": 85.0, "Fixed Income": 12.0,
                "Options": 2.0, "Commodities": 0.0, "Other": 1.0,
            },
            option_positions=options,
            bond_positions=bonds,
            commodity_positions=commodities,
            put_call_ratio=0.3,
            options_sentiment="Neutral",
            bond_duration=1.2,
        )

    def _mock_point72(self, filing_date: str, report_date: str) -> InstitutionalFiling:
        """Point72 — multi-strategy, active options book."""
        aum = 35_000_000_000
        buys = [
            PositionChange("NVDA", "Equity", "ADD", 800_000, 9.6e8, 2.8),
            PositionChange("AMZN", "Equity", "NEW", 1_500_000, 2.7e8, 0.8),
            PositionChange("UBER", "Equity", "ADD", 3_000_000, 2.1e8, 0.6),
            PositionChange("CRWD", "Equity", "NEW", 500_000, 1.8e8, 0.5),
            PositionChange("XLE", "ETF", "ADD", 2_000_000, 1.6e8, 0.5),
        ]
        sells = [
            PositionChange("INTC", "Equity", "EXIT", -8_000_000, -2.0e8, 0.0),
            PositionChange("DIS", "Equity", "REDUCE", -3_000_000, -2.7e8, 0.4),
            PositionChange("PFE", "Equity", "REDUCE", -5_000_000, -1.5e8, 0.3),
        ]
        options = [
            OptionPosition("NVDA", "CALL", 140.0, "2026-06-20", 4_000, 3.6e8, "ADD", 1.0),
            OptionPosition("AMZN", "CALL", 220.0, "2026-03-21", 3_000, 1.8e8, "NEW", 0.5),
            OptionPosition("SPY", "PUT", 490.0, "2026-06-20", 5_000, 2.4e8, "ADD", 0.7),
            OptionPosition("QQQ", "PUT", 460.0, "2026-03-21", 3_500, 1.7e8, "NEW", 0.5),
            OptionPosition("XLE", "CALL", 90.0, "2026-09-19", 2_000, 1.0e8, "NEW", 0.3),
        ]
        bonds = [
            BondPosition("US Treasury", "Treasury", "2027-03-31", 4.875, 3.0e9, 2.95e9, "HOLD", 8.6),
            BondPosition("Ford Motor", "Corporate HY", "2031-01-15", 6.10, 5.0e8, 4.8e8, "NEW", 1.4),
            BondPosition("Verizon Comm", "Corporate IG", "2030-03-22", 4.016, 8.0e8, 7.6e8, "HOLD", 2.3),
            BondPosition("HCA Healthcare", "Corporate HY", "2029-06-15", 5.375, 4.0e8, 3.9e8, "ADD", 1.1),
        ]
        commodities = [
            CommodityPosition("Crude Oil", "ETF", "XLE", 1.6e8, "ADD", 0.5),
            CommodityPosition("Natural Gas", "Futures", "NG=F", 8.0e7, "NEW", 0.2),
        ]
        return InstitutionalFiling(
            fund_name="Point72 Asset Management, L.P.",
            filing_date=filing_date,
            report_period=report_date,
            total_aum=aum,
            top_buys=buys,
            top_sells=sells,
            sector_allocation={
                "Technology": 35.0, "Consumer Cyclical": 18.0,
                "Healthcare": 15.0, "Energy": 12.0,
                "Communication Services": 10.0, "Other": 10.0,
            },
            asset_class_allocation={
                "Equities": 50.0, "Options": 25.0,
                "Fixed Income": 18.0, "Commodities": 5.0, "Other": 2.0,
            },
            option_positions=options,
            bond_positions=bonds,
            commodity_positions=commodities,
            put_call_ratio=1.1,
            options_sentiment="Neutral",
            bond_duration=4.1,
        )


    def _mock_two_sigma(self, filing_date: str, report_date: str) -> InstitutionalFiling:
        """Two Sigma — systematic quant, diversified across asset classes."""
        aum = 60_000_000_000
        buys = [
            PositionChange("AAPL", "Equity", "ADD", 2_000_000, 3.6e8, 1.2),
            PositionChange("GOOGL", "Equity", "ADD", 1_000_000, 1.7e8, 0.6),
            PositionChange("IWM", "ETF", "ADD", 3_000_000, 6.0e8, 2.0),
            PositionChange("EEM", "ETF", "NEW", 5_000_000, 2.5e8, 0.8),
        ]
        sells = [
            PositionChange("NFLX", "Equity", "EXIT", -200_000, -1.2e8, 0.0),
            PositionChange("BABA", "Equity", "REDUCE", -3_000_000, -2.4e8, 0.3),
        ]
        options = [
            OptionPosition("SPY", "CALL", 510.0, "2026-03-21", 8_000, 4.1e8, "ADD", 1.4),
            OptionPosition("IWM", "CALL", 210.0, "2026-06-20", 5_000, 2.1e8, "NEW", 0.7),
            OptionPosition("EEM", "PUT", 40.0, "2026-06-20", 4_000, 1.0e8, "NEW", 0.3),
            OptionPosition("QQQ", "PUT", 450.0, "2026-09-19", 3_000, 1.4e8, "ADD", 0.5),
        ]
        bonds = [
            BondPosition("US Treasury", "Treasury", "2026-09-30", 5.125, 6.0e9, 5.95e9, "HOLD", 10.0),
            BondPosition("US Treasury", "Treasury", "2029-02-15", 4.25, 3.0e9, 2.9e9, "ADD", 5.0),
            BondPosition("Alphabet Inc", "Corporate IG", "2031-08-15", 2.25, 1.0e9, 9.0e8, "HOLD", 1.7),
        ]
        commodities = [
            CommodityPosition("Gold", "ETF", "GLD", 2.0e8, "HOLD", 0.7),
            CommodityPosition("Broad Commodities", "ETF", "PDBC", 1.2e8, "NEW", 0.4),
        ]
        return InstitutionalFiling(
            fund_name="Two Sigma Investments, LP",
            filing_date=filing_date,
            report_period=report_date,
            total_aum=aum,
            top_buys=buys,
            top_sells=sells,
            sector_allocation={
                "Technology": 38.0, "Consumer Cyclical": 15.0,
                "Financials": 14.0, "Healthcare": 12.0,
                "Industrials": 10.0, "Other": 11.0,
            },
            asset_class_allocation={
                "Equities": 48.0, "Options": 18.0,
                "Fixed Income": 27.0, "Commodities": 4.0, "Other": 3.0,
            },
            option_positions=options,
            bond_positions=bonds,
            commodity_positions=commodities,
            put_call_ratio=1.0,
            options_sentiment="Neutral",
            bond_duration=3.5,
        )

    def _mock_millennium(self, filing_date: str, report_date: str) -> InstitutionalFiling:
        """Millennium Management — multi-strategy, broad coverage."""
        aum = 68_000_000_000
        buys = [
            PositionChange("META", "Equity", "ADD", 1_800_000, 9.0e8, 2.2),
            PositionChange("AMZN", "Equity", "ADD", 2_500_000, 4.5e8, 1.1),
            PositionChange("COIN", "Equity", "NEW", 1_000_000, 2.5e8, 0.6),
            PositionChange("ARM", "Equity", "NEW", 800_000, 1.6e8, 0.4),
        ]
        sells = [
            PositionChange("INTC", "Equity", "EXIT", -6_000_000, -1.5e8, 0.0),
            PositionChange("WBA", "Equity", "EXIT", -4_000_000, -5.0e7, 0.0),
            PositionChange("T", "Equity", "REDUCE", -8_000_000, -1.6e8, 0.2),
        ]
        options = [
            OptionPosition("META", "CALL", 600.0, "2026-06-20", 3_000, 1.8e8, "ADD", 0.5),
            OptionPosition("AMZN", "CALL", 230.0, "2026-03-21", 4_000, 2.4e8, "NEW", 0.7),
            OptionPosition("SPY", "PUT", 480.0, "2026-06-20", 6_000, 2.9e8, "ADD", 0.8),
            OptionPosition("COIN", "CALL", 280.0, "2026-09-19", 2_000, 1.4e8, "NEW", 0.4),
        ]
        bonds = [
            BondPosition("US Treasury", "Treasury", "2027-12-31", 4.625, 5.0e9, 4.9e9, "HOLD", 7.4),
            BondPosition("Amazon.com Inc", "Corporate IG", "2032-05-12", 3.875, 1.5e9, 1.4e9, "NEW", 2.2),
            BondPosition("Sprint Capital", "Corporate HY", "2028-03-15", 6.875, 6.0e8, 5.9e8, "HOLD", 0.9),
        ]
        commodities = [
            CommodityPosition("Gold", "ETF", "GLD", 1.5e8, "HOLD", 0.4),
        ]
        return InstitutionalFiling(
            fund_name="Millennium Management LLC",
            filing_date=filing_date,
            report_period=report_date,
            total_aum=aum,
            top_buys=buys,
            top_sells=sells,
            sector_allocation={
                "Technology": 40.0, "Communication Services": 20.0,
                "Consumer Cyclical": 15.0, "Financials": 12.0,
                "Healthcare": 8.0, "Other": 5.0,
            },
            asset_class_allocation={
                "Equities": 52.0, "Options": 20.0,
                "Fixed Income": 22.0, "Commodities": 3.0, "Other": 3.0,
            },
            option_positions=options,
            bond_positions=bonds,
            commodity_positions=commodities,
            put_call_ratio=0.9,
            options_sentiment="Speculative Long",
            bond_duration=3.8,
        )

    def _mock_de_shaw(self, filing_date: str, report_date: str) -> InstitutionalFiling:
        """D.E. Shaw — systematic quant, broad multi-asset."""
        aum = 55_000_000_000
        buys = [
            PositionChange("MSFT", "Equity", "ADD", 1_500_000, 5.6e8, 1.6),
            PositionChange("NVDA", "Equity", "ADD", 600_000, 7.2e8, 2.1),
            PositionChange("SPY", "ETF", "ADD", 1_000_000, 5.0e8, 1.5),
            PositionChange("TLT", "ETF", "ADD", 2_000_000, 1.8e8, 0.5),
        ]
        sells = [
            PositionChange("BIDU", "Equity", "EXIT", -2_000_000, -1.8e8, 0.0),
            PositionChange("SNAP", "Equity", "EXIT", -10_000_000, -1.0e8, 0.0),
        ]
        options = [
            OptionPosition("SPY", "CALL", 515.0, "2026-03-21", 10_000, 5.2e8, "ADD", 1.5),
            OptionPosition("SPY", "PUT", 470.0, "2026-06-20", 7_000, 3.2e8, "ADD", 0.9),
            OptionPosition("NVDA", "CALL", 155.0, "2026-06-20", 3_000, 2.7e8, "NEW", 0.8),
            OptionPosition("TLT", "CALL", 95.0, "2026-09-19", 4_000, 1.2e8, "NEW", 0.3),
        ]
        bonds = [
            BondPosition("US Treasury", "Treasury", "2026-03-31", 5.375, 8.0e9, 7.95e9, "ADD", 14.5),
            BondPosition("US Treasury", "Treasury", "2030-11-15", 4.125, 4.0e9, 3.85e9, "HOLD", 7.3),
            BondPosition("Meta Platforms", "Corporate IG", "2032-05-15", 4.45, 1.0e9, 9.8e8, "NEW", 1.8),
        ]
        commodities = [
            CommodityPosition("Gold", "ETF", "GLD", 2.5e8, "ADD", 0.7),
            CommodityPosition("Crude Oil", "Futures", "CL=F", 1.0e8, "HOLD", 0.3),
        ]
        return InstitutionalFiling(
            fund_name="D.E. Shaw & Co., L.P.",
            filing_date=filing_date,
            report_period=report_date,
            total_aum=aum,
            top_buys=buys,
            top_sells=sells,
            sector_allocation={
                "Technology": 42.0, "Financials": 18.0,
                "Healthcare": 12.0, "Consumer Cyclical": 10.0,
                "Industrials": 8.0, "Other": 10.0,
            },
            asset_class_allocation={
                "Equities": 44.0, "Options": 22.0,
                "Fixed Income": 28.0, "Commodities": 4.0, "Other": 2.0,
            },
            option_positions=options,
            bond_positions=bonds,
            commodity_positions=commodities,
            put_call_ratio=1.05,
            options_sentiment="Neutral",
            bond_duration=4.0,
        )

    def _mock_tiger_global(self, filing_date: str, report_date: str) -> InstitutionalFiling:
        """Tiger Global — growth equity focus, minimal derivatives."""
        aum = 25_000_000_000
        buys = [
            PositionChange("NVDA", "Equity", "ADD", 1_200_000, 1.44e9, 5.8),
            PositionChange("MSFT", "Equity", "ADD", 900_000, 3.4e8, 1.4),
            PositionChange("SNOW", "Equity", "NEW", 2_000_000, 3.0e8, 1.2),
            PositionChange("DDOG", "Equity", "ADD", 1_500_000, 2.1e8, 0.8),
        ]
        sells = [
            PositionChange("SHOP", "Equity", "REDUCE", -3_000_000, -3.3e8, 0.5),
            PositionChange("TWLO", "Equity", "EXIT", -5_000_000, -2.5e8, 0.0),
        ]
        options = [
            OptionPosition("NVDA", "CALL", 145.0, "2026-06-20", 5_000, 4.5e8, "ADD", 1.8),
            OptionPosition("MSFT", "CALL", 460.0, "2026-09-19", 2_000, 1.2e8, "NEW", 0.5),
        ]
        bonds = [
            BondPosition("US Treasury", "Treasury", "2026-06-30", 5.25, 2.0e9, 1.98e9, "HOLD", 8.0),
        ]
        commodities = []
        return InstitutionalFiling(
            fund_name="Tiger Global Management LLC",
            filing_date=filing_date,
            report_period=report_date,
            total_aum=aum,
            top_buys=buys,
            top_sells=sells,
            sector_allocation={
                "Technology": 65.0, "Communication Services": 15.0,
                "Consumer Cyclical": 12.0, "Healthcare": 5.0, "Other": 3.0,
            },
            asset_class_allocation={
                "Equities": 72.0, "Options": 15.0,
                "Fixed Income": 10.0, "Commodities": 0.0, "Other": 3.0,
            },
            option_positions=options,
            bond_positions=bonds,
            commodity_positions=commodities,
            put_call_ratio=0.2,
            options_sentiment="Speculative Long",
            bond_duration=1.5,
        )

    def _mock_appaloosa(self, filing_date: str, report_date: str) -> InstitutionalFiling:
        """Appaloosa Management — value/distressed, macro-aware."""
        aum = 14_000_000_000
        buys = [
            PositionChange("GOOGL", "Equity", "ADD", 800_000, 1.36e8, 1.0),
            PositionChange("META", "Equity", "ADD", 500_000, 2.5e8, 1.8),
            PositionChange("ORCL", "Equity", "NEW", 1_000_000, 1.5e8, 1.1),
            PositionChange("XOM", "Equity", "ADD", 1_500_000, 1.8e8, 1.3),
        ]
        sells = [
            PositionChange("PARA", "Equity", "EXIT", -8_000_000, -1.2e8, 0.0),
            PositionChange("WBD", "Equity", "REDUCE", -5_000_000, -5.0e7, 0.2),
        ]
        options = [
            OptionPosition("SPY", "PUT", 475.0, "2026-06-20", 3_000, 1.4e8, "ADD", 1.0),
            OptionPosition("GOOGL", "CALL", 185.0, "2026-09-19", 2_000, 1.1e8, "NEW", 0.8),
            OptionPosition("XOM", "CALL", 125.0, "2026-06-20", 1_500, 7.5e7, "NEW", 0.5),
        ]
        bonds = [
            BondPosition("US Treasury", "Treasury", "2027-06-30", 4.75, 1.5e9, 1.48e9, "HOLD", 10.7),
            BondPosition("Caesars Entmt", "Corporate HY", "2030-02-15", 8.125, 3.0e8, 3.1e8, "ADD", 2.1),
            BondPosition("Macy's Inc", "Corporate HY", "2028-01-15", 5.875, 2.0e8, 1.95e8, "NEW", 1.4),
        ]
        commodities = [
            CommodityPosition("Crude Oil", "ETF", "XLE", 1.2e8, "ADD", 0.9),
            CommodityPosition("Gold", "ETF", "GLD", 8.0e7, "HOLD", 0.6),
        ]
        return InstitutionalFiling(
            fund_name="Appaloosa Management LP",
            filing_date=filing_date,
            report_period=report_date,
            total_aum=aum,
            top_buys=buys,
            top_sells=sells,
            sector_allocation={
                "Technology": 35.0, "Communication Services": 22.0,
                "Energy": 15.0, "Consumer Cyclical": 12.0,
                "Financials": 8.0, "Other": 8.0,
            },
            asset_class_allocation={
                "Equities": 58.0, "Options": 12.0,
                "Fixed Income": 22.0, "Commodities": 5.0, "Other": 3.0,
            },
            option_positions=options,
            bond_positions=bonds,
            commodity_positions=commodities,
            put_call_ratio=1.3,
            options_sentiment="Hedging",
            bond_duration=4.5,
        )

    def _mock_generic(self, fund_name: str, filing_date: str, report_date: str) -> InstitutionalFiling:
        """Generic fallback mock for any unlisted fund."""
        aum = 20_000_000_000
        buys = [
            PositionChange("GOOGL", "Equity", "ADD", 1_000_000, 1.7e8, 1.0),
            PositionChange("AMD", "Equity", "NEW", 500_000, 8.0e7, 0.5),
        ]
        sells = [
            PositionChange("NFLX", "Equity", "EXIT", -200_000, -1.2e8, 0.0),
        ]
        options = [
            OptionPosition("SPY", "PUT", 480.0, "2026-06-20", 2_000, 9.6e7, "NEW", 0.5),
        ]
        bonds = [
            BondPosition("US Treasury", "Treasury", "2027-03-31", 4.875, 2.0e9, 1.98e9, "HOLD", 10.0),
        ]
        commodities = [
            CommodityPosition("Gold", "ETF", "GLD", 5.0e7, "HOLD", 0.3),
        ]
        return InstitutionalFiling(
            fund_name=fund_name,
            filing_date=filing_date,
            report_period=report_date,
            total_aum=aum,
            top_buys=buys,
            top_sells=sells,
            sector_allocation={
                "Technology": 40.0, "Communication Services": 20.0,
                "Consumer Cyclical": 20.0, "Other": 20.0,
            },
            asset_class_allocation={
                "Equities": 60.0, "Options": 10.0,
                "Fixed Income": 25.0, "Commodities": 3.0, "Other": 2.0,
            },
            option_positions=options,
            bond_positions=bonds,
            commodity_positions=commodities,
            put_call_ratio=1.0,
            options_sentiment="Neutral",
            bond_duration=3.0,
        )


    # -----------------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------------

    def fetch_latest_filings(self, limit: int = 10, allow_simulated: bool = False) -> List[InstitutionalFiling]:
        """
        Fetch the latest real 13F filings from SEC EDGAR for the top funds.

        Real EDGAR data is the default and the ONLY data presented as
        institutional holdings. Simulated filings are produced ONLY when
        ``allow_simulated`` is explicitly True (demo mode) and are always
        labelled as estimates. When real data is unavailable and simulation is
        not requested, an explicitly-marked UNAVAILABLE filing is returned —
        never fabricated data.
        """
        filings = []
        for fund in self.top_funds[:limit]:
            cik = FUND_CIKS.get(fund)
            edgar_data = None
            if cik:
                edgar_data = self._fetch_from_edgar(cik)

            if edgar_data and edgar_data.get("rows"):
                filing = self._build_filing_from_edgar(fund, edgar_data)
                if filing and filing.total_aum > 0:
                    filings.append(filing)
                    continue

            if allow_simulated:
                mock = self._generate_mock_filing(fund)
                mock.data_source = PROVENANCE_SIMULATED
                mock.is_simulated = True
                filings.append(mock)
            else:
                filings.append(self._empty_unavailable_filing(fund))

        return filings

    def _empty_unavailable_filing(self, fund_name: str) -> InstitutionalFiling:
        """Return an explicitly-unavailable filing (never fabricated data)."""
        return InstitutionalFiling(
            fund_name=fund_name,
            filing_date="",
            report_period="",
            total_aum=0.0,
            top_buys=[],
            top_sells=[],
            sector_allocation={},
            asset_class_allocation={},
            option_positions=[],
            bond_positions=[],
            commodity_positions=[],
            put_call_ratio=0.0,
            options_sentiment="N/A",
            bond_duration=0.0,
            data_source=PROVENANCE_UNAVAILABLE,
            is_simulated=False,
            data_available=False,
        )

    def generate_ai_insights(self, filing: InstitutionalFiling) -> str:
        """Generate LLM-powered institutional-grade insight for a filing."""
        if not self.has_llm:
            return "LLM Engine unavailable. Unable to generate contextual insight."

        # Build options summary
        total_call_notional = sum(
            o.notional_value for o in filing.option_positions if o.contract_type == "CALL"
        )
        total_put_notional = sum(
            o.notional_value for o in filing.option_positions if o.contract_type == "PUT"
        )
        options_summary = (
            f"Total call notional: ${total_call_notional/1e9:.2f}B, "
            f"total put notional: ${total_put_notional/1e9:.2f}B, "
            f"put/call ratio: {filing.put_call_ratio:.2f} ({filing.options_sentiment})"
        )

        # Build bond summary
        bond_types = {}
        for b in filing.bond_positions:
            bond_types[b.bond_type] = bond_types.get(b.bond_type, 0) + b.market_value
        bond_summary = ", ".join(
            [f"{k}: ${v/1e9:.1f}B" for k, v in bond_types.items()]
        )

        # Build commodity summary
        commodity_summary = ", ".join(
            [f"{c.commodity} via {c.ticker} (${c.notional_value/1e6:.0f}M)"
             for c in filing.commodity_positions]
        ) or "None"

        provenance = filing.provenance_label()
        sim_warning = ""
        if filing.is_simulated:
            sim_warning = (
                "IMPORTANT: This filing is a SIMULATED ESTIMATE used for demonstration, NOT "
                "actual SEC data. State this clearly at the top of your analysis and do not "
                "present the figures as real holdings.\n\n"
            )
        prompt = f"""
Analyze the following comprehensive SEC 13F filing for {filing.fund_name}.
DATA PROVENANCE: {provenance}
Report Period: {filing.report_period} | Filing Date: {filing.filing_date}
Total AUM: ${filing.total_aum / 1e9:.1f}B

{sim_warning}ASSET CLASS ALLOCATION:
{json.dumps(filing.asset_class_allocation, indent=2)}

TOP EQUITY BUYS:
{', '.join([f"{b.action} {b.symbol} ({b.shares_changed:,} shares, ${b.value_changed/1e6:.0f}M)" for b in filing.top_buys if b.value_changed > 0])}

TOP EQUITY SELLS:
{', '.join([f"{s.action} {s.symbol} ({abs(s.shares_changed):,} shares, ${abs(s.value_changed)/1e6:.0f}M)" for s in filing.top_sells])}

OPTIONS POSITIONING:
{options_summary}
Key option positions: {', '.join([f"{o.contract_type} {o.symbol} ${o.strike} exp {o.expiry} ({o.contracts:,} contracts)" for o in filing.option_positions[:5]])}

FIXED INCOME POSITIONING:
Weighted average duration: {filing.bond_duration:.1f} years
Bond breakdown: {bond_summary}

COMMODITY EXPOSURE:
{commodity_summary}

SECTOR ALLOCATION:
{json.dumps(filing.sector_allocation, indent=2)}

ANALYSIS TASKS:
1. Identify the overarching macro thesis and strategy this fund is expressing.
2. Interpret the put/call ratio of {filing.put_call_ratio:.2f} — is this hedging, speculation, or income generation?
3. Analyze the bond duration of {filing.bond_duration:.1f} years — what does this imply about the fund's rate expectations?
4. Interpret the commodity exposure — what macro thesis does it support?
5. Cross-reference the report period with macro events (Fed policy, inflation, geopolitical tensions, AI narrative).
6. Identify the most likely profitable positions and the highest-risk bets.
7. Provide an institutional-grade synthesis of the overall positioning.

Be highly analytical, signal-dense, and directional. NO EMOJIS.
"""
        try:
            insight = _call_llm(
                prompt=prompt,
                system=(
                    "You are an elite institutional quant researcher and macro strategist. "
                    "Produce signal-dense, directional analysis. STRICT RULE: NO EMOJIS ALLOWED."
                ),
            )
            if not insight:
                return f"The local LLM engine (LM Studio) is not responding. Heuristic Summary:\n- Fund: {filing.fund_name}\n- Put/Call Ratio: {filing.put_call_ratio:.2f} ({filing.options_sentiment})\n- Bond Duration: {filing.bond_duration:.1f} years\n- Action: Monitor positioning for shifts in macro regime."
            return insight
        except Exception as e:
            return f"Error generating insight: {e}"

    # -----------------------------------------------------------------------
    # AGGREGATE ANALYSIS METHODS
    # -----------------------------------------------------------------------

    # Aggregate flow is rebuilt from raw filings on every call, and the build
    # (parse + aggregate 10 funds) costs seconds. The ML ensemble calls it once
    # per predict (and sometimes twice per analysis), so scanners / signal loops
    # would otherwise pay seconds of redundant work per symbol. Filings only
    # refresh daily, so a short TTL cache is safe and makes repeated reads free.
    _FLOW_CACHE: Optional[Dict[str, Any]] = None
    _FLOW_CACHE_TS: float = 0.0
    _FLOW_CACHE_TTL_S: float = 600.0  # 10 min — filings refresh at most daily

    def get_global_smart_money_flow(self) -> Dict[str, Any]:
        """Aggregate net institutional equity flow across all funds."""
        _now = time.time()
        if (SEC13FEngine._FLOW_CACHE is not None
                and (_now - SEC13FEngine._FLOW_CACHE_TS) < SEC13FEngine._FLOW_CACHE_TTL_S):
            return SEC13FEngine._FLOW_CACHE

        filings = [f for f in self.fetch_latest_filings() if f.data_available]
        ticker_net_flow: Dict[str, float] = {}

        for f in filings:
            for b in f.top_buys:
                ticker_net_flow[b.symbol] = ticker_net_flow.get(b.symbol, 0) + b.value_changed
            for s in f.top_sells:
                ticker_net_flow[s.symbol] = ticker_net_flow.get(s.symbol, 0) + s.value_changed

        sorted_flows = sorted(ticker_net_flow.items(), key=lambda x: x[1], reverse=True)
        top_inflows = [x for x in sorted_flows if x[1] > 0][:8]
        top_outflows = sorted([x for x in sorted_flows if x[1] < 0], key=lambda x: x[1])[:8]

        result = {
            "top_inflows": top_inflows,
            "top_outflows": top_outflows,
            "net_flow_map": ticker_net_flow,
        }
        SEC13FEngine._FLOW_CACHE = result
        SEC13FEngine._FLOW_CACHE_TS = _now
        return result

    def get_cross_fund_options_flow(self) -> Dict[str, Any]:
        """Aggregate options positioning across all funds — net calls vs puts by underlying."""
        filings = [f for f in self.fetch_latest_filings() if f.data_available]
        underlying_data: Dict[str, Dict[str, float]] = {}

        for f in filings:
            for opt in f.option_positions:
                sym = opt.symbol
                if sym not in underlying_data:
                    underlying_data[sym] = {"call_notional": 0.0, "put_notional": 0.0, "funds": 0}
                if opt.contract_type == "CALL":
                    underlying_data[sym]["call_notional"] += opt.notional_value
                else:
                    underlying_data[sym]["put_notional"] += opt.notional_value
                underlying_data[sym]["funds"] += 1

        # Compute net P/C ratio per underlying
        result = []
        for sym, data in underlying_data.items():
            call_n = data["call_notional"]
            put_n = data["put_notional"]
            total = call_n + put_n
            pc_ratio = put_n / call_n if call_n > 0 else float("inf")
            sentiment = (
                "Bearish/Hedged" if pc_ratio > 1.5
                else "Bullish" if pc_ratio < 0.5
                else "Neutral"
            )
            result.append({
                "symbol": sym,
                "call_notional": call_n,
                "put_notional": put_n,
                "total_notional": total,
                "pc_ratio": pc_ratio,
                "sentiment": sentiment,
                "fund_count": data["funds"],
            })

        result.sort(key=lambda x: x["total_notional"], reverse=True)

        # Aggregate totals
        total_calls = sum(r["call_notional"] for r in result)
        total_puts = sum(r["put_notional"] for r in result)
        aggregate_pc = total_puts / total_calls if total_calls > 0 else 1.0

        return {
            "by_underlying": result,
            "total_call_notional": total_calls,
            "total_put_notional": total_puts,
            "aggregate_pc_ratio": aggregate_pc,
            "aggregate_sentiment": (
                "Hedging" if aggregate_pc > 1.2
                else "Speculative Long" if aggregate_pc < 0.7
                else "Neutral"
            ),
        }

    def get_bond_market_positioning(self) -> Dict[str, Any]:
        """Aggregate bond holdings — duration, credit quality, Treasury vs corporate."""
        filings = [f for f in self.fetch_latest_filings() if f.data_available]
        bond_type_totals: Dict[str, float] = {}
        total_market_value = 0.0
        weighted_duration_sum = 0.0

        issuer_data: Dict[str, Dict] = {}

        for f in filings:
            for b in f.bond_positions:
                bond_type_totals[b.bond_type] = (
                    bond_type_totals.get(b.bond_type, 0) + b.market_value
                )
                total_market_value += b.market_value
                weighted_duration_sum += b.market_value * f.bond_duration

                key = f"{b.issuer} ({b.bond_type})"
                if key not in issuer_data:
                    issuer_data[key] = {
                        "issuer": b.issuer,
                        "bond_type": b.bond_type,
                        "total_market_value": 0.0,
                        "avg_coupon": [],
                    }
                issuer_data[key]["total_market_value"] += b.market_value
                issuer_data[key]["avg_coupon"].append(b.coupon)

        # Finalize issuer averages
        issuer_list = []
        for key, data in issuer_data.items():
            data["avg_coupon"] = (
                sum(data["avg_coupon"]) / len(data["avg_coupon"])
                if data["avg_coupon"] else 0.0
            )
            issuer_list.append(data)
        issuer_list.sort(key=lambda x: x["total_market_value"], reverse=True)

        weighted_avg_duration = (
            weighted_duration_sum / total_market_value if total_market_value > 0 else 0.0
        )

        # Duration bucket breakdown
        duration_buckets = {"Short (0-3yr)": 0.0, "Medium (3-7yr)": 0.0, "Long (7yr+)": 0.0}
        for f in filings:
            d = f.bond_duration
            total_bonds = sum(b.market_value for b in f.bond_positions)
            if d < 3:
                duration_buckets["Short (0-3yr)"] += total_bonds
            elif d < 7:
                duration_buckets["Medium (3-7yr)"] += total_bonds
            else:
                duration_buckets["Long (7yr+)"] += total_bonds

        return {
            "bond_type_breakdown": bond_type_totals,
            "total_market_value": total_market_value,
            "weighted_avg_duration": weighted_avg_duration,
            "duration_buckets": duration_buckets,
            "top_issuers": issuer_list[:15],
        }

    def get_commodity_exposure(self) -> Dict[str, Any]:
        """Aggregate commodity exposure across all funds."""
        filings = [f for f in self.fetch_latest_filings() if f.data_available]
        commodity_totals: Dict[str, Dict] = {}

        for f in filings:
            for c in f.commodity_positions:
                key = c.commodity
                if key not in commodity_totals:
                    commodity_totals[key] = {
                        "commodity": c.commodity,
                        "total_notional": 0.0,
                        "funds": [],
                        "instruments": set(),
                        "tickers": set(),
                    }
                commodity_totals[key]["total_notional"] += c.notional_value
                commodity_totals[key]["funds"].append(f.fund_name)
                commodity_totals[key]["instruments"].add(c.instrument)
                commodity_totals[key]["tickers"].add(c.ticker)

        result = []
        for key, data in commodity_totals.items():
            result.append({
                "commodity": data["commodity"],
                "total_notional": data["total_notional"],
                "fund_count": len(set(data["funds"])),
                "funds": list(set(data["funds"])),
                "instruments": list(data["instruments"]),
                "tickers": list(data["tickers"]),
            })
        result.sort(key=lambda x: x["total_notional"], reverse=True)

        total_commodity_notional = sum(r["total_notional"] for r in result)

        return {
            "by_commodity": result,
            "total_notional": total_commodity_notional,
            "commodity_allocation": {
                r["commodity"]: r["total_notional"] / total_commodity_notional * 100
                for r in result
            } if total_commodity_notional > 0 else {},
        }

    def get_asset_class_flows(self) -> Dict[str, Any]:
        """Net flows broken down by asset class across all funds."""
        filings = [f for f in self.fetch_latest_filings() if f.data_available]
        asset_class_aum: Dict[str, float] = {}
        asset_class_net_flow: Dict[str, float] = {}

        for f in filings:
            for asset_class, pct in f.asset_class_allocation.items():
                alloc_value = f.total_aum * pct / 100
                asset_class_aum[asset_class] = (
                    asset_class_aum.get(asset_class, 0) + alloc_value
                )

            # Equity net flow
            equity_flow = sum(b.value_changed for b in f.top_buys) + sum(
                s.value_changed for s in f.top_sells
            )
            asset_class_net_flow["Equities"] = (
                asset_class_net_flow.get("Equities", 0) + equity_flow
            )

            # Options net flow (calls positive, puts negative for sentiment)
            opt_flow = sum(
                o.notional_value if o.contract_type == "CALL" else -o.notional_value
                for o in f.option_positions
            )
            asset_class_net_flow["Options"] = (
                asset_class_net_flow.get("Options", 0) + opt_flow
            )

            # Bond flow
            bond_flow = sum(
                b.market_value if b.action in ("ADD", "NEW") else -b.market_value
                for b in f.bond_positions
            )
            asset_class_net_flow["Fixed Income"] = (
                asset_class_net_flow.get("Fixed Income", 0) + bond_flow
            )

            # Commodity flow
            comm_flow = sum(
                c.notional_value if c.action in ("ADD", "NEW") else -c.notional_value
                for c in f.commodity_positions
            )
            asset_class_net_flow["Commodities"] = (
                asset_class_net_flow.get("Commodities", 0) + comm_flow
            )

        return {
            "asset_class_aum": asset_class_aum,
            "asset_class_net_flow": asset_class_net_flow,
        }

    def get_options_sentiment_summary(self) -> Dict[str, Any]:
        """Overall options market sentiment from institutional positioning."""
        filings = [f for f in self.fetch_latest_filings() if f.data_available]
        sentiment_counts: Dict[str, int] = {}
        pc_ratios = []
        total_call_notional = 0.0
        total_put_notional = 0.0

        for f in filings:
            sentiment_counts[f.options_sentiment] = (
                sentiment_counts.get(f.options_sentiment, 0) + 1
            )
            pc_ratios.append(f.put_call_ratio)
            for opt in f.option_positions:
                if opt.contract_type == "CALL":
                    total_call_notional += opt.notional_value
                else:
                    total_put_notional += opt.notional_value

        avg_pc_ratio = sum(pc_ratios) / len(pc_ratios) if pc_ratios else 1.0
        aggregate_sentiment = (
            "Hedging" if avg_pc_ratio > 1.2
            else "Speculative Long" if avg_pc_ratio < 0.7
            else "Neutral"
        )

        return {
            "avg_put_call_ratio": avg_pc_ratio,
            "aggregate_sentiment": aggregate_sentiment,
            "sentiment_distribution": sentiment_counts,
            "total_call_notional": total_call_notional,
            "total_put_notional": total_put_notional,
            "pc_ratio_by_fund": {
                f.fund_name: f.put_call_ratio for f in filings
            },
        }


# ---------------------------------------------------------------------------
# SINGLETON
# ---------------------------------------------------------------------------

_engine_instance: Optional[SEC13FEngine] = None


def get_sec_13f_engine() -> SEC13FEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = SEC13FEngine()
    return _engine_instance
