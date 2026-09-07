"""
Canonical Analytical Context
=============================

Single authoritative SecurityContext / AnalysisContext consumed by EVERY
analytical module. No module independently "figures out" what company/security
it is analyzing.

Also provides:
  - FinancialDataStore   — canonical financial inputs with full provenance
  - ProvenanceEngine     — source/status/confidence tracking for every value

Architecture rule:
  DATA -> ENTITY RESOLUTION -> DATA NORMALIZATION -> DATA PROVENANCE
  -> ANALYTICAL ENGINES -> CROSS-MODULE CONSISTENCY -> NUMERICAL QC
  -> SEMANTIC QC -> ECONOMIC PLAUSIBILITY QC -> CONFIDENCE/DATA-QUALITY GATING
  -> INVESTMENT CONCLUSION -> VISUALIZATION

Author: Octavian Terminal
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from enum import Enum
import hashlib


# ────────────────────────────────────────────────────────────────────────────
# PROVENANCE — every number carries its origin
# ────────────────────────────────────────────────────────────────────────────

class ProvenanceStatus(str, Enum):
    """Classification for every data point."""
    OBSERVED = "OBSERVED"               # Directly observed from a source
    REPORTED = "REPORTED"               # Sourced from financial statements
    DERIVED = "DERIVED"                 # Calculated from sourced data
    ESTIMATED = "ESTIMATED"             # Model estimate
    ASSUMED = "ASSUMED"                 # User/model assumption
    MARKET_IMPLIED = "MARKET_IMPLIED"   # Implied by market prices
    CONSENSUS = "CONSENSUS"             # Sell-side / analyst consensus
    INFERRED = "INFERRED"               # AI/statistical inference
    STALE = "STALE"                     # Previously observed, now outdated
    UNAVAILABLE = "UNAVAILABLE"         # Data not available


class MissingReason(str, Enum):
    """Why data is absent — NEVER silently substitute."""
    NOT_AVAILABLE = "NOT AVAILABLE"
    NOT_FETCHED = "NOT FETCHED"
    STALE = "STALE"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT APPLICABLE"
    INSUFFICIENT_HISTORY = "INSUFFICIENT HISTORY"
    SOURCE_ERROR = "SOURCE ERROR"
    CONFLICTING_SOURCES = "CONFLICTING SOURCES"


@dataclass
class ProvenancePoint:
    """Every important number carries this metadata."""
    label: str
    value: Optional[float] = None
    status: ProvenanceStatus = ProvenanceStatus.UNAVAILABLE
    source: str = ""
    source_timestamp: str = ""          # ISO format when the source was observed
    unit: str = ""                      # e.g. "USD", "shares", "percentage"
    currency: str = "USD"
    period: str = ""                    # e.g. "FY2025A", "NTM", "TTM"
    methodology: str = ""               # How the value was derived
    confidence: float = 0.0             # 0..1
    transformations: List[str] = field(default_factory=list)
    missing_reason: Optional[MissingReason] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        if self.missing_reason:
            d["missing_reason"] = self.missing_reason.value
        return d

    @property
    def is_available(self) -> bool:
        return (self.status not in (ProvenanceStatus.UNAVAILABLE, ProvenanceStatus.STALE)
                and self.value is not None)

    @property
    def is_verified(self) -> bool:
        return self.status in (ProvenanceStatus.OBSERVED, ProvenanceStatus.REPORTED)

    @property
    def requires_disclosure(self) -> bool:
        """True when the value's limitations must be disclosed."""
        return self.status in (ProvenanceStatus.ESTIMATED, ProvenanceStatus.ASSUMED,
                               ProvenanceStatus.INFERRED, ProvenanceStatus.MARKET_IMPLIED)


# ────────────────────────────────────────────────────────────────────────────
# FINANCIAL DATA STORE — single source of truth for all modules
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class FinancialDataStore:
    """Canonical financial values. Every module pulls from here — no module
    independently generates its own revenue/EBIT/FCF numbers."""

    # Income statement
    revenue: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("Revenue"))
    revenue_growth: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("Revenue Growth"))
    ebit: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("EBIT"))
    ebit_margin: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("EBIT Margin"))
    net_income: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("Net Income"))
    eps: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("EPS"))
    da_expense: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("D&A"))

    # Balance sheet
    total_debt: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("Total Debt"))
    cash: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("Cash"))
    net_debt: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("Net Debt"))
    total_assets: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("Total Assets"))

    # Cash flow
    operating_cf: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("Operating CF"))
    capex: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("CapEx"))
    fcf: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("Free Cash Flow"))
    fcf_margin: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("FCF Margin"))

    # Per-share / market
    shares_outstanding: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("Shares Outstanding"))
    current_price: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("Current Price"))
    market_cap: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("Market Cap"))

    # Valuation inputs
    wacc: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("WACC"))
    terminal_growth: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("Terminal Growth"))
    tax_rate: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("Tax Rate"))
    risk_free_rate: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("Risk-Free Rate"))
    equity_risk_premium: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("ERP"))
    beta: ProvenancePoint = field(default_factory=lambda: ProvenancePoint("Beta"))

    # Fiscal metadata
    fiscal_year_end: str = ""
    reporting_currency: str = "USD"
    reporting_period: str = ""          # e.g. "FY2025A", "TTM", "Latest Q"

    def availability_map(self) -> Dict[str, bool]:
        """Map of which fields have actual data."""
        result = {}
        for field_name in self.__dataclass_fields__:
            val = getattr(self, field_name)
            if isinstance(val, ProvenancePoint):
                result[field_name] = val.is_available
        return result

    def completeness_score(self) -> float:
        """Fraction of core financial fields with data."""
        core_fields = ["revenue", "ebit_margin", "fcf", "shares_outstanding",
                       "current_price", "wacc"]
        avail = self.availability_map()
        present = sum(1 for f in core_fields if avail.get(f, False))
        return present / len(core_fields) if core_fields else 0.0

    def get_available_fields(self) -> List[str]:
        return [k for k, v in self.availability_map().items() if v]

    def get_missing_fields(self) -> List[str]:
        return [k for k, v in self.availability_map().items() if not v]


# ────────────────────────────────────────────────────────────────────────────
# SECURITY CONTEXT — the one canonical object consumed by every module
# ────────────────────────────────────────────────────────────────────────────

class IntegrityStatus(str, Enum):
    """Overall integrity status of the analysis context."""
    VERIFIED = "VERIFIED"
    CONDITIONAL = "CONDITIONAL"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


class SecurityType(str, Enum):
    EQUITY = "Equity"
    ETF = "ETF"
    CRYPTO = "Crypto"
    FOREX = "Forex"
    FUTURES = "Futures"
    INDEX = "Index"
    OTHER = "Other"


@dataclass
class SecurityContext:
    """Canonical context object consumed by EVERY analytical module.

    No downstream module may independently substitute another security,
    company, industry, product, competitor, or financial dataset.
    """

    # Identity
    ticker: str = ""
    company_legal_name: str = ""
    company_display_name: str = ""
    exchange: str = ""
    security_type: SecurityType = SecurityType.EQUITY
    sector: str = ""
    industry: str = ""
    sub_industry: str = ""
    country: str = ""

    # Business
    primary_segments: List[str] = field(default_factory=list)
    major_products: List[str] = field(default_factory=list)
    relevant_competitors: List[str] = field(default_factory=list)
    relevant_kpis: List[str] = field(default_factory=list)
    relevant_macro_sensitivities: List[str] = field(default_factory=list)
    relevant_valuation_methodologies: List[str] = field(default_factory=list)

    # Financial data — the single source of truth
    financials: FinancialDataStore = field(default_factory=FinancialDataStore)

    # Metadata
    valuation_date: str = ""            # ISO format
    data_timestamp: str = ""            # When this context was assembled
    source_registry: Dict[str, str] = field(default_factory=dict)
    confidence_metadata: Dict[str, Any] = field(default_factory=dict)

    # Integrity
    integrity_status: IntegrityStatus = IntegrityStatus.CONDITIONAL
    integrity_violations: List[Dict[str, str]] = field(default_factory=list)
    entity_integrity_score: float = 0.0  # 0-100

    # Context ID for traceability
    context_id: str = field(default_factory=lambda: hashlib.sha256(
        str(datetime.now(timezone.utc).timestamp()).encode()).hexdigest()[:12])

    def to_summary(self) -> Dict[str, Any]:
        """Compact summary for the audit panel."""
        return {
            "ticker": self.ticker,
            "name": self.company_display_name or self.ticker,
            "sector": self.sector,
            "security_type": self.security_type.value,
            "country": self.country,
            "integrity": self.integrity_status.value,
            "completeness": f"{self.financials.completeness_score():.0%}",
            "entity_score": f"{self.entity_integrity_score:.0f}/100",
            "context_id": self.context_id,
        }

    @staticmethod
    def from_quick_quote(ticker: str, price: Optional[float] = None,
                         company_name: str = "", sector: str = "",
                         price_timestamp: str = "",
                         price_source: str = "live quote") -> "SecurityContext":
        """Build a minimal context from just a ticker and quote (the common case).
        The resulting context is CONDITIONAL until enriched with fundamentals."""
        ctx = SecurityContext(
            ticker=ticker,
            company_display_name=company_name or ticker,
            sector=sector,
            valuation_date=price_timestamp or datetime.now(timezone.utc).isoformat(),
            data_timestamp=datetime.now(timezone.utc).isoformat(),
            integrity_status=IntegrityStatus.CONDITIONAL,
        )
        if price is not None and price > 0:
            ctx.financials.current_price = ProvenancePoint(
                label="Current Price",
                value=float(price),
                status=ProvenanceStatus.OBSERVED,
                source=price_source,
                source_timestamp=price_timestamp or "",
                unit="USD/share",
                currency="USD",
                confidence=1.0,
            )
        else:
            ctx.financials.current_price = ProvenancePoint(
                label="Current Price",
                status=ProvenanceStatus.UNAVAILABLE,
                missing_reason=MissingReason.NOT_AVAILABLE,
            )
        return ctx

    @staticmethod
    def from_fundamentals(ticker: str, fund: Dict[str, Any],
                          price: float = 0.0, price_ts: str = "",
                          price_src: str = "live quote") -> "SecurityContext":
        """Build an enriched context from fundamentals dict (from _dd_fetch_fundamentals)."""
        ctx = SecurityContext.from_quick_quote(
            ticker, price,
            company_name=fund.get("company_name", ""),
            sector=fund.get("sector", ""),
            price_timestamp=price_ts,
            price_source=price_src,
        )
        has_fund = bool(fund.get("revenue_m") and fund.get("shares_m"))
        if has_fund:
            status = ProvenanceStatus.REPORTED
            src = "SEC filings / financial data feed"
            ts = fund.get("filing_date", "")

            ctx.financials.revenue = ProvenancePoint(
                "Revenue", float(fund["revenue_m"]), status, src, ts,
                unit="USD M", currency="USD",
                period=fund.get("fiscal_period", "TTM"), confidence=0.9)
            ctx.financials.ebit_margin = ProvenancePoint(
                "EBIT Margin", float(fund.get("ebit_margin_pct", 20.0)) / 100.0,
                status, src, ts, unit="ratio", confidence=0.9)
            ctx.financials.shares_outstanding = ProvenancePoint(
                "Shares Outstanding", float(fund["shares_m"]),
                status, src, ts, unit="M shares", confidence=0.9)
            ctx.financials.tax_rate = ProvenancePoint(
                "Tax Rate", float(fund.get("tax_rate_pct", 21.0)) / 100.0,
                status, src, ts, unit="ratio", confidence=0.85)
            ctx.financials.total_debt = ProvenancePoint(
                "Total Debt", float(fund.get("debt_m", 0)),
                status, src, ts, unit="USD M", confidence=0.9)
            ctx.financials.cash = ProvenancePoint(
                "Cash", float(fund.get("cash_m", 0)),
                status, src, ts, unit="USD M", confidence=0.9)
            ctx.financials.net_debt = ProvenancePoint(
                "Net Debt",
                float(fund.get("debt_m", 0)) - float(fund.get("cash_m", 0)),
                ProvenanceStatus.DERIVED, src, ts, unit="USD M", confidence=0.85,
                methodology="total_debt - cash")
            ctx.financials.eps = ProvenancePoint(
                "EPS", float(fund.get("eps", 0)),
                status, src, ts, unit="USD/share", confidence=0.9)
            ctx.financials.reporting_period = fund.get("fiscal_period", "TTM")
            ctx.financials.fiscal_year_end = fund.get("fiscal_year_end", "")
            ctx.integrity_status = IntegrityStatus.CONDITIONAL  # Still conditional until cash flow verified

        # Cash flow is almost always DATA UNAVAILABLE from the current feed
        ctx.financials.operating_cf = ProvenancePoint(
            "Operating CF", status=ProvenanceStatus.UNAVAILABLE,
            missing_reason=MissingReason.NOT_AVAILABLE,
            notes="Cash flow statement not in current fundamentals feed")
        ctx.financials.capex = ProvenancePoint(
            "CapEx", status=ProvenanceStatus.UNAVAILABLE,
            missing_reason=MissingReason.NOT_AVAILABLE)
        ctx.financials.fcf = ProvenancePoint(
            "FCF", status=ProvenanceStatus.UNAVAILABLE,
            missing_reason=MissingReason.NOT_AVAILABLE,
            notes="FCF requires cash flow statement (DATA UNAVAILABLE)")

        return ctx

    def is_critical_field_missing(self) -> Tuple[bool, List[str]]:
        """Check if any critical fields are missing. Returns (any_missing, list_of_fields)."""
        critical = ["current_price", "revenue", "shares_outstanding"]
        missing = [f for f in critical
                   if not getattr(self.financials, f).is_available]
        return bool(missing), missing

    def data_freshness_score(self) -> float:
        """0-100: how fresh is the data."""
        if not self.financials.current_price.is_available:
            return 0.0
        # Simple heuristic: price data within 24h = fresh
        score = 50.0  # Base: we have a price
        if self.financials.revenue.is_available:
            score += 30.0
        if self.financials.fcf.is_available:
            score += 20.0
        return min(100.0, score)


# ────────────────────────────────────────────────────────────────────────────
# PROVENANCE ENGINE — centralized provenance tracking
# ────────────────────────────────────────────────────────────────────────────

class ProvenanceEngine:
    """Central registry for provenance tracking across all analytical outputs."""

    def __init__(self):
        self._registry: Dict[str, ProvenancePoint] = {}
        self._source_conflicts: List[Dict[str, Any]] = []

    def register(self, key: str, pp: ProvenancePoint) -> None:
        """Register a provenance point. If the same key already exists with a
        conflicting value, record a source conflict."""
        if key in self._registry:
            existing = self._registry[key]
            if (existing.is_available and pp.is_available
                    and existing.value is not None and pp.value is not None
                    and abs(existing.value - pp.value) / max(abs(existing.value), 1e-9) > 0.01):
                self._source_conflicts.append({
                    "key": key,
                    "source_a": existing.source,
                    "value_a": existing.value,
                    "source_b": pp.source,
                    "value_b": pp.value,
                    "preferred": pp.source,  # Latest registration wins
                    "impact": "MATERIAL" if abs(existing.value - pp.value) / max(abs(existing.value), 1e-9) > 0.05 else "MINOR",
                })
        self._registry[key] = pp

    def get(self, key: str) -> Optional[ProvenancePoint]:
        return self._registry.get(key)

    def get_value(self, key: str) -> Optional[float]:
        pp = self._registry.get(key)
        return pp.value if pp and pp.is_available else None

    def get_conflicts(self) -> List[Dict[str, Any]]:
        return list(self._source_conflicts)

    def has_material_conflicts(self) -> bool:
        return any(c.get("impact") == "MATERIAL" for c in self._source_conflicts)

    def provenance_report(self) -> str:
        """Generate a provenance summary for the audit panel."""
        total = len(self._registry)
        available = sum(1 for pp in self._registry.values() if pp.is_available)
        verified = sum(1 for pp in self._registry.values() if pp.is_verified)
        lines = [
            f"**PROVENANCE:** {available}/{total} values available, "
            f"{verified}/{total} verified or reported.",
        ]
        if self._source_conflicts:
            lines.append(f"**SOURCE CONFLICTS:** {len(self._source_conflicts)} detected.")
            for c in self._source_conflicts:
                lines.append(f"  - {c['key']}: {c['source_a']}={c['value_a']} vs "
                             f"{c['source_b']}={c['value_b']} [{c['impact']}]")
        return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# FACTORY — build SecurityContext from what's available at query time
# ────────────────────────────────────────────────────────────────────────────

def build_security_context(ticker: str, live_data: Optional[Dict] = None,
                           fundamentals: Optional[Dict] = None,
                           query: str = "") -> SecurityContext:
    """Build the canonical SecurityContext from whatever data is available.

    This is the SINGLE entry point for creating the context. Every analytical
    module calls this (or receives the context from its caller).
    """
    td = (live_data or {}).get(ticker, {})
    price = float(td.get("price", 0.0) or 0.0)
    qdate = str(td.get("quote_date") or "") or None
    qsrc = str(td.get("source") or "live quote")

    if fundamentals and fundamentals.get("revenue_m") and fundamentals.get("shares_m"):
        ctx = SecurityContext.from_fundamentals(
            ticker, fundamentals, price,
            price_ts=qdate or "",
            price_src=qsrc,
        )
    else:
        ctx = SecurityContext.from_quick_quote(
            ticker, price,
            price_timestamp=qdate or "",
            price_source=qsrc,
        )

    # Infer security type from ticker
    if "/" in ticker or "=X" in ticker:
        ctx.security_type = SecurityType.FOREX
    elif "=F" in ticker.upper():
        ctx.security_type = SecurityType.FUTURES
    elif ticker.upper().endswith("-USD"):
        ctx.security_type = SecurityType.CRYPTO
    elif ticker.startswith("^"):
        ctx.security_type = SecurityType.INDEX

    # Populate DCF valuation inputs when fundamentals are present
    if ctx.financials.revenue.is_available:
        # WACC: MODEL ASSUMPTION (default ~9.5%)
        ctx.financials.wacc = ProvenancePoint(
            "WACC", 0.095, ProvenanceStatus.ASSUMED,
            "model default", "", unit="ratio",
            methodology="CAPM with model default inputs",
            notes="ASSUMPTION — not computed from live CAPM inputs")
        ctx.financials.terminal_growth = ProvenancePoint(
            "Terminal Growth", 0.028, ProvenanceStatus.ASSUMED,
            "model default", "", unit="ratio",
            methodology="long-run nominal GDP proxy")
        ctx.financials.risk_free_rate = ProvenancePoint(
            "Risk-Free Rate", 0.045, ProvenanceStatus.ASSUMED,
            "model default", "", unit="ratio",
            notes="ASSUMPTION — not pulled from live yield curve")
        ctx.financials.equity_risk_premium = ProvenancePoint(
            "ERP", 0.055, ProvenanceStatus.ASSUMED,
            "model default", "", unit="ratio")
        ctx.financials.beta = ProvenancePoint(
            "Beta", 1.2, ProvenanceStatus.ASSUMED,
            "model default", "", unit="ratio",
            notes="ASSUMPTION — not computed from live regression")

    return ctx