"""
Analytical Integrity Engine
============================

Generalized integrity checks that work for EVERY security, sector, and industry.

Modules:
  - EntityIntegrityEngine   — verifies all claims refer to the right entity
  - SemanticContamination   — detects wrong-company/industry narrative bleed
  - EconomicPlausibility    — mathematically valid != economically sensible
  - HardConfidenceEngine    — weighted average + HARD constraints

All engines consume the canonical SecurityContext and return structured results.

Author: Octavian Terminal
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
import re

from analytical_context import (
    SecurityContext, FinancialDataStore, ProvenancePoint,
    ProvenanceStatus, IntegrityStatus, SecurityType, ProvenanceEngine,
)


# ────────────────────────────────────────────────────────────────────────────
# SEVERITY LEVELS
# ────────────────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    """Impact classification for integrity violations."""
    INFO = "INFO"           # Informational, no impact on output
    WARNING = "WARNING"     # Degraded confidence, but analysis proceeds
    CRITICAL = "CRITICAL"   # Must affect downstream behavior (rating, confidence)


@dataclass
class IntegrityViolation:
    """A single integrity issue found during validation."""
    check: str              # What check was performed
    severity: Severity
    description: str
    evidence: str = ""      # What was found that triggered the violation
    field: str = ""         # Which data/field is affected
    recommended_action: str = ""


@dataclass
class IntegrityReport:
    """Complete integrity assessment for an analysis."""
    violations: List[IntegrityViolation] = field(default_factory=list)
    entity_integrity_score: float = 100.0  # 0-100
    semantic_contamination_score: float = 100.0  # 0-100
    economic_plausibility_score: float = 100.0  # 0-100
    overall_integrity: IntegrityStatus = IntegrityStatus.VERIFIED
    blocks_rating: bool = False
    summary: str = ""


# ────────────────────────────────────────────────────────────────────────────
# ENTITY INTEGRITY ENGINE
# ────────────────────────────────────────────────────────────────────────────

# Broad set of English/financial stopwords that are NEVER valid tickers.
# This is the first defense against treating prose words as securities.
_ENTITY_STOPWORDS: Set[str] = {
    "THE", "AND", "OR", "BUT", "IF", "THEN", "ELSE", "WHEN", "WHAT", "WHY",
    "HOW", "IS", "ARE", "AM", "WAS", "WERE", "BE", "BEEN", "BEING",
    "DO", "DID", "DOES", "I", "YOU", "HE", "SHE", "IT", "WE", "THEY",
    "ME", "HIM", "HER", "US", "THEM", "THIS", "THAT", "THESE", "THOSE",
    "IN", "ON", "AT", "BY", "FOR", "WITH", "ABOUT", "AGAINST", "BETWEEN",
    "INTO", "THROUGH", "DURING", "BEFORE", "AFTER", "ABOVE", "BELOW",
    "FROM", "UP", "DOWN", "OUT", "OVER", "UNDER", "AGAIN", "FURTHER",
    "ONCE", "TO", "A", "AN", "AS", "SECTOR", "SECTORS", "SHORT", "LONG",
    "LOOK", "RIGHT", "NOW", "BEAR", "BEARISH", "BULL", "BULLISH",
    "TICKER", "TICKERS", "MARKET", "MARKETS", "PRICE", "PRICES",
    "TREND", "TRENDS", "POSITION", "STRATEGY", "STRATEGIES",
}

# Financial-metric tokens that must NEVER be treated as security names.
# These are the "DCF for FCF of EBITDA" type problems.
_FINANCIAL_METRIC_TOKENS: Set[str] = {
    "DCF", "FCF", "EBITDA", "EBIT", "EPS", "WACC", "CAPM", "CAPEX",
    "NOPAT", "NWC", "NPV", "IRR", "ROE", "ROA", "ROIC", "ROCE",
    "GPU", "CPU", "ASIC", "API", "SaaS", "AI", "ML", "LLM", "NLP",
    "CAGR", "YOY", "QOQ", "TTM", "NTM", "FY", "H1", "H2", "Q1", "Q2",
    "Q3", "Q4", "P/E", "P/B", "P/S", "EV", "COGS", "SG&A", "R&D",
    "GM", "PM", "OPM", "NPM", "CFO", "CFI", "CFF", "PPE", "AR", "AP",
    "IPO", "SPAC", "M&A", "LBO", "ETF", "REIT", "MLP", "BDC",
}


class EntityIntegrityEngine:
    """Verifies that all claims in an analysis refer to the correct entity.

    Works for EVERY security — no hard-coded company-specific logic.
    """

    @staticmethod
    def validate_ticker(ctx: SecurityContext) -> List[IntegrityViolation]:
        """Check that the ticker is valid and not a stopword/metric token."""
        violations = []
        upper = ctx.ticker.upper().strip()

        if not upper:
            violations.append(IntegrityViolation(
                check="ticker-validity",
                severity=Severity.CRITICAL,
                description="No ticker provided — cannot verify entity.",
            ))
            return violations

        if upper in _ENTITY_STOPWORDS:
            violations.append(IntegrityViolation(
                check="ticker-stopword",
                severity=Severity.CRITICAL,
                description=f"Ticker '{ctx.ticker}' is an English stopword — not a valid security.",
                evidence=f"'{ctx.ticker}' matched entity stopword list.",
                recommended_action="Extract the actual security ticker from the query.",
            ))

        if upper in _FINANCIAL_METRIC_TOKENS:
            violations.append(IntegrityViolation(
                check="ticker-metric-token",
                severity=Severity.CRITICAL,
                description=f"Ticker '{ctx.ticker}' is a financial metric token — cannot be a security.",
                evidence=f"'{ctx.ticker}' matched financial-metric token list.",
                recommended_action="Extract the actual security ticker from the query.",
            ))

        # Pattern check: tickers should be 1-6 uppercase letters with optional digits
        if not re.match(r'^[A-Z]{1,6}$', upper) and "/" not in ctx.ticker and "=" not in ctx.ticker and "-" not in ctx.ticker:
            violations.append(IntegrityViolation(
                check="ticker-pattern",
                severity=Severity.WARNING,
                description=f"Ticker '{ctx.ticker}' has unusual format for a US equity.",
                evidence=f"Pattern mismatch: expected 1-6 uppercase letters.",
            ))

        return violations

    @staticmethod
    def compute_entity_score(ctx: SecurityContext) -> Tuple[float, List[IntegrityViolation]]:
        """Compute entity integrity score (0-100) from available identity data."""
        violations = EntityIntegrityEngine.validate_ticker(ctx)
        score = 100.0

        # Critical: ticker validity
        for v in violations:
            if v.severity == Severity.CRITICAL:
                score = min(score, 0.0)

        # Degraded: missing identity fields
        if not ctx.company_display_name:
            violations.append(IntegrityViolation(
                check="entity-name",
                severity=Severity.WARNING,
                description="Company display name is empty.",
                recommended_action="Verify the company name from ticker lookup.",
            ))
            score = min(score, 60.0)

        if not ctx.sector:
            violations.append(IntegrityViolation(
                check="entity-sector",
                severity=Severity.WARNING,
                description="Sector classification is missing.",
                recommended_action="Classify the company's sector from its business description.",
            ))
            score = min(score, 70.0)

        return max(0.0, score), violations


# ────────────────────────────────────────────────────────────────────────────
# SEMANTIC CONTAMINATION DETECTION
# ────────────────────────────────────────────────────────────────────────────

class SemanticContaminationDetector:
    """Detects when narrative content belongs to a different company/industry.

    Dynamically constructs a relevance vocabulary from the SecurityContext
    and checks incoming content against it.
    """

    # Industry-contrary term clusters — if the subject is NOT in this industry,
    # these terms in the narrative are highly suspicious.
    _INDUSTRY_CONTRARY_CLUSTERS: Dict[str, List[str]] = {
        "Semiconductors": [
            "GPU architecture", "CUDA", "fab", "wafer", "node process",
            "foundry", "lithography", "chiplet", "HBM", "memory bandwidth",
            "tensor core", "ASIC accelerator", "networking fabric",
        ],
        "Pharmaceuticals": [
            "clinical trial", "FDA approval", "pipeline", "patent cliff",
            "biologic", "small molecule", "phase III", "drug candidate",
            "therapeutic area", "orphan drug",
        ],
        "Banking": [
            "net interest margin", "loan book", "deposit beta", "CET1",
            "risk-weighted assets", "loan-loss provision", "NIM",
            "cost-income ratio", "trading book", "Basel III",
        ],
        "Oil & Gas": [
            "upstream", "downstream", "barrel", "rig count", "E&P",
            "refining margin", "crack spread", "reserves", "production volume",
        ],
        "Software": [
            "SaaS", "ARR", "MRR", "churn", "CAC", "LTV", "cloud revenue",
            "subscription", "seat-based", "multi-tenant",
        ],
        "Retail": [
            "same-store sales", "foot traffic", "inventory turnover",
            "SKU", "omnichannel", "comp sales", "square footage",
        ],
    }

    # Generic suspicious patterns — any narrative that introduces a clearly
    # different company name, ticker, or product should be flagged.
    _GENERIC_SUSPICIOUS_PATTERNS = [
        # Another company's products mentioned as if belonging to subject
        (re.compile(r'\b(iPhone|iPad|MacBook|AirPods|Apple Watch)\b', re.I), "smartphone/consumer electronics"),
        (re.compile(r'\b(Windows|Azure|Office 365|Xbox|LinkedIn|Bing)\b', re.I), "Microsoft ecosystem"),
        (re.compile(r'\b(AWS|Amazon Prime|Alexa|Kindle|Fire TV)\b', re.I), "Amazon ecosystem"),
        (re.compile(r'\b(Search|Android|YouTube|Gmail|Chrome|Maps)\b', re.I), "Google/Alphabet ecosystem"),
    ]

    @staticmethod
    def build_relevance_vocabulary(ctx: SecurityContext) -> Set[str]:
        """Build the set of terms that are RELEVANT to this entity."""
        vocab = set()
        vocab.add(ctx.ticker.upper())
        if ctx.company_display_name:
            for part in ctx.company_display_name.split():
                if len(part) > 2:
                    vocab.add(part.lower())
        for seg in ctx.primary_segments:
            for w in seg.split():
                if len(w) > 2:
                    vocab.add(w.lower())
        for prod in ctx.major_products:
            for w in prod.split():
                if len(w) > 2:
                    vocab.add(w.lower())
        for comp in ctx.relevant_competitors:
            vocab.add(comp.upper())
        return vocab

    @staticmethod
    def classify_industry(ctx: SecurityContext) -> Optional[str]:
        """Map the context to one of the known industry clusters."""
        sector = ctx.sector.lower() if ctx.sector else ""
        industry = ctx.industry.lower() if ctx.industry else ""

        if any(w in sector for w in ["technology", "tech", "semiconductor"]):
            return "Semiconductors"
        if any(w in (sector + industry) for w in ["pharma", "biotech", "healthcare"]):
            return "Pharmaceuticals"
        if any(w in sector for w in ["financial", "bank", "insurance"]):
            return "Banking"
        if any(w in sector for w in ["energy", "oil", "gas"]):
            return "Oil & Gas"
        if any(w in (sector + industry) for w in ["software", "cloud", "saas"]):
            return "Software"
        if any(w in sector for w in ["retail", "consumer"]):
            return "Retail"
        return None

    @staticmethod
    def scan_content(content: str, ctx: SecurityContext) -> List[IntegrityViolation]:
        """Scan generated content for semantic contamination."""
        violations = []
        if not content:
            return violations

        content_lower = content.lower()
        relevance_vocab = SemanticContaminationDetector.build_relevance_vocabulary(ctx)

        # Check for contrary-industry terms
        subject_industry = SemanticContaminationDetector.classify_industry(ctx)
        if subject_industry:
            for industry, terms in SemanticContaminationDetector._INDUSTRY_CONTRARY_CLUSTERS.items():
                if industry == subject_industry:
                    continue  # These terms are EXPECTED for the subject's industry
                for term in terms:
                    if term.lower() in content_lower:
                        violations.append(IntegrityViolation(
                            check="semantic-contamination",
                            severity=Severity.WARNING,
                            description=f"Narrative contains '{term}' — this is {industry} "
                                        f"terminology, but the subject is classified as {subject_industry}.",
                            evidence=f"Found '{term}' in generated content.",
                            field="narrative",
                            recommended_action="Verify this term is relevant to the subject company.",
                        ))

        # Check for competing company product mentions
        for pattern, ecosystem_name in SemanticContaminationDetector._GENERIC_SUSPICIOUS_PATTERNS:
            matches = pattern.findall(content)
            if matches:
                # Check if the subject IS that company (relevance vocab match)
                ecosystem_tokens = ecosystem_name.lower().split()
                if not any(t in " ".join(relevance_vocab).lower() for t in ecosystem_tokens):
                    violations.append(IntegrityViolation(
                        check="competing-ecosystem-contamination",
                        severity=Severity.WARNING if len(matches) <= 2 else Severity.CRITICAL,
                        description=f"Narrative references {ecosystem_name} products "
                                    f"({', '.join(matches[:3])}) — may belong to a different company.",
                        evidence=f"Found {len(matches)} references to {ecosystem_name}.",
                        field="narrative",
                        recommended_action="Remove references to competing products unless part of "
                                          "competitive analysis section.",
                    ))

        return violations

    @staticmethod
    def compute_contamination_score(violations: List[IntegrityViolation]) -> float:
        """Score based on contamination violations (100 = clean)."""
        if not violations:
            return 100.0
        critical_count = sum(1 for v in violations if v.severity == Severity.CRITICAL)
        warning_count = sum(1 for v in violations if v.severity == Severity.WARNING)
        if critical_count > 0:
            return max(0.0, 100.0 - critical_count * 40.0 - warning_count * 10.0)
        return max(20.0, 100.0 - warning_count * 15.0)


# ────────────────────────────────────────────────────────────────────────────
# ECONOMIC PLAUSIBILITY ENGINE
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class PlausibilityCheck:
    """Result of a single economic plausibility test."""
    label: str
    passed: bool
    actual_value: Optional[float] = None
    threshold: Optional[float] = None
    description: str = ""
    severity: Severity = Severity.WARNING


class EconomicPlausibilityEngine:
    """Checks whether mathematically-correct results make economic sense.

    Different question from numerical QC:
      NUMERICAL QC: Is the math correct?
      ECONOMIC QC:  Does the result make economic sense?
    """

    # Long-run economic constraints (observable bounds over full market cycles)
    LONG_RUN_REVENUE_CAGR_CAP = 0.30     # 30% sustained is extraordinary
    LONG_RUN_EBIT_MARGIN_CAP = 0.65      # 65% EBIT margins are extreme
    LONG_RUN_FCF_MARGIN_CAP = 0.55       # 55% FCF margins are extreme
    LONG_RUN_TERMINAL_GROWTH_CAP = 0.06   # Terminal growth above long-run nominal GDP
    LONG_RUN_EV_REVENUE_CAP = 50.0       # 50x EV/Revenue
    LONG_RUN_EV_EBITDA_CAP = 50.0        # 50x EV/EBITDA
    MIN_TERMINAL_VALUE_CONTRIBUTION = 0.40  # TV < 40% of total is suspicious
    MAX_TERMINAL_VALUE_CONTRIBUTION = 0.95  # TV > 95% is a perpetuity bet

    @staticmethod
    def check_revenue_growth_plausibility(
        implied_cagr: Optional[float], ctx: SecurityContext
    ) -> PlausibilityCheck:
        """Check if implied revenue CAGR is economically plausible."""
        if implied_cagr is None:
            return PlausibilityCheck(
                "revenue-growth-plausibility", True,
                description="No revenue CAGR to check.",
                severity=Severity.INFO)

        if implied_cagr > EconomicPlausibilityEngine.LONG_RUN_REVENUE_CAGR_CAP:
            return PlausibilityCheck(
                "revenue-growth-plausibility", False,
                actual_value=implied_cagr,
                threshold=EconomicPlausibilityEngine.LONG_RUN_REVENUE_CAGR_CAP,
                description=f"Implied revenue CAGR {implied_cagr:.1%} exceeds "
                            f"long-run observable bound of "
                            f"{EconomicPlausibilityEngine.LONG_RUN_REVENUE_CAGR_CAP:.0%}.",
                severity=Severity.WARNING)

        if implied_cagr < -0.15:
            return PlausibilityCheck(
                "revenue-growth-plausibility", False,
                actual_value=implied_cagr,
                threshold=-0.15,
                description=f"Implied revenue CAGR {implied_cagr:.1%} implies severe "
                            f"contraction — verify against company outlook.",
                severity=Severity.WARNING)

        return PlausibilityCheck("revenue-growth-plausibility", True)

    @staticmethod
    def check_margin_plausibility(
        ebit_margin: Optional[float], fcf_margin: Optional[float]
    ) -> List[PlausibilityCheck]:
        """Check if margins are economically plausible."""
        checks = []

        if ebit_margin is not None:
            if ebit_margin > EconomicPlausibilityEngine.LONG_RUN_EBIT_MARGIN_CAP:
                checks.append(PlausibilityCheck(
                    "ebit-margin-plausibility", False,
                    actual_value=ebit_margin,
                    threshold=EconomicPlausibilityEngine.LONG_RUN_EBIT_MARGIN_CAP,
                    description=f"EBIT margin {ebit_margin:.1%} exceeds long-run "
                                f"observable bound. Sustainably above 65% is rare.",
                    severity=Severity.WARNING))
            elif ebit_margin < -0.05:
                checks.append(PlausibilityCheck(
                    "ebit-margin-plausibility", False,
                    actual_value=ebit_margin,
                    threshold=-0.05,
                    description=f"EBIT margin {ebit_margin:.1%} is negative — "
                                f"verify if this is structural or temporary.",
                    severity=Severity.WARNING))
            else:
                checks.append(PlausibilityCheck("ebit-margin-plausibility", True))

        if fcf_margin is not None:
            if fcf_margin > EconomicPlausibilityEngine.LONG_RUN_FCF_MARGIN_CAP:
                checks.append(PlausibilityCheck(
                    "fcf-margin-plausibility", False,
                    actual_value=fcf_margin,
                    threshold=EconomicPlausibilityEngine.LONG_RUN_FCF_MARGIN_CAP,
                    description=f"FCF margin {fcf_margin:.1%} is extreme — requires "
                                f"negative capex/working-capital, which is unsustainable.",
                    severity=Severity.CRITICAL))

        return checks

    @staticmethod
    def check_terminal_value(
        tv_contribution: Optional[float], terminal_growth: Optional[float]
    ) -> List[PlausibilityCheck]:
        """Check terminal value assumptions are economically plausible."""
        checks = []

        if terminal_growth is not None:
            if terminal_growth > EconomicPlausibilityEngine.LONG_RUN_TERMINAL_GROWTH_CAP:
                checks.append(PlausibilityCheck(
                    "terminal-growth-plausibility", False,
                    actual_value=terminal_growth,
                    threshold=EconomicPlausibilityEngine.LONG_RUN_TERMINAL_GROWTH_CAP,
                    description=f"Terminal growth {terminal_growth:.1%} exceeds "
                                f"long-run nominal GDP — the company would outgrow "
                                f"the economy indefinitely.",
                    severity=Severity.CRITICAL))

        if tv_contribution is not None:
            if tv_contribution > EconomicPlausibilityEngine.MAX_TERMINAL_VALUE_CONTRIBUTION:
                checks.append(PlausibilityCheck(
                    "tv-contribution-plausibility", False,
                    actual_value=tv_contribution,
                    threshold=EconomicPlausibilityEngine.MAX_TERMINAL_VALUE_CONTRIBUTION,
                    description=f"Terminal value is {tv_contribution:.0%} of total "
                                f"enterprise value — this is a perpetuity bet, not "
                                f"a DCF. The explicit forecast period is nearly "
                                f"irrelevant.",
                    severity=Severity.CRITICAL))
            elif tv_contribution < EconomicPlausibilityEngine.MIN_TERMINAL_VALUE_CONTRIBUTION:
                checks.append(PlausibilityCheck(
                    "tv-contribution-plausibility", False,
                    actual_value=tv_contribution,
                    threshold=EconomicPlausibilityEngine.MIN_TERMINAL_VALUE_CONTRIBUTION,
                    description=f"Terminal value is only {tv_contribution:.0%} of "
                                f"total — unusual for a going concern.",
                    severity=Severity.WARNING))

        return checks

    @staticmethod
    def check_dcf_outputs(
        dcf_fvs: Optional[Dict[str, Any]], ctx: SecurityContext
    ) -> Tuple[List[PlausibilityCheck], List[IntegrityViolation]]:
        """Run economic plausibility checks on DCF outputs."""
        checks = []
        violations = []

        if not dcf_fvs or not isinstance(dcf_fvs, dict):
            return checks, violations

        base = dcf_fvs.get("Base")
        current_price = ctx.financials.current_price.value

        if base is not None and current_price and current_price > 0:
            # Implied EV/Revenue multiple check
            if ctx.financials.revenue.is_available and ctx.financials.shares_outstanding.is_available:
                revenue = ctx.financials.revenue.value
                shares = ctx.financials.shares_outstanding.value
                if revenue and shares and shares > 0:
                    ev_revenue = (base * shares) / revenue
                    if ev_revenue > EconomicPlausibilityEngine.LONG_RUN_EV_REVENUE_CAP:
                        checks.append(PlausibilityCheck(
                            "ev-revenue-plausibility", False,
                            actual_value=ev_revenue,
                            threshold=EconomicPlausibilityEngine.LONG_RUN_EV_REVENUE_CAP,
                            description=f"Implied EV/Revenue {ev_revenue:.1f}x is extreme.",
                            severity=Severity.WARNING))

            # Upside/downside ratio check
            upside = (base / current_price) - 1.0
            if upside > 2.0:
                checks.append(PlausibilityCheck(
                    "dcf-upside-plausibility", False,
                    actual_value=upside,
                    threshold=2.0,
                    description=f"DCF implies {upside:.0%} upside from current price. "
                                f"A gap this large typically reflects either a data error "
                                f"or an assumption breakdown, not a market mispricing.",
                    severity=Severity.WARNING))

        # Convert checks to violations for critical issues
        for c in checks:
            if not c.passed and c.severity == Severity.CRITICAL:
                violations.append(IntegrityViolation(
                    check=c.label,
                    severity=Severity.CRITICAL,
                    description=c.description,
                    evidence=f"Value: {c.actual_value}, Threshold: {c.threshold}",
                ))

        return checks, violations

    @staticmethod
    def run_full_check(
        ctx: SecurityContext,
        implied_cagr: Optional[float] = None,
        tv_contribution: Optional[float] = None,
        terminal_growth: Optional[float] = None,
        dcf_fvs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[PlausibilityCheck], List[IntegrityViolation]]:
        """Run all economic plausibility checks."""
        all_checks = []
        all_violations = []

        # Revenue growth
        all_checks.append(
            EconomicPlausibilityEngine.check_revenue_growth_plausibility(
                implied_cagr, ctx))

        # Margins
        ebit_margin = ctx.financials.ebit_margin.value
        fcf_margin = ctx.financials.fcf_margin.value
        all_checks.extend(
            EconomicPlausibilityEngine.check_margin_plausibility(
                ebit_margin, fcf_margin))

        # Terminal value
        all_checks.extend(
            EconomicPlausibilityEngine.check_terminal_value(
                tv_contribution, terminal_growth))

        # DCF outputs
        dcf_checks, dcf_violations = EconomicPlausibilityEngine.check_dcf_outputs(
            dcf_fvs, ctx)
        all_checks.extend(dcf_checks)
        all_violations.extend(dcf_violations)

        return all_checks, all_violations

    @staticmethod
    def compute_plausibility_score(
        checks: List[PlausibilityCheck], violations: List[IntegrityViolation]
    ) -> float:
        """0-100 economic plausibility score."""
        total = len(checks) + len(violations)
        if total == 0:
            return 100.0
        critical = (sum(1 for c in checks if not c.passed and c.severity == Severity.CRITICAL)
                    + sum(1 for v in violations if v.severity == Severity.CRITICAL))
        failing = (sum(1 for c in checks if not c.passed and c.severity != Severity.CRITICAL)
                   + sum(1 for v in violations if v.severity != Severity.CRITICAL))
        score = 100.0 - critical * 35.0 - failing * 8.0
        return max(0.0, score)


# ────────────────────────────────────────────────────────────────────────────
# HARD CONFIDENCE ENGINE
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ConfidenceComponent:
    """A single component of the confidence calculation."""
    name: str
    score: float           # 0-100
    weight: float          # 0-1
    hard_ceiling: Optional[float] = None  # Maximum allowed (None = no cap)
    hard_floor: Optional[float] = None    # Minimum allowed (None = no floor)
    constrained: bool = False             # Whether a constraint was applied
    constraint_reason: str = ""


@dataclass
class ConfidenceResult:
    """Complete confidence assessment."""
    components: List[ConfidenceComponent] = field(default_factory=list)
    weighted_average: float = 0.0
    hard_constrained_score: float = 0.0   # After applying hard ceilings/floors
    overall_confidence: float = 0.0       # Final confidence (0-100)
    blocking_failures: List[str] = field(default_factory=list)
    degradation_reasons: List[str] = field(default_factory=list)


class HardConfidenceEngine:
    """Confidence with HARD CONSTRAINTS — not just a weighted average.

    A critical failure in any component creates a ceiling that other strong
    components CANNOT mathematically average away.

    Confidence = MIN(weighted_average, lowest_ceiling_of_critical_failures)
    """

    # Default weights (matching the existing disclosed formula)
    DEFAULT_WEIGHTS = {
        "data_quality": 0.30,
        "model_robustness": 0.25,
        "forecast_certainty": 0.25,
        "regime_clarity": 0.20,
        "entity_integrity": 0.15,       # New components
        "economic_plausibility": 0.10,
    }

    @staticmethod
    def compute_data_quality(ctx: SecurityContext) -> ConfidenceComponent:
        """Data quality component with hard ceiling based on missing critical data."""
        completeness = ctx.financials.completeness_score()

        if not ctx.financials.current_price.is_available:
            return ConfidenceComponent(
                "data_quality", 22.0, HardConfidenceEngine.DEFAULT_WEIGHTS["data_quality"],
                hard_ceiling=30.0, constrained=True,
                constraint_reason="No live price — data quality cannot exceed 30%.")

        score = min(82.0, 35.0 + 45.0 * completeness)

        # Critical fields missing = hard ceiling
        missing_critical, critical_fields = ctx.is_critical_field_missing()
        ceiling = None
        if missing_critical:
            if "revenue" in critical_fields and "shares_outstanding" in critical_fields:
                ceiling = 50.0  # Quote-only: fundamentals unavailable
            elif "current_price" in critical_fields:
                ceiling = 35.0

        return ConfidenceComponent(
            "data_quality", score,
            HardConfidenceEngine.DEFAULT_WEIGHTS["data_quality"],
            hard_ceiling=ceiling,
            constrained=ceiling is not None,
            constraint_reason=f"Missing critical fields: {', '.join(critical_fields)}" if missing_critical else "")

    @staticmethod
    def compute_model_robustness(ctx: SecurityContext) -> ConfidenceComponent:
        """Model robustness — higher when reported fundamentals are attached."""
        score = 62.0 if ctx.financials.revenue.is_available else 45.0
        return ConfidenceComponent(
            "model_robustness", score,
            HardConfidenceEngine.DEFAULT_WEIGHTS["model_robustness"])

    @staticmethod
    def compute_forecast_certainty(scenario_returns: List[float]) -> ConfidenceComponent:
        """Forecast certainty from scenario dispersion."""
        import numpy as np
        rets = scenario_returns or [0.0]
        disp = float(np.std(rets)) if len(rets) > 1 else 0.15
        score = max(30.0, min(70.0, 58.0 - disp * 55.0))
        return ConfidenceComponent(
            "forecast_certainty", score,
            HardConfidenceEngine.DEFAULT_WEIGHTS["forecast_certainty"])

    @staticmethod
    def compute_entity_integrity(entity_score: float) -> ConfidenceComponent:
        """Entity integrity as a confidence component with hard ceiling."""
        ceiling = None
        if entity_score < 30.0:
            ceiling = 40.0  # Critical entity issues cap overall confidence
        return ConfidenceComponent(
            "entity_integrity", entity_score,
            HardConfidenceEngine.DEFAULT_WEIGHTS["entity_integrity"],
            hard_ceiling=ceiling,
            constrained=entity_score < 30.0,
            constraint_reason="Critical entity integrity failure." if entity_score < 30.0 else "")

    @staticmethod
    def compute_economic_plausibility(plausibility_score: float) -> ConfidenceComponent:
        """Economic plausibility as a confidence component."""
        ceiling = None
        if plausibility_score < 30.0:
            ceiling = 50.0
        return ConfidenceComponent(
            "economic_plausibility", plausibility_score,
            HardConfidenceEngine.DEFAULT_WEIGHTS["economic_plausibility"],
            hard_ceiling=ceiling,
            constrained=plausibility_score < 30.0,
            constraint_reason="Multiple economic plausibility failures." if plausibility_score < 30.0 else "")

    @staticmethod
    def compute_full_confidence(
        ctx: SecurityContext,
        scenario_returns: List[float],
        entity_score: float = 100.0,
        plausibility_score: float = 100.0,
    ) -> ConfidenceResult:
        """Compute full confidence with hard constraints.

        The final score is the MINIMUM of the weighted average and the lowest
        hard ceiling from any component that has a constraint. This prevents
        strong components from averaging away critical failures.
        """
        components = [
            HardConfidenceEngine.compute_data_quality(ctx),
            HardConfidenceEngine.compute_model_robustness(ctx),
            HardConfidenceEngine.compute_forecast_certainty(scenario_returns),
            ConfidenceComponent(  # regime_clarity — existing component
                "regime_clarity", 62.0,
                HardConfidenceEngine.DEFAULT_WEIGHTS["regime_clarity"]),
            HardConfidenceEngine.compute_entity_integrity(entity_score),
            HardConfidenceEngine.compute_economic_plausibility(plausibility_score),
        ]

        # Compute weighted average
        total_weight = sum(c.weight for c in components)
        weighted_avg = (sum(c.score * c.weight for c in components)
                        / total_weight if total_weight > 0 else 0.0)

        # Apply hard constraints: find the lowest ceiling from constrained components
        constrained = [c for c in components if c.constrained and c.hard_ceiling is not None]
        if constrained:
            lowest_ceiling = min(c.hard_ceiling for c in constrained)
            hard_constrained_score = min(weighted_avg, lowest_ceiling)
        else:
            hard_constrained_score = weighted_avg

        # Blocking failures: critical issues that should prevent a rating
        blocking = []
        degradation = []
        for c in components:
            if c.constrained and c.hard_ceiling is not None:
                degradation.append(f"{c.name}: capped at {c.hard_ceiling:.0f}% — {c.constraint_reason}")
            if c.constrained and c.hard_ceiling is not None and c.hard_ceiling < 40.0:
                blocking.append(f"{c.name}: {c.constraint_reason}")

        return ConfidenceResult(
            components=components,
            weighted_average=round(weighted_avg, 1),
            hard_constrained_score=round(hard_constrained_score, 1),
            overall_confidence=round(hard_constrained_score, 1),
            blocking_failures=blocking,
            degradation_reasons=degradation,
        )


# ────────────────────────────────────────────────────────────────────────────
# FULL INTEGRITY CHECK — runs all engines against a SecurityContext
# ────────────────────────────────────────────────────────────────────────────

def run_full_integrity_check(
    ctx: SecurityContext,
    narrative_content: str = "",
    scenario_returns: Optional[List[float]] = None,
    implied_cagr: Optional[float] = None,
    tv_contribution: Optional[float] = None,
    terminal_growth: Optional[float] = None,
    dcf_fvs: Optional[Dict[str, Any]] = None,
) -> IntegrityReport:
    """Run all integrity checks and return a comprehensive report.

    This is the SINGLE entry point for integrity validation. Every analysis
    must pass through this gate before producing a final conclusion.
    """
    report = IntegrityReport()

    # 1. Entity integrity
    entity_score, entity_violations = EntityIntegrityEngine.compute_entity_score(ctx)
    report.violations.extend(entity_violations)
    report.entity_integrity_score = entity_score

    # 2. Semantic contamination
    if narrative_content:
        semantic_violations = SemanticContaminationDetector.scan_content(
            narrative_content, ctx)
        report.violations.extend(semantic_violations)
        report.semantic_contamination_score = (
            SemanticContaminationDetector.compute_contamination_score(semantic_violations))
    else:
        report.semantic_contamination_score = 100.0

    # 3. Economic plausibility
    plausibility_checks, plausibility_violations = EconomicPlausibilityEngine.run_full_check(
        ctx, implied_cagr, tv_contribution, terminal_growth, dcf_fvs)
    report.violations.extend(plausibility_violations)
    report.economic_plausibility_score = (
        EconomicPlausibilityEngine.compute_plausibility_score(
            plausibility_checks, plausibility_violations))

    # 4. Determine overall integrity status
    critical_count = sum(1 for v in report.violations if v.severity == Severity.CRITICAL)
    warning_count = sum(1 for v in report.violations if v.severity == Severity.WARNING)

    if critical_count > 0 or report.entity_integrity_score < 30.0:
        report.overall_integrity = IntegrityStatus.INVALID
        report.blocks_rating = True
    elif (report.entity_integrity_score < 70.0
          or report.economic_plausibility_score < 60.0
          or report.semantic_contamination_score < 60.0):
        report.overall_integrity = IntegrityStatus.DEGRADED
        report.blocks_rating = report.economic_plausibility_score < 30.0
    elif warning_count > 3 or report.entity_integrity_score < 90.0:
        report.overall_integrity = IntegrityStatus.CONDITIONAL
    else:
        report.overall_integrity = IntegrityStatus.VERIFIED

    # 5. Build summary
    parts = []
    if report.overall_integrity == IntegrityStatus.VERIFIED:
        parts.append("All integrity checks passed.")
    elif report.overall_integrity == IntegrityStatus.CONDITIONAL:
        parts.append(f"Integrity: CONDITIONAL ({warning_count} warnings).")
    elif report.overall_integrity == IntegrityStatus.DEGRADED:
        parts.append(f"Integrity: DEGRADED ({critical_count} critical, {warning_count} warnings).")
    else:
        parts.append(f"Integrity: INVALID ({critical_count} critical failures).")

    if report.blocks_rating:
        parts.append("RATING BLOCKED — critical integrity failures present.")
    if critical_count > 0:
        for v in [v for v in report.violations if v.severity == Severity.CRITICAL]:
            parts.append(f"  CRITICAL: {v.description}")

    report.summary = "\n".join(parts)
    return report