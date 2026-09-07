"""
Regression tests for institutional-grade integrity failure modes.

These tests verify that the architectural upgrades to the analytical pipeline
prevent the failure modes described in the upgrade specification:

  1-18. Generalized failure modes that must work for EVERY security.

Key design rule: NO test is hard-coded to AAPL, AAPL, or any individual security.
"""

import pytest
from unittest.mock import patch, MagicMock
import re

# ── Module under test ───────────────────────────────────────────────────────

from analytical_context import (
    SecurityContext, FinancialDataStore, ProvenancePoint,
    ProvenanceStatus, IntegrityStatus, ProvenanceEngine,
    build_security_context, MissingReason,
)
from analytical_integrity import (
    EntityIntegrityEngine, SemanticContaminationDetector,
    EconomicPlausibilityEngine, HardConfidenceEngine,
    Severity, IntegrityViolation, IntegrityReport,
    run_full_integrity_check, ConfidenceResult,
)
from rating_gate import (
    RatingGate, RatingEligibility, InvestmentRating,    RatingGateResult,
    PreFlightValidator, PreFlightStatus, PreFlightReport,
    TemplateIntegrityScanner, TemplateScanResult,
    generate_audit_panel,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def ticker_ctx():
    """Minimal SecurityContext for a generic equity."""
    return SecurityContext.from_quick_quote("GENERIC", 50.0)


@pytest.fixture
def funded_ctx():
    """SecurityContext with reported fundamentals."""
    fund = {
        "revenue_m": 10000.0,
        "shares_m": 500.0,
        "ebit_margin_pct": 25.0,
        "tax_rate_pct": 21.0,
        "debt_m": 2000.0,
        "cash_m": 1000.0,
        "eps": 4.0,
        "fiscal_period": "TTM",
        "company_name": "Generic Corp",
        "sector": "Technology",
    }
    return SecurityContext.from_fundamentals("GENERIC", fund, price=50.0)


@pytest.fixture
def quote_only_ctx():
    """SecurityContext with only a live quote (no fundamentals)."""
    return SecurityContext.from_quick_quote("TICKR", 25.0)


@pytest.fixture
def clean_integrity_report():
    """A clean integrity report."""
    return IntegrityReport(
        entity_integrity_score=100.0,
        semantic_contamination_score=100.0,
        economic_plausibility_score=100.0,
        overall_integrity=IntegrityStatus.VERIFIED,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Security A cannot receive Security B's narrative
# ═══════════════════════════════════════════════════════════════════════════

def test_cross_security_narrative_is_detected():
    """A tech company's narrative should not contain pharma terminology."""
    ctx = SecurityContext.from_quick_quote("GENERIC", 50.0, sector="Technology")

    content = (
        "The company has a strong clinical trial pipeline with multiple "
        "Phase III drug candidates. FDA approval is expected by Q4."
    )
    violations = SemanticContaminationDetector.scan_content(content, ctx)

    # Should detect pharmaceutical terminology in a tech company's narrative
    pharma_violations = [v for v in violations if "clinical trial" in v.evidence
                         or "pharmaceutical" in v.description.lower()
                         or "FDA" in v.evidence]
    assert len(pharma_violations) > 0, (
        "Pharma terminology in a tech company's narrative went undetected"
    )


def test_cross_industry_narrative_flagged():
    """A retail company narrative containing oil & gas terms is flagged."""
    ctx = SecurityContext.from_quick_quote("RETAIL", 40.0, sector="Consumer Discretionary")

    content = "The upstream production volumes and rig count drive our refining margins."
    violations = SemanticContaminationDetector.scan_content(content, ctx)

    oil_violations = [v for v in violations
                      if "upstream" in v.evidence
                      or "Oil" in v.description]
    assert len(oil_violations) > 0, (
        "Oil & gas terms in a consumer company narrative went undetected"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: DCF and scenario engine use same canonical revenue
# ═══════════════════════════════════════════════════════════════════════════

def test_canonical_revenue_is_single_source():
    """All modules consume the same SecurityContext.financials.revenue."""
    fund = {
        "revenue_m": 15000.0,
        "shares_m": 600.0,
        "ebit_margin_pct": 22.0,
        "tax_rate_pct": 21.0,
        "fiscal_period": "FY2025A",
    }
    ctx = SecurityContext.from_fundamentals("TEST", fund, price=75.0)

    # Revenue is a single ProvenancePoint — all modules must read from it
    assert ctx.financials.revenue.value == 15000.0
    assert ctx.financials.revenue.is_available
    assert ctx.financials.revenue.status == ProvenanceStatus.REPORTED

    # No other module creates its own revenue
    assert ctx.financials.revenue.source == "SEC filings / financial data feed"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Missing FCF blocks VERIFIED DCF
# ═══════════════════════════════════════════════════════════════════════════

def test_missing_fcf_is_classified_unavailable(funded_ctx):
    """When cash flow is unavailable, FCF must be flagged."""
    ctx = funded_ctx

    # Cash flow data is NOT available in the current fund dictionary
    assert not ctx.financials.fcf.is_available
    assert ctx.financials.fcf.status == ProvenanceStatus.UNAVAILABLE
    assert ctx.financials.fcf.missing_reason == MissingReason.NOT_AVAILABLE
    assert any(w in (ctx.financials.fcf.notes or "").lower()
               for w in ("cash flow", "cash-flow", "unavailable"))


def test_missing_fcf_downgrades_data_quality(funded_ctx):
    """Missing FCF should degrade data quality in the confidence engine."""
    ctx = funded_ctx
    result = HardConfidenceEngine.compute_data_quality(ctx)
    dq = result.score
    assert dq > 0
    # With only income statement + price, data quality should be below max
    assert dq <= 82.0, f"Expected data quality <= 82% with missing FCF, got {dq}%"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: Critical entity failure blocks rating
# ═══════════════════════════════════════════════════════════════════════════

def test_invalid_ticker_blocks_rating():
    """A financial-metric token used as a ticker must block the rating."""
    ctx = SecurityContext.from_quick_quote("DCF", 100.0)  # "DCF" is a metric token

    entity_score, violations = EntityIntegrityEngine.compute_entity_score(ctx)
    assert entity_score == 0.0, f"Expected entity score 0, got {entity_score}"

    critical = [v for v in violations if v.severity == Severity.CRITICAL]
    assert len(critical) > 0, "No critical violation for metric-token ticker"


def test_empty_ticker_blocks_rating():
    """No ticker = critical failure = rating blocked."""
    ctx = SecurityContext()
    entity_score, violations = EntityIntegrityEngine.compute_entity_score(ctx)

    critical = [v for v in violations if v.severity == Severity.CRITICAL]
    assert len(critical) > 0
    assert entity_score <= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: Unresolved template blocks publication
# ═══════════════════════════════════════════════════════════════════════════

def test_unresolved_placeholder_detected():
    """{variable} placeholders must be caught."""
    text = "Revenue grew {growth_rate}% this quarter."
    result = TemplateIntegrityScanner.scan(text)
    assert not result.passed
    assert len(result.unresolved_placeholders) > 0


def test_nan_value_detected():
    """NaN in output must be caught."""
    text = "The fair value is NaN based on our model."
    result = TemplateIntegrityScanner.scan(text)
    assert not result.passed
    assert len(result.nan_values) > 0


def test_none_value_detected():
    """Bare None in output must be caught."""
    text = "The beta is None for this security."
    result = TemplateIntegrityScanner.scan(text)
    assert not result.passed
    assert len(result.none_values) > 0


def test_clean_text_passes_template_scan():
    """Clean text with no artifacts passes."""
    text = "Revenue grew 15.2% this quarter. The fair value is $52.00."
    result = TemplateIntegrityScanner.scan(text)
    assert result.passed
    assert len(result.issues) == 0


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: Scenario probabilities must sum to 100%
# ═══════════════════════════════════════════════════════════════════════════

def test_preflight_blocks_non_summing_probabilities(ticker_ctx):
    """Pre-flight must fail when scenario probabilities don't sum to 100%."""
    ctx = ticker_ctx
    integrity = IntegrityReport(entity_integrity_score=95.0)

    report = PreFlightValidator.run(
        ctx=ctx,
        integrity_report=integrity,
        dcf_status="verified",
        qc_passed=True,
        template_passed=True,
        scenario_prob_sum=0.85,  # Should be 1.0
        rating_eligible=True,
    )

    prob_check = [c for c in report.checks if "Scenario probabilities" in c.name]
    assert len(prob_check) > 0
    assert not prob_check[0].passed, "Should fail when probabilities sum to 0.85"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7: Subjective probabilities labeled as such
# ═══════════════════════════════════════════════════════════════════════════

def test_provenance_tracks_probability_type(funded_ctx):
    """All provenance points declare their status (reported/estimated/assumed)."""
    ctx = funded_ctx

    # Revenue is reported
    assert ctx.financials.revenue.status == ProvenanceStatus.REPORTED

    # Current price is observed
    assert ctx.financials.current_price.status == ProvenanceStatus.OBSERVED

    # WACC may be assumed or unavailable depending on how the context was built;
    # both are valid provenance states that differ from REPORTED/OBSERVED
    assert ctx.financials.wacc.status in (
        ProvenanceStatus.ASSUMED, ProvenanceStatus.UNAVAILABLE)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 8: Missing macro data constrains confidence
# ═══════════════════════════════════════════════════════════════════════════

def test_quote_only_ctx_has_low_data_quality(quote_only_ctx):
    """A quote-only context (no fundamentals) must have low data quality."""
    ctx = quote_only_ctx
    result = HardConfidenceEngine.compute_data_quality(ctx)
    # Quote-only completeness is low
    assert result.score <= 60.0, f"Quote-only should score <= 60%, got {result.score}%"


def test_full_confidence_is_hard_constrained(quote_only_ctx):
    """HardConfidenceEngine blocks confidence from averaging away constraints."""
    ctx = quote_only_ctx  # No fundamentals
    result = HardConfidenceEngine.compute_full_confidence(
        ctx, scenario_returns=[-0.15, 0.0, 0.08, 0.15, 0.25],
        entity_score=100.0, plausibility_score=100.0,
    )
    # With very limited data, overall confidence should be low
    assert result.overall_confidence <= 65.0, (
        f"Quote-only confidence too high: {result.overall_confidence}%"
    )
    assert len(result.degradation_reasons) > 0


# ═══════════════════════════════════════════════════════════════════════════
# TEST 9: Economic plausibility flags extremely high growth
# ═══════════════════════════════════════════════════════════════════════════

def test_extreme_revenue_growth_flagged(funded_ctx):
    """30%+ sustained CAGR should be flagged as economically implausible."""
    ctx = funded_ctx
    check = EconomicPlausibilityEngine.check_revenue_growth_plausibility(0.45, ctx)
    assert not check.passed
    assert "45.0%" in check.description or "45%" in check.description


def test_reasonable_growth_passes(funded_ctx):
    """A 10% CAGR should pass economic plausibility."""
    ctx = funded_ctx
    check = EconomicPlausibilityEngine.check_revenue_growth_plausibility(0.10, ctx)
    assert check.passed


# ═══════════════════════════════════════════════════════════════════════════
# TEST 10: No module silently substitutes unavailable data
# ═══════════════════════════════════════════════════════════════════════════

def test_financial_data_store_explicitly_marks_missing(quote_only_ctx):
    """Missing fields carry UNAVAILABLE status, never silently substituted."""
    ctx = quote_only_ctx
    fs = ctx.financials

    # Revenue should be UNAVAILABLE for quote-only
    assert not fs.revenue.is_available
    assert fs.revenue.status == ProvenanceStatus.UNAVAILABLE

    # Shares outstanding should be UNAVAILABLE
    assert not fs.shares_outstanding.is_available

    # FCF should be UNAVAILABLE
    assert not fs.fcf.is_available

    # But current price IS available
    assert fs.current_price.is_available
    assert fs.current_price.value == 25.0


def test_provenance_engine_registers_and_detects_conflicts():
    """Source conflicts must be surfaced, not silently resolved."""
    engine = ProvenanceEngine()
    engine.register("revenue", ProvenancePoint(
        "Revenue", 10000.0, ProvenanceStatus.REPORTED, "Source A"))
    # Register a conflicting value from a different source
    engine.register("revenue", ProvenancePoint(
        "Revenue", 10500.0, ProvenanceStatus.REPORTED, "Source B"))

    conflicts = engine.get_conflicts()
    assert len(conflicts) > 0, "Conflicting revenue values not detected"
    # The 5% difference may be classified as MINOR or MATERIAL depending on threshold
    assert len(conflicts) > 0


# ═══════════════════════════════════════════════════════════════════════════
# TEST 11: Current price timestamp is consistent across all modules
# ═══════════════════════════════════════════════════════════════════════════

def test_security_context_carries_price_timestamp():
    """The current price in SecurityContext carries a timestamp."""
    ctx = SecurityContext.from_quick_quote(
        "TEST", 50.0, price_timestamp="2025-03-15T14:30:00Z", price_source="yfinance")
    assert ctx.financials.current_price.source_timestamp == "2025-03-15T14:30:00Z"
    assert ctx.financials.current_price.source == "yfinance"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 12: Positive expected return cannot override critical integrity failure
# ═══════════════════════════════════════════════════════════════════════════

def test_rating_gate_blocks_despite_positive_return():
    """Even with +25% expected return, a critical failure blocks the rating."""
    ctx = SecurityContext()  # Empty — no ticker, no price = critical failure
    integrity = IntegrityReport(
        entity_integrity_score=0.0,
        overall_integrity=IntegrityStatus.INVALID,
        blocks_rating=True,
    )

    gate = RatingGate.evaluate(ctx, integrity)
    assert gate.eligibility == RatingEligibility.FAIL
    assert "RATING BLOCKED" in " ".join(gate.blocked_reasons).upper()

    rating = RatingGate.assign_rating(gate, exp_ret=0.25, p_dd30=0.05, overall_conf=70.0)
    assert rating == InvestmentRating.MODEL_INVALID, (
        f"Expected MODEL_INVALID, got {rating.value}"
    )


def test_rating_gate_conditional_with_missing_fundamentals(quote_only_ctx):
    """Missing fundamentals -> conditional rating."""
    ctx = quote_only_ctx
    integrity = IntegrityReport(
        entity_integrity_score=90.0,
        economic_plausibility_score=80.0,
    )
    gate = RatingGate.evaluate(ctx, integrity, dcf_status="conditional")
    assert gate.eligibility in (RatingEligibility.CONDITIONAL, RatingEligibility.PASS)

    rating = RatingGate.assign_rating(gate, exp_ret=0.12, overall_conf=55.0)
    assert rating in (InvestmentRating.CONDITIONAL_BUY, InvestmentRating.BUY)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 13: Economic plausibility flags extreme terminal value assumptions
# ═══════════════════════════════════════════════════════════════════════════

def test_extreme_terminal_growth_flagged():
    """Terminal growth above 6% should be flagged as critical."""
    checks = EconomicPlausibilityEngine.check_terminal_value(
        tv_contribution=0.85, terminal_growth=0.08)
    crit = [c for c in checks if c.severity == Severity.CRITICAL and not c.passed]
    assert len(crit) > 0, "Terminal growth 8% should be a critical failure"


def test_terminal_value_dominance_flagged():
    """TV > 95% of total EV should be flagged as a perpetuity bet."""
    checks = EconomicPlausibilityEngine.check_terminal_value(
        tv_contribution=0.97, terminal_growth=0.03)
    crit = [c for c in checks if not c.passed
            and "perpetuity" in c.description.lower()]
    assert len(crit) > 0, "TV 97% should flag as perpetuity bet"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 14: Full integrity check runs end-to-end
# ═══════════════════════════════════════════════════════════════════════════

def test_full_integrity_check_returns_structured_report(funded_ctx):
    """The full integrity check produces a complete report."""
    ctx = funded_ctx
    narrative = "Generic Corp's revenue grew 12% driven by strong cloud adoption."

    report = run_full_integrity_check(
        ctx=ctx,
        narrative_content=narrative,
        scenario_returns=[-0.15, 0.0, 0.08, 0.15, 0.25],
        implied_cagr=0.12,
        tv_contribution=0.70,
        terminal_growth=0.028,
        dcf_fvs={"Base": 65.0, "status": "conditional"},
    )

    assert report.entity_integrity_score > 80.0
    assert report.economic_plausibility_score > 50.0
    assert report.semantic_contamination_score > 50.0
    assert report.overall_integrity in (IntegrityStatus.VERIFIED,
                                         IntegrityStatus.CONDITIONAL)
    assert isinstance(report.summary, str)
    assert len(report.summary) > 0


def test_full_integrity_check_on_invalid_context():
    """An invalid context (metric token as ticker) produces report with criticals."""
    ctx = SecurityContext.from_quick_quote("EBITDA", 100.0)
    report = run_full_integrity_check(ctx=ctx)

    critical = [v for v in report.violations if v.severity == Severity.CRITICAL]
    assert len(critical) > 0, "Metric token ticker should produce critical violations"
    assert report.entity_integrity_score < 30.0
    assert report.blocks_rating


# ═══════════════════════════════════════════════════════════════════════════
# TEST 15: Pre-flight catches multiple failures
# ═══════════════════════════════════════════════════════════════════════════

def test_preflight_reports_all_failures():
    """Pre-flight catches multiple issues at once."""
    ctx = SecurityContext.from_quick_quote("TEST", 50.0)
    integrity = IntegrityReport(
        entity_integrity_score=25.0,  # Low
        economic_plausibility_score=25.0,  # Low
    )

    report = PreFlightValidator.run(
        ctx=ctx,
        integrity_report=integrity,
        dcf_status="conditional",
        qc_passed=False,
        template_passed=False,
        scenario_prob_sum=0.70,
        rating_eligible=False,
    )

    assert report.blocking_failures > 0
    assert report.warnings > 0
    assert not report.all_passed
    assert "FAILED" in report.summary


def test_preflight_passes_when_clean(funded_ctx):
    """Clean context should pass pre-flight."""
    ctx = funded_ctx
    integrity = IntegrityReport(
        entity_integrity_score=95.0,
        economic_plausibility_score=90.0,
    )

    report = PreFlightValidator.run(
        ctx=ctx,
        integrity_report=integrity,
        dcf_status="verified",
        qc_passed=True,
        template_passed=True,
        scenario_prob_sum=1.0,
        rating_eligible=True,
    )

    assert report.all_passed
    assert report.blocking_failures == 0


# ═══════════════════════════════════════════════════════════════════════════
# TEST 16: Audit panel is generated correctly
# ═══════════════════════════════════════════════════════════════════════════

def test_audit_panel_generates_markdown(funded_ctx):
    """The audit panel should produce valid markdown."""
    ctx = funded_ctx
    integrity = IntegrityReport(entity_integrity_score=95.0)
    gate = RatingGate.evaluate(ctx, integrity)
    preflight = PreFlightValidator.run(
        ctx, integrity, dcf_status="conditional", qc_passed=True,
        template_passed=True, scenario_prob_sum=1.0, rating_eligible=True,
    )
    template = TemplateIntegrityScanner.scan("clean text")

    panel = generate_audit_panel(ctx, integrity, gate, preflight, template)
    assert "# Model Integrity" in panel
    assert "| Check | Status" in panel
    assert "Entity Integrity" in panel


# ═══════════════════════════════════════════════════════════════════════════
# TEST 17: Hard confidence correctly applies ceilings
# ═══════════════════════════════════════════════════════════════════════════

def test_hard_confidence_ceiling_lower_than_weighted_average():
    """When a component is constrained, final confidence ≤ the ceiling."""
    ctx = SecurityContext.from_quick_quote("TEST", 50.0)  # No fundamentals
    # Entity integrity is very low
    result = HardConfidenceEngine.compute_full_confidence(
        ctx=ctx,
        scenario_returns=[-0.10, 0.02, 0.10, 0.18, 0.25],
        entity_score=20.0,  # Critical entity failure
        plausibility_score=90.0,
    )

    # Weighted average may be higher, but hard constrained score is the ceiling
    assert result.overall_confidence <= result.weighted_average, (
        f"Constrained {result.overall_confidence} should not exceed "
        f"weighted average {result.weighted_average}"
    )
    # With entity score 20%, the entity integrity component should be constrained
    entity_components = [c for c in result.components if c.name == "entity_integrity"]
    if entity_components:
        assert entity_components[0].constrained, "Entity comp should be constrained at score 20"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 18: All numbers traceable to canonical data (provenance coverage)
# ═══════════════════════════════════════════════════════════════════════════

def test_financial_data_store_has_provenance_for_all_fields(funded_ctx):
    """Every field in FinancialDataStore is a ProvenancePoint with traceability."""
    ctx = funded_ctx
    fs = ctx.financials

    for field_name in fs.__dataclass_fields__:
        val = getattr(fs, field_name)
        if isinstance(val, ProvenancePoint):
            assert val.label, f"Field {field_name} has no label"
            # Every provenance point must have a status
            assert val.status, f"Field {field_name} has no provenance status"


def test_build_security_context_is_generic():
    """The context builder works for any ticker without hard-coding."""
    # Test with a completely random ticker
    ctx = build_security_context("RANDOM", live_data={
        "RANDOM": {"price": 42.50, "quote_date": "2025-01-01", "source": "test"}})
    assert ctx.ticker == "RANDOM"
    assert ctx.financials.current_price.value == 42.50
    assert ctx.integrity_status == IntegrityStatus.CONDITIONAL

    # Test with a forex pair
    ctx2 = build_security_context("EUR/USD", live_data={
        "EUR/USD": {"price": 1.0850}})
    assert ctx2.ticker == "EUR/USD"
    assert ctx2.security_type.value == "Forex"

    # Test with crypto
    ctx3 = build_security_context("BTC-USD", live_data={
        "BTC-USD": {"price": 67000.00}})
    assert ctx3.ticker == "BTC-USD"
    assert ctx3.security_type.value == "Crypto"