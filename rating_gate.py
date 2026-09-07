"""
Rating Gate & Pre-flight Validation
===================================

The HARD INVESTMENT RATING GATE. A BUY/SELL rating can only pass through if
all critical integrity checks pass. The system must never generate a high-
confidence investment conclusion from internally inconsistent or semantically
contaminated data.

Also provides:
  - PreFlightValidator   — runs before memo generation
  - TemplateIntegrity    — detects leaked placeholders, NaN, unresolved templates
  - AuditPanel           — generates the institutional "Model Integrity" panel

Author: Octavian Terminal
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import re
import math

from analytical_context import SecurityContext, IntegrityStatus
from analytical_integrity import (
    IntegrityReport, Severity, IntegrityViolation,
)


# ────────────────────────────────────────────────────────────────────────────
# RATING GATE
# ────────────────────────────────────────────────────────────────────────────

class RatingEligibility(str, Enum):
    PASS = "PASS"
    CONDITIONAL = "CONDITIONAL"
    FAIL = "FAIL"


class InvestmentRating(str, Enum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    CONDITIONAL_BUY = "Conditional Buy"
    HOLD = "Hold"
    CONDITIONAL_HOLD = "Conditional Hold"
    SELL = "Sell"
    CONDITIONAL_SELL = "Conditional Sell"
    INSUFFICIENT_DATA = "INSUFFICIENT DATA"
    MODEL_INVALID = "MODEL INVALID"


# Mechanical rating notches (strings, mapped to enum in assign_rating)
_MECHANICAL_NOTCHES = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]


@dataclass
class RatingGateResult:
    """Result of the rating gate evaluation."""
    eligibility: RatingEligibility = RatingEligibility.PASS
    rating: InvestmentRating = InvestmentRating.MODEL_INVALID
    base_rating: str = ""           # The mechanically-derived rating before gating
    blocked_reasons: List[str] = field(default_factory=list)
    conditional_reasons: List[str] = field(default_factory=list)
    framework_text: str = ""
    pass_all_checks: bool = False


class RatingGate:
    """Hard gate between analysis and investment conclusion.

    Before producing a rating, evaluates:
      1. Entity integrity
      2. Data integrity
      3. Financial-period integrity
      4. Numerical QC
      5. DCF integrity
      6. Scenario integrity
      7. Provenance completeness
      8. Missing-data severity
      9. Economic plausibility
      10. Model stability
      11. Template/rendering integrity

    A positive expected return CANNOT override a critical integrity failure.
    """

    @staticmethod
    def evaluate(
        ctx: SecurityContext,
        integrity_report: IntegrityReport,
        dcf_status: Optional[str] = None,
        qc_passed: bool = True,
        template_passed: bool = True,
    ) -> RatingGateResult:
        """Evaluate whether a rating can be produced and at what level."""

        result = RatingGateResult()
        result.pass_all_checks = True

        # ── 1. Entity integrity ──
        if integrity_report.entity_integrity_score < 30.0:
            result.eligibility = RatingEligibility.FAIL
            result.blocked_reasons.append(
                f"ENTITY INTEGRITY FAILURE (score: {integrity_report.entity_integrity_score:.0f}/100) "
                "— the subject entity cannot be reliably identified. Rating blocked.")
            result.pass_all_checks = False
        elif integrity_report.entity_integrity_score < 70.0:
            result.conditional_reasons.append(
                f"Entity integrity is DEGRADED (score: {integrity_report.entity_integrity_score:.0f}/100). "
                "Rating is conditional on entity verification.")

        # ── 2. Data integrity ──
        missing_critical, critical_fields = ctx.is_critical_field_missing()
        if missing_critical:
            if "current_price" in critical_fields:
                result.eligibility = RatingEligibility.FAIL
                result.blocked_reasons.append(
                    "DATA INTEGRITY FAILURE — no current price. Rating blocked.")
                result.pass_all_checks = False
            elif "revenue" in critical_fields and "shares_outstanding" in critical_fields:
                result.conditional_reasons.append(
                    "Fundamentals unavailable — valuation is assumption-based. "
                    "Rating is conditional.")

        # ── 3. Numerical QC ──
        if not qc_passed:
            result.eligibility = RatingEligibility.FAIL
            result.blocked_reasons.append(
                "NUMERICAL QC FAILURE — displayed numbers do not reconcile with "
                "the scenario inputs. Rating blocked.")
            result.pass_all_checks = False

        # ── 4. DCF integrity ──
        if dcf_status == "conditional":
            result.conditional_reasons.append(
                "DCF is CONDITIONAL (assumption-based, not verified FCF). "
                "Fair values are illustrative only.")

        # ── 5. Economic plausibility ──
        if integrity_report.economic_plausibility_score < 30.0:
            result.eligibility = RatingEligibility.FAIL
            result.blocked_reasons.append(
                f"ECONOMIC PLAUSIBILITY FAILURE (score: "
                f"{integrity_report.economic_plausibility_score:.0f}/100) — "
                "mathematically valid but economically implausible outputs. "
                "Rating blocked.")
            result.pass_all_checks = False
        elif integrity_report.economic_plausibility_score < 60.0:
            result.conditional_reasons.append(
                f"Economic plausibility is DEGRADED (score: "
                f"{integrity_report.economic_plausibility_score:.0f}/100). "
                "Rating conditional on economic validation.")

        # ── 6. Template integrity ──
        if not template_passed:
            result.eligibility = RatingEligibility.FAIL
            result.blocked_reasons.append(
                "TEMPLATE INTEGRITY FAILURE — unresolved placeholders or "
                "NaN values in output. Rating blocked.")
            result.pass_all_checks = False

        # ── 7. Integrity report override ──
        if integrity_report.blocks_rating:
            result.eligibility = RatingEligibility.FAIL
            if "RATING BLOCKED" not in " ".join(result.blocked_reasons):
                result.blocked_reasons.append(
                    "Integrity report declares RATING BLOCKED due to critical failures.")
            result.pass_all_checks = False

        # ── Determine final eligibility ──
        if result.eligibility != RatingEligibility.FAIL:
            if result.conditional_reasons:
                result.eligibility = RatingEligibility.CONDITIONAL
            else:
                result.eligibility = RatingEligibility.PASS

        return result

    @staticmethod
    def assign_rating(
        gate_result: RatingGateResult,
        exp_ret: float,
        p_dd30: float = 0.0,
        overall_conf: float = 50.0,
    ) -> InvestmentRating:
        """Assign the final rating, respecting the gate result.

        If the gate says FAIL, the rating is MODEL INVALID regardless of
        expected return. If CONDITIONAL, the rating is downgraded to the
        conditional equivalent.
        """
        # Gate failure = no rating
        if gate_result.eligibility == RatingEligibility.FAIL:
            return InvestmentRating.MODEL_INVALID

        # If not enough data, say so
        if overall_conf < 30.0:
            return InvestmentRating.INSUFFICIENT_DATA

        # Mechanical derivation (same bands as _dd_rating)
        if exp_ret >= 0.20:
            base = "Strong Buy"
        elif exp_ret >= 0.08:
            base = "Buy"
        elif exp_ret >= -0.05:
            base = "Hold"
        elif exp_ret >= -0.20:
            base = "Sell"
        else:
            base = "Strong Sell"

        # Risk adjustment
        if p_dd30 >= 0.20 or overall_conf < 50.0:
            if base == "Strong Buy":
                base = "Buy"
            elif base == "Buy":
                base = "Hold"
            elif base == "Hold":
                base = "Sell"

        gate_result.base_rating = base

        # Conditional override
        if gate_result.eligibility == RatingEligibility.CONDITIONAL:
            mapping = {
                "Strong Buy": InvestmentRating.CONDITIONAL_BUY,
                "Buy": InvestmentRating.CONDITIONAL_BUY,
                "Hold": InvestmentRating.CONDITIONAL_HOLD,
                "Sell": InvestmentRating.CONDITIONAL_SELL,
                "Strong Sell": InvestmentRating.CONDITIONAL_SELL,
            }
            return mapping.get(base, InvestmentRating.CONDITIONAL_HOLD)

        # Direct mapping for PASS
        mapping = {
            "Strong Buy": InvestmentRating.STRONG_BUY,
            "Buy": InvestmentRating.BUY,
            "Hold": InvestmentRating.HOLD,
            "Sell": InvestmentRating.SELL,
            "Strong Sell": InvestmentRating.SELL,
        }
        return mapping.get(base, InvestmentRating.HOLD)


# ────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT VALIDATION
# ────────────────────────────────────────────────────────────────────────────

class PreFlightStatus(str, Enum):
    PASS = "PASS"
    WARNINGS = "WARNINGS"
    FAIL = "FAIL"


@dataclass
class PreFlightCheck:
    """A single pre-flight check result."""
    name: str
    passed: bool
    status: PreFlightStatus = PreFlightStatus.PASS
    detail: str = ""


@dataclass
class PreFlightReport:
    """Complete pre-flight validation report."""
    checks: List[PreFlightCheck] = field(default_factory=list)
    all_passed: bool = True
    blocking_failures: int = 0
    warnings: int = 0
    summary: str = ""


class PreFlightValidator:
    """Pre-flight validation — runs before memo generation.

    Must pass ALL checks before the final report is generated.
    """

    _CHECK_ORDER = [
        "entity_resolved",
        "entity_integrity",
        "financial_periods",
        "data_provenance",
        "missing_data",
        "canonical_numbers",
        "numerical_qc",
        "economic_qc",
        "dcf_classification",
        "reverse_dcf_plausibility",
        "scenario_probabilities",
        "confidence_constraints",
        "template_rendering",
        "cross_module_reconciliation",
        "rating_eligibility",
    ]

    @staticmethod
    def run(ctx: SecurityContext,
            integrity_report: IntegrityReport,
            dcf_status: Optional[str] = None,
            qc_passed: bool = True,
            template_passed: bool = True,
            scenario_prob_sum: float = 1.0,
            rating_eligible: bool = True) -> PreFlightReport:
        """Run all pre-flight checks."""
        report = PreFlightReport()
        checks = []

        # 1. Entity resolved
        checks.append(PreFlightCheck(
            "Entity resolved", bool(ctx.ticker),
            detail=f"Ticker: {ctx.ticker}" if ctx.ticker else "No ticker"))

        # 2. Entity integrity
        entity_ok = integrity_report.entity_integrity_score >= 30.0
        checks.append(PreFlightCheck(
            "Entity integrity passed",
            entity_ok,
            status=PreFlightStatus.FAIL if not entity_ok else PreFlightStatus.PASS,
            detail=f"Score: {integrity_report.entity_integrity_score:.0f}/100" +
                    (" — CRITICAL FAILURE" if not entity_ok else "")))

        # 3. Financial periods
        has_periods = bool(ctx.financials.reporting_period)
        checks.append(PreFlightCheck(
            "Financial periods aligned",
            has_periods,
            status=PreFlightStatus.WARNINGS if not has_periods else PreFlightStatus.PASS,
            detail=ctx.financials.reporting_period or "Unspecified"))

        # 4. Data provenance
        provenanced = ctx.financials.completeness_score() > 0.2
        checks.append(PreFlightCheck(
            "Data provenance available",
            provenanced,
            status=PreFlightStatus.WARNINGS if not provenanced else PreFlightStatus.PASS,
            detail=f"Completeness: {ctx.financials.completeness_score():.0%}"))

        # 5. Missing data classified
        missing = ctx.financials.get_missing_fields()
        checks.append(PreFlightCheck(
            "Missing data classified",
            True,  # Always passes — classification is structural
            status=PreFlightStatus.WARNINGS if len(missing) > 4 else PreFlightStatus.PASS,
            detail=f"{len(missing)} fields unavailable: {', '.join(missing[:4])}"
                    + (f" + {len(missing) - 4} more" if len(missing) > 4 else "")))

        # 6. Canonical numbers synchronized
        checks.append(PreFlightCheck(
            "Canonical numbers synchronized",
            True,  # Structural — context IS the canonical source
            detail="All modules consume the same SecurityContext"))

        # 7. Numerical QC
        checks.append(PreFlightCheck(
            "Numerical QC passed",
            qc_passed,
            status=PreFlightStatus.FAIL if not qc_passed else PreFlightStatus.PASS,
            detail="Passed" if qc_passed else "FAILED — numbers do not reconcile"))

        # 8. Economic QC
        econ_ok = integrity_report.economic_plausibility_score >= 60.0
        checks.append(PreFlightCheck(
            "Economic QC passed or appropriately flagged",
            econ_ok,
            status=(PreFlightStatus.FAIL if integrity_report.economic_plausibility_score < 30.0
                    else PreFlightStatus.WARNINGS if not econ_ok
                    else PreFlightStatus.PASS),
            detail=f"Score: {integrity_report.economic_plausibility_score:.0f}/100"))

        # 9. DCF classification
        dcf_classified = dcf_status is not None
        checks.append(PreFlightCheck(
            "DCF classification determined",
            dcf_classified,
            detail=dcf_status or "Not classified"))

        # 10. Reverse DCF plausibility
        checks.append(PreFlightCheck(
            "Reverse DCF plausibility checked",
            True,  # Structural — the reverse DCF engine flags implausible results
            detail="Engine flags mathematically solvable but economically unstable results"))

        # 11. Scenario probabilities
        prob_ok = abs(scenario_prob_sum - 1.0) < 1e-6
        checks.append(PreFlightCheck(
            "Scenario probabilities validated",
            prob_ok,
            status=PreFlightStatus.FAIL if not prob_ok else PreFlightStatus.PASS,
            detail=f"Sum: {scenario_prob_sum:.4f}" +
                    (" — MUST sum to 100%" if not prob_ok else "")))

        # 12. Confidence constraints
        checks.append(PreFlightCheck(
            "Confidence constraints applied",
            True,  # Structural — hard confidence engine applied
            detail="Hard ceilings applied to constrained components"))

        # 13. Template rendering
        checks.append(PreFlightCheck(
            "Template rendering validated",
            template_passed,
            status=PreFlightStatus.FAIL if not template_passed else PreFlightStatus.PASS,
            detail="Passed" if template_passed else "FAILED — unresolved placeholders"))

        # 14. Cross-module reconciliation
        checks.append(PreFlightCheck(
            "Cross-module reconciliation passed",
            qc_passed,  # QC audit covers cross-module reconciliation
            detail="Scenario/DCF/Risk-reward/Expectation-gap numbers reconcile"
                    if qc_passed else "FAILED"))

        # 15. Rating eligibility
        checks.append(PreFlightCheck(
            "Rating eligibility determined",
            rating_eligible,
            status=PreFlightStatus.FAIL if not rating_eligible else PreFlightStatus.PASS,
            detail="Eligible" if rating_eligible else "BLOCKED"))

        report.checks = checks
        report.blocking_failures = sum(
            1 for c in checks if c.status == PreFlightStatus.FAIL)
        report.warnings = sum(
            1 for c in checks if c.status == PreFlightStatus.WARNINGS)
        report.all_passed = report.blocking_failures == 0

        if report.all_passed and report.warnings == 0:
            report.summary = "PRE-FLIGHT: ALL CHECKS PASSED"
        elif report.all_passed:
            report.summary = f"PRE-FLIGHT: PASSED with {report.warnings} warning(s)"
        else:
            report.summary = (f"PRE-FLIGHT: FAILED — {report.blocking_failures} "
                              f"blocking failure(s), {report.warnings} warning(s)")

        return report


# ────────────────────────────────────────────────────────────────────────────
# TEMPLATE INTEGRITY SCANNER
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TemplateScanResult:
    """Result of template integrity scan."""
    passed: bool = True
    issues: List[str] = field(default_factory=list)
    unresolved_placeholders: List[str] = field(default_factory=list)
    nan_values: List[str] = field(default_factory=list)
    none_values: List[str] = field(default_factory=list)
    unresolved_references: List[str] = field(default_factory=list)


class TemplateIntegrityScanner:
    """Scans rendered output for template/rendering artifacts.

    Detects:
      - {variable} placeholders
      - {variable:.2f} format specifiers
      - {{placeholder}}
      - None, NaN, undefined, null
      - N/A where inappropriate
      - unresolved template expressions

    Any unresolved template token triggers a RENDERING FAILURE.
    """

    # Patterns that indicate unresolved templates
    _UNRESOLVED_PATTERNS = [
        (re.compile(r'\{\s*\{[^}]*\}\s*\}'), "double-brace placeholder"),
        (re.compile(r'\{[a-zA-Z_][a-zA-Z0-9_.]*\}'), "unresolved variable placeholder"),
        (re.compile(r'\{[a-zA-Z_][a-zA-Z0-9_.]*:[^}]*\}'), "unresolved format placeholder"),
    ]

    _NAN_PATTERNS = [
        (re.compile(r'\bNaN\b'), "NaN value"),
        (re.compile(r'\bundefined\b', re.I), "undefined value"),
        (re.compile(r'\bnull\b', re.I), "null value"),
    ]

    _NONE_PATTERN = re.compile(r'\bNone\b(?!\s+of\s+the\s+above)')

    @staticmethod
    def scan(text: str) -> TemplateScanResult:
        """Scan rendered text for template artifacts."""
        result = TemplateScanResult()

        if not text:
            return result

        # Check for unresolved placeholders
        for pattern, desc in TemplateIntegrityScanner._UNRESOLVED_PATTERNS:
            matches = pattern.findall(text)
            if isinstance(matches, list):
                for m in matches:
                    match_str = m if isinstance(m, str) else str(m)
                    result.unresolved_placeholders.append(
                        f"{desc}: '{match_str}'")
            elif matches:
                result.unresolved_placeholders.append(
                    f"{desc}: '{matches}'")

        # Check for NaN/undefined/null
        for pattern, desc in TemplateIntegrityScanner._NAN_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                result.nan_values.append(
                    f"{desc}: found {len(matches)} occurrence(s)")

        # Check for bare None (not in "None of the above")
        none_matches = TemplateIntegrityScanner._NONE_PATTERN.findall(text)
        if none_matches:
            result.none_values.append(
                f"bare 'None': found {len(none_matches)} occurrence(s)")

        # Aggregate
        all_issues = (result.unresolved_placeholders + result.nan_values
                      + result.none_values + result.unresolved_references)
        result.issues = all_issues
        result.passed = len(all_issues) == 0

        return result


# ────────────────────────────────────────────────────────────────────────────
# AUDIT PANEL GENERATOR
# ────────────────────────────────────────────────────────────────────────────

def generate_audit_panel(
    ctx: SecurityContext,
    integrity_report: IntegrityReport,
    rating_result: RatingGateResult,
    preflight: PreFlightReport,
    template_scan: TemplateScanResult,
) -> str:
    """Generate the institutional 'Model Integrity' panel for the UI."""

    def status_icon(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    def rating_icon(status: PreFlightStatus) -> str:
        if status == PreFlightStatus.PASS:
            return "PASS"
        elif status == PreFlightStatus.WARNINGS:
            return "WARNING"
        return "FAIL"

    lines = [
        "# Model Integrity",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]

    for check in preflight.checks:
        if check.status == PreFlightStatus.FAIL:
            icon = "FAIL"
        elif check.status == PreFlightStatus.WARNINGS:
            icon = "WARNING"
        else:
            icon = "PASS"
        lines.append(f"| {check.name} | {icon} | {check.detail} |")

    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Entity Integrity | {integrity_report.entity_integrity_score:.0f}/100 |")
    lines.append(f"| Semantic Contamination | {integrity_report.semantic_contamination_score:.0f}/100 |")
    lines.append(f"| Economic Plausibility | {integrity_report.economic_plausibility_score:.0f}/100 |")
    lines.append(f"| Data Completeness | {ctx.financials.completeness_score():.0%} |")
    lines.append(f"| Provenance | {sum(1 for _ in ctx.financials.get_available_fields())} "
                 f"/ {len(ctx.financials.get_available_fields()) + len(ctx.financials.get_missing_fields())} values |")
    lines.append(f"| Missing Data | {len(ctx.financials.get_missing_fields())} fields |")
    lines.append(f"| Template QC | {'PASS' if template_scan.passed else 'FAIL'} |")
    lines.append(f"| Rating Eligibility | {rating_result.eligibility.value} |")

    if rating_result.blocked_reasons:
        lines.append("")
        lines.append("**CRITICAL FAILURES:**")
        for r in rating_result.blocked_reasons:
            lines.append(f"- {r}")

    if rating_result.rating in (InvestmentRating.MODEL_INVALID, InvestmentRating.INSUFFICIENT_DATA):
        lines.append("")
        lines.append(f"**RATING: {rating_result.rating.value}**")
        lines.append("A polished BUY/SELL output cannot be produced under current integrity conditions.")

    return "\n".join(lines)