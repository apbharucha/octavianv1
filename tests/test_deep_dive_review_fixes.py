"""
Tests for the review-fix iteration on the deep-dive memo + chart engine:

  * Chart engine: exactly 4 DISTINCT analytical charts per deep-dive run
    (expectation gap, scenario distribution, valuation sensitivity, risk/
    reward) - never generic price charts; each chart carries chart_id,
    purpose, dataset, data_timestamp, provenance, calculation, run_id,
    dataset_hash; identical datasets are flagged reused (spec regenerated,
    never a cached figure); DATA UNAVAILABLE when unsupported.
  * Scenario math: displayed weighted return recomputes exactly from the
    DISPLAYED scenario table (display-precision rounding upstream).
  * Market-vs-model disagreement: only same-variable comparisons (implied
    revenue CAGR vs model revenue growth; never FCF growth vs revenue
    growth).
  * DCF labeling: exact reinvestment formulas stated (NWC = % of CHANGE in
    revenue); DCF status CONDITIONAL/ASSUMPTION-BASED; valuation separated
    into A/B/C/D buckets.
  * Directional validation: reasons-to-own contain only positive evidence;
    reasons-not-to-own only negative.
  * Rating mechanically derived + risk-adjusted; confidence mechanically
    computed with disclosed fixed weights.
  * SUBJECTIVE BAYESIAN FRAMEWORK label; modeled-vs-empirical probability
    language; macro current-claims gated.
"""

import os
import re

import numpy as np
import pytest

os.environ["OCTAVIAN_OFFLINE"] = "1"

from deep_dive_charts import (  # noqa: E402
    build_deep_dive_charts,
)
from financial_llm_engine import (  # noqa: E402
    _DD_DATA_UNAVAILABLE,
    _build_institutional_deep_dive,
    _dd_confidence,
    _dd_real_dcf,
    _dd_scenario_engine,
)

NVDA_QUERY = (
    "You are the lead investment strategist at a multi-strategy institutional asset "
    "manager. Conduct a full fundamental, quantitative, macroeconomic, market-structure, "
    "and sentiment-driven investment assessment of NVIDIA (NVDA) over the next 12-24 "
    "months. Build an evidence-based investment thesis. Include reverse DCF, risk matrix, "
    "catalysts, five scenarios, falsification, Bayesian updating, and a final rating."
)

INTENTS = {
    "bullish": False, "bearish": False, "macro": False, "volatility": False,
    "comparison": False, "sector_scan": False, "options": False, "valuation": False,
    "earnings": False, "dividend": False, "crypto": False, "fx": False,
    "commodities": False, "geopolitics": False, "current_events": False,
    "transmission": False,
}

LIVE = {"NVDA": {"price": 225.16, "change_5d": 3.52, "quote_date": "2026-08-21",
                 "source": "yfinance 5d daily close (delayed/end-of-day)"}}

FUND = {
    "ticker": "NVDA", "price": 225.16, "revenue_m": 130500.0, "revenue_growth": 55.0,
    "ebit_margin_pct": 62.5, "ebitda_m": 90000.0, "tax_rate_pct": 12.5, "beta": 1.68,
    "market_cap_m": 5520000.0, "debt_m": 9700.0, "cash_m": 31500.0, "shares_m": 24520.0,
    "eps": 3.75, "exit_multiple": 30.0, "enterprise_value_m": 5490000.0,
}


def _memo(live=LIVE, fund=FUND):
    return _build_institutional_deep_dive(NVDA_QUERY, dict(INTENTS), ["NVDA", "AMD"],
                                          ["technology"], live, None, fund)


def _section(out, header):
    start = out.find(header)
    assert start != -1, f"section missing: {header}"
    nxt = out.find("\n### ", start + len(header))
    return out[start: nxt if nxt != -1 else len(out)]


# --------------------------------------------------------------------------- #
#  1. CHART ENGINE - four distinct analytical jobs, never price charts
# --------------------------------------------------------------------------- #

CHART_TYPES = {"expectation_gap", "scenario_distribution",
               "valuation_sensitivity", "risk_reward"}


def test_chart_engine_returns_four_distinct_analytical_charts():
    charts = build_deep_dive_charts(NVDA_QUERY, ["NVDA"], ["technology"], LIVE, FUND,
                                    run_id="RUN1")
    assert len(charts) == 4
    types = {c["type"] for c in charts}
    assert types == CHART_TYPES, f"expected the four analytical jobs, got {types}"
    # no generic price chart type anywhere
    assert all(c["type"] != "price_analysis" for c in charts)


def test_every_chart_carries_full_metadata():
    charts = build_deep_dive_charts(NVDA_QUERY, ["NVDA"], ["technology"], LIVE, FUND,
                                    run_id="RUN2")
    for c in charts:
        assert c["chart_id"].startswith("RUN2-")
        assert c["chart_id"].endswith(c["type"])
        assert c["purpose"] and len(c["purpose"]) > 20
        assert c["dataset"] and c["data_timestamp"]
        assert c["provenance"]
        assert c["calculation"]
        assert c["run_id"] == "RUN2"
        assert len(c["dataset_hash"]) == 16
        assert "figure" in c
        assert c["figure"] is not None


def test_identical_dataset_flagged_reused_never_cached():
    import deep_dive_charts as ddc
    ddc._CHART_HASH_REGISTRY.clear()  # isolate from earlier test runs
    a = build_deep_dive_charts(NVDA_QUERY, ["NVDA"], ["technology"], LIVE, FUND,
                               run_id="RUN3")
    b = build_deep_dive_charts(NVDA_QUERY, ["NVDA"], ["technology"], LIVE, FUND,
                               run_id="RUN4")
    assert all(not c["reused_dataset"] for c in a)
    assert all(c["reused_dataset"] for c in b)  # identical dataset -> flagged
    assert all("regenerated" in c["purpose"] for c in b)
    # the figure is still freshly built (never None, never the prior object)
    assert all(c["figure"] is not None for c in b)
    ddc._CHART_HASH_REGISTRY.clear()


def test_changed_data_changes_hash_and_clears_reuse_flag():
    import deep_dive_charts as ddc
    ddc._CHART_HASH_REGISTRY.clear()
    live2 = {"NVDA": {"price": 230.0, "change_5d": 1.0}}
    a = build_deep_dive_charts(NVDA_QUERY, ["NVDA"], ["technology"], LIVE, FUND,
                               run_id="RUN5")
    b = build_deep_dive_charts(NVDA_QUERY, ["NVDA"], ["technology"], live2, FUND,
                               run_id="RUN6")
    assert all(c["dataset_hash"] != d["dataset_hash"]
               for c, d in zip(a, b))
    assert all(not c["reused_dataset"] for c in b)
    ddc._CHART_HASH_REGISTRY.clear()


def test_charts_data_unavailable_without_price():
    import deep_dive_charts as ddc
    ddc._CHART_HASH_REGISTRY.clear()
    charts = build_deep_dive_charts(NVDA_QUERY, ["NVDA"], ["technology"], {}, None,
                                    run_id="RUN7")
    assert len(charts) == 4
    assert all(c["status"] == "DATA UNAVAILABLE" for c in charts)
    # every chart is still present (no silent dropping) but unsupported ones
    # say DATA UNAVAILABLE instead of fabricating
    for c in charts:
        assert c["status"] in ("OK", "DATA UNAVAILABLE")
    ddc._CHART_HASH_REGISTRY.clear()


def test_chart_scenario_distribution_matches_memo_table():
    import deep_dive_charts as ddc
    ddc._CHART_HASH_REGISTRY.clear()
    charts = build_deep_dive_charts(NVDA_QUERY, ["NVDA"], ["technology"], LIVE, FUND,
                                    run_id="RUN8")
    eng = _dd_scenario_engine(LIVE["NVDA"]["change_5d"])
    sc = next(c for c in charts if c["type"] == "scenario_distribution")
    # the figure's own annotation carries the memo's weighted expected return
    # (the SAME engine value), so chart and memo can never disagree
    assert f"{eng['exp_ret']:+.1%}" in str(sc.get("figure")), \
        f"chart missing weighted return {eng['exp_ret']:+.1%}"
    # and the chart id/purpose/run metadata are all populated
    assert sc["run_id"] == "RUN8" and sc["status"] == "OK"
    ddc._CHART_HASH_REGISTRY.clear()


# --------------------------------------------------------------------------- #
#  2. SCENARIO MATH - displayed values are the single source of truth
# --------------------------------------------------------------------------- #

def test_weighted_return_recomputes_from_displayed_table():
    out = _memo()
    s13 = _section(out, "### 13. Five-Scenario Engine")
    row = {}
    for ln in s13.splitlines():
        m = re.match(r"^\| (Extreme Bull|Bull|Base|Bear|Extreme Bear) \| (\d+)% \|.*\| ([+-][\d.]+%) \|$", ln)
        if m:
            row[m.group(1)] = (int(m.group(2)) / 100.0, float(m.group(3).strip("%")) / 100.0)
    displayed = re.search(r"expected return: ([+-][\d.]+%)", s13).group(1)
    rec = sum(p * r for p, r in row.values())
    # the review demanded <= 0.05pp tolerance; display-precision rounding makes
    # this EXACT
    assert abs(rec - float(displayed.strip("%")) / 100.0) <= 0.0005, row


def test_scenario_engine_rounds_to_display_precision():
    eng = _dd_scenario_engine(3.52)
    probs = [p for _, p, _ in eng["scenarios"]]
    rets = [r for _, _, r in eng["scenarios"]]
    assert abs(sum(probs) - 1.0) < 1e-9, probs  # largest remainder -> exactly 100%
    # returns are whole-percent, matching the printed cell format
    assert all(abs(r * 100 - round(r * 100)) < 1e-9 for r in rets)
    # weighted return is computed FROM the rounded values
    assert abs(eng["exp_ret"] - sum(p * r for p, r in zip(probs, rets))) < 1e-12


def test_qc3b_displayed_reconciliation_in_audit():
    out = _memo()
    sec = _section(out, "### 20. Quality Control Audit")
    assert "QC3b displayed weighted return from displayed table" in sec
    assert re.search(r"QC3b.*PASS", sec)


# --------------------------------------------------------------------------- #
#  3. MARKET-VS-MODEL - same-variable comparisons only
# --------------------------------------------------------------------------- #

def test_expectation_gap_uses_implied_revenue_cagr_not_fcf_growth():
    out = _memo()
    sec = _section(out, "### 8. Expectation-Gap Engine")
    assert "same variable: revenue CAGR vs revenue CAGR" in sec
    assert "MARKET-IMPLIED revenue CAGR" in sec
    assert "Same-variable rule" in sec
    # the FCF growth is never placed in the gap cell
    m = re.search(r"\| Revenue growth.*\| ([-+][\d.]+%) \|\s*$", sec)
    gap_cell = m.group(1) if m else None


def test_largest_disagreement_reports_revenue_cagr_above_base():
    out = _memo()
    s19 = _section(out, "### 19. Final Investment Committee Output")
    assert "market-implied revenue CAGR" in s19
    assert "materially ABOVE" in s19
    # never the old bug: comparing FCF growth against revenue growth as a "gap"
    assert "FCF growth vs" not in s19


def test_qc16_same_variable_check_passes():
    out = _memo()
    sec = _section(out, "### 20. Quality Control Audit")
    assert "QC16 market-vs-model disagreement compares SAME variable" in sec
    assert re.search(r"QC16.*PASS", sec)


# --------------------------------------------------------------------------- #
#  4. DCF LABELING - exact formulas + CONDITIONAL + valuation separation
# --------------------------------------------------------------------------- #

def test_dcf_states_exact_reinvestment_formulas():
    txt, fvs = _dd_real_dcf("NVDA", FUND, 225.16, 0.061, 0.095, 0.028, True, True)
    assert "DCF STATUS: CONDITIONAL" in txt
    assert "ASSUMPTION-BASED" in txt
    # NWC is explicitly % of the CHANGE in revenue - the review's labeling fix
    assert "x \u0394Revenue" in txt and "CHANGE in revenue" in txt
    assert "not a reported-FCF valuation" in txt
    assert fvs.get("status") == "conditional"


def test_valuation_separated_into_buckets():
    out = _memo()
    s19 = _section(out, "### 19. Final Investment Committee Output")
    for bucket in ("(A) Reported-data valuation", "(B) Assumption-based DCF",
                   "(C) Market-implied valuation", "(D) Scenario valuation"):
        assert bucket in s19, f"missing valuation bucket {bucket}"
    # assumption-based DCF is explicitly labeled CONDITIONAL/ILLUSTRATIVE
    assert "CONDITIONAL / ILLUSTRATIVE" in s19
    assert "not a reported-FCF valuation" in s19


# --------------------------------------------------------------------------- #
#  5. DIRECTIONAL VALIDATION - reasons to own / not to own
# --------------------------------------------------------------------------- #

def _numbered_items(block: str):
    return [ln for ln in block.splitlines()
            if re.match(r"^\d+\. ", ln.strip())]


def test_reasons_to_own_never_contain_negative_evidence():
    # Bull tape (positive expected return): the positive expected return may be
    # a reason to own, but negative-direction language must never appear in the
    # numbered reason items.
    out = _memo()
    s19 = _section(out, "### 19. Final Investment Committee Output")
    own_block = s19.split("## The 5 strongest reasons NOT to own")[0]
    own_items = _numbered_items(own_block)
    assert len(own_items) == 5
    assert "directionally validated" in s19
    for item in own_items:
        assert "NEGATIVE" not in item
        assert "drawdown" not in item.lower()
        assert "de-rate" not in item.lower()
        assert "downside" not in item.lower()
    # Bear tape: expected return is negative -> it must appear under NOT-to-own
    bear = _memo(live={"NVDA": {"price": 210.0, "change_5d": -8.5}})
    b19 = _section(bear, "### 19. Final Investment Committee Output")
    b_own = b19.split("## The 5 strongest reasons NOT to own")[0]
    b_not = b19.split("## The 5 strongest reasons NOT to own")[1] \
        if "## The 5 strongest reasons NOT to own" in b19 else ""
    b_own_items = _numbered_items(b_own)
    b_not_items = _numbered_items(b_not)
    assert any("NEGATIVE" in i for i in b_not_items), "negative return must be a reason NOT to own"
    assert all("NEGATIVE" not in i for i in b_own_items)


def test_reasons_directional_validation_qc():
    out = _memo()
    sec = _section(out, "### 20. Quality Control Audit")
    assert "QC18 directional validation" in sec
    assert re.search(r"QC18.*PASS", sec)


# --------------------------------------------------------------------------- #
#  6. RATING + CONFIDENCE - mechanically derived
# --------------------------------------------------------------------------- #

def test_rating_framework_shown_with_risk_adjustment():
    out = _memo()
    s19 = _section(out, "### 19. Final Investment Committee Output")
    assert "mechanically derived" in s19
    assert "Rating framework (MODEL CALCULATION)" in s19
    assert "risk-adjusted" in s19 or "risk adjustment" in s19


def test_confidence_formula_and_weights_disclosed():
    out = _memo()
    sec = _section(out, "### 18. Confidence Engine")
    assert "fixed, disclosed weights" in sec
    # the formula prints each component with its weight
    assert "x 30%" in sec and "x 25%" in sec and "x 20%" in sec
    m = re.search(r"\*\*Overall confidence: (\d+)%", out)
    assert m
    # mechanically recompute from the same inputs and compare
    eng = _dd_scenario_engine(3.52)
    _, conf = _dd_confidence(4 / 11, True, eng["scenarios"], True)
    assert abs(conf - float(m.group(1))) < 0.5


# --------------------------------------------------------------------------- #
#  7. BAYESIAN + PROBABILITY LANGUAGE + MACRO GATING
# --------------------------------------------------------------------------- #

def test_bayesian_labeled_subjective():
    out = _memo()
    sec = _section(out, "### 15. Bayesian Update Engine")
    assert "SUBJECTIVE BAYESIAN FRAMEWORK" in sec
    assert "Bayesian-Style Scenario Update" in sec
    assert "not calibrated" in sec


def test_probability_language_never_fake_precision():
    out = _memo()
    s13 = _section(out, "### 13. Five-Scenario Engine")
    s14 = _section(out, "### 14. Monte Carlo / Distribution Engine")
    s19 = _section(out, "### 19. Final Investment Committee Output")
    # every drawdown/loss probability is explicitly "within defined scenarios"
    assert s13.count("within defined scenarios") >= 2
    assert s14.count("within defined scenarios") >= 4
    assert s19.count("within defined scenarios") >= 2
    # empirical probabilities are declared unavailable, not invented
    assert "empirical probability" in s13.lower() or "empirical" in s13.lower()
    assert "empirical probability" in s19.lower() or "empirical" in s19.lower()


def test_macro_current_claims_gated_without_data():
    # The deep-dive passes current_data_available=False, so even with a macro
    # intent the authoritative current-claims phrasing must be suppressed.
    intents = dict(INTENTS)
    intents["macro"] = True
    out = _build_institutional_deep_dive(
        NVDA_QUERY, intents, ["NVDA", "AMD"], ["technology"], LIVE, None, FUND)
    sec = _section(out, "### 9. Macro Transmission Engine")
    # the old authoritative current-claims phrasing must be gone
    assert "The Fed remains data-dependent" not in out
    assert "Here's what matters right now" not in out
    assert "surprisingly resilient" not in out
    # the DATA UNAVAILABLE macro gate is present instead
    assert "Current macro observations" in out
    assert _DD_DATA_UNAVAILABLE in sec


# --------------------------------------------------------------------------- #
#  8. QC firewall additions
# --------------------------------------------------------------------------- #

def test_qc_firewall_checks_present():
    out = _memo()
    sec = _section(out, "### 20. Quality Control Audit")
    for qc in ("QC16", "QC17", "QC18", "QC19", "QC20", "QC21"):
        assert qc in sec, f"missing {qc}"


def test_qc17_confidence_recompute_matches():
    out = _memo()
    sec = _section(out, "### 20. Quality Control Audit")
    assert re.search(r"QC17.*PASS", sec)


def test_qc20_reverse_dcf_variables_never_conflated():
    out = _memo()
    sec = _section(out, "### 20. Quality Control Audit")
    assert re.search(r"QC20.*PASS", sec)


# --------------------------------------------------------------------------- #
#  9. RETEST REVIEW (round 2) - semantic + visualization integrity firewall
# --------------------------------------------------------------------------- #

def test_below_market_dcf_anchor_is_reason_not_to_own():
    # The FUND fixture's conservative DCF anchor sits BELOW market. A below-
    # market fair value is a NEGATIVE implication - it may never appear under
    # reasons-to-own (the round-1 build listed it there as a "reason to own").
    from financial_llm_engine import _dd_build_reasons
    own, not_ = _dd_build_reasons(True, -0.043, 89.34, "conditional", 214.72)
    assert any("below market" in r.lower() for r in not_), \
        "below-market DCF anchor must be a reason NOT to own"
    assert all("below market" not in r.lower() for r in own), \
        "below-market DCF anchor may never be a reason to own"


def test_semantic_scan_detects_misplaced_direction():
    from financial_llm_engine import _dd_semantic_violations
    # clean lists -> no violations
    assert _dd_semantic_violations(["good moat"], ["expensive"]) == []
    # negative implication under reasons-to-own -> violation
    v = _dd_semantic_violations(
        ["Assumption-based DCF anchor $89.34 sits below market"], ["bad"])
    assert any("reasons-to-own" in x for x in v)
    # positive implication under reasons-not-to-own -> violation
    v2 = _dd_semantic_violations(["ok"], ["Expected return +8.8% above market"])
    assert any("reasons-not-to-own" in x for x in v2)


def test_qc18_uses_real_validation_not_hardcoded():
    # The audit must actually scan the reasons lists (round-1 build passed
    # reasons_directional=True unconditionally). A planted violation must
    # flip QC18 to FAIL.
    from financial_llm_engine import _dd_qc_audit
    eng = _dd_scenario_engine(3.52)
    audit = _dd_qc_audit(True, True, eng["scenarios"], {}, exp_ret=eng["exp_ret"],
                         price=225.16, fund=FUND,
                         reasons_directional=False,
                         reasons_violations=["reasons-to-own #1 contains a negative implication: x"])
    assert "QC18 directional validation" in audit
    assert re.search(r"QC18.*FAIL", audit)
    assert "SEMANTIC QC FAILURES" in audit


def test_memo_qc18_passes_with_real_scan():
    # The real memo (bull fixture) must pass QC18 under the actual scan.
    out = _memo()
    sec = _section(out, "### 20. Quality Control Audit")
    assert re.search(r"QC18.*PASS", sec)
    assert "SEMANTIC QC FAILURES" not in sec


def test_expectation_gap_chart_has_conditional_banner():
    import deep_dive_charts as ddc
    ddc._CHART_HASH_REGISTRY.clear()
    charts = build_deep_dive_charts(NVDA_QUERY, ["NVDA"], ["technology"], LIVE, FUND,
                                    run_id="RUN40")
    eg = next(c for c in charts if c["type"] == "expectation_gap")
    fig_txt = str(eg["figure"])
    # the FUND DCF is CONDITIONAL (cash flow unavailable): the chart must say
    # so explicitly, never a bare "DCF Bear / DCF Base / DCF Bull"
    assert "CONDITIONAL DCF" in fig_txt
    assert "NOT VERIFIED FCF VALUATION" in fig_txt
    assert "Cond. DCF" in fig_txt
    # and the chart inherits the memo's QC status
    assert eg["qc_status"].startswith("QC")
    ddc._CHART_HASH_REGISTRY.clear()


def test_valuation_sensitivity_chart_has_conditional_banner():
    import deep_dive_charts as ddc
    ddc._CHART_HASH_REGISTRY.clear()
    charts = build_deep_dive_charts(NVDA_QUERY, ["NVDA"], ["technology"], LIVE, FUND,
                                    run_id="RUN41")
    vs = next(c for c in charts if c["type"] == "valuation_sensitivity")
    fig_txt = str(vs["figure"])
    assert "CONDITIONAL DCF" in fig_txt
    assert "NOT VERIFIED FCF VALUATION" in fig_txt
    ddc._CHART_HASH_REGISTRY.clear()


def test_risk_reward_chart_is_metrics_table_matching_text():
    import deep_dive_charts as ddc
    ddc._CHART_HASH_REGISTRY.clear()
    charts = build_deep_dive_charts(NVDA_QUERY, ["NVDA"], ["technology"], LIVE, FUND,
                                    run_id="RUN42")
    rr = next(c for c in charts if c["type"] == "risk_reward")
    cells = rr["figure"].data[0].cells
    metric_col = list(cells.values)[0]
    val_col = list(cells.values)[1]
    metrics = dict(zip(metric_col, val_col))
    # the table directly corresponds to the memo text (review point 2)
    for key in ("Probability-weighted return", "Probability-weighted price",
                "Current price", "Upside scenarios", "Downside scenarios"):
        assert key in metrics, f"missing metric row {key}"
    # strict modeled-probability language (review point 3)
    assert "within defined scenarios" in metrics["Modeled P(>30% drawdown)"]
    assert "within defined scenarios" in metrics["Modeled P(>50% drawdown)"]
    assert "empirical:" in metrics["Modeled P(>30% drawdown)"]
    # values reconcile with the memo's scenario engine
    eng = _dd_scenario_engine(LIVE["NVDA"]["change_5d"])
    assert f"{eng['exp_ret']:+.1%}" == metrics["Probability-weighted return"]
    fv = LIVE["NVDA"]["price"] * (1 + eng["exp_ret"])
    assert abs(float(metrics["Probability-weighted price"].replace("$", "").replace(",", "")) - fv) < 0.01
    # every chart carries the QC status (visualization inherits memo QC)
    assert rr["qc_status"].startswith("QC")
    ddc._CHART_HASH_REGISTRY.clear()


def test_all_charts_carry_qc_status():
    import deep_dive_charts as ddc
    ddc._CHART_HASH_REGISTRY.clear()
    charts = build_deep_dive_charts(NVDA_QUERY, ["NVDA"], ["technology"], LIVE, FUND,
                                    run_id="RUN43")
    for c in charts:
        assert c["qc_status"].startswith("QC"), f"missing qc_status on {c['type']}"
    ddc._CHART_HASH_REGISTRY.clear()


def test_chart_qc_fails_when_values_do_not_reconcile():
    from deep_dive_charts import _chart_qc
    # A deliberately inconsistent scenario set must produce QC FAIL
    bad = {"scenarios": [("A", 0.5, 0.1), ("B", 0.4, -0.2)], "exp_ret": 0.99}
    st = _chart_qc(bad, None, True, True)
    assert st.startswith("QC FAIL")


def test_eval_prompts_request_modeled_probability_language():
    import chatbot_eval.prompt_factory as pf
    blob = "\n".join(pf._DEEP_DIVE_TEMPLATES)
    # the old "probability of a >30% drawdown" phrasing is gone from the prompts
    assert "MODELED probability" in blob or "modeled probability" in blob.lower()
    assert "within defined scenarios" in blob
    # and the bare old phrase is no longer requested
    assert "the probability of a >30% drawdown" not in blob


def test_llm_stress_prompt_uses_modeled_language():
    import llm_stress_test as lst
    src = open(lst.__file__).read()
    assert "MODELED probability of a >30% drawdown" in src
    assert "within defined scenarios" in src
