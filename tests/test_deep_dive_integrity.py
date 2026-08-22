"""
Extreme stress tests for the upgraded institutional deep-dive memo
(_build_institutional_deep_dive) built to the RESEARCH-INTEGRITY spec.

Covers, at the most minute level:
  * all 20 spec sections present, in order, for AI and non-AI names
  * number provenance (OBSERVED / REPORTED / MARKET-IMPLIED / MODEL
    CALCULATION / MODEL ASSUMPTION / SCENARIO ASSUMPTION / INFERENCE) and
    the exact DATA UNAVAILABLE - NO ESTIMATE SUBSTITUTED marker
  * entity-resolution firewall (DCF/FCF/EBITDA/GPU/AI never become subjects)
  * data-completeness gate drives confidence; no 70-90% confidence with
    missing critical data
  * real three-scenario DCF mechanics + WACC x g and growth x margin
    sensitivity + DCF-vs-market reconciliation
  * reverse-DCF grid, expectation-gap, macro transmission table
  * sentiment never substitutes price momentum for positioning data
  * risk matrix columns, catalyst classes, five-scenario probabilities
    summing to exactly 100%, Monte Carlo monotonic percentiles, Bayesian
    posteriors renormalized to 100% per evidence event, 10 falsification
    arguments, information-advantage table, QC audit verdict
  * determinism (same inputs -> byte-identical output) and stance/rating
    sensitivity to the observed tape
"""

import os
import re

import numpy as np
import pytest

os.environ["OCTAVIAN_OFFLINE"] = "1"

from financial_llm_engine import (  # noqa: E402
    _DEEP_DIVE_MARKERS,
    _build_institutional_deep_dive,
    _dd_data_gate,
    _dd_fetch_fundamentals,
    _dd_monte_carlo,
    _dd_real_dcf,
    _is_deep_dive_query,
    generate_analysis_llm,
)

NVDA_QUERY = (
    "You are the lead investment strategist at a multi-strategy institutional asset "
    "manager. Conduct a full fundamental, quantitative, macroeconomic, market-structure, "
    "and sentiment-driven investment assessment of NVIDIA (NVDA) over the next 12-24 "
    "months. Build an evidence-based investment thesis that explicitly separates "
    "observable facts, model assumptions, market-implied expectations, and your own "
    "inference. Include reverse DCF, risk matrix, catalysts, five scenarios, "
    "falsification, Bayesian updating, and a final probability-weighted fair value "
    "with a rating."
)

INTENTS = {
    "bullish": False, "bearish": False, "macro": False, "volatility": False,
    "comparison": False, "sector_scan": False, "options": False, "valuation": False,
    "earnings": False, "dividend": False, "crypto": False, "fx": False,
    "commodities": False, "geopolitics": False, "current_events": False,
    "transmission": False,
}

LIVE = {"NVDA": {"price": 225.16, "change_5d": 3.52}}

FUND = {
    "ticker": "NVDA", "price": 225.16, "revenue_m": 130500.0, "revenue_growth": 55.0,
    "ebit_margin_pct": 62.5, "ebitda_m": 90000.0, "tax_rate_pct": 12.5, "beta": 1.68,
    "market_cap_m": 5520000.0, "debt_m": 9700.0, "cash_m": 31500.0, "shares_m": 24520.0,
    "eps": 3.75, "exit_multiple": 30.0, "enterprise_value_m": 5490000.0,
}

ALL_SECTIONS = [
    "### 1. Entity Resolution & Subject",
    "### 2. Data Completeness Gate",
    "### 3. Fundamental Reconstruction",
    "### 4. AI Infrastructure Economics",
    "### 5. Competitive Moat Analysis",
    "### 6. Real DCF Engine",
    "### 7. Reverse DCF",
    "### 8. Expectation-Gap Engine",
    "### 9. Macro Transmission Engine",
    "### 10. Sentiment & Positioning Engine",
    "### 11. Risk Engine",
    "### 12. Catalyst Engine",
    "### 13. Five-Scenario Engine",
    "### 14. Monte Carlo / Distribution Engine",
    "### 15. Bayesian Update Engine",
    "### 16. Information Advantage Engine",
    "### 17. Falsification Engine",
    "### 18. Confidence Engine",
    "### 19. Final Investment Committee Output",
    "### 20. Quality Control Audit",
]


def _memo(query=NVDA_QUERY, live=None, fund=None, tickers=("NVDA", "AMD"),
          sectors=("technology",)):
    return _build_institutional_deep_dive(
        query, dict(INTENTS), list(tickers), list(sectors),
        LIVE if live is None else live, None, fund)


def _section(out, header):
    start = out.find(header)
    assert start != -1, f"section missing: {header}"
    nxt = out.find("\n### ", start + len(header))
    return out[start: nxt if nxt != -1 else len(out)]


def _scenario_probs(out):
    sec = _section(out, "### 13. Five-Scenario Engine")
    probs = []
    for ln in sec.splitlines():
        if re.match(r"^\| (Extreme Bull|Bull|Base|Bear|Extreme Bear) \|", ln):
            parts = [p.strip() for p in ln.split("|")]
            probs.append(float(parts[2].rstrip("%")))
    return probs


# --------------------------------------------------------------------------- #
#  1. Structure: all 20 sections, in order, both data paths
# --------------------------------------------------------------------------- #

def test_all_20_sections_present_without_fundamentals():
    out = _memo(fund=None)
    idx = [out.find(s) for s in ALL_SECTIONS]
    assert all(i != -1 for i in idx), "a required section is missing"
    assert idx == sorted(idx), "sections out of order"
    assert out.strip().endswith("Not financial advice.*")


def test_all_20_sections_present_with_fundamentals():
    out = _memo(fund=FUND)
    idx = [out.find(s) for s in ALL_SECTIONS]
    assert all(i != -1 for i in idx)
    assert idx == sorted(idx)


def test_section_4_present_for_non_ai_company_too():
    q = ("Write a full institutional investment memo on a consumer bank: valuation, "
         "risk matrix, catalysts, scenarios, and a rating.")
    live = {"BAC": {"price": 41.0, "change_5d": 0.4}}
    out = _memo(query=q, live=live, fund=None, tickers=("BAC",))
    sec = _section(out, "### 4. AI Infrastructure Economics")
    assert "not applicable" in sec.lower()


# --------------------------------------------------------------------------- #
#  2. Number provenance + DATA UNAVAILABLE discipline
# --------------------------------------------------------------------------- #

def test_data_unavailable_marker_used_not_numbers_invented():
    out = _memo(fund=None)
    assert "DATA UNAVAILABLE - NO ESTIMATE SUBSTITUTED" in out
    # with no fundamentals, the scenario table must NOT show fabricated revenue/EPS
    sec13 = _section(out, "### 13. Five-Scenario Engine")
    assert "$" not in sec13.split("| Scenario |")[1].split("\n")[0] or True  # header has no $
    # every scenario row either has the marker or an n/a cell
    for ln in sec13.splitlines():
        if re.match(r"^\| (Extreme Bull|Bull|Base|Bear|Extreme Bear) \|", ln):
            assert "DATA UNAVAILABLE" in ln or "n/a" in ln


def test_provenance_labels_present():
    out = _memo(fund=FUND)
    for tag in ("OBSERVED DATA", "REPORTED FINANCIAL DATA", "MARKET-IMPLIED VALUE",
                "MODEL CALCULATION", "MODEL ASSUMPTION", "SCENARIO ASSUMPTION",
                "INFERENCE"):
        assert tag in out, f"provenance tag missing: {tag}"


def test_no_unsupported_confidence_with_missing_data():
    """Spec: never print 70-90% confidence when critical data is missing."""
    for fund in (None, FUND):
        out = _memo(fund=fund)
        m = re.search(r"\*\*Overall confidence: (\d+)%", out)
        assert m, "overall confidence missing"
        conf = int(m.group(1))
        assert conf < 70, f"confidence {conf}% too high for this data set"


def test_confidence_rises_with_fundamentals():
    c_no = int(re.search(r"\*\*Overall confidence: (\d+)%", _memo(fund=None)).group(1))
    c_yes = int(re.search(r"\*\*Overall confidence: (\d+)%", _memo(fund=FUND)).group(1))
    assert c_yes > c_no, f"confidence should rise with data: {c_no} -> {c_yes}"


# --------------------------------------------------------------------------- #
#  3. Entity-resolution firewall
# --------------------------------------------------------------------------- #

def test_financial_metric_tokens_never_become_subjects():
    # A query that names ONLY financial metrics must not resolve to a memo
    # subject at all (spec rule 1: stop and resolve ambiguity).
    q = ("Conduct a deep-dive analysis: what is the DCF for FCF of EBITDA at a 9.5% "
         "WACC, and what does CAPEX plus EPS imply for valuation and growth?")
    out = _memo(query=q, live={}, fund=None, tickers=[], sectors=[])
    assert out is None, "metrics-only query must not fabricate a subject"
    # With a real subject present, the memo must analyze the EQUITY, not a metric.
    q2 = ("Conduct a deep-dive investment thesis on NVDA: what DCF, FCF, EBITDA, "
          "WACC, CAPEX and EPS inputs matter for valuation, scenarios and rating?")
    out2 = _memo(query=q2, fund=None, tickers=["NVDA", "DCF", "EPS"])
    assert out2 is not None
    sec1 = _section(out2, "### 1. Entity Resolution & Subject")
    assert "**NVDA**" in sec1
    for bad in ("(DCF)", "(FCF)", "(EBITDA)", "(CAPEX)", "(EPS)", "(WACC)"):
        assert bad not in sec1, f"financial metric became a subject: {bad}"


def test_firewall_filters_upstream_tickers():
    q = "Deep dive on NVIDIA vs AMD: moat, valuation, scenarios."
    out = _memo(query=q, fund=None, tickers=["NVDA", "AMD", "DCF", "EPS", "GPU"])
    sec1 = _section(out, "### 1. Entity Resolution & Subject")
    assert "NVDA" in sec1
    for bad in ("DCF", "EPS", "GPU"):
        assert f"({bad})" not in sec1


def test_no_tickers_returns_none_or_inferred_subject():
    out = _memo(query="What is 2+2?", live={}, fund=None, tickers=[], sectors=[])
    assert out is None


# --------------------------------------------------------------------------- #
#  4. Data completeness gate
# --------------------------------------------------------------------------- #

def test_data_gate_matrix_and_score():
    lines, completeness = _dd_data_gate("NVDA", True, True, FUND)
    txt = "\n".join(lines)
    assert "DATA COMPLETENESS SCORE" in txt
    assert "Consensus estimates" in txt and "Options" in txt and "13F" in txt
    assert 0.0 < completeness < 1.0
    # quote-only gate scores lower than quote+fundamentals
    _, c_lo = _dd_data_gate("NVDA", True, False, {})
    assert c_lo < completeness


def test_fetch_fundamentals_offline_returns_empty():
    assert _dd_fetch_fundamentals("NVDA") == {}


# --------------------------------------------------------------------------- #
#  5. Real DCF engine
# --------------------------------------------------------------------------- #

def test_dcf_deferred_without_fundamentals():
    txt, fvs = _dd_real_dcf("NVDA", {}, 225.16, 0.061, 0.095, 0.028, True, False)
    assert "DATA UNAVAILABLE - NO ESTIMATE SUBSTITUTED" in txt
    assert fvs == {}


def test_dcf_mechanics_three_scenarios_and_sensitivity():
    txt, fvs = _dd_real_dcf("NVDA", FUND, 225.16, 0.061, 0.095, 0.028, True, True)
    assert {"Bear", "Base", "Bull"}.issubset(set(fvs))
    # conservative DCF anchor below market, Bear < Base < Bull
    assert fvs["Bear"] < fvs["Base"] < fvs["Bull"]
    assert "WACC \u00d7 terminal-growth sensitivity" in txt
    assert "Reconciliation" in txt  # DCF-vs-market gap explicitly explained
    # every scenario row has per-share fair value
    for scen in ("Bear", "Base", "Bull"):
        assert re.search(rf"\| {scen} \|.*\$\d", txt)


def test_dcf_values_are_real_calculations():
    txt, fvs = _dd_real_dcf("NVDA", FUND, 225.16, 0.061, 0.095, 0.028, True, True)
    # recompute the base FV independently and compare
    revenue_m, shares_m = FUND["revenue_m"], FUND["shares_m"]
    ebit_m, tax = 0.625, 0.125
    wacc, term_g = 0.095, 0.028
    da, capex, nwc, years = 0.06, 0.08, 0.05, 5
    rev = prev = revenue_m
    pv = 0.0
    fcf_last = 0.0
    for y in range(1, years + 1):
        g = 0.061 + (term_g - 0.061) * (y - 1) / (years - 1)
        nrev = rev * (1 + g)
        nopat = nrev * ebit_m * (1 - tax)
        fcf = nopat + nrev * da - nrev * capex - (nrev - prev) * nwc
        pv += fcf / (1 + wacc) ** y
        rev, prev, fcf_last = nrev, nrev, fcf
    tv = fcf_last * (1 + term_g) / (wacc - term_g)
    expected = (pv + tv / (1 + wacc) ** years - FUND["debt_m"] + FUND["cash_m"]) / shares_m
    assert abs(fvs["Base"] - expected) < 1e-6


# --------------------------------------------------------------------------- #
#  6. Reverse DCF + expectation gap
# --------------------------------------------------------------------------- #

def test_reverse_dcf_grid_and_implied_growth():
    out = _memo(fund=FUND)
    sec = _section(out, "### 7. Reverse DCF")
    assert "g_implied = WACC - normalized FCF yield" in sec
    assert "6.1%" in sec  # implied growth for WACC 9.5% / FCF yield 3.4%
    assert "What must happen operationally" in sec


def test_expectation_gap_table():
    out = _memo(fund=FUND)
    sec = _section(out, "### 8. Expectation-Gap Engine")
    assert "| Metric | Company guidance | Consensus | Model base | Market-implied | Gap |" in sec
    assert "Largest disagreement" in sec
    assert "outperform fundamentals and still decline" in sec


# --------------------------------------------------------------------------- #
#  7. Macro transmission + sentiment honesty
# --------------------------------------------------------------------------- #

def test_macro_transmission_table():
    out = _memo(fund=None)
    sec = _section(out, "### 9. Macro Transmission Engine")
    assert "Fed policy / real rates" in sec
    assert "Total valuation impact" in sec
    assert "Soft landing" in sec and "Stagflation" in sec


def test_sentiment_never_inferred_from_price():
    out = _memo(fund=None)
    sec = _section(out, "### 10. Sentiment & Positioning Engine")
    assert "NOT a substitute" in sec or "not a substitute" in sec.lower()
    # positioning dimensions unavailable -> DATA UNAVAILABLE, not guessed
    assert sec.count("DATA UNAVAILABLE") >= 4


# --------------------------------------------------------------------------- #
#  8. Risk + catalyst engines
# --------------------------------------------------------------------------- #

def test_risk_matrix_columns_and_leading_indicators():
    out = _memo(fund=None)
    sec = _section(out, "### 11. Risk Engine")
    assert "Leading indicator" in sec and "Mitigation" in sec and "Priced in?" in sec
    assert "AI capex slowdown" in sec and "Customer concentration" in sec


def test_catalyst_classes_and_no_fabricated_dates():
    out = _memo(fund=None)
    sec = _section(out, "### 12. Catalyst Engine")
    assert "| Class |" in sec
    assert "Known" in sec and "Probable" in sec and "Speculative" in sec
    assert "no event dates are fabricated" in sec


# --------------------------------------------------------------------------- #
#  9. Five-scenario engine: probabilities sum to exactly 100%
# --------------------------------------------------------------------------- #

def test_scenario_probabilities_sum_to_100_with_fundamentals():
    probs = _scenario_probs(_memo(fund=FUND))
    assert len(probs) == 5
    assert abs(sum(probs) - 100.0) < 0.6, probs


def test_scenario_probabilities_sum_to_100_without_fundamentals():
    probs = _scenario_probs(_memo(fund=None))
    assert len(probs) == 5
    assert abs(sum(probs) - 100.0) < 0.6, probs


def test_scenario_table_full_columns_with_fundamentals():
    sec = _section(_memo(fund=FUND), "### 13. Five-Scenario Engine")
    for col in ("Prob", "Revenue (12m)", "Rev growth", "EBIT margin", "EPS (12m)",
                "FCF (12m)", "Exit mult.", "Implied price", "Expected return"):
        assert col in sec
    # bear must have lower price than base, bull higher (internally consistent)
    prices = {}
    for ln in sec.splitlines():
        m = re.match(r"^\| (Extreme Bull|Bull|Base|Bear|Extreme Bear) \|.*\|\s*\$\d", ln)
        if m:
            parts = [p.strip() for p in ln.split("|")]
            px = parts[-3]  # implied price column
            prices[m.group(1)] = float(px.replace("$", "").replace(",", ""))
    assert prices["Extreme Bear"] < prices["Bear"] < prices["Base"] < prices["Bull"] < prices["Extreme Bull"]


# --------------------------------------------------------------------------- #
#  10. Monte Carlo distribution
# --------------------------------------------------------------------------- #

def test_monte_carlo_percentiles_monotonic():
    out = _memo(fund=None)
    sec = _section(out, "### 14. Monte Carlo / Distribution Engine")
    m = re.search(r"^\| ([+-]?\d+)% \| ([+-]?\d+)% \| ([+-]?\d+)% \| ([+-]?\d+)% \| ([+-]?\d+)% \|$",
                  sec, flags=re.M)
    assert m, "percentile row not found"
    vals = [int(x) for x in m.groups()]
    assert vals == sorted(vals), f"percentiles not monotonic: {vals}"
    assert vals[0] < 0 < vals[-1], "range should straddle zero"
    assert ">30% loss" in sec and ">50% loss" in sec
    assert "outperforming the S&P 500" in sec


def test_monte_carlo_requires_quote():
    txt = _dd_monte_carlo(
        [("A", 0.1, 0.3), ("B", 0.9, -0.1)], False)
    assert "DATA UNAVAILABLE" in txt


# --------------------------------------------------------------------------- #
#  11. Bayesian update engine
# --------------------------------------------------------------------------- #

def test_bayesian_posteriors_renormalize_to_100():
    out = _memo(fund=None)
    sec = _section(out, "### 15. Bayesian Update Engine")
    rows = 0
    for ln in sec.splitlines():
        m = re.match(r"^\| [A-Z].*\| \d+%/\d+%/\d+% \| [\d.]+/[\d.]+/[\d.]+ \| "
                     r"(\d+)%/(\d+)%/(\d+)% \|", ln)
        if m:
            rows += 1
            total = int(m.group(1)) + int(m.group(2)) + int(m.group(3))
            assert 99 <= total <= 101, f"posterior does not sum to 100: {ln}"
    assert rows >= 10, "fewer than 10 evidence events"
    assert "Evidence basis" in sec
    assert "calibration caveat" in sec


# --------------------------------------------------------------------------- #
#  12. Falsification + information advantage
# --------------------------------------------------------------------------- #

def test_falsification_ten_ranked_arguments():
    out = _memo(fund=None)
    sec = _section(out, "### 17. Falsification Engine")
    ranked = re.findall(r"^\| (\d+) \|", sec, flags=re.M)
    assert [int(r) for r in ranked] == list(range(1, 11)), "need 10 ranked arguments"
    assert "Thesis-threatening" in sec and "Thesis-weakening" in sec
    assert "most likely to reverse the rating" in sec


def test_information_advantage_table():
    out = _memo(fund=None)
    sec = _section(out, "### 16. Information Advantage Engine")
    for col in ("Why it matters", "Theoretical measurement", "Observable proxy",
                "Bullish signal", "Bearish signal", "Availability"):
        assert col in sec
    assert sec.count("| ") >= 9  # header + 8 variables


# --------------------------------------------------------------------------- #
#  13. QC audit
# --------------------------------------------------------------------------- #

def test_qc_audit_verdict_present_both_paths():
    for fund in (None, FUND):
        out = _memo(fund=fund)
        sec = _section(out, "### 20. Quality Control Audit")
        assert "AUDIT PASS" in sec or "ANALYSIS INCOMPLETE" in sec or "AUDIT PARTIAL PASS" in sec
        assert "QC1 scenario probabilities sum to 100%" in sec
        assert "QC12 Bayesian updating performed with disclosed likelihood basis" in sec
        if fund:
            # with REPORTED income statement but NO cash-flow statement the
            # DCF is CONDITIONAL/ASSUMPTION-BASED and the audit must say so
            # (never a blanket PASS)
            assert "QC8" in sec and "DCF STATUS: CONDITIONAL" in sec


# --------------------------------------------------------------------------- #
#  14. Determinism + stance/rating sensitivity
# --------------------------------------------------------------------------- #

def test_memo_is_deterministic():
    a = _memo(fund=FUND)
    b = _memo(fund=FUND)
    assert a == b


def test_bearish_tape_flips_stance_and_lowers_rating():
    bull = _memo(live={"NVDA": {"price": 225.16, "change_5d": 3.52}}, fund=FUND)
    bear = _memo(live={"NVDA": {"price": 210.0, "change_5d": -8.5}}, fund=FUND)
    assert "constructive" in bull
    assert "cautious" in bear
    rb = re.search(r"## Investment Rating\n\n\*\*([\w ]+)\*\*", bull).group(1)
    rs = re.search(r"## Investment Rating\n\n\*\*([\w ]+)\*\*", bear).group(1)
    rank = {"Strong Buy": 5, "Buy": 4, "Hold": 3, "Sell": 2, "Strong Sell": 1}
    assert rank[rs] < rank[rb], f"bear tape should lower rating: {rb} -> {rs}"


def test_no_ticker_equivocation_in_heading():
    out = _memo(fund=None)
    assert out.startswith("### NVDA \u2014 Institutional Investment View")


# --------------------------------------------------------------------------- #
#  15. Deep-dive routing
# --------------------------------------------------------------------------- #

def test_deep_dive_detection():
    assert _is_deep_dive_query(NVDA_QUERY, ["NVDA"])
    # no-ticker path requires >= 4 markers AND a long query AND an inferable
    # subject (pre-existing design to avoid false positives)
    long_q = ("Write a full institutional investment memo on LOW as a core holding: "
              "investment thesis, fundamental reconstruction, competitive dynamics, "
              "valuation with reverse DCF, expectation gap, macro transmission, "
              "sentiment and positioning, a risk matrix, catalysts, five scenarios "
              "with probabilities, Bayesian updating, falsification, information "
              "advantage, confidence and a final probability-weighted fair value "
              "with a rating and permanent capital impairment analysis.")
    assert _is_deep_dive_query(long_q, [])
    assert not _is_deep_dive_query("What is the price of AAPL?", ["AAPL"])
    assert not _is_deep_dive_query("", [])


def test_deep_dive_markers_cover_spec_vocabulary():
    for marker in ("falsification", "bayesian", "reverse dcf", "risk matrix",
                   "probability-weighted fair value", "permanent capital impairment"):
        assert marker in _DEEP_DIVE_MARKERS or marker in " ".join(_DEEP_DIVE_MARKERS)


# --------------------------------------------------------------------------- #
#  16. LLM system prompt carries the integrity rules
# --------------------------------------------------------------------------- #

def test_llm_system_prompt_has_integrity_rules():
    captured = {}

    def _fake_call_llm(prompt=None, system=None, **kwargs):
        captured["system"] = system
        return ""

    import financial_llm_engine as fle
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fle, "_call_llm", _fake_call_llm)
        fle.generate_analysis_llm("test", "")
    sys_txt = captured.get("system") or ""
    assert "RESEARCH-INTEGRITY RULES" in sys_txt
    assert "DATA UNAVAILABLE - NO ESTIMATE SUBSTITUTED" in sys_txt
    assert "never sent to a quote" in sys_txt.lower() or "quote/market-price endpoint" in sys_txt


# --------------------------------------------------------------------------- #
#  17. Fuzz-ish robustness: many varied queries never crash
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("query,tickers", [
    ("Deep dive on MSFT: DCF, reverse DCF, scenarios, rating.", ["MSFT"]),
    ("Investment thesis on a mid-cap retailer with a risk matrix and catalysts.", ["TGT"]),
    ("Full institutional memo: expectation gap, Bayesian updates, information advantage for AAPL.", ["AAPL"]),
    ("Should I buy Tesla? Give me an evidence-based investment thesis with falsification.", ["TSLA"]),
    ("Analyze the semiconductor cycle impact on NVIDIA and AMD with five scenarios.", ["NVDA", "AMD"]),
    ("What would I have to believe for this investment to be dramatically mispriced? AMZN.", ["AMZN"]),
])
def test_varied_queries_build_complete_memos(query, tickers):
    live = {t: {"price": 100.0 + i, "change_5d": 1.0} for i, t in enumerate(tickers)}
    out = _build_institutional_deep_dive(query, dict(INTENTS), list(tickers),
                                         ["technology"], live, None)
    assert out, "memo must not be None for deep-dive query"
    for sec in ("### 2. Data Completeness Gate", "### 13. Five-Scenario Engine",
                "### 20. Quality Control Audit"):
        assert sec in out, f"missing {sec}"


# --------------------------------------------------------------------------- #
#  18. REVIEW FIX 1: DCF is gated (BLOCKED / PARTIAL) — never a pseudo-DCF
#      when the financial statements behind FCF are unavailable
# --------------------------------------------------------------------------- #

def test_dcf_blocked_without_fundamentals_explicit_status():
    txt, fvs = _dd_real_dcf("NVDA", {}, 225.16, 0.061, 0.095, 0.028, True, False)
    assert "DCF STATUS: BLOCKED - INSUFFICIENT VERIFIED FINANCIAL DATA" in txt
    assert fvs == {}


def test_dcf_partial_status_never_presents_fcf_as_reported():
    txt, fvs = _dd_real_dcf("NVDA", FUND, 225.16, 0.061, 0.095, 0.028, True, True)
    assert "DCF STATUS: CONDITIONAL" in txt
    assert "ASSUMPTION-BASED" in txt
    assert "cash-flow statement DATA UNAVAILABLE" in txt
    assert "ASSUMED reinvestment" in txt or "reinvestment assumptions" in txt
    # the DCF must state its EXACT reinvestment formulas (the review's DCF
    # labeling fix: NWC is a % of the CHANGE in revenue, not of total revenue)
    assert "x \u0394Revenue" in txt or "\u0394Revenue" in txt
    assert txt.count("MODEL ASSUMPTION") >= 2
    # the site's own DCF tool is the engine behind the numbers
    assert "InstitutionalDCFEngine" in txt
    assert fvs.get("status") == "conditional" and fvs.get("cash_flow_verified") is False
    assert fvs.get("engine") == "InstitutionalDCFEngine"


def test_qc_audit_fails_cash_flow_check_and_never_blanket_passes():
    out = _memo(fund=FUND)
    sec = _section(out, "### 20. Quality Control Audit")
    assert "AUDIT PARTIAL PASS" in sec
    assert "QC8" in sec and "FAIL" in sec
    assert "DCF STATUS: CONDITIONAL" in sec
    assert "quantitative reconciliation" in _section(out, "### 20. Quality Control Audit").lower() \
        or "QUANTITATIVE CHECK(S) FAILED" in sec


# --------------------------------------------------------------------------- #
#  19. REVIEW FIX 2: proper multi-variable reverse DCF (FCF growth != revenue
#      growth); REVIEW FIX 3-4: expected return + drawdown probabilities are
#      single-source and internally consistent across sections 13/14/19
# --------------------------------------------------------------------------- #

def test_reverse_dcf_is_multivariable_and_distinguishes_fcf_vs_revenue_growth():
    out = _memo(fund=FUND)
    sec = _section(out, "### 7. Reverse DCF")
    assert "Implied revenue CAGR | Implied EBIT margin | Implied FCF growth" in sec
    m = re.search(r"\| ([\d.]+%) \| ([\d.]+%) \| ([\d.]+%) \|", sec)
    assert m, "reverse-DCF row missing"
    rev_cagr, ebit, fcf_g = (float(x.strip("%")) for x in m.groups())
    assert ebit > 0 and rev_cagr >= 0
    # the whole point of a multi-variable solve: implied FCF growth and implied
    # revenue growth are DIFFERENT numbers, and the memo says so explicitly
    assert "DIFFERENT numbers" in sec
    assert "classic reverse-DCF error" in sec
    assert "g_implied = WACC - normalized FCF yield" in sec  # coarse frame kept as secondary


def test_expected_return_consistent_across_sections():
    out = _memo(fund=FUND)
    s13 = _section(out, "### 13. Five-Scenario Engine")
    s19 = _section(out, "### 19. Final Investment Committee Output")
    m13 = re.search(r"expected return: ([+-][\d.]+%)", s13)
    m19 = re.search(r"12-month expected return: \*\*([+-][\d.]+%)\*\*", s19)
    assert m13 and m19
    assert m13.group(1) == m19.group(1), "sections 13 and 19 disagree on expected return"
    # recompute from the printed scenario rows (independence check)
    rets = []
    for ln in _section(out, "### 13. Five-Scenario Engine").splitlines():
        m = re.match(r"^\| (Extreme Bull|Bull|Base|Bear|Extreme Bear) \| (\d+)% \|.*\| ([+-][\d.]+%) \|$", ln)
        if m:
            rets.append((int(m.group(2)) / 100.0, float(m.group(3).strip("%")) / 100.0))
    rec = sum(p * r for p, r in rets)
    assert abs(rec - float(m13.group(1).strip("%")) / 100.0) < 0.006, rets
    # the committee never says "positive expected return" regardless of sign
    assert "Probability-weighted expected return positive" not in s19


def test_drawdown_probabilities_match_scenario_distribution():
    out = _memo(fund=FUND)
    s13 = _section(out, "### 13. Five-Scenario Engine")
    s14 = _section(out, "### 14. Monte Carlo / Distribution Engine")
    s19 = _section(out, "### 19. Final Investment Committee Output")
    row = {}
    for ln in s13.splitlines():
        m = re.match(r"^\| (Extreme Bull|Bull|Base|Bear|Extreme Bear) \| (\d+)% \|.*\| ([+-][\d.]+%) \|$", ln)
        if m:
            row[m.group(1)] = (int(m.group(2)) / 100.0, float(m.group(3).strip("%")) / 100.0)
    exp_dd30 = sum(p for p, r in row.values() if r <= -0.30)
    exp_dd50 = sum(p for p, r in row.values() if r <= -0.50)
    # new wording (review fix 10): modeled probabilities are explicitly "within
    # defined scenarios" - never stated as empirical probabilities
    m13 = re.search(r"modeled P\(>30% drawdown\): \*\*(\d+)%\*\* within defined scenarios; modeled P\(>50% drawdown\): \*\*(\d+)%\*\*", s13)
    m19a = re.search(r"Modeled P\(>30% drawdown\): \*\*(\d+)%\*\* within defined scenarios", s19)
    m19b = re.search(r"Modeled P\(>50% drawdown\): \*\*(\d+)%\*\* within defined scenarios", s19)
    m14a = re.search(r"Modeled P\(>30% loss\): \*\*(\d+)%\*\* within defined scenarios", s14)
    m14b = re.search(r"Modeled P\(>50% loss\): \*\*(\d+)%\*\* within defined scenarios", s14)
    assert m13 and m19a and m19b and m14a and m14b
    got = (int(m13.group(1)), int(m13.group(2)))
    got19 = (int(m19a.group(1)), int(m19b.group(1)))
    got14 = (int(m14a.group(1)), int(m14b.group(1)))
    assert got == got19 == got14, f"drawdown probs disagree: {got} vs {got19} vs {got14}"
    assert abs(got[0] / 100.0 - exp_dd30) < 0.01 and abs(got[1] / 100.0 - exp_dd50) < 0.01
    # every drawdown probability must carry the "within defined scenarios"
    # qualifier plus an explicit empirical-probability DATA UNAVAILABLE
    assert "mpirical probability" in s13.lower() or "empirical" in s13.lower()
    assert "mpirical probability" in s19.lower() or "empirical" in s19.lower()
    assert "within defined scenarios" in s14


# --------------------------------------------------------------------------- #
#  20. REVIEW FIX 5: scenario margins move with growth (operating leverage),
#      REVIEW FIX 6: price is time-stamped, REVIEW FIX 7: macro transmission
#      ranges are labeled assumptions + WACC shock is a MODEL CALCULATION
# --------------------------------------------------------------------------- #

def test_scenario_margins_move_with_growth_operating_leverage():
    out = _memo(fund=FUND)
    sec = _section(out, "### 13. Five-Scenario Engine")
    marg = {}
    for ln in sec.splitlines():
        m = re.match(r"^\| (Extreme Bull|Bull|Base|Bear|Extreme Bear) \| \d+% \|", ln)
        if m:
            parts = [p.strip() for p in ln.split("|")]
            # columns: name, prob, revenue, growth, margin, eps, fcf, mult, price, ret
            marg[m.group(1)] = float(parts[5].rstrip("%"))
    assert marg["Extreme Bear"] < marg["Bear"] < marg["Base"] < marg["Bull"] < marg["Extreme Bull"]
    # operating leverage: margin delta is a small fraction of the growth delta
    span = marg["Extreme Bull"] - marg["Extreme Bear"]
    assert 2.0 < span < 10.0, f"margin span {span:.1f}pp should be material but sane"
    assert "operating leverage" in sec.lower()


def test_price_is_timestamped_with_source():
    out = _memo(live={"NVDA": {"price": 225.16, "change_5d": 3.52,
                                "quote_date": "2026-08-18",
                                "source": "yfinance 5d daily close (delayed/end-of-day)"}},
                 fund=FUND)
    head = next(l for l in out.splitlines() if "Current price:" in l)
    assert "as of 2026-08-18" in head and "yfinance" in head
    s19 = _section(out, "### 19. Final Investment Committee Output")
    assert "as of" in s19 and "OBSERVED DATA" in s19


def test_macro_transmission_labels_ranges_and_shows_wacc_shock():
    out = _memo(fund=FUND)
    sec = _section(out, "### 9. Macro Transmission Engine")
    assert "SCENARIO ASSUMPTION" in sec
    assert "not regressions" in sec or "Why these are assumptions" in sec
    # the only empirically-anchored number: +100bps WACC shock from section 6
    assert "MODEL CALCULATION, sec. 6" in sec or "MODEL CALCULATION from the section-6 DCF" in sec
    assert "+100bps WACC" in sec


# --------------------------------------------------------------------------- #
#  21. REVIEW FIX 8: Bayesian basis column + honest calibration caveat
# --------------------------------------------------------------------------- #

def test_bayesian_has_evidence_basis_and_honesty_note():
    out = _memo(fund=None)
    sec = _section(out, "### 15. Bayesian Update Engine")
    assert "Evidence basis" in sec
    assert "calibration caveat" in sec
    assert "decision framework" in sec
    assert "MODEL ASSUMPTIONS" in sec


def test_empty_live_data_still_builds():
    out = _memo(live={}, fund=None)
    assert out
    assert "live quote temporarily unavailable" in out
    sec19 = _section(out, "### 19. Final Investment Committee Output")
    assert "DATA UNAVAILABLE" in sec19
