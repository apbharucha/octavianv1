"""
Tests for the institutional Financial Model Generator upgrade suite:

1. Trading Comps engine (relevance, stats, implied valuation, outliers)
2. Precedent Transactions engine (relevance, adjusted range, premium analysis)
3. IPO engine (market pulse, share bridge, pricing, proceeds, scenarios)
4. Cross-model valuation bridge (weighting, convergence, outliers)
5. Market consensus / dislocation layer
6. Model QA auditor
7. Investment memo generator
8. Upgraded M&A / LBO institutional fields (earnouts, synergy schedule,
   PIK, returns attribution, scenario cases)
"""

import pytest

from comps_engine import (
    CompsCompany, CompsInputs, get_comps_engine,
)
from precedent_transactions_engine import (
    PrecedentTransaction, PrecedentInputs, get_precedent_engine,
)
from ipo_engine import IPOAssumptions, get_ipo_engine
from market_consensus_engine import (
    default_market_pulse_inputs, MarketPulseInputs, get_market_pulse_engine,
    reverse_dcf_expectations, get_consensus_dislocation_engine,
    ConsensusEstimates, SentimentSnapshot,
)
from valuation_bridge import get_valuation_bridge
from model_audit import run_model_qa
from investment_memo import MemoContext, get_memo_generator
from mna_model_engine import MnAAssumptions, get_mna_engine
from lbo_model_engine import LBOAssumptions, get_lbo_engine


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_subject(**kw):
    base = dict(
        ticker="SUB", name="Subject Co", sector="Technology",
        business_model="SaaS", revenue=1000, revenue_growth_pct=25,
        ebitda_margin_pct=25, ebitda=250, ebit=200, fcf=180,
        net_income=150, net_debt=100, market_cap=5000,
        shares_outstanding=50, price=100,
    )
    base.update(kw)
    return CompsCompany(**base)


def make_peers(n=6):
    return [
        CompsCompany(
            ticker=f"P{i}", sector="Technology", business_model="SaaS",
            revenue=800 + i * 50, revenue_growth_pct=20 + i,
            ebitda_margin_pct=25 + i, ebitda=200 + i * 10, ebit=160,
            fcf=150, net_income=120, net_debt=0,
            market_cap=4000 + i * 200, shares_outstanding=40, price=100)
        for i in range(n)
    ]


def make_transactions(n=6):
    return [
        PrecedentTransaction(
            deal_name=f"Deal{i}", buyer="Acquirer", seller="Target",
            enterprise_value=1500 + i * 100, equity_value=1200,
            seller_revenue=500, seller_revenue_growth_pct=15,
            seller_ebitda=100 + i * 5, seller_ebitda_margin_pct=20,
            seller_ebit=80, seller_fcf=70, control_premium_pct=25 + i,
            buyer_type="strategic", structure="cash", market_regime="neutral",
        )
        for i in range(n)
    ]


def make_ipo_assumptions(**kw):
    base = dict(
        ticker="SUB", company_name="Subject Co", sector="Technology",
        revenue=1000, revenue_growth_pct=25, ebitda=250,
        ebitda_margin_pct=25, ebit=200, fcf=180, net_income=150,
        net_debt=100, shares_outstanding_m=50,
        options_m=5, options_strike=20, warrants_m=2, warrants_strike=30,
        rsus_m=3, primary_shares_m=12, secondary_shares_m=5,
        market_pulse=default_market_pulse_inputs(), base_revenue=1000,
    )
    base.update(kw)
    return IPOAssumptions(**base)


# ─────────────────────────────────────────────────────────────────────────────
# Trading Comps
# ─────────────────────────────────────────────────────────────────────────────

def test_comps_runs_and_has_key_outputs():
    result = get_comps_engine().run_comps(
        CompsInputs(subject=make_subject(), peers=make_peers()))
    assert result.recommendation
    assert result.conclusion
    assert "Blended (weighted)" in result.implied_valuation
    assert result.implied_range[1] >= result.implied_range[0] >= 0
    assert result.implied_valuation["Blended (weighted)"] > 0
    assert result.peers_df is not None and not result.peers_df.empty
    assert len(result.relevance) == 6


def test_comps_stats_and_outliers():
    peers = make_peers()
    # Inject an obvious outlier: one peer trading at a wildly rich multiple
    peers[2].ebitda = 30  # tiny EBITDA → huge EV/EBITDA
    result = get_comps_engine().run_comps(
        CompsInputs(subject=make_subject(), peers=peers))
    s = result.stats.get("EV/EBITDA", {})
    assert s["count"] >= 5
    assert s["q1"] <= s["median"] <= s["q3"]
    assert any(o["ticker"] == "P2" for o in result.outliers)


def test_comps_relevance_penalizes_mismatch():
    peers = [CompsCompany(ticker="FAR", sector="Energy", business_model="Hardware",
                          revenue=900, revenue_growth_pct=2, ebitda_margin_pct=8,
                          ebitda=72, market_cap=1000, shares_outstanding=40, price=25)]
    result = get_comps_engine().run_comps(
        CompsInputs(subject=make_subject(), peers=peers))
    assert result.relevance["FAR"]["score"] < 0.5


def test_comps_regressions_populated():
    result = get_comps_engine().run_comps(
        CompsInputs(subject=make_subject(), peers=make_peers(6)))
    reg = result.regressions or {}
    assert reg.get("ev_ebitda_vs_growth", {}).get("n", 0) >= 3


# ─────────────────────────────────────────────────────────────────────────────
# Precedent Transactions
# ─────────────────────────────────────────────────────────────────────────────

def test_precedents_runs():
    inputs = PrecedentInputs(
        subject_name="SUB", subject_sector="Technology",
        subject_revenue=1000, subject_revenue_growth_pct=25,
        subject_ebitda=250, subject_ebitda_margin_pct=25,
        subject_ebit=200, subject_fcf=180, subject_net_debt=100,
        subject_shares=50, current_market_regime="neutral",
        transactions=make_transactions())
    result = get_precedent_engine().run_precedents(inputs)
    assert result.core_deals
    assert result.adjusted_range[1] >= result.adjusted_range[0] >= 0
    assert result.control_premium_analysis["median_premium_pct"] > 0
    assert "Blended (weighted)" in result.implied_valuation
    assert result.conclusion


def test_precedents_multiples_math():
    t = make_transactions(1)[0]
    m = t.multiples()
    assert m["EV/EBITDA"] == pytest.approx(
        t.enterprise_value / t.seller_ebitda)
    assert m["EV/Revenue"] == pytest.approx(
        t.enterprise_value / t.seller_revenue)


# ─────────────────────────────────────────────────────────────────────────────
# IPO engine
# ─────────────────────────────────────────────────────────────────────────────

def test_ipo_full_run():
    result = get_ipo_engine().run_ipo(make_ipo_assumptions())
    lo, hi = result.price_range
    assert lo > 0 and hi >= lo
    assert result.gross_proceeds == pytest.approx(
        result.primary_proceeds + result.secondary_proceeds)
    assert result.gross_proceeds > result.net_proceeds
    assert result.market_pulse["label"] in result.market_pulse["scale"]
    assert result.share_count_bridge["total_shares_post_ipo"] > 0
    assert result.dilution_pct > 0
    assert len(result.scenarios) >= 3
    total_prob = sum(s.probability for s in result.scenarios)
    assert total_prob == pytest.approx(1.0, abs=1e-6)


def test_ipo_share_bridge_treasury_stock_method():
    a = make_ipo_assumptions(options_m=10, options_strike=10,
                             shares_outstanding_m=50)
    # Offer price well above strike → all options in the money, net of buyback
    from ipo_engine import build_share_count_bridge
    bridge = build_share_count_bridge(a, offer_price=100)
    assert bridge["options_itm"] == 10.0
    assert bridge["options_net_tsm"] < 10.0  # buyback reduces net
    assert bridge["total_shares_post_ipo"] > 50


def test_ipo_market_pulse_scales():
    engine = get_market_pulse_engine()
    good = engine.evaluate(MarketPulseInputs(
        vix_level=12, ten_year_yield_pct=3.5, credit_spread_bps=120,
        recent_ipo_avg_first_day_return_pct=30, recent_ipo_pop_pct=80,
        ipo_withdrawals_12m=1, risk_appetite=85, equity_regime="bull"))
    bad = engine.evaluate(MarketPulseInputs(
        vix_level=40, ten_year_yield_pct=6.0, credit_spread_bps=400,
        recent_ipo_avg_first_day_return_pct=-10, recent_ipo_pop_pct=20,
        ipo_withdrawals_12m=15, risk_appetite=20, equity_regime="bear"))
    assert good["label"] in ("Highly supportive", "Supportive")
    assert bad["label"] in ("Challenging", "Highly challenging")
    assert good["observed_signals"] >= 6  # explanation requires evidence


# ─────────────────────────────────────────────────────────────────────────────
# Consensus / dislocation
# ─────────────────────────────────────────────────────────────────────────────

def test_reverse_dcf_expectations():
    impl = reverse_dcf_expectations(
        model_price=60, current_price=40, shares=50, net_debt=100,
        base_revenue=1000, model_revenue_growth=0.10,
        model_ebit_margin=0.15, wacc=0.10, terminal_growth=0.025)
    assert impl.model_price == 60
    assert impl.upside_pct == pytest.approx(50.0)
    assert 0 < impl.implied_revenue_growth < 1
    assert impl.gap_label


def test_dislocation_detects_upside():
    impl = reverse_dcf_expectations(
        model_price=80, current_price=40, shares=50, net_debt=100,
        base_revenue=1000, model_revenue_growth=0.10,
        model_ebit_margin=0.15, wacc=0.10, terminal_growth=0.025)
    report = get_consensus_dislocation_engine().analyze(
        implied=impl, model_price=80, current_price=40, model_upside_pct=1.0)
    assert any(f.direction == "Positive" for f in report.findings)
    assert report.overall_reading


def test_sentiment_composite_labeling():
    s = SentimentSnapshot(news_sentiment=0.6, analyst_sentiment=0.4,
                          sector_sentiment=0.2)
    assert s.label() == "Bullish sentiment"
    s2 = SentimentSnapshot()
    assert s2.label() == "No sentiment data"


# ─────────────────────────────────────────────────────────────────────────────
# Valuation bridge
# ─────────────────────────────────────────────────────────────────────────────

def test_bridge_weighting_and_outliers():
    bridge = get_valuation_bridge()
    methods = [
        bridge.add_dcf(type("D", (), {"fair_value_per_share": 55,
                                      "scenario_weighted_value": 50,
                                      "mc_percentiles": {"p25": 45, "p75": 65}})),
        bridge.add_market(40),
        bridge.add_market(41),  # duplicate-ish narrow method
    ]
    methods = [m for m in methods if m]
    result = bridge.run(methods, current_price=40)
    assert result.weighted_mid > 0
    assert result.weighted_range[1] >= result.weighted_range[0]
    assert set(result.weights.keys()) == {m.key for m in methods}
    assert result.valuation_statement


def test_bridge_integrates_all_models():
    bridge = get_valuation_bridge()
    comps = get_comps_engine().run_comps(
        CompsInputs(subject=make_subject(), peers=make_peers()))
    prec = get_precedent_engine().run_precedents(PrecedentInputs(
        subject_name="SUB", subject_sector="Technology", subject_revenue=1000,
        subject_revenue_growth_pct=25, subject_ebitda=250,
        subject_ebitda_margin_pct=25, subject_ebit=200, subject_fcf=180,
        subject_net_debt=100, subject_shares=50, current_market_regime="neutral",
        transactions=make_transactions()))
    ipo = get_ipo_engine().run_ipo(make_ipo_assumptions())
    methods = [
        bridge.add_dcf(type("D", (), {"fair_value_per_share": 55,
                                      "scenario_weighted_value": 50,
                                      "mc_percentiles": {"p25": 45, "p75": 65}})),
        bridge.add_comps(comps),
        bridge.add_precedents(prec),
        bridge.add_ipo(ipo),
        bridge.add_market(40),
    ]
    methods = [m for m in methods if m]
    assert len(methods) >= 4
    result = bridge.run(methods, current_price=40)
    assert result.weighted_mid > 0
    assert result.drivers


# ─────────────────────────────────────────────────────────────────────────────
# Model QA
# ─────────────────────────────────────────────────────────────────────────────

def test_qa_dcf_pass():
    r = type("D", (), {
        "fair_value_per_share": 55, "scenario_weighted_value": 50,
        "mc_percentiles": {"p25": 45, "p75": 65},
        "enterprise_value": 5000, "equity_value": 2500, "net_debt": 2500,
        "wacc": 0.10, "pv_terminal_value": 3000,
        "terminal_growth": 0.025,
        "line_items": type("L", (), {"empty": True})(),
    })()
    qa = run_model_qa(r, "dcf")
    assert qa.status in ("PASS", "WARN")
    assert qa.pass_count >= 1


def test_qa_fails_impossible_terminal():
    r = type("D", (), {
        "fair_value_per_share": 55, "scenario_weighted_value": 50,
        "mc_percentiles": {"p25": 45, "p75": 65},
        "enterprise_value": 5000, "equity_value": 2500, "net_debt": 2500,
        "wacc": 0.05, "pv_terminal_value": 3000,
        "terminal_growth": 0.06,  # g >= WACC → Gordon growth invalid
        "line_items": type("L", (), {"empty": True})(),
    })()
    qa = run_model_qa(r, "dcf")
    assert any(c.status == "FAIL" for c in qa.checks)


# ─────────────────────────────────────────────────────────────────────────────
# Investment memo
# ─────────────────────────────────────────────────────────────────────────────

def test_memo_generation():
    memo = get_memo_generator().generate(MemoContext(
        title="Subject Co", ticker="SUB", subject_type="IPO",
        fair_value=55, current_price=40,
        market_pulse_label="Supportive",
        scenarios=[{"label": "Base", "probability": 0.5, "price": 55.0,
                    "rationale": "Central path"}],
        risks=["Demand weakness"], catalysts=["Index inclusion"]))
    assert "## 17. Final Conclusion" in memo
    assert "Subject Co" in memo
    assert "Supportive" in memo


# ─────────────────────────────────────────────────────────────────────────────
# Upgraded M&A institutional fields
# ─────────────────────────────────────────────────────────────────────────────

def make_mna(**kw):
    base = dict(
        acquirer_ticker="ACQ", target_ticker="TGT",
        acquirer_price=100, acquirer_eps=5, acquirer_shares=1000,
        target_price=50, target_eps=2, target_shares=200,
        offer_premium=0.30, percent_stock=0.5, percent_cash=0.5,
        cost_of_debt=0.06, tax_rate=0.21, pre_tax_synergies=100,
        earnout_value=50, earnout_probability=0.8,
        synergy_probability=0.9, buyer_hold_years=4,
    )
    base.update(kw)
    return MnAAssumptions(**base)


def test_mna_upgrade_fields_present():
    res = get_mna_engine().run_mna(make_mna())
    assert res.earnout_metrics["value"] == 50
    assert res.earnout_metrics["expected_value"] == pytest.approx(40.0)
    assert not res.synergy_schedule.empty
    assert res.buyer_returns["hold_years"] == 4
    assert res.seller_proceeds["total_expected_proceeds"] == pytest.approx(
        res.total_deal_value + 40.0)
    assert res.max_purchase_price > 0
    assert res.value_bridge["net_value_creation"] != 0
    assert not res.transaction_scenarios.empty
    # FCF accretion is computed
    assert "fcf_accretion_pct" in res.fcf_accretion


def test_mna_earnout_affects_eps():
    no_earn = get_mna_engine().run_mna(make_mna(earnout_value=0))
    with_earn = get_mna_engine().run_mna(make_mna(earnout_value=200,
                                                  earnout_probability=1.0))
    assert with_earn.pro_forma_eps < no_earn.pro_forma_eps


# ─────────────────────────────────────────────────────────────────────────────
# Upgraded LBO institutional fields
# ─────────────────────────────────────────────────────────────────────────────

def make_lbo(**kw):
    base = dict(
        ticker="TGT", target_name="Target", entry_year=2026, exit_year=2031,
        ltm_ebitda=500, entry_multiple=10, exit_multiple=10,
        leverage_multiple=5, interest_rate=0.08, tax_rate=0.25,
        pik_interest_pct=0.3, second_lien_multiple=1.0,
    )
    base.update(kw)
    return LBOAssumptions(**base)


def test_lbo_upgrade_fields_present():
    res = get_lbo_engine().run_lbo(make_lbo())
    assert res.returns_attribution["total_uplift"] == pytest.approx(
        res.exit_equity_value - res.equity_amount)
    assert set(res.returns_attribution["components"].keys()) == {
        "Operating (EBITDA growth)", "Multiple expansion / contraction",
        "Deleveraging", "Net cash at exit"}
    assert not res.scenario_cases.empty
    assert list(res.scenario_cases["Scenario"]) == ["Downside", "Base", "Upside"]
    assert res.debt_structure["second_lien"] == pytest.approx(500.0)
    assert res.debt_structure["pik_share"] == pytest.approx(0.3)


def test_lbo_pik_increases_debt_balance():
    no_pik = get_lbo_engine().run_lbo(make_lbo(pik_interest_pct=0.0))
    pik = get_lbo_engine().run_lbo(make_lbo(pik_interest_pct=1.0))
    # PIK rolls interest into principal → higher ending debt than cash-pay
    last_no = no_pik.cash_flows["Ending TLB Balance"].iloc[-1]
    last_pik = pik.cash_flows["Ending TLB Balance"].iloc[-1]
    assert last_pik >= last_no


def test_lbo_scenario_probabilities_sum_to_one():
    res = get_lbo_engine().run_lbo(make_lbo())
    assert res.scenario_cases["Probability"].sum() == pytest.approx(1.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# New Excel workbook builders (live-formula institutional deliverables)
# ─────────────────────────────────────────────────────────────────────────────

def test_comps_workbook_builds():
    from ib_excel_engine import build_comps_workbook
    comps = get_comps_engine().run_comps(
        CompsInputs(subject=make_subject(), peers=make_peers()))
    data = build_comps_workbook(comps.peers_df)
    assert isinstance(data, bytes) and len(data) > 1000


def test_precedents_workbook_builds():
    from ib_excel_engine import build_precedents_workbook
    prec = get_precedent_engine().run_precedents(PrecedentInputs(
        subject_name="SUB", subject_sector="Technology", subject_revenue=1000,
        subject_revenue_growth_pct=25, subject_ebitda=250,
        subject_ebitda_margin_pct=25, subject_ebit=200, subject_fcf=180,
        subject_net_debt=100, subject_shares=50, current_market_regime="neutral",
        transactions=make_transactions()))
    data = build_precedents_workbook(prec)
    assert isinstance(data, bytes) and len(data) > 1000


def test_ipo_workbook_builds():
    from ib_excel_engine import build_ipo_workbook
    res = get_ipo_engine().run_ipo(make_ipo_assumptions())
    data = build_ipo_workbook(res)
    assert isinstance(data, bytes) and len(data) > 1000


def test_bridge_workbook_builds():
    from ib_excel_engine import build_bridge_workbook
    bridge = get_valuation_bridge()
    methods = [
        bridge.add_dcf(type("D", (), {"fair_value_per_share": 55,
                                      "scenario_weighted_value": 50,
                                      "mc_percentiles": {"p25": 45, "p75": 65}})),
        bridge.add_market(40),
    ]
    methods = [m for m in methods if m]
    result = bridge.run(methods, current_price=40)
    data = build_bridge_workbook(result)
    assert isinstance(data, bytes) and len(data) > 1000


# ─────────────────────────────────────────────────────────────────────────────
# New pitchbook builders (investment-banking PowerPoint deliverables)
# ─────────────────────────────────────────────────────────────────────────────

def test_comps_pitchbook_builds():
    from presentation_generator import get_presentation_generator
    comps = get_comps_engine().run_comps(
        CompsInputs(subject=make_subject(), peers=make_peers()))
    data = get_presentation_generator().generate_comps_pitchbook(comps)
    assert isinstance(data, bytes) and len(data) > 1000


def test_precedents_pitchbook_builds():
    from presentation_generator import get_presentation_generator
    prec = get_precedent_engine().run_precedents(PrecedentInputs(
        subject_name="SUB", subject_sector="Technology", subject_revenue=1000,
        subject_revenue_growth_pct=25, subject_ebitda=250,
        subject_ebitda_margin_pct=25, subject_ebit=200, subject_fcf=180,
        subject_net_debt=100, subject_shares=50, current_market_regime="neutral",
        transactions=make_transactions()))
    data = get_presentation_generator().generate_precedents_pitchbook(prec)
    assert isinstance(data, bytes) and len(data) > 1000


def test_ipo_pitchbook_builds():
    from presentation_generator import get_presentation_generator
    res = get_ipo_engine().run_ipo(make_ipo_assumptions())
    data = get_presentation_generator().generate_ipo_pitchbook(res)
    assert isinstance(data, bytes) and len(data) > 1000


# ─────────────────────────────────────────────────────────────────────────────
# Zero / negative denominator robustness (institutional data edge cases)
# ─────────────────────────────────────────────────────────────────────────────

def test_comps_handles_zero_ebitda_peers():
    peers = make_peers(3)
    peers[1].ebitda = 0.0   # EV/EBITDA must be None, not crash or poison stats
    peers[2].ebitda = -50.0  # negative EBITDA (pre-revenue style) handled
    result = get_comps_engine().run_comps(
        CompsInputs(subject=make_subject(), peers=peers))
    assert result.peers_df is not None
    m = result.peers_df[result.peers_df["Ticker"] == "P1"]["EV/EBITDA"].iloc[0]
    assert m is None or (isinstance(m, float) and m != m)  # NaN/None, never a number


def test_precedents_handles_zero_ebitda():
    tx = make_transactions(1)
    tx[0].seller_ebitda = 0.0
    inputs = PrecedentInputs(
        subject_name="SUB", subject_sector="Technology", subject_revenue=1000,
        subject_revenue_growth_pct=25, subject_ebitda=250,
        subject_ebitda_margin_pct=25, subject_ebit=200, subject_fcf=180,
        subject_net_debt=100, subject_shares=50, current_market_regime="neutral",
        transactions=tx)
    result = get_precedent_engine().run_precedents(inputs)
    m = tx[0].multiples()
    assert m["EV/EBITDA"] is None
    assert m["EV/Revenue"] is not None  # unaffected metric still computed


def test_ipo_handles_negative_ebitda():
    # Loss-making issuer: EBITDA < 0 → EBITDA-multiple methods must be skipped
    res = get_ipo_engine().run_ipo(make_ipo_assumptions(ebitda=-100))
    lo, hi = res.price_range
    assert lo > 0 and hi >= lo
    assert res.valuation_methods  # still priced off revenue-based methods
