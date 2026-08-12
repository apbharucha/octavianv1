"""
Institutional-Grade M&A Accretion / Dilution Model

Goldman-Sachs-standard analyst output: a fully-specified accretion/dilution
model with contribution analysis (the standard "EPS walk"), 2-way sensitivity
tables (premium x % stock, premium x synergies), transaction fees, a sources &
uses check, and a layered narrative (executive verdict -> institutional detail
-> plain-English risks).

Institutional upgrades (additive — the core math is unchanged):
* Earnout / contingent consideration with probability weighting
* Synergy realization schedule (phasing, probability, implementation costs)
* Buyer return analysis (IRR over hold period) and seller proceeds breakdown
* Maximum purchase price analysis (walk-away / value-neutral price)
* Value-creation bridge (standalone value + synergy value + financing)
* FCF accretion / (dilution) alongside EPS accretion

Author: Octavian Terminal
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class MnAAssumptions:
    acquirer_ticker: str
    target_ticker: str

    # Financials
    acquirer_price: float
    acquirer_eps: float
    acquirer_shares: float

    target_price: float
    target_eps: float
    target_shares: float

    # Deal Terms
    offer_premium: float                      # e.g. 0.30 for 30% premium
    percent_stock: float                      # 0.0 to 1.0
    percent_cash: float                       # 0.0 to 1.0

    # Financing
    cost_of_debt: float                       # pre-tax
    tax_rate: float

    # Synergies & Costs
    pre_tax_synergies: float                  # $M

    # Optional fields (all defaulted)
    acquirer_net_debt: float = 0.0            # $M (positive = net debt)
    target_net_debt: float = 0.0              # $M
    cash_interest_income_rate: float = 0.0    # return on foregone cash / retained cash
    integration_costs: float = 0.0            # one-time, pre-tax, $M
    advisory_fee_pct: float = 0.005           # 50bps of deal value
    financing_fee_pct: float = 0.01           # 100bps of debt raised

    # ── Institutional upgrades (all defaulted → fully backward compatible) ──
    # Earnout / contingent consideration
    earnout_value: float = 0.0                # $M max contingent consideration
    earnout_probability: float = 1.0          # 0..1 probability of paying out
    earnout_years: int = 2                    # typical earnout measurement window

    # Synergy realization
    synergy_phase_years: List[float] = field(default_factory=list)   # % realized per year
    synergy_probability: float = 1.0          # 0..1 probability of realizing synergies
    synergy_implementation_costs: float = 0.0 # $M one-time costs to capture synergies

    # Buyer return / value creation inputs
    buyer_hold_years: int = 3
    acquirer_revenue: float = 0.0             # $M
    target_revenue: float = 0.0               # $M
    acquirer_ebitda: float = 0.0              # $M
    target_ebitda: float = 0.0                # $M
    acquirer_fcf: float = 0.0                 # $M
    target_fcf: float = 0.0                   # $M
    acquirer_net_income: float = 0.0          # $M (fallback if EPS not reliable)
    target_net_income: float = 0.0            # $M
    acquirer_ebit: float = 0.0
    target_ebit: float = 0.0

    # Synergy split (cost vs revenue) — used for value bridge & narrative
    cost_synergy_pct: float = 0.60
    revenue_synergy_pct: float = 0.40

    # Financing detail
    existing_target_debt_assumed: float = 0.0   # $M of target debt assumed (vs refinanced)
    new_debt_issuance: float = 0.0              # $M override of debt raised (None → derived)


@dataclass
class MnASensitivity:
    """2-way accretion/dilution tables."""
    premium_stock_table: pd.DataFrame      # rows=premium, cols=%stock
    premium_synergy_table: pd.DataFrame    # rows=premium, cols=synergy level
    breakeven_synergies: Optional[float]   # $M pre-tax synergies for 0% acc/dil


@dataclass
class MnAResult:
    assumptions: MnAAssumptions
    offer_price: float
    total_deal_value: float
    total_uses: float
    total_sources: float
    sources_uses_check: float               # should be ~0
    new_shares_issued: float
    pro_forma_shares: float
    pro_forma_net_income: float
    pro_forma_eps: float
    accretion_dilution_dollar: float
    accretion_dilution_pct: float
    is_accretive: bool
    contribution_analysis: pd.DataFrame     # EPS walk
    sensitivity: MnASensitivity
    deal_metrics: Dict[str, float]
    conclusion: str = ""

    # ── Institutional upgrade fields (defaulted → backward compatible) ──────
    earnout_metrics: Dict[str, float] = field(default_factory=dict)
    synergy_schedule: pd.DataFrame = field(default_factory=pd.DataFrame)
    buyer_returns: Dict[str, float] = field(default_factory=dict)
    seller_proceeds: Dict[str, float] = field(default_factory=dict)
    max_purchase_price: float = 0.0
    value_bridge: Dict[str, float] = field(default_factory=dict)
    fcf_accretion: Dict[str, float] = field(default_factory=dict)
    transaction_scenarios: pd.DataFrame = field(default_factory=pd.DataFrame)


class InstitutionalMnAEngine:

    _PREMIUM_RANGE = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    _STOCK_RANGE = [0.0, 0.25, 0.5, 0.75, 1.0]
    _SYNERGY_MULTIPLIERS = [0.0, 0.5, 1.0, 1.5, 2.0]

    def _check_assumptions(self, a: MnAAssumptions) -> List[str]:
        """Validate inputs; return list of warnings (empty = all good)."""
        warnings = []
        if abs((a.percent_stock + a.percent_cash) - 1.0) > 1e-6:
            warnings.append(
                f"Stock ({a.percent_stock:.0%}) + cash ({a.percent_cash:.0%}) "
                f"consideration does not sum to 100%."
            )
        if a.acquirer_price <= 0 or a.target_price <= 0:
            warnings.append("Share prices must be positive.")
        if a.acquirer_shares <= 0 or a.target_shares <= 0:
            warnings.append("Shares outstanding must be positive.")
        if a.acquirer_eps <= 0:
            warnings.append("Acquirer EPS is non-positive — accretion math will be degenerate.")
        if a.offer_premium < 0:
            warnings.append("Offer premium should not be negative.")
        if a.cost_of_debt < 0 or a.tax_rate < 0 or a.tax_rate > 0.5:
            warnings.append("Cost of debt / tax rate outside normal ranges.")
        if not (0 <= a.earnout_probability <= 1):
            warnings.append("Earnout probability outside [0,1].")
        if not (0 <= a.synergy_probability <= 1):
            warnings.append("Synergy probability outside [0,1].")
        return warnings

    def _accretion_dilution(self, a: MnAAssumptions) -> Dict[str, float]:
        """Core accretion/dilution math. Returns the intermediate dict."""
        offer_price = a.target_price * (1 + a.offer_premium)
        total_deal_value = offer_price * a.target_shares  # $M

        # Fees
        advisory_fee = total_deal_value * a.advisory_fee_pct
        cash_needed = total_deal_value * a.percent_cash
        debt_raised = cash_needed
        financing_fee = debt_raised * a.financing_fee_pct
        total_fees = advisory_fee + financing_fee

        # Uses = purchase of target equity + refinance of target net debt + fees
        target_equity_purchase = total_deal_value
        total_uses = target_equity_purchase + max(a.target_net_debt, 0.0) + total_fees

        # Sources = stock consideration + new debt + cash from balance sheet
        equity_consideration = total_deal_value * a.percent_stock
        new_shares_issued = equity_consideration / a.acquirer_price if a.acquirer_price > 0 else 0.0
        cash_sources = max(a.target_net_debt, 0.0)  # debt refinanced from new debt
        total_sources = cash_sources + debt_raised + equity_consideration
        # The residual (equity consideration + fees not covered by debt/cash)
        # is funded by the acquirer's existing cash / new sponsor equity:
        balance_cash = max(total_uses - total_sources, 0.0)
        total_sources += balance_cash
        sources_uses_check = total_sources - total_uses

        pro_forma_shares = a.acquirer_shares + new_shares_issued

        # Interest on new debt (cash portion), net of tax
        new_interest_expense = debt_raised * a.cost_of_debt
        after_tax_interest = new_interest_expense * (1 - a.tax_rate)

        # After-tax synergies and one-time integration costs
        after_tax_synergies = a.pre_tax_synergies * (1 - a.tax_rate)
        after_tax_integration = a.integration_costs * (1 - a.tax_rate)

        # Earnout (probability-weighted contingent consideration) — a pure
        # expense in the base case, amortized over the earnout window.
        expected_earnout = a.earnout_value * a.earnout_probability
        after_tax_earnout_annual = (
            expected_earnout * (1 - a.tax_rate) / max(a.earnout_years, 1)
        )

        acquirer_net_income = a.acquirer_eps * a.acquirer_shares  # $M
        target_net_income = a.target_eps * a.target_shares        # $M

        pro_forma_net_income = (
            acquirer_net_income
            + target_net_income
            + after_tax_synergies
            - after_tax_interest
            - after_tax_integration
            - after_tax_earnout_annual
        )

        pro_forma_eps = pro_forma_net_income / pro_forma_shares if pro_forma_shares > 0 else 0.0
        accretion_dollar = pro_forma_eps - a.acquirer_eps
        accretion_pct = accretion_dollar / a.acquirer_eps if a.acquirer_eps > 0 else 0.0

        return {
            "offer_price": offer_price,
            "total_deal_value": total_deal_value,
            "advisory_fee": advisory_fee,
            "financing_fee": financing_fee,
            "total_fees": total_fees,
            "debt_raised": debt_raised,
            "cash_needed": cash_needed,
            "equity_consideration": equity_consideration,
            "new_shares_issued": new_shares_issued,
            "pro_forma_shares": pro_forma_shares,
            "pro_forma_net_income": pro_forma_net_income,
            "pro_forma_eps": pro_forma_eps,
            "accretion_dollar": accretion_dollar,
            "accretion_pct": accretion_pct,
            "after_tax_synergies": after_tax_synergies,
            "after_tax_interest": after_tax_interest,
            "after_tax_integration": after_tax_integration,
            "acquirer_net_income": acquirer_net_income,
            "target_net_income": target_net_income,
            "total_uses": total_uses,
            "total_sources": total_sources,
            "sources_uses_check": sources_uses_check,
            "balance_cash": balance_cash,
            "target_equity_purchase": target_equity_purchase,
            "expected_earnout": expected_earnout,
            "after_tax_earnout_annual": after_tax_earnout_annual,
        }

    def _contribution_analysis(self, a: MnAAssumptions, m: Dict[str, float]) -> pd.DataFrame:
        """Standard IB 'EPS bridge': show each driver's cps (per-share) contribution.

        Rows: acquirer standalone EPS, target EPS contribution (P/E effect),
        after-tax synergies, after-tax integration costs, after-tax interest
        (financing effect), net dilution from new shares, pro forma EPS.
        """
        pf_shares = m["pro_forma_shares"]
        base = a.acquirer_eps
        rows = []

        def add(label, delta, kind):
            rows.append({"Driver": label, "EPS Impact ($)": delta,
                         "Cumulative EPS ($)": base + sum(r["EPS Impact ($)"] for r in rows),
                         "Type": kind})

        # Contribution of target earnings on a per-share basis
        tgt_cps = m["target_net_income"] / pf_shares
        add("Target net income contribution (P/E effect)", tgt_cps, "Operating")
        # Synergies
        syn_cps = m["after_tax_synergies"] / pf_shares
        add("After-tax cost & revenue synergies", syn_cps, "Synergy")
        # Integration costs
        int_cps = -m["after_tax_integration"] / pf_shares
        add("One-time integration costs (after-tax)", int_cps, "One-time")
        # Earnout (if any)
        if m.get("after_tax_earnout_annual", 0) > 0:
            earn_cps = -m["after_tax_earnout_annual"] / pf_shares
            add("Earnout amortization (after-tax)", earn_cps, "One-time")
        # Financing
        fin_cps = -m["after_tax_interest"] / pf_shares
        add("After-tax interest on new debt (financing effect)", fin_cps, "Financing")
        # Dilution from new shares issued (pro forma share count effect)
        # The dilution is implicit: base EPS on pf shares = (acquirer NI)/pf_shares
        acq_ni_cps = m["acquirer_net_income"] / pf_shares
        dil_cps = acq_ni_cps - base
        add("Share-count dilution from stock consideration", dil_cps, "Dilution")

        pro_forma = m["pro_forma_eps"]
        add("Pro forma EPS", pro_forma - base, "Result")
        return pd.DataFrame(rows)

    def _sensitivity(self, a: MnAAssumptions) -> MnASensitivity:
        """2-way sensitivity: accretion/dilution % vs (premium, %stock) and
        (premium, synergy level)."""
        prem_rows, stock_cols = [], self._STOCK_RANGE
        for prem in self._PREMIUM_RANGE:
            row = {"Premium": f"{prem:.0%}"}
            for s in stock_cols:
                mod = MnAAssumptions(**{**a.__dict__, "offer_premium": prem,
                                        "percent_stock": s, "percent_cash": 1 - s})
                row[f"{s:.0%}"] = self._accretion_dilution(mod)["accretion_pct"]
            prem_rows.append(row)
        prem_stock = pd.DataFrame(prem_rows)

        prem_rows2 = []
        base_syn = a.pre_tax_synergies
        for prem in self._PREMIUM_RANGE:
            row = {"Premium": f"{prem:.0%}"}
            for mult in self._SYNERGY_MULTIPLIERS:
                mod = MnAAssumptions(**{**a.__dict__, "offer_premium": prem,
                                        "pre_tax_synergies": base_syn * mult})
                row[f"{mult:.0f}x"] = self._accretion_dilution(mod)["accretion_pct"]
            prem_rows2.append(row)
        prem_syn = pd.DataFrame(prem_rows2)

        # Breakeven synergies: the pre-tax synergy level that zeroes accretion
        breakeven = None
        try:
            # Linear solve: accretion_pct(syn) = (NI0 + syn*(1-t))/pf_shares / eps - 1
            pf_shares = self._accretion_dilution(a)["pro_forma_shares"]
            ni0 = (self._accretion_dilution(a)["pro_forma_net_income"]
                   - self._accretion_dilution(a)["after_tax_synergies"])
            eps = a.acquirer_eps
            # need: (ni0 + syn_at)/pf_shares = eps  => syn_at = eps*pf_shares - ni0
            syn_at = eps * pf_shares - ni0
            breakeven = syn_at / (1 - a.tax_rate) if (1 - a.tax_rate) > 0 else None
        except Exception:
            breakeven = None

        return MnASensitivity(premium_stock_table=prem_stock,
                              premium_synergy_table=prem_syn,
                              breakeven_synergies=breakeven)

    # ── Institutional upgrade: earnout metrics ───────────────────────────────
    def _earnout_metrics(self, a: MnAAssumptions, m: Dict[str, float]) -> Dict[str, float]:
        if a.earnout_value <= 0:
            return {"value": 0.0, "probability": 0.0, "expected_value": 0.0,
                    "annual_after_tax_eps_impact": 0.0, "pct_of_deal": 0.0}
        expected = a.earnout_value * a.earnout_probability
        annual_after_tax = expected * (1 - a.tax_rate) / max(a.earnout_years, 1)
        eps_impact = annual_after_tax / max(m["pro_forma_shares"], 1e-9)
        return {
            "value": a.earnout_value,
            "probability": a.earnout_probability,
            "expected_value": expected,
            "annual_after_tax_eps_impact": eps_impact,
            "pct_of_deal": a.earnout_value / max(m["total_deal_value"], 1e-9),
            "years": a.earnout_years,
        }

    # ── Institutional upgrade: synergy realization schedule ─────────────────
    def _synergy_schedule(self, a: MnAAssumptions) -> pd.DataFrame:
        years = [1, 2, 3]
        phases = a.synergy_phase_years or [0.50, 0.80, 1.00]
        while len(phases) < 3:
            phases.append(1.0)
        phases = phases[:3]
        rows = []
        for i, yr in enumerate(years):
            pct = phases[i]
            run_rate = a.pre_tax_synergies * pct
            prob_adj = run_rate * a.synergy_probability
            cost = a.synergy_implementation_costs if i == 0 else 0.0
            rows.append({
                "Year": yr,
                "Realization %": pct,
                "Pre-tax run-rate synergies ($M)": run_rate,
                "Probability-adjusted ($M)": prob_adj,
                "Implementation costs ($M)": cost,
            })
        return pd.DataFrame(rows)

    # ── Institutional upgrade: buyer returns ─────────────────────────────────
    def _buyer_returns(self, a: MnAAssumptions, m: Dict[str, float]) -> Dict[str, float]:
        """Approximate acquirer-shareholder returns over the hold period from
        pro forma EPS growth (EPS accretion + organic growth)."""
        hold = max(a.buyer_hold_years, 1)
        if a.acquirer_price <= 0 or m["pro_forma_eps"] <= 0:
            return {"irr_pct": 0.0, "moic": 1.0, "terminal_eps": 0.0, "note": ""}
        organic_growth = 0.05  # placeholder for organic EPS growth (analyst-editable)
        terminal_eps = m["pro_forma_eps"] * (1 + organic_growth) ** hold
        terminal_price = a.acquirer_price * (terminal_eps / a.acquirer_eps) if a.acquirer_eps > 0 else a.acquirer_price
        moic = terminal_price / a.acquirer_price if a.acquirer_price > 0 else 1.0
        irr = moic ** (1 / hold) - 1
        return {
            "irr_pct": irr,
            "moic": moic,
            "hold_years": hold,
            "terminal_eps_estimate": terminal_eps,
            "note": "Acquirer-shareholder IRR estimate from pro forma EPS compounding; "
                    "organic growth assumed at 5% (edit to refine).",
        }

    # ── Institutional upgrade: seller proceeds ───────────────────────────────
    def _seller_proceeds(self, a: MnAAssumptions, m: Dict[str, float]) -> Dict[str, float]:
        total = m["total_deal_value"]
        return {
            "cash_proceeds": total * a.percent_cash,
            "stock_proceeds": total * a.percent_stock,
            "expected_earnout": a.earnout_value * a.earnout_probability,
            "total_expected_proceeds": total + a.earnout_value * a.earnout_probability,
            "price_per_share": m["offer_price"],
        }

    # ── Institutional upgrade: max purchase price ────────────────────────────
    def _max_purchase_price(self, a: MnAAssumptions) -> float:
        """Walk-away / value-neutral price: the offer that leaves the deal
        exactly EPS-neutral (0% accretion) at the given synergy level."""
        try:
            # Binary search the offer price that leaves accretion ≈ 0 at the
            # modeled synergy level. PF NI = acq_ni + tgt_ni + syn_at − int_at −
            # earnout_at; the stock portion grows shares 1-for-1 with price.
            acq_eps = a.acquirer_eps
            if a.target_price <= 0 or acq_eps <= 0:
                return 0.0
            low, high = a.target_price * 0.5, a.target_price * 4.0
            for _ in range(120):
                mid = (low + high) / 2
                test = MnAAssumptions(**{**a.__dict__})
                test.offer_premium = (mid / a.target_price) - 1
                acc = self._accretion_dilution(test)["accretion_pct"]
                if acc > 0:
                    low = mid
                else:
                    high = mid
            return max(round((low + high) / 2, 2), 0.0)
        except Exception:
            return 0.0

    # ── Institutional upgrade: value creation bridge ─────────────────────────
    def _value_bridge(self, a: MnAAssumptions, m: Dict[str, float]) -> Dict[str, float]:
        """Decompose value created/(destroyed) for acquirer shareholders."""
        acq_eq = a.acquirer_price * a.acquirer_shares
        if acq_eq <= 0:
            return {"standalone_value": 0.0, "synergy_value": 0.0,
                    "financing_impact": 0.0, "net_value_creation": 0.0}
        standalone_value = acq_eq
        # Value of synergies (capitalized at acquirer P/E)
        acq_pe = a.acquirer_price / a.acquirer_eps if a.acquirer_eps > 0 else 1.0
        synergy_value = m["after_tax_synergies"] * acq_pe * a.synergy_probability
        # Financing impact: interest drag + fees, capitalized
        financing_impact = -(m["after_tax_interest"] + m["total_fees"]) * acq_pe
        integration_impact = -m["after_tax_integration"] * acq_pe
        net = standalone_value + synergy_value + financing_impact + integration_impact
        return {
            "standalone_value": standalone_value,
            "synergy_value": synergy_value,
            "financing_impact": financing_impact,
            "integration_impact": integration_impact,
            "net_value_creation": net - standalone_value,
        }

    # ── Institutional upgrade: FCF accretion ─────────────────────────────────
    def _fcf_accretion(self, a: MnAAssumptions, m: Dict[str, float]) -> Dict[str, float]:
        acq_fcf = a.acquirer_fcf or (a.acquirer_net_income + a.acquirer_ebitda * 0) or 0
        tgt_fcf = a.target_fcf or 0
        # If FCF not supplied, estimate from EBITDA with a cash-conversion proxy
        if acq_fcf <= 0 and a.acquirer_ebitda > 0:
            acq_fcf = a.acquirer_ebitda * 0.6
        if tgt_fcf <= 0 and a.target_ebitda > 0:
            tgt_fcf = a.target_ebitda * 0.6
        syn_fcf = a.pre_tax_synergies * (1 - a.tax_rate) * a.synergy_probability
        pf_fcf = acq_fcf + tgt_fcf + syn_fcf - m["after_tax_interest"] - m["total_fees"]
        pf_fcf_ps = pf_fcf / max(m["pro_forma_shares"], 1e-9)
        acq_fcf_ps = acq_fcf / max(a.acquirer_shares, 1e-9)
        accretion = (pf_fcf_ps / acq_fcf_ps - 1) if acq_fcf_ps > 0 else 0.0
        return {
            "acquirer_fcf": acq_fcf, "target_fcf": tgt_fcf, "synergy_fcf": syn_fcf,
            "pro_forma_fcf": pf_fcf, "pro_forma_fcf_per_share": pf_fcf_ps,
            "fcf_accretion_pct": accretion,
            "note": "FCF proxies estimated at 60% EBITDA conversion when actual FCF not supplied.",
        }

    # ── Institutional upgrade: transaction scenarios ─────────────────────────
    def _transaction_scenarios(self, a: MnAAssumptions) -> pd.DataFrame:
        rows = []
        scenarios = [
            ("Base", 0.5, 0.0, 0.0, 1.0),
            ("Synergy over-delivery", 0.2, 0.0, 0.3, 1.0),
            ("Synergy miss", 0.15, 0.0, -0.5, 0.7),
            ("Financing cost shock", 0.15, 0.02, 0.0, 1.0),
        ]
        for label, prob, rate_delta, syn_mult, syn_prob in scenarios:
            mod = MnAAssumptions(**{**a.__dict__,
                                    "cost_of_debt": a.cost_of_debt + rate_delta,
                                    "pre_tax_synergies": max(a.pre_tax_synergies * (1 + syn_mult), 0),
                                    "synergy_probability": syn_prob})
            m = self._accretion_dilution(mod)
            rows.append({
                "Scenario": label, "Probability": prob,
                "Accretion/(Dilution) %": m["accretion_pct"],
                "Pro Forma EPS": m["pro_forma_eps"],
                "Synergy Level ($M)": mod.pre_tax_synergies,
                "Cost of Debt": mod.cost_of_debt,
            })
        return pd.DataFrame(rows)

    def run_mna(self, a: MnAAssumptions) -> MnAResult:
        m = self._accretion_dilution(a)
        contribution = self._contribution_analysis(a, m)
        sensitivity = self._sensitivity(a)

        deal_metrics = {
            "offer_premium": a.offer_premium,
            "offer_pe": (m["offer_price"] / a.target_eps) if a.target_eps > 0 else float("nan"),
            "target_pe": (a.target_price / a.target_eps) if a.target_eps > 0 else float("nan"),
            "acquirer_pe": (a.acquirer_price / a.acquirer_eps) if a.acquirer_eps > 0 else float("nan"),
            "implied_ev": m["total_deal_value"] + max(a.target_net_debt, 0.0),
            "fees_pct_of_deal": (m["total_fees"] / m["total_deal_value"]) if m["total_deal_value"] else 0.0,
            "percent_stock": a.percent_stock,
            "percent_cash": a.percent_cash,
            "new_shares_pct": (m["new_shares_issued"] / m["pro_forma_shares"]) if m["pro_forma_shares"] else 0.0,
        }

        conclusion = self._generate_conclusion(a, m, contribution, deal_metrics)

        return MnAResult(
            assumptions=a,
            offer_price=m["offer_price"],
            total_deal_value=m["total_deal_value"],
            total_uses=m["total_uses"],
            total_sources=m["total_sources"],
            sources_uses_check=m["sources_uses_check"],
            new_shares_issued=m["new_shares_issued"],
            pro_forma_shares=m["pro_forma_shares"],
            pro_forma_net_income=m["pro_forma_net_income"],
            pro_forma_eps=m["pro_forma_eps"],
            accretion_dilution_dollar=m["accretion_dollar"],
            accretion_dilution_pct=m["accretion_pct"],
            is_accretive=(m["accretion_dollar"] > 0),
            contribution_analysis=contribution,
            sensitivity=sensitivity,
            deal_metrics=deal_metrics,
            conclusion=conclusion,
            # ── Institutional upgrades ──
            earnout_metrics=self._earnout_metrics(a, m),
            synergy_schedule=self._synergy_schedule(a),
            buyer_returns=self._buyer_returns(a, m),
            seller_proceeds=self._seller_proceeds(a, m),
            max_purchase_price=self._max_purchase_price(a),
            value_bridge=self._value_bridge(a, m),
            fcf_accretion=self._fcf_accretion(a, m),
            transaction_scenarios=self._transaction_scenarios(a),
        )

    def _generate_conclusion(self, a, m, contribution, dm) -> str:
        lines = []
        ad_pct = m["accretion_pct"]
        pf_eps = m["pro_forma_eps"]
        eps0 = a.acquirer_eps

        # ── Layered narrative ────────────────────────────────────────────────
        # Layer 0: verdict
        if ad_pct > 0.02:
            verdict = "PROCEED"
            verdict_note = ("the deal is immediately accretive and should be well-received by the market; "
                            "move to board approval and shareholder vote preparation")
        elif ad_pct > -0.02:
            verdict = "PROCEED WITH CAUTION (RENEGOTIATE TERMS)"
            verdict_note = ("near-breakeven on accretion — market reaction will hinge on the strategic "
                            "narrative and the credibility of the modeled revenue synergies")
        else:
            verdict = "PASS OR RENEGOTIATE"
            verdict_note = ("significant dilution raises execution risk — push the premium down, shift "
                            "consideration toward cash, or find a larger synergy pool before proceeding")

        lines.append(f"**Recommendation: {verdict}**")
        lines.append(
            f"{a.acquirer_ticker} buying {a.target_ticker} at a {a.offer_premium*100:.0f}% premium "
            f"(${m['offer_price']:,.2f}/share, ${m['total_deal_value']:,.0f}M total) is **{ad_pct*100:+.1f}% "
            f"{'accretive' if ad_pct > 0 else 'dilutive'}** to acquirer EPS on a pro forma basis "
            f"(${pf_eps:.2f} vs ${eps0:.2f} standalone). In plain terms: {verdict_note}."
        )

        # Layer 1: deal structure & financing
        lines.append(f"\n**Deal Structure & Financing:**")
        lines.append(
            f"  - Stock consideration: ${m['equity_consideration']:,.0f}M ({a.percent_stock*100:.0f}% of deal) — "
            f"{m['new_shares_issued']:,.0f}M new shares ({dm['new_shares_pct']*100:.1f}% dilution to existing holders)"
        )
        lines.append(
            f"  - Cash consideration: ${m['cash_needed']:,.0f}M ({a.percent_cash*100:.0f}% of deal) — "
            f"financed at {a.cost_of_debt*100:.1f}% pre-tax cost of debt "
            f"(${m['after_tax_interest']:,.0f}M annual after-tax interest)"
        )
        lines.append(
            f"  - Transaction fees: ${m['total_fees']:,.0f}M "
            f"({dm['fees_pct_of_deal']*100:.1f}% of deal value; advisory {a.advisory_fee_pct*100:.0f}bps, "
            f"financing {a.financing_fee_pct*100:.0f}bps of debt raised)"
        )
        if m.get("expected_earnout", 0) > 0:
            lines.append(
                f"  - Earnout: ${a.earnout_value:,.0f}M contingent consideration at "
                f"{a.earnout_probability:.0%} probability (expected value ${m['expected_earnout']:,.0f}M)"
            )
        lines.append(f"  - Sources = Uses check: **${m['sources_uses_check']:,.0f}M** (Δ = 0 = balanced)")

        # Layer 2: contribution analysis (the EPS walk)
        lines.append(f"\n**Contribution Analysis (EPS Bridge):**")
        for _, r in contribution.iterrows():
            label = r["Driver"]
            val = r["EPS Impact ($)"]
            if r["Type"] == "Result":
                lines.append(f"  - **{label}: ${val:+.2f}**")
            else:
                lines.append(f"  - {label}: ${val:+.2f}")

        # Layer 3: synergy dependency
        lines.append(f"\n**Synergy Analysis:**")
        lines.append(
            f"  - Pre-tax synergies assumed: ${a.pre_tax_synergies:,.0f}M (after-tax: ${m['after_tax_synergies']:,.0f}M)"
        )
        if a.synergy_probability < 1.0:
            lines.append(f"  - Synergy realization probability: {a.synergy_probability:.0%}")
        if a.pre_tax_synergies > 0:
            if m["after_tax_synergies"] / pf_eps < -m["accretion_dollar"]:
                lines.append(
                    "  - **Without synergies, the deal would be dilutive.** The accretive outcome depends "
                    "entirely on achieving the projected synergies."
                )
            else:
                lines.append(
                    "  - The deal is accretive even without synergies. Synergies provide additional upside "
                    "but are not required to justify the transaction."
                )
        if sensitivity_breakeven := self._breakeven_str(a, m):
            lines.append(f"  - **Breakeven pre-tax synergies: {sensitivity_breakeven}**")
        if self._max_purchase_price(a) > 0:
            lines.append(
                f"  - **Maximum (EPS-neutral) purchase price: ${self._max_purchase_price(a):,.2f}/share** "
                f"(at modeled synergies; above this level the deal turns dilutive)"
            )

        # Layer 4: valuation context
        lines.append(
            f"\n**Valuation Context:** Offering {dm['offer_pe']:.1f}x target earnings vs "
            f"{dm['target_pe']:.1f}x pre-deal (acquirer trades at {dm['acquirer_pe']:.1f}x). "
            f"The {a.offer_premium*100:.0f}% premium is "
            f"{'within' if a.offer_premium <= 0.35 else 'above'} the typical 20-35% range for strategic acquisitions."
        )

        # Layer 5: value creation
        vb = self._value_bridge(a, m)
        if vb.get("net_value_creation", 0) != 0:
            lines.append(
                f"\n**Value Creation Bridge:** standalone acquirer value ${vb['standalone_value']:,.0f}M "
                f"+ synergies ${vb['synergy_value']:,.0f}M − financing/integration drag "
                f"${-vb['financing_impact'] - vb['integration_impact']:,.0f}M = "
                f"**{vb['net_value_creation']:+,.0f}M net value creation** for acquirer shareholders."
            )

        # Layer 6: risks in plain English (novice-friendly)
        risks = []
        if a.pre_tax_synergies > 0 and m["after_tax_synergies"] / pf_eps < -m["accretion_dollar"]:
            risks.append("Without the projected synergies, the deal turns dilutive — the accretion is entirely "
                         "dependent on hitting the synergy targets.")
        if dm["new_shares_pct"] > 0.10:
            risks.append("Shareholders face meaningful dilution (over 10% of the pro forma share count is new stock) "
                         "— watch for stock-price pressure after announcement.")
        if a.offer_premium > 0.35:
            risks.append("The premium is above the typical 20-35% range — the market may view the price as rich, "
                         "and an overpayment leaves less room for integration slips.")
        if m["total_fees"] > 0 and dm["fees_pct_of_deal"] > 0.02:
            risks.append(f"Transaction fees of ${m['total_fees']:,.0f}M ({dm['fees_pct_of_deal']*100:.1f}% of deal) "
                         "are high — they directly reduce accretion and should be scrutinized.")
        if a.earnout_value > 0 and a.earnout_probability < 1:
            risks.append("Contingent consideration creates post-close negotiation/accounting complexity; "
                         "earnout targets may drive short-term management behavior.")
        if not risks:
            risks.append("The main exposure is integration execution: realizing the modeled cost saves and "
                         "keeping key target talent and customers.")
        lines.append("\n**Key Risks (Plain English):**")
        for r in risks:
            lines.append(f"  - {r}")

        return "\n".join(lines)

    def _breakeven_str(self, a: MnAAssumptions, m: Dict[str, float]) -> str:
        """Human-readable breakeven synergies from the sensitivity run."""
        try:
            be = self._sensitivity(a).breakeven_synergies
            if be is None:
                return ""
            if be <= 0:
                return "none required (accretive without synergies)"
            return f"${be:,.0f}M pre-tax"
        except Exception:
            return ""


_mna_engine = None


def get_mna_engine() -> InstitutionalMnAEngine:
    global _mna_engine
    if _mna_engine is None:
        _mna_engine = InstitutionalMnAEngine()
    return _mna_engine
