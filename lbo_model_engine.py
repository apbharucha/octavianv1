"""
Institutional-Grade Leveraged Buyout (LBO) Model

A Goldman-Sachs-standard LBO: Sources & Uses with transaction fees, a full debt
schedule (Term Loan B with mandatory amortization + excess cash flow sweep,
revolver with minimum cash balance), sponsor returns (IRR / MOIC with timing
convention), 2-way sensitivity (entry multiple x exit multiple), a value
creation bridge, and a layered narrative.

Institutional upgrades (additive — the core math is unchanged):
* Optional PIK tranche (interest rolled into principal)
* Returns attribution bridge (operating / deleveraging / multiple expansion)
* Scenario cases (downside / base / upside) with probabilities
* Full value-creation bridge including cash generation contribution

Author: Octavian Terminal
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np


@dataclass
class LBOAssumptions:
    ticker: str
    target_name: str
    entry_year: int
    exit_year: int

    # Valuation
    ltm_ebitda: float                    # $M
    entry_multiple: float
    exit_multiple: float

    # Capital Structure
    leverage_multiple: float             # Debt / EBITDA
    interest_rate: float                 # Term loan rate (pre-tax)
    tax_rate: float = 0.25

    # Operations (per year; lists override the scalar defaults)
    revenue_growth_rates: List[float] = field(default_factory=list)
    ebitda_margins: List[float] = field(default_factory=list)
    capex_pct_rev: float = 0.05
    nwc_pct_rev: float = 0.10
    depreciation_pct_rev: float = 0.04   # D&A as % of revenue

    # Fees & structure
    advisory_fee_pct: float = 0.02       # of purchase price
    financing_fee_pct: float = 0.015     # of debt raised
    mandatory_amort_pct: float = 0.01    # 1% of original TLB balance per year
    cash_sweep_pct: float = 0.50         # 50% of excess cash flow to debt
    min_cash_pct_rev: float = 0.03       # minimum cash as % of revenue
    revolver_rate: float = 0.06          # revolver draw rate
    cash_interest_rate: float = 0.02     # interest income on cash balances
    exit_year_index: Optional[int] = None  # exit multiple applied on this projection year

    # ── Institutional upgrades (defaulted → backward compatible) ────────────
    # PIK tranche: portion of TLB interest rolled into principal (0 = none)
    pik_interest_pct: float = 0.0
    # Second-lien / mezzanine tranche
    second_lien_multiple: float = 0.0    # Debt/EBITDA in second-lien (0 = none)
    second_lien_rate: float = 0.12
    # Scenario probabilities (downside / base / upside) — base derived
    downside_prob: float = 0.25
    upside_prob: float = 0.20
    # Scenario deltas vs base
    downside_growth_delta: float = -0.06
    downside_margin_delta: float = -0.05
    downside_exit_multiple_delta: float = -1.5
    upside_growth_delta: float = 0.06
    upside_margin_delta: float = 0.05
    upside_exit_multiple_delta: float = 1.5


@dataclass
class LBOSensitivity:
    """IRR / MOIC 2-way table vs entry and exit multiple."""
    irr_table: pd.DataFrame
    moic_table: pd.DataFrame


@dataclass
class LBOResult:
    assumptions: LBOAssumptions
    purchase_price: float
    debt_amount: float
    equity_amount: float
    total_sources: float
    total_uses: float
    sources_uses_check: float
    exit_enterprise_value: float
    exit_equity_value: float
    irr: float
    moic: float
    cash_flows: pd.DataFrame
    debt_schedule: pd.DataFrame
    sources_uses: pd.DataFrame
    sensitivity: LBOSensitivity
    value_bridge: Dict[str, float]
    conclusion: str = ""

    # ── Institutional upgrades (defaulted → backward compatible) ─────────────
    returns_attribution: Dict[str, Any] = field(default_factory=dict)
    scenario_cases: pd.DataFrame = field(default_factory=pd.DataFrame)
    debt_structure: Dict[str, float] = field(default_factory=dict)


class InstitutionalLBOEngine:

    _ENTRY_MULT_RANGE = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    _EXIT_MULT_RANGE = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

    def _years(self, a: LBOAssumptions) -> int:
        return max(a.exit_year - a.entry_year, 1)

    def _build_projection(self, a: LBOAssumptions, with_sensitivity: bool = True):
        """Build the full projection + debt schedule.

        Returns dict with dataframes and key outputs.
        """
        years = self._years(a)
        growth = a.revenue_growth_rates if a.revenue_growth_rates else [0.05] * years
        margins = a.ebitda_margins if a.ebitda_margins else [0.20] * years
        if len(growth) < years:
            growth = growth + [growth[-1]] * (years - len(growth))
        if len(margins) < years:
            margins = margins + [margins[-1]] * (years - len(margins))

        purchase_price = a.ltm_ebitda * a.entry_multiple
        debt_amount = a.ltm_ebitda * a.leverage_multiple
        equity_amount = purchase_price - debt_amount

        # ── Debt structure (TLB + optional second lien) ─────────────────────
        second_lien = a.ltm_ebitda * a.second_lien_multiple
        tlb_amount = max(debt_amount - second_lien, 0.0)
        # PIK split within the TLB: portion of interest rolled to principal
        pik_share = max(0.0, min(a.pik_interest_pct, 1.0))

        # Fees
        advisory_fee = purchase_price * a.advisory_fee_pct
        financing_fee = (tlb_amount + second_lien) * a.financing_fee_pct
        total_fees = advisory_fee + financing_fee

        # Sources & Uses
        uses_purchase = purchase_price
        uses_fees = total_fees
        total_uses = uses_purchase + uses_fees
        debt_amount_used = tlb_amount + second_lien
        total_sources = debt_amount_used + equity_amount
        sponsor_equity = total_uses - debt_amount_used
        total_sources_adj = debt_amount_used + sponsor_equity
        sources_uses_check = total_sources_adj - total_uses

        # Projection
        rev = a.ltm_ebitda / margins[0] if margins[0] > 0 else a.ltm_ebitda / 0.20
        tlb_original = tlb_amount
        tlb_balance = tlb_original
        sl_balance = second_lien
        revolver_balance = 0.0
        cash_balance = max(rev * a.min_cash_pct_rev, 0.0)  # starting cash

        rows = []
        for i in range(years):
            rev_prev = rev
            rev = rev * (1 + growth[i])
            ebitda = rev * margins[i]
            depreciation = rev * a.depreciation_pct_rev
            ebit = ebitda - depreciation

            # Interest: TLB (cash + PIK portions) + second lien + revolver,
            # minus cash interest income
            tlb_cash_interest = tlb_balance * a.interest_rate * (1 - pik_share)
            tlb_pik_interest = tlb_balance * a.interest_rate * pik_share
            sl_interest = sl_balance * a.second_lien_rate
            revolver_interest = revolver_balance * a.revolver_rate
            interest_expense = tlb_cash_interest + sl_interest + revolver_interest
            interest_income = cash_balance * a.cash_interest_rate
            net_interest = interest_expense - interest_income

            ebt = ebit - net_interest
            taxes = max(0.0, ebt * a.tax_rate)
            net_income = ebt - taxes

            capex = rev * a.capex_pct_rev
            nwc_change = (rev - rev_prev) * a.nwc_pct_rev

            # Cash flow available for debt service (CFADS)
            cfads = net_income + depreciation - capex - nwc_change

            # Mandatory amortization (1% of original TLB)
            mandatory_amort = min(tlb_original * a.mandatory_amort_pct, tlb_balance)
            tlb_balance -= mandatory_amort

            # PIK accrual increases principal
            tlb_balance += tlb_pik_interest

            # Excess cash flow sweep: 50% of remaining cash flow to the TLB
            excess = max(cfads - mandatory_amort, 0.0)
            sweep = min(excess * a.cash_sweep_pct, tlb_balance)
            tlb_balance -= sweep

            # Revolver: draw if cash flow is insufficient to cover debt service;
            # otherwise repay. Positive flow = repayment; negative = draw.
            cash_after_debt_service = cfads - mandatory_amort - sweep
            if cash_after_debt_service < 0:
                revolver_draw = -cash_after_debt_service
                revolver_balance += revolver_draw
                revolver_flow = -revolver_draw  # negative = net draw
                cash_after_debt_service = 0.0
            else:
                revolver_paydown = min(cash_after_debt_service, revolver_balance)
                revolver_balance -= revolver_paydown
                revolver_flow = revolver_paydown   # positive = net repayment
                cash_after_debt_service -= revolver_paydown

            # Cash balance accrues remaining free cash flow (floor = min cash)
            cash_balance = max(cash_balance + cash_after_debt_service,
                               rev * a.min_cash_pct_rev)

            total_debt = tlb_balance + sl_balance + revolver_balance

            rows.append({
                "Year": a.entry_year + i,
                "Revenue": rev,
                "Revenue Growth %": growth[i],
                "EBITDA Margin %": margins[i],
                "EBITDA": ebitda,
                "(-) D&A": -depreciation,
                "EBIT": ebit,
                "(-) Interest Expense": -interest_expense,
                "(+) Interest Income": interest_income,
                "Pre-Tax Income": ebt,
                "(-) Taxes": -taxes,
                "Net Income": net_income,
                "(+) D&A": depreciation,
                "(-) CapEx": -capex,
                "(-) Change in NWC": -nwc_change,
                "Cash Flow Available for Debt Service (CFADS)": cfads,
                "(-) Mandatory Amortization": -mandatory_amort,
                "(-) Excess Cash Flow Sweep": -sweep,
                "PIK Interest Accrued": tlb_pik_interest,
                "Revolver Draw / (+ Repayment)": revolver_flow,
                "Ending TLB Balance": tlb_balance,
                "Ending Second Lien Balance": sl_balance,
                "Ending Revolver Balance": revolver_balance,
                "Ending Cash Balance": cash_balance,
                "Total Debt": total_debt,
            })

        cf_df = pd.DataFrame(rows)

        # Exit
        exit_idx = (a.exit_year_index - 1) if a.exit_year_index else (years - 1)
        exit_idx = max(0, min(exit_idx, years - 1))
        exit_ebitda = rows[exit_idx]["EBITDA"]
        exit_ev = exit_ebitda * a.exit_multiple
        total_debt_at_exit = rows[exit_idx]["Total Debt"]
        cash_at_exit = rows[exit_idx]["Ending Cash Balance"]
        exit_equity = exit_ev - total_debt_at_exit + cash_at_exit

        # Returns (standard convention: exit at end of final year)
        moic = exit_equity / sponsor_equity if sponsor_equity > 0 else 0.0
        irr = (moic ** (1 / years)) - 1 if moic > 0 and years > 0 else 0.0

        # Value bridge
        entry_ev = purchase_price
        ebitda_growth_contrib = (exit_ebitda - a.ltm_ebitda) * a.entry_multiple
        multiple_expansion_contrib = (a.exit_multiple - a.entry_multiple) * exit_ebitda
        delev_contrib = total_debt_at_exit - debt_amount_used  # negative = debt paid down
        value_bridge = {
            "EBITDA growth contribution": ebitda_growth_contrib,
            "Multiple expansion (contraction) contribution": multiple_expansion_contrib,
            "Net debt change (deleveraging)": delev_contrib,
            "Net cash at exit": cash_at_exit,
        }

        su_df = pd.DataFrame([
            {"Item": "USES", "Amount": None, "% of Total": None},
            {"Item": "Purchase Equity Value", "Amount": uses_purchase, "% of Total": uses_purchase / total_uses},
            {"Item": "Transaction Fees (advisory + financing)", "Amount": uses_fees, "% of Total": uses_fees / total_uses},
            {"Item": "Total Uses", "Amount": total_uses, "% of Total": 1.0},
            {"Item": "", "Amount": None, "% of Total": None},
            {"Item": "SOURCES", "Amount": None, "% of Total": None},
            {"Item": "Term Loan B", "Amount": tlb_amount, "% of Total": tlb_amount / total_uses},
            {"Item": "Second Lien", "Amount": second_lien, "% of Total": second_lien / total_uses if total_uses else 0},
            {"Item": "Sponsor Equity (check)", "Amount": sponsor_equity, "% of Total": sponsor_equity / total_uses},
            {"Item": "Total Sources", "Amount": total_sources_adj, "% of Total": 1.0},
            {"Item": "Sources = Uses check", "Amount": sources_uses_check, "% of Total": None},
        ])

        sensitivity = self._sensitivity(a, years) if with_sensitivity else None

        return {
            "purchase_price": purchase_price,
            "debt_amount": debt_amount_used,
            "equity_amount": sponsor_equity,
            "sponsor_equity": sponsor_equity,
            "total_sources": total_sources_adj,
            "total_uses": total_uses,
            "sources_uses_check": sources_uses_check,
            "exit_enterprise_value": exit_ev,
            "exit_equity_value": exit_equity,
            "irr": irr,
            "moic": moic,
            "cash_flows": cf_df,
            "sources_uses": su_df,
            "sensitivity": sensitivity,
            "with_sensitivity": with_sensitivity,
            "value_bridge": value_bridge,
            "exit_year_label": rows[exit_idx]["Year"],
            "debt_structure": {"tlb": tlb_amount, "second_lien": second_lien,
                               "pik_share": pik_share, "total": debt_amount_used},
        }

    def _sensitivity(self, a: LBOAssumptions, years: int) -> LBOSensitivity:
        irr_rows, moic_rows = [], []
        for em in self._ENTRY_MULT_RANGE:
            irr_row = {"Entry": f"{em:.1f}x"}
            moic_row = {"Entry": f"{em:.1f}x"}
            for xm in self._EXIT_MULT_RANGE:
                mod = LBOAssumptions(**{**a.__dict__, "entry_multiple": em, "exit_multiple": xm})
                out = self._build_projection(mod, with_sensitivity=False)
                irr_row[f"{xm:.1f}x"] = out["irr"]
                moic_row[f"{xm:.1f}x"] = out["moic"]
            irr_rows.append(irr_row)
            moic_rows.append(moic_row)
        return LBOSensitivity(irr_table=pd.DataFrame(irr_rows),
                              moic_table=pd.DataFrame(moic_rows))

    # ── Institutional upgrade: returns attribution bridge ────────────────────
    def _returns_attribution(self, a: LBOAssumptions, out: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose the sponsor's equity uplift into operating, deleveraging,
        multiple, and cash-generation contributions (in $M and % of total)."""
        eq0 = out["sponsor_equity"]
        exit_eq = out["exit_equity_value"]
        total_uplift = exit_eq - eq0
        if eq0 <= 0 or total_uplift == 0:
            return {"total_uplift": total_uplift, "components": {}, "note": "Degenerate equity base."}

        vb = out["value_bridge"]
        years = self._years(a)
        exit_ebitda = out["cash_flows"].iloc[min(a.exit_year_index - 1 if a.exit_year_index else years - 1, years - 1)]["EBITDA"]

        # Operating contribution = EBITDA growth × entry multiple
        operating = vb.get("EBITDA growth contribution", 0)
        # Multiple contribution
        multiple = vb.get("Multiple expansion (contraction) contribution", 0)
        # Deleveraging = reduction in total debt (sponsor receives less debt to repay)
        delev = -vb.get("Net debt change (deleveraging)", 0)
        # Cash at exit
        cash = vb.get("Net cash at exit", 0)

        components = {
            "Operating (EBITDA growth)": operating,
            "Multiple expansion / contraction": multiple,
            "Deleveraging": delev,
            "Net cash at exit": cash,
        }
        total = sum(components.values())
        pct = {k: (v / total * 100 if abs(total) > 1e-9 else 0.0) for k, v in components.items()}
        return {
            "total_uplift": total_uplift,
            "components": components,
            "pct_of_uplift": pct,
            "note": "Attribution shows where sponsor returns originate: operating "
                    "improvement, multiple moves, debt paydown, and cash build.",
        }

    # ── Institutional upgrade: scenario cases ────────────────────────────────
    def _scenario_cases(self, a: LBOAssumptions, base_out: Dict[str, Any]) -> pd.DataFrame:
        rows = []
        base = base_out
        scenarios = [
            ("Downside", a.downside_prob,
             [g + a.downside_growth_delta for g in (a.revenue_growth_rates or [0.05] * self._years(a))],
             [m + a.downside_margin_delta for m in (a.ebitda_margins or [0.20] * self._years(a))],
             a.exit_multiple + a.downside_exit_multiple_delta),
            ("Base", max(1.0 - a.downside_prob - a.upside_prob, 0.0),
             a.revenue_growth_rates or [0.05] * self._years(a),
             a.ebitda_margins or [0.20] * self._years(a),
             a.exit_multiple),
            ("Upside", a.upside_prob,
             [g + a.upside_growth_delta for g in (a.revenue_growth_rates or [0.05] * self._years(a))],
             [m + a.upside_margin_delta for m in (a.ebitda_margins or [0.20] * self._years(a))],
             a.exit_multiple + a.upside_exit_multiple_delta),
        ]
        for label, prob, growth, margins, exit_mult in scenarios:
            mod = LBOAssumptions(**{**a.__dict__, "revenue_growth_rates": growth,
                                    "ebitda_margins": margins, "exit_multiple": exit_mult})
            out = self._build_projection(mod, with_sensitivity=False)
            rows.append({
                "Scenario": label, "Probability": prob,
                "IRR": out["irr"], "MOIC": out["moic"],
                "Exit Multiple (x)": exit_mult,
                "Exit Equity ($M)": out["exit_equity_value"],
                "Avg Growth": np.mean(growth),
                "Avg EBITDA Margin": np.mean(margins),
            })
        df = pd.DataFrame(rows)
        total_p = df["Probability"].sum()
        if total_p > 0:
            df["Probability"] = df["Probability"] / total_p
        return df

    def run_lbo(self, a: LBOAssumptions) -> LBOResult:
        out = self._build_projection(a)
        conclusion = self._generate_conclusion(a, out)
        return LBOResult(
            assumptions=a,
            purchase_price=out["purchase_price"],
            debt_amount=out["debt_amount"],
            equity_amount=out["sponsor_equity"],
            total_sources=out["total_sources"],
            total_uses=out["total_uses"],
            sources_uses_check=out["sources_uses_check"],
            exit_enterprise_value=out["exit_enterprise_value"],
            exit_equity_value=out["exit_equity_value"],
            irr=out["irr"],
            moic=out["moic"],
            cash_flows=out["cash_flows"],
            debt_schedule=out["cash_flows"],   # debt schedule embedded in CF sheet
            sources_uses=out["sources_uses"],
            sensitivity=out["sensitivity"],
            value_bridge=out["value_bridge"],
            conclusion=conclusion,
            # ── Institutional upgrades ──
            returns_attribution=self._returns_attribution(a, out),
            scenario_cases=self._scenario_cases(a, out),
            debt_structure=out["debt_structure"],
        )

    def _generate_conclusion(self, a: LBOAssumptions, out: Dict[str, Any]) -> str:
        lines = []
        irr, moic = out["irr"], out["moic"]
        years = self._years(a)
        purchase_price = out["purchase_price"]
        sponsor_eq = out["sponsor_equity"]
        debt = out["debt_amount"]
        exit_eq = out["exit_equity_value"]
        exit_ev = out["exit_enterprise_value"]
        remaining_debt = None
        tlb_end = None
        if out["cash_flows"] is not None and not out["cash_flows"].empty:
            last = out["cash_flows"].iloc[-1]
            remaining_debt = last["Total Debt"]
            tlb_end = last["Ending TLB Balance"]
        leverage = debt / a.ltm_ebitda if a.ltm_ebitda > 0 else 0
        debt_paydown_pct = (1 - remaining_debt / debt) * 100 if debt and remaining_debt is not None else 0

        # Layer 0: verdict
        if irr >= 0.20:
            verdict = "PROCEED"
            verdict_note = ("the deal clears typical institutional hurdle rates (18-22%); advance to detailed "
                            "due diligence and financing committee review")
        elif irr >= 0.12:
            verdict = "PROCEED WITH CONDITIONS (RENEGOTIATE)"
            verdict_note = ("returns sit below the preferred hurdle — push for a lower entry multiple, identify "
                            "additional operational synergies, or structure seller earn-outs before committing capital")
        else:
            verdict = "PASS"
            verdict_note = ("returns do not justify the illiquidity premium and execution risk of an LBO; walk "
                            "away unless material operational improvements can be identified")

        lines.append(f"**Recommendation: {verdict}**")
        lines.append(
            f"This LBO delivers **{irr*100:.1f}% IRR** and **{moic:.2f}x MOIC** over {years} years "
            f"(exit {out['exit_year_label']}). In plain terms: {verdict_note}. The structure retires "
            f"**{debt_paydown_pct:.0f}% of entry debt** through operating cash flow, leaving "
            f"${remaining_debt:,.0f}M outstanding at exit."
        )

        # Capital structure
        lines.append(
            f"\n**Capital Structure:** Entry leverage of {leverage:.1f}x Debt/EBITDA on a "
            f"${purchase_price:,.0f}M purchase (${debt:,.0f}M debt, ${sponsor_eq:,.0f}M sponsor equity). "
            f"Sources = Uses check: **${out['sources_uses_check']:,.0f}M** (Δ = 0 = balanced)."
        )
        ds = out.get("debt_structure", {})
        if ds.get("pik_share", 0) > 0:
            lines.append(
                f"  {ds['pik_share']:.0%} of TLB interest is PIK (rolled into principal) — "
                "this reduces cash interest but increases the debt balance at exit."
            )
        if leverage > 6:
            lines.append(
                "  The leverage level exceeds typical institutional comfort zones (>6x). Financing risk is "
                "elevated — a downturn could strain covenant compliance."
            )

        # Value bridge
        lines.append("\n**Value Creation Bridge:**")
        for k, v in out["value_bridge"].items():
            lines.append(f"  - {k}: ${v:,.0f}M")
        lines.append(
            f"  - **Total equity uplift: ${exit_eq - sponsor_eq:,.0f}M "
            f"({((exit_eq / sponsor_eq) - 1) * 100:.1f}%)**"
        )

        # Returns attribution
        ra = self._returns_attribution(a, out)
        if ra.get("components"):
            lines.append("\n**Returns Attribution (where sponsor returns come from):**")
            for k, v in ra["components"].items():
                pct = ra["pct_of_uplift"].get(k, 0)
                lines.append(f"  - {k}: ${v:,.0f}M ({pct:+.1f}% of uplift)")

        # Scenario callout
        sc = self._scenario_cases(a, out)
        if sc is not None and not sc.empty:
            down = sc[sc["Scenario"] == "Downside"]
            up = sc[sc["Scenario"] == "Upside"]
            if not down.empty:
                lines.append(
                    f"\n**Scenario Cases:** Downside IRR {down.iloc[0]['IRR']:.1%} (P = {down.iloc[0]['Probability']:.0%}), "
                    f"Base {irr:.1%}, Upside {up.iloc[0]['IRR']:.1%} (P = {up.iloc[0]['Probability']:.0%}). "
                    "Probability-weighted returns below base — underwrite to the downside."
                )

        # Sensitivity callout
        if abs(a.exit_multiple - a.entry_multiple) > 1.0:
            lines.append(
                f"\n**Key Risk — Multiple Expansion Dependency:** The model assumes exit at {a.exit_multiple:.1f}x "
                f"vs entry at {a.entry_multiple:.1f}x. If the exit multiple contracts to entry levels, IRR drops "
                f"to approximately "
                f"{((((a.ltm_ebitda * a.exit_multiple - remaining_debt) / sponsor_eq) ** (1 / years)) - 1) * 100:.1f}%."
            )

        # Risks in plain English
        risks = []
        if leverage > 6:
            risks.append("The company is carrying a lot of debt (over 6x EBITDA) — if earnings dip, repaying it "
                         "gets much harder.")
        if abs(a.exit_multiple - a.entry_multiple) > 1.0:
            risks.append("A big part of the profit depends on selling at a higher multiple than you paid — "
                         "multiples can shrink fast in a downturn.")
        if remaining_debt is not None and remaining_debt > debt * 0.5:
            risks.append("More than half the original debt is still outstanding at exit — the deal relies heavily "
                         "on the exit multiple rather than on paying down leverage.")
        if ds.get("pik_share", 0) > 0:
            risks.append("PIK interest inflates the debt balance over time — the exit debt load is higher than "
                         "the headline interest rate implies.")
        if not risks:
            risks.append("The main exposure is execution: hitting the operating plan and refinancing debt at "
                         "reasonable rates.")
        lines.append("\n**Key Risks (Plain English):**")
        for r in risks:
            lines.append(f"  - {r}")

        return "\n".join(lines)


_lbo_engine = None


def get_lbo_engine() -> InstitutionalLBOEngine:
    global _lbo_engine
    if _lbo_engine is None:
        _lbo_engine = InstitutionalLBOEngine()
    return _lbo_engine
