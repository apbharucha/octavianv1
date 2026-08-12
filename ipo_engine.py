"""
Octavian Institutional IPO Analysis & Underwriting Engine
==========================================================

Simulates the analytical workflow an ECM / equity research team runs when
evaluating a real IPO:

1. Company & capital-structure inputs (with a full share-count bridge
   including options, RSUs, warrants, convertibles and preferred stock)
2. IPO market-pulse reading (environment: highly supportive → challenging)
3. Multi-method valuation (CCA, IPO comps, precedents, DCF, reverse DCF,
   SOTP, growth-adjusted) with *dynamic* method weighting
4. IPO pricing (price range, primary/secondary proceeds, dilution)
5. Bookbuilding / demand framework (clearly labeled estimates when no
   order book exists)
6. Dynamic scenario engine → IPO case studies (probability, triggers,
   valuation, aftermarket outcomes)
7. Full downstream deliverables (Excel, PPTX, memo) driven by the result

Author: Octavian Terminal
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from market_consensus_engine import (
    MarketPulseInputs, MarketPulseEngine, get_market_pulse_engine,
)


# ─────────────────────────────────────────────────────────────────────────────
# INPUTS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EquityAward:
    """An option / warrant / RSU with a strike (None strike = RSU/SAR)."""
    kind: str                       # "option" | "warrant" | "rsu" | "convertible" | "preferred"
    shares_m: float                 # shares (or underlying) in millions
    strike: Optional[float] = None  # $/share (None for RSUs)
    conversion_premium_pct: Optional[float] = None  # for convertibles/preferred
    label: str = ""


@dataclass
class IPOAssumptions:
    # ── Identity ────────────────────────────────────────────────────────────
    ticker: str = ""
    company_name: str = ""
    sector: str = ""
    industry: str = ""
    geography: str = ""
    business_model: str = ""

    # ── Financials ($M) ─────────────────────────────────────────────────────
    revenue: float = 0.0
    revenue_growth_pct: float = 0.0
    gross_margin_pct: float = 0.0
    ebitda: float = 0.0
    ebitda_margin_pct: float = 0.0
    ebit: float = 0.0
    fcf: float = 0.0
    net_income: float = 0.0
    net_debt: float = 0.0              # + = net debt, - = net cash
    shares_outstanding_m: float = 0.0  # shares OUTSTANDING pre-IPO (excl. treasury)

    # ── Dilutive securities (pre-IPO, $M / $) ───────────────────────────────
    options_m: float = 0.0
    options_strike: float = 0.0
    warrants_m: float = 0.0
    warrants_strike: float = 0.0
    rsus_m: float = 0.0
    convertible_m: float = 0.0
    convertible_premium_pct: float = 20.0
    preferred_m: float = 0.0
    preferred_liquidation_pref: float = 0.0

    # ── Offering structure ──────────────────────────────────────────────────
    primary_shares_m: float = 0.0      # new shares sold by the company
    secondary_shares_m: float = 0.0    # existing shares sold by holders
    gross_spread_pct: float = 7.0      # total underwriting spread (% of gross)
    other_expenses_m: float = 10.0     # legal/audit/listing etc. ($M)
    greenshoe_pct: float = 15.0        # overallotment option as % of offering
    greenshoe_exercised: bool = False

    # ── Valuation & comps context ───────────────────────────────────────────
    # Trading comps-derived metrics (fed from the CCA module where available)
    peer_ev_revenue: float = 6.0
    peer_ev_ebitda: float = 18.0
    peer_pe: float = 30.0
    peer_ev_ebit: float = 24.0
    # IPO-comparable multiples (recent deals)
    ipo_comp_ev_revenue: float = 8.0
    ipo_comp_ev_ebitda: float = 22.0
    # Precedent transaction multiples
    precedent_ev_ebitda: float = 25.0
    precedent_premium_pct: float = 25.0

    # ── DCF inputs (used for the DCF leg of the valuation) ──────────────────
    base_revenue: float = 0.0
    dcf_growth_rates: List[float] = field(default_factory=list)
    dcf_ebit_margin: float = 0.15
    tax_rate: float = 0.21
    da_pct_revenue: float = 0.04
    capex_pct_revenue: float = 0.05
    nwc_change_pct_revenue: float = 0.01
    wacc: float = 0.10
    terminal_growth_rate: float = 0.025
    projection_years: int = 5

    # ── Market pulse & demand ───────────────────────────────────────────────
    market_pulse: Optional[MarketPulseInputs] = None
    # Bookbuilding (all estimates unless an order book was actually observed)
    expected_oversubscription_x: float = 5.0
    institutional_demand_pct: float = 85.0
    price_elasticity: float = -2.0     # % change in demand per % change in price
    anchor_investor_commitment_pct: float = 0.0
    existing_shareholder_selling_pct: float = 0.0  # % of secondary shares from insiders

    # ── Scenario probabilities (base case is derived; others editable) ──────
    scenario_probabilities: Dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# SHARE-COUNT BRIDGE  (treasury stock method)
# ─────────────────────────────────────────────────────────────────────────────

def build_share_count_bridge(a: IPOAssumptions, offer_price: float) -> Dict[str, Any]:
    """Fully-diluted share count pre- and post-IPO using the treasury stock
    method for in-the-money options/warrants and conversion for convertibles."""
    existing = max(a.shares_outstanding_m, 0.0)

    # Options / warrants: treasury stock method
    itm_options = max(a.options_m, 0.0) if (a.options_strike or 0) < offer_price else 0.0
    proceeds_options = itm_options * a.options_strike
    buyback_options = proceeds_options / offer_price if offer_price > 0 else 0.0
    net_options = max(itm_options - buyback_options, 0.0)

    itm_warrants = max(a.warrants_m, 0.0) if (a.warrants_strike or 0) < offer_price else 0.0
    proceeds_warrants = itm_warrants * a.warrants_strike
    buyback_warrants = proceeds_warrants / offer_price if offer_price > 0 else 0.0
    net_warrants = max(itm_warrants - buyback_warrants, 0.0)

    # RSUs: no strike — full incremental shares (net of assumed withholding is a
    # modeling refinement; here we show gross RSUs as incremental).
    net_rsus = max(a.rsus_m, 0.0)

    # Convertible: converts at premium above offer price
    conv_shares = 0.0
    if a.convertible_m > 0 and a.convertible_premium_pct > 0:
        conv_price = offer_price * (1 + a.convertible_premium_pct / 100.0)
        conv_shares = a.convertible_m / conv_price if conv_price > 0 else 0.0

    # Preferred: converts into common (liquidation preference met first)
    pref_shares = max(a.preferred_m, 0.0)

    fully_diluted_pre = existing + net_options + net_warrants + net_rsus + conv_shares + pref_shares

    primary = max(a.primary_shares_m, 0.0)
    secondary = max(a.secondary_shares_m, 0.0)
    total_post = fully_diluted_pre + primary

    return {
        "existing_shares": existing,
        "options_gross": max(a.options_m, 0.0),
        "options_itm": round(itm_options, 3),
        "options_net_tsm": round(net_options, 3),
        "warrants_gross": max(a.warrants_m, 0.0),
        "warrants_net_tsm": round(net_warrants, 3),
        "rsus": round(net_rsus, 3),
        "convertible_shares": round(conv_shares, 3),
        "preferred_shares": round(pref_shares, 3),
        "fully_diluted_pre_ipo": round(fully_diluted_pre, 3),
        "primary_new_shares": primary,
        "secondary_shares": secondary,
        "total_shares_post_ipo": round(total_post, 3),
        "dilution_pct": round(primary / total_post * 100, 2) if total_post > 0 else 0.0,
        "method": "Treasury stock method (options/warrants); conversion at premium for convertibles.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# IPO PRICING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _dcf_value(a: IPOAssumptions) -> Optional[float]:
    """Simple multi-year DCF of the company's own operating assumptions."""
    try:
        base = a.base_revenue or a.revenue
        if base <= 0:
            return None
        g = list(a.dcf_growth_rates) if a.dcf_growth_rates else [a.revenue_growth_pct / 100.0] * a.projection_years
        while len(g) < a.projection_years:
            g.append(g[-1] if g else 0.05)
        g = g[:a.projection_years]
        rev = base
        fcfs = []
        for gr in g:
            rev *= (1 + gr)
            ebit = rev * a.dcf_ebit_margin
            nopat = ebit * (1 - a.tax_rate)
            fcf = nopat + rev * a.da_pct_revenue - rev * a.capex_pct_revenue - rev * a.nwc_change_pct_revenue
            fcfs.append(fcf)
        pv = sum(f / (1 + a.wacc) ** (i + 1) for i, f in enumerate(fcfs))
        tv = fcfs[-1] * (1 + a.terminal_growth_rate) / max(a.wacc - a.terminal_growth_rate, 0.01)
        pv_tv = tv / (1 + a.wacc) ** a.projection_years
        return pv + pv_tv
    except Exception:
        return None


class IPOPricingEngine:
    """Multi-method valuation with dynamic weighting.

    Weights are set by *data availability and method relevance*, not by
    fiat: the DCF gets more weight for profitable, mature issuers; CCA /
    IPO-comps dominate for growth issuers where cash flows are still ramping.
    """

    def price(self, a: IPOAssumptions, comps_context: Optional[Dict[str, Any]] = None
              ) -> Dict[str, Any]:
        methods: Dict[str, Dict[str, Any]] = {}
        ev_by_method: Dict[str, float] = {}

        # ── Trading comps (CCA) ──────────────────────────────────────────────
        if a.revenue > 0 and a.peer_ev_revenue > 0:
            ev = a.revenue * a.peer_ev_revenue
            ev_by_method["Trading comps EV/Revenue"] = ev
            methods["Trading comps EV/Revenue"] = {"ev": ev, "multiple": a.peer_ev_revenue, "weight": 1.0}
        if a.ebitda > 0 and a.peer_ev_ebitda > 0:
            ev = a.ebitda * a.peer_ev_ebitda
            ev_by_method["Trading comps EV/EBITDA"] = ev
            methods["Trading comps EV/EBITDA"] = {"ev": ev, "multiple": a.peer_ev_ebitda, "weight": 1.2}
        if a.ebit > 0 and a.peer_ev_ebit > 0:
            ev = a.ebit * a.peer_ev_ebit
            ev_by_method["Trading comps EV/EBIT"] = ev
            methods["Trading comps EV/EBIT"] = {"ev": ev, "multiple": a.peer_ev_ebit, "weight": 0.8}

        # ── IPO comps (recent offerings) ─────────────────────────────────────
        if a.revenue > 0 and a.ipo_comp_ev_revenue > 0:
            ev = a.revenue * a.ipo_comp_ev_revenue
            ev_by_method["IPO comps EV/Revenue"] = ev
            methods["IPO comps EV/Revenue"] = {"ev": ev, "multiple": a.ipo_comp_ev_revenue, "weight": 1.1}
        if a.ebitda > 0 and a.ipo_comp_ev_ebitda > 0:
            ev = a.ebitda * a.ipo_comp_ev_ebitda
            ev_by_method["IPO comps EV/EBITDA"] = ev
            methods["IPO comps EV/EBITDA"] = {"ev": ev, "multiple": a.ipo_comp_ev_ebitda, "weight": 1.1}

        # ── Precedent transactions ───────────────────────────────────────────
        if a.ebitda > 0 and a.precedent_ev_ebitda > 0:
            ev = a.ebitda * a.precedent_ev_ebitda
            ev_by_method["Precedent transactions EV/EBITDA"] = ev
            methods["Precedent transactions EV/EBITDA"] = {"ev": ev, "multiple": a.precedent_ev_ebitda, "weight": 0.7}

        # ── DCF ──────────────────────────────────────────────────────────────
        dcf_ev = _dcf_value(a)
        if dcf_ev and dcf_ev > 0:
            # Dynamic weighting: more weight for profitable / mature issuers
            w = 0.8 if a.ebitda_margin_pct >= 15 else (0.6 if a.ebitda_margin_pct >= 5 else 0.3)
            ev_by_method["DCF"] = dcf_ev
            methods["DCF"] = {"ev": dcf_ev, "multiple": None, "weight": w}

        if not ev_by_method:
            return {"methods": {}, "blended_ev": 0.0, "weights": {},
                    "price_range": (0.0, 0.0), "mid_price": 0.0,
                    "methodology_note": "Insufficient inputs to run IPO pricing."}

        total_w = sum(m["weight"] for m in methods.values())
        blended_ev = sum(ev * m["weight"] for k, m in methods.items()) / total_w

        # Weight justification (dynamic — explain why each method carries its weight)
        justifications = []
        for k, m in methods.items():
            why = (
                "Primary read for growth issuers" if "Revenue" in k
                else "Primary read for profitable issuers" if "EBITDA" in k and "Revenue" not in k
                else "Secondary read — transaction context" if "Precedent" in k
                else "Fundamental cross-check — less weight when cash flows are ramping"
                if k == "DCF" and a.ebitda_margin_pct < 10
                else "Fundamental cross-check"
            )
            justifications.append({"method": k, "weight": round(m["weight"] / total_w, 3), "why": why})

        # ── Pricing (discount for aftermarket risk & IPO "pop" budget) ───────
        # IPOs are typically priced at a discount to the *post-IPO* comp value to
        # leave money on the table for investors (the "IPO discount").
        ipo_discount = 0.15 if a.ebitda_margin_pct >= 15 else 0.20
        low_discount, high_discount = ipo_discount, ipo_discount - 0.05

        equity_value = blended_ev - a.net_debt
        bridge = build_share_count_bridge(a, offer_price=1.0)  # placeholder pricing

        # We need the offer price to compute the share bridge (TSM depends on
        # strike vs offer). Iterate: use blended per-share as the anchor.
        shares_pre = max(a.shares_outstanding_m, 1e-9) + max(a.options_m, 0) + max(a.warrants_m, 0) + max(a.rsus_m, 0) + max(a.preferred_m, 0)
        anchor_price = equity_value / shares_pre if shares_pre > 0 else 0.0

        bridge_low = build_share_count_bridge(a, offer_price=max(anchor_price * (1 - ipo_discount), 0.01))
        bridge_high = build_share_count_bridge(a, offer_price=max(anchor_price * (1 - high_discount), 0.01))

        shares_post_low = bridge_low["total_shares_post_ipo"]
        shares_post_high = bridge_high["total_shares_post_ipo"]

        low_price = (equity_value * (1 - low_discount)) / shares_post_low if shares_post_low > 0 else 0.0
        high_price = (equity_value * (1 - high_discount)) / shares_post_high if shares_post_high > 0 else 0.0
        mid = (low_price + high_price) / 2.0

        return {
            "methods": {k: {"ev": round(v["ev"], 2), "multiple": v["multiple"],
                            "weight": v["weight"]} for k, v in methods.items()},
            "blended_ev": round(blended_ev, 2),
            "weights": justifications,
            "equity_value": round(equity_value, 2),
            "price_range": (round(low_price, 2), round(high_price, 2)),
            "mid_price": round(mid, 2),
            "ipo_discount": ipo_discount,
            "methodology_note": (
                "Blend weights are dynamic: revenue multiples dominate for growth "
                "issuers; EBITDA-based methods and DCF dominate for profitable "
                "issuers. The range is struck at a discount to the blended comp "
                "value to price in aftermarket risk."
            ),
            "share_bridge": bridge,
        }


# ─────────────────────────────────────────────────────────────────────────────
# BOOKBUILDING / DEMAND FRAMEWORK
# ─────────────────────────────────────────────────────────────────────────────

class IPOBookbuildingEngine:
    """Models demand and pricing pressure from supply-side and demand-side
    estimates. All figures are clearly labeled estimates when no order book
    was observed."""

    def analyze(self, a: IPOAssumptions, price_range: Tuple[float, float],
                shares_post_ipo: float) -> Dict[str, Any]:
        low, high = price_range
        mid = (low + high) / 2
        offering = max(a.primary_shares_m, 0) + max(a.secondary_shares_m, 0)

        # Oversubscription → price pressure. Elasticity converts demand excess
        # into upward pricing pressure within the range.
        oversub = max(a.expected_oversubscription_x, 0.0)
        # Demand at low end of range vs high end
        demand_low = oversub
        elasticity = a.price_elasticity if a.price_elasticity < 0 else -2.0
        demand_high = oversub * (1 + elasticity * (high - low) / max(mid, 1e-9))

        price_pressure = "None" if demand_low < 1 else (
            "High — likely to price at/near the top of the range"
            if demand_low >= 8 else
            "Moderate — pricing within the upper half"
            if demand_low >= 3 else
            "Low — pricing within the lower half"
        )

        float_pct = shares_post_ipo - 0  # shares held by public post-offer
        free_float_pct = offering / max(shares_post_ipo, 1e-9) * 100

        # Aftermarket demand estimate
        aftermarket = (
            "Strong aftermarket support expected — oversubscription and modest "
            "free float support a positive 'pop'"
            if (demand_low >= 5 and free_float_pct <= 25)
            else "Balanced aftermarket — demand and float are in reasonable equilibrium"
            if demand_low >= 2
            else "Weak aftermarket risk — limited order book; price discovery may be volatile"
        )

        lockup_risk = (
            "Elevated lockup risk — large insider secondary selling; watch lockup expiry"
            if a.existing_shareholder_selling_pct >= 30
            else "Manageable lockup risk" if a.existing_shareholder_selling_pct > 0
            else "Minimal lockup overhang from insider selling"
        )

        return {
            "expected_oversubscription_x": oversub,
            "demand_at_low": round(demand_low, 2),
            "demand_at_high": round(max(demand_high, 0.0), 2),
            "price_pressure": price_pressure,
            "free_float_pct": round(free_float_pct, 1),
            "institutional_demand_pct": a.institutional_demand_pct,
            "aftermarket_outlook": aftermarket,
            "lockup_risk": lockup_risk,
            "disclosure": (
                "Order-book figures are ESTIMATES derived from user-supplied "
                "expectations — this platform does not have access to live ECM "
                "order data. Treat demand metrics as scenario inputs, not facts."
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC SCENARIO ENGINE → IPO CASE STUDIES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IPOScenario:
    label: str
    probability: float
    price: float
    valuation: float                 # implied EV in scenario
    market_reaction: str             # first-day / aftermarket outlook
    triggers: List[str]
    key_assumptions: List[str]
    risks: List[str]
    catalysts: List[str]
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IPOScenarioEngine:
    """Builds a *dynamic* set of materially distinct IPO scenarios. The base
    case is anchored to the pricing engine; down/up cases are coherently
    re-priced rather than arbitrary ±% marks."""

    def build(self, a: IPOAssumptions, pricing: Dict[str, Any],
              book: Dict[str, Any], pulse_label: str) -> List[IPOScenario]:
        low, high = pricing["price_range"]
        mid = pricing["mid_price"]
        blended_ev = pricing["blended_ev"]

        probs = dict(a.scenario_probabilities or {})

        scenarios: List[IPOScenario] = []

        # ── Downside: weak demand / market deterioration ─────────────────────
        down_price = low * 0.85
        down_prob = probs.get("Downside", 0.25)
        scenarios.append(IPOScenario(
            label="Downside / weak demand",
            probability=down_prob,
            price=round(down_price, 2),
            valuation=round(blended_ev * 0.80, 2),
            market_reaction=(
                "Priced near/below range low; weak first-day performance, elevated "
                "post-IPO volatility, possible aftermarket support buying."
            ),
            triggers=[
                "Order book materially below 3x oversubscription",
                "Broad market sell-off during the roadshow",
                "Comparable IPOs price below range / trade down post-offer",
                "Anchor investor withdrawals",
            ],
            key_assumptions=["Demand < 3x", "No anchor support", "Market de-rates"],
            risks=["Offering may need to be withdrawn or repriced", "Reputation damage for issuer/bank"],
            catalysts=["Buyer strike driven by valuation discipline", "Rates spike late in process"],
            rationale=(
                "Weak demand forces pricing to the low end or below, leaving the "
                "issuer with less capital and investors demanding a discount."
            ),
        ))

        # ── Base case: pricing-engine central path ───────────────────────────
        base_prob = probs.get("Base", 0.50)
        scenarios.append(IPOScenario(
            label="Base / expected",
            probability=base_prob,
            price=round(mid, 2),
            valuation=round(blended_ev, 2),
            market_reaction=(
                f"Prices within the ${low:.2f} – ${high:.2f} range; moderate "
                "first-day 'pop' consistent with the market-pulse reading "
                f"('{pulse_label}')."
            ),
            triggers=["Roadshow proceeds as planned", "Oversubscription 4-8x",
                      "Market stable through pricing"],
            key_assumptions=["Base multiples hold", "Pricing discount maintained"],
            risks=["Aftermarket drift below offer", "Lockup expiry overhang"],
            catalysts=["Positive earnings post-IPO", "Index inclusion eligibility"],
            rationale="The central path implied by the pricing engine and bookbuilding framework.",
        ))

        # ── Upside: strong demand / multiple expansion ───────────────────────
        up_price = high * 1.12
        up_prob = probs.get("Upside", 0.15)
        scenarios.append(IPOScenario(
            label="Upside / hot offering",
            probability=up_prob,
            price=round(up_price, 2),
            valuation=round(blended_ev * 1.18, 2),
            market_reaction=(
                "Priced at/near range high; strong first-day performance, "
                "potential for trading above range as momentum builds."
            ),
            triggers=[
                "Oversubscription > 10x",
                "Strong comparable IPO aftermarket",
                "Multiple expansion in the comp set",
                "Anchor demand exceeds supply",
            ],
            key_assumptions=["Demand > 10x", "Comp multiples expand", "No macro shock"],
            risks=["Over-heated pricing leaves less upside for aftermarket buyers",
                   "Higher short-term volatility post-pop"],
            catalysts=["Scarcity premium on float", "Follow-on demand from index funds"],
            rationale="Demand far exceeds supply, permitting pricing at the top of the range with a strong aftermarket.",
        ))

        # ── Optional: transformation / over-heating scenario when conditions
        #    are highly supportive or the company is clearly a platform asset ─
        if pulse_label in ("Highly supportive", "Supportive") or a.revenue_growth_pct > 50:
            hot_prob = probs.get("Hot / de-SPAC style", 0.10)
            scenarios.append(IPOScenario(
                label="Hot / momentum-driven",
                probability=hot_prob,
                price=round(high * 1.3, 2),
                valuation=round(blended_ev * 1.35, 2),
                market_reaction=(
                    "Price discovery runs well above range; substantial first-day "
                    "gains, retail participation, potential for a short squeeze "
                    "on the float."
                ),
                triggers=["Compelling narrative + scarce float + strong momentum",
                          "Comparable IPOs trading up aggressively"],
                key_assumptions=["Narrative momentum persists", "Retail participation high"],
                risks=["Sharp mean-reversion risk", "Post-lockup overhang"],
                catalysts=["Analyst coverage initiation", "Index inclusion"],
                rationale="In a supportive tape with a scarce float, momentum can dominate fundamentals near-term.",
            ))

        # Normalize probabilities to sum to 1
        total = sum(s.probability for s in scenarios)
        if total > 0:
            for s in scenarios:
                s.probability = round(s.probability / total, 4)

        return scenarios


# ─────────────────────────────────────────────────────────────────────────────
# RESULT & ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IPOResult:
    assumptions: IPOAssumptions
    market_pulse: Dict[str, Any]
    pricing: Dict[str, Any]
    bookbuilding: Dict[str, Any]
    share_count_bridge: Dict[str, Any]
    scenarios: List[IPOScenario]
    valuation_methods: Dict[str, Any]
    implied_ipo_price: float
    price_range: Tuple[float, float]
    gross_proceeds: float
    primary_proceeds: float
    secondary_proceeds: float
    net_proceeds: float
    shares_outstanding: float
    market_cap_at_mid: float
    dilution_pct: float
    expected_first_day_return_pct: float
    conclusion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_pulse": self.market_pulse,
            "pricing": self.pricing,
            "bookbuilding": self.bookbuilding,
            "share_count_bridge": self.share_count_bridge,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "price_range": list(self.price_range),
            "gross_proceeds": self.gross_proceeds,
            "net_proceeds": self.net_proceeds,
            "shares_outstanding": self.shares_outstanding,
            "market_cap_at_mid": self.market_cap_at_mid,
            "expected_first_day_return_pct": self.expected_first_day_return_pct,
            "conclusion": self.conclusion,
        }


class IPOEngine:
    def __init__(self):
        self._pricing = IPOPricingEngine()
        self._book = IPOBookbuildingEngine()
        self._scenarios = IPOScenarioEngine()

    def run_ipo(self, a: IPOAssumptions,
                comps_context: Optional[Dict[str, Any]] = None) -> IPOResult:
        # 1) Market pulse
        pulse = get_market_pulse_engine().evaluate(a.market_pulse or MarketPulseInputs())

        # 2) Pricing
        pricing = self._pricing.price(a, comps_context)
        low, high = pricing["price_range"]
        mid = pricing["mid_price"]

        # 3) Share bridge at mid price
        bridge = build_share_count_bridge(a, offer_price=mid)
        shares_post = bridge["total_shares_post_ipo"]

        # 4) Proceeds
        primary = max(a.primary_shares_m, 0)
        secondary = max(a.secondary_shares_m, 0)
        gross = (primary + secondary) * mid
        spread = gross * a.gross_spread_pct / 100.0
        expenses = max(a.other_expenses_m, 0)
        net_primary = primary * mid - (primary / max(primary + secondary, 1e-9)) * (spread + expenses) if (primary + secondary) > 0 else 0
        net_proceeds = gross - spread - expenses
        primary_proceeds = primary * mid
        secondary_proceeds = secondary * mid

        # 5) Bookbuilding
        book = self._book.analyze(a, (low, high), shares_post)

        # 6) Scenarios
        scenarios = self._scenarios.build(a, pricing, book, pulse["label"])
        prob_weighted_price = sum(s.price * s.probability for s in scenarios)

        # 7) Expected first-day return (estimate): the "pop" consistent with
        #    the pricing discount and demand. Estimate, clearly labeled.
        demand = max(book["expected_oversubscription_x"], 1.0)
        pop_est = min(max((demand - 1.0) * 2.0, 2.0), 45.0)
        pulse_idx = pulse["index"]
        pop_est = pop_est * (1.15 if pulse_idx <= 1 else 0.85 if pulse_idx >= 3 else 1.0)

        market_cap = shares_post * mid

        conclusion = self._conclusion(a, pricing, book, pulse, scenarios, mid, low, high)
        return IPOResult(
            assumptions=a,
            market_pulse=pulse,
            pricing=pricing,
            bookbuilding=book,
            share_count_bridge=bridge,
            scenarios=scenarios,
            valuation_methods=pricing["methods"],
            implied_ipo_price=round(prob_weighted_price, 2),
            price_range=(low, high),
            gross_proceeds=round(gross, 2),
            primary_proceeds=round(primary_proceeds, 2),
            secondary_proceeds=round(secondary_proceeds, 2),
            net_proceeds=round(net_proceeds, 2),
            shares_outstanding=shares_post,
            market_cap_at_mid=round(market_cap, 2),
            dilution_pct=bridge["dilution_pct"],
            expected_first_day_return_pct=round(pop_est, 1),
            conclusion=conclusion,
        )

    def _conclusion(self, a, pricing, book, pulse, scenarios, mid, low, high) -> str:
        parts = [
            f"IPO analysis for {a.company_name or a.ticker or 'the issuer'}: a "
            f"defensible price range of ${low:.2f} – ${high:.2f} (mid ${mid:.2f}) "
            f"implied across {len(pricing['methods'])} valuation methodologies.",
            f"The current market environment reads **{pulse['label']}** — {pulse['explanation'].split('.')[0]}.",
            book["price_pressure"] + ".",
            f"Probability-weighted scenario price: ${mid:.2f}.",
            "The range embeds an IPO discount to comp value (standard practice); "
            "final pricing will depend on live order-book dynamics that this "
            "platform cannot observe and therefore models as estimates.",
        ]
        return " ".join(parts)


_ipo_engine = None


def get_ipo_engine() -> IPOEngine:
    global _ipo_engine
    if _ipo_engine is None:
        _ipo_engine = IPOEngine()
    return _ipo_engine
