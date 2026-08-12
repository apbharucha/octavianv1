"""
Octavian Investment Memo Generator
==================================

Produces a fully written, analytical (not promotional) investment-analysis
document from any model context in the Financial Model Generator. The memo
structure follows institutional standards:

    1. Situation        8. Scenario construction     15. Why it could fail
    2. Company          9. Key assumptions           16. What would invalidate
    3. Industry        10. Risks                         the thesis
    4. Market env      11. Catalysts                 17. Final conclusion
    5. Consensus       12. Financial analysis
    6. Sentiment       13. Transaction analysis
    7. Valuation       14. Why the case works

The generator composes the memo from the *actual* model outputs passed in,
so the document always reflects the real analysis rather than boilerplate.

Author: Octavian Terminal
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class MemoContext:
    """Everything the memo can draw on. All optional; the memo adapts."""
    title: str = "Investment Analysis"
    ticker: str = ""
    company_name: str = ""
    sector: str = ""
    subject_type: str = ""            # "Equity" | "IPO" | "M&A" | "LBO" | "Comps" etc.

    # Fundamentals
    revenue: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    ebitda_margin_pct: Optional[float] = None
    net_debt: Optional[float] = None

    # Valuation
    fair_value: Optional[float] = None
    current_price: Optional[float] = None
    valuation_methods: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    valuation_narrative: str = ""

    # Market / consensus / sentiment
    market_pulse_label: str = ""
    market_pulse_explanation: str = ""
    consensus_summary: str = ""
    sentiment_summary: str = ""
    dislocation_summary: str = ""

    # Scenarios
    scenarios: List[Dict[str, Any]] = field(default_factory=list)

    # Risks / catalysts
    risks: List[str] = field(default_factory=list)
    catalysts: List[str] = field(default_factory=list)

    # Transaction-specific
    transaction_summary: str = ""
    transaction_outcome: str = ""

    # QA
    qa_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InvestmentMemoGenerator:

    def generate(self, ctx: MemoContext) -> str:
        L: List[str] = []
        L.append(f"# Investment Memorandum — {ctx.title}")
        L.append(f"**Prepared by:** Octavian Terminal  |  **Date:** {datetime.now().strftime('%B %d, %Y')}")
        if ctx.ticker:
            L.append(f"**Subject:** {ctx.company_name or ctx.ticker} ({ctx.ticker})"
                     + (f" — {ctx.sector}" if ctx.sector else ""))
        L.append("---")

        # 1. Situation
        L.append("## 1. Situation")
        if ctx.subject_type == "IPO":
            L.append(self._para(
                f"This memo evaluates the initial public offering of {ctx.company_name or ctx.ticker}. "
                f"The objective is to determine a defensible price range, understand the demand "
                f"environment, and stress-test the offering across scenarios."
            ))
        elif ctx.subject_type == "M&A":
            L.append(self._para(ctx.transaction_summary or
                                "This memo evaluates a proposed acquisition, its accretion/dilution "
                                "profile, financing, synergies, and value creation."))
        elif ctx.subject_type == "LBO":
            L.append(self._para(ctx.transaction_summary or
                                "This memo evaluates the leveraged buyout of the subject, sponsor "
                                "returns, debt capacity, and downside protection."))
        else:
            L.append(self._para(
                f"This memo evaluates the investment case for {ctx.company_name or ctx.ticker or 'the subject'}, "
                f"drawing on the institutional models in the Financial Model Generator."
            ))

        # 2. Company
        L.append("## 2. Company")
        if ctx.revenue is not None:
            L.append(self._para(
                f"The company generates ${ctx.revenue:,.0f}M revenue, growing at "
                f"{ctx.revenue_growth_pct:.1f}% with an EBITDA margin of {ctx.ebitda_margin_pct:.1f}%. "
                f"Net debt is ${ctx.net_debt:,.0f}M."
            ))
        else:
            L.append(self._para("Financial profile: see model outputs for revenue, margins and capital structure."))

        # 3. Industry
        L.append("## 3. Industry")
        L.append(self._para(
            f"Operating in {ctx.sector or 'the relevant'} industry. Competitive dynamics, cyclicality, "
            "and regulatory exposure are the primary industry-level variables that condition the forecast."
        ))

        # 4. Market environment
        L.append("## 4. Market Environment")
        if ctx.market_pulse_label:
            L.append(self._para(
                f"Market conditions read **{ctx.market_pulse_label}**. {ctx.market_pulse_explanation}"
            ))
        else:
            L.append(self._para("Market conditions: no market-pulse inputs supplied; treat valuation "
                                "as independent of issuance-window conditions."))

        # 5. Consensus
        L.append("## 5. Consensus")
        L.append(self._para(ctx.consensus_summary or
                            "No analyst consensus inputs were supplied. The memo therefore relies on "
                            "model-implied expectations rather than sell-side estimates."))

        # 6. Sentiment
        L.append("## 6. Sentiment")
        L.append(self._para(ctx.sentiment_summary or
                            "No sentiment signals were supplied. Positioning is treated as neutral."))

        # 7. Valuation
        L.append("## 7. Valuation")
        if ctx.valuation_methods:
            L.append("| Methodology | Low | High | Mid | Weight |")
            L.append("|---|---|---|---|---|")
            for k, v in ctx.valuation_methods.items():
                lo = v.get("low"); hi = v.get("high"); mid = v.get("mid"); w = v.get("weight")
                L.append(
                    f"| {k} | ${lo:,.2f} | ${hi:,.2f} | ${mid:,.2f} | {w:.0%} |"
                    if all(x is not None for x in (lo, hi, mid, w)) else f"| {k} | — | — | — | — |"
                )
            L.append("")
        if ctx.fair_value is not None and ctx.current_price:
            gap = ctx.fair_value / ctx.current_price - 1
            L.append(self._para(
                f"Weighted fair value is ${ctx.fair_value:.2f} vs market ${ctx.current_price:.2f} "
                f"({gap:+.1%}). {ctx.valuation_narrative}"
            ))
        elif ctx.fair_value is not None:
            L.append(self._para(f"Weighted fair value is ${ctx.fair_value:.2f}. {ctx.valuation_narrative}"))

        # 8. Scenario construction
        L.append("## 8. Scenario Construction")
        if ctx.scenarios:
            L.append("The analysis is built around the following materially distinct scenarios:")
            L.append("")
            for s in ctx.scenarios:
                prob = s.get("probability", 0)
                px = s.get("price")
                L.append(f"- **{s.get('label','Scenario')}** (P = {prob:.0%}): "
                         f"{('prices at $%.2f.' % px) if px else 'see model outputs.'} "
                         f"{s.get('rationale', '')}")
        else:
            L.append(self._para("Scenario analysis was not generated for this context."))

        # 9. Key assumptions
        L.append("## 9. Key Assumptions")
        L.append(self._para(
            "The valuation is most sensitive to growth, margins, discount rate, and exit/terminal "
            "assumptions. Each model's sensitivity tables identify the highest-value sensitivities; "
            "these should be revisited before any decision."
        ))

        # 10. Risks
        L.append("## 10. Risks")
        if ctx.risks:
            for r in ctx.risks:
                L.append(f"- {r}")
        else:
            L.append(self._para("No structured risk list was supplied; see the model risk output."))
        L.append("")

        # 11. Catalysts
        L.append("## 11. Catalysts")
        if ctx.catalysts:
            for c in ctx.catalysts:
                L.append(f"- {c}")
        else:
            L.append(self._para("No structured catalyst list was supplied; see the model catalyst output."))
        L.append("")

        # 12. Financial analysis
        L.append("## 12. Financial Analysis")
        L.append(self._para(
            "Financial analysis is grounded in the operating model: revenue build, margin trajectory, "
            "working-capital and capex intensity, and free-cash-flow conversion. Historical anchoring "
            "and market-implied expectations (reverse DCF) provide the discipline for the forecast."
        ))

        # 13. Transaction analysis
        L.append("## 13. Transaction Analysis")
        if ctx.transaction_outcome:
            L.append(self._para(ctx.transaction_outcome))
        else:
            L.append(self._para("No transaction-specific analysis was generated for this context."))

        # 14. Why the case works
        L.append("## 14. Why the Case Works")
        L.append(self._para(
            "The case works if the operating plan is delivered at the modeled cost of capital and the "
            "market re-rates the company toward the model-implied value. The margin of safety is the "
            "gap between market price and weighted fair value, adjusted for method reliability."
        ))

        # 15. Why it could fail
        L.append("## 15. Why It Could Fail")
        L.append(self._para(
            "The case fails if growth stalls, margins compress, the discount rate rises, or sentiment "
            "de-rates the multiple. For transaction contexts, integration/execution risk, financing "
            "conditions, and exit-market deterioration are the principal failure modes."
        ))

        # 16. What would invalidate the thesis
        L.append("## 16. What Would Invalidate the Thesis")
        L.append(self._para(
            "The thesis is invalidated if the market-implied expectations prove unachievable "
            "(sustained miss on growth or margins), if the weighted valuation range collapses toward "
            "or below market price, or if key transaction assumptions (synergies, financing, exit "
            "multiple) break."
        ))

        # 17. Final conclusion
        L.append("## 17. Final Conclusion")
        if ctx.fair_value is not None and ctx.current_price:
            gap = ctx.fair_value / ctx.current_price - 1
            if gap > 0.15:
                verdict = "the models indicate the security is undervalued; the case for ownership is supported."
            elif gap < -0.15:
                verdict = "the models indicate the security is overvalued; discipline favors restraint."
            else:
                verdict = "the models and market are broadly aligned; conviction requires differentiated information."
            L.append(self._para(f"On balance, {verdict}"))
        else:
            L.append(self._para("Conclusion pending full model outputs."))
        if ctx.qa_summary:
            L.append("")
            L.append(f"> **Model QA:** {ctx.qa_summary}")

        L.append("")
        L.append("---")
        L.append("*This memorandum is analytical, not promotional. It is generated from model outputs "
                "and user-supplied inputs and does not constitute investment advice.*")
        return "\n\n".join(L)

    def _para(self, text: str) -> str:
        return text


_memo_generator = None


def get_memo_generator() -> InvestmentMemoGenerator:
    global _memo_generator
    if _memo_generator is None:
        _memo_generator = InvestmentMemoGenerator()
    return _memo_generator
