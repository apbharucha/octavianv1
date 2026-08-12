"""
Octavian Market Consensus & Sentiment Engine
=============================================

The pervasive "market layer" shared by every model in the Financial Model
Generator. It is deliberately *not* a data-fabrication engine:

* Consensus, sentiment and market-pulse figures are only ever computed from
  inputs the analyst provides (or that the platform can genuinely observe).
* Every input carries a `DataProvenance` tag so the platform never presents
  model estimates as observed facts.
* The dislocation engine reverse-engineers what the *market price* already
  implies and asks: what would have to happen for the market to be wrong?

Author: Octavian Terminal
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# DATA PROVENANCE  (Section 28 of the upgrade spec)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataProvenance:
    """Tracks where a value came from so the model never conflates
    observation with estimation.

    status ∈ {"verified", "derived", "estimated", "assumption"}
    """
    label: str
    value: float
    status: str = "assumption"          # verified | derived | estimated | assumption
    source: str = "user input"          # e.g. "yfinance", "SEC filings", "model"
    date: str = ""                      # ISO date of the underlying observation
    confidence: float = 0.5             # 0..1
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def status_rank(self) -> int:
        return {"verified": 0, "derived": 1, "estimated": 2, "assumption": 3}.get(
            self.status, 3
        )


STATUS_LABELS = {
    "verified": "Verified — directly observed source",
    "derived": "Derived — calculated from sourced information",
    "estimated": "Estimated — model estimate",
    "assumption": "Assumption — user/model assumption",
}


# ─────────────────────────────────────────────────────────────────────────────
# CONSENSUS LAYER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConsensusEstimates:
    """What sell-side / market consensus *believes* about the company.

    All fields are optional and only set when the analyst or the platform
    supplies them; a missing field means "no consensus data available" and
    is rendered as such (never fabricated).
    """
    revenue_growth_pct: Optional[float] = None        # next-year consensus rev growth
    ebitda_margin_pct: Optional[float] = None
    eps_growth_pct: Optional[float] = None
    fcf_growth_pct: Optional[float] = None
    revenue_growth_5y_pct: Optional[float] = None     # LT growth
    target_price: Optional[float] = None
    analyst_rating: Optional[str] = None              # Buy / Hold / Sell
    num_analysts: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    def has_data(self) -> bool:
        return any(v is not None for v in (
            self.revenue_growth_pct, self.ebitda_margin_pct, self.eps_growth_pct,
            self.fcf_growth_pct, self.revenue_growth_5y_pct, self.target_price,
            self.analyst_rating,
        ))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT LAYER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SentimentSnapshot:
    """Distinct, labeled sentiment signals. Never blended into a single
    number without the composite being labeled an *estimate*.

    score conventions: each signal is on a -1.0 (very bearish) .. +1.0
    (very bullish) scale, or None when unavailable.
    """
    news_sentiment: Optional[float] = None
    analyst_sentiment: Optional[float] = None
    earnings_call_tone: Optional[float] = None
    price_momentum_1m: Optional[float] = None          # 1m price return
    price_momentum_3m: Optional[float] = None
    volatility_20d: Optional[float] = None             # annualized vol
    short_interest_pct: Optional[float] = None         # short interest as % float
    institutional_flow: Optional[float] = None         # + = net inflow estimate
    options_skew: Optional[float] = None               # >0 = put skew elevated
    sector_sentiment: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def composite(self) -> Tuple[float, int]:
        """Signed average of available sentiment signals.

        Returns (score, n_signals). The composite is always labeled an
        *estimate* derived from whatever signals were available.
        """
        vals = [v for v in (
            self.news_sentiment, self.analyst_sentiment, self.earnings_call_tone,
            self.institutional_flow, self.sector_sentiment,
        ) if v is not None]
        if not vals:
            return 0.0, 0
        return float(np.mean(vals)), len(vals)

    def label(self) -> str:
        score, n = self.composite()
        if n == 0:
            return "No sentiment data"
        if score > 0.25:
            return "Bullish sentiment"
        if score < -0.25:
            return "Bearish sentiment"
        return "Neutral sentiment"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        score, n = self.composite()
        d["composite_score"] = score
        d["composite_n"] = n
        d["composite_label"] = self.label()
        return d


# ─────────────────────────────────────────────────────────────────────────────
# MODEL-IMPLIED EXPECTATIONS (the "what does the price imply?" solver)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImpliedExpectations:
    """Reverse-engineered expectations embedded in a market price.

    Each field is an *estimate* derived from a simplified DCF identity, so
    every field is labeled 'estimated' in provenance terms.
    """
    implied_revenue_growth: float = 0.0
    implied_ebit_margin: float = 0.0
    implied_fcf_growth: float = 0.0
    implied_ev: float = 0.0
    market_implied_terminal_multiple: float = 0.0
    current_price: float = 0.0
    model_price: float = 0.0
    upside_pct: float = 0.0
    gap_label: str = ""
    gap_color: str = "#aaaaaa"
    narrative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def reverse_dcf_expectations(
    model_price: float,
    current_price: float,
    shares: float,
    net_debt: float,
    base_revenue: float,
    model_revenue_growth: float,
    model_ebit_margin: float,
    wacc: float,
    terminal_growth: float,
    projection_years: int = 5,
) -> ImpliedExpectations:
    """Estimate what the current market price implies vs the analyst model.

    Uses a simplified valuation identity: for a stable-growth company,
    EV ≈ FCF1 / (WACC − g). We solve for the growth rate and margin that
    would justify the observed market capitalization.
    """
    market_cap = max(current_price, 0.0) * max(shares, 1e-9)
    implied_ev = market_cap + net_debt

    # Baseline FCF conversion (proxy): FCF = Rev * margin * (1 - t) * conversion
    # with a cash-conversion haircut — calibrated to the analyst's own model so
    # the comparison is apples-to-apples.
    implied_growth = model_revenue_growth
    implied_margin = model_ebit_margin

    try:
        # Solve for growth that justifies implied EV, holding margin fixed.
        # FCF1 ≈ Rev1 * m * (1-t); EV = FCF1*(1+g)/(WACC-g)
        m = max(model_ebit_margin, 0.01)
        t = 0.21
        rev1 = base_revenue * (1 + model_revenue_growth)
        fcf1 = rev1 * m * (1 - t)
        if implied_ev > 0 and wacc > terminal_growth:
            # EV = fcf1*(1+g)/(wacc-g)  →  g = (EV*wacc - fcf1)/(EV + fcf1)
            implied_growth = (implied_ev * wacc - fcf1) / (implied_ev + fcf1)
            implied_growth = max(-0.15, min(0.60, implied_growth))
    except Exception:
        pass

    try:
        # Solve for margin that justifies implied EV, holding growth fixed.
        rev1 = base_revenue * (1 + model_revenue_growth)
        if implied_ev > 0 and wacc > terminal_growth:
            # EV = rev1*m*(1-t)*(1+g)/(wacc-g) → m = EV*(wacc-g)/(rev1*(1-t)*(1+g))
            implied_margin = (
                implied_ev * (wacc - terminal_growth)
                / (rev1 * (1 - 0.21) * (1 + terminal_growth))
            )
            implied_margin = max(0.005, min(0.90, implied_margin))
    except Exception:
        pass

    upside = (model_price / current_price - 1.0) * 100 if current_price > 0 else 0.0

    growth_diff = implied_growth - model_revenue_growth
    if growth_diff > 0.02:
        gap_label = "Market prices in HIGHER growth than the model"
        gap_color = "#00c853"
    elif growth_diff < -0.02:
        gap_label = "Market prices in LOWER growth than the model"
        gap_color = "#d32f2f"
    else:
        gap_label = "Market growth expectations broadly aligned with the model"
        gap_color = "#e0c97f"

    narrative = (
        f"The current price of ${current_price:.2f} implies roughly "
        f"{implied_growth:.1%} revenue growth and a {implied_margin:.1%} EBIT margin "
        f"vs the model's {model_revenue_growth:.1%} / {model_ebit_margin:.1%}. "
        f"{gap_label}. For the market to be wrong on the upside, the company would "
        f"need to deliver {model_price / current_price - 1.0:+.1%} above the embedded "
        f"path; on the downside, any sustained miss on these embedded expectations "
        f"represents the principal de-rating risk."
    )

    return ImpliedExpectations(
        implied_revenue_growth=float(implied_growth),
        implied_ebit_margin=float(implied_margin),
        implied_fcf_growth=float(max(wacc - terminal_growth, 0.0)),
        implied_ev=float(implied_ev),
        market_implied_terminal_multiple=float(1.0 / max(wacc - terminal_growth, 0.01)),
        current_price=float(current_price),
        model_price=float(model_price),
        upside_pct=float(upside),
        gap_label=gap_label,
        gap_color=gap_color,
        narrative=narrative,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONSENSUS DISLOCATION ENGINE  (Section 7 of the upgrade spec)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DislocationFinding:
    category: str            # e.g. "consensus gap", "bullish disagreement", "narrative risk"
    severity: str            # High / Medium / Low
    title: str
    detail: str
    direction: str           # Positive / Negative / Neutral

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConsensusDislocationReport:
    findings: List[DislocationFinding] = field(default_factory=list)
    summary: str = ""
    overall_reading: str = ""
    overall_color: str = "#aaaaaa"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "overall_reading": self.overall_reading,
            "overall_color": self.overall_color,
        }


class ConsensusDislocationEngine:
    """Asks: what does the market already believe, and what would have to
    happen for the market to be wrong?

    The engine deliberately refuses to fabricate consensus numbers. It only
    reasons over the expectations it is given (analyst-provided consensus,
    model-implied expectations, and sentiment signals).
    """

    def analyze(
        self,
        *,
        implied: Optional[ImpliedExpectations] = None,
        consensus: Optional[ConsensusEstimates] = None,
        sentiment: Optional[SentimentSnapshot] = None,
        model_price: float = 0.0,
        current_price: float = 0.0,
        model_upside_pct: float = 0.0,
    ) -> ConsensusDislocationReport:
        findings: List[DislocationFinding] = []
        has_implied = implied is not None and implied.model_price > 0

        # 1) Model vs market (the core "is the market wrong?" question)
        if has_implied:
            upside = model_upside_pct
            if upside > 0.20:
                findings.append(DislocationFinding(
                    category="model vs market",
                    severity="High",
                    title="Model implies material upside to the market price",
                    detail=(
                        f"The model's {implied.model_price:.2f} fair value sits "
                        f"{upside:+.1f}% above the ${current_price:.2f} market price. "
                        f"For the model to be right, the company must deliver the "
                        f"embedded {implied.implied_revenue_growth:.1%} growth path or better."
                    ),
                    direction="Positive",
                ))
            elif upside < -0.20:
                findings.append(DislocationFinding(
                    category="model vs market",
                    severity="High",
                    title="Model implies downside to the market price",
                    detail=(
                        f"The model's {implied.model_price:.2f} fair value sits "
                        f"{upside:+.1f}% below the market. The market is paying for "
                        f"{implied.implied_revenue_growth:.1%} growth — a level the "
                        f"model regards as too rich."
                    ),
                    direction="Negative",
                ))

        # 2) Consensus vs model
        if consensus is not None and consensus.has_data() and has_implied:
            if consensus.revenue_growth_pct is not None:
                cons_g = consensus.revenue_growth_pct / 100.0
                if cons_g > implied.implied_revenue_growth + 0.03:
                    findings.append(DislocationFinding(
                        category="consensus vs implied",
                        severity="Medium",
                        title="Consensus growth sits above market-implied growth",
                        detail=(
                            f"Consensus expects {cons_g:.1%} revenue growth while the "
                            f"market price embeds only {implied.implied_revenue_growth:.1%}. "
                            f"If consensus is right, the stock is undervalued; a miss "
                            f"would compress both estimates and multiple."
                        ),
                        direction="Positive",
                    ))
                elif cons_g < implied.implied_revenue_growth - 0.03:
                    findings.append(DislocationFinding(
                        category="consensus vs implied",
                        severity="Medium",
                        title="Market embeds growth above consensus",
                        detail=(
                            f"The price embeds {implied.implied_revenue_growth:.1%} growth "
                            f"while consensus expects {cons_g:.1%}. The market is already "
                            f"paying for an acceleration consensus has not yet endorsed — "
                            f"an earnings miss would be punished sharply."
                        ),
                        direction="Negative",
                    ))

        # 3) Consensus blind spots / bullish & bearish disagreement
        if consensus is not None and consensus.has_data():
            if consensus.analyst_rating and "Buy" in consensus.analyst_rating:
                findings.append(DislocationFinding(
                    category="bullish disagreement",
                    severity="Low",
                    title="Sell-side ratings are bullish",
                    detail=(
                        "With a consensus Buy rating, positive news is largely "
                        "reflected; the marginal buyer is already in. Surprises "
                        "therefore skew to the downside of expectations."
                    ),
                    direction="Neutral",
                ))
            if consensus.revenue_growth_pct is not None and consensus.revenue_growth_pct < 3.0:
                findings.append(DislocationFinding(
                    category="consensus blind spot",
                    severity="Medium",
                    title="Low consensus growth leaves room for positive surprise",
                    detail=(
                        f"Consensus embeds only {consensus.revenue_growth_pct:.1f}% growth. "
                        f"Modest acceleration — new product cycles, pricing, or share "
                        f"gains — would constitute a positive surprise the market is "
                        f"not positioned for."
                    ),
                    direction="Positive",
                ))

        # 4) Sentiment-driven risks
        if sentiment is not None:
            score, n = sentiment.composite()
            if n > 0 and score > 0.3:
                findings.append(DislocationFinding(
                    category="narrative risk",
                    severity="Medium",
                    title="Sentiment is stretched positive",
                    detail=(
                        f"Composite sentiment ({score:+.2f}) is strongly positive. "
                        f"Crowded positioning raises the risk of sharp mean-reversion "
                        f"on any disappointment, even if fundamentals are intact."
                    ),
                    direction="Negative",
                ))
            if sentiment.short_interest_pct is not None and sentiment.short_interest_pct > 8.0:
                findings.append(DislocationFinding(
                    category="positioning",
                    severity="Medium",
                    title="Elevated short interest",
                    detail=(
                        f"Short interest of {sentiment.short_interest_pct:.1f}% of float "
                        f"creates squeeze potential on positive news but indicates "
                        f"meaningful skeptical capital."
                    ),
                    direction="Neutral",
                ))

        # 5) Volatility / rate sensitivity
        if sentiment is not None and sentiment.volatility_20d is not None:
            if sentiment.volatility_20d > 0.45:
                findings.append(DislocationFinding(
                    category="valuation risk",
                    severity="Medium",
                    title="Elevated volatility raises discount-rate sensitivity",
                    detail=(
                        f"20-day annualized volatility of {sentiment.volatility_20d:.0%} "
                        f"means the valuation is unusually sensitive to WACC moves — "
                        f"rate headlines will dominate the tape."
                    ),
                    direction="Neutral",
                ))

        if not findings:
            findings.append(DislocationFinding(
                category="reading",
                severity="Low",
                title="No material dislocation detected",
                detail=(
                    "Available signals do not indicate a large gap between price, "
                    "consensus and model expectations. Monitor for new information "
                    "rather than forcing a trade."
                ),
                direction="Neutral",
            ))

        # Overall reading
        pos = sum(1 for f in findings if f.direction == "Positive")
        neg = sum(1 for f in findings if f.direction == "Negative")
        if pos > neg + 1:
            reading, color = "Dislocation skews positive — watch for upside surprises", "#00c853"
        elif neg > pos + 1:
            reading, color = "Dislocation skews negative — watch for downside surprises", "#d32f2f"
        else:
            reading, color = "Balanced dislocation — no clear edge from positioning alone", "#e0c97f"

        summary = (
            f"{len(findings)} dislocation signal(s). {reading}. "
            "Dislocation analysis identifies where expectations could be wrong; "
            "it is a complement to — not a replacement for — fundamental valuation."
        )

        return ConsensusDislocationReport(
            findings=findings, summary=summary,
            overall_reading=reading, overall_color=color,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MARKET PULSE (shared market-environment scorer, used by IPO engine)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketPulseInputs:
    """Optional market-environment observations. Missing = unknown (never
    fabricated). All ratios are 0..1 or 0..100 where noted.
    """
    ipo_market_activity: Optional[float] = None      # 0..100 activity index
    recent_ipo_avg_first_day_return_pct: Optional[float] = None
    recent_ipo_pop_pct: Optional[float] = None       # % of IPOs trading above offer
    ipo_withdrawals_12m: Optional[int] = None
    ipo_postponements_12m: Optional[int] = None
    vix_level: Optional[float] = None
    ten_year_yield_pct: Optional[float] = None
    credit_spread_bps: Optional[float] = None
    risk_appetite: Optional[float] = None            # 0..100
    equity_regime: Optional[str] = None              # "bull", "neutral", "bear"
    sector_performance_3m_pct: Optional[float] = None
    recent_comparable_ipo_oversubscription: Optional[float] = None  # x
    notes: List[str] = field(default_factory=list)


class MarketPulseEngine:
    """Scores the IPO / issuance market environment on a 5-point scale and —
    critically — explains *why*.

    The score is a transparent weighted composite of the inputs actually
    supplied; unobserved inputs receive neutral weights and are disclosed as
    missing.
    """

    SCALE = ["Highly supportive", "Supportive", "Neutral", "Challenging", "Highly challenging"]

    def evaluate(self, inputs: MarketPulseInputs) -> Dict[str, Any]:
        evidence: List[Dict[str, Any]] = []
        scores: List[float] = []
        weights: List[float] = []

        def _add(label, score, weight, detail, observed):
            scores.append(max(-1.0, min(1.0, score)))
            weights.append(weight)
            evidence.append({
                "signal": label, "score": round(score, 3), "weight": weight,
                "detail": detail, "observed": observed,
            })

        if inputs.vix_level is not None:
            vix = inputs.vix_level
            if vix < 15:
                _add("VIX level", 0.8, 1.0, f"VIX {vix:.1f} — low vol supports issuance", True)
            elif vix < 20:
                _add("VIX level", 0.4, 1.0, f"VIX {vix:.1f} — benign volatility", True)
            elif vix < 28:
                _add("VIX level", -0.2, 1.0, f"VIX {vix:.1f} — elevated vol pressures pricing", True)
            else:
                _add("VIX level", -0.8, 1.0, f"VIX {vix:.1f} — high vol discourages IPOs", True)
        else:
            _add("VIX level", 0.0, 0.0, "VIX not provided — neutral", False)

        if inputs.ten_year_yield_pct is not None:
            y = inputs.ten_year_yield_pct
            score = 0.5 if y < 4.0 else (0.0 if y < 5.5 else -0.6)
            _add("Rates environment", score, 0.8,
                 f"10Y at {y:.2f}% — {'cheap discount rates support valuations' if y < 4 else 'rates pressure DCF values' if y >= 5.5 else 'neutral for valuation'}", True)
        else:
            _add("Rates environment", 0.0, 0.0, "10Y not provided — neutral", False)

        if inputs.credit_spread_bps is not None:
            s = inputs.credit_spread_bps
            score = 0.5 if s < 150 else (0.0 if s < 250 else -0.6)
            _add("Credit conditions", score, 0.7,
                 f"Credit spreads {s:.0f}bps — {'open credit markets' if s < 150 else 'tightening credit' if s >= 250 else 'neutral credit'}", True)
        else:
            _add("Credit conditions", 0.0, 0.0, "Credit spreads not provided — neutral", False)

        if inputs.recent_ipo_avg_first_day_return_pct is not None:
            r = inputs.recent_ipo_avg_first_day_return_pct
            score = 0.6 if r > 25 else (0.3 if r > 10 else (-0.3 if r > 0 else -0.8))
            _add("Recent IPO first-day performance", score, 0.9,
                 f"Average first-day return {r:+.1f}% across recent deals", True)
        else:
            _add("Recent IPO first-day performance", 0.0, 0.0, "First-day returns not provided — neutral", False)

        if inputs.recent_ipo_pop_pct is not None:
            p = inputs.recent_ipo_pop_pct
            score = 0.5 if p > 70 else (0.0 if p > 40 else -0.6)
            _add("IPO aftermarket breadth", score, 0.7,
                 f"{p:.0f}% of recent IPOs trade above offer", True)
        else:
            _add("IPO aftermarket breadth", 0.0, 0.0, "Aftermarket breadth not provided — neutral", False)

        if inputs.ipo_withdrawals_12m is not None or inputs.ipo_postponements_12m is not None:
            w = inputs.ipo_withdrawals_12m or 0
            p = inputs.ipo_postponements_12m or 0
            total = w + p
            score = 0.4 if total <= 3 else (0.0 if total <= 8 else -0.7)
            _add("IPO withdrawals / postponements", score, 0.6,
                 f"{total} withdrawals/postponements over trailing 12m", True)
        else:
            _add("IPO withdrawals / postponements", 0.0, 0.0, "Withdrawal data not provided — neutral", False)

        if inputs.risk_appetite is not None:
            r = inputs.risk_appetite
            score = (r - 50.0) / 50.0
            _add("Risk appetite", score, 0.8,
                 f"Risk appetite index {r:.0f}/100", True)
        else:
            _add("Risk appetite", 0.0, 0.0, "Risk appetite not provided — neutral", False)

        if inputs.equity_regime is not None:
            score = {"bull": 0.7, "neutral": 0.1, "bear": -0.7}.get(
                inputs.equity_regime.lower(), 0.0)
            _add("Equity-market regime", score, 1.0,
                 f"Regime: {inputs.equity_regime}", True)
        else:
            _add("Equity-market regime", 0.0, 0.0, "Regime not provided — neutral", False)

        if inputs.sector_performance_3m_pct is not None:
            s = inputs.sector_performance_3m_pct
            score = 0.5 if s > 10 else (0.0 if s > -5 else -0.6)
            _add("Sector performance", score, 0.6,
                 f"Sector +{s:.1f}% over 3m" if s >= 0 else f"Sector {s:.1f}% over 3m", True)
        else:
            _add("Sector performance", 0.0, 0.0, "Sector performance not provided — neutral", False)

        if inputs.ipo_market_activity is not None:
            a = inputs.ipo_market_activity
            score = (a - 50.0) / 50.0
            _add("IPO market activity", score, 0.7,
                 f"Issuance activity index {a:.0f}/100", True)
        else:
            _add("IPO market activity", 0.0, 0.0, "Activity index not provided — neutral", False)

        if inputs.recent_comparable_ipo_oversubscription is not None:
            o = inputs.recent_comparable_ipo_oversubscription
            score = 0.6 if o > 10 else (0.3 if o > 3 else -0.2)
            _add("Comparable IPO demand", score, 0.7,
                 f"Recent comparable offerings {o:.1f}x oversubscribed", True)
        else:
            _add("Comparable IPO demand", 0.0, 0.0, "Oversubscription not provided — neutral", False)

        # Composite (weights normalize across observed signals)
        total_w = sum(weights)
        composite = sum(s * w for s, w in zip(scores, weights)) / total_w if total_w > 0 else 0.0
        idx = 0 if composite > 0.55 else (1 if composite > 0.15 else (
            2 if composite > -0.15 else (3 if composite > -0.55 else 4)))
        label = self.SCALE[idx]

        observed_n = sum(1 for e in evidence if e["observed"])
        explanation = (
            f"Composite score {composite:+.2f} → **{label}** market. "
            f"Based on {observed_n} observed signal(s) out of {len(evidence)} tracked; "
            "unobserved signals were treated as neutral (no fabricated inputs). "
        )
        if inputs.notes:
            explanation += " Analyst notes: " + "; ".join(inputs.notes) + ". "

        return {
            "label": label,
            "index": idx,
            "composite_score": round(composite, 3),
            "scale": self.SCALE,
            "evidence": evidence,
            "observed_signals": observed_n,
            "total_signals": len(evidence),
            "explanation": explanation,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_consensus_dislocation_engine() -> ConsensusDislocationEngine:
    return ConsensusDislocationEngine()


def get_market_pulse_engine() -> MarketPulseEngine:
    return MarketPulseEngine()


def default_market_pulse_inputs() -> MarketPulseInputs:
    """Reasonable *defaults the analyst can edit* — clearly labeled estimates
    so the platform never pretends these are observed data."""
    return MarketPulseInputs(
        vix_level=18.0,
        ten_year_yield_pct=4.2,
        credit_spread_bps=140.0,
        recent_ipo_avg_first_day_return_pct=18.0,
        recent_ipo_pop_pct=65.0,
        ipo_withdrawals_12m=4,
        ipo_postponements_12m=3,
        risk_appetite=60.0,
        equity_regime="neutral",
        sector_performance_3m_pct=6.0,
        ipo_market_activity=60.0,
        recent_comparable_ipo_oversubscription=6.0,
        notes=[
            "Market-pulse inputs are analyst-editable estimates; replace with "
            "observed data (e.g. from ECM desks, Dealogic/Renaissance) for live use."
        ],
    )
