"""
Octavian Comparable Company Analysis (Trading Comps) Engine
===========================================================

An institutional CCA engine that goes well beyond a multiples table:

* peer universe with explicit *relevance scores* and reasons
* full multiples set (EV/Rev, EV/EBITDA, EV/EBIT, P/E, P/FCF, EV/FCF)
* distributional stats (mean / median / quartiles / high / low)
* implied valuation of the subject company from the comp set
* premium / discount positioning analysis
* growth-adjusted and margin-adjusted comparison
* multiple-vs-growth and multiple-vs-margin regressions
* outlier detection
* written conclusion explaining *why* the subject should trade where it does

Author: Octavian Terminal
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np


@dataclass
class CompsCompany:
    """One comparable company (or the subject). All monetary values $M."""
    ticker: str
    name: str = ""
    sector: str = ""
    geography: str = ""
    business_model: str = ""          # e.g. "SaaS", "Hardware", "Services"
    revenue: float = 0.0
    revenue_growth_pct: float = 0.0
    gross_margin_pct: float = 0.0
    ebitda: float = 0.0
    ebitda_margin_pct: float = 0.0
    ebit: float = 0.0
    fcf: float = 0.0
    net_income: float = 0.0
    net_debt: float = 0.0             # + = net debt, - = net cash
    market_cap: float = 0.0
    shares_outstanding: float = 0.0
    price: float = 0.0
    recurring_revenue_pct: float = 0.0  # % of revenue that is recurring
    capital_intensity: float = 0.5      # 0..1 proxy (capex/sales)
    user_weight: float = 1.0            # analyst override of relevance weight

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def enterprise_value(self) -> float:
        return self.market_cap + self.net_debt

    def multiples(self) -> Dict[str, Optional[float]]:
        ev = self.enterprise_value
        return {
            "EV/Revenue": ev / self.revenue if self.revenue > 0 else None,
            "EV/EBITDA": ev / self.ebitda if self.ebitda > 0 else None,
            "EV/EBIT": ev / self.ebit if self.ebit > 0 else None,
            "EV/FCF": ev / self.fcf if self.fcf > 0 else None,
            "P/E": self.market_cap / self.net_income if self.net_income > 0 else None,
            "P/FCF": self.market_cap / self.fcf if self.fcf > 0 else None,
        }


@dataclass
class CompsInputs:
    subject: CompsCompany
    peers: List[CompsCompany] = field(default_factory=list)
    # Manual override of peer relevance weights (ticker -> 0..1). Default
    # relevance is computed by the engine.
    relevance_overrides: Dict[str, float] = field(default_factory=dict)
    # Growth/margin adjustment preferences
    adjust_for_growth: bool = True
    adjust_for_margin: bool = True


@dataclass
class CompsResult:
    subject: CompsCompany
    peers: List[CompsCompany]
    peers_df: pd.DataFrame
    relevance: Dict[str, Dict[str, Any]]          # ticker -> {score, reasons}
    multiples_df: pd.DataFrame                    # ticker x multiple
    stats: Dict[str, Dict[str, float]]            # multiple -> {mean, median, q1, q3, high, low}
    implied_valuation: Dict[str, float]           # method -> implied FV/share for subject
    implied_range: Tuple[float, float]
    premium_discount: Dict[str, Any]
    regressions: Dict[str, Dict[str, float]]      # EV/EBITDA vs growth/margin
    outliers: List[Dict[str, Any]]
    recommendation: str
    conclusion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject.to_dict(),
            "peers": [p.to_dict() for p in self.peers],
            "relevance": self.relevance,
            "stats": self.stats,
            "implied_valuation": self.implied_valuation,
            "implied_range": list(self.implied_range),
            "premium_discount": self.premium_discount,
            "regressions": self.regressions,
            "outliers": self.outliers,
            "recommendation": self.recommendation,
            "conclusion": self.conclusion,
        }


class CompsEngine:

    # Multiples the engine computes (with which to derive implied valuation)
    _MULTIPLES = ["EV/Revenue", "EV/EBITDA", "EV/EBIT", "EV/FCF", "P/E", "P/FCF"]

    # ── Relevance scoring ────────────────────────────────────────────────────
    def _relevance_score(self, subj: CompsCompany, peer: CompsCompany,
                         override: Optional[float] = None) -> Tuple[float, List[str]]:
        if override is not None:
            return max(0.0, min(1.0, override)), ["Analyst override applied"]
        reasons = []
        score = 1.0
        decay = 0.25

        if subj.sector and peer.sector:
            if subj.sector.lower() == peer.sector.lower():
                reasons.append(f"Sector match: {peer.sector}")
            else:
                score -= decay
                reasons.append(f"Sector mismatch: {peer.sector} vs {subj.sector}")

        if subj.business_model and peer.business_model:
            if subj.business_model.lower() == peer.business_model.lower():
                reasons.append(f"Business model match: {peer.business_model}")
            else:
                score -= decay * 0.8
                reasons.append(f"Business model differs: {peer.business_model}")

        # Growth proximity
        if subj.revenue_growth_pct > 0:
            g_diff = abs(peer.revenue_growth_pct - subj.revenue_growth_pct)
            if g_diff > 15:
                score -= decay * 0.7
                reasons.append(f"Growth profile differs materially ({peer.revenue_growth_pct:.0f}% vs {subj.revenue_growth_pct:.0f}%)")
            elif g_diff <= 5:
                reasons.append(f"Growth profile comparable ({peer.revenue_growth_pct:.0f}%)")

        # Margin proximity
        if subj.ebitda_margin_pct > 0:
            m_diff = abs(peer.ebitda_margin_pct - subj.ebitda_margin_pct)
            if m_diff > 15:
                score -= decay * 0.7
                reasons.append(f"Margin profile differs ({peer.ebitda_margin_pct:.0f}% vs {subj.ebitda_margin_pct:.0f}%)")
            elif m_diff <= 5:
                reasons.append(f"Margin profile comparable ({peer.ebitda_margin_pct:.0f}%)")

        # Scale proximity (log distance on market cap)
        if subj.market_cap > 0 and peer.market_cap > 0:
            scale_ratio = max(peer.market_cap, subj.market_cap) / max(min(peer.market_cap, subj.market_cap), 1e-9)
            if scale_ratio > 20:
                score -= decay * 0.5
                reasons.append(f"Scale mismatch (mcaps differ >20x)")
            elif scale_ratio <= 5:
                reasons.append(f"Scale comparable")

        # Geography
        if subj.geography and peer.geography and subj.geography.lower() != peer.geography.lower():
            score -= decay * 0.3
            reasons.append(f"Geography differs: {peer.geography}")

        # Recurring revenue similarity (when both known)
        if subj.recurring_revenue_pct > 0 and peer.recurring_revenue_pct > 0:
            if abs(peer.recurring_revenue_pct - subj.recurring_revenue_pct) > 40:
                score -= decay * 0.4
                reasons.append(f"Recurring-revenue mix differs ({peer.recurring_revenue_pct:.0f}%)")

        score = max(0.05, min(1.0, score))
        if score >= 0.75:
            reasons.insert(0, "High relevance to subject")
        elif score >= 0.5:
            reasons.insert(0, "Moderate relevance to subject")
        else:
            reasons.insert(0, "Limited relevance — consider excluding")
        return round(score, 3), reasons

    # ── Stats ────────────────────────────────────────────────────────────────
    def _stats(self, values: List[Optional[float]]) -> Dict[str, float]:
        vals = sorted(v for v in values if v is not None and np.isfinite(v) and v > 0)
        if not vals:
            return {"mean": None, "median": None, "q1": None, "q3": None,
                    "high": None, "low": None, "count": 0}
        arr = np.array(vals)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "q1": float(np.percentile(arr, 25)),
            "q3": float(np.percentile(arr, 75)),
            "high": float(arr.max()),
            "low": float(arr.min()),
            "count": int(len(arr)),
        }

    # ── Regression-based valuation ───────────────────────────────────────────
    def _regression(self, x: List[float], y: List[float]) -> Optional[Dict[str, float]]:
        """Simple OLS y = a + b*x with R². Returns None when degenerate."""
        pairs = [(float(a), float(b)) for a, b in zip(x, y)
                 if a is not None and b is not None
                 and np.isfinite(a) and np.isfinite(b)]
        if len(pairs) < 3:
            return None
        xs = np.array([p[0] for p in pairs])
        ys = np.array([p[1] for p in pairs])
        if np.std(xs) < 1e-12:
            return None
        b, a = np.polyfit(xs, ys, 1)
        y_pred = a + b * xs
        ss_res = float(np.sum((ys - y_pred) ** 2))
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return {"slope": float(b), "intercept": float(a), "r2": float(r2),
                "n": len(pairs)}

    # ── Main run ─────────────────────────────────────────────────────────────
    def run_comps(self, inputs: CompsInputs) -> CompsResult:
        subj = inputs.subject
        peers = inputs.peers or []

        # Relevance
        relevance: Dict[str, Dict[str, Any]] = {}
        for p in peers:
            override = inputs.relevance_overrides.get(p.ticker)
            score, reasons = self._relevance_score(subj, p, override)
            relevance[p.ticker] = {"score": score, "reasons": reasons}

        # Core set: relevance >= 0.5 (or analyst override)
        core = [p for p in peers if relevance[p.ticker]["score"] >= 0.5]
        if not core:
            core = peers  # never fall to zero peers if user supplied some

        # Build peer table
        rows = []
        for p in core:
            m = p.multiples()
            rows.append({
                "Ticker": p.ticker, "Sector": p.sector,
                "Revenue ($M)": p.revenue,
                "Growth %": p.revenue_growth_pct,
                "EBITDA Margin %": p.ebitda_margin_pct,
                "Market Cap ($M)": p.market_cap,
                "Net Debt ($M)": p.net_debt,
                "EV ($M)": p.enterprise_value,
                **m,
                "Relevance": relevance[p.ticker]["score"],
            })
        peers_df = pd.DataFrame(rows) if rows else pd.DataFrame()

        # Multiples matrix (subject last row) for the UI table
        mult_rows = []
        for p in core + [subj]:
            m = p.multiples()
            row = {"Ticker": p.ticker, **{k: (round(v, 2) if v is not None else None) for k, v in m.items()}}
            row["is_subject"] = (p.ticker == subj.ticker)
            mult_rows.append(row)
        multiples_df = pd.DataFrame(mult_rows)

        # Distributional stats per multiple (core set only)
        stats: Dict[str, Dict[str, float]] = {}
        for mult in self._MULTIPLES:
            vals = [p.multiples()[mult] for p in core]
            stats[mult] = self._stats(vals)

        # ── Implied valuation of the subject ─────────────────────────────────
        # EV/Revenue and EV/EBITDA carry the most weight (typical IB practice);
        # P/E and P/FCF convert from equity-value multiples. We also blend a
        # growth-adjusted and a regression-adjusted read where possible.
        implied: Dict[str, float] = {}
        weights: Dict[str, float] = {}

        def _apply_ev_multiple(mult_key: str, weight: float):
            s = stats.get(mult_key, {})
            med = s.get("median")
            if med and subj.revenue > 0 and mult_key == "EV/Revenue":
                ev = subj.revenue * med
                fv = (ev - subj.net_debt) / max(subj.shares_outstanding, 1e-9)
                implied[f"{mult_key} (median)"] = max(0.0, fv)
                weights[f"{mult_key} (median)"] = weight
            elif med and subj.ebitda > 0 and mult_key == "EV/EBITDA":
                ev = subj.ebitda * med
                fv = (ev - subj.net_debt) / max(subj.shares_outstanding, 1e-9)
                implied[f"{mult_key} (median)"] = max(0.0, fv)
                weights[f"{mult_key} (median)"] = weight
            elif med and subj.ebit > 0 and mult_key == "EV/EBIT":
                ev = subj.ebit * med
                fv = (ev - subj.net_debt) / max(subj.shares_outstanding, 1e-9)
                implied[f"{mult_key} (median)"] = max(0.0, fv)
                weights[f"{mult_key} (median)"] = weight
            elif med and subj.fcf > 0 and mult_key == "EV/FCF":
                ev = subj.fcf * med
                fv = (ev - subj.net_debt) / max(subj.shares_outstanding, 1e-9)
                implied[f"{mult_key} (median)"] = max(0.0, fv)
                weights[f"{mult_key} (median)"] = weight

        _apply_ev_multiple("EV/Revenue", 1.0)
        _apply_ev_multiple("EV/EBITDA", 1.2)
        _apply_ev_multiple("EV/EBIT", 0.8)
        _apply_ev_multiple("EV/FCF", 0.7)

        s_pe = stats.get("P/E", {})
        if s_pe.get("median") and subj.net_income > 0:
            fv = (subj.net_income * s_pe["median"]) / max(subj.shares_outstanding, 1e-9)
            implied["P/E (median)"] = max(0.0, fv)
            weights["P/E (median)"] = 0.9
        s_pfcf = stats.get("P/FCF", {})
        if s_pfcf.get("median") and subj.fcf > 0:
            fv = (subj.fcf * s_pfcf["median"]) / max(subj.shares_outstanding, 1e-9)
            implied["P/FCF (median)"] = max(0.0, fv)
            weights["P/FCF (median)"] = 0.7

        # Growth-adjusted: PEG-style — premium/discount to median multiple
        if inputs.adjust_for_growth and stats.get("EV/EBITDA", {}).get("median"):
            med = stats["EV/EBITDA"]["median"]
            peer_g = np.mean([p.revenue_growth_pct for p in core]) if core else subj.revenue_growth_pct
            if peer_g > 0 and subj.revenue_growth_pct > 0:
                growth_adj = med * (1 + (subj.revenue_growth_pct - peer_g) / 100.0)
                ev = subj.ebitda * max(growth_adj, 0.5)
                fv = (ev - subj.net_debt) / max(subj.shares_outstanding, 1e-9)
                implied["Growth-adjusted EV/EBITDA"] = max(0.0, fv)
                weights["Growth-adjusted EV/EBITDA"] = 0.9

        # Regression-based: EV/EBITDA vs growth
        if inputs.adjust_for_growth and len(core) >= 3:
            reg = self._regression(
                [p.revenue_growth_pct for p in core],
                [p.multiples()["EV/EBITDA"] for p in core],
            )
            if reg and reg["n"] >= 3:
                pred = reg["intercept"] + reg["slope"] * subj.revenue_growth_pct
                ev = subj.ebitda * max(pred, 0.5)
                fv = (ev - subj.net_debt) / max(subj.shares_outstanding, 1e-9)
                implied["Regression EV/EBITDA (growth)"] = max(0.0, fv)
                weights["Regression EV/EBITDA (growth)"] = 0.6
            self._reg_ev_ebitda = reg  # stash for output

        # Margin-adjusted regression
        if inputs.adjust_for_margin and len(core) >= 3:
            reg = self._regression(
                [p.ebitda_margin_pct for p in core],
                [p.multiples()["EV/EBITDA"] for p in core],
            )
            if reg and reg["n"] >= 3:
                pred = reg["intercept"] + reg["slope"] * subj.ebitda_margin_pct
                ev = subj.ebitda * max(pred, 0.5)
                fv = (ev - subj.net_debt) / max(subj.shares_outstanding, 1e-9)
                implied["Regression EV/EBITDA (margin)"] = max(0.0, fv)
                weights["Regression EV/EBITDA (margin)"] = 0.6
            self._reg_ev_ebitda_margin = reg

        # Blended value (relevance- and method-weighted, not a naive average)
        if implied:
            total_w = sum(weights.values())
            blended = sum(v * weights[k] for k, v in implied.items()) / total_w
        else:
            blended = subj.price if subj.price > 0 else 0.0
        implied["Blended (weighted)"] = blended

        if implied:
            non_blend = [v for k, v in implied.items() if k != "Blended (weighted)" and v > 0]
            lo, hi = (min(non_blend), max(non_blend)) if non_blend else (blended, blended)
        else:
            lo = hi = blended
        implied_range = (lo, hi)

        # ── Premium / discount positioning ───────────────────────────────────
        premium_discount: Dict[str, Any] = {}
        if subj.price > 0:
            premium_discount["premium_to_blended"] = blended / subj.price - 1.0
            premium_discount["premium_to_median_ev_ebitda"] = (
                (subj.ebitda * stats.get("EV/EBITDA", {}).get("median", 0) - subj.net_debt)
                / max(subj.shares_outstanding, 1e-9) / subj.price - 1.0
                if stats.get("EV/EBITDA", {}).get("median") and subj.ebitda > 0 else None
            )
        premium_discount["current_price"] = subj.price

        # ── Outlier detection (IQR rule per multiple, degenerate-IQR safe) ────
        # When the peer set is clustered (IQR ≈ 0), a fixed floor of 5% of the
        # median keeps the rule meaningful instead of silently skipping.
        outliers: List[Dict[str, Any]] = []
        for mult in self._MULTIPLES:
            s = stats.get(mult, {})
            q1, q3, median = s.get("q1"), s.get("q3"), s.get("median")
            if q1 is None or q3 is None or median is None:
                continue
            iqr = max(q3 - q1, max(median * 0.05, 0.1))
            for p in core:
                v = p.multiples()[mult]
                if v is None:
                    continue
                if v < q1 - 1.5 * iqr or v > q3 + 1.5 * iqr:
                    outliers.append({
                        "ticker": p.ticker, "multiple": mult,
                        "value": round(v, 2),
                        "band": f"Q1 {q1:.1f} – Q3 {q3:.1f}",
                        "note": "Statistical outlier by IQR rule — verify financials or exclude from the core set.",
                    })

        # ── Recommendation & conclusion ──────────────────────────────────────
        rec = self._build_recommendation(subj, blended, implied_range, stats, core)
        return CompsResult(
            subject=subj,
            peers=core,
            peers_df=peers_df,
            relevance=relevance,
            multiples_df=multiples_df,
            stats=stats,
            implied_valuation=implied,
            implied_range=implied_range,
            premium_discount=premium_discount,
            regressions={
                "ev_ebitda_vs_growth": getattr(self, "_reg_ev_ebitda", None) or {},
                "ev_ebitda_vs_margin": getattr(self, "_reg_ev_ebitda_margin", None) or {},
            },
            outliers=outliers,
            recommendation=rec["label"],
            conclusion=rec["text"],
        )

    def _build_recommendation(self, subj: CompsCompany, blended: float,
                              implied_range: Tuple[float, float],
                              stats: Dict, core: List[CompsCompany]) -> Dict[str, str]:
        parts = []
        if subj.price > 0 and blended > 0:
            pos = blended / subj.price - 1.0
            if pos > 0.15:
                label = "Undervalued vs comps"
                parts.append(
                    f"{subj.ticker} trades at ${subj.price:.2f} vs a comp-blended value of "
                    f"${blended:.2f} ({pos:+.1f}%), implying the market applies a discount "
                    f"to the peer set.")
            elif pos < -0.15:
                label = "Overvalued vs comps"
                parts.append(
                    f"{subj.ticker} trades at ${subj.price:.2f} vs a comp-blended value of "
                    f"${blended:.2f} ({pos:+.1f}%), implying a premium to the peer set that "
                    f"must be justified by growth, margins or strategic characteristics.")
            else:
                label = "In line with comps"
                parts.append(
                    f"{subj.ticker} trades at ${subj.price:.2f} vs a comp-blended value of "
                    f"${blended:.2f} ({pos:+.1f}%) — within the peer range.")
        else:
            label = "Comps positioning indeterminate"
            parts.append("Insufficient price data to position the subject within the peer set.")

        if core:
            med_growth = np.median([p.revenue_growth_pct for p in core])
            med_margin = np.median([p.ebitda_margin_pct for p in core])
            parts.append(
                f"Relative to the {len(core)}-company core set, the subject's "
                f"{subj.revenue_growth_pct:.1f}% growth and {subj.ebitda_margin_pct:.1f}% "
                f"EBITDA margin compare with medians of {med_growth:.1f}% and {med_margin:.1f}%."
            )
        parts.append(
            f"The defensible comp-driven valuation range is ${implied_range[0]:.2f} – "
            f"${implied_range[1]:.2f} per share. Trading comps measure *market* pricing of "
            f"comparable businesses; they embed sentiment and should be triangulated with "
            f"DCF, precedents and transaction logic before forming a conclusion."
        )
        return {"label": label, "text": " ".join(parts)}


_comps_engine = None


def get_comps_engine() -> CompsEngine:
    global _comps_engine
    if _comps_engine is None:
        _comps_engine = CompsEngine()
    return _comps_engine
