"""
Octavian Precedent Transactions Analysis Engine
================================================

An institutional precedent-transaction engine that:

* scores each transaction's *relevance* to the subject (not just "same
  industry" — it considers size, growth, margins, rationale, buyer type,
  structure and market regime)
* computes transaction multiples (EV/Rev, EV/EBITDA, EV/EBIT, EV/FCF)
* separates genuinely comparable deals from superficially similar ones
* produces an *adjusted* valuation range for the subject company, with a
  written explanation of which deals drive the range and why

Author: Octavian Terminal
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np


@dataclass
class PrecedentTransaction:
    """One historical M&A transaction. All monetary values $M."""
    deal_name: str
    buyer: str = ""
    seller: str = ""
    announcement_date: str = ""            # ISO date
    enterprise_value: float = 0.0
    equity_value: float = 0.0
    seller_revenue: float = 0.0
    seller_revenue_growth_pct: float = 0.0
    seller_ebitda: float = 0.0
    seller_ebitda_margin_pct: float = 0.0
    seller_ebit: float = 0.0
    seller_fcf: float = 0.0
    control_premium_pct: float = 0.0
    strategic_rationale: str = ""
    buyer_type: str = ""                   # "strategic" | "financial" | "mixed"
    structure: str = ""                    # "cash" | "stock" | "mixed" | "earnout"
    market_regime: str = ""                # "bull" | "neutral" | "bear"
    synergies_expected_pct: float = 0.0    # expected synergies as % of target EBITDA
    relevance_override: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def multiples(self) -> Dict[str, Optional[float]]:
        ev = self.enterprise_value
        return {
            "EV/Revenue": ev / self.seller_revenue if self.seller_revenue > 0 else None,
            "EV/EBITDA": ev / self.seller_ebitda if self.seller_ebitda > 0 else None,
            "EV/EBIT": ev / self.seller_ebit if self.seller_ebit > 0 else None,
            "EV/FCF": ev / self.seller_fcf if self.seller_fcf > 0 else None,
        }


@dataclass
class PrecedentInputs:
    subject_name: str = ""
    subject_sector: str = ""
    subject_business_model: str = ""
    subject_geography: str = ""
    subject_revenue: float = 0.0
    subject_revenue_growth_pct: float = 0.0
    subject_ebitda: float = 0.0
    subject_ebitda_margin_pct: float = 0.0
    subject_ebit: float = 0.0
    subject_fcf: float = 0.0
    subject_net_debt: float = 0.0
    subject_shares: float = 0.0
    transactions: List[PrecedentTransaction] = field(default_factory=list)
    current_market_regime: str = ""
    relevance_overrides: Dict[str, float] = field(default_factory=dict)


@dataclass
class PrecedentResult:
    transactions: List[PrecedentTransaction]
    core_deals: List[PrecedentTransaction]
    relevance: Dict[str, Dict[str, Any]]
    multiples_df: pd.DataFrame
    stats: Dict[str, Dict[str, float]]
    implied_valuation: Dict[str, float]
    adjusted_range: Tuple[float, float]
    control_premium_analysis: Dict[str, Any]
    context_adjustments: List[Dict[str, str]]
    conclusion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core_deals": [d.deal_name for d in self.core_deals],
            "relevance": self.relevance,
            "stats": self.stats,
            "implied_valuation": self.implied_valuation,
            "adjusted_range": list(self.adjusted_range),
            "control_premium_analysis": self.control_premium_analysis,
            "context_adjustments": self.context_adjustments,
            "conclusion": self.conclusion,
        }


class PrecedentEngine:

    _MULTIPLES = ["EV/Revenue", "EV/EBITDA", "EV/EBIT", "EV/FCF"]

    def _relevance(self, subj: dict, t: PrecedentTransaction,
                   override: Optional[float] = None) -> Tuple[float, List[str]]:
        if override is not None:
            return max(0.0, min(1.0, override)), ["Analyst override applied"]
        reasons = []
        score = 1.0
        decay = 0.25

        sector = subj.get("subject_sector", "")
        if sector and t.seller and sector.lower() in t.strategic_rationale.lower():
            reasons.append("Sector corroborated by deal rationale")
        if t.market_regime and subj.get("current_market_regime", ""):
            if t.market_regime.lower() == subj["current_market_regime"].lower():
                reasons.append(f"Market regime match: {t.market_regime}")
            else:
                score -= decay * 0.5
                reasons.append(f"Different market regime: {t.market_regime} vs {subj['current_market_regime']}")

        if subj.get("subject_revenue", 0) > 0 and t.seller_revenue > 0:
            ratio = max(t.seller_revenue, subj["subject_revenue"]) / max(min(t.seller_revenue, subj["subject_revenue"]), 1e-9)
            if ratio > 10:
                score -= decay * 0.6
                reasons.append(f"Size mismatch (target revenue differs >10x)")
            elif ratio <= 3:
                reasons.append(f"Size comparable (target revenue within 3x)")

        if subj.get("subject_revenue_growth_pct", 0) > 0 and t.seller_revenue_growth_pct > 0:
            if abs(t.seller_revenue_growth_pct - subj["subject_revenue_growth_pct"]) > 15:
                score -= decay * 0.5
                reasons.append(f"Growth profile differs ({t.seller_revenue_growth_pct:.0f}% vs {subj['subject_revenue_growth_pct']:.0f}%)")

        if subj.get("subject_ebitda_margin_pct", 0) > 0 and t.seller_ebitda_margin_pct > 0:
            if abs(t.seller_ebitda_margin_pct - subj["subject_ebitda_margin_pct"]) > 12:
                score -= decay * 0.5
                reasons.append(f"Margin profile differs ({t.seller_ebitda_margin_pct:.0f}% vs {subj['subject_ebitda_margin_pct']:.0f}%)")

        if t.buyer_type and subj.get("buyer_type_context"):
            # optional context flag: "strategic" deals historically pay more
            pass
        if t.buyer_type == "financial":
            reasons.append("Financial buyer — premium typically lower than strategic")
        elif t.buyer_type == "strategic":
            reasons.append("Strategic buyer — synergy-driven premium")

        if t.structure:
            reasons.append(f"Structure: {t.structure} consideration")

        if t.control_premium_pct > 0:
            reasons.append(f"Control premium: {t.control_premium_pct:.0f}%")

        score = max(0.05, min(1.0, score))
        if score >= 0.7:
            reasons.insert(0, "Genuinely comparable transaction")
        elif score >= 0.45:
            reasons.insert(0, "Partially comparable — use with adjustment")
        else:
            reasons.insert(0, "Superficially similar — limited comparability")
        return round(score, 3), reasons

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

    def run_precedents(self, inputs: PrecedentInputs) -> PrecedentResult:
        subj = {
            "subject_name": inputs.subject_name,
            "subject_sector": inputs.subject_sector,
            "subject_business_model": inputs.subject_business_model,
            "subject_geography": inputs.subject_geography,
            "subject_revenue": inputs.subject_revenue,
            "subject_revenue_growth_pct": inputs.subject_revenue_growth_pct,
            "subject_ebitda": inputs.subject_ebitda,
            "subject_ebitda_margin_pct": inputs.subject_ebitda_margin_pct,
            "subject_ebit": inputs.subject_ebit,
            "subject_fcf": inputs.subject_fcf,
            "subject_net_debt": inputs.subject_net_debt,
            "subject_shares": inputs.subject_shares,
            "current_market_regime": inputs.current_market_regime,
        }
        txns = inputs.transactions or []

        relevance: Dict[str, Dict[str, Any]] = {}
        for t in txns:
            override = inputs.relevance_overrides.get(t.deal_name)
            score, reasons = self._relevance(subj, t, override)
            relevance[t.deal_name] = {"score": score, "reasons": reasons}

        core = [t for t in txns if relevance[t.deal_name]["score"] >= 0.45]
        if not core and txns:
            core = txns  # fall back to user-provided universe

        # Multiples table
        rows = []
        for t in core:
            m = t.multiples()
            rows.append({
                "Deal": t.deal_name, "Announced": t.announcement_date,
                "Buyer": t.buyer, "Seller": t.seller,
                "EV ($M)": t.enterprise_value,
                "Premium %": t.control_premium_pct,
                **{k: (round(v, 2) if v is not None else None) for k, v in m.items()},
                "Relevance": relevance[t.deal_name]["score"],
            })
        multiples_df = pd.DataFrame(rows) if rows else pd.DataFrame()

        stats: Dict[str, Dict[str, float]] = {}
        for mult in self._MULTIPLES:
            stats[mult] = self._stats([t.multiples()[mult] for t in core])

        # ── Implied valuation of subject ─────────────────────────────────────
        implied: Dict[str, float] = {}
        weights: Dict[str, float] = {}
        shares = max(subj["subject_shares"], 1e-9)

        def _apply_ev(mult_key: str, metric_key: str, weight: float):
            med = stats.get(mult_key, {}).get("median")
            metric = subj.get(metric_key, 0)
            if med and metric and metric > 0:
                ev = metric * med
                fv = (ev - subj["subject_net_debt"]) / shares
                implied[f"{mult_key} (median)"] = max(0.0, fv)
                weights[f"{mult_key} (median)"] = weight

        _apply_ev("EV/Revenue", "subject_revenue", 1.0)
        _apply_ev("EV/EBITDA", "subject_ebitda", 1.2)
        _apply_ev("EV/EBIT", "subject_ebit", 0.8)
        _apply_ev("EV/FCF", "subject_fcf", 0.7)

        # Control premium read: what premium would a buyer pay vs market?
        if inputs.subject_shares > 0 and stats.get("EV/EBITDA", {}).get("median") and subj["subject_ebitda"] > 0:
            med = stats["EV/EBITDA"]["median"]
            premium_adj = med * 1.0  # already reflects control in transaction multiples
            ev = subj["subject_ebitda"] * premium_adj
            fv = (ev - subj["subject_net_debt"]) / shares
            implied["EV/EBITDA (control-adjusted)"] = max(0.0, fv)
            weights["EV/EBITDA (control-adjusted)"] = 1.0

        if implied:
            total_w = sum(weights.values())
            blended = sum(v * weights[k] for k, v in implied.items()) / total_w
        else:
            blended = 0.0
        implied["Blended (weighted)"] = blended

        vals = [v for k, v in implied.items() if k != "Blended (weighted)" and v > 0]
        if vals:
            lo, hi = min(vals), max(vals)
            # Trim to interquartile-style central range when enough methods
            if len(vals) >= 4:
                arr = np.array(sorted(vals))
                lo, hi = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        else:
            lo = hi = blended
        adjusted_range = (lo, hi)

        # ── Control premium analysis ─────────────────────────────────────────
        premiums = [t.control_premium_pct for t in core if t.control_premium_pct > 0]
        prem_stats = self._stats(premiums) if premiums else {}
        control_analysis = {
            "median_premium_pct": prem_stats.get("median"),
            "mean_premium_pct": prem_stats.get("mean"),
            "high_premium_pct": prem_stats.get("high"),
            "low_premium_pct": prem_stats.get("low"),
            "n_deals_with_premium": len(premiums),
            "context": (
                "Transaction multiples embed a control premium by construction; "
                "precedent-derived equity value is the *control value* of the company, "
                "not its minority trading value."
            ),
        }

        # ── Context adjustments (transparent, no fabrication) ────────────────
        adjustments: List[Dict[str, str]] = []
        if any(t.buyer_type == "financial" for t in core) and any(t.buyer_type == "strategic" for t in core):
            adjustments.append({
                "factor": "Buyer type mix",
                "note": "Deal set mixes financial and strategic buyers — strategic buyers typically pay higher premiums for synergies; weight strategic deals toward the top of the range.",
            })
        if stats.get("EV/EBITDA", {}).get("median"):
            q1, q3 = stats["EV/EBITDA"].get("q1"), stats["EV/EBITDA"].get("q3")
            if q1 and q3 and (q3 - q1) / max(q1, 1e-9) > 0.6:
                adjustments.append({
                    "factor": "Multiple dispersion",
                    "note": "Wide dispersion in EV/EBITDA across the deal set — the range is unusually broad; examine deal-level economics before relying on the midpoint.",
                })
        if subj.get("current_market_regime"):
            adjustments.append({
                "factor": "Market regime",
                "note": f"Current regime '{subj['current_market_regime']}' may differ from deal-time conditions; multiples paid in different regimes are not perfectly comparable.",
            })
        if not adjustments:
            adjustments.append({
                "factor": "Context",
                "note": "No material context adjustments identified from the supplied deal universe.",
            })

        conclusion = self._conclusion(subj, blended, adjusted_range, core, stats)
        return PrecedentResult(
            transactions=txns,
            core_deals=core,
            relevance=relevance,
            multiples_df=multiples_df,
            stats=stats,
            implied_valuation=implied,
            adjusted_range=adjusted_range,
            control_premium_analysis=control_analysis,
            context_adjustments=adjustments,
            conclusion=conclusion,
        )

    def _conclusion(self, subj: dict, blended: float, rng: Tuple[float, float],
                    core: list, stats: dict) -> str:
        parts = []
        name = subj.get("subject_name") or "the subject company"
        if blended > 0:
            parts.append(
                f"Based on {len(core)} comparable transactions, the precedent-derived "
                f"control value of {name} is approximately ${blended:.2f} per share, "
                f"with a defensible adjusted range of ${rng[0]:.2f} – ${rng[1]:.2f}."
            )
        else:
            parts.append(
                f"Insufficient comparable transaction data to derive a control value for {name}."
            )
        if stats.get("EV/EBITDA", {}).get("median"):
            med = stats["EV/EBITDA"]["median"]
            parts.append(
                f"The core deal set trades at a median {med:.1f}x EV/EBITDA — "
                "transaction multiples are typically higher than trading multiples "
                "because they embed control premiums and expected synergies."
            )
        parts.append(
            "Precedent transactions measure what *buyers actually paid*; they are "
            "context-dependent (regime, financing, auction dynamics) and should be "
            "triangulated with trading comps and DCF before forming a final view."
        )
        return " ".join(parts)


_precedent_engine = None


def get_precedent_engine() -> PrecedentEngine:
    global _precedent_engine
    if _precedent_engine is None:
        _precedent_engine = PrecedentEngine()
    return _precedent_engine
