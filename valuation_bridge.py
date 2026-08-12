"""
Octavian Cross-Model Valuation Bridge
=====================================

Unifies valuations from every model in the Financial Model Generator into a
single institutional valuation dashboard:

    DCF → CCA → Precedents → IPO → LBO → M&A implied

The bridge never averages blindly. Each methodology is weighted by
*relevance and reliability* for the specific situation, and the output
explains why some methods deserve more weight, flags outliers, and reports
convergence / divergence between methods.

Author: Octavian Terminal
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class BridgeMethod:
    """One valuation methodology feeding the bridge."""
    key: str
    label: str
    low: float
    high: float
    mid: float
    base_weight: float = 1.0        # method-relevance weight (before adjustment)
    source: str = ""                # which model produced it
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValuationBridgeResult:
    methods: List[BridgeMethod]
    weights: Dict[str, float]
    weighted_mid: float
    weighted_range: Tuple[float, float]
    convergence: str
    outliers: List[str]
    drivers: List[str]
    narrative: str = ""
    valuation_statement: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "methods": [m.to_dict() for m in self.methods],
            "weights": self.weights,
            "weighted_mid": self.weighted_mid,
            "weighted_range": list(self.weighted_range),
            "convergence": self.convergence,
            "outliers": self.outliers,
            "drivers": self.drivers,
            "narrative": self.narrative,
            "valuation_statement": self.valuation_statement,
        }


class ValuationBridgeEngine:

    # Method relevance: how much weight a method *deserves* in a given
    # situation, before data-quality adjustment.
    _BASE_WEIGHTS = {
        "dcf": 1.0,
        "comps": 1.0,
        "precedents": 0.8,
        "ipo": 0.8,
        "lbo": 0.7,
        "mna": 0.7,
        "sotp": 0.9,
        "market": 0.6,
    }

    def add_dcf(self, result) -> Optional[BridgeMethod]:
        if result is None:
            return None
        fv = getattr(result, "fair_value_per_share", 0)
        if not fv or fv <= 0:
            return None
        # Range: scenario low/high if present, else ±15%
        lo = getattr(result, "scenario_weighted_value", 0)
        mc_p = getattr(result, "mc_percentiles", {}) or {}
        low = mc_p.get("p25") or fv * 0.85
        high = mc_p.get("p75") or fv * 1.15
        return BridgeMethod(
            key="dcf", label="DCF (fundamental value)",
            low=low, high=high, mid=fv,
            base_weight=self._BASE_WEIGHTS["dcf"],
            source="Discounted Cash Flow",
            note=("Discounted cash-flow value; most reliable when cash flows are "
                  "established and WACC is well-estimated."),
        )

    def add_comps(self, result) -> Optional[BridgeMethod]:
        if result is None:
            return None
        rng = getattr(result, "implied_range", None)
        if not rng or not rng[1]:
            return None
        lo, hi = float(rng[0]), float(rng[1])
        mid = getattr(result, "implied_valuation", {}).get("Blended (weighted)")
        if not mid:
            mid = (lo + hi) / 2
        n_core = len(getattr(result, "peers", []) or [])
        return BridgeMethod(
            key="comps", label="Trading comps (market pricing)",
            low=lo, high=hi, mid=mid,
            base_weight=self._BASE_WEIGHTS["comps"] * min(1.0, n_core / 5.0 + 0.3),
            source=f"Comparable company analysis ({n_core} peers)",
            note="Reflects how the market currently prices comparable businesses — embeds sentiment.",
        )

    def add_precedents(self, result) -> Optional[BridgeMethod]:
        if result is None:
            return None
        rng = getattr(result, "adjusted_range", None)
        if not rng or not rng[1]:
            return None
        lo, hi = float(rng[0]), float(rng[1])
        mid = getattr(result, "implied_valuation", {}).get("Blended (weighted)")
        if not mid:
            mid = (lo + hi) / 2
        n = len(getattr(result, "core_deals", []) or [])
        return BridgeMethod(
            key="precedents", label="Precedent transactions (control value)",
            low=lo, high=hi, mid=mid,
            base_weight=self._BASE_WEIGHTS["precedents"] * min(1.0, n / 4.0 + 0.25),
            source=f"Precedent transactions ({n} core deals)",
            note="Control value — typically carries a premium over trading value.",
        )

    def add_ipo(self, result) -> Optional[BridgeMethod]:
        if result is None:
            return None
        rng = getattr(result, "price_range", None)
        if not rng or not rng[1]:
            return None
        lo, hi = float(rng[0]), float(rng[1])
        mid = getattr(result, "implied_ipo_price", 0) or (lo + hi) / 2
        return BridgeMethod(
            key="ipo", label="IPO pricing (public-market value)",
            low=lo, high=hi, mid=mid,
            base_weight=self._BASE_WEIGHTS["ipo"],
            source="IPO analysis & underwriting engine",
            note="Public-market value post-IPO; discount to comp value embeds aftermarket risk.",
        )

    def add_lbo(self, result, entry_multiple: Optional[float] = None,
                shares: float = 1.0) -> Optional[BridgeMethod]:
        """LBO-derived implied value: what a sponsor could pay and still hit
        its hurdle (back-solve the entry multiple that yields target IRR)."""
        if result is None:
            return None
        # Simplified back-solve on the sponsor's reported return profile
        irr = getattr(result, "irr", 0) or 0
        moic = getattr(result, "moic", 0) or 0
        entry_ev = getattr(result, "purchase_price", 0)
        net_debt = getattr(result, "debt_amount", 0) or 0
        if not entry_ev:
            return None
        target_irr = 0.20
        # If sponsor IRR > hurdle, there is room to pay more: scale up to the
        # price that would just clear the hurdle (approximation).
        scale = (target_irr / max(irr, 1e-9)) ** 0.5 if irr > 0 else 1.0
        mid_ev = entry_ev * min(scale, 1.25)
        lo_ev, hi_ev = entry_ev, max(entry_ev * 1.15, mid_ev)
        eq_value = (mid_ev - net_debt) / max(shares, 1e-9)
        return BridgeMethod(
            key="lbo", label="LBO-derived value (sponsor view)",
            low=(lo_ev - net_debt) / max(shares, 1e-9),
            high=(hi_ev - net_debt) / max(shares, 1e-9),
            mid=max(eq_value, 0.0),
            base_weight=self._BASE_WEIGHTS["lbo"],
            source="Leveraged buyout underwriting",
            note="Maximum price a financial sponsor could pay while clearing hurdle rates.",
        )

    def add_mna(self, result, shares: float = 1.0) -> Optional[BridgeMethod]:
        """M&A implied value: offer price the target could command."""
        if result is None:
            return None
        offer = getattr(result, "offer_price", 0)
        if not offer:
            return None
        prem = getattr(getattr(result, "assumptions", None), "offer_premium", 0.3) or 0.3
        # Standalone value = offer price without premium
        standalone = offer / (1 + prem) if (1 + prem) > 0 else offer
        return BridgeMethod(
            key="mna", label="M&A implied value (takeout value)",
            low=standalone, high=offer, mid=(standalone + offer) / 2,
            base_weight=self._BASE_WEIGHTS["mna"],
            source="M&A accretion / dilution model",
            note="Takeout value including control premium — a ceiling on minority value.",
        )

    def add_market(self, price: float) -> BridgeMethod:
        return BridgeMethod(
            key="market", label="Market price (observed)",
            low=price, high=price, mid=price,
            base_weight=self._BASE_WEIGHTS["market"],
            source="Live market",
            note="Observed trading price — the reference point for all valuation gaps.",
        )

    def run(self, methods: List[BridgeMethod], current_price: float = 0.0
            ) -> ValuationBridgeResult:
        if not methods:
            return ValuationBridgeResult(
                methods=[], weights={}, weighted_mid=0.0, weighted_range=(0, 0),
                convergence="No valuation methods available",
                outliers=[], drivers=[], valuation_statement="Insufficient data.",
            )

        # Weights: base relevance × data quality (range width penalty)
        weights: Dict[str, float] = {}
        for m in methods:
            w = m.base_weight
            span = (m.high - m.low) / max(m.mid, 1e-9) if m.mid else 2.0
            # Narrow ranges (high precision) get a small boost; absurdly wide
            # ranges get penalized (uninformative methods).
            if 0 < span < 0.4:
                w *= 1.15
            elif span > 1.0:
                w *= 0.6
            weights[m.key] = max(w, 0.05)

        total_w = sum(weights.values())
        norm = {k: v / total_w for k, v in weights.items()}
        weighted_mid = sum(m.mid * norm[m.key] for m in methods)
        lo = sum(m.low * norm[m.key] for m in methods)
        hi = sum(m.high * norm[m.key] for m in methods)

        # Outliers: methods more than 1.5 IQR away from the weighted mid
        mids = [m.mid for m in methods]
        arr = np.array(mids)
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = max(q3 - q1, 1e-9)
        outliers = [
            m.label for m in methods
            if abs(m.mid - weighted_mid) > 1.5 * iqr and len(methods) >= 3
        ]

        # Convergence reading
        spread = (max(mids) - min(mids)) / max(weighted_mid, 1e-9) if len(mids) > 1 else 0
        if spread < 0.15:
            convergence = "High convergence — methodologies align closely; valuation conclusion is robust."
        elif spread < 0.40:
            convergence = "Moderate convergence — methodologies broadly agree; range driven by method choice."
        else:
            convergence = "Material divergence — methodologies disagree; understand the drivers before weighting."

        # Drivers: which methods move the weighted answer most
        contrib = sorted(
            ((m.label, m.mid * norm[m.key]) for m in methods),
            key=lambda x: -abs(x[1] - weighted_mid * 0), reverse=True,
        )
        drivers = [f"{label}: contributes ${abs(v):,.2f} to the weighted value" for label, v in contrib[:3]]

        # Narrative + statement
        narrative = self._narrative(methods, norm, weighted_mid, current_price)
        if current_price > 0 and weighted_mid > 0:
            gap = weighted_mid / current_price - 1
            statement = (
                f"Cross-model weighted fair value is ${weighted_mid:.2f} vs market "
                f"${current_price:.2f} ({gap:+.1%}). "
                + ("The models suggest the market is undervaluing the company."
                   if gap > 0.15 else
                   "The models suggest the market is overvaluing the company."
                   if gap < -0.15 else
                   "The models and market are broadly aligned.")
            )
        else:
            statement = f"Cross-model weighted fair value is ${weighted_mid:.2f} per share."

        return ValuationBridgeResult(
            methods=methods, weights=norm, weighted_mid=weighted_mid,
            weighted_range=(lo, hi), convergence=convergence,
            outliers=outliers, drivers=drivers,
            narrative=narrative, valuation_statement=statement,
        )

    def _narrative(self, methods, norm, mid, price) -> str:
        top = sorted(methods, key=lambda m: -norm[m.key])[:2]
        parts = [
            f"The weighted valuation of ${mid:.2f} is driven primarily by "
            + " and ".join(f"{m.label} ({norm[m.key]:.0%} weight)" for m in top) + ".",
            "Weights reflect both methodological relevance to this situation and "
            "the precision of each method's range.",
        ]
        if price > 0:
            parts.append(f"Market price of ${price:.2f} provides the reference point for the gap analysis.")
        return " ".join(parts)


_bridge_engine = None


def get_valuation_bridge() -> ValuationBridgeEngine:
    global _bridge_engine
    if _bridge_engine is None:
        _bridge_engine = ValuationBridgeEngine()
    return _bridge_engine
