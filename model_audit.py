"""
Octavian Model Audit & Quality Control Engine
==============================================

Runs institutional QA checks over any generated model result so every
deliverable carries a visible "Model Integrity" dashboard. Checks are
*additive*: they never mutate the model, and they tolerate results from any
engine (DCF, M&A, LBO, CCA, Precedents, IPO, Valuation Bridge) by inspecting
whatever fields are present.

Check severity conventions:
  PASS   — check succeeded
  WARN   — unusual but internally consistent (documented)
  FAIL   — model integrity issue that must be flagged before the output is used

Author: Octavian Terminal
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


@dataclass
class QACheck:
    module: str
    check: str
    status: str            # PASS | WARN | FAIL
    detail: str = ""
    value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QAReport:
    checks: List[QACheck] = field(default_factory=list)
    summary: str = ""
    status: str = "PASS"   # aggregate: PASS if no FAIL, WARN if any WARN only, FAIL if any FAIL
    pass_count: int = 0
    warn_count: int = 0
    fail_count: int = 0

    def add(self, module: str, check: str, status: str, detail: str = "",
            value: Optional[str] = None):
        self.checks.append(QACheck(module=module, check=check, status=status,
                                   detail=detail, value=value))
        if status == "FAIL":
            self.fail_count += 1
        elif status == "WARN":
            self.warn_count += 1
        else:
            self.pass_count += 1

    def finalize(self) -> "QAReport":
        self.status = "FAIL" if self.fail_count else ("WARN" if self.warn_count else "PASS")
        parts = [f"{self.pass_count} passed"]
        if self.warn_count:
            parts.append(f"{self.warn_count} warning(s)")
        if self.fail_count:
            parts.append(f"{self.fail_count} failed")
        self.summary = (
            f"Model QA: {', '.join(parts)}. "
            + ("All integrity checks passed." if self.status == "PASS"
               else "Review warnings/failures before relying on outputs." if self.status == "WARN"
               else "Integrity failures detected — outputs should not be used without correction.")
        )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checks": [c.to_dict() for c in self.checks],
            "summary": self.summary,
            "status": self.status,
            "pass_count": self.pass_count,
            "warn_count": self.warn_count,
            "fail_count": self.fail_count,
        }


def _safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


class ModelAuditor:
    """Runs structural QA over model results."""

    def audit(self, result: Any, module: str = "model") -> QAReport:
        qa = QAReport()
        r = result

        # ── Generic structural checks ─────────────────────────────────────────
        if r is None:
            qa.add(module, "Result present", "FAIL", "No result object supplied.")
            return qa.finalize()

        if hasattr(r, "sources_uses_check"):
            self._check_sources_uses(qa, module, r.sources_uses_check)
        if hasattr(r, "total_sources") and hasattr(r, "total_uses"):
            if _safe_float(r.total_uses) > 0 and abs(_safe_float(r.total_sources)) > 0:
                ratio = _safe_float(r.total_sources) / _safe_float(r.total_uses)
                qa.add(module, "Sources / Uses ratio",
                       "PASS" if _approx(ratio, 1.0, 1e-4) else "WARN",
                       f"Sources ${r.total_sources:,.1f}M vs Uses ${r.total_uses:,.1f}M (ratio {ratio:.4f}).",
                       f"{ratio:.4f}")

        if hasattr(r, "net_debt") and hasattr(r, "equity_value") and hasattr(r, "enterprise_value"):
            ev = _safe_float(r.enterprise_value)
            nd = _safe_float(r.net_debt)
            eq = _safe_float(r.equity_value)
            if abs(ev) > 0:
                qa.add(module, "EV = Equity + Net debt identity",
                       "PASS" if _approx(ev, eq + nd, max(abs(ev) * 1e-3, 0.5)) else "WARN",
                       f"EV {ev:,.1f} vs Equity {eq:,.1f} + Net debt {nd:,.1f}.")
        return qa.finalize()

    def _check_sources_uses(self, qa: QAReport, module: str, check_value):
        v = _safe_float(check_value)
        if abs(v) < 1e-6:
            qa.add(module, "Sources = Uses check", "PASS",
                   f"Δ = {v:,.2f} (balanced within tolerance).", f"{v:,.4f}")
        elif abs(v) < 1e-2:
            qa.add(module, "Sources = Uses check", "WARN",
                   f"Δ = {v:,.2f} — small residual; verify rounding.", f"{v:,.4f}")
        else:
            qa.add(module, "Sources = Uses check", "FAIL",
                   f"Δ = {v:,.2f} — sources do not reconcile with uses.", f"{v:,.4f}")

    # ── DCF ──────────────────────────────────────────────────────────────────
    def audit_dcf(self, result: Any) -> QAReport:
        qa = QAReport()
        if result is None:
            qa.add("DCF", "Result present", "FAIL", "No DCF result object supplied.")
            return qa.finalize()
        r = result

        ev = _safe_float(getattr(r, "enterprise_value", 0))
        eq = _safe_float(getattr(r, "equity_value", 0))
        nd = _safe_float(getattr(r, "net_debt", 0))
        fv = _safe_float(getattr(r, "fair_value_per_share", 0))
        shares = _safe_float(getattr(getattr(r, "assumptions", None), "shares_outstanding", 0))

        qa.add("DCF", "Valuation bridge balances",
               "PASS" if _approx(ev, eq + nd, max(abs(ev) * 1e-3, 0.5)) else "WARN",
               f"EV {ev:,.1f} = Equity {eq:,.1f} + Net debt {nd:,.1f}.")
        if shares > 0 and abs(eq) > 0:
            implied = eq / shares
            qa.add("DCF", "Fair value / share consistent with equity value",
                   "PASS" if _approx(implied, fv, max(abs(implied) * 1e-3, 0.01)) else "WARN",
                   f"Equity/shares = {implied:.2f} vs reported FV {fv:.2f}.")
        if not getattr(r, "line_items", pd.DataFrame()).empty:
            li = r.line_items
            fcf_rows = [c for c in li.columns if "FCF" in str(c).upper()]
            if fcf_rows:
                try:
                    vals = pd.to_numeric(li[fcf_rows[0]], errors="coerce").dropna()
                    if len(vals) and (vals < -1e9).any():
                        qa.add("DCF", "FCF plausibility", "WARN",
                               "Negative FCFs observed in projection — verify working-capital and capex assumptions.")
                except Exception:
                    pass
        if hasattr(r, "wacc"):
            w = _safe_float(r.wacc)
            g = _safe_float(getattr(getattr(r, "assumptions", None), "terminal_growth_rate", 0))
            # Accept either the assumptions field or a direct attribute
            if not g and hasattr(r, "terminal_growth"):
                g = _safe_float(r.terminal_growth)
            if w > 0 and g >= w:
                qa.add("DCF", "Terminal value validity", "FAIL",
                       f"Terminal growth {g:.2%} ≥ WACC {w:.2%} — Gordon growth is undefined.")
            else:
                qa.add("DCF", "Terminal value validity", "PASS",
                       f"WACC {w:.2%} > terminal growth {g:.2%}.")
        if hasattr(r, "pv_terminal_value") and ev > 0:
            tv_pct = _safe_float(r.pv_terminal_value) / ev
            if tv_pct > 0.85:
                qa.add("DCF", "Terminal value concentration", "WARN",
                       f"PV(TV) = {tv_pct:.0%} of EV — valuation is dominated by the terminal value; "
                       "verify terminal assumptions carefully.")
        return qa.finalize()

    # ── M&A ──────────────────────────────────────────────────────────────────
    def audit_mna(self, result: Any) -> QAReport:
        qa = QAReport()
        if result is None:
            qa.add("M&A", "Result present", "FAIL", "No M&A result object supplied.")
            return qa.finalize()
        r = result
        self._check_sources_uses(qa, "M&A", getattr(r, "sources_uses_check", None))

        contrib = getattr(r, "contribution_analysis", None)
        if contrib is not None and not contrib.empty and "Type" in contrib.columns and "EPS Impact ($)" in contrib.columns:
            drivers = contrib[contrib["Type"] != "Result"]
            base = _safe_float(getattr(getattr(r, "assumptions", None), "acquirer_eps", 0))
            total = _safe_float(drivers["EPS Impact ($)"].sum())
            pf_eps = _safe_float(getattr(r, "pro_forma_eps", 0))
            qa.add("M&A", "EPS bridge sums to pro forma EPS",
                   "PASS" if _approx(base + total, pf_eps, max(abs(pf_eps) * 1e-3, 1e-6)) else "FAIL",
                   f"Standalone {base:.2f} + drivers {total:+.2f} = {base + total:.2f} vs PF EPS {pf_eps:.2f}.")
        if hasattr(r, "accretion_dilution_pct"):
            qa.add("M&A", "Accretion / dilution computed",
                   "PASS", f"{_safe_float(r.accretion_dilution_pct):+.2%}")
        return qa.finalize()

    # ── LBO ──────────────────────────────────────────────────────────────────
    def audit_lbo(self, result: Any) -> QAReport:
        qa = QAReport()
        if result is None:
            qa.add("LBO", "Result present", "FAIL", "No LBO result object supplied.")
            return qa.finalize()
        r = result
        self._check_sources_uses(qa, "LBO", getattr(r, "sources_uses_check", None))

        cf = getattr(r, "cash_flows", None)
        if cf is not None and not cf.empty:
            if "Ending TLB Balance" in cf.columns:
                tlb = pd.to_numeric(cf["Ending TLB Balance"], errors="coerce").dropna()
                if len(tlb):
                    if (tlb < -1e-9).any():
                        qa.add("LBO", "Debt balance never negative", "FAIL",
                               "TLB balance goes negative in the projection.")
                    else:
                        qa.add("LBO", "Debt balance never negative", "PASS")
            if "Total Debt" in cf.columns:
                td = pd.to_numeric(cf["Total Debt"], errors="coerce").dropna()
                if len(td) > 1 and td.iloc[-1] > td.iloc[0] * 1.001:
                    qa.add("LBO", "Debt paydown path", "WARN",
                           "Total debt rises over the projection — verify cash sweep and amortization.")
                elif len(td) > 1:
                    qa.add("LBO", "Debt paydown path", "PASS")
        if hasattr(r, "moic") and hasattr(r, "equity_amount"):
            moic = _safe_float(r.moic)
            eq = _safe_float(r.equity_amount)
            xe = _safe_float(getattr(r, "exit_equity_value", 0))
            if eq > 0 and abs(xe) > 0:
                qa.add("LBO", "MOIC consistency",
                       "PASS" if _approx(xe / eq, moic, max(abs(moic) * 1e-3, 1e-6)) else "WARN",
                       f"Exit equity / sponsor equity = {xe / eq:.3f}x vs reported {moic:.3f}x.")
        return qa.finalize()

    # ── Comps ────────────────────────────────────────────────────────────────
    def audit_comps(self, result: Any) -> QAReport:
        qa = QAReport()
        if result is None:
            qa.add("Comps", "Result present", "FAIL", "No comps result object supplied.")
            return qa.finalize()
        r = result
        peers = getattr(r, "peers_df", None)
        if peers is not None and not peers.empty:
            qa.add("Comps", "Peer universe populated", "PASS",
                   f"{len(peers)} comparable companies analyzed.")
            n_nan = peers.isna().sum().sum()
            if n_nan:
                qa.add("Comps", "Data completeness", "WARN",
                       f"{n_nan} missing data points across the peer table — statistics computed on available data.")
        implied = getattr(r, "implied_valuation", {}) or {}
        if implied:
            vals = [v for v in implied.values() if isinstance(v, (int, float)) and v > 0]
            if len(vals) >= 2:
                spread = (max(vals) - min(vals)) / np.mean(vals)
                if spread > 1.5:
                    qa.add("Comps", "Implied valuation dispersion", "WARN",
                           f"Implied value spread of {spread:.0%} across methods — verify comp-set relevance.")
        return qa.finalize()

    # ── IPO ──────────────────────────────────────────────────────────────────
    def audit_ipo(self, result: Any) -> QAReport:
        qa = QAReport()
        if result is None:
            qa.add("IPO", "Result present", "FAIL", "No IPO result object supplied.")
            return qa.finalize()
        r = result
        if hasattr(r, "share_count_bridge") and hasattr(r, "shares_outstanding"):
            bridge = getattr(r, "share_count_bridge", {}) or {}
            total = _safe_float(bridge.get("total_shares_post_ipo", 0))
            reported = _safe_float(r.shares_outstanding)
            if total > 0 and reported > 0:
                qa.add("IPO", "Share-count bridge reconciles",
                       "PASS" if _approx(total, reported, max(reported * 1e-3, 1e-6)) else "FAIL",
                       f"Bridge total {total:,.1f}M vs reported {reported:,.1f}M.")
        if hasattr(r, "gross_proceeds") and hasattr(r, "primary_proceeds") and hasattr(r, "secondary_proceeds"):
            gp = _safe_float(r.gross_proceeds)
            pp = _safe_float(r.primary_proceeds)
            sp = _safe_float(r.secondary_proceeds)
            if abs(gp) > 0:
                qa.add("IPO", "Proceeds reconcile",
                       "PASS" if _approx(gp, pp + sp, max(abs(gp) * 1e-3, 1e-6)) else "WARN",
                       f"Gross {gp:,.1f} vs primary {pp:,.1f} + secondary {sp:,.1f}.")
        return qa.finalize()

    # ── Valuation bridge ─────────────────────────────────────────────────────
    def audit_bridge(self, result: Any) -> QAReport:
        qa = QAReport()
        if result is None:
            qa.add("Bridge", "Result present", "FAIL", "No bridge result object supplied.")
            return qa.finalize()
        r = result
        methods = getattr(r, "methods", []) or []
        if not methods:
            qa.add("Bridge", "Valuation methods present", "FAIL", "No valuation methods registered.")
        else:
            qa.add("Bridge", "Valuation methods present", "PASS", f"{len(methods)} methodologies compared.")
        ranges = [m.get("range") for m in methods if m.get("range")]
        if ranges and len(ranges) >= 2:
            lows = [lo for lo, hi in ranges if lo and hi]
            highs = [hi for lo, hi in ranges if lo and hi]
            if lows and highs:
                span = (max(highs) - min(lows)) / max(np.mean([(lo + hi) / 2 for lo, hi in ranges]), 1e-9)
                if span > 1.0:
                    qa.add("Bridge", "Valuation convergence", "WARN",
                           f"Methodology range spans {span:.0%} of the mid — divergence is material; "
                           "understand the drivers before weighting.")
        return qa.finalize()


def run_model_qa(result: Any, module: str = "model") -> QAReport:
    """Dispatch to the correct auditor based on the result's shape."""
    auditor = ModelAuditor()
    mod = (module or "").lower()
    if mod == "dcf":
        return auditor.audit_dcf(result)
    if mod == "mna" or mod == "ma" or mod == "m&a":
        return auditor.audit_mna(result)
    if mod == "lbo":
        return auditor.audit_lbo(result)
    if mod == "comps":
        return auditor.audit_comps(result)
    if mod == "ipo":
        return auditor.audit_ipo(result)
    if mod == "bridge":
        return auditor.audit_bridge(result)
    return auditor.audit(result, module=mod or "model")


def get_model_auditor() -> ModelAuditor:
    return ModelAuditor()
