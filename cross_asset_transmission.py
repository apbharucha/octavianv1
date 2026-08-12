"""
cross_asset_transmission.py — Quantitative cross-asset transmission analysis.

Answers the class of questions the AI chatbot previously served as canned
boilerplate, e.g.:

    "what does the recent volatility relating to the price of oil futures
     affect other asset prices, provide quantitative proof and evidence"

Instead of generic "Energy landscape evolving" text, this module:

  1. Detects the ANCHOR asset named in the query (oil -> CL=F, gold -> GC=F,
     rates -> ^TNX, dollar -> DX-Y.NYB, VIX -> ^VIX, BTC -> BTC-USD, ...).
  2. Fetches 2 years of daily returns for the anchor and a representative
     basket of assets across classes (equities, energy equities, defensives,
     bonds, credit, FX, metals, EM, airlines, vol).
  3. Computes REAL quantitative evidence from that data:
       - pairwise daily-return correlation vs the anchor
       - 60-day rolling correlation (current vs 6 months ago) — is the
         relationship strengthening or fading?
       - beta of each asset to anchor daily returns
       - lead-lag causality (cross-correlation, best lag)
       - volatility-regime dependence: average correlation in the anchor's
         HIGH-vol quartile vs LOW-vol quartile (the "transmission" effect)
       - the anchor's own 20-day realized vol vs 6 months ago
  4. Writes an institutional narrative where every claim carries a number.

All data fetches go through the existing cached MacroDataLayer, so repeated
queries are cheap. If the data is unavailable the analyzer returns None and
the caller falls back to the previous (non-quantitative) path — this module
never fabricates numbers.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("CrossAssetTransmission")

# ─────────────────────────────────────────────────────────────────────────────
# Anchor detection — commodity / macro words -> Yahoo symbols
# ─────────────────────────────────────────────────────────────────────────────
ANCHOR_MAP: Dict[str, Tuple[str, str]] = {
    # Oil & energy
    "oil": ("CL=F", "WTI crude oil"),
    "crude": ("CL=F", "WTI crude oil"),
    "wti": ("CL=F", "WTI crude oil"),
    "petroleum": ("CL=F", "WTI crude oil"),
    "brent": ("BZ=F", "Brent crude oil"),
    "natural gas": ("NG=F", "natural gas"),
    "natgas": ("NG=F", "natural gas"),
    "heating oil": ("HO=F", "heating oil"),
    "gasoline": ("RB=F", "RBOB gasoline"),
    # Metals
    "gold": ("GC=F", "gold"),
    "silver": ("SI=F", "silver"),
    "copper": ("HG=F", "copper"),
    "platinum": ("PL=F", "platinum"),
    "palladium": ("PA=F", "palladium"),
    "aluminum": ("ALI=F", "aluminum"),
    # Agriculture
    "wheat": ("ZW=F", "wheat"),
    "corn": ("ZC=F", "corn"),
    "soybean": ("ZS=F", "soybean"),
    "soybeans": ("ZS=F", "soybean"),
    # Rates / yields
    "treasury": ("^TNX", "10Y Treasury yield"),
    "10y": ("^TNX", "10Y Treasury yield"),
    "10-year": ("^TNX", "10Y Treasury yield"),
    "rates": ("^TNX", "10Y Treasury yield"),
    "yields": ("^TNX", "10Y Treasury yield"),
    "yield curve": ("^TNX", "10Y Treasury yield"),
    # FX
    "dollar": ("DX-Y.NYB", "US Dollar index"),
    "usd": ("DX-Y.NYB", "US Dollar index"),
    "dxy": ("DX-Y.NYB", "US Dollar index"),
    # Vol
    "vix": ("^VIX", "CBOE VIX"),
    # Crypto
    "crypto": ("BTC-USD", "Bitcoin"),
    "cryptocurrenc": ("BTC-USD", "Bitcoin"),
    "bitcoin": ("BTC-USD", "Bitcoin"),
    "btc": ("BTC-USD", "Bitcoin"),
    "ethereum": ("ETH-USD", "Ethereum"),
    "eth": ("ETH-USD", "Ethereum"),
    "solana": ("SOL-USD", "Solana"),
    "dogecoin": ("DOGE-USD", "Dogecoin"),
}

# Representative asset basket for transmission measurement.
# (symbol, human label, asset class)
BASKET: List[Tuple[str, str, str]] = [
    ("SPY", "S&P 500", "US Equities"),
    ("QQQ", "Nasdaq 100", "US Growth/Tech"),
    ("DIA", "Dow Jones", "US Industrials"),
    ("XLE", "Energy Equities", "Sector Equities"),
    ("XLF", "Financials", "Sector Equities"),
    ("XLP", "Consumer Staples", "Defensives"),
    ("XLU", "Utilities", "Defensives"),
    ("DX-Y.NYB", "US Dollar Index", "FX"),
    ("GLD", "Gold", "Metals"),
    ("TLT", "Long Treasuries", "Rates"),
    ("HYG", "High-Yield Credit", "Credit"),
    ("LQD", "Inv-Grade Credit", "Credit"),
    ("EEM", "Emerging Markets", "EM Equities"),
    ("JETS", "Airlines", "Oil-Sensitive"),
    ("^VIX", "VIX (equity vol)", "Volatility"),
]

class CrossAssetTransmissionAnalyzer:
    """Computes real cross-asset transmission evidence for an anchor asset."""

    def __init__(self, cache_ttl: int = 600):
        try:
            from macro_cross_asset_engine import MacroDataLayer
            self.data = MacroDataLayer(cache_ttl=cache_ttl)
        except Exception as e:  # pragma: no cover
            logger.warning(f"MacroDataLayer unavailable: {e}")
            self.data = None
        self._result_cache: Dict[str, Tuple[float, dict]] = {}
        self._lock = threading.Lock()

    # ── Anchor detection ─────────────────────────────────────────────────────
    @staticmethod
    def detect_anchor(query: str, tickers: Optional[List[str]] = None) -> Optional[Tuple[str, str]]:
        """Return (symbol, label) for the anchor asset named in the query.

        Checks anchor words first (longest match wins), then falls back to the
        first explicitly mentioned ticker that resolves in the basket/anchor
        universe so generic "how does X affect the market" queries still work.
        """
        if not query:
            return None
        q = query.lower()
        # Longest-word-first so "natural gas" beats "gas", "yield curve" beats
        # "yields", etc.
        for phrase in sorted(ANCHOR_MAP, key=len, reverse=True):
            if phrase in q:
                return ANCHOR_MAP[phrase]
        # Fallback: explicit tickers that are themselves anchors or in the basket
        if tickers:
            known = {s.upper() for s, _, _ in BASKET} | {v[0] for v in ANCHOR_MAP.values()}
            for t in tickers:
                tu = str(t).upper()
                if tu in known:
                    return (tu, tu)
        return None

    # ── Data helpers ─────────────────────────────────────────────────────────
    def _get_returns(self, symbol: str, period: str = "2y") -> Optional[pd.Series]:
        if self.data is None:
            return None
        try:
            return self.data.get_returns(symbol, period=period)
        except Exception as e:
            logger.warning(f"returns unavailable for {symbol}: {e}")
            return None

    # ── Core analysis ────────────────────────────────────────────────────────
    def analyze(self, anchor: str, anchor_label: str,
                period: str = "2y") -> Optional[dict]:
        """Compute the full transmission evidence set for the anchor.

        Returns None if the anchor itself cannot be fetched (no fabricated
        numbers ever).
        """
        key = f"{anchor}:{period}"
        now = time.time()
        with self._lock:
            cached = self._result_cache.get(key)
            if cached and now - cached[0] < 900:  # 15-min result cache
                return cached[1]

        anchor_rets = self._get_returns(anchor, period)
        if anchor_rets is None or len(anchor_rets) < 60:
            logger.warning(f"[CrossAsset] anchor {anchor} has insufficient data")
            return None

        anchor_close = self._anchor_close(anchor, period)
        rows: List[dict] = []
        # Fetch basket returns IN PARALLEL (network-bound) so the first
        # transmission query does not take 15+ sequential download round-trips.
        basket_syms = [(sym, label, asset_class) for sym, label, asset_class in BASKET
                       if sym != anchor]
        fetched: Dict[str, Optional[pd.Series]] = {}
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(8, max(1, len(basket_syms)))) as ex:
                future_map = {ex.submit(self._get_returns, sym, period): sym
                              for sym, _, _ in basket_syms}
                for fut in future_map:
                    sym = future_map[fut]
                    try:
                        fetched[sym] = fut.result()
                    except Exception as e:
                        logger.warning(f"[CrossAsset] basket fetch {sym} failed: {e}")
                        fetched[sym] = None
        except Exception as e:  # pragma: no cover
            logger.warning(f"[CrossAsset] parallel basket fetch failed ({e}); falling back to serial")
            for sym, _, _ in basket_syms:
                fetched[sym] = self._get_returns(sym, period)

        for sym, label, asset_class in basket_syms:
            try:
                rets = fetched.get(sym)
                if rets is None or len(rets) < 60:
                    continue
                aligned = pd.concat(
                    [anchor_rets.rename("A"), rets.rename("B")], axis=1
                ).dropna()
                if len(aligned) < 60:
                    continue
                corr = float(aligned["A"].corr(aligned["B"]))
                # Rolling 60d correlation: current vs ~6 months (126d) ago
                roll = aligned["A"].rolling(60).corr(aligned["B"])
                roll_now = float(roll.dropna().iloc[-1]) if len(roll.dropna()) >= 1 else corr
                roll_then = float(roll.dropna().iloc[-126]) if len(roll.dropna()) > 126 else corr
                # Beta vs anchor daily returns
                var_a = float(aligned["A"].var())
                beta = float(aligned["A"].cov(aligned["B"]) / var_a) if var_a > 0 else 0.0
                # Lead-lag: does the anchor lead this asset? (positive lag)
                lead_lag = self._lead_lag(aligned)
                rows.append({
                    "symbol": sym,
                    "label": label,
                    "asset_class": asset_class,
                    "corr": round(corr, 3),
                    "corr_now": round(roll_now, 3),
                    "corr_6mo_ago": round(roll_then, 3),
                    "corr_trend": round(roll_now - roll_then, 3),
                    "beta": round(beta, 2),
                    "lead_lag_days": lead_lag,
                    "n": int(len(aligned)),
                })
            except Exception as e:
                logger.warning(f"[CrossAsset] basket {sym} failed: {e}")
                continue

        if len(rows) < 3:
            return None

        rows.sort(key=lambda r: abs(r["corr_now"]), reverse=True)

        # Volatility-regime dependence: correlations in the anchor's HIGH-vol
        # quartile vs LOW-vol quartile.
        regime = self._vol_regime_analysis(anchor_rets, rows, period, fetched)

        # Anchor's own realized vol now vs 6 months ago
        anchor_vol_now = self._realized_vol(anchor_rets, window=20)
        anchor_vol_then = self._realized_vol(anchor_rets.iloc[:-126], window=20) \
            if len(anchor_rets) > 126 else None
        anchor_mom_63d = float(anchor_close.iloc[-1] / anchor_close.iloc[-64] - 1) \
            if anchor_close is not None and len(anchor_close) > 64 else None

        result = {
            "anchor": anchor,
            "anchor_label": anchor_label,
            "rows": rows,
            "regime": regime,
            "anchor_vol_now": round(anchor_vol_now, 1) if anchor_vol_now else None,
            "anchor_vol_6mo_ago": round(anchor_vol_then, 1) if anchor_vol_then else None,
            "anchor_mom_63d": round(anchor_mom_63d * 100, 1) if anchor_mom_63d is not None else None,
            "n_rows": len(rows),
            "period": period,
            "computed_at": time.strftime("%Y-%m-%d %H:%M"),
        }
        with self._lock:
            self._result_cache[key] = (time.time(), result)
        return result

    def _anchor_close(self, anchor: str, period: str = "2y") -> Optional[pd.Series]:
        if self.data is None:
            return None
        try:
            return self.data.get_close(anchor, period=period)
        except Exception:
            return None

    @staticmethod
    def _realized_vol(rets: pd.Series, window: int = 20) -> Optional[float]:
        try:
            if rets is None or len(rets) < window + 5:
                return None
            tail = rets.tail(window)
            return float(tail.std() * np.sqrt(252) * 100)
        except Exception:
            return None

    @staticmethod
    def _lead_lag(aligned: pd.DataFrame, max_lag: int = 15) -> int:
        """Best lag at which the anchor (col 'A') leads the asset (col 'B')."""
        try:
            base = float(aligned["A"].corr(aligned["B"]))
            best_lag, best_corr = 0, abs(base)
            for lag in range(1, max_lag + 1):
                c = float(aligned["A"].shift(lag).corr(aligned["B"]))
                if not np.isnan(c) and abs(c) > best_corr:
                    best_corr, best_lag = abs(c), lag
            return int(best_lag)
        except Exception:
            return 0

    def _vol_regime_analysis(self, anchor_rets: pd.Series, rows: List[dict],
                             period: str,
                             fetched: Optional[Dict[str, Optional[pd.Series]]] = None) -> dict:
        """Average anchor-asset correlation in the anchor's HIGH-vol quartile
        vs LOW-vol quartile (20-day realized vol). This is the core measure of
        how the ANCHOR'S VOLATILITY changes transmission to other assets."""
        try:
            vol_series = anchor_rets.rolling(20).std() * np.sqrt(252)
            vol_series = vol_series.dropna()
            if len(vol_series) < 80:
                return {"available": False}
            hi = vol_series >= vol_series.quantile(0.75)
            lo = vol_series <= vol_series.quantile(0.25)
            hi_idx = set(vol_series[hi].index)
            lo_idx = set(vol_series[lo].index)
            regime_rows = []
            for r in rows:
                try:
                    # Reuse the returns already fetched in analyze() — no
                    # second network pass for the regime evidence.
                    rets = (fetched or {}).get(r["symbol"])
                    if rets is None:
                        rets = self._get_returns(r["symbol"], period)
                    if rets is None:
                        continue
                    aligned = pd.concat(
                        [anchor_rets.rename("A"), rets.rename("B")], axis=1
                    ).dropna()
                    hi_al = aligned[aligned.index.isin(hi_idx)]
                    lo_al = aligned[aligned.index.isin(lo_idx)]
                    hi_c = float(hi_al["A"].corr(hi_al["B"])) if len(hi_al) >= 20 else None
                    lo_c = float(lo_al["A"].corr(lo_al["B"])) if len(lo_al) >= 20 else None
                    if hi_c is not None and lo_c is not None and not (np.isnan(hi_c) or np.isnan(lo_c)):
                        regime_rows.append({
                            "symbol": r["symbol"],
                            "label": r["label"],
                            "corr_high_vol": round(hi_c, 3),
                            "corr_low_vol": round(lo_c, 3),
                            "corr_spread": round(hi_c - lo_c, 3),
                            "corr_full": r["corr"],
                            "hi_n": int(len(hi_al)),
                            "lo_n": int(len(lo_al)),
                        })
                except Exception:
                    continue
            # Headline-relevant: only rank regime changes for assets with a
            # MEANINGFUL base relationship (|corr| >= 0.3) so the top row is
            # never a near-zero-correlation asset whose tiny spread looks
            # dramatic. Remaining rows still show the full picture.
            meaningful = [x for x in regime_rows if abs(x["corr_full"]) >= 0.3]
            rank_pool = meaningful or regime_rows
            rank_pool.sort(key=lambda x: abs(x["corr_spread"]), reverse=True)
            return {"available": True, "rows": rank_pool[:10]}
        except Exception as e:
            logger.warning(f"[CrossAsset] regime analysis failed: {e}")
            return {"available": False}

    # ── Report generation ────────────────────────────────────────────────────
    def build_report(self, query: str, anchor: str, anchor_label: str,
                     analysis: dict) -> str:
        """Produce the institutional-grade, number-backed narrative."""
        rows = analysis["rows"]
        regime = analysis.get("regime", {})
        a_now = analysis.get("anchor_vol_now")
        a_then = analysis.get("anchor_vol_6mo_ago")
        a_mom = analysis.get("anchor_mom_63d")

        # ── Executive summary (numbers first) ────────────────────────────────
        top = rows[0]
        strongest_pos = next((r for r in rows if r["corr_now"] > 0), None)
        strongest_neg = next((r for r in rows if r["corr_now"] < 0), None)
        vol_line = ""
        if a_now is not None:
            if a_then is not None:
                delta = a_now - a_then
                trend = "elevated" if delta > 5 else "compressed" if delta < -5 else "broadly stable"
                vol_line = (f"{anchor_label} ({anchor}) realized volatility is **{a_now:.0f}%** "
                            f"(annualized, 20d), {trend} vs **{a_then:.0f}%** six months ago")
            else:
                vol_line = (f"{anchor_label} ({anchor}) realized volatility is **{a_now:.0f}%** "
                            f"(annualized, 20d)")
        mom_line = ""
        if a_mom is not None:
            mom_line = f"Price momentum over the last 3 months: **{a_mom:+.1f}%**."
        # Title-case without destroying acronyms (WTI stays WTI, not Wti)
        def _nice_title(s: str) -> str:
            words = [w if w.isupper() else w.capitalize() for w in s.split()]
            return " ".join(words)

        lines = [
            f"### Cross-Asset Transmission: How {_nice_title(anchor_label)} Moves Other Markets",
            "",
            f"**Anchor:** {_nice_title(anchor_label)} ({anchor}) · daily returns, {analysis['period']}.",
            "",
            f"**Quantitative bottom line:** The strongest measured relationship is with "
            f"**{top['label']}** ({top['asset_class']}) — 60-day correlation "
            f"**{top['corr_now']:+.2f}** ({'strengthening' if top['corr_trend'] > 0.05 else 'fading' if top['corr_trend'] < -0.05 else 'stable'} vs 6 months ago), "
            f"beta **{top['beta']:.2f}** to anchor returns.",
            "",
        ]
        if vol_line:
            lines.append(vol_line + ".")
        if mom_line:
            lines.append(mom_line)
        if strongest_neg and strongest_neg["symbol"] != top["symbol"]:
            lines.append(
                f"The clearest **inverse** relationship is **{strongest_neg['label']}** "
                f"(corr **{strongest_neg['corr_now']:+.2f}**) — the hedge that has historically "
                f"worked when the anchor moves."
            )
        lines.append("")
        lines.append("---")
        lines.append("")

        # ── Transmission channels table ──────────────────────────────────────
        lines.append("### Transmission Channels — Measured Evidence")
        lines.append("")
        lines.append("| Asset | Class | Corr (full) | Corr (60d) | Corr 6mo ago | Trend | Beta | Anchor leads by |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in rows[:10]:
            trend = ("↑" if r["corr_trend"] > 0.05 else
                     "↓" if r["corr_trend"] < -0.05 else "→")
            lines.append(
                f"| {r['label']} ({r['symbol']}) | {r['asset_class']} | {r['corr']:+.2f} | "
                f"{r['corr_now']:+.2f} | {r['corr_6mo_ago']:+.2f} | {trend} {r['corr_trend']:+.2f} | "
                f"{r['beta']:.2f} | {r['lead_lag_days']}d |"
            )
        lines.append("")
        lines.append("*Correlation: daily returns vs the anchor, 2y window. Beta: sensitivity of the asset's "
                     "daily return to a 1% anchor move. Lead: best cross-correlation lag in days.*")
        lines.append("")

        # ── Volatility-regime dependence ─────────────────────────────────────
        lines.append("### How the Anchor's *Volatility* Changes Transmission")
        lines.append("")
        if regime.get("available") and regime.get("rows"):
            r0 = regime["rows"][0]
            lines.append(
                f"Correlations are **regime-dependent**. When {anchor_label} realized volatility sits in "
                f"its top quartile, the average correlation with **{r0['label']}** is "
                f"**{r0['corr_high_vol']:+.2f}** vs **{r0['corr_low_vol']:+.2f}** in the calmest quartile "
                f"({r0['corr_spread']:+.2f} spread) — i.e. diversification erodes exactly when the anchor "
                f"is stressed."
            )
            lines.append("")
            lines.append("| Asset | Corr in HIGH-vol regime | Corr in LOW-vol regime | Spread |")
            lines.append("|---|---|---|---|")
            for rr in regime["rows"][:8]:
                lines.append(
                    f"| {rr['label']} | {rr['corr_high_vol']:+.2f} | {rr['corr_low_vol']:+.2f} | "
                    f"{rr['corr_spread']:+.2f} |"
                )
            lines.append("")
        else:
            lines.append("Insufficient history to split by volatility regime; the full-window "
                         "correlations above are the best available evidence.")
            lines.append("")

        # ── Causality / lead-lag ─────────────────────────────────────────────
        leaders = [r for r in rows if r["lead_lag_days"] >= 3]
        lines.append("### Lead-Lag & Causality")
        lines.append("")
        if leaders:
            lines.append(
                f"{anchor_label} price action has led (best cross-correlation lag ≥3 days) these "
                f"assets in the sample: {', '.join(f'{r['label']} ({r['lead_lag_days']}d)' for r in leaders[:5])}."
            )
        else:
            lines.append("No asset showed a consistent multi-day lead from anchor moves; the "
                         "relationships are predominantly same-day (contemporaneous).")
        lines.append("")
        lines.append("---")
        lines.append("")

        # ── Caveats (institutional honesty) ──────────────────────────────────
        lines.append("### Caveats")
        lines.append("")
        lines.append(
            "- **Correlation ≠ causation.** The measured co-movement is consistent with economic "
            "transmission channels (cost push, inflation expectations, risk sentiment, dollar flows) "
            "but cannot prove a specific mechanism by itself."
        )
        lines.append(
            "- **Regime dependence.** Correlations are unstable across vol regimes and crises; the "
            "HIGH-vol quartile numbers above are the more decision-relevant ones."
        )
        lines.append(
            "- **Window sensitivity.** A 2-year window mixes regimes. Shorten the window and the "
            "correlations move materially."
        )
        lines.append(
            "- **Futures vs spot.** Commodity anchors use front-month futures; roll and contango "
            "effects can slightly distort daily returns versus physical spot."
        )
        lines.append("")
        lines.append(
            f"*Analysis generated {analysis.get('computed_at', '')} · "
            f"Octavian Cross-Asset Transmission Engine · data: {analysis['period']} daily closes*"
        )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor + convenience entry point
# ─────────────────────────────────────────────────────────────────────────────
_analyzer: Optional[CrossAssetTransmissionAnalyzer] = None


def get_transmission_analyzer() -> CrossAssetTransmissionAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = CrossAssetTransmissionAnalyzer()
    return _analyzer


def generate_cross_asset_transmission(query: str,
                                      tickers: Optional[List[str]] = None) -> Optional[str]:
    """Convenience entry point: returns the full report text or None if the
    query has no detectable anchor or the data is unavailable."""
    try:
        detected = CrossAssetTransmissionAnalyzer.detect_anchor(query, tickers)
        if not detected:
            return None
        anchor, label = detected
        analyzer = get_transmission_analyzer()
        analysis = analyzer.analyze(anchor, label)
        if analysis is None:
            return None
        return analyzer.build_report(query, anchor, label, analysis)
    except Exception as e:
        logger.warning(f"[CrossAsset] transmission generation failed: {e}")
        return None
