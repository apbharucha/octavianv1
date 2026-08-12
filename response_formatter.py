"""
Octavian Institutional Response Formatter
==========================================
Converts raw quantitative data into high-fidelity institutional reports.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class FormattedResponse:
    """Structured response with user-facing text and hidden analysis."""
    user_text: str
    raw_analysis: Dict[str, Any]
    show_reasoning_available: bool
    response_metadata: Dict[str, Any]


class ResponseFormatter:
    """Formats AI responses for premium institutional experience."""

    # ── Public entry points ────────────────────────────────────────────────

    def format_analysis_response(
        self,
        query: str,
        symbol_analyses: Dict[str, Any],
        intents: List[str],
        market_context: Dict[str, Any],
        timeframe: str,
    ) -> FormattedResponse:
        user_text = self._generate_institutional_synthesis(
            query, symbol_analyses, intents, market_context, timeframe
        )
        raw_analysis = {
            "symbol_analyses": symbol_analyses,
            "market_context": market_context,
            "regime_state": market_context.get("volatility_regime", "Unknown"),
            "bayesian_consensus": market_context.get("master_bias", "Neutral"),
            "detailed_metrics": self._create_quantitative_breakdown(symbol_analyses),
        }
        return FormattedResponse(
            user_text=user_text,
            raw_analysis=raw_analysis,
            show_reasoning_available=True,
            response_metadata={
                "symbols": list(symbol_analyses.keys()),
                "confidence": self._calculate_aggregate_confidence(symbol_analyses),
                "fidelity": "Institutional",
            },
        )

    def format_scan_response(
        self, opportunities: List[Any], timeframe: str
    ) -> FormattedResponse:
        """Format a market-wide scan result into institutional text."""
        if not opportunities:
            text = (
                "### Octavian Market Scan\n\n"
                "No high-conviction opportunities were identified across the scanned universe "
                "at this time. Market conditions may be range-bound or lacking directional clarity.\n\n"
                "**Suggestion:** Narrow the scan to a specific asset class or revisit after the next "
                "market session."
            )
        else:
            lines = [
                "### Octavian Unbiased Market Scan",
                f"*{len(opportunities)} opportunities ranked by risk-adjusted profit potential*\n",
                f"**Timeframe:** {timeframe.replace('_', ' ').title()}\n",
                "---",
            ]
            for i, opp in enumerate(opportunities[:10], 1):
                sym  = getattr(opp, "symbol", "?")
                prob = getattr(opp, "profit_probability", 0.5)
                ret  = getattr(opp, "expected_return", 0.0)
                conf = getattr(opp, "confidence_score", 0.5)
                sig  = getattr(opp, "signal_direction", "NEUTRAL")
                lines.append(
                    f"**{i}. {sym}** — {sig} | "
                    f"Profit Probability: {prob:.1%} | "
                    f"Expected Return: {ret:+.2%} | "
                    f"Confidence: {conf:.1%}"
                )
            text = "\n".join(lines)

        return FormattedResponse(
            user_text=text,
            raw_analysis={"opportunities": [vars(o) if hasattr(o, "__dict__") else {} for o in opportunities]},
            show_reasoning_available=False,
            response_metadata={"timeframe": timeframe, "count": len(opportunities)},
        )

    # ── Core synthesis ─────────────────────────────────────────────────────

    def _generate_institutional_synthesis(
        self,
        query: str,
        symbol_analyses: Dict[str, Any],
        intents: List[str],
        market_context: Dict[str, Any],
        timeframe: str,
    ) -> str:
        """
        Produces a substantive, directional, institutional-grade response.
        Each symbol gets: signal verdict, key technicals, price context,
        volatility, entry/exit thesis, and risk factors.
        """
        regime    = market_context.get("volatility_regime", "Normal")
        risk_mode = market_context.get("risk_mode", "Neutral")
        master_bias = market_context.get("master_bias", "NEUTRAL")
        conviction  = market_context.get("master_conviction", 0.5)
        tf_label    = timeframe.replace("_", " ").title()

        lines = [
            "## Octavian Intelligence Report",
            f"*Query: \"{query}\"*  |  *Timeframe: {tf_label}*\n",
            f"> **Market Regime:** {regime}  |  **Risk Sentiment:** {risk_mode}  "
            f"|  **Octavian Bias:** {master_bias} ({conviction*100:.0f}% conviction)\n",
            "---",
        ]

        for sym, ana in symbol_analyses.items():
            lines.extend(self._format_single_symbol(sym, ana, intents))
            lines.append("---")

        # Cross-asset conclusion if multiple symbols
        if len(symbol_analyses) > 1:
            lines.append(self._multi_asset_conclusion(symbol_analyses, master_bias))

        return "\n".join(lines)

    def _format_single_symbol(
        self, sym: str, ana: Dict[str, Any], intents: List[str]
    ) -> List[str]:
        pred    = ana.get("prediction", {})
        metrics = ana.get("metrics", {})
        tech    = ana.get("technical", {})
        signal  = pred.get("signal", "NEUTRAL")
        conf    = pred.get("confidence", 0.5)
        prob    = pred.get("bullish_prob", 0.5)
        exp_ret = pred.get("expected_return", 0.0)
        rar     = pred.get("risk_adjusted_return", 0.0)
        ivol    = pred.get("institutional_vol", 0.2)
        price   = metrics.get("current_price", 0.0)
        chg     = metrics.get("daily_change", 0.0)
        rsi_val = metrics.get("rsi", 0.0)
        vol_pct = metrics.get("annual_vol_pct", ivol * 100)
        asset_t = ana.get("asset_type", "Asset")

        # Signal emoji
        sig_icon = {"BULLISH": "", "BEARISH": "", "NEUTRAL": ""}.get(signal, "")

        lines = [
            f"### {sig_icon} {sym}  ({asset_t})",
        ]

        # Price line
        if price:
            chg_str = f"{chg:+.2f}%" if chg else ""
            lines.append(f"**Current Price:** ${price:,.4f}  {chg_str}")

        # Signal verdict
        lines.append(
            f"**Signal:** {signal}  |  "
            f"**Profit Probability:** {prob:.1%}  |  "
            f"**Model Confidence:** {conf:.1%}"
        )
        lines.append(
            f"**Expected Return:** {exp_ret:+.2%}  |  "
            f"**Risk-Adjusted Return:** {rar:.2f}  |  "
            f"**Annual Volatility:** {vol_pct:.1f}%"
        )
        lines.append("")

        # Key technical signals
        primary = tech.get("primary_signals", [])
        secondary = tech.get("secondary_signals", [])
        entry_sigs = ana.get("entry_signals", [])
        all_sigs = primary or entry_sigs
        if all_sigs:
            lines.append("**Key Technical Signals:**")
            for s in all_sigs[:4]:
                lines.append(f"- {s}")
            lines.append("")

        # RSI context
        if rsi_val:
            if rsi_val > 70:
                lines.append(f" **RSI {rsi_val:.1f}** — Overbought territory. Pullback risk elevated.")
            elif rsi_val < 30:
                lines.append(f" **RSI {rsi_val:.1f}** — Oversold. Potential mean-reversion / bounce setup.")
            else:
                lines.append(f"**RSI {rsi_val:.1f}** — Neutral momentum band.")
            lines.append("")

        # Directional thesis
        lines.append("**Directional Thesis:**")
        if signal == "BULLISH":
            lines.append(
                f"{sym} shows a bullish technical structure. Price action, momentum, and trend alignment "
                f"suggest upside potential of approximately {exp_ret*100:+.1f}% on a {ana.get('timeframe_scope','swing').replace('_',' ')} basis. "
                f"Watch for a hold above key moving averages as confirmation."
            )
        elif signal == "BEARISH":
            lines.append(
                f"{sym} is exhibiting bearish characteristics. Momentum is deteriorating and the trend "
                f"structure favors further downside. Expected move: {exp_ret*100:+.1f}%. "
                f"Short-side exposure is justified with disciplined stops above recent swing highs."
            )
        else:
            lines.append(
                f"{sym} is in a consolidation or range-bound phase. No clear directional edge at this time. "
                f"Wait for a confirmed breakout above resistance or breakdown below support before committing capital."
            )
        lines.append("")

        # Risk factors
        risk_factors = ana.get("risk_factors", [])
        if risk_factors:
            lines.append("**Risk Factors:**")
            for rf in risk_factors[:3]:
                lines.append(f"- {rf}")
            lines.append("")

        # Catalysts
        catalysts = ana.get("profit_catalysts", [])
        if catalysts:
            lines.append("**Potential Catalysts:**")
            for c in catalysts[:2]:
                lines.append(f"- {c}")
            lines.append("")

        return lines

    def _multi_asset_conclusion(
        self, symbol_analyses: Dict[str, Any], master_bias: str
    ) -> str:
        bullish = [s for s, a in symbol_analyses.items() if a.get("prediction", {}).get("signal") == "BULLISH"]
        bearish = [s for s, a in symbol_analyses.items() if a.get("prediction", {}).get("signal") == "BEARISH"]
        lines = ["### Cross-Asset Summary"]
        if bullish:
            lines.append(f" **Bullish:** {', '.join(bullish)}")
        if bearish:
            lines.append(f" **Bearish:** {', '.join(bearish)}")
        neutral = [s for s in symbol_analyses if s not in bullish and s not in bearish]
        if neutral:
            lines.append(f" **Neutral / Mixed:** {', '.join(neutral)}")
        lines.append(f"\n*Octavian dominant market bias: **{master_bias}***")
        return "\n".join(lines)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _create_quantitative_breakdown(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        breakdown = {}
        for sym, ana in analyses.items():
            breakdown[sym] = {
                "ml_ensemble":        ana.get("prediction", {}),
                "technical_signals":  ana.get("technical", {}),
                "volatility_forecast": ana.get("metrics", {}),
                "unbiased_metrics":   ana.get("unbiased_analysis", {}),
            }
        return breakdown

    def _calculate_aggregate_confidence(self, analyses: Dict[str, Any]) -> float:
        confs = [a.get("prediction", {}).get("confidence", 0.5) for a in analyses.values()]
        return sum(confs) / len(confs) if confs else 0.5


_formatter_singleton = None


def get_response_formatter() -> "ResponseFormatter":
    """Return the shared ResponseFormatter instance (module-level singleton)."""
    global _formatter_singleton
    if _formatter_singleton is None:
        _formatter_singleton = ResponseFormatter()
    return _formatter_singleton
