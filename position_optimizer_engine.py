"""
Position Optimizer Engine  Octavian Terminal
Grades existing user positions and suggests mathematically superior optimizations.
"""

import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from unbiased_market_analyzer import UnbiasedMarketAnalyzer
from risk_engine import _get_asset_data, calculate_advanced_risk_metrics
import options_engine
from advanced_ml_engine import get_ensemble_engine
from institutional_analytics_engine import run_macro_analysis, run_micro_analysis
import re

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

@dataclass
class PositionGrade:
    symbol: str
    alpha_score: float # 0-100 (Model confidence)
    risk_score: float # 0-100 (Inversely proportional to VaR/Volatility)
    efficiency_score: float # 0-100 (Combined)
    expected_return_30d: float 
    capital_efficiency: float # ROI / Capital Required
    convexity_type: str # Linear, Convex, Concave
    status: str # "Strong Hold", "Optimize Needed", "Immediate Exit"
    liquidity_score: float = 0.0 # 0-100 (Volume/Spread based)
    risk_adjusted_grade: str = "B" # A, B, C, D, F
    recommendations: List[str] = field(default_factory=list)
    predictive_insights: Dict[str, Any] = field(default_factory=dict)
    # Institutional & Macro Extensions
    macro_sensitivity: Dict[str, float] = field(default_factory=dict)
    sentiment_score: float = 0.0  # -100 to 100
    institutional_conviction: float = 0.0  # 0-100
    volatility_forecast_30d: float = 0.0
    # Multi-leg additions
    strategy_name: str = "Single Leg"
    legs: List[Dict[str, Any]] = field(default_factory=list)
    aggregate_greeks: Dict[str, float] = field(default_factory=lambda: {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0})

class PositionOptimizerEngine:
    """The central intelligence for positioning optimization."""
    
    def __init__(self):
        self.analyzer = UnbiasedMarketAnalyzer()
        self.options_engine = options_engine.get_options_engine()
        self.ml_engine = get_ensemble_engine()

    def _clamp(self, x: float, lo: float = 0.0, hi: float = 100.0) -> float:
        return max(lo, min(hi, x))

    def _safe_corr(self, a: pd.Series, b: pd.Series) -> float:
        try:
            df = pd.concat([a, b], axis=1).dropna()
            if len(df) < 10:
                return 0.0
            return float(np.corrcoef(df.iloc[:, 0], df.iloc[:, 1])[0, 1])
        except Exception:
            return 0.0

    def _compute_technical_score(self, data: pd.DataFrame) -> float:
        try:
            close = data["Close"].astype(float)
            if len(close) < 30:
                return 50.0
            ret_21 = (close.iloc[-1] / close.iloc[-22] - 1) * 100 if len(close) > 21 else 0.0
            ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1] if len(close) >= 50 else ema20
            trend = 60.0 if ema20 >= ema50 else 40.0
            mom = 50.0 + np.tanh(ret_21 / 10.0) * 25.0
            return float(self._clamp((trend * 0.55 + mom * 0.45)))
        except Exception:
            return 50.0

    def _compute_fundamental_score(self, symbol: str) -> float:
        if not HAS_YF:
            return 55.0
        try:
            info = yf.Ticker(symbol).info or {}
            pe = info.get("trailingPE")
            margin = info.get("profitMargins")
            roe = info.get("returnOnEquity")
            rev_g = info.get("revenueGrowth")
            debt_to_eq = info.get("debtToEquity")

            score = 50.0
            if pe is not None and pe > 0:
                score += 8.0 if pe < 25 else (2.0 if pe < 40 else -6.0)
            if margin is not None:
                score += float(np.tanh(margin * 4.0) * 12.0)
            if roe is not None:
                score += float(np.tanh(roe * 3.0) * 10.0)
            if rev_g is not None:
                score += float(np.tanh(rev_g * 3.0) * 12.0)
            if debt_to_eq is not None:
                score += -8.0 if debt_to_eq > 250 else (-3.0 if debt_to_eq > 120 else 4.0)

            return float(self._clamp(score))
        except Exception:
            return 55.0

    def _compute_sentiment_score(self, symbol: str, analysis: Any) -> float:
        # Returns [-100, 100]
        score = 0.0
        try:
            # Heuristic from analyzer artifacts
            catalysts = getattr(analysis, "profit_catalysts", []) or []
            risks = getattr(analysis, "risk_factors", []) or []
            score += min(40.0, len(catalysts) * 6.0)
            score -= min(40.0, len(risks) * 6.0)
        except Exception:
            pass

        # Lightweight news title lexicon
        if HAS_YF:
            try:
                news_items = (yf.Ticker(symbol).news or [])[:20]
                pos_words = {"beat", "upgrade", "surge", "strong", "record", "bullish", "growth"}
                neg_words = {"miss", "downgrade", "lawsuit", "weak", "fall", "bearish", "cut"}
                nscore = 0
                for n in news_items:
                    title = str(n.get("title", "")).lower()
                    nscore += sum(1 for w in pos_words if w in title)
                    nscore -= sum(1 for w in neg_words if w in title)
                score += np.tanh(nscore / 8.0) * 35.0
            except Exception:
                pass

        return float(max(-100.0, min(100.0, score)))

    def _determine_driving_factor(
        self,
        asset_type: str,
        vol: float,
        sentiment_score: float,
        macro_bias: str
    ) -> str:
        at = (asset_type or "").upper()
        if at == "OPTION":
            return "TECHNICAL" if vol > 0.35 else "FUNDAMENTAL"
        if at in {"FOREX", "FUTURES", "CRYPTO"}:
            return "TECHNICAL" if abs(sentiment_score) < 20 else "SENTIMENT"
        # STOCK default: fundamental overweight
        if macro_bias in {"RISK_OFF", "BEARISH"} and vol > 0.4:
            return "TECHNICAL"
        return "FUNDAMENTAL"

    def _score_weights_by_context(self, driving_factor: str) -> Dict[str, float]:
        # Fundamental intentionally higher by default
        if driving_factor == "TECHNICAL":
            return {"fundamental": 0.30, "technical": 0.45, "model": 0.20, "sentiment": 0.05}
        if driving_factor == "SENTIMENT":
            return {"fundamental": 0.25, "technical": 0.25, "model": 0.20, "sentiment": 0.30}
        return {"fundamental": 0.45, "technical": 0.30, "model": 0.20, "sentiment": 0.05}

    def _extract_option_liquidity_score(self, symbol: str, metadata: Dict[str, Any]) -> float:
        if not HAS_YF:
            return 55.0
        try:
            expiry = metadata.get("expiry")
            strike = float(metadata.get("strike")) if metadata.get("strike") is not None else None
            otype = str(metadata.get("option_type", "call")).lower()

            tk = yf.Ticker(symbol)
            if not expiry:
                exps = tk.options or []
                if not exps:
                    return 55.0
                expiry = exps[0]

            chain = tk.option_chain(expiry)
            df = chain.calls if otype == "call" else chain.puts
            if df is None or df.empty:
                return 55.0

            # fix for "['vol'] not in index": normalize expected columns
            if "vol" not in df.columns:
                if "impliedVolatility" in df.columns:
                    df["vol"] = pd.to_numeric(df["impliedVolatility"], errors="coerce").fillna(0.0)
                else:
                    df["vol"] = 0.0

            if strike is not None and "strike" in df.columns:
                df = df.assign(_d=(df["strike"] - strike).abs()).sort_values("_d").head(3)

            bid = pd.to_numeric(df.get("bid", 0), errors="coerce").fillna(0.0).mean()
            ask = pd.to_numeric(df.get("ask", 0), errors="coerce").fillna(0.0).mean()
            volume = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0).mean()
            oi = pd.to_numeric(df.get("openInterest", 0), errors="coerce").fillna(0.0).mean()
            spread = (ask - bid) if ask >= bid else ask

            spread_penalty = min(40.0, max(0.0, spread * 100.0))
            depth_bonus = min(35.0, np.log10(max(1.0, volume + oi)) * 8.0)
            vol_bonus = min(15.0, np.log10(max(1.0, float(df["vol"].mean()) * 1000 + 1.0)) * 5.0)

            score = 55.0 - spread_penalty + depth_bonus + vol_bonus
            return float(self._clamp(score))
        except Exception:
            return 55.0

    def _dynamic_grade_from_distribution(self, score: float, ref_series: np.ndarray) -> str:
        q20, q40, q60, q80 = np.percentile(ref_series, [20, 40, 60, 80])
        if score >= q80: return "A"
        if score >= q60: return "B"
        if score >= q40: return "C"
        if score >= q20: return "D"
        return "F"

    def _dynamic_status_from_models(self, alpha: float, risk: float, exp_ret_30d: float, ml_decision: str, confidence: float) -> str:
        # fully model-driven composition
        if ml_decision == "BULLISH" and alpha > 60 and exp_ret_30d > 0:
            return "Model-Backed Long Bias"
        if ml_decision == "BEARISH" and alpha > 55 and exp_ret_30d < 0:
            return "Model-Backed Defensive Bias"
        if confidence < 0.45 or abs(exp_ret_30d) < 1.0:
            return "Low-Edge / Rebalance Candidate"
        return "Conditional Hold"

    def _dynamic_recommendations(self, factor_scores: Dict[str, float], driving_factor: str, ml_decision: str) -> List[str]:
        ranked = sorted(factor_scores.items(), key=lambda kv: kv[1], reverse=True)
        weak = sorted(factor_scores.items(), key=lambda kv: kv[1])[:2]
        recs = [
            f"Primary decision driver: {driving_factor.lower()}",
            f"Model direction: {ml_decision}",
            f"Top contributors: {ranked[0][0]}={ranked[0][1]:.1f}, {ranked[1][0]}={ranked[1][1]:.1f}",
            f"Weak contributors to improve: {weak[0][0]}={weak[0][1]:.1f}, {weak[1][0]}={weak[1][1]:.1f}",
        ]
        return recs

    async def grade_position(self, symbol: str, entry_price: float, quantity: float, 
                      asset_type: str = "STOCK", metadata: Dict[str, Any] = None) -> PositionGrade:
        """
        Analyze and grade a single position with institutional-grade predictive metrics.
        Metadata can include: strike, expiry, option_type, leverage, exchange, etc.
        """
        if metadata is None: metadata = {}
        try:
            from data_sources import get_realtime_price
            from risk_engine import _get_asset_data
            
            # 1. Fetch Comprehensive Data
            data = _get_asset_data(symbol)
            if data.empty:
                return self._fallback_grade(symbol)
            
            rt_price_data = get_realtime_price(symbol)
            current_price = rt_price_data[0] if rt_price_data else data['Close'].iloc[-1]
            pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price != 0 else 0
            
            # 2. Predictive Intelligence (Async)
            analysis = await self.analyzer.analyze_unbiased(symbol)

            # Build options context for model fusion when relevant
            options_context = None
            if (asset_type or "").upper() == "OPTION":
                strike_ctx = metadata.get("strike")
                expiry_ctx = metadata.get("expiry")
                option_type_ctx = metadata.get("option_type", "call")
                if strike_ctx and expiry_ctx:
                    try:
                        expiry_dt_ctx = datetime.strptime(expiry_ctx, "%Y-%m-%d")
                        dte_days_ctx = max(1, (expiry_dt_ctx - datetime.now()).days)
                        # implied vol from metadata when available, else realized
                        # vol from the asset's own returns, else a neutral floor
                        realized_vol = (float(data['Close'].pct_change().std() * np.sqrt(252))
                                        if len(data) > 20 else 0.25)
                        iv_ctx = float(max(float(metadata.get("iv") or realized_vol), 0.05))
                        options_context = {
                            "spot": float(current_price),
                            "strike": float(strike_ctx),
                            "dte_days": int(dte_days_ctx),
                            "iv": iv_ctx,
                            "option_type": str(option_type_ctx).lower(),
                            "market_view": 0.0,
                            "confidence": 0.5,
                        }
                    except Exception:
                        options_context = None

            ml_results = self.ml_engine.analyze_symbol_ensemble(
                data,
                symbol,
                asset_type=asset_type,
                options_context=options_context
            )
            model_score = (ml_results['confidence'] * 100) if ml_results['decision'] != "NEUTRAL" else 50.0

            # NEW: fundamental + technical + sentiment
            fundamental_score = self._compute_fundamental_score(symbol)
            technical_score = self._compute_technical_score(data)
            sentiment_score = self._compute_sentiment_score(symbol, analysis)
            sentiment_01 = (sentiment_score + 100.0) / 200.0
            sentiment_component = sentiment_01 * 100.0

            # 3. Risk Engine
            returns = data['Close'].pct_change().dropna()
            risk_metrics = calculate_advanced_risk_metrics(returns)
            vol = risk_metrics['volatility']
            risk_score = max(5, 100 - (vol * 150))
            
            # 4. Institutional & Macro Context
            from master_strategy_engine import get_master_engine
            master_outlook = get_master_engine().get_dominant_outlook()

            macro_sens = {
                'rates_sensitivity': 0.0,
                'equity_beta': 1.0,
            }
            try:
                rates = _get_asset_data("^TNX")['Close'].pct_change()
                spy = _get_asset_data("SPY")['Close'].pct_change()
                macro_sens['rates_sensitivity'] = round(self._safe_corr(returns.tail(90), rates.tail(90)), 2)
                macro_sens['equity_beta'] = round(self._safe_corr(returns.tail(90), spy.tail(90)), 2)
            except Exception:
                pass

            v_ma_20 = data['Volume'].tail(20).mean() if 'Volume' in data.columns else 0
            v_current = data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
            vol_surge_score = min(100, max(10, (v_current / max(v_ma_20, 1)) * 50)) if v_ma_20 else 50.0

            alignment_score = 50.0
            beta = macro_sens.get('equity_beta', 1.0)
            if master_outlook.bias == "BULLISH":
                alignment_score = (50.0 + (master_outlook.conviction * 50.0)) if beta > 0.4 else (50.0 - (master_outlook.conviction * 20.0))
            elif master_outlook.bias == "BEARISH":
                alignment_score = (50.0 + (master_outlook.conviction * 50.0)) if beta < -0.1 else (50.0 - (master_outlook.conviction * 20.0))
            inst_conviction = (vol_surge_score * 0.4 + alignment_score * 0.6)

            # Context-driven driving factor
            driving_factor = self._determine_driving_factor(
                asset_type, vol, sentiment_score, master_outlook.bias
            )
            w = self._score_weights_by_context(driving_factor)

            base_prob = float(getattr(analysis, "profit_probability", 0.5))
            heuristic_model_score = base_prob * 100.0
            blended_model_score = heuristic_model_score * 0.6 + model_score * 0.4

            alpha_score = (
                w["fundamental"] * fundamental_score
                + w["technical"] * technical_score
                + w["model"] * blended_model_score
                + w["sentiment"] * sentiment_component
            )
            alpha_score = self._clamp(alpha_score)

            exp_ret_30d = (
                float(getattr(analysis, "expected_return", 0.0)) * 0.6
                + float(ml_results.get("predicted_return", 0.0)) * 0.4
            ) * 100

            leverage_factor = 1.0
            convexity = "Linear"
            liquidity_score = 70.0

            if asset_type == "OPTION":
                leverage_factor = 10.0
                convexity = "Convex"
                liquidity_score = self._extract_option_liquidity_score(symbol, metadata)
                strike = metadata.get('strike')
                option_type = metadata.get('option_type', 'call')
                expiry = metadata.get('expiry')
                if strike and expiry:
                    try:
                        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
                        t_days = (expiry_dt - datetime.now()).days
                        T = max(1, t_days) / 365
                        greeks = self.options_engine.black_scholes(current_price, strike, T, vol, option_type)
                        alpha_score = self._clamp(alpha_score * 0.8 + (0.5 + greeks['delta'] * (1 if option_type == 'call' else -1)) * 20.0)
                    except Exception:
                        pass
            elif asset_type == "FOREX":
                leverage_factor = metadata.get('leverage', 20.0)
                liquidity_score = max(60.0, liquidity_score)
            elif asset_type == "FUTURE":
                leverage_factor = metadata.get('multiplier', 50.0)
                convexity = "Linear Leverage"
                liquidity_score = max(70.0, liquidity_score)
            elif asset_type == "CRYPTO":
                leverage_factor = 1.0
                liquidity_score = max(30, liquidity_score - 20)

            if 'Volume' in data.columns and asset_type != "OPTION":
                avg_vol = data['Volume'].tail(20).mean()
                liquidity_score = min(100, max(10, np.log10(avg_vol + 1) * 10))

            cap_efficiency = (abs(exp_ret_30d) * leverage_factor)
            efficiency = (alpha_score * 0.4 + risk_score * 0.3 + (100 - min(100, abs(pnl_pct))) * 0.2 + liquidity_score * 0.1)

            ra_score = (alpha_score * 0.6 + risk_score * 0.4)
            ra_grade = "A" if ra_score > 85 else "B" if ra_score > 70 else "C" if ra_score > 55 else "D" if ra_score > 40 else "F"

            factor_scores = {
                'fundamental': round(fundamental_score, 1),
                'technical': round(technical_score, 1),
                'sentiment': round(sentiment_component, 1),
                'model': round(blended_model_score, 1),
            }

            # dynamic status/grade/recommendations (no hardcoded strategy templates)
            status = self._dynamic_status_from_models(
                alpha_score, risk_score, exp_ret_30d, ml_results['decision'], float(ml_results.get('confidence', 0.5))
            )
            ref = np.array([alpha_score, risk_score, efficiency, liquidity_score, 50.0, 65.0, 35.0, 80.0, 20.0], dtype=float)
            ra_grade = self._dynamic_grade_from_distribution((alpha_score * 0.6 + risk_score * 0.4), ref)
            recommendations = self._dynamic_recommendations(factor_scores, driving_factor, ml_results['decision'])

            predictive_insights = {
                'expected_30d_move': f"{exp_ret_30d:+.1f}%",
                'model_edge': f"{(alpha_score - 50):+.1f}%",
                'volatility_regime': "High" if vol > 0.4 else "Moderate" if vol > 0.2 else "Low",
                'reward_risk_ratio': round(abs(exp_ret_30d/100) / max(vol, 0.05), 2),
                'liquidity': "Deep" if liquidity_score > 80 else "Normal" if liquidity_score > 50 else "Thin",
                'ml_decision': ml_results['decision'],
                'driving_factor': driving_factor,
                'factor_scores': factor_scores
            }

            return PositionGrade(
                symbol=symbol,
                alpha_score=round(alpha_score, 1),
                risk_score=round(risk_score, 1),
                efficiency_score=round(efficiency, 1),
                expected_return_30d=round(exp_ret_30d, 1),
                capital_efficiency=round(cap_efficiency, 1),
                convexity_type=convexity,
                status=status,
                liquidity_score=round(liquidity_score, 1),
                risk_adjusted_grade=ra_grade,
                recommendations=recommendations,
                predictive_insights=predictive_insights,
                macro_sensitivity=macro_sens,
                sentiment_score=round(sentiment_score, 1),
                institutional_conviction=round(inst_conviction, 1),
                volatility_forecast_30d=round(vol * 1.1, 3)
            )
        except Exception as e:
            print(f"Optimizer grading complex error for {symbol}: {e}")
            return self._fallback_grade(symbol)

    async def grade_strategy(self, symbol: str, legs: List[Dict[str, Any]], 
                           risk_tolerance: str = "Balanced") -> PositionGrade:
        """
        Grade a multi-leg strategy (Spread, Straddle, etc.) by aggregating legs.
        'legs' list of {type, strike, expiry, side, quantity, entry_price}
        """
        if not legs: return self._fallback_grade(symbol)
        
        try:
            from data_sources import get_realtime_price
            from risk_engine import _get_asset_data
            
            data = _get_asset_data(symbol)
            current_price = get_realtime_price(symbol)
            if current_price == 0: current_price = float(data['Close'].iloc[-1])
            
            agg_greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
            total_cost = 0.0
            total_alpha = 0.0
            
            # Analyze each leg
            for leg in legs:
                expiry_dt = datetime.strptime(leg['expiry'], "%Y-%m-%d")
                T = max(1, (expiry_dt - datetime.now()).days) / 365
                vol = 0.25 # Default fallback vol
                try: 
                    # Try to get real vol from data
                    vol = data['Close'].pct_change().std() * np.sqrt(252)
                except: pass
                
                g = self.options_engine.black_scholes(current_price, leg['strike'], T, vol, leg['option_type'])
                qty = leg['quantity']
                side = 1 if leg['side'].lower() in ['buy', 'long'] else -1
                
                agg_greeks['delta'] += g['delta'] * qty * side
                agg_greeks['gamma'] += g['gamma'] * qty * side
                agg_greeks['theta'] += g['theta'] * qty * side
                agg_greeks['vega'] += g['vega'] * qty * side
                
                total_cost += leg['entry_price'] * qty * side
                
            # Compute consolidated scores
            # Alpha is harder for spreads, use delta-neutrality or directional bias
            bias = 1.0 if agg_greeks['delta'] > 0 else -1.0
            # Simple alpha heuristic for demo
            alpha_score = 50 + (agg_greeks['delta'] * current_price / (abs(total_cost) + 1)) * 10
            alpha_score = min(95, max(5, alpha_score))
            
            risk_score = 100 - min(90, abs(agg_greeks['vega']) * 100)
            
            # Identify strategy type
            strat_name = "Custom Multi-Leg"
            if len(legs) == 2:
                if legs[0]['strike'] == legs[1]['strike']: strat_name = "Straddle/Strangle"
                else: strat_name = "Spread"
            
            return PositionGrade(
                symbol=symbol,
                alpha_score=round(alpha_score, 1),
                risk_score=round(risk_score, 1),
                efficiency_score=round((alpha_score + risk_score) / 2, 1),
                expected_return_30d=round(alpha_score - 50, 1),
                capital_efficiency=round(abs(agg_greeks['delta'] * current_price) / (abs(total_cost) + 1), 2),
                convexity_type="Non-Linear",
                status="Strategic Hold",
                liquidity_score=75.0,
                risk_adjusted_grade="A" if alpha_score > 70 and risk_score > 60 else "B",
                recommendations=[f"Strategy Greeks: Net Delta {agg_greeks['delta']:.2f}, Net Theta {agg_greeks['theta']:.2f}"],
                predictive_insights={'strategy': strat_name, 'net_delta': round(agg_greeks['delta'], 2)},
                strategy_name=strat_name,
                legs=legs,
                aggregate_greeks=agg_greeks
            )
        except Exception as e:
            print(f"Strategy grading error: {e}")
            return self._fallback_grade(symbol)

    def optimize_position(self, symbol: str, grade: PositionGrade) -> List[Dict[str, Any]]:
        """Fully model-driven optimization suggestions (no static strategy templates)."""
        suggestions = []
        fs = grade.predictive_insights.get("factor_scores", {})
        ml_decision = grade.predictive_insights.get("ml_decision", "NEUTRAL")
        driver = grade.predictive_insights.get("driving_factor", "FUNDAMENTAL")
        rr = grade.predictive_insights.get("reward_risk_ratio", 0)

        suggestions.append({
            "type": "MODEL_ALIGNMENT",
            "strategy": f"{ml_decision}_{driver}",
            "reason": f"Decision generated from model outputs; reward/risk={rr}",
            "details": f"Use highest-ranked factors: {sorted(fs.items(), key=lambda x: x[1], reverse=True)[:2]}"
        })

        if grade.alpha_score < 50:
            suggestions.append({
                "type": "EDGE_RECOVERY",
                "strategy": "Reduce directional exposure",
                "reason": "Model-estimated edge is below neutral threshold",
                "details": "Re-enter only when model edge and confidence recover."
            })

        if grade.liquidity_score < 45:
            suggestions.append({
                "type": "LIQUIDITY_CONTROL",
                "strategy": "Execution constraint mode",
                "reason": "Low liquidity from live chain/volume metrics",
                "details": "Use smaller clips and stricter limit-order execution."
            })

        return suggestions

    async def predict_and_optimize_position(
        self, symbol: str, entry_price: float, quantity: float, 
        position_type: str = "LONG", asset_type: str = "STOCK", 
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        HIGH-LEVEL ORCHESTRATION:
        1. Predict current position trajectory (next 5-30 days)
        2. Fetch live market data to assess real edge
        3. Generate position transformation recommendations
        4. Calculate delta-neutral hedges if needed
        5. Suggest best alternative (stock vs options conversion)
        """
        if metadata is None:
            metadata = {}

        from data_sources import get_realtime_price, get_stock
        from risk_engine import _get_asset_data
        
        # ── STEP 1: Get Current State ──
        try:
            current_price_tuple = get_realtime_price(symbol)
            current_price = current_price_tuple[0] if current_price_tuple else None
            
            if not current_price or current_price <= 0:
                # Fallback to historical close
                df = _get_asset_data(symbol)
                if df.empty:
                    return {
                        "status": "error",
                        "message": f"Could not fetch live data for {symbol}",
                        "symbol": symbol
                    }
                current_price = float(df['Close'].iloc[-1])
            
            # Calculate current position P&L
            position_value = current_price * quantity
            entry_value = entry_price * quantity
            unrealized_pnl = position_value - entry_value
            unrealized_pnl_pct = (unrealized_pnl / entry_value * 100) if entry_value > 0 else 0
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Price fetch failed: {str(e)}",
                "symbol": symbol
            }

        # ── STEP 2: Grade Existing Position ──
        grade = await self.grade_position(symbol, entry_price, quantity, asset_type, metadata)

        # ── STEP 3: Predict 5/14/30-Day Performance ──
        predictions = self._generate_position_predictions(
            symbol, current_price, entry_price, grade, asset_type
        )

        # ── STEP 4: Transformation Recommendations ──
        transformations = self._generate_position_transformations(
            symbol, current_price, quantity, entry_price, grade, 
            asset_type, position_type, unrealized_pnl_pct
        )

        # ── STEP 5: Live Hedge / Risk Mitigation ──
        hedges = self._suggest_hedges(
            symbol, current_price, quantity, grade, position_type, unrealized_pnl_pct
        )

        # ── STEP 6: Comprehensive Summary ──
        return {
            "status": "success",
            "symbol": symbol,
            "current_state": {
                "entry_price": round(entry_price, 2),
                "current_price": round(current_price, 2),
                "quantity": quantity,
                "position_type": position_type,
                "asset_type": asset_type,
                "position_value": round(position_value, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
            },
            "position_grade": {
                "alpha_score": grade.alpha_score,
                "risk_score": grade.risk_score,
                "efficiency_score": grade.efficiency_score,
                "status": grade.status,
                "risk_adjusted_grade": grade.risk_adjusted_grade,
                "expected_return_30d": grade.expected_return_30d,
            },
            "predictions": predictions,
            "transformations": transformations,
            "hedges": hedges,
            "model_drivers": {
                "primary_driver": grade.predictive_insights.get("driving_factor", "FUNDAMENTAL"),
                "ml_decision": grade.predictive_insights.get("ml_decision", "NEUTRAL"),
                "volatility_regime": grade.predictive_insights.get("volatility_regime", "Moderate"),
                "factor_scores": grade.predictive_insights.get("factor_scores", {}),
            }
        }

    def _generate_position_predictions(
        self, symbol: str, current_price: float, entry_price: float, 
        grade: PositionGrade, asset_type: str
    ) -> Dict[str, Any]:
        """Predict position trajectory over multiple timeframes."""
        
        exp_ret_pct = grade.expected_return_30d
        vol = grade.volatility_forecast_30d if grade.volatility_forecast_30d > 0 else 0.25
        
        # Scenario-based predictions
        predictions = {
            "base_case_5d": {
                "expected_move_pct": round(exp_ret_pct * 0.15, 2),
                "price_target": round(current_price * (1 + exp_ret_pct * 0.0015), 2),
                "confidence": round(grade.institutional_conviction, 1),
                "reasoning": f"Model expects {exp_ret_pct:+.1f}% move over 30d, pro-rata 5d move is ~{exp_ret_pct * 0.15:+.2f}%"
            },
            "base_case_14d": {
                "expected_move_pct": round(exp_ret_pct * 0.45, 2),
                "price_target": round(current_price * (1 + exp_ret_pct * 0.0045), 2),
                "confidence": round(grade.institutional_conviction * 0.95, 1),
                "reasoning": f"Two-week projection with model conviction decay (~95% of 30d)"
            },
            "base_case_30d": {
                "expected_move_pct": round(exp_ret_pct, 2),
                "price_target": round(current_price * (1 + exp_ret_pct / 100), 2),
                "confidence": round(grade.institutional_conviction * 0.85, 1),
                "reasoning": f"Model base case: {exp_ret_pct:+.1f}% return expectation over 30 days"
            },
            "bull_case_30d": {
                "expected_move_pct": round(exp_ret_pct * 1.5, 2),
                "price_target": round(current_price * (1 + exp_ret_pct * 1.5 / 100), 2),
                "confidence": round(grade.institutional_conviction * 0.6, 1),
                "probability": "25-30%",
                "catalyst": "Positive surprise in fundamentals or macro tailwinds"
            },
            "bear_case_30d": {
                "expected_move_pct": round(exp_ret_pct * 0.5 - vol * 15, 2),
                "price_target": round(current_price * (1 + (exp_ret_pct * 0.5 - vol * 15) / 100), 2),
                "confidence": round(grade.institutional_conviction * 0.4, 1),
                "probability": "15-20%",
                "catalyst": "Negative reversal or macro shock"
            },
        }
        
        return predictions

    def _generate_position_transformations(
        self, symbol: str, current_price: float, quantity: float, entry_price: float,
        grade: PositionGrade, current_asset_type: str, position_type: str, 
        unrealized_pnl_pct: float
    ) -> List[Dict[str, Any]]:
        """Generate position transformation recommendations (stock → options, spreads, etc)."""
        
        transformations = []
        
        # ── TRANSFORMATION 1: Stock → Long Call (if bullish) ──
        if current_asset_type == "STOCK" and position_type == "LONG" and grade.expected_return_30d > 3:
            call_recs = self._recommend_call_spread(
                symbol, current_price, quantity, entry_price, grade
            )
            transformations.append({
                "type": "STOCK_TO_LONG_CALL",
                "from_position": f"Long {quantity} {symbol} @ ${entry_price:.2f}",
                "to_position": call_recs["strategy"],
                "reason": f"Convert linear stock position into leveraged call spread (20-30x capital efficiency gain)",
                "details": call_recs["details"],
                "capital_freed": round((quantity * current_price) * 0.7, 2),
                "max_profit": call_recs["max_profit"],
                "max_loss": call_recs["max_loss"],
                "breakeven": call_recs["breakeven"],
                "implementation": {
                    "step_1": f"Sell {quantity} shares @ ${current_price:.2f} (or place limit order at resistance)",
                    "step_2": f"Buy {call_recs['contracts']} call contracts at {call_recs['strike_atm']} strike",
                    "step_3": f"Sell {call_recs['contracts']} call contracts at {call_recs['strike_otm']} strike",
                    "time_to_execute": "1-2 trading sessions"
                },
                "confidence": round(grade.alpha_score, 1),
                "net_exposure": call_recs.get("net_delta", 0)
            })

        # ── TRANSFORMATION 2: Stock → Put Spread (if bearish) ──
        if current_asset_type == "STOCK" and position_type == "LONG" and grade.expected_return_30d < -3:
            put_recs = self._recommend_put_spread(
                symbol, current_price, quantity, entry_price, grade
            )
            transformations.append({
                "type": "STOCK_TO_PROTECTION_SPREAD",
                "from_position": f"Long {quantity} {symbol} @ ${entry_price:.2f}",
                "to_position": put_recs["strategy"],
                "reason": f"Convert to zero-cost or low-cost collar to protect against downside",
                "details": put_recs["details"],
                "cost": round(put_recs["cost"], 2),
                "protection_level": round(put_recs["protection_strike"], 2),
                "max_protected_loss": round(put_recs["max_loss"], 2),
                "upside_capped_at": round(put_recs["max_profit"], 2),
                "implementation": {
                    "step_1": f"Buy {put_recs['contracts']} put @ {put_recs['protection_strike']} strike (protection)",
                    "step_2": f"Sell {put_recs['contracts']} call @ {put_recs['cap_strike']} strike (finance premium)",
                    "step_3": f"Net cost: ${put_recs['cost']:.2f} (or zero if premium matches)",
                    "time_to_execute": "Same day"
                },
                "confidence": round(grade.risk_score, 1),
            })

        # ── TRANSFORMATION 3: Stock → Iron Condor (if neutral/ranging) ──
        if (current_asset_type == "STOCK" and abs(grade.expected_return_30d) < 2 
            and grade.volatility_forecast_30d < 0.30):
            condor_recs = self._recommend_iron_condor(
                symbol, current_price, quantity, entry_price, grade
            )
            transformations.append({
                "type": "STOCK_TO_IRON_CONDOR",
                "from_position": f"Long {quantity} {symbol} @ ${entry_price:.2f}",
                "to_position": condor_recs["strategy"],
                "reason": f"Stock is range-bound. Liquidate and harvest premium via iron condor (income play)",
                "details": condor_recs["details"],
                "capital_required": round(condor_recs["capital_required"], 2),
                "max_profit": round(condor_recs["max_profit"], 2),
                "max_loss": round(condor_recs["max_loss"], 2),
                "profit_range": f"${condor_recs['lower_strike']:.2f} - ${condor_recs['upper_strike']:.2f}",
                "implementation": {
                    "step_1": f"Sell {condor_recs['contracts']} put spread @ {condor_recs['put_strikes']}",
                    "step_2": f"Sell {condor_recs['contracts']} call spread @ {condor_recs['call_strikes']}",
                    "step_3": f"Collect premium: ${condor_recs['max_profit']:.2f}",
                    "time_to_execute": "1 trading session"
                },
                "confidence": round(100 - abs(grade.expected_return_30d) * 5, 1),
            })

        # ── TRANSFORMATION 4: Stock → Covered Call (if at resistance) ──
        if current_asset_type == "STOCK" and position_type == "LONG" and unrealized_pnl_pct > 5:
            cc_recs = self._recommend_covered_call(
                symbol, current_price, quantity, entry_price, grade
            )
            transformations.append({
                "type": "STOCK_TO_COVERED_CALL",
                "from_position": f"Long {quantity} {symbol} @ ${entry_price:.2f}",
                "to_position": cc_recs["strategy"],
                "reason": f"Lock in gains at resistance while generating additional income via calls",
                "details": cc_recs["details"],
                "income_generated": round(cc_recs["premium"], 2),
                "income_yield": round((cc_recs["premium"] / (current_price * quantity)) * 100, 2),
                "upside_capped_at": round(cc_recs["call_strike"], 2),
                "implementation": {
                    "step_1": f"Keep {quantity} shares (already held)",
                    "step_2": f"Sell {cc_recs['contracts']} calls @ ${cc_recs['call_strike']:.2f} strike",
                    "step_3": f"Collect ${cc_recs['premium']:.2f} premium",
                    "time_to_execute": "Immediate"
                },
                "confidence": round(grade.alpha_score * 0.9, 1),  # Slightly lower because upside is capped
            })

        # ── TRANSFORMATION 5: Stay or Hold (if already optimal) ──
        if grade.efficiency_score > 75 and grade.status == "Strong Hold":
            transformations.append({
                "type": "HOLD_CURRENT",
                "recommendation": "MAINTAIN POSITION",
                "reason": f"Position is already optimized for current market regime ({grade.driving_factor})",
                "details": f"Alpha score {grade.alpha_score:.0f}, Risk score {grade.risk_score:.0f} — keep as is",
                "expected_hold_period": "5-30 days (until signal changes)",
                "rebalance_trigger": f"If unrealized P&L drops below ${entry_price * quantity * 0.95:.2f} or model signal reverses"
            })

        return transformations

    def _suggest_hedges(
        self, symbol: str, current_price: float, quantity: float, grade: PositionGrade,
        position_type: str, unrealized_pnl_pct: float
    ) -> List[Dict[str, Any]]:
        """Suggest live hedges based on risk profile."""
        
        hedges = []
        
        # ── HEDGE 1: Protective Put (if downside risk is high) ──
        if grade.risk_score < 40 and unrealized_pnl_pct > 0:
            hedges.append({
                "type": "PROTECTIVE_PUT",
                "trigger": f"Risk score {grade.risk_score:.0f} is elevated",
                "strategy": f"Buy {quantity // 100} put contracts at {round(current_price * 0.95, 2)} strike",
                "cost": round((current_price * 0.02) * (quantity // 100), 2),
                "protection": f"Floors loss at 5% below current price",
                "when_to_use": "If you want to hold long but worried about 10%+ drop",
                "max_protected_loss": round((current_price * 0.05) * quantity, 2),
                "breakeven_if_unexercised": round(unrealized_pnl_pct - 2, 2)
            })

        # ── HEDGE 2: Collar (zero-cost hedge) ──
        if position_type == "LONG" and unrealized_pnl_pct > 8:
            hedges.append({
                "type": "ZERO_COST_COLLAR",
                "trigger": f"Large unrealized gain ({unrealized_pnl_pct:+.1f}%)",
                "strategy": "Buy downside put + Sell upside call to finance it",
                "put_strike": round(current_price * 0.92, 2),
                "call_strike": round(current_price * 1.10, 2),
                "net_cost": 0.0,
                "protection": "Downside protected; upside capped at 10%",
                "when_to_use": "Lock in gains without paying for protection",
            })

        # ── HEDGE 3: Reduce Size (if conviction is declining) ──
        if grade.institutional_conviction < 40 and unrealized_pnl_pct > 3:
            hedges.append({
                "type": "PARTIAL_EXIT",
                "trigger": "Institutional conviction is waning",
                "recommendation": f"Sell 50% of position ({quantity // 2} shares) at current price",
                "reason": "Lock in half the gains; keep half for upside with reduced risk",
                "cash_generated": round((current_price * (quantity // 2)), 2),
                "remaining_position": quantity // 2,
                "risk_reduction": "50%"
            })

        return hedges

    # ── Private Helper Methods for Transformations ──

    def _recommend_call_spread(
        self, symbol: str, current_price: float, quantity: float, entry_price: float, grade: PositionGrade
    ) -> Dict[str, Any]:
        """Generate Bull Call Spread recommendation (LONG stock → BULL CALL SPREAD)."""
        
        contracts = max(1, quantity // 100)
        
        # ATM strike + OTM strike (typically 0.30-0.50 delta)
        iv = grade.volatility_forecast_30d if grade.volatility_forecast_30d > 0 else 0.25
        strike_atm = round(current_price)
        strike_otm = round(current_price * (1 + iv * 0.15))  # ~+15% OTM buffer
        
        dte = 30
        T = dte / 365
        
        # Black-Scholes prices
        opt_engine = self.options_engine
        call_atm = opt_engine.black_scholes(current_price, strike_atm, T, iv, "call")
        call_otm = opt_engine.black_scholes(current_price, strike_otm, T, iv, "call")
        
        net_debit = (call_atm["price"] - call_otm["price"]) * 100 * contracts
        max_profit = ((strike_otm - strike_atm) * 100 * contracts) - net_debit
        max_loss = net_debit
        breakeven = strike_atm + (net_debit / (100 * contracts))
        
        return {
            "strategy": f"Bull Call Spread: BUY {contracts}x ${strike_atm:.0f} call / SELL {contracts}x ${strike_otm:.0f} call",
            "details": f"Reduce capital by 70%, keep ~70% of upside potential, limited downside risk",
            "contracts": contracts,
            "strike_atm": strike_atm,
            "strike_otm": strike_otm,
            "dte": dte,
            "net_debit": round(net_debit, 2),
            "max_profit": round(max_profit, 2),
            "max_loss": round(max_loss, 2),
            "breakeven": round(breakeven, 2),
            "net_delta": round(call_atm["delta"] - call_otm["delta"], 2),
            "capital_efficiency": round(max_profit / max_loss if max_loss > 0 else 0, 2)
        }

    def _recommend_put_spread(
        self, symbol: str, current_price: float, quantity: float, entry_price: float, grade: PositionGrade
    ) -> Dict[str, Any]:
        """Generate Put Spread (protection) recommendation."""
        
        contracts = max(1, quantity // 100)
        iv = grade.volatility_forecast_30d if grade.volatility_forecast_30d > 0 else 0.25
        
        protection_strike = round(current_price * 0.95)  # 5% downside
        cap_strike = round(current_price * 1.08)  # Cap upside @ 8%
        
        dte = 45
        T = dte / 365
        
        put_protect = self.options_engine.black_scholes(current_price, protection_strike, T, iv, "put")
        call_cap = self.options_engine.black_scholes(current_price, cap_strike, T, iv, "call")
        
        cost = ((put_protect["price"] - call_cap["price"]) * 100 * contracts)
        max_loss = (protection_strike - current_price) * quantity + cost
        max_profit = (cap_strike - current_price) * quantity - cost
        
        return {
            "strategy": f"Protective Collar: BUY {contracts}x ${protection_strike:.0f} put / SELL {contracts}x ${cap_strike:.0f} call",
            "details": f"Zero-cost (or low-cost) downside protection with capped upside",
            "contracts": contracts,
            "protection_strike": protection_strike,
            "cap_strike": cap_strike,
            "dte": dte,
            "cost": round(cost, 2),
            "max_loss": round(max_loss, 2),
            "max_profit": round(max_profit, 2),
        }

    def _recommend_iron_condor(
        self, symbol: str, current_price: float, quantity: float, entry_price: float, grade: PositionGrade
    ) -> Dict[str, Any]:
        """Generate Iron Condor recommendation (income strategy)."""
        
        contracts = max(1, quantity // 100)
        iv = grade.volatility_forecast_30d if grade.volatility_forecast_30d > 0 else 0.20
        
        # 0.20 delta strikes (typical for condor wings)
        put_short_strike = round(current_price * 0.92)
        put_long_strike = round(current_price * 0.88)
        call_short_strike = round(current_price * 1.08)
        call_long_strike = round(current_price * 1.12)
        
        dte = 45
        T = dte / 365
        
        opt_engine = self.options_engine
        put_short = opt_engine.black_scholes(current_price, put_short_strike, T, iv, "put")
        put_long = opt_engine.black_scholes(current_price, put_long_strike, T, iv, "put")
        call_short = opt_engine.black_scholes(current_price, call_short_strike, T, iv, "call")
        call_long = opt_engine.black_scholes(current_price, call_long_strike, T, iv, "call")
        
        max_profit = ((put_short["price"] - put_long["price"]) + (call_short["price"] - call_long["price"])) * 100 * contracts
        max_loss = ((put_short_strike - put_long_strike) * 100 * contracts) - max_profit
        
        return {
            "strategy": f"Iron Condor: Sell {contracts}x Put Spread + Sell {contracts}x Call Spread",
            "details": f"Liquidate stock; harvest premium in range-bound market",
            "contracts": contracts,
            "put_strikes": f"Sell {put_short_strike:.0f} / Buy {put_long_strike:.0f}",
            "call_strikes": f"Sell {call_short_strike:.0f} / Buy {call_long_strike:.0f}",
            "dte": dte,
            "lower_strike": put_long_strike,
            "upper_strike": call_long_strike,
            "max_profit": round(max_profit, 2),
            "max_loss": round(max_loss, 2),
            "capital_required": round(max_loss, 2),
            "profit_margin": round((max_profit / max_loss * 100) if max_loss > 0 else 0, 1)
        }

    def _recommend_covered_call(
        self, symbol: str, current_price: float, quantity: float, entry_price: float, grade: PositionGrade
    ) -> Dict[str, Any]:
        """Generate Covered Call recommendation (income on gains)."""
        
        contracts = max(1, quantity // 100)
        iv = grade.volatility_forecast_30d if grade.volatility_forecast_30d > 0 else 0.25
        
        # Sell call at +15% premium (typical for cc)
        call_strike = round(current_price * 1.15)
        
        dte = 30
        T = dte / 365
        
        call_option = self.options_engine.black_scholes(current_price, call_strike, T, iv, "call")
        premium = call_option["price"] * 100 * contracts
        
        return {
            "strategy": f"Covered Call: Keep {quantity} shares / SELL {contracts}x ${call_strike:.0f} calls",
            "details": f"Generate income while holding; cap upside at {round((call_strike - current_price) / current_price * 100, 1)}%",
            "contracts": contracts,
            "call_strike": call_strike,
            "dte": dte,
            "premium": round(premium, 2),
            "yield": round((premium / (current_price * quantity)) * 100, 2),
            "assignment_probability": round(call_option["delta"] * 100, 1)
        }

    def _fallback_grade(self, symbol: str) -> PositionGrade:
        return PositionGrade(
            symbol=symbol,
            alpha_score=50.0,
            risk_score=50.0,
            efficiency_score=50.0,
            expected_return_30d=0.0,
            capital_efficiency=0.0,
            convexity_type="Unknown",
            status="Insufficient Data",
            liquidity_score=0.0,
            risk_adjusted_grade="C",
            recommendations=["Market data link failed. Check symbol formatting or API connectivity."],
            predictive_insights={'liquidity': 'N/A', 'volatility_regime': 'Unknown', 'model_edge': '0.0%'}
        )

def get_position_optimizer():
    return PositionOptimizerEngine()

def render_position_optimizer_ui():
    import streamlit as st
    from data_sources import get_realtime_price

    st.title("Position Optimizer")
    st.caption(
        "Institutional position grading: entry, sizing, convexity, risk-adjusted "
        "outcome, and portfolio contribution for a single position."
    )
    symbol = st.text_input("Symbol", value="AAPL")
    entry_price = st.number_input("Entry Price", value=150.0)
    quantity = st.number_input("Quantity", value=100)
    asset_type = st.selectbox("Asset Type", ["STOCK", "OPTION", "FOREX", "CRYPTO"], index=0)
    metadata = {}

    # "Get Market Price" button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Get Market Price"):
            price_tuple = get_realtime_price(symbol)
            if price_tuple and price_tuple[0]:
                st.success(f"Market price for {symbol}: ${price_tuple[0]:.2f}")
            else:
                st.error("Could not fetch market price.")

    with col2:
        if st.button("Run Position Optimization"):
            import asyncio
            engine = get_position_optimizer()
            grade = asyncio.run(engine.grade_position(symbol, entry_price, quantity, asset_type, metadata))
            # Expanded output
            st.markdown(f"## Position Optimization Report")
            st.markdown(f"### {grade.symbol}")
            st.markdown(f"**Strategy:** {grade.strategy_name}")
            st.markdown(f"**Status:** {grade.status}")
            st.markdown(f"**Outcome Grade:** {grade.risk_adjusted_grade}")
            st.markdown(f"**Type:** {grade.convexity_type}")
            st.markdown("---")
            st.markdown(f"**Alpha:** {grade.alpha_score:.1f}%")
            st.markdown(f"**Efficiency:** {grade.efficiency_score:.1f}%")
            st.markdown(f"**Risk Score:** {grade.risk_score:.1f}%")
            st.markdown(f"**Liquidity:** {grade.liquidity_score:.1f}%")
            st.markdown(f"**Exp Return:** {grade.expected_return_30d:+.1f}%")
            st.markdown(f"**Edge:** {float(grade.alpha_score - 50):+.1f}%")
            st.markdown(f"**Regime:** {grade.predictive_insights.get('volatility_regime', 'N/A')}")
            st.markdown(f"**Macro Sensitivity:** {grade.macro_sensitivity}")
            st.markdown(f"**Institutional Conviction:** {grade.institutional_conviction:.1f}")
            st.markdown(f"**Sentiment Score:** {grade.sentiment_score:.1f}")
            st.markdown("---")
            st.markdown("#### Predictive Outlook")
            st.markdown(f"- Expected 30d Move: {grade.predictive_insights.get('expected_30d_move', 'N/A')}")
            st.markdown(f"- Reward/Risk Ratio: {grade.predictive_insights.get('reward_risk_ratio', 'N/A')}")
            st.markdown(f"- ML Decision: {grade.predictive_insights.get('ml_decision', 'N/A')}")
            st.markdown(f"- Primary Driver: {grade.predictive_insights.get('driving_factor', 'N/A')}")
            st.markdown("---")
            st.markdown("#### Recommendations")
            for rec in grade.recommendations:
                st.write(f"- {rec}")
            st.markdown("---")
            st.markdown(f"[CHART: {grade.symbol}] [PREDICTIVE: {grade.symbol}]")
            # Show all factor scores
            st.markdown("#### Factor Scores")
            for k, v in grade.predictive_insights.get("factor_scores", {}).items():
                st.write(f"{k.title()}: {v:.1f}")
            # Show legs if multi-leg
            if grade.legs:
                st.markdown("#### Strategy Legs")
                for leg in grade.legs:
                    st.write(leg)
            # Show aggregate greeks
            if grade.aggregate_greeks:
                st.markdown("#### Aggregate Greeks")
                for k, v in grade.aggregate_greeks.items():
                    st.write(f"{k.title()}: {v:.3f}")
