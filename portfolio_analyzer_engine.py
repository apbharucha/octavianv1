"""
Portfolio Analyzer Engine - Octavian Terminal
Comprehensive portfolio analysis incorporating past, present, and predictive metrics.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from paper_trading_system import get_paper_trading_system
from types import SimpleNamespace
from risk_engine import correlation_matrix, portfolio_var, _get_asset_data
from advanced_ml_engine import get_ensemble_engine
from institutional_analytics_engine import detect_regime, run_macro_analysis, run_micro_analysis, generate_market_scenarios

@dataclass
class PortfolioAnalytics:
    account_id: str
    total_value: float
    cash_balance: float
    total_pnl: float
    total_pnl_pct: float
    
    # Past Performance (Historical)
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Present Allocation & Risk
    volatility_annual: float = 0.0
    var_95_daily: float = 0.0
    beta_to_spy: float = 1.0
    diversification_score: float = 0.0 # 0-100
    sector_allocation: Dict[str, float] = field(default_factory=dict)
    correlation_avg: float = 0.0
    
    # Predictive & Future
    expected_return_30d: float = 0.0
    regime_alignment: str = "Neutral"
    scenario_impacts: Dict[str, float] = field(default_factory=dict)
    
    # Grading
    health_grade: str = "B" # A-F
    health_score: float = 70.0 # 0-100
    improvement_ideas: List[Dict[str, str]] = field(default_factory=list)

class PortfolioAnalyzerEngine:
    """Institutional-grade portfolio analytics intelligence."""
    
    def __init__(self):
        self.pts = get_paper_trading_system()
        self.ml_engine = get_ensemble_engine()

    def _resolve_or_create_account(self, account_id: str):
        # 1) Direct account lookup
        account = self.pts.get_account(account_id)
        if account:
            return account

        # 2) Treat incoming value as user_id and reuse first existing account
        try:
            if hasattr(self.pts, "list_accounts"):
                existing = self.pts.list_accounts(account_id)
                if existing:
                    return existing[0]
        except Exception:
            pass

        # 3) Create a default account for this user_id (correct create_account signature)
        try:
            if hasattr(self.pts, "create_account"):
                created = self.pts.create_account(
                    user_id=account_id,
                    account_name=f"{account_id}_primary",
                    initial_balance=100000.0
                )
                if created:
                    return created
                # Some implementations return None but still persist; retry list/get
                if hasattr(self.pts, "list_accounts"):
                    existing = self.pts.list_accounts(account_id)
                    if existing:
                        return existing[0]
        except Exception:
            pass

        # 4) Final synthetic fallback
        return SimpleNamespace(
            account_id=account_id,
            initial_balance=100000.0,
            current_balance=100000.0,
            total_pnl=0.0,
            total_pnl_pct=0.0
        )
        
    async def analyze_portfolio(self, account_id: str) -> PortfolioAnalytics:
        """Runs a full tripartite (Past, Present, Future) analysis of the portfolio."""
        account = self._resolve_or_create_account(account_id)
        positions = self.pts.get_positions(account_id) if hasattr(self.pts, "get_positions") else []
        option_positions = self.pts.get_option_positions(account_id) if hasattr(self.pts, "get_option_positions") else []
        all_positions = positions + option_positions
        
        trades = self.pts.get_trade_history(account_id, limit=500) if hasattr(self.pts, "get_trade_history") else []

        past_metrics = self._calculate_past_performance(trades, account.initial_balance)
        present_metrics = await self._analyze_present_state(all_positions, account)
        predictive_metrics = await self._run_predictive_simulations(all_positions, present_metrics)
        grading = self._grade_portfolio(past_metrics, present_metrics, predictive_metrics)

        total_pos_val = sum(getattr(p, "current_price", 0.0) * getattr(p, "quantity", 0.0) * (100 if hasattr(p, "underlying_symbol") else (50 if "=F" in getattr(p, "symbol", "") else 1)) for p in all_positions) if all_positions else 0.0

        # Only pass fields that PortfolioAnalytics expects
        pa_kwargs = dict(
            account_id=account_id,
            total_value=float(account.current_balance + total_pos_val),
            cash_balance=float(account.current_balance),
            total_pnl=float(getattr(account, "total_pnl", 0.0)),
            total_pnl_pct=float(getattr(account, "total_pnl_pct", 0.0)),
        )
        pa_kwargs.update({k: v for k, v in past_metrics.items() if k in PortfolioAnalytics.__dataclass_fields__})
        pa_kwargs.update({k: v for k, v in present_metrics.items() if k in PortfolioAnalytics.__dataclass_fields__})
        pa_kwargs.update({k: v for k, v in predictive_metrics.items() if k in PortfolioAnalytics.__dataclass_fields__})
        pa_kwargs.update({k: v for k, v in grading.items() if k in PortfolioAnalytics.__dataclass_fields__})

        return PortfolioAnalytics(**pa_kwargs)
        
    def _calculate_past_performance(self, trades: List[Any], initial_balance: float) -> Dict[str, Any]:
        """Analyzes trade history for risk-adjusted return metrics."""
        if not trades:
            return {'sharpe_ratio': 0.0, 'win_rate': 0.0, 'profit_factor': 1.0, 'sortino_ratio': 0.0}

        try:
            pnl_list = []
            for t in trades:
                pnl_val = None
                for k in ("pnl", "realized_pnl", "total_pnl", "profit_loss"):
                    if hasattr(t, k):
                        pnl_val = getattr(t, k)
                        break
                    if isinstance(t, dict) and k in t:
                        pnl_val = t[k]
                        break
                if pnl_val is not None:
                    try:
                        pnl_list.append(float(pnl_val))
                    except Exception:
                        pass

            if not pnl_list:
                return {'sharpe_ratio': 0.0, 'win_rate': 0.0, 'profit_factor': 1.0, 'sortino_ratio': 0.0}

            wins = [p for p in pnl_list if p > 0]
            losses = [p for p in pnl_list if p < 0]

            win_rate = len(wins) / len(pnl_list)
            profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else (2.0 if wins else 1.0)

            returns = pd.Series(pnl_list) / max(float(initial_balance), 1.0)
            sharpe = (returns.mean() / (returns.std() + 1e-6)) * np.sqrt(252) if len(returns) > 5 else 0.0

            downside = returns[returns < 0]
            sortino = (returns.mean() / (downside.std() + 1e-6)) * np.sqrt(252) if len(downside) > 3 else sharpe

            # max drawdown from cumulative return proxy
            curve = (1 + returns).cumprod()
            mdd = float(((curve / curve.cummax()) - 1).min()) * 100.0 if len(curve) else 0.0

            return {
                'sharpe_ratio': round(float(sharpe), 2),
                'win_rate': round(win_rate * 100, 1),
                'profit_factor': round(float(profit_factor), 2),
                'sortino_ratio': round(float(sortino), 2),
                'max_drawdown': round(abs(mdd), 2)
            }
        except Exception:
            return {'sharpe_ratio': 0.0, 'win_rate': 0.0, 'profit_factor': 1.0, 'sortino_ratio': 0.0, 'max_drawdown': 0.0}

    async def _analyze_present_state(self, positions: List[Any], account: Any) -> Dict[str, Any]:
        """Analyzes current holdings for allocation, correlation, and risk."""
        if not positions:
            return {
                'diversification_score': 100.0,
                'volatility_annual': 0.0,
                'var_95_daily': 0.0,
                'correlation_avg': 0.0,
                'beta_to_spy': 0.0,
                'sector_allocation': {}
            }
            
        symbols = [getattr(p, "underlying_symbol", p.symbol) for p in positions]
        weights = [(p.current_price * p.quantity * (100 if hasattr(p, "underlying_symbol") else (50 if "=F" in getattr(p, "symbol", "") else 1))) for p in positions]
        total_pos_value = sum(weights)
        norm_weights = [w / total_pos_value for w in weights] if total_pos_value > 0 else [1.0 / len(positions)] * len(positions)

        var_95, vol_annual = portfolio_var(symbols, norm_weights)
        corr_df = correlation_matrix(symbols)
        avg_corr = float(corr_df.mean().mean()) if not corr_df.empty else 0.0

        # beta to SPY (weighted)
        beta_to_spy = 1.0
        try:
            spy = _get_asset_data("SPY")["Close"].pct_change()
            betas = []
            for s in symbols:
                r = _get_asset_data(s)["Close"].pct_change()
                tmp = pd.concat([r, spy], axis=1).dropna()
                if len(tmp) >= 30 and tmp.iloc[:, 1].var() > 0:
                    b = np.cov(tmp.iloc[:, 0], tmp.iloc[:, 1])[0, 1] / tmp.iloc[:, 1].var()
                    betas.append(float(b))
                else:
                    betas.append(1.0)
            beta_to_spy = float(np.dot(norm_weights, betas))
        except Exception:
            beta_to_spy = 1.0
        
        sector_map = {}
        try:
            from ticker_universe import TickerUniverse
            tu = TickerUniverse()
            for p in positions:
                info = tu.get_ticker_info(p.symbol)
                sector = info.get('sector', 'Unknown')
                weight = (p.current_price * p.quantity) / (total_pos_value + max(account.current_balance, 1))
                sector_map[sector] = sector_map.get(sector, 0.0) + weight * 100
        except Exception:
            pass
            
        concentration = sum(w**2 for w in norm_weights)
        div_score = 100 * (1.0 - avg_corr * 0.5 - concentration * 0.5)
        
        return {
            'volatility_annual': round(float(vol_annual), 2),
            'var_95_daily': round(float(var_95), 2),
            'correlation_avg': round(float(avg_corr), 2),
            'diversification_score': round(float(max(0, div_score)), 1),
            'sector_allocation': sector_map,
            'beta_to_spy': round(float(beta_to_spy), 2)
        }

    async def _run_predictive_simulations(self, positions: List[Any], present_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Forecasts performance across multiple macro scenarios grounded in Octavian Master Strategy."""
        if not positions:
            return {'scenario_impacts': {}, 'master_bias': 'Neutral', 'master_conviction': 0.0, 'forecast_period': "30-90 Days", 'primary_risk_factor': "None"}
            
        from master_strategy_engine import get_master_engine
        master_outlook = get_master_engine().get_dominant_outlook()
        
        scenarios = generate_market_scenarios()
        impacts = {}
        beta = present_metrics.get('beta_to_spy', 1.0)
        
        for s in scenarios:
            # support object or dict scenario formats
            if isinstance(s, dict):
                name = s.get("name", "Scenario")
                equity_move = float(s.get("expected_equity_move", s.get("impact_pct", 0.0)))
                prob = float(s.get("probability", 0.2))
                chain = s.get("causal_chain", [])
            else:
                name = getattr(s, "name", "Scenario")
                equity_move = float(getattr(s, "expected_equity_move", 0.0))
                prob = float(getattr(s, "probability", 0.2))
                chain = getattr(s, "causal_chain", [])

            impact_val = equity_move * beta
            prob_adj = 1.0
            if master_outlook.bias == "BULLISH":
                prob_adj = 1.0 + master_outlook.conviction if impact_val > 0 else 1.0 - (master_outlook.conviction * 0.5)
            elif master_outlook.bias == "BEARISH":
                prob_adj = 1.0 + master_outlook.conviction if impact_val < 0 else 1.0 - (master_outlook.conviction * 0.5)

            adj_prob = min(0.95, max(0.01, prob * prob_adj))
            impacts[name] = {
                'impact_pct': round(float(impact_val), 2),
                'probability': round(float(adj_prob), 2),
                'causal_chain': chain,
                'is_likely': adj_prob > 0.3
            }
            
        return {
            'scenario_impacts': impacts,
            'master_bias': master_outlook.bias,
            'master_conviction': master_outlook.conviction,
            'forecast_period': "30-90 Days",
            'primary_risk_factor': "Inflation Repricing" if master_outlook.bias == "BEARISH" else "Liquidity Contraction"
        }

    def _portfolio_model_signal(self, present: Dict[str, Any], past: Dict[str, Any], future: Dict[str, Any]) -> Dict[str, float]:
        # Dynamic feature map; no static playbook
        sharpe = float(past.get("sharpe_ratio", 0.0))
        sortino = float(past.get("sortino_ratio", 0.0))
        div = float(present.get("diversification_score", 50.0))
        corr = float(present.get("correlation_avg", 0.0))
        var95 = float(present.get("var_95_daily", 0.0))
        beta = float(present.get("beta_to_spy", 1.0))
        conviction = float(future.get("master_conviction", 0.0))

        feature_scores = {
            "risk_adjusted_performance": np.clip((sharpe + sortino) * 12.0, 0, 100),
            "diversification": np.clip(div, 0, 100),
            "correlation_penalty": np.clip(100 - corr * 100, 0, 100),
            "tail_risk_penalty": np.clip(100 - var95 * 10, 0, 100),
            "beta_stability": np.clip(100 - abs(beta - 1.0) * 40.0, 0, 100),
            "macro_alignment": np.clip(50 + conviction * 50, 0, 100),
        }

        # Weighted portfolio health driven by model factors
        weights = {
            "risk_adjusted_performance": 0.28,
            "diversification": 0.22,
            "correlation_penalty": 0.16,
            "tail_risk_penalty": 0.16,
            "beta_stability": 0.10,
            "macro_alignment": 0.08,
        }

        health = sum(feature_scores[k] * weights[k] for k in feature_scores)
        return {"health_score": float(health), "factor_scores": feature_scores}

    def _dynamic_health_grade(self, score: float, ref: np.ndarray) -> str:
        q20, q40, q60, q80 = np.percentile(ref, [20, 40, 60, 80])
        if score >= q80: return "A"
        if score >= q60: return "B"
        if score >= q40: return "C"
        if score >= q20: return "D"
        return "F"

    def _grade_portfolio(self, past: Dict, present: Dict, future: Dict) -> Dict[str, Any]:
        """Model-driven portfolio grading and improvement ideas."""
        model_out = self._portfolio_model_signal(present, past, future)
        score = model_out["health_score"]
        fs = model_out["factor_scores"]

        ref = np.array(list(fs.values()) + [35, 45, 55, 65, 75, 85], dtype=float)
        grade = self._dynamic_health_grade(score, ref)

        ranked_weak = sorted(fs.items(), key=lambda kv: kv[1])[:3]
        ideas = [
            {
                "title": f"Improve {k.replace('_', ' ').title()}",
                "strategy": "Reallocate weights based on model contributor deficits.",
                "impact": f"Current contributor score: {v:.1f}"
            }
            for k, v in ranked_weak
        ]

        return {
            'health_score': round(float(score), 1),
            'health_grade': grade,
            'improvement_ideas': ideas
        }

    # ── NEW: entry point for manual / loaded positions ───────────────────────
    async def analyze_portfolio_from_positions(
        self, loaded_positions: list
    ) -> "PortfolioAnalytics":
        """
        Run full tripartite analytics directly from a list of position dicts.
        Each dict must contain: symbol, quantity, entry_price, current_price, asset_type.
        Works without a paper-trading account — all metrics computed from live price data.
        """
        from types import SimpleNamespace

        # Build lightweight position objects that match what the engine expects
        pos_objects = []
        total_entry = 0.0
        total_market = 0.0
        for p in loaded_positions:
            symbol       = str(p.get("symbol", "")).upper().strip()
            quantity     = float(p.get("quantity") or 0)
            entry_price  = float(p.get("entry_price") or 0)
            current_price = p.get("current_price")
            asset_type   = str(p.get("asset_type", "equity")).lower()

            if current_price is None:
                # Try a fresh fetch so risk metrics aren't skipped
                try:
                    from portfolio_analyzer import _fetch_live_price
                    current_price = _fetch_live_price(symbol, asset_type) or entry_price
                except Exception:
                    current_price = entry_price

            current_price = float(current_price or entry_price)
            multiplier = 100.0 if asset_type == "option" else 1.0
            mv = quantity * current_price * multiplier
            ev = quantity * entry_price * multiplier
            total_entry  += ev
            total_market += mv

            pos_objects.append(SimpleNamespace(
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                current_price=current_price,
                asset_type=asset_type,
                market_value=mv,
                entry_value=ev,
            ))

        total_pnl     = total_market - total_entry
        total_pnl_pct = (total_pnl / total_entry * 100.0) if total_entry > 0 else 0.0

        # For past-performance metrics we compute synthetic daily returns from
        # 1-year historical data for each symbol, weighted by portfolio weight
        past_metrics = await self._calculate_past_from_prices(pos_objects, total_market)
        present_metrics = await self._analyze_present_state(pos_objects,
                                                             SimpleNamespace(
                                                                 account_id="manual",
                                                                 initial_balance=total_entry,
                                                                 current_balance=0.0,
                                                                 total_pnl=total_pnl,
                                                                 total_pnl_pct=total_pnl_pct,
                                                             ))
        predictive_metrics = await self._run_predictive_simulations(pos_objects, present_metrics)
        grading = self._grade_portfolio(past_metrics, present_metrics, predictive_metrics)

        pa_kwargs = dict(
            account_id="manual",
            total_value=float(total_market),
            cash_balance=0.0,
            total_pnl=float(total_pnl),
            total_pnl_pct=float(total_pnl_pct),
        )
        pa_kwargs.update({k: v for k, v in past_metrics.items()     if k in PortfolioAnalytics.__dataclass_fields__})
        pa_kwargs.update({k: v for k, v in present_metrics.items()  if k in PortfolioAnalytics.__dataclass_fields__})
        pa_kwargs.update({k: v for k, v in predictive_metrics.items() if k in PortfolioAnalytics.__dataclass_fields__})
        pa_kwargs.update({k: v for k, v in grading.items()          if k in PortfolioAnalytics.__dataclass_fields__})

        return PortfolioAnalytics(**pa_kwargs)

    async def _calculate_past_from_prices(
        self, positions: list, total_market_value: float
    ) -> dict:
        """
        Compute Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor from
        1-year daily price history of each holding, weighted by portfolio weight.
        """
        try:
            import pandas as pd
            import numpy as np
            from risk_engine import _get_asset_data

            if not positions or total_market_value <= 0:
                return {'sharpe_ratio': 0.0, 'win_rate': 0.0,
                        'profit_factor': 1.0, 'sortino_ratio': 0.0, 'max_drawdown': 0.0}

            weighted_returns = None

            for pos in positions:
                w = pos.market_value / total_market_value
                try:
                    df = _get_asset_data(pos.symbol)
                    if df is None or df.empty or "Close" not in df.columns:
                        continue
                    close = df["Close"].squeeze().dropna().astype(float)
                    if len(close) < 30:
                        continue
                    r = close.pct_change().dropna()
                    wr = r * w
                    if weighted_returns is None:
                        weighted_returns = wr
                    else:
                        weighted_returns = weighted_returns.add(wr, fill_value=0)
                except Exception:
                    continue

            if weighted_returns is None or len(weighted_returns) < 20:
                return {'sharpe_ratio': 0.0, 'win_rate': 0.0,
                        'profit_factor': 1.0, 'sortino_ratio': 0.0, 'max_drawdown': 0.0}

            wr = weighted_returns.dropna()
            mean_r = wr.mean()
            std_r  = wr.std()
            sharpe  = (mean_r / (std_r + 1e-9)) * np.sqrt(252) if std_r > 0 else 0.0
            down    = wr[wr < 0]
            sortino = (mean_r / (down.std() + 1e-9)) * np.sqrt(252) if len(down) > 3 else sharpe
            curve   = (1 + wr).cumprod()
            mdd     = float(((curve / curve.cummax()) - 1).min()) * 100.0

            wins   = wr[wr > 0]
            losses = wr[wr < 0]
            win_rate = len(wins) / len(wr) * 100.0 if len(wr) else 0.0
            profit_factor = (abs(wins.sum()) / (abs(losses.sum()) + 1e-9)) if len(losses) else 2.0

            return {
                'sharpe_ratio':  round(float(sharpe), 2),
                'sortino_ratio': round(float(sortino), 2),
                'max_drawdown':  round(abs(mdd), 2),
                'win_rate':      round(win_rate, 1),
                'profit_factor': round(float(profit_factor), 2),
            }
        except Exception:
            return {'sharpe_ratio': 0.0, 'win_rate': 0.0,
                    'profit_factor': 1.0, 'sortino_ratio': 0.0, 'max_drawdown': 0.0}


def get_portfolio_analyzer():
    return PortfolioAnalyzerEngine()
