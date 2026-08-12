import numpy as np
import pandas as pd
from data_sources import get_stock, get_fx, get_futures_proxy

def _get_asset_data(symbol):
    """Get data for any asset type"""
    # Try stock first
    df = get_stock(symbol)
    if not df.empty:
        return df
    
    # Try futures
    if "=F" in symbol or symbol.startswith("ES") or symbol.startswith("NQ"):
        df = get_futures_proxy(symbol)
        if not df.empty:
            return df
    
    # Try FX (handle standard notations like USD/JPY, USD-JPY, USD_JPY)
    if any(x in symbol for x in ["=X", "_", "-", "/"]):
        # Normalize to Oanda format (USD_JPY)
        fx_symbol_oanda = symbol.replace("=X", "").replace("-", "_").replace("/", "_")
        try:
            df = get_fx(fx_symbol_oanda)
            if not df.empty:
                return df
        except Exception:
            pass # Fallback to Yahoo
            
        # Fallback to Yahoo format (USDJPY=X or JPY=X)
        # Try constructing Yahoo symbol: USD/JPY -> USDJPY=X
        yf_symbol = symbol.replace("/", "").replace("-", "").replace("_", "") + "=X"
        df = get_stock(yf_symbol)
        if not df.empty:
            return df
            
    # Try as stock again (for crypto like BTC-USD)
    return get_stock(symbol)

def correlation_matrix(symbols):
    prices = {}

    for s in symbols:
        try:
            df = _get_asset_data(s)
            if not df.empty:
                close_col = df["Close"]
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                # Ensure index is datetime and sorted
                close_col.index = pd.to_datetime(close_col.index)
                prices[s] = close_col.sort_index()
        except Exception as e:
            print(f"Error fetching data for {s} in correlation: {e}")

    if not prices:
        return pd.DataFrame()

    # Create combined DataFrame
    df = pd.DataFrame(prices)
    
    # Handle missing data more robustly
    # 1. Forward fill (limit to 3 days to avoid stale data)
    df = df.ffill(limit=3)
    # 2. Backward fill for the very start
    df = df.bfill(limit=3)
    # 3. Only then drop remaining NaNs
    df = df.dropna()
    
    if df.empty or len(df) < 2:
        return pd.DataFrame()
    
    returns = df.pct_change().dropna()
    if returns.empty:
        return pd.DataFrame()
    
    return returns.corr()

def portfolio_var(symbols, weights, confidence=0.95):
    prices = {}
    valid_weights = []
    
    # Ensure symbols and weights match length initially
    if len(symbols) != len(weights):
        weights = [1.0/len(symbols)] * len(symbols)

    for i, s in enumerate(symbols):
        try:
            df = _get_asset_data(s)
            if not df.empty:
                close_col = df["Close"]
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                close_col.index = pd.to_datetime(close_col.index)
                prices[s] = close_col.sort_index()
                valid_weights.append(weights[i])
        except Exception:
            continue

    if not prices:
        return 0, 0

    # Normalize weights for valid assets
    total_weight = sum(valid_weights)
    if total_weight > 0:
        valid_weights = [w / total_weight for w in valid_weights]
    else:
        valid_weights = [1.0 / len(valid_weights)] * len(valid_weights)

    df = pd.DataFrame(prices)
    df = df.ffill(limit=3).bfill(limit=3).dropna()
    
    if df.empty or len(df) < 10:
        return 0, 0
    
    returns = df.pct_change().dropna()
    if returns.empty or len(returns) < 5:
        return 0, 0

    cov = returns.cov()
    if cov.empty:
        return 0, 0
    
    w_array = np.array(valid_weights)
    
    # Variance calculation: w^T * Cov * w
    portfolio_vol = np.sqrt(np.dot(w_array, np.dot(cov, w_array)))

    # Use Monte Carlo or historical percentile for VaR
    portfolio_returns = returns.dot(w_array)
    if len(portfolio_returns) == 0:
        return 0, 0
    
    var = np.percentile(portfolio_returns, (1 - confidence) * 100)
    
    # Annualize volatility (assuming daily returns)
    annual_vol = portfolio_vol * np.sqrt(252)
    
    return round(var * 100, 2), round(annual_vol * 100, 2)

def position_size(account_size, risk_pct, stop_pct):
    risk_amount = account_size * risk_pct
    size = risk_amount / stop_pct
    return round(size, 2)


def calculate_advanced_risk_metrics(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.95,
    horizon: int = 1,
    monte_carlo_paths: int = 5000,
) -> dict:
    """
    Compute advanced risk metrics from a return series.
    Returns decimal-form metrics (e.g. 0.02 == 2%).
    """
    try:
        r = pd.Series(returns).dropna().astype(float)
    except Exception:
        return {
            "volatility": 0.0,
            "var_historical": 0.0,
            "var_parametric": 0.0,
            "var_monte_carlo": 0.0,
            "cvar": 0.0,
        }

    if r.empty:
        return {
            "volatility": 0.0,
            "var_historical": 0.0,
            "var_parametric": 0.0,
            "var_monte_carlo": 0.0,
            "cvar": 0.0,
        }

    # Scale to requested horizon with square-root-of-time rule
    h = max(int(horizon), 1)
    scale = np.sqrt(h)
    vol = float(r.std(ddof=1)) * scale
    mu = float(r.mean()) * h

    alpha = max(1e-6, min(1 - confidence, 0.5))
    pct = alpha * 100

    # Historical VaR / CVaR
    hist_q = float(np.percentile(r.values, pct)) * scale
    var_historical = abs(hist_q)
    tail = r[r <= np.percentile(r.values, pct)]
    cvar = abs(float(tail.mean()) * scale) if not tail.empty else var_historical

    # Parametric VaR via empirical quantile of simulated normal
    sim_norm = np.random.normal(loc=mu, scale=max(vol, 1e-12), size=10000)
    param_q = float(np.percentile(sim_norm, pct))
    var_parametric = abs(param_q)

    # Monte Carlo VaR using historical drift/vol assumptions
    mc = np.random.normal(loc=mu, scale=max(vol, 1e-12), size=max(1000, int(monte_carlo_paths)))
    mc_q = float(np.percentile(mc, pct))
    var_monte_carlo = abs(mc_q)

    return {
        "volatility": float(max(0.0, vol * np.sqrt(252))),  # annualized volatility
        "var_historical": float(max(0.0, var_historical)),
        "var_parametric": float(max(0.0, var_parametric)),
        "var_monte_carlo": float(max(0.0, var_monte_carlo)),
        "cvar": float(max(0.0, cvar)),
    }


class InstitutionalRiskEngine:
    """
    Advanced Institutional Risk Engine for Octavian.
    Extends base risk metrics with stress testing and probabilistic simulation.
    """
    
    def __init__(self):
        self.scenarios = {
            "Lehman_Crisis": {"SPY": -0.45, "TLT": 0.20, "GLD": 0.15, "USO": -0.60},
            "COVID_Shock": {"SPY": -0.30, "TLT": 0.15, "GLD": -0.05, "USO": -0.70},
            "Vol_Mageddon": {"SPY": -0.15, "TLT": -0.10, "GLD": 0.10, "VIX": 1.50},
            "Tech_Bubble_Burst": {"SPY": -0.25, "XLK": -0.50, "TLT": 0.10, "GLD": 0.05}
        }

    def run_portfolio_stress_test(self, symbols: list[str], weights: list[float]) -> dict:
        """Evaluate portfolio impact across major historical stress scenarios."""
        results = {}
        for name, impacts in self.scenarios.items():
            port_impact = 0
            for sym, weight in zip(symbols, weights):
                # Use proxy or correlation-adjusted impact if symbol not in impacts
                impact = impacts.get(sym, impacts.get("SPY", -0.20))
                port_impact += weight * impact
            results[name] = float(port_impact)
        return results

    def monte_carlo_portfolio_simulation(self, returns: pd.DataFrame, weights: list[float], days: int = 252, sims: int = 5000) -> dict:
        """Simulate 5,000+ paths for probabilistic portfolio evolution."""
        if returns.empty: return {}
        
        mu = returns.mean()
        cov = returns.cov()
        w = np.array(weights)
        
        port_mu = np.dot(w, mu)
        port_std = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        
        # Simulate paths using geometric brownian motion assumption
        sim_results = np.random.normal(port_mu, port_std, (days, sims))
        cum_returns = np.cumprod(1 + sim_results, axis=0)
        
        final_returns = cum_returns[-1, :]
        return {
            "expected_annual_return": float(np.mean(final_returns) - 1),
            "p5_downside_var": float(np.percentile(final_returns, 5) - 1),
            "p95_upside_pot": float(np.percentile(final_returns, 95) - 1),
            "prob_of_drawdown_gt_20pct": float(np.mean(np.min(cum_returns, axis=0) < 0.80))
        }

    def calculate_kelly_fraction(self, win_prob: float, win_loss_ratio: float) -> float:
        """
        Kelly Criterion for optimal position sizing: K% = W - (1-W)/R
        Institutional risk management uses a 'half-Kelly' or 'quarter-Kelly' for safety.
        """
        if win_loss_ratio <= 0: return 0.0
        kelly = win_prob - (1 - win_prob) / win_loss_ratio
        return float(np.clip(kelly, 0, 1.0))

    def get_risk_adjusted_grade(self, sharpe: float, sortino: float, max_drawdown: float) -> str:
        """Score a strategy based on institutional risk/reward ratios."""
        score = (sharpe * 0.4) + (sortino * 0.4) - (max_drawdown * 0.2)
        if score > 2.5: return "AAA (Institutional Elite)"
        if score > 1.5: return "A (Institutional Grade)"
        if score > 0.8: return "B (Standard Alpha)"
        return "C (High Tail Risk)"
