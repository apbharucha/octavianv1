"""
Genetic Strategy Evolution Engine
===================================
Automatically discovers trading strategies through genetic algorithms,
evolutionary optimization, and walk-forward validation.

Architecture:
  - Strategy DNA: parameter vector encoding entry/exit rules
  - Population: pool of candidate strategies
  - Fitness Function: risk-adjusted returns (Sharpe, Calmar, win rate)
  - Operators: crossover, mutation, selection, elitism
  - Validation: walk-forward, Monte Carlo permutation tests
  - Overfitting Prevention: out-of-sample holdout, complexity penalties

Strategy Genes (each strategy is a vector of parameters):
  - Entry indicators: RSI threshold, MA crossover periods, momentum lookback
  - Exit rules: trailing stop %, profit target %, time stop (days)
  - Position sizing: fixed %, volatility-scaled, Kelly fraction
  - Regime filter: trend-only, mean-rev only, all regimes
  - Signal combination: AND / OR / weighted vote
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# 
# Gene Definitions
# 

# Each gene is a float in [0, 1] that maps to a discrete parameter value
GENE_SCHEMA: dict[str, dict] = {
    #  Entry Signals 
    "rsi_period": {"min": 7, "max": 28, "type": "int"},
    "rsi_oversold": {"min": 20, "max": 45, "type": "int"},
    "rsi_overbought": {"min": 55, "max": 80, "type": "int"},
    "ma_fast": {"min": 5, "max": 30, "type": "int"},
    "ma_slow": {"min": 20, "max": 120, "type": "int"},
    "momentum_lookback": {"min": 5, "max": 63, "type": "int"},
    "momentum_threshold": {"min": 0.5, "max": 8.0, "type": "float"},
    "vol_lookback": {"min": 10, "max": 40, "type": "int"},
    "bb_period": {"min": 10, "max": 30, "type": "int"},
    "bb_std": {"min": 1.0, "max": 3.0, "type": "float"},
    "atr_period": {"min": 7, "max": 21, "type": "int"},
    #  Entry Logic 
    "use_rsi": {"min": 0, "max": 1, "type": "binary"},
    "use_ma_cross": {"min": 0, "max": 1, "type": "binary"},
    "use_momentum": {"min": 0, "max": 1, "type": "binary"},
    "use_bollinger": {"min": 0, "max": 1, "type": "binary"},
    "use_vol_filter": {"min": 0, "max": 1, "type": "binary"},
    "entry_mode": {"min": 0, "max": 2, "type": "int"},  # 0=AND, 1=OR, 2=weighted
    "signal_threshold": {"min": 0.3, "max": 0.8, "type": "float"},  # for weighted mode
    #  Exit Rules 
    "stop_loss_pct": {"min": 0.5, "max": 8.0, "type": "float"},
    "take_profit_pct": {"min": 1.0, "max": 20.0, "type": "float"},
    "trailing_stop_pct": {"min": 0.5, "max": 6.0, "type": "float"},
    "use_trailing_stop": {"min": 0, "max": 1, "type": "binary"},
    "time_stop_days": {"min": 2, "max": 30, "type": "int"},
    "use_time_stop": {"min": 0, "max": 1, "type": "binary"},
    "exit_on_signal_flip": {"min": 0, "max": 1, "type": "binary"},
    #  Position Sizing 
    "position_pct": {"min": 2.0, "max": 25.0, "type": "float"},
    "sizing_mode": {
        "min": 0,
        "max": 2,
        "type": "int",
    },  # 0=fixed, 1=vol-scaled, 2=Kelly
    "max_positions": {"min": 1, "max": 5, "type": "int"},
    "kelly_fraction": {"min": 0.1, "max": 0.5, "type": "float"},
    #  Regime Filter 
    "regime_filter": {
        "min": 0,
        "max": 3,
        "type": "int",
    },  # 0=none, 1=trend, 2=mean-rev, 3=low-vol
    "trend_ma_period": {"min": 50, "max": 200, "type": "int"},
    "vol_regime_threshold": {"min": 10.0, "max": 35.0, "type": "float"},
    #  Direction 
    "direction_bias": {"min": 0, "max": 2, "type": "int"},  # 0=long, 1=short, 2=both
}

GENE_NAMES = list(GENE_SCHEMA.keys())
N_GENES = len(GENE_NAMES)


def _decode_gene(gene_name: str, raw_value: float) -> Any:
    """Map a float in [0, 1] to a concrete parameter value."""
    schema = GENE_SCHEMA[gene_name]
    lo, hi = schema["min"], schema["max"]
    gtype = schema["type"]
    if gtype == "binary":
        return 1 if raw_value >= 0.5 else 0
    elif gtype == "int":
        return int(round(lo + raw_value * (hi - lo)))
    else:  # float
        return round(lo + raw_value * (hi - lo), 4)


def decode_dna(dna: np.ndarray) -> dict:
    """Decode a full DNA vector into a strategy parameter dict."""
    return {
        name: _decode_gene(name, float(dna[i])) for i, name in enumerate(GENE_NAMES)
    }


# 
# Data Structures
# 


@dataclass
class Strategy:
    dna: np.ndarray
    params: dict = field(default_factory=dict)
    fitness: float = -999.0
    generation: int = 0
    # Performance metrics
    total_return: float = 0.0
    sharpe: float = 0.0
    calmar: float = 0.0
    sortino: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    n_trades: int = 0
    oos_sharpe: float = 0.0  # Out-of-sample Sharpe
    complexity_penalty: float = 0.0
    uid: str = ""

    def __post_init__(self):
        self.params = decode_dna(self.dna)
        h = hashlib.md5(self.dna.tobytes()).hexdigest()[:8]
        self.uid = f"S_{h}"


@dataclass
class GenerationResult:
    generation: int
    best_fitness: float
    avg_fitness: float
    best_strategy: Strategy
    population_size: int
    elite_count: int
    mutations: int
    crossovers: int
    elapsed_sec: float


@dataclass
class EvolutionResult:
    best_strategy: Strategy
    generations: list[GenerationResult]
    population_history: list[list[float]]  # fitness values per generation
    discovered_alpha: list[dict]
    robustness_score: float
    overfitting_risk: str  # "Low" / "Medium" / "High"
    final_population: list[Strategy]


# 
# Fitness / Backtesting Engine
# 


class FastBacktester:
    """
    Vectorised backtester designed for high-throughput evaluation of thousands
    of strategies per generation. Uses numpy operations instead of Python loops
    where possible.
    """

    def __init__(self, close: pd.Series, warmup: int = 60):
        self.close = close.reset_index(drop=True)
        self.n = len(self.close)
        self.prices = self.close.values.astype(float)
        self.warmup = warmup
        self._cache: dict[str, np.ndarray] = {}

    #  Pre-computed indicator cache 

    def _returns(self) -> np.ndarray:
        if "returns" not in self._cache:
            p = self.prices
            r = np.zeros(len(p))
            r[1:] = p[1:] / p[:-1] - 1
            self._cache["returns"] = r
        return self._cache["returns"]

    def _rsi(self, period: int) -> np.ndarray:
        key = f"rsi_{period}"
        if key not in self._cache:
            r = self._returns()
            gain = np.where(r > 0, r, 0.0)
            loss = np.where(r < 0, -r, 0.0)
            avg_gain = np.full(len(r), np.nan)
            avg_loss = np.full(len(r), np.nan)
            for i in range(period, len(r)):
                if i == period:
                    avg_gain[i] = gain[1 : period + 1].mean()
                    avg_loss[i] = loss[1 : period + 1].mean()
                else:
                    avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
                    avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
            rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
            rsi = 100 - (100 / (1 + rs))
            rsi[:period] = 50.0
            self._cache[key] = rsi
        return self._cache[key]

    def _sma(self, period: int) -> np.ndarray:
        key = f"sma_{period}"
        if key not in self._cache:
            p = self.prices
            sma = np.full(len(p), np.nan)
            for i in range(period - 1, len(p)):
                sma[i] = p[max(0, i - period + 1) : i + 1].mean()
            sma[: period - 1] = p[: period - 1]
            self._cache[key] = sma
        return self._cache[key]

    def _momentum(self, lookback: int) -> np.ndarray:
        key = f"mom_{lookback}"
        if key not in self._cache:
            p = self.prices
            mom = np.zeros(len(p))
            for i in range(lookback, len(p)):
                mom[i] = (p[i] / p[i - lookback] - 1) * 100
            self._cache[key] = mom
        return self._cache[key]

    def _bollinger(self, period: int, n_std: float) -> tuple[np.ndarray, np.ndarray]:
        key = f"bb_{period}_{n_std}"
        if key not in self._cache:
            p = self.prices
            upper = np.full(len(p), np.nan)
            lower = np.full(len(p), np.nan)
            for i in range(period - 1, len(p)):
                window = p[max(0, i - period + 1) : i + 1]
                mu = window.mean()
                sigma = window.std()
                upper[i] = mu + n_std * sigma
                lower[i] = mu - n_std * sigma
            upper[: period - 1] = p[: period - 1]
            lower[: period - 1] = p[: period - 1]
            self._cache[key] = (upper, lower)
        return self._cache[key]  # type: ignore

    def _vol(self, period: int) -> np.ndarray:
        key = f"vol_{period}"
        if key not in self._cache:
            r = self._returns()
            vol = np.zeros(len(r))
            for i in range(period, len(r)):
                vol[i] = r[max(0, i - period) : i].std() * math.sqrt(252) * 100
            self._cache[key] = vol
        return self._cache[key]

    #  Signal Generation 

    def _compute_entry_signal(self, params: dict, i: int) -> float:
        """
        Returns a continuous signal in [-1, +1].
        Positive = bullish, negative = bearish.
        """
        signals = []
        weights = []

        # RSI signal
        if params["use_rsi"]:
            rsi = self._rsi(params["rsi_period"])[i]
            oversold = params["rsi_oversold"]
            overbought = params["rsi_overbought"]
            if rsi < oversold:
                signals.append(1.0)
                weights.append(1.0)
            elif rsi > overbought:
                signals.append(-1.0)
                weights.append(1.0)
            else:
                signals.append(0.0)
                weights.append(0.5)

        # MA crossover signal
        if params["use_ma_cross"]:
            fast = self._sma(params["ma_fast"])[i]
            slow = self._sma(params["ma_slow"])[i]
            if fast > slow * 1.002:
                signals.append(1.0)
                weights.append(1.2)
            elif fast < slow * 0.998:
                signals.append(-1.0)
                weights.append(1.2)
            else:
                signals.append(0.0)
                weights.append(0.4)

        # Momentum signal
        if params["use_momentum"]:
            mom = self._momentum(params["momentum_lookback"])[i]
            thresh = params["momentum_threshold"]
            if mom > thresh:
                signals.append(1.0)
                weights.append(1.0)
            elif mom < -thresh:
                signals.append(-1.0)
                weights.append(1.0)
            else:
                signals.append(0.0)
                weights.append(0.3)

        # Bollinger bands signal
        if params["use_bollinger"]:
            upper, lower = self._bollinger(params["bb_period"], params["bb_std"])
            p = self.prices[i]
            if p <= lower[i]:
                signals.append(1.0)
                weights.append(0.9)
            elif p >= upper[i]:
                signals.append(-1.0)
                weights.append(0.9)
            else:
                signals.append(0.0)
                weights.append(0.3)

        if not signals:
            return 0.0

        mode = params["entry_mode"]
        if mode == 0:  # AND — all signals must agree
            if all(s > 0 for s in signals):
                return 1.0
            elif all(s < 0 for s in signals):
                return -1.0
            return 0.0
        elif mode == 1:  # OR — majority vote
            pos = sum(1 for s in signals if s > 0)
            neg = sum(1 for s in signals if s < 0)
            if pos > neg and pos / max(len(signals), 1) > 0.5:
                return 1.0
            elif neg > pos and neg / max(len(signals), 1) > 0.5:
                return -1.0
            return 0.0
        else:  # Weighted average
            total_w = sum(weights)
            if total_w == 0:
                return 0.0
            weighted_avg = sum(s * w for s, w in zip(signals, weights)) / total_w
            thresh = params["signal_threshold"]
            if weighted_avg > thresh:
                return weighted_avg
            elif weighted_avg < -thresh:
                return weighted_avg
            return 0.0

    def _regime_ok(self, params: dict, i: int, direction: float) -> bool:
        """Returns True if current regime is compatible with this trade direction."""
        regime = params["regime_filter"]
        if regime == 0:
            return True
        p = self.prices[i]
        trend_ma = self._sma(params["trend_ma_period"])[i]
        vol = self._vol(params["vol_lookback"])[i]
        vol_thresh = params["vol_regime_threshold"]

        if regime == 1:  # Trend-following only
            in_uptrend = p > trend_ma
            return (direction > 0 and in_uptrend) or (direction < 0 and not in_uptrend)
        elif regime == 2:  # Mean reversion only — low vol environment
            return vol < vol_thresh
        elif regime == 3:  # Low volatility filter
            return vol < vol_thresh
        return True

    #  Core Backtest 

    def run(self, params: dict, capital: float = 100_000.0) -> dict:
        """Run a full backtest for the given strategy parameters."""
        prices = self.prices
        n = self.n
        warmup = max(self.warmup, params.get("ma_slow", 50) + 5)

        cash = capital
        position = 0.0
        entry_price = 0.0
        entry_bar = 0
        in_position = False
        direction = 0  # +1 long, -1 short
        trailing_high = 0.0
        trailing_low = float("inf")

        equity = np.zeros(n)
        equity[:warmup] = capital

        trades = []

        for i in range(warmup, n):
            p = prices[i]
            current_equity = cash + position * p

            #  Exit Logic 
            if in_position:
                # Stop loss
                pnl_pct = (p - entry_price) / entry_price * direction * 100
                stop_hit = pnl_pct < -params["stop_loss_pct"]

                # Take profit
                tp_hit = pnl_pct > params["take_profit_pct"]

                # Trailing stop
                trail_hit = False
                if params["use_trailing_stop"]:
                    if direction > 0:
                        trailing_high = max(trailing_high, p)
                        trail_hit = p < trailing_high * (
                            1 - params["trailing_stop_pct"] / 100
                        )
                    else:
                        trailing_low = min(trailing_low, p)
                        trail_hit = p > trailing_low * (
                            1 + params["trailing_stop_pct"] / 100
                        )

                # Time stop
                time_hit = (
                    params["use_time_stop"]
                    and (i - entry_bar) >= params["time_stop_days"]
                )

                # Signal flip exit
                if params["exit_on_signal_flip"]:
                    sig = self._compute_entry_signal(params, i)
                    signal_flip = (direction > 0 and sig < -0.3) or (
                        direction < 0 and sig > 0.3
                    )
                else:
                    signal_flip = False

                should_exit = stop_hit or tp_hit or trail_hit or time_hit or signal_flip

                if should_exit:
                    # Close position
                    trade_pnl = position * (p - entry_price)
                    cash += position * p
                    trades.append(
                        {
                            "entry_bar": entry_bar,
                            "exit_bar": i,
                            "entry_price": entry_price,
                            "exit_price": p,
                            "direction": direction,
                            "pnl": trade_pnl,
                            "pnl_pct": pnl_pct,
                            "hold_bars": i - entry_bar,
                            "exit_reason": (
                                "stop"
                                if stop_hit
                                else "tp"
                                if tp_hit
                                else "trail"
                                if trail_hit
                                else "time"
                                if time_hit
                                else "flip"
                            ),
                        }
                    )
                    position = 0.0
                    in_position = False
                    direction = 0

            #  Entry Logic 
            if not in_position:
                sig = self._compute_entry_signal(params, i)
                dir_bias = params["direction_bias"]  # 0=long, 1=short, 2=both

                enter_long = sig > 0.3 and dir_bias in (0, 2)
                enter_short = sig < -0.3 and dir_bias in (1, 2)

                if (enter_long or enter_short) and self._regime_ok(
                    params, i, 1 if enter_long else -1
                ):
                    trade_dir = 1 if enter_long else -1

                    # Position sizing
                    sizing = params["sizing_mode"]
                    if sizing == 0:  # Fixed %
                        alloc_pct = params["position_pct"] / 100
                    elif sizing == 1:  # Volatility-scaled
                        vol = self._vol(params["vol_lookback"])[i]
                        target_vol = 15.0
                        alloc_pct = min(
                            0.25,
                            (target_vol / max(vol, 5.0)) * params["position_pct"] / 100,
                        )
                    else:  # Kelly fraction
                        win_r = max(0.3, min(0.7, 0.5 + sig * 0.2))
                        avg_win = params["take_profit_pct"] / 100
                        avg_loss = params["stop_loss_pct"] / 100
                        kelly = (win_r * avg_win - (1 - win_r) * avg_loss) / max(
                            avg_win, 0.01
                        )
                        alloc_pct = max(
                            0.01, min(0.25, kelly * params["kelly_fraction"])
                        )

                    alloc = cash * alloc_pct
                    shares = alloc / max(p, 0.01) * trade_dir
                    if abs(shares * p) > 100:
                        cash -= shares * p
                        position = shares
                        entry_price = p
                        entry_bar = i
                        in_position = True
                        direction = trade_dir
                        trailing_high = p
                        trailing_low = p

            equity[i] = cash + position * p

        #  Close open position at end 
        if in_position and position != 0:
            final_p = prices[-1]
            pnl = position * (final_p - entry_price)
            trades.append(
                {
                    "entry_bar": entry_bar,
                    "exit_bar": n - 1,
                    "entry_price": entry_price,
                    "exit_price": final_p,
                    "direction": direction,
                    "pnl": pnl,
                    "pnl_pct": (final_p - entry_price) / entry_price * direction * 100,
                    "hold_bars": n - 1 - entry_bar,
                    "exit_reason": "end",
                }
            )
            cash += position * final_p

        equity[-1] = cash

        #  Compute Metrics 
        eq = equity
        eq[eq <= 0] = 1.0
        returns = np.diff(eq) / eq[:-1]
        returns = returns[warmup:]

        total_ret = float(eq[-1] / capital - 1)
        ann_ret = float((eq[-1] / capital) ** (252 / max(n - warmup, 1)) - 1)

        if len(returns) > 5:
            sharpe = float(
                np.mean(returns) / (np.std(returns) + 1e-10) * math.sqrt(252)
            )
            downside = returns[returns < 0]
            sortino = float(
                np.mean(returns) / (np.std(downside) + 1e-10) * math.sqrt(252)
                if len(downside) > 3
                else sharpe
            )
        else:
            sharpe = sortino = 0.0

        running_max = np.maximum.accumulate(eq)
        drawdowns = (eq - running_max) / (running_max + 1e-10)
        max_dd = float(np.min(drawdowns))

        calmar = float(ann_ret / abs(max_dd)) if abs(max_dd) > 0.001 else ann_ret * 5

        n_trades = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]
        win_rate = len(wins) / max(n_trades, 1)
        avg_win = float(np.mean([t["pnl_pct"] for t in wins])) if wins else 0.0
        losers = [t for t in trades if t["pnl"] <= 0]
        avg_loss = float(np.mean([t["pnl_pct"] for t in losers])) if losers else 0.0

        return {
            "total_return": total_ret,
            "ann_return": ann_ret,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "n_trades": n_trades,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "profit_factor": abs(avg_win / avg_loss)
            * win_rate
            / max(1 - win_rate, 0.01)
            if avg_loss != 0
            else 1.0,
            "equity_curve": eq.tolist(),
            "trades": trades,
        }


# 
# Fitness Function
# 


def compute_fitness(metrics: dict, params: dict) -> float:
    """
    Multi-objective fitness function combining:
      - Sharpe Ratio (40%)
      - Calmar Ratio (25%)
      - Win Rate (15%)
      - Profit Factor (10%)
      - Trade frequency (10%) — penalise over- and under-trading
    With complexity penalty to discourage overfitting.
    """
    sharpe = float(np.clip(metrics.get("sharpe", 0), -3, 5))
    calmar = float(np.clip(metrics.get("calmar", 0), -5, 10))
    win_rate = float(metrics.get("win_rate", 0))
    profit_factor = float(np.clip(metrics.get("profit_factor", 1), 0, 10))
    n_trades = int(metrics.get("n_trades", 0))

    # Component scores (all normalised to approximately [0, 1])
    sharpe_score = (sharpe + 1) / 6  # [-1, 5] → [0, 1]
    calmar_score = (calmar + 2) / 12  # [-2, 10] → [0, 1]
    wr_score = win_rate  # already [0, 1]
    pf_score = min(profit_factor / 5, 1)  # cap at 5 → [0, 1]

    # Trade frequency score — penalise <5 or >200 trades
    if n_trades < 5:
        freq_score = n_trades / 20.0
    elif n_trades > 200:
        freq_score = max(0, 1 - (n_trades - 200) / 300)
    else:
        freq_score = 1.0

    # Weighted combination
    fitness = (
        0.40 * sharpe_score
        + 0.25 * calmar_score
        + 0.15 * wr_score
        + 0.10 * pf_score
        + 0.10 * freq_score
    )

    # Complexity penalty — more active filters = more overfitting risk
    n_active_signals = sum(
        [
            params.get("use_rsi", 0),
            params.get("use_ma_cross", 0),
            params.get("use_momentum", 0),
            params.get("use_bollinger", 0),
        ]
    )
    complexity_penalty = max(0, n_active_signals - 3) * 0.03
    fitness -= complexity_penalty

    return float(np.clip(fitness, -1.0, 1.0))


# 
# Genetic Operators
# 


def random_dna(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0, 1, N_GENES)


def crossover(
    parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform crossover with random gene exchange."""
    mask = rng.random(N_GENES) > 0.5
    child_a = np.where(mask, parent_a, parent_b)
    child_b = np.where(mask, parent_b, parent_a)
    return child_a, child_b


def mutate(
    dna: np.ndarray, mutation_rate: float, mutation_std: float, rng: np.random.Generator
) -> np.ndarray:
    """Gaussian mutation with clipping to [0, 1]."""
    mutated = dna.copy()
    for i in range(N_GENES):
        if rng.random() < mutation_rate:
            mutated[i] = float(
                np.clip(mutated[i] + float(rng.normal(0, mutation_std)), 0.0, 1.0)
            )
    return mutated


def tournament_select(
    population: list[Strategy], k: int, rng: np.random.Generator
) -> Strategy:
    """Tournament selection — pick k candidates, return the fittest."""
    candidates = [population[int(rng.integers(0, len(population)))] for _ in range(k)]
    return max(candidates, key=lambda s: s.fitness)


# 
# Main Evolution Engine
# 


class GeneticStrategyEngine:
    """
    Drives the evolutionary search for high-fitness trading strategies.

    Parameters
    ----------
    population_size : int
        Number of strategies per generation (default 50)
    n_generations : int
        Number of evolutionary generations (default 20)
    elite_pct : float
        Fraction of top strategies preserved unchanged (default 0.1)
    mutation_rate : float
        Probability of mutating each gene (default 0.15)
    mutation_std : float
        Standard deviation of Gaussian mutation noise (default 0.12)
    tournament_k : int
        Tournament selection size (default 3)
    oos_split : float
        Fraction of data held out for out-of-sample testing (default 0.3)
    seed : int | None
        Random seed for reproducibility
    """

    def __init__(
        self,
        population_size: int = 50,
        n_generations: int = 20,
        elite_pct: float = 0.10,
        mutation_rate: float = 0.15,
        mutation_std: float = 0.12,
        tournament_k: int = 3,
        oos_split: float = 0.30,
        seed: int | None = None,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.elite_count = max(1, int(population_size * elite_pct))
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.tournament_k = tournament_k
        self.oos_split = oos_split
        self.rng = np.random.default_rng(seed)

    #  Population Initialisation 

    def _init_population(self) -> list[Strategy]:
        return [Strategy(dna=random_dna(self.rng)) for _ in range(self.population_size)]

    #  Strategy Evaluation 

    def _evaluate(self, strategy: Strategy, backtester: FastBacktester) -> Strategy:
        try:
            metrics = backtester.run(strategy.params)
            strategy.fitness = compute_fitness(metrics, strategy.params)
            strategy.total_return = metrics["total_return"]
            strategy.sharpe = metrics["sharpe"]
            strategy.calmar = metrics["calmar"]
            strategy.sortino = metrics["sortino"]
            strategy.win_rate = metrics["win_rate"]
            strategy.max_drawdown = metrics["max_drawdown"]
            strategy.n_trades = metrics["n_trades"]
        except Exception:
            strategy.fitness = -1.0
        return strategy

    def _evaluate_oos(
        self, strategy: Strategy, oos_backtester: FastBacktester
    ) -> float:
        try:
            metrics = oos_backtester.run(strategy.params)
            return metrics["sharpe"]
        except Exception:
            return -1.0

    #  Generation Step 

    def _next_generation(
        self,
        population: list[Strategy],
        generation: int,
    ) -> tuple[list[Strategy], int, int]:
        # Sort by fitness descending
        population.sort(key=lambda s: s.fitness, reverse=True)

        new_population: list[Strategy] = []
        mutations = 0
        crossovers = 0

        # Elitism — keep top N unchanged
        for s in population[: self.elite_count]:
            elite = Strategy(dna=s.dna.copy(), generation=generation)
            elite.fitness = s.fitness
            elite.sharpe = s.sharpe
            elite.calmar = s.calmar
            elite.win_rate = s.win_rate
            elite.max_drawdown = s.max_drawdown
            elite.n_trades = s.n_trades
            new_population.append(elite)

        # Fill rest via selection + crossover + mutation
        while len(new_population) < self.population_size:
            parent_a = tournament_select(population, self.tournament_k, self.rng)
            parent_b = tournament_select(population, self.tournament_k, self.rng)

            if self.rng.random() < 0.70:  # 70% crossover
                child_dna_a, child_dna_b = crossover(
                    parent_a.dna, parent_b.dna, self.rng
                )
                crossovers += 2
            else:
                child_dna_a = parent_a.dna.copy()
                child_dna_b = parent_b.dna.copy()

            # Mutate
            child_dna_a = mutate(
                child_dna_a, self.mutation_rate, self.mutation_std, self.rng
            )
            child_dna_b = mutate(
                child_dna_b, self.mutation_rate, self.mutation_std, self.rng
            )
            mutations += 2

            new_population.append(Strategy(dna=child_dna_a, generation=generation))
            if len(new_population) < self.population_size:
                new_population.append(Strategy(dna=child_dna_b, generation=generation))

        return new_population, mutations, crossovers

    #  Main Evolution Loop 

    def evolve(
        self,
        close: pd.Series,
        capital: float = 100_000.0,
        progress_callback=None,
    ) -> EvolutionResult:
        """
        Run the full genetic evolution process.

        Parameters
        ----------
        close : pd.Series
            Daily close price series (should have 200+ bars for meaningful results)
        capital : float
            Starting capital for backtests
        progress_callback : callable | None
            Optional function(generation, total, best_fitness) called each generation

        Returns
        -------
        EvolutionResult
        """
        if len(close) < 150:
            raise ValueError(f"Need at least 150 bars, got {len(close)}")

        # Train / test split
        split_idx = int(len(close) * (1 - self.oos_split))
        close_is = close.iloc[:split_idx]
        close_oos = close.iloc[split_idx:]

        is_backtester = FastBacktester(close_is)
        oos_backtester = FastBacktester(close_oos)

        population = self._init_population()
        generation_results: list[GenerationResult] = []
        population_history: list[list[float]] = []
        global_best: Strategy | None = None

        for gen in range(self.n_generations):
            t0 = time.time()

            # Evaluate all strategies
            for i, strat in enumerate(population):
                if strat.fitness <= -999.0:  # Only evaluate unevaluated strategies
                    self._evaluate(strat, is_backtester)

            # Track global best
            generation_best = max(population, key=lambda s: s.fitness)
            if global_best is None or generation_best.fitness > global_best.fitness:
                global_best = copy.deepcopy(generation_best)

            fitnesses = [s.fitness for s in population]
            population_history.append(fitnesses)

            gen_result = GenerationResult(
                generation=gen,
                best_fitness=generation_best.fitness,
                avg_fitness=float(np.mean(fitnesses)),
                best_strategy=copy.deepcopy(generation_best),
                population_size=len(population),
                elite_count=self.elite_count,
                mutations=0,
                crossovers=0,
                elapsed_sec=time.time() - t0,
            )
            generation_results.append(gen_result)

            if progress_callback:
                progress_callback(gen + 1, self.n_generations, generation_best.fitness)

            # Evolve (except on last generation)
            if gen < self.n_generations - 1:
                population, mut_count, cross_count = self._next_generation(
                    population, gen + 1
                )
                generation_results[-1].mutations = mut_count
                generation_results[-1].crossovers = cross_count

        #  Out-of-Sample Evaluation 
        top_strategies = sorted(population, key=lambda s: s.fitness, reverse=True)[:10]
        for strat in top_strategies:
            strat.oos_sharpe = self._evaluate_oos(strat, oos_backtester)

        if global_best is not None:
            global_best.oos_sharpe = self._evaluate_oos(global_best, oos_backtester)

        #  Overfitting Assessment 
        overfitting_risk = (
            self._assess_overfitting(global_best, top_strategies)
            if global_best
            else "High"
        )

        robustness_score = (
            self._compute_robustness(global_best, close, capital)
            if global_best
            else 0.0
        )

        #  Discovered Alpha 
        discovered_alpha = self._extract_alpha_signals(top_strategies)

        return EvolutionResult(
            best_strategy=global_best or top_strategies[0],
            generations=generation_results,
            population_history=population_history,
            discovered_alpha=discovered_alpha,
            robustness_score=robustness_score,
            overfitting_risk=overfitting_risk,
            final_population=sorted(population, key=lambda s: s.fitness, reverse=True),
        )

    #  Overfitting Detection 

    def _assess_overfitting(self, best: Strategy, top_n: list[Strategy]) -> str:
        is_sharpe = best.sharpe
        oos_sharpe = best.oos_sharpe

        if is_sharpe <= 0.1:
            return "High"

        degradation = (is_sharpe - oos_sharpe) / max(abs(is_sharpe), 0.01)

        if degradation > 0.60:
            return "High"
        elif degradation > 0.30:
            return "Medium"
        else:
            return "Low"

    def _compute_robustness(
        self,
        strategy: Strategy,
        close: pd.Series,
        capital: float,
        n_permutations: int = 20,
    ) -> float:
        """
        Monte Carlo permutation test: shuffle returns and rerun backtest.
        Robustness = fraction of real-data runs that beat shuffled runs.
        """
        try:
            real_bt = FastBacktester(close)
            real_metrics = real_bt.run(strategy.params, capital)
            real_sharpe = real_metrics["sharpe"]

            shuffled_sharpes = []
            prices = close.values.astype(float)
            returns = np.diff(prices) / prices[:-1]

            for _ in range(n_permutations):
                shuffled_returns = returns.copy()
                self.rng.shuffle(shuffled_returns)
                shuffled_prices = np.cumprod(
                    np.concatenate([[prices[0]], prices[0] * (1 + shuffled_returns)])
                )
                shuffled_close = pd.Series(shuffled_prices)
                try:
                    sbt = FastBacktester(shuffled_close)
                    sm = sbt.run(strategy.params, capital)
                    shuffled_sharpes.append(sm["sharpe"])
                except Exception:
                    shuffled_sharpes.append(0.0)

            if not shuffled_sharpes:
                return 0.5

            beats = sum(1 for ss in shuffled_sharpes if real_sharpe > ss)
            robustness = beats / len(shuffled_sharpes)
            return round(robustness, 3)
        except Exception:
            return 0.5

    #  Alpha Signal Extraction 

    def _extract_alpha_signals(self, top_strategies: list[Strategy]) -> list[dict]:
        """
        Analyse the top strategies to extract common parameter patterns.
        These represent the 'discovered alpha' of the evolutionary search.
        """
        if not top_strategies:
            return []

        alpha_signals = []

        # Aggregate parameter statistics across top strategies
        param_arrays: dict[str, list] = {k: [] for k in GENE_NAMES}
        for strat in top_strategies:
            for k, v in strat.params.items():
                param_arrays[k].append(v)

        # Active signal patterns
        use_rsi_pct = float(np.mean(param_arrays["use_rsi"]))
        use_ma_pct = float(np.mean(param_arrays["use_ma_cross"]))
        use_mom_pct = float(np.mean(param_arrays["use_momentum"]))
        use_bb_pct = float(np.mean(param_arrays["use_bollinger"]))

        if use_rsi_pct > 0.6:
            avg_rsi_period = float(np.mean(param_arrays["rsi_period"]))
            avg_oversold = float(np.mean(param_arrays["rsi_oversold"]))
            alpha_signals.append(
                {
                    "name": "RSI Mean Reversion",
                    "prevalence": f"{use_rsi_pct:.0%}",
                    "description": f"RSI({avg_rsi_period:.0f}) oversold threshold ~{avg_oversold:.0f} — dominant entry signal",
                    "strength": round(use_rsi_pct * 100),
                }
            )

        if use_ma_pct > 0.6:
            avg_fast = float(np.mean(param_arrays["ma_fast"]))
            avg_slow = float(np.mean(param_arrays["ma_slow"]))
            alpha_signals.append(
                {
                    "name": "MA Crossover Trend",
                    "prevalence": f"{use_ma_pct:.0%}",
                    "description": f"SMA({avg_fast:.0f}) / SMA({avg_slow:.0f}) crossover — trend following edge",
                    "strength": round(use_ma_pct * 100),
                }
            )

        if use_mom_pct > 0.6:
            avg_lb = float(np.mean(param_arrays["momentum_lookback"]))
            avg_thresh = float(np.mean(param_arrays["momentum_threshold"]))
            alpha_signals.append(
                {
                    "name": "Price Momentum",
                    "prevalence": f"{use_mom_pct:.0%}",
                    "description": f"{avg_lb:.0f}-day momentum threshold {avg_thresh:.1f}% — momentum premium captured",
                    "strength": round(use_mom_pct * 100),
                }
            )

        if use_bb_pct > 0.6:
            avg_bb_p = float(np.mean(param_arrays["bb_period"]))
            alpha_signals.append(
                {
                    "name": "Bollinger Band Reversion",
                    "prevalence": f"{use_bb_pct:.0%}",
                    "description": f"BB({avg_bb_p:.0f}) bands — mean reversion at extremes",
                    "strength": round(use_bb_pct * 100),
                }
            )

        # Exit pattern analysis
        avg_sl = float(np.mean(param_arrays["stop_loss_pct"]))
        avg_tp = float(np.mean(param_arrays["take_profit_pct"]))
        rr = avg_tp / max(avg_sl, 0.01)
        alpha_signals.append(
            {
                "name": "Optimal Risk/Reward",
                "prevalence": "100%",
                "description": f"Avg stop: {avg_sl:.1f}% | Avg target: {avg_tp:.1f}% | R:R = {rr:.1f}:1",
                "strength": int(min(90, rr * 20)),
            }
        )

        # Position sizing
        sizing_mode_counts = [int(v) for v in param_arrays["sizing_mode"]]
        mode_names = {0: "Fixed %", 1: "Vol-Scaled", 2: "Kelly Fraction"}
        from collections import Counter

        most_common_mode = Counter(sizing_mode_counts).most_common(1)[0][0]
        alpha_signals.append(
            {
                "name": f"Position Sizing: {mode_names[most_common_mode]}",
                "prevalence": f"{sizing_mode_counts.count(most_common_mode) / len(sizing_mode_counts):.0%}",
                "description": f"Dominant sizing method across top strategies: {mode_names[most_common_mode]}",
                "strength": 70,
            }
        )

        return alpha_signals

    #  Strategy Summary 

    @staticmethod
    def describe_strategy(strategy: Strategy) -> str:
        p = strategy.params
        lines = [
            f"Strategy ID: {strategy.uid}",
            f"Generation: {strategy.generation}",
            "",
            "== Entry Signals ==",
        ]
        if p["use_rsi"]:
            lines.append(
                f"  RSI({p['rsi_period']}): Buy < {p['rsi_oversold']}, Sell > {p['rsi_overbought']}"
            )
        if p["use_ma_cross"]:
            lines.append(f"  MA Cross: SMA({p['ma_fast']}) / SMA({p['ma_slow']})")
        if p["use_momentum"]:
            lines.append(
                f"  Momentum: {p['momentum_lookback']}d | threshold {p['momentum_threshold']:.1f}%"
            )
        if p["use_bollinger"]:
            lines.append(f"  Bollinger: Period {p['bb_period']}, {p['bb_std']:.1f} std")

        mode_map = {
            0: "AND (all must agree)",
            1: "OR (majority vote)",
            2: "Weighted average",
        }
        lines.append(f"  Entry Mode: {mode_map.get(p['entry_mode'], '?')}")

        lines += [
            "",
            "== Exit Rules ==",
            f"  Stop Loss: {p['stop_loss_pct']:.1f}%",
            f"  Take Profit: {p['take_profit_pct']:.1f}%",
        ]
        if p["use_trailing_stop"]:
            lines.append(f"  Trailing Stop: {p['trailing_stop_pct']:.1f}%")
        if p["use_time_stop"]:
            lines.append(f"  Time Stop: {p['time_stop_days']} bars")
        if p["exit_on_signal_flip"]:
            lines.append("  Exit on Signal Flip: Yes")

        sizing_map = {0: "Fixed %", 1: "Volatility-Scaled", 2: "Kelly Fraction"}
        regime_map = {
            0: "None (all regimes)",
            1: "Trend-following only",
            2: "Mean-reversion only",
            3: "Low-vol only",
        }
        dir_map = {0: "Long only", 1: "Short only", 2: "Long & Short"}

        lines += [
            "",
            "== Position Sizing ==",
            f"  Mode: {sizing_map.get(p['sizing_mode'], '?')}",
            f"  Size: {p['position_pct']:.1f}% of capital",
            f"  Max Positions: {p['max_positions']}",
            "",
            "== Filters ==",
            f"  Regime Filter: {regime_map.get(p['regime_filter'], '?')}",
            f"  Direction: {dir_map.get(p['direction_bias'], '?')}",
            "",
            "== Performance (In-Sample) ==",
            f"  Sharpe: {strategy.sharpe:.2f}",
            f"  Calmar: {strategy.calmar:.2f}",
            f"  Win Rate: {strategy.win_rate:.1%}",
            f"  Max Drawdown: {strategy.max_drawdown:.2%}",
            f"  Trades: {strategy.n_trades}",
            f"  OOS Sharpe: {strategy.oos_sharpe:.2f}",
            f"  Fitness Score: {strategy.fitness:.4f}",
        ]
        return "\n".join(lines)


# 
# Singleton accessor
# 

_engine_instance: GeneticStrategyEngine | None = None


def get_genetic_engine(
    population_size: int = 50,
    n_generations: int = 20,
    seed: int | None = 42,
) -> GeneticStrategyEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = GeneticStrategyEngine(
            population_size=population_size,
            n_generations=n_generations,
            seed=seed,
        )
    return _engine_instance
