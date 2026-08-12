"""
Octavian Advanced Backtesting Engine
Professional-grade backtesting with detailed metrics, rolling analytics,
and comprehensive graph data generation.

Author: APB - Octavian Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from quant_ensemble_model import get_quant_ensemble, QuantSignal


@dataclass
class BacktestTrade:
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str  # LONG or SHORT
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl_pct: Optional[float]
    pnl_dollar: Optional[float]
    signal_confidence: float
    signal_direction: str
    hold_bars: int = 0
    exit_reason: str = ""
    max_favorable: float = 0.0  # Max favorable excursion
    max_adverse: float = 0.0   # Max adverse excursion


@dataclass
class BacktestResult:
    symbol: str
    period: str
    strategy: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_return_dollar: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    max_drawdown_dollar: float
    max_drawdown_duration_bars: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_win_dollar: float
    avg_loss_dollar: float
    largest_win_pct: float
    largest_loss_pct: float
    avg_hold_bars: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    expectancy: float  # Average $ per trade
    kelly_fraction: float
    # Graph data
    equity_curve: List[float]
    equity_timestamps: List[Any]
    drawdown_curve: List[float]
    rolling_sharpe: List[float]
    rolling_win_rate: List[float]
    monthly_returns: Dict[str, float]
    trade_log: List[BacktestTrade]
    # Benchmark comparison
    benchmark_return: float
    alpha: float
    beta: float
    information_ratio: float


class AdvancedBacktester:
    """
    Professional backtesting engine that tests the Quant Ensemble Model
    against historical data with realistic execution assumptions.
    """

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital
        self.quant_model = get_quant_ensemble()
        self.commission_pct = 0.001  # 10 bps
        self.slippage_pct = 0.0005  # 5 bps
        self.max_position_pct = 0.15  # Max 15% per position

    def run_backtest(self, df: pd.DataFrame, symbol: str,
                     lookback_window: int = 40, rebalance_every: int = 1,
                     use_stop_loss: bool = True, use_take_profit: bool = True) -> Optional[BacktestResult]:
        if df is None or df.empty or len(df) < lookback_window + 10:
            return None
        
        # Robustly extract Close column (handles multi-level yfinance columns)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna().astype(float)
        
        if len(close) < lookback_window + 10:
            return None
        
        volumes = None
        if "Volume" in df.columns:
            v = df["Volume"]
            if isinstance(v, pd.DataFrame):
                v = v.iloc[:, 0]
            volumes = v.dropna().astype(float).values
            # Align volumes length with close
            if len(volumes) != len(close):
                volumes = None
        
        prices = close.values
        timestamps = close.index.tolist()
        n = len(prices)
        
        capital = self.initial_capital
        position = 0.0
        entry_price = 0.0
        entry_time = None
        entry_idx = 0
        entry_conf = 0.0
        entry_dir = ""
        equity = [capital]
        equity_ts = [timestamps[lookback_window - 1]]
        trades: List[BacktestTrade] = []
        peak_equity = capital
        max_fav = 0.0
        max_adv = 0.0

        for i in range(lookback_window, n):
            cp = prices[i]
            pw = prices[max(0, i - lookback_window):i + 1]
            vw = volumes[max(0, i - lookback_window):i + 1] if volumes is not None and len(volumes) > i else None

            # Track unrealized P&L and check stops
            unr = 0.0
            if position != 0:
                if position > 0:
                    unr = (cp - entry_price) / entry_price
                else:
                    unr = (entry_price - cp) / entry_price
                max_fav = max(max_fav, unr)
                max_adv = min(max_adv, unr)
                
                # Stop loss
                if use_stop_loss and unr <= -0.02:
                    if position > 0:
                        ep = cp * (1 - self.slippage_pct)
                        pnl_p = (ep - entry_price) / entry_price
                        capital += position * ep * (1 - self.commission_pct)
                    else:
                        ep = cp * (1 + self.slippage_pct)
                        pnl_p = (entry_price - ep) / entry_price
                        capital -= abs(position) * ep * (1 + self.commission_pct)
                    pnl_d = pnl_p * abs(position) * entry_price
                    trades.append(BacktestTrade(entry_time, timestamps[i], symbol,
                        "LONG" if position > 0 else "SHORT", entry_price, ep,
                        abs(position), pnl_p, pnl_d, entry_conf, entry_dir,
                        i - entry_idx, "STOP_LOSS", max_fav, max_adv))
                    position = 0.0; max_fav = 0.0; max_adv = 0.0
                
                # Take profit
                elif use_take_profit and unr >= 0.04:
                    if position > 0:
                        ep = cp * (1 - self.slippage_pct)
                        pnl_p = (ep - entry_price) / entry_price
                        capital += position * ep * (1 - self.commission_pct)
                    else:
                        ep = cp * (1 + self.slippage_pct)
                        pnl_p = (entry_price - ep) / entry_price
                        capital -= abs(position) * ep * (1 + self.commission_pct)
                    pnl_d = pnl_p * abs(position) * entry_price
                    trades.append(BacktestTrade(entry_time, timestamps[i], symbol,
                        "LONG" if position > 0 else "SHORT", entry_price, ep,
                        abs(position), pnl_p, pnl_d, entry_conf, entry_dir,
                        i - entry_idx, "TAKE_PROFIT", max_fav, max_adv))
                    position = 0.0; max_fav = 0.0; max_adv = 0.0

            # Calculate current equity for position sizing
            if position > 0:
                cur_eq = capital + position * cp
            elif position < 0:
                cur_eq = capital - abs(position) * cp
            else:
                cur_eq = capital
            cur_eq = max(cur_eq, 0.01)

            # Generate signals on rebalance bars
            if i % rebalance_every == 0 and len(pw) >= 20:
                try:
                    signal = self.quant_model.predict(pw, vw)
                except Exception:
                    signal = None
                    
                if signal is not None:
                    if position == 0:
                        # ENTRY LOGIC
                        kelly_size = signal.optimal_position_size
                        effective_size = max(kelly_size * 2, 0.05) if signal.confidence > 0.15 else kelly_size * 2
                        effective_size = min(effective_size, self.max_position_pct)
                        
                        if signal.direction == "BULLISH" and signal.confidence > 0.15:
                            pv = cur_eq * effective_size
                            if pv > 50:
                                ap = cp * (1 + self.slippage_pct)
                                qty = pv / ap
                                position = qty; entry_price = ap; entry_time = timestamps[i]
                                entry_idx = i; entry_conf = signal.confidence; entry_dir = signal.direction
                                capital -= qty * ap * (1 + self.commission_pct); max_fav = 0.0; max_adv = 0.0
                        
                        elif signal.direction == "BEARISH" and signal.confidence > 0.20:
                            pv = cur_eq * effective_size * 0.7
                            if pv > 50:
                                ap = cp * (1 - self.slippage_pct)
                                qty = pv / ap
                                position = -qty; entry_price = ap; entry_time = timestamps[i]
                                entry_idx = i; entry_conf = signal.confidence; entry_dir = signal.direction
                                capital += qty * ap * (1 - self.commission_pct); max_fav = 0.0; max_adv = 0.0
                                
                    elif position > 0 and signal.direction == "BEARISH" and signal.confidence > 0.25:
                        # Exit long on bearish reversal
                        ep = cp * (1 - self.slippage_pct)
                        pnl_p = (ep - entry_price) / entry_price
                        pnl_d = pnl_p * position * entry_price
                        capital += position * ep * (1 - self.commission_pct)
                        trades.append(BacktestTrade(entry_time, timestamps[i], symbol, "LONG",
                            entry_price, ep, position, pnl_p, pnl_d, entry_conf, entry_dir,
                            i - entry_idx, "SIGNAL_REVERSAL", max_fav, max_adv))
                        position = 0.0; max_fav = 0.0; max_adv = 0.0
                        
                    elif position < 0 and signal.direction == "BULLISH" and signal.confidence > 0.25:
                        # Exit short on bullish reversal
                        ep = cp * (1 + self.slippage_pct)
                        pnl_p = (entry_price - ep) / entry_price
                        pnl_d = pnl_p * abs(position) * entry_price
                        capital -= abs(position) * ep * (1 + self.commission_pct)
                        trades.append(BacktestTrade(entry_time, timestamps[i], symbol, "SHORT",
                            entry_price, ep, abs(position), pnl_p, pnl_d, entry_conf, entry_dir,
                            i - entry_idx, "SIGNAL_REVERSAL", max_fav, max_adv))
                        position = 0.0; max_fav = 0.0; max_adv = 0.0

            # Re-update equity for the bar recording
            if position > 0:
                cur_eq = capital + position * cp
            elif position < 0:
                cur_eq = capital - abs(position) * cp
            else:
                cur_eq = capital
            
            cur_eq = max(cur_eq, 0.01)  # prevent negative equity
            equity.append(cur_eq)
            equity_ts.append(timestamps[i])
            if cur_eq > peak_equity:
                peak_equity = cur_eq

        # Close any remaining position at end
        if position != 0:
            fp = prices[-1]
            if position > 0:
                pnl_p = (fp - entry_price) / entry_price
                capital += position * fp * (1 - self.commission_pct)
            else:
                pnl_p = (entry_price - fp) / entry_price
                capital -= abs(position) * fp * (1 + self.commission_pct)
            pnl_d = pnl_p * abs(position) * entry_price
            
            trades.append(BacktestTrade(entry_time, timestamps[-1], symbol,
                "LONG" if position > 0 else "SHORT", entry_price, fp,
                abs(position), pnl_p, pnl_d, entry_conf, entry_dir, 
                n - 1 - entry_idx,
                "END_OF_PERIOD", max_fav, max_adv))
            position = 0.0

        final_capital = capital if position == 0 else capital
        # Recalculate final from last equity point
        if equity:
            final_capital = equity[-1]
            
        return self._compute_metrics(symbol, equity, equity_ts, trades, prices, timestamps, final_capital)

    def _compute_metrics(self, symbol, equity, equity_ts, trades, prices, timestamps, final_capital) -> BacktestResult:
        ea = np.array(equity, dtype=float)
        n_eq = len(ea)
        trp = (final_capital - self.initial_capital) / self.initial_capital
        trd = final_capital - self.initial_capital
        n_years = max(n_eq / 252, 0.01)
        ann = (1 + trp) ** (1 / n_years) - 1 if trp > -1 else -1
        er = np.diff(ea) / ea[:-1]
        er = er[np.isfinite(er)]
        sharpe = (np.mean(er) / np.std(er)) * np.sqrt(252) if len(er) > 1 and np.std(er) > 0 else 0.0
        neg_r = er[er < 0]
        sortino = (np.mean(er) / np.std(neg_r)) * np.sqrt(252) if len(neg_r) > 0 and np.std(neg_r) > 0 else 0.0
        peak = np.maximum.accumulate(ea)
        dd_curve = (ea - peak) / peak
        max_dd = float(np.abs(np.min(dd_curve)))
        max_dd_dollar = float(np.max(peak - ea))
        calmar = ann / max_dd if max_dd > 0 else 0.0
        in_dd = dd_curve < 0
        dd_lengths = []
        c = 0
        for v in in_dd:
            if v:
                c += 1
            else:
                if c > 0:
                    dd_lengths.append(c)
                c = 0
        if c > 0:
            dd_lengths.append(c)
        max_dd_dur = max(dd_lengths) if dd_lengths else 0
        tt = len(trades)
        winners = [t for t in trades if t.pnl_pct and t.pnl_pct > 0]
        losers = [t for t in trades if t.pnl_pct and t.pnl_pct <= 0]
        wc = len(winners)
        lc = len(losers)
        wr = wc / tt if tt > 0 else 0
        awp = float(np.mean([t.pnl_pct for t in winners])) if winners else 0
        alp = float(np.mean([t.pnl_pct for t in losers])) if losers else 0
        awd = float(np.mean([t.pnl_dollar for t in winners if t.pnl_dollar])) if winners else 0
        ald = float(np.mean([t.pnl_dollar for t in losers if t.pnl_dollar])) if losers else 0
        lwp = max((t.pnl_pct for t in winners), default=0)
        llp = min((t.pnl_pct for t in losers), default=0)
        gp = sum(t.pnl_dollar for t in winners if t.pnl_dollar) if winners else 0
        gl = abs(sum(t.pnl_dollar for t in losers if t.pnl_dollar)) if losers else 1e-8
        pf = gp / gl if gl > 0 else (float('inf') if gp > 0 else 0)
        ah = float(np.mean([t.hold_bars for t in trades])) if trades else 0
        mcw = mcl = cw = cl = 0
        for t in trades:
            if t.pnl_pct and t.pnl_pct > 0:
                cw += 1
                cl = 0
            else:
                cl += 1
                cw = 0
            mcw = max(mcw, cw)
            mcl = max(mcl, cl)
        exp = trd / tt if tt > 0 else 0
        kelly = wr - (1 - wr) / abs(awp / alp) if wr > 0 and alp != 0 else 0.0

        # Rolling Sharpe
        w = min(30, n_eq // 4)
        rs_vals = []
        if w > 5:
            for i in range(w, len(er)):
                ch = er[i - w:i]
                rs_vals.append(float(np.mean(ch) / np.std(ch) * np.sqrt(252)) if np.std(ch) > 0 else 0.0)

        # Rolling win rate
        rw_vals = []
        for i in range(len(trades)):
            start = max(0, i - 20)
            chunk = trades[start:i + 1]
            if chunk:
                wr_c = sum(1 for t in chunk if t.pnl_pct and t.pnl_pct > 0) / len(chunk)
                rw_vals.append(float(wr_c))

        # Monthly returns
        monthly = {}
        if len(equity_ts) > 1:
            try:
                eq_series = pd.Series(ea, index=pd.DatetimeIndex(equity_ts))
                monthly_eq = eq_series.resample('ME').last()
                monthly_rets = monthly_eq.pct_change().dropna()
                for idx_m, val in monthly_rets.items():
                    monthly[idx_m.strftime("%Y-%m")] = float(val)
            except Exception:
                pass

        # Benchmark & alpha/beta
        br = (prices[-1] / prices[0] - 1) if prices[0] > 0 else 0
        alpha_val = 0.0
        beta_val = 1.0
        ir = 0.0
        if len(er) > 20:
            bench_r = np.diff(prices[-len(er) - 1:]) / prices[-len(er) - 1:-1]
            bench_r = bench_r[:len(er)]
            if len(bench_r) == len(er) and np.std(bench_r) > 0:
                cov = np.cov(er, bench_r)
                beta_val = cov[0, 1] / (np.var(bench_r) + 1e-8)
                alpha_val = float(np.mean(er) - beta_val * np.mean(bench_r)) * 252
                te = np.std(er - bench_r)
                ir = (np.mean(er) - np.mean(bench_r)) / (te + 1e-8) * np.sqrt(252)

        return BacktestResult(
            symbol=symbol, period=f"{len(prices)} bars", strategy="Quant Ensemble",
            initial_capital=self.initial_capital, final_capital=final_capital,
            total_return_pct=trp, total_return_dollar=trd, annualized_return=ann,
            sharpe_ratio=float(sharpe), sortino_ratio=float(sortino), calmar_ratio=float(calmar),
            max_drawdown_pct=max_dd, max_drawdown_dollar=max_dd_dollar,
            max_drawdown_duration_bars=max_dd_dur, total_trades=tt,
            winning_trades=wc, losing_trades=lc, win_rate=wr, profit_factor=pf,
            avg_win_pct=awp, avg_loss_pct=alp, avg_win_dollar=awd, avg_loss_dollar=ald,
            largest_win_pct=float(lwp), largest_loss_pct=float(llp),
            avg_hold_bars=ah, max_consecutive_wins=mcw, max_consecutive_losses=mcl,
            expectancy=exp, kelly_fraction=float(kelly),
            equity_curve=equity, equity_timestamps=equity_ts,
            drawdown_curve=dd_curve.tolist(), rolling_sharpe=rs_vals,
            rolling_win_rate=rw_vals, monthly_returns=monthly, trade_log=trades,
            benchmark_return=br, alpha=float(alpha_val), beta=float(beta_val),
            information_ratio=float(ir),
        )

    def run(self, df: pd.DataFrame, symbol: str, **kwargs) -> Optional[BacktestResult]:
        """Alias for run_backtest for compatibility"""
        return self.run_backtest(df, symbol, **kwargs)
