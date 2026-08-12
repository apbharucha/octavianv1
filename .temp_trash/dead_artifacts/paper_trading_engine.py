"""
Octavian Paper Trading Engine
Author: APB - Octavian Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json, os, uuid, threading, time

from data_sources import get_stock, get_fx, get_futures_proxy
from trading_system.risk_manager import AdvancedRiskManager
from trading_system.simulation_grader import SimulationGrader
from trading_system.trading_data_types import TradeAlert

try:
    from data_sources import get_latest_price as _ds_get_latest_price
    _HAS_LATEST = True
except ImportError:
    _HAS_LATEST = False


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class AssetClass(Enum):
    EQUITY = "EQUITY"
    OPTION = "OPTION"
    FUTURE = "FUTURE"
    FOREX = "FOREX"
    CRYPTO = "CRYPTO"


@dataclass
class PaperOrder:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    asset_class: AssetClass = AssetClass.EQUITY
    multiplier: float = 1.0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    commission: float = 0.0
    notes: str = ""
    # Option specific
    strike: Optional[float] = None
    expiry: Optional[str] = None
    option_type: Optional[str] = None


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_cost: float
    asset_class: AssetClass = AssetClass.EQUITY
    multiplier: float = 1.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    market_value: float = 0.0
    # Option specific
    strike: Optional[float] = None
    expiry: Optional[str] = None
    option_type: Optional[str] = None

    def update_price(self, price: float):
        self.current_price = price
        self.market_value = abs(self.quantity) * price * self.multiplier
        if self.avg_cost > 0 and self.quantity != 0:
            if self.quantity > 0:
                self.unrealized_pnl = (price - self.avg_cost) * self.quantity * self.multiplier
            else:
                self.unrealized_pnl = (self.avg_cost - price) * abs(self.quantity) * self.multiplier
            self.unrealized_pnl_pct = self.unrealized_pnl / (abs(self.quantity) * self.avg_cost * self.multiplier)


@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    cash: float
    positions_value: float
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    positions_count: int


PAPER_STATE_FILE = "octavian_paper_trading_state.json"


class PaperTradingEngine:
    def __init__(self, initial_capital: float = 100_000.0, profile_id: str = "default"):
        self.profile_id = profile_id
        self.state_file = f"octavian_paper_trading_state_{self.profile_id}.json"
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[PaperOrder] = []
        self.order_history: List[PaperOrder] = []
        self.trade_log: List[Dict] = []
        self.realized_pnl = 0.0
        self.commission_rate = 0.001
        self.bot_enabled = False
        self.bot_config: Dict[str, Any] = {
            "symbols": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"],
            "max_position_pct": 0.10,
            "min_confidence": 0.35,
            "rebalance_interval_sec": 300,
        }
        self._bot_thread: Optional[threading.Thread] = None
        self._bot_stop = threading.Event()

        # New Components
        self.risk_manager = AdvancedRiskManager(total_capital=initial_capital)
        self.grader = SimulationGrader()
        self.active_alerts: List['TradeAlert'] = []

        self._load_state()

    def _load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.cash = state.get("cash", self.initial_capital)
                self.realized_pnl = state.get("realized_pnl", 0.0)
                for pd_ in state.get("positions", []):
                    ac_str = pd_.get("asset_class", "EQUITY")
                    try:
                        ac = AssetClass[ac_str]
                    except KeyError:
                        ac = AssetClass.EQUITY
                        
                    pos = Position(
                        symbol=pd_["symbol"], 
                        quantity=pd_["quantity"], 
                        avg_cost=pd_["avg_cost"],
                        realized_pnl=pd_.get("realized_pnl", 0),
                        asset_class=ac,
                        multiplier=pd_.get("multiplier", 1.0),
                        strike=pd_.get("strike"),
                        expiry=pd_.get("expiry"),
                        option_type=pd_.get("option_type")
                    )
                    self.positions[pos.symbol] = pos
                self.trade_log = state.get("trade_log", [])[-500:]
        except Exception:
            pass

    def _save_state(self):
        try:
            state = {
                "cash": self.cash, "realized_pnl": self.realized_pnl,
                "positions": [{"symbol": p.symbol, "quantity": p.quantity,
                               "avg_cost": p.avg_cost, "realized_pnl": p.realized_pnl,
                               "asset_class": p.asset_class.name, "multiplier": p.multiplier,
                               "strike": p.strike, "expiry": p.expiry, "option_type": p.option_type}
                              for p in self.positions.values()],
                "trade_log": self.trade_log[-500:],
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception:
            pass

    def get_live_price(self, symbol: str) -> Optional[float]:
        try:
            # Determine yfinance-compatible symbol
            yf_sym = symbol
            if '/' in symbol:
                yf_sym = symbol.replace('/', '') + '=X'

            # Try fresh price first (bypasses cache)
            if _HAS_LATEST:
                p = _ds_get_latest_price(yf_sym)
                if p is not None and p > 0:
                    return p

            # Fallback to existing logic
            if '/' in symbol:
                fx_key = symbol.replace('/', '_')
                df = get_fx(fx_key)
                if df is not None and not df.empty:
                    close = df["Close"]
                    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
                    return float(close.iloc[-1])
                df = get_stock(yf_sym, period="1d")
            elif '=F' in symbol:
                df = get_futures_proxy(symbol, period="1d")
            else:
                df = get_stock(symbol, period="1d")
            if df is not None and not df.empty:
                close = df["Close"]
                if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
                return float(close.iloc[-1])
        except Exception:
            pass
        return None

    def place_order(self, symbol: str, side: str, quantity: float,
                    order_type: str = "MARKET", limit_price: float = None,
                    stop_price: float = None, asset_class: str = "EQUITY",
                    multiplier: float = 1.0, strike: float = None, 
                    expiry: str = None, option_type: str = None, **kwargs) -> PaperOrder:
        try:
            ac = AssetClass[asset_class.upper()]
        except:
            ac = AssetClass.EQUITY
            
        order = PaperOrder(
            order_id=str(uuid.uuid4())[:8], symbol=symbol.upper().strip(),
            side=OrderSide(side.upper()), order_type=OrderType(order_type.upper()),
            quantity=quantity, limit_price=limit_price, stop_price=stop_price,
            asset_class=ac, multiplier=multiplier, strike=strike,
            expiry=expiry, option_type=option_type
        )
        if order.order_type == OrderType.MARKET:
            self._execute_market_order(order)
        else:
            self.orders.append(order)
        return order

    def _execute_market_order(self, order: PaperOrder):
        price = self.get_live_price(order.symbol)
        if price is None:
            order.status = OrderStatus.REJECTED
            order.notes = "Could not fetch price"
            self.order_history.append(order)
            return

        commission = price * order.quantity * order.multiplier * self.commission_rate
        pnl = 0.0

        if order.side == OrderSide.BUY:
            total_cost = price * order.quantity * order.multiplier + commission
            if total_cost > self.cash:
                order.status = OrderStatus.REJECTED
                order.notes = f"Insufficient funds: need ${total_cost:.2f}, have ${self.cash:.2f}"
                self.order_history.append(order)
                return
            self.cash -= total_cost
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                total_qty = pos.quantity + order.quantity
                if total_qty != 0:
                    pos.avg_cost = (pos.avg_cost * pos.quantity + price * order.quantity) / total_qty
                pos.quantity = total_qty
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol, quantity=order.quantity, avg_cost=price,
                    asset_class=order.asset_class, multiplier=order.multiplier,
                    strike=order.strike, expiry=order.expiry, option_type=order.option_type)

            # Register position with risk manager
            self.risk_manager.register_position(order.symbol, {
                "entry_price": price,
                "units": order.quantity,
                "position_dollars": price * order.quantity * order.multiplier,
                "risk_dollars": 0  # Will be updated if stop loss is set
            })

        elif order.side == OrderSide.SELL:
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                pnl = (price - pos.avg_cost) * min(order.quantity, max(0, pos.quantity)) * order.multiplier
                self.realized_pnl += pnl
                pos.realized_pnl += pnl
                pos.quantity -= order.quantity
                if abs(pos.quantity) < 0.0001:
                    self.risk_manager.close_position(order.symbol, price, datetime.now().isoformat())
                    del self.positions[order.symbol]
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol, quantity=-order.quantity, avg_cost=price,
                    asset_class=order.asset_class, multiplier=order.multiplier,
                    strike=order.strike, expiry=order.expiry, option_type=order.option_type)
                pnl = 0
            self.cash += price * order.quantity * order.multiplier - commission

        # Update risk manager capital and daily PnL
        snapshot = self.get_portfolio_snapshot()
        self.risk_manager.update_capital(snapshot.total_value)
        if pnl != 0:
            self.risk_manager.update_daily_pnl(pnl)

        order.status = OrderStatus.FILLED
        order.filled_price = price
        order.filled_quantity = order.quantity
        order.filled_at = datetime.now()
        order.commission = commission
        self.order_history.append(order)
        self.trade_log.append({
            "timestamp": datetime.now().isoformat(), "symbol": order.symbol,
            "side": order.side.value, "quantity": order.quantity,
            "price": price, "commission": commission, "pnl": pnl,
        })
        self._save_state()

    def check_pending_orders(self):
        filled = []
        for order in self.orders:
            if order.status != OrderStatus.PENDING:
                continue
            price = self.get_live_price(order.symbol)
            if price is None:
                continue
            should_fill = False
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and price <= order.limit_price:
                    should_fill = True
                elif order.side == OrderSide.SELL and price >= order.limit_price:
                    should_fill = True
            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.SELL and price <= order.stop_price:
                    should_fill = True
                elif order.side == OrderSide.BUY and price >= order.stop_price:
                    should_fill = True
            if should_fill:
                order.order_type = OrderType.MARKET
                self._execute_market_order(order)
                filled.append(order)
        for o in filled:
            if o in self.orders:
                self.orders.remove(o)

    def cancel_order(self, order_id: str) -> bool:
        for order in self.orders:
            if order.order_id == order_id and order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                self.order_history.append(order)
                self.orders.remove(order)
                return True
        return False

    def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        positions_value = 0.0
        unrealized = 0.0
        for pos in self.positions.values():
            price = self.get_live_price(pos.symbol)
            if price:
                pos.update_price(price)
            positions_value += pos.market_value
            unrealized += pos.unrealized_pnl
        return PortfolioSnapshot(
            timestamp=datetime.now(), cash=self.cash,
            positions_value=positions_value, total_value=self.cash + positions_value,
            unrealized_pnl=unrealized, realized_pnl=self.realized_pnl,
            positions_count=len(self.positions),
        )

    def reset(self):
        self.cash = self.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.order_history.clear()
        self.trade_log.clear()
        self.realized_pnl = 0.0
        self._save_state()

    def start_bot(self, config: Optional[Dict] = None):
        if config:
            self.bot_config.update(config)
        self.bot_enabled = True
        self._bot_stop.clear()
        self._bot_thread = threading.Thread(target=self._bot_loop, daemon=True)
        self._bot_thread.start()

    def stop_bot(self):
        self.bot_enabled = False
        self._bot_stop.set()
        if self._bot_thread:
            self._bot_thread.join(timeout=10)

    def _bot_loop(self):
        try:
            from quant_ensemble_model import get_quant_ensemble
        except ImportError:
            return
        quant = get_quant_ensemble()
        interval = self.bot_config.get("rebalance_interval_sec", 300)
        symbols = self.bot_config.get("symbols", ["SPY"])
        min_conf = self.bot_config.get("min_confidence", 0.35)
        max_pos_pct = self.bot_config.get("max_position_pct", 0.10)

        while not self._bot_stop.is_set():
            try:
                snapshot = self.get_portfolio_snapshot()
                total_value = snapshot.total_value
                for symbol in symbols:
                    try:
                        df = get_stock(symbol, period="3mo")
                        if df is None or df.empty:
                            continue
                        close = df["Close"]
                        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
                        prices = close.dropna().astype(float).values
                        if len(prices) < 40:
                            continue
                        signal = quant.predict(prices)
                        current_price = prices[-1]
                        held_qty = self.positions.get(symbol, Position(symbol, 0, 0)).quantity

                        if signal.direction == "BULLISH" and signal.confidence > min_conf and held_qty <= 0:
                            pos_value = total_value * min(signal.optimal_position_size * 2, max_pos_pct)
                            qty = max(1, int(pos_value / current_price))
                            if held_qty < 0:
                                self.place_order(symbol, "BUY", abs(held_qty))
                            self.place_order(symbol, "BUY", qty)
                        elif signal.direction == "BEARISH" and signal.confidence > min_conf + 0.05 and held_qty > 0:
                            self.place_order(symbol, "SELL", held_qty)
                    except Exception:
                        pass
                self.check_pending_orders()
                self._save_state()
            except Exception:
                pass
            self._bot_stop.wait(timeout=interval)

    def get_performance_summary(self) -> Dict[str, Any]:
        snapshot = self.get_portfolio_snapshot()
        total_return = (snapshot.total_value - self.initial_capital) / self.initial_capital
        return {
            "initial_capital": self.initial_capital,
            "current_value": snapshot.total_value,
            "cash": self.cash,
            "positions_value": snapshot.positions_value,
            "total_return_pct": total_return,
            "total_return_dollar": snapshot.total_value - self.initial_capital,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": snapshot.unrealized_pnl,
            "total_trades": len(self.trade_log),
            "open_positions": len(self.positions),
            "pending_orders": len(self.orders),
            "bot_enabled": self.bot_enabled,
        }

    # --- New Advanced Trading Methods ---
    
    def add_trade_alert(self, alert: 'TradeAlert'):
        """
        Register a proactive trade alert.
        """
        self.active_alerts.append(alert)
        print(f"Added Trade Alert: {alert.symbol} {alert.direction.value} @ {alert.trigger_price}")
        
    def check_trade_alerts(self):
        """
        Check if any trade alerts have been triggered by current market prices.
        """
        triggered_alerts = []
        
        for alert in self.active_alerts:
            # 1. Get Live Price
            price_data = self.get_live_price(alert.symbol)
            if not price_data: continue
            
            # get_live_price returns float directly in this class
            current_price = price_data
            
            # 2. Check Trigger Condition
            triggered = False
            if alert.direction.value == "Long":
                # For Stop Entry Long: Trigger is ABOVE current price usually (Breakout)
                # But if we just added it, we wait for price to cross UP
                # Simplification: If Price >= Trigger
                if current_price >= alert.trigger_price:
                    triggered = True
            elif alert.direction.value == "Short":
                # For Stop Entry Short: Trigger is BELOW current price
                if current_price <= alert.trigger_price:
                    triggered = True
            
            if triggered:
                print(f"TRIGGER HIT: {alert.symbol} @ {current_price} (Target: {alert.trigger_price})")
                self._execute_alert(alert, current_price)
                triggered_alerts.append(alert)
                
        # Remove triggered alerts
        for alert in triggered_alerts:
            if alert in self.active_alerts:
                self.active_alerts.remove(alert)
                
    def _execute_alert(self, alert: 'TradeAlert', execution_price: float):
        """
        Execute a triggered alert as a real order.
        """
        # 1. Final Risk Check
        risk_check = self.risk_manager.check_trade_viability(
            symbol=alert.symbol,
            price=execution_price,
            stop_loss=alert.position_spec.stop_loss_price,
            entry_price=execution_price,
            trade_type=alert.trade_type
        )
        
        if not risk_check["approved"]:
            print(f"Trade Rejected by Risk Manager: {risk_check['reason']}")
            return
            
        units = alert.position_spec.position_size_units
        # Optional: Adjust units based on final risk check if needed
        # units = risk_check["max_units"]
        
        # 2. Place Order
        order = self.place_order(
            symbol=alert.symbol,
            side="BUY" if alert.direction.value == "Long" else "SELL",
            quantity=units,
            order_type="MARKET", # Execute immediately upon trigger
            stop_price=alert.position_spec.stop_loss_price,
            limit_price=alert.position_spec.take_profit_price
        )
        
        if order:
            print(f"EXECUTED ALERT: {alert.symbol} - {units} units @ {execution_price}")
            
    def get_advanced_performance_report(self):
        """
        Get detailed grading report with full trade breakdown.
        """
        summary = self.get_performance_summary()

        # Calculate win rate from closed trades
        closed_trades = [t for t in self.trade_log if 'pnl' in t]
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0.0

        # Calculate Sharpe ratio from trade log (if sufficient history)
        sharpe_ratio = 0.0
        if len(closed_trades) >= 10:
            pnls = [t.get('pnl', 0) for t in closed_trades]
            returns = np.array(pnls) / self.initial_capital
            if np.std(returns) > 0:
                sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)

        # Calculate max drawdown
        max_drawdown_pct = 0.0
        if len(self.trade_log) >= 2:
            cumulative_pnl = 0
            peak = 0
            max_dd = 0
            for trade in self.trade_log:
                cumulative_pnl += trade.get('pnl', 0)
                if cumulative_pnl > peak:
                    peak = cumulative_pnl
                dd = peak - cumulative_pnl
                if dd > max_dd:
                    max_dd = dd
            max_drawdown_pct = max_dd / self.initial_capital if self.initial_capital > 0 else 0

        # Get total PnL
        total_pnl_dollars = summary['realized_pnl'] + summary['unrealized_pnl']

        grade = self.grader.calculate_grade(
            total_pnl_dollars=total_pnl_dollars,
            total_return_pct=summary['total_return_pct'],
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            trade_count=len(closed_trades)
        )

        # Add trade breakdown
        grade['trade_breakdown'] = []
        for i, trade in enumerate(self.trade_log[-20:]):  # Last 20 trades
            grade['trade_breakdown'].append({
                'trade_num': len(self.trade_log) - 20 + i + 1,
                'timestamp': trade.get('timestamp', 'N/A'),
                'symbol': trade.get('symbol', 'N/A'),
                'side': trade.get('side', 'N/A'),
                'quantity': trade.get('quantity', 0),
                'price': trade.get('price', 0),
                'pnl': trade.get('pnl', 0),
                'pnl_pct': (trade.get('pnl', 0) / (trade.get('price', 1) * trade.get('quantity', 1)) * 100) if trade.get('price', 0) > 0 else 0
            })

        grade['risk_metrics'] = self.risk_manager.get_portfolio_risk_metrics()

        return grade


_paper_engines = {}

def get_paper_engine() -> PaperTradingEngine:
    global _paper_engines
    base_name = "default"
    
    # Try to load current profile name if Streamlit is available
    try:
        import streamlit as st
        from trader_profile import get_trader_profile
        if 'active_profile' in st.session_state:
            # We use active profile name or safe formatting
            prof = get_trader_profile()
            name = prof.get('name', '')
            if name:
                # Sanitize the name for file system
                base_name = "".join(x for x in name if x.isalnum() or x in " _-").replace(" ", "_").lower()
    except Exception:
        pass
        
    if base_name not in _paper_engines:
        _paper_engines[base_name] = PaperTradingEngine(profile_id=base_name)
    return _paper_engines[base_name]
