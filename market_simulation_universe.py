"""
Market Simulation Universe — Agent-Based Market Microstructure Engine
======================================================================
Simulates realistic market dynamics using multiple agent types, order book
mechanics, liquidity shocks, and behavioral biases.

Architecture:
  - Order Book: Double-auction limit order book with price-time priority
  - Agent Types:
      * Fundamentalists  — value-anchored, mean-reverting
      * Trend Followers  — momentum-driven, extrapolating
      * Market Makers    — liquidity providers, spread capturers
      * Noise Traders    — random order flow, liquidity consumers
      * Institutional    — large, patient, VWAP-style execution
      * HFT Agents       — latency-sensitive arbitrageurs
  - Market Events: earnings, macro shocks, liquidity crises
  - Outputs: price path, volume, order flow imbalance, bid-ask spread,
             realized volatility, regime labels

Usage:
    sim = MarketSimulator(n_agents=100, n_steps=500)
    result = sim.run(initial_price=100.0)
    df = result.to_dataframe()
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

# 
# Enums & Constants
# 


class Side(Enum):
    BID = "BID"
    ASK = "ASK"


class AgentType(Enum):
    FUNDAMENTALIST = "Fundamentalist"
    TREND_FOLLOWER = "Trend Follower"
    MARKET_MAKER = "Market Maker"
    NOISE_TRADER = "Noise Trader"
    INSTITUTIONAL = "Institutional"
    HFT = "HFT"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class EventType(Enum):
    NONE = "none"
    EARNINGS_BEAT = "earnings_beat"
    EARNINGS_MISS = "earnings_miss"
    MACRO_SHOCK = "macro_shock"
    LIQUIDITY_SHOCK = "liquidity_shock"
    LARGE_BUYER = "large_buyer"
    LARGE_SELLER = "large_seller"
    VIX_SPIKE = "vix_spike"


AGENT_DEFAULTS: dict[AgentType, dict] = {
    AgentType.FUNDAMENTALIST: {
        "pct": 0.20,
        "capital_mean": 500_000,
        "capital_std": 200_000,
        "activity_prob": 0.15,
        "order_size_mean": 50,
        "order_size_std": 20,
        "patience": 5,  # bars willing to wait
        "overreaction_bias": 0.0,
    },
    AgentType.TREND_FOLLOWER: {
        "pct": 0.25,
        "capital_mean": 300_000,
        "capital_std": 150_000,
        "activity_prob": 0.20,
        "order_size_mean": 40,
        "order_size_std": 15,
        "lookback_mean": 20,
        "lookback_std": 8,
        "overreaction_bias": 0.3,
    },
    AgentType.MARKET_MAKER: {
        "pct": 0.10,
        "capital_mean": 1_000_000,
        "capital_std": 300_000,
        "activity_prob": 0.90,
        "order_size_mean": 100,
        "order_size_std": 30,
        "spread_bps_mean": 10,
        "spread_bps_std": 4,
    },
    AgentType.NOISE_TRADER: {
        "pct": 0.25,
        "capital_mean": 100_000,
        "capital_std": 50_000,
        "activity_prob": 0.10,
        "order_size_mean": 20,
        "order_size_std": 10,
        "overreaction_bias": 0.5,
    },
    AgentType.INSTITUTIONAL: {
        "pct": 0.10,
        "capital_mean": 5_000_000,
        "capital_std": 2_000_000,
        "activity_prob": 0.05,
        "order_size_mean": 200,
        "order_size_std": 80,
        "slice_factor": 10,  # splits large orders into slices
    },
    AgentType.HFT: {
        "pct": 0.10,
        "capital_mean": 2_000_000,
        "capital_std": 500_000,
        "activity_prob": 0.60,
        "order_size_mean": 30,
        "order_size_std": 10,
        "latency_advantage": 0.5,  # probability of front-running
    },
}


# 
# Order & Trade Data Structures
# 


@dataclass(order=True)
class Order:
    price: float
    time_priority: int
    side: Side = field(compare=False)
    quantity: int = field(compare=False)
    order_type: OrderType = field(compare=False)
    agent_id: int = field(compare=False)
    order_id: int = field(compare=False)

    def __post_init__(self):
        # Bids prioritise highest price; asks lowest price
        if self.side == Side.BID:
            object.__setattr__(self, "_sort_price", -self.price)
        else:
            object.__setattr__(self, "_sort_price", self.price)


@dataclass
class Trade:
    step: int
    price: float
    quantity: int
    aggressor_side: Side
    buyer_id: int
    seller_id: int


@dataclass
class MarketSnapshot:
    step: int
    mid_price: float
    bid_price: float
    ask_price: float
    spread: float
    volume: int
    order_flow_imbalance: float  # (buy_vol - sell_vol) / total_vol
    realized_vol_20: float  # 20-bar annualised vol
    depth_bid: int  # shares on bid side
    depth_ask: int  # shares on ask side
    trades: list[Trade]
    event: EventType
    fundamental_value: float


# 
# Order Book
# 


class OrderBook:
    """
    Double-sided limit order book with price-time priority matching.
    Bids: sorted descending by price (highest first).
    Asks: sorted ascending by price (lowest first).
    """

    def __init__(self):
        self._bids: list[Order] = []  # sorted: best bid first (highest price)
        self._asks: list[Order] = []  # sorted: best ask first (lowest price)
        self._order_counter = 0
        self._time_counter = 0
        self.last_trade_price: float | None = None
        self.total_buy_volume: int = 0
        self.total_sell_volume: int = 0
        self._trades_this_step: list[Trade] = []

    def _next_order_id(self) -> int:
        self._order_counter += 1
        return self._order_counter

    def _next_time(self) -> int:
        self._time_counter += 1
        return self._time_counter

    @property
    def best_bid(self) -> float | None:
        return self._bids[0].price if self._bids else None

    @property
    def best_ask(self) -> float | None:
        return self._asks[0].price if self._asks else None

    @property
    def mid_price(self) -> float | None:
        b = self.best_bid
        a = self.best_ask
        if b is not None and a is not None:
            return (b + a) / 2
        return b or a

    @property
    def spread(self) -> float:
        b = self.best_bid
        a = self.best_ask
        if b is not None and a is not None:
            return a - b
        return 0.0

    @property
    def bid_depth(self) -> int:
        return sum(o.quantity for o in self._bids)

    @property
    def ask_depth(self) -> int:
        return sum(o.quantity for o in self._asks)

    def reset_step_stats(self):
        self.total_buy_volume = 0
        self.total_sell_volume = 0
        self._trades_this_step = []

    def submit_limit_order(
        self, side: Side, price: float, quantity: int, agent_id: int, step: int
    ) -> list[Trade]:
        """Submit a limit order; match immediately if crossing."""
        order = Order(
            price=price,
            time_priority=self._next_time(),
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            agent_id=agent_id,
            order_id=self._next_order_id(),
        )
        trades = self._match(order, step)
        if order.quantity > 0:
            self._insert(order)
        return trades

    def submit_market_order(
        self, side: Side, quantity: int, agent_id: int, step: int
    ) -> list[Trade]:
        """Market order — matches against best available liquidity."""
        price = 1e10 if side == Side.BID else 0.0
        order = Order(
            price=price,
            time_priority=self._next_time(),
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            agent_id=agent_id,
            order_id=self._next_order_id(),
        )
        trades = self._match(order, step)
        return trades

    def _match(self, incoming: Order, step: int) -> list[Trade]:
        trades = []
        opposite = self._asks if incoming.side == Side.BID else self._bids

        while incoming.quantity > 0 and opposite:
            best = opposite[0]

            # Check if order crosses
            if incoming.side == Side.BID and incoming.price < best.price:
                break
            if incoming.side == Side.ASK and incoming.price > best.price:
                break

            # Execute trade
            exec_qty = min(incoming.quantity, best.quantity)
            exec_price = best.price  # passive side sets price

            trade = Trade(
                step=step,
                price=exec_price,
                quantity=exec_qty,
                aggressor_side=incoming.side,
                buyer_id=incoming.agent_id
                if incoming.side == Side.BID
                else best.agent_id,
                seller_id=best.agent_id
                if incoming.side == Side.BID
                else incoming.agent_id,
            )
            trades.append(trade)
            self._trades_this_step.append(trade)

            incoming.quantity -= exec_qty
            best.quantity -= exec_qty

            if incoming.side == Side.BID:
                self.total_buy_volume += exec_qty
            else:
                self.total_sell_volume += exec_qty

            self.last_trade_price = exec_price

            if best.quantity == 0:
                opposite.pop(0)

        return trades

    def _insert(self, order: Order):
        """Insert a remaining unmatched limit order into the book."""
        if order.side == Side.BID:
            # Bids: descending price, then ascending time
            i = 0
            while i < len(self._bids):
                if order.price > self._bids[i].price or (
                    order.price == self._bids[i].price
                    and order.time_priority < self._bids[i].time_priority
                ):
                    break
                i += 1
            self._bids.insert(i, order)
        else:
            # Asks: ascending price, then ascending time
            i = 0
            while i < len(self._asks):
                if order.price < self._asks[i].price or (
                    order.price == self._asks[i].price
                    and order.time_priority < self._asks[i].time_priority
                ):
                    break
                i += 1
            self._asks.insert(i, order)

    def cancel_stale_orders(self, current_mid: float, tolerance_pct: float = 0.05):
        """Remove orders that are far from the current mid price."""
        threshold = current_mid * tolerance_pct
        self._bids = [
            o for o in self._bids if abs(o.price - current_mid) < threshold * 3
        ]
        self._asks = [
            o for o in self._asks if abs(o.price - current_mid) < threshold * 3
        ]

    def get_step_trades(self) -> list[Trade]:
        return list(self._trades_this_step)


# 
# Agent Base Class
# 


class Agent:
    def __init__(
        self,
        agent_id: int,
        agent_type: AgentType,
        capital: float,
        rng: np.random.Generator,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capital = capital
        self.position = 0  # shares held (can be negative for shorts)
        self.pnl = 0.0
        self.rng = rng
        self.last_price = 0.0
        self.trades_done = 0

    def decide(
        self,
        step: int,
        book: OrderBook,
        price_history: list[float],
        fundamental_value: float,
        market_vol: float,
    ) -> list[tuple]:
        """
        Returns list of (side, price, quantity, order_type) tuples.
        Subclasses override this method.
        """
        return []

    def _rand_size(self, mean: int, std: int) -> int:
        return max(1, int(self.rng.normal(mean, std)))

    def _rand_price(self, mid: float, std_pct: float) -> float:
        return max(0.01, mid * (1 + float(self.rng.normal(0, std_pct))))


# 
# Concrete Agent Types
# 


class FundamentalistAgent(Agent):
    """
    Buys when price is below fundamental value, sells above.
    Anchored to a mean-reverting fundamental estimate.
    """

    def __init__(self, agent_id, capital, rng, params):
        super().__init__(agent_id, AgentType.FUNDAMENTALIST, capital, rng)
        self.activity_prob = params.get("activity_prob", 0.15)
        self.order_size_mean = params.get("order_size_mean", 50)
        self.order_size_std = params.get("order_size_std", 20)
        self.conviction_threshold = float(
            rng.uniform(0.005, 0.025)
        )  # % mispricing to act

    def decide(self, step, book, price_history, fundamental_value, market_vol):
        if self.rng.random() > self.activity_prob:
            return []
        mid = book.mid_price
        if mid is None or fundamental_value <= 0:
            return []
        mispricing = (fundamental_value - mid) / fundamental_value
        if abs(mispricing) < self.conviction_threshold:
            return []
        qty = self._rand_size(self.order_size_mean, self.order_size_std)
        if mispricing > 0:  # undervalued → buy
            limit_price = round(mid * (1 + float(self.rng.uniform(0, 0.002))), 2)
            return [(Side.BID, limit_price, qty, OrderType.LIMIT)]
        else:  # overvalued → sell
            limit_price = round(mid * (1 - float(self.rng.uniform(0, 0.002))), 2)
            return [(Side.ASK, limit_price, qty, OrderType.LIMIT)]


class TrendFollowerAgent(Agent):
    """
    Extrapolates recent price momentum. Buys into uptrends, sells into downtrends.
    Introduces positive feedback loops and can amplify moves.
    """

    def __init__(self, agent_id, capital, rng, params):
        super().__init__(agent_id, AgentType.TREND_FOLLOWER, capital, rng)
        self.activity_prob = params.get("activity_prob", 0.20)
        self.order_size_mean = params.get("order_size_mean", 40)
        self.order_size_std = params.get("order_size_std", 15)
        self.lookback = max(
            5,
            int(
                rng.normal(
                    params.get("lookback_mean", 20),
                    params.get("lookback_std", 8),
                )
            ),
        )
        self.threshold = float(rng.uniform(0.003, 0.015))
        self.overreaction = params.get("overreaction_bias", 0.3)

    def decide(self, step, book, price_history, fundamental_value, market_vol):
        if self.rng.random() > self.activity_prob:
            return []
        if len(price_history) < self.lookback + 1:
            return []
        mid = book.mid_price
        if mid is None:
            return []
        past = price_history[-self.lookback]
        if past <= 0:
            return []
        momentum = (price_history[-1] - past) / past
        if abs(momentum) < self.threshold:
            return []
        qty = self._rand_size(self.order_size_mean, self.order_size_std)
        # Overreaction bias: trend followers push orders more aggressively
        aggression = 1.0 + self.overreaction * abs(momentum) * 10
        qty = min(int(qty * aggression), qty * 3)
        if momentum > 0:  # uptrend → buy
            price = round(mid * (1 + float(self.rng.uniform(0, 0.003))), 2)
            return [(Side.BID, price, qty, OrderType.LIMIT)]
        else:  # downtrend → sell
            price = round(mid * (1 - float(self.rng.uniform(0, 0.003))), 2)
            return [(Side.ASK, price, qty, OrderType.LIMIT)]


class MarketMakerAgent(Agent):
    """
    Continuously quotes bid and ask around mid, earns the spread.
    Adjusts quotes based on inventory and volatility.
    """

    def __init__(self, agent_id, capital, rng, params):
        super().__init__(agent_id, AgentType.MARKET_MAKER, capital, rng)
        self.activity_prob = params.get("activity_prob", 0.90)
        self.order_size_mean = params.get("order_size_mean", 100)
        self.order_size_std = params.get("order_size_std", 30)
        self.spread_bps = float(
            rng.normal(
                params.get("spread_bps_mean", 10),
                params.get("spread_bps_std", 4),
            )
        )
        self.spread_bps = max(2, self.spread_bps)
        self.inventory_limit = 500  # max net shares before de-risking

    def decide(self, step, book, price_history, fundamental_value, market_vol):
        if self.rng.random() > self.activity_prob:
            return []
        mid = book.mid_price
        if mid is None:
            return []

        # Widen spread if volatile or heavy inventory
        vol_adj = 1.0 + max(0, market_vol - 20) / 10
        inv_adj = 1.0 + abs(self.position) / self.inventory_limit * 0.5
        effective_spread = self.spread_bps / 10000 * vol_adj * inv_adj

        half = effective_spread / 2
        bid_price = round(mid * (1 - half), 2)
        ask_price = round(mid * (1 + half), 2)
        if bid_price <= 0 or ask_price <= bid_price:
            return []

        # Skew quotes to manage inventory
        qty = self._rand_size(self.order_size_mean, self.order_size_std)
        inv_skew = min(0.3, abs(self.position) / max(self.inventory_limit, 1))
        bid_qty = int(qty * (1 - inv_skew if self.position > 0 else 1 + inv_skew * 0.5))
        ask_qty = int(qty * (1 - inv_skew if self.position < 0 else 1 + inv_skew * 0.5))
        bid_qty = max(1, bid_qty)
        ask_qty = max(1, ask_qty)

        return [
            (Side.BID, bid_price, bid_qty, OrderType.LIMIT),
            (Side.ASK, ask_price, ask_qty, OrderType.LIMIT),
        ]


class NoiseTraderAgent(Agent):
    """
    Generates random order flow. Represents uninformed retail activity.
    """

    def __init__(self, agent_id, capital, rng, params):
        super().__init__(agent_id, AgentType.NOISE_TRADER, capital, rng)
        self.activity_prob = params.get("activity_prob", 0.10)
        self.order_size_mean = params.get("order_size_mean", 20)
        self.order_size_std = params.get("order_size_std", 10)

    def decide(self, step, book, price_history, fundamental_value, market_vol):
        if self.rng.random() > self.activity_prob:
            return []
        mid = book.mid_price
        if mid is None:
            return []
        side = Side.BID if self.rng.random() > 0.5 else Side.ASK
        qty = self._rand_size(self.order_size_mean, self.order_size_std)
        spread_noise = float(self.rng.normal(0, 0.002))
        if side == Side.BID:
            price = round(mid * (1 + max(-0.005, spread_noise)), 2)
        else:
            price = round(mid * (1 + min(0.005, spread_noise)), 2)
        price = max(0.01, price)
        return [(side, price, qty, OrderType.LIMIT)]


class InstitutionalAgent(Agent):
    """
    Large patient buyer/seller that executes via VWAP-style slicing.
    Represents pension funds, hedge funds, mutual funds.
    """

    def __init__(self, agent_id, capital, rng, params):
        super().__init__(agent_id, AgentType.INSTITUTIONAL, capital, rng)
        self.activity_prob = params.get("activity_prob", 0.05)
        self.order_size_mean = params.get("order_size_mean", 200)
        self.order_size_std = params.get("order_size_std", 80)
        self.slice_factor = params.get("slice_factor", 10)
        # Pre-determine direction bias for this agent's lifecycle
        self.direction_bias = 1 if rng.random() > 0.5 else -1
        self.remaining_order = 0
        self.order_side: Side = Side.BID

    def decide(self, step, book, price_history, fundamental_value, market_vol):
        mid = book.mid_price
        if mid is None:
            return []

        # Initiate new large order occasionally
        if self.remaining_order == 0 and self.rng.random() < self.activity_prob:
            total_size = self._rand_size(
                self.order_size_mean * self.slice_factor,
                self.order_size_std * self.slice_factor,
            )
            self.remaining_order = total_size
            self.order_side = Side.BID if self.direction_bias > 0 else Side.ASK

        if self.remaining_order <= 0:
            return []

        # Slice the order
        slice_size = max(1, self.remaining_order // self.slice_factor)
        slice_size = min(slice_size, self.remaining_order)

        # VWAP-style: work orders passively within the spread
        if self.order_side == Side.BID:
            price = round(mid * (1 - float(self.rng.uniform(0.0005, 0.002))), 2)
        else:
            price = round(mid * (1 + float(self.rng.uniform(0.0005, 0.002))), 2)

        self.remaining_order -= slice_size
        return [(self.order_side, max(0.01, price), slice_size, OrderType.LIMIT)]


class HFTAgent(Agent):
    """
    High-frequency trading agent with latency advantage.
    Performs statistical arbitrage and front-running of order flow.
    """

    def __init__(self, agent_id, capital, rng, params):
        super().__init__(agent_id, AgentType.HFT, capital, rng)
        self.activity_prob = params.get("activity_prob", 0.60)
        self.order_size_mean = params.get("order_size_mean", 30)
        self.order_size_std = params.get("order_size_std", 10)
        self.latency_advantage = params.get("latency_advantage", 0.5)
        self.max_position = 100  # keeps position flat intraday

    def decide(self, step, book, price_history, fundamental_value, market_vol):
        if self.rng.random() > self.activity_prob:
            return []
        mid = book.mid_price
        if mid is None:
            return []

        orders = []

        # Mean-reversion stat arb within very tight range
        if len(price_history) >= 5:
            short_avg = float(np.mean(price_history[-5:]))
            deviation = (mid - short_avg) / max(short_avg, 0.01)
            if abs(deviation) > 0.001 and abs(self.position) < self.max_position:
                qty = self._rand_size(self.order_size_mean, self.order_size_std)
                if deviation > 0:  # price above short avg → sell
                    price = round(mid * (1 + 0.0001), 2)
                    orders.append((Side.ASK, price, qty, OrderType.LIMIT))
                else:  # price below short avg → buy
                    price = round(mid * (1 - 0.0001), 2)
                    orders.append((Side.BID, price, qty, OrderType.LIMIT))

        # Inventory flattening
        if abs(self.position) > self.max_position * 0.8:
            qty = max(1, abs(self.position) // 2)
            if self.position > 0:
                orders.append((Side.ASK, round(mid, 2), qty, OrderType.MARKET))
            else:
                orders.append((Side.BID, round(mid, 2), qty, OrderType.MARKET))

        return orders


# 
# Simulation Result
# 


@dataclass
class SimulationResult:
    snapshots: list[MarketSnapshot]
    agent_pnls: dict[int, float]
    agent_types: dict[int, AgentType]
    events: list[tuple[int, EventType, str]]
    final_price: float
    total_volume: int
    n_trades: int
    params: dict

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for s in self.snapshots:
            rows.append(
                {
                    "step": s.step,
                    "mid_price": s.mid_price,
                    "bid_price": s.bid_price,
                    "ask_price": s.ask_price,
                    "spread": s.spread,
                    "volume": s.volume,
                    "ofi": s.order_flow_imbalance,
                    "realized_vol": s.realized_vol_20,
                    "bid_depth": s.depth_bid,
                    "ask_depth": s.depth_ask,
                    "n_trades": len(s.trades),
                    "event": s.event.value,
                    "fundamental_value": s.fundamental_value,
                }
            )
        return pd.DataFrame(rows)

    def price_series(self) -> pd.Series:
        return pd.Series(
            [s.mid_price for s in self.snapshots],
            name="price",
        )

    def volume_series(self) -> pd.Series:
        return pd.Series(
            [s.volume for s in self.snapshots],
            name="volume",
        )

    def ofi_series(self) -> pd.Series:
        return pd.Series(
            [s.order_flow_imbalance for s in self.snapshots],
            name="ofi",
        )

    def regime_labels(self) -> list[str]:
        """Classify each step into a market regime."""
        labels = []
        prices = [s.mid_price for s in self.snapshots]
        for i, snap in enumerate(self.snapshots):
            vol = snap.realized_vol_20
            if i < 20:
                labels.append("Warming Up")
                continue
            recent_ret = (prices[i] - prices[i - 20]) / max(prices[i - 20], 0.01)
            if vol > 35:
                labels.append("High Vol / Crisis")
            elif vol > 22 and recent_ret < -0.05:
                labels.append("Bear / Drawdown")
            elif recent_ret > 0.05 and vol < 22:
                labels.append("Bull / Trending")
            elif abs(recent_ret) < 0.02 and vol < 18:
                labels.append("Low Vol / Range")
            else:
                labels.append("Transition")
        return labels

    def agent_type_pnl_summary(self) -> pd.DataFrame:
        rows = []
        type_pnl: dict[str, list[float]] = {}
        for agent_id, pnl in self.agent_pnls.items():
            atype = self.agent_types.get(agent_id, AgentType.NOISE_TRADER).value
            type_pnl.setdefault(atype, []).append(pnl)
        for atype, pnls in type_pnl.items():
            rows.append(
                {
                    "Agent Type": atype,
                    "Count": len(pnls),
                    "Total PnL ($)": round(sum(pnls), 2),
                    "Avg PnL ($)": round(float(np.mean(pnls)), 2),
                    "Win Rate": round(
                        sum(1 for p in pnls if p > 0) / max(len(pnls), 1), 2
                    ),
                }
            )
        return pd.DataFrame(rows).sort_values("Total PnL ($)", ascending=False)


# 
# Market Simulator
# 


class MarketSimulator:
    """
    Full agent-based market simulation engine.

    Parameters
    ----------
    n_agents : int
        Total number of agents (default 100)
    n_steps : int
        Number of simulation steps (default 500)
    fundamental_vol : float
        Annualised volatility of the fundamental value process (default 0.15)
    event_prob : float
        Per-step probability of a market event (default 0.01)
    seed : int | None
        Random seed for reproducibility
    agent_mix : dict | None
        Override agent type proportions (keys = AgentType, values = fractions summing to 1)
    """

    def __init__(
        self,
        n_agents: int = 100,
        n_steps: int = 500,
        fundamental_vol: float = 0.15,
        event_prob: float = 0.01,
        seed: int | None = None,
        agent_mix: dict | None = None,
    ):
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.fundamental_vol = fundamental_vol
        self.event_prob = event_prob
        self.rng = np.random.default_rng(seed)
        self.agent_mix = agent_mix

    #  Agent Factory 

    def _build_agents(self) -> list[Agent]:
        agents: list[Agent] = []
        agent_id = 0

        mix = self.agent_mix or {t: d["pct"] for t, d in AGENT_DEFAULTS.items()}
        # Normalise
        total = sum(mix.values())
        mix = {t: v / total for t, v in mix.items()}

        for atype, frac in mix.items():
            n = max(1, int(self.n_agents * frac))
            defaults = AGENT_DEFAULTS.get(atype, {})
            for _ in range(n):
                capital = max(
                    10_000,
                    float(
                        self.rng.normal(
                            defaults.get("capital_mean", 200_000),
                            defaults.get("capital_std", 50_000),
                        )
                    ),
                )
                if atype == AgentType.FUNDAMENTALIST:
                    agents.append(
                        FundamentalistAgent(agent_id, capital, self.rng, defaults)
                    )
                elif atype == AgentType.TREND_FOLLOWER:
                    agents.append(
                        TrendFollowerAgent(agent_id, capital, self.rng, defaults)
                    )
                elif atype == AgentType.MARKET_MAKER:
                    agents.append(
                        MarketMakerAgent(agent_id, capital, self.rng, defaults)
                    )
                elif atype == AgentType.NOISE_TRADER:
                    agents.append(
                        NoiseTraderAgent(agent_id, capital, self.rng, defaults)
                    )
                elif atype == AgentType.INSTITUTIONAL:
                    agents.append(
                        InstitutionalAgent(agent_id, capital, self.rng, defaults)
                    )
                elif atype == AgentType.HFT:
                    agents.append(HFTAgent(agent_id, capital, self.rng, defaults))
                agent_id += 1

        return agents

    #  Fundamental Value Process 

    def _simulate_fundamental(self, initial: float) -> list[float]:
        """Geometric Brownian Motion for the fundamental (fair) value."""
        dt = 1 / 252
        mu = 0.05  # long-run drift
        sigma = self.fundamental_vol
        path = [initial]
        for _ in range(self.n_steps):
            shock = float(self.rng.normal(0, 1))
            next_v = path[-1] * math.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * shock
            )
            path.append(max(0.01, next_v))
        return path

    #  Event Engine 

    def _sample_event(
        self, step: int, fundamental: list[float]
    ) -> tuple[EventType, str, float]:
        """Returns (event_type, description, fundamental_shock_multiplier)."""
        if self.rng.random() > self.event_prob:
            return EventType.NONE, "", 1.0

        events = [
            (EventType.EARNINGS_BEAT, "Earnings beat — EPS +15% above consensus", 1.06),
            (EventType.EARNINGS_MISS, "Earnings miss — EPS -12% below consensus", 0.94),
            (EventType.MACRO_SHOCK, "Fed hawkish surprise — rates +50bps", 0.96),
            (
                EventType.LIQUIDITY_SHOCK,
                "Liquidity shock — market-wide margin calls",
                0.90,
            ),
            (EventType.LARGE_BUYER, "Large institutional block buy detected", 1.02),
            (EventType.LARGE_SELLER, "Large institutional block sell detected", 0.98),
            (EventType.VIX_SPIKE, "VIX spike — implied vol surges +40%", 0.93),
        ]
        idx = int(self.rng.integers(0, len(events)))
        etype, desc, mult = events[idx]
        return etype, desc, mult

    #  Realised Volatility 

    def _realized_vol(self, price_history: list[float], window: int = 20) -> float:
        if len(price_history) < window + 1:
            return 15.0
        recent = price_history[-(window + 1) :]
        log_rets = [
            math.log(recent[i] / max(recent[i - 1], 0.01))
            for i in range(1, len(recent))
        ]
        if not log_rets:
            return 15.0
        vol = float(np.std(log_rets)) * math.sqrt(252) * 100
        return round(vol, 2)

    #  Main Run 

    def run(self, initial_price: float = 100.0) -> SimulationResult:
        """
        Execute the full market simulation.

        Returns
        -------
        SimulationResult with full snapshot history and agent analytics.
        """
        agents = self._build_agents()
        book = OrderBook()
        fundamental_path = self._simulate_fundamental(initial_price)

        price_history: list[float] = [initial_price]
        snapshots: list[MarketSnapshot] = []
        events_log: list[tuple[int, EventType, str]] = []

        # Seed the order book with initial quotes
        mid = initial_price
        for agent in agents:
            agent.last_price = mid
        book.submit_limit_order(Side.BID, round(mid * 0.999, 2), 500, -1, 0)
        book.submit_limit_order(Side.ASK, round(mid * 1.001, 2), 500, -2, 0)

        for step in range(1, self.n_steps + 1):
            book.reset_step_stats()
            fundamental_value = fundamental_path[min(step, len(fundamental_path) - 1)]
            current_mid = book.mid_price or price_history[-1]

            # Market event
            event_type, event_desc, fv_mult = self._sample_event(step, fundamental_path)
            if event_type != EventType.NONE:
                # Directly shock the fundamental for this step
                fundamental_value *= fv_mult
                events_log.append((step, event_type, event_desc))

                # Inject large order on event
                shock_side = Side.BID if fv_mult > 1 else Side.ASK
                shock_qty = int(self.rng.integers(200, 600))
                shock_price = (
                    round(current_mid * 1.005, 2)
                    if shock_side == Side.BID
                    else round(current_mid * 0.995, 2)
                )
                book.submit_limit_order(shock_side, shock_price, shock_qty, -99, step)

            # Shuffle agents to prevent ordering bias
            agent_order = list(range(len(agents)))
            self.rng.shuffle(agent_order)

            market_vol = self._realized_vol(price_history)

            for idx in agent_order:
                agent = agents[idx]
                try:
                    orders = agent.decide(
                        step, book, price_history, fundamental_value, market_vol
                    )
                    for side, price, qty, otype in orders:
                        if otype == OrderType.LIMIT:
                            trades = book.submit_limit_order(
                                side, price, qty, agent.agent_id, step
                            )
                        else:
                            trades = book.submit_market_order(
                                side, qty, agent.agent_id, step
                            )

                        # Update agent position and PnL
                        for t in trades:
                            if t.buyer_id == agent.agent_id:
                                agent.position += t.quantity
                                agent.capital -= t.price * t.quantity
                            elif t.seller_id == agent.agent_id:
                                agent.position -= t.quantity
                                agent.capital += t.price * t.quantity
                            agent.trades_done += 1
                except Exception:
                    pass

            # Stale order cleanup every 10 steps
            if step % 10 == 0:
                current_mid_now = book.mid_price or price_history[-1]
                book.cancel_stale_orders(current_mid_now, tolerance_pct=0.08)
                # Re-seed if book is empty
                if book.best_bid is None or book.best_ask is None:
                    p = price_history[-1]
                    book.submit_limit_order(
                        Side.BID, round(p * 0.998, 2), 200, -1, step
                    )
                    book.submit_limit_order(
                        Side.ASK, round(p * 1.002, 2), 200, -2, step
                    )

            # Record snapshot
            mid_now = book.mid_price or price_history[-1]
            bid_now = book.best_bid or mid_now * 0.999
            ask_now = book.best_ask or mid_now * 1.001
            spread_now = ask_now - bid_now

            step_trades = book.get_step_trades()
            step_volume = book.total_buy_volume + book.total_sell_volume
            buy_vol = book.total_buy_volume
            sell_vol = book.total_sell_volume
            ofi = (buy_vol - sell_vol) / max(buy_vol + sell_vol, 1)

            snapshot = MarketSnapshot(
                step=step,
                mid_price=round(mid_now, 4),
                bid_price=round(bid_now, 4),
                ask_price=round(ask_now, 4),
                spread=round(spread_now, 4),
                volume=step_volume,
                order_flow_imbalance=round(ofi, 4),
                realized_vol_20=self._realized_vol(price_history),
                depth_bid=book.bid_depth,
                depth_ask=book.ask_depth,
                trades=step_trades,
                event=event_type,
                fundamental_value=round(fundamental_value, 4),
            )
            snapshots.append(snapshot)
            price_history.append(mid_now)

        # Compute final agent PnLs (mark to market at last price)
        final_price = price_history[-1]
        agent_pnls = {}
        agent_types_map = {}
        for agent in agents:
            mtm_pnl = agent.capital + agent.position * final_price - agent.capital
            # Simplified: track position value change
            agent_pnls[agent.agent_id] = round(agent.position * final_price, 2)
            agent_types_map[agent.agent_id] = agent.agent_type

        total_volume = sum(s.volume for s in snapshots)
        total_trades = sum(len(s.trades) for s in snapshots)

        return SimulationResult(
            snapshots=snapshots,
            agent_pnls=agent_pnls,
            agent_types=agent_types_map,
            events=events_log,
            final_price=round(final_price, 4),
            total_volume=total_volume,
            n_trades=total_trades,
            params={
                "n_agents": self.n_agents,
                "n_steps": self.n_steps,
                "initial_price": initial_price,
                "fundamental_vol": self.fundamental_vol,
                "event_prob": self.event_prob,
            },
        )


# 
# Preset Scenarios
# 

SIMULATION_PRESETS: dict[str, dict] = {
    "Normal Market": {
        "n_agents": 100,
        "n_steps": 500,
        "fundamental_vol": 0.15,
        "event_prob": 0.008,
        "description": "Typical equity market with balanced agent population.",
    },
    "High Frequency Battle": {
        "n_agents": 150,
        "n_steps": 300,
        "fundamental_vol": 0.12,
        "event_prob": 0.005,
        "agent_mix": {
            AgentType.HFT: 0.40,
            AgentType.MARKET_MAKER: 0.20,
            AgentType.INSTITUTIONAL: 0.10,
            AgentType.NOISE_TRADER: 0.15,
            AgentType.FUNDAMENTALIST: 0.10,
            AgentType.TREND_FOLLOWER: 0.05,
        },
        "description": "HFT-dominated market with tight spreads and rapid mean-reversion.",
    },
    "Liquidity Crisis": {
        "n_agents": 80,
        "n_steps": 400,
        "fundamental_vol": 0.45,
        "event_prob": 0.04,
        "agent_mix": {
            AgentType.NOISE_TRADER: 0.40,
            AgentType.TREND_FOLLOWER: 0.30,
            AgentType.FUNDAMENTALIST: 0.10,
            AgentType.INSTITUTIONAL: 0.10,
            AgentType.MARKET_MAKER: 0.05,
            AgentType.HFT: 0.05,
        },
        "description": "Crisis scenario: high volatility, frequent shocks, market makers retreat.",
    },
    "Momentum Mania": {
        "n_agents": 120,
        "n_steps": 600,
        "fundamental_vol": 0.20,
        "event_prob": 0.015,
        "agent_mix": {
            AgentType.TREND_FOLLOWER: 0.50,
            AgentType.NOISE_TRADER: 0.20,
            AgentType.INSTITUTIONAL: 0.10,
            AgentType.MARKET_MAKER: 0.10,
            AgentType.FUNDAMENTALIST: 0.05,
            AgentType.HFT: 0.05,
        },
        "description": "Trend-follower-dominated market producing momentum bubbles and crashes.",
    },
    "Efficient Market": {
        "n_agents": 100,
        "n_steps": 500,
        "fundamental_vol": 0.12,
        "event_prob": 0.005,
        "agent_mix": {
            AgentType.FUNDAMENTALIST: 0.40,
            AgentType.MARKET_MAKER: 0.25,
            AgentType.HFT: 0.15,
            AgentType.INSTITUTIONAL: 0.10,
            AgentType.NOISE_TRADER: 0.05,
            AgentType.TREND_FOLLOWER: 0.05,
        },
        "description": "Fundamentalist-dominated market: prices track fair value closely.",
    },
}


def get_simulator(
    preset_name: str = "Normal Market", seed: int | None = 42
) -> MarketSimulator:
    """Create a MarketSimulator from a named preset."""
    preset = SIMULATION_PRESETS.get(preset_name, SIMULATION_PRESETS["Normal Market"])
    kwargs = {
        "n_agents": preset.get("n_agents", 100),
        "n_steps": preset.get("n_steps", 500),
        "fundamental_vol": preset.get("fundamental_vol", 0.15),
        "event_prob": preset.get("event_prob", 0.01),
        "seed": seed,
    }
    if "agent_mix" in preset:
        kwargs["agent_mix"] = preset["agent_mix"]
    return MarketSimulator(**kwargs)
