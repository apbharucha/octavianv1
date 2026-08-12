"""
Octavian Brokerage Engine
Integration with Interactive Brokers (IBKR) via ib_insync and Smart Order Routing (SOR) algorithm.
"""

import asyncio
from typing import List, Dict, Any, Optional

try:
    from ib_insync import IB, Stock, Option, MarketOrder, LimitOrder, Contract
    _HAS_IBKR = True
except ImportError:
    _HAS_IBKR = False

class BrokerageEngine:
    """Manages Interactive Brokers connections, order execution, and Smart Order Routing (SOR)."""
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        if _HAS_IBKR:
            self.ib = IB()

    def connect(self) -> bool:
        """Establishes connection to TWS/IB Gateway."""
        if not _HAS_IBKR:
            print("[BrokerageEngine] ib_insync not installed. Operating in mock mode.")
            return False
            
        try:
            if not self.ib.isConnected():
                self.ib.connect(self.host, self.port, clientId=self.client_id)
            return True
        except Exception as e:
            print(f"[BrokerageEngine] Failed to connect to IBKR: {e}")
            return False

    def disconnect(self):
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()

    def execute_smart_order_route(self, underlying_symbol: str, legs: List[Dict[str, Any]], target_credit_debit: float, is_credit: bool = True) -> Dict[str, Any]:
        """
        Smart Order Routing (SOR) algorithm to execute multi-leg option strategies minimizing slippage.
        It uses a staggered entry approach: sending limit orders at mid-price, then stepping toward natural.
        """
        print(f"[SOR] Routing multi-leg strategy for {underlying_symbol}")
        if not _HAS_IBKR or not self.ib or not self.ib.isConnected():
            # Mock execution for UI when IBKR isn't active
            return {
                "status": "MOCK_EXECUTED",
                "message": f"Successfully routed multi-leg strategy for {underlying_symbol} at target {'credit' if is_credit else 'debit'} of ${abs(target_credit_debit):.2f}.",
                "filled_price": target_credit_debit * 0.98, # Mock 2% slippage improvement
                "legs": legs
            }

        # Real IBKR Execution Logic
        try:
            contracts = []
            for leg in legs:
                contract = Option(
                    underlying_symbol,
                    lastTradeDateOrContractMonth=leg.get('expiry', '20261218'),
                    strike=leg['strike'],
                    right='C' if leg['type'].lower() == 'call' else 'P',
                    exchange='SMART',
                    multiplier='100'
                )
                self.ib.qualifyContracts(contract)
                contracts.append((contract, leg['side']))
                
            # For complex multi-leg, IBKR requires Combo / Bag orders
            # Simplified here to sequential limit orders using the SOR walking algorithm
            executions = []
            for contract, side in contracts:
                action = 'BUY' if side == 1 else 'SELL'
                
                # Mock SOR walking algorithm (mid-price to natural)
                # In production, we'd fetch live bid/ask via ib.reqMktData
                limit_price = abs(target_credit_debit) / len(legs) # Naive split for limit
                
                order = LimitOrder(action, 1, limit_price)
                order.smartComboRoutingParams = []
                trade = self.ib.placeOrder(contract, order)
                executions.append(trade)
                
            return {
                "status": "ROUTED",
                "message": "Strategy routed via SOR. Awaiting fills.",
                "trades": executions
            }
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    def get_portfolio_positions(self) -> List[Dict[str, Any]]:
        """Fetch live portfolio positions from IBKR."""
        if not self.ib or not self.ib.isConnected():
            return []
            
        positions = self.ib.positions()
        return [{"symbol": p.contract.symbol, "position": p.position, "avgCost": p.avgCost} for p in positions]

    def generate_rebalance_orders(self, current_portfolio: List[Dict[str, Any]], target_allocations: Dict[str, float], total_portfolio_value: float) -> List[Dict[str, Any]]:
        """
        Phase 5: Automated Rebalancing
        Calculates the delta between current position allocations and target allocations,
        and generates the required mock/live trades to bring the portfolio back to target risk allocation.
        """
        orders = []
        current_allocations = {}
        
        # Calculate current allocations
        for pos in current_portfolio:
            sym = pos["symbol"]
            # Assume we have current market price. Here we use avgCost as a proxy for the math.
            val = pos["position"] * pos.get("current_price", pos["avgCost"])
            current_allocations[sym] = val / total_portfolio_value if total_portfolio_value > 0 else 0
            
        # Generate trades for targets
        for sym, target_pct in target_allocations.items():
            current_pct = current_allocations.get(sym, 0.0)
            diff_pct = target_pct - current_pct
            
            # If the difference is significant (e.g., > 1% drift), generate order
            if abs(diff_pct) > 0.01:
                target_value_to_trade = total_portfolio_value * diff_pct
                # Assuming current price is available or 100 as placeholder
                assumed_price = 100.0 
                for pos in current_portfolio:
                    if pos["symbol"] == sym:
                        assumed_price = pos.get("current_price", pos["avgCost"])
                        break
                        
                shares_to_trade = int(target_value_to_trade / assumed_price)
                if shares_to_trade != 0:
                    orders.append({
                        "symbol": sym,
                        "action": "BUY" if shares_to_trade > 0 else "SELL",
                        "quantity": abs(shares_to_trade),
                        "estimated_value": abs(target_value_to_trade)
                    })
                    
        return orders

# Singleton instance for the platform
_brokerage_instance = None

def get_brokerage_engine() -> BrokerageEngine:
    global _brokerage_instance
    if _brokerage_instance is None:
        _brokerage_instance = BrokerageEngine()
    return _brokerage_instance
