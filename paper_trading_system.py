"""
Paper Trading System with Multiple Accounts

Provides comprehensive paper trading functionality:
1. Multiple paper trading accounts per user
2. Position tracking and P&L calculation
3. Trade logging with full AI reasoning
4. Automated strategy mode with AI-driven trades
5. Real-time market data integration

Author: APB - Octavian Team
"""

import sqlite3
import pandas as pd
import numpy as np
import uuid # Import uuid for generating unique IDs
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from contextlib import contextmanager

from database_manager import get_database_manager
from data_sources import get_stock, get_fx, get_futures_proxy

logger = logging.getLogger(__name__)


class TradeAction(Enum):
    """Trade action types."""
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


class AccountStatus(Enum):
    """Account status types."""
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    CLOSED = "CLOSED"


@dataclass
class PaperTradingAccount:
    """Paper trading account data structure."""
    account_id: str
    user_id: str
    account_name: str
    initial_balance: float
    current_balance: float
    total_pnl: float
    total_pnl_pct: float
    realized_pnl: float
    status: AccountStatus
    automation_enabled: bool
    automation_config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class Position:
    """Trading position data structure."""
    position_id: str
    account_id: str
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_time: datetime
    updated_at: datetime


@dataclass
class OptionPosition:
    """Option contract position data structure."""
    position_id: str
    account_id: str
    symbol: str  # Contract symbol e.g. 'AAPL260515C00150000'
    underlying_symbol: str
    option_type: str  # 'CALL' or 'PUT'
    strike: float
    expiration: str
    side: str  # 'LONG' or 'SHORT'
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime
    updated_at: datetime


@dataclass
class Trade:
    """Trade execution data structure."""
    trade_id: str
    account_id: str
    symbol: str
    action: TradeAction
    quantity: float
    price: float
    total_value: float
    strategy_name: str
    ai_reasoning: str
    market_context: Dict[str, Any]
    timestamp: datetime


class PaperTradingSystem:
    """Comprehensive paper trading system with multiple accounts."""

    # Legacy user_id aliases → canonical user_id
    _USER_ID_ALIASES: Dict[str, str] = {
        "trader_1": "default_user",
    }

    def _normalize_user_id(self, user_id: str) -> str:
        """Normalize a user_id: strip whitespace, lower-case lookup for aliases.

        Maps legacy IDs (e.g. ``"trader_1"``) to their canonical equivalents
        (e.g. ``"default_user"``) so that old data and new data are always
        resolved to the same account bucket.

        ``user_id`` may arrive as an ``int`` (e.g. the numeric row id stored in
        ``st.session_state`` by ``auth_engine``), so it is coerced to ``str``
        before any string operations. ``None`` is passed through unchanged.
        """
        if user_id is None:
            return user_id
        if not isinstance(user_id, str):
            user_id = str(user_id)
        stripped = user_id.strip()
        return self._USER_ID_ALIASES.get(stripped.lower(), stripped)

    def __init__(self, db_manager=None):
        # We need a re-entrant lock since Streamlit re-runs the script
        # and _get_database_manager might be called multiple times.
        self.lock = threading.RLock()
        self.db_manager = db_manager if db_manager else get_database_manager()
        self._initialize_tables()
    
    def _initialize_tables(self):
        """Initialize paper trading database tables."""
        with self.db_manager._get_connection() as conn:
            cursor = conn.cursor()
            
            # Paper trading accounts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS paper_trading_accounts (
                    account_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    account_name TEXT NOT NULL,
                    initial_balance REAL NOT NULL,
                    current_balance REAL NOT NULL,
                    total_pnl REAL DEFAULT 0,
                    total_pnl_pct REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    status TEXT DEFAULT 'ACTIVE',
                    automation_enabled BOOLEAN DEFAULT 0,
                    automation_config TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Positions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS paper_trading_positions (
                    position_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL,
                    unrealized_pnl REAL DEFAULT 0,
                    unrealized_pnl_pct REAL DEFAULT 0,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (account_id) REFERENCES paper_trading_accounts (account_id)
                )
            """)
            
            # Trades log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS paper_trading_trades (
                    trade_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    total_value REAL NOT NULL,
                    strategy_name TEXT,
                    ai_reasoning TEXT,
                    market_context TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (account_id) REFERENCES paper_trading_accounts (account_id)
                )
            """)
            
            # Options Positions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS paper_trading_options_positions (
                    position_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    underlying_symbol TEXT NOT NULL,
                    option_type TEXT NOT NULL, -- 'CALL' or 'PUT'
                    strike REAL NOT NULL,
                    expiration DATE NOT NULL,
                    side TEXT NOT NULL, -- 'LONG' or 'SHORT'
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_market_price REAL,
                    unrealized_pnl REAL DEFAULT 0,
                    contract_symbol TEXT,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (account_id) REFERENCES paper_trading_accounts (account_id)
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_account 
                ON paper_trading_positions(account_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_account 
                ON paper_trading_trades(account_id, timestamp)
            """)
            
            conn.commit()
            logger.info("Paper trading tables initialized")
    
    def create_account(self, user_id: str, account_name: str,
                      initial_balance: float = 100000.0) -> Optional[PaperTradingAccount]:
        """Create a new paper trading account."""
        try:
            user_id = self._normalize_user_id(user_id)
            with self.lock:
                account_id = f"PT_{user_id}_{uuid.uuid4()}"
                
                with self.db_manager._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO paper_trading_accounts 
                        (account_id, user_id, account_name, initial_balance, current_balance)
                        VALUES (?, ?, ?, ?, ?)
                    """, (account_id, user_id, account_name, initial_balance, initial_balance))
                    conn.commit()
                
                logger.info(f"Created paper trading account {account_id}")
                return self.get_account(account_id)
        
        except Exception as e:
            logger.error(f"Error creating paper trading account: {e}")
            return None
    
    def get_account(self, account_id: str) -> Optional[PaperTradingAccount]:
        """Get paper trading account details."""
        try:
            with self.db_manager._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM paper_trading_accounts WHERE account_id = ?
                """, (account_id,))
                
                row = cursor.fetchone()
                if row:
                    return PaperTradingAccount(
                        account_id=row['account_id'],
                        user_id=row['user_id'],
                        account_name=row['account_name'],
                        initial_balance=row['initial_balance'],
                        current_balance=row['current_balance'],
                        total_pnl=row['total_pnl'],
                        total_pnl_pct=row['total_pnl_pct'],
                        realized_pnl=row['realized_pnl'],
                        status=AccountStatus(row['status']),
                        automation_enabled=bool(row['automation_enabled']),
                        automation_config=json.loads(row['automation_config'] or '{}'),
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at'])
                    )
                return None
        
        except Exception as e:
            logger.error(f"Error getting account {account_id}: {e}")
            return None
    
    def list_accounts(self, user_id: str) -> List[PaperTradingAccount]:
        """List all paper trading accounts for a user."""
        try:
            user_id = self._normalize_user_id(user_id)
            with self.db_manager._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM paper_trading_accounts 
                    WHERE LOWER(user_id) = LOWER(?)
                    ORDER BY created_at DESC
                """, (user_id,))
                
                accounts = []
                for row in cursor.fetchall():
                    accounts.append(PaperTradingAccount(
                        account_id=row['account_id'],
                        user_id=row['user_id'],
                        account_name=row['account_name'],
                        initial_balance=row['initial_balance'],
                        current_balance=row['current_balance'],
                        total_pnl=row['total_pnl'],
                        total_pnl_pct=row['total_pnl_pct'],
                        realized_pnl=row['realized_pnl'],
                        status=AccountStatus(row['status']),
                        automation_enabled=bool(row['automation_enabled']),
                        automation_config=json.loads(row['automation_config'] or '{}'),
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at'])
                    ))
                
                return accounts
        
        except Exception as e:
            logger.error(f"Error listing accounts for user {user_id}: {e}")
            return []
    def delete_account(self, account_id: str) -> bool:
        """Delete a paper trading account and all associated data."""
        try:
            with self.lock:
                with self.db_manager._get_connection() as conn:
                    cursor = conn.cursor()

                    # Delete trades
                    cursor.execute("""
                        DELETE FROM paper_trading_trades WHERE account_id = ?
                    """, (account_id,))

                    # Delete positions
                    cursor.execute("""
                        DELETE FROM paper_trading_positions WHERE account_id = ?
                    """, (account_id,))

                    # Delete account
                    cursor.execute("""
                        DELETE FROM paper_trading_accounts WHERE account_id = ?
                    """, (account_id,))

                    conn.commit()

                    logger.info(f"Deleted paper trading account {account_id}")
                    return True

        except Exception as e:
            logger.error(f"Error deleting account {account_id}: {e}")
            return False
    
    def execute_trade(self, account_id: str, symbol: str, action: TradeAction,
                     quantity: float, price: Optional[float] = None,
                     strategy_name: str = "Manual", ai_reasoning: str = "",
                     market_context: Optional[Dict] = None) -> Optional[Trade]:
        """Execute a paper trade."""
        try:
            with self.lock:
                # Get account
                account = self.get_account(account_id)
                if not account or account.status != AccountStatus.ACTIVE:
                    logger.error(f"Account {account_id} not active")
                    return None
                
                # Get current price if not provided
                if price is None:
                    price = self._get_current_price(symbol)
                    if price is None:
                        logger.error(f"Could not get price for {symbol}")
                        return None
                
                total_value = quantity * price
                
                # Check if account has sufficient balance for BUY/SHORT
                if action in [TradeAction.BUY, TradeAction.SHORT]:
                    if total_value > account.current_balance:
                        logger.error(f"Insufficient balance for trade")
                        return None
                
                # Create trade record
                trade_id = f"TR_{account_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                trade = Trade(
                    trade_id=trade_id,
                    account_id=account_id,
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    price=price,
                    total_value=total_value,
                    strategy_name=strategy_name,
                    ai_reasoning=ai_reasoning,
                    market_context=market_context or {},
                    timestamp=datetime.now()
                )
                
                # Store trade
                with self.db_manager._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO paper_trading_trades
                        (trade_id, account_id, symbol, action, quantity, price, 
                         total_value, strategy_name, ai_reasoning, market_context)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trade.trade_id, trade.account_id, trade.symbol,
                        trade.action.value, trade.quantity, trade.price,
                        trade.total_value, trade.strategy_name, trade.ai_reasoning,
                        json.dumps(trade.market_context)
                    ))
                    conn.commit()
                
                # Update positions and balance
                pnl = self._update_position(account_id, symbol, action, quantity, price)
                self._update_account_balance(account_id, action, total_value, pnl)
                
                logger.info(f"Executed trade {trade_id}: {action.value} {quantity} {symbol} @ {price}")
                return trade
        
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol."""
        try:
            # Determine asset type and fetch data
            if '/' in symbol or 'USD' in symbol and '=' not in symbol:
                df = get_fx(symbol)
            elif '=F' in symbol:
                df = get_futures_proxy(symbol)
            else:
                df = get_stock(symbol)
            
            if df is not None and not df.empty:
                close_col = df['Close']
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                return float(close_col.iloc[-1])
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def _update_position(self, account_id: str, symbol: str, action: TradeAction,
                        quantity: float, price: float) -> float:
        """Update position after trade execution and return realized P&L."""
        realized_pnl = 0.0
        try:
            with self.db_manager._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get existing position — side-aware so a long and a short in the
                # same symbol are tracked independently and never collide.
                target_side = 'LONG' if action in (TradeAction.BUY, TradeAction.SELL) else 'SHORT'
                cursor.execute("""
                    SELECT * FROM paper_trading_positions
                    WHERE account_id = ? AND symbol = ? AND side = ?
                """, (account_id, symbol, target_side))
                
                existing = cursor.fetchone()
                
                if action == TradeAction.BUY:
                    if existing:
                        # Add to existing long position
                        new_quantity = existing['quantity'] + quantity
                        new_avg_price = ((existing['quantity'] * existing['entry_price']) + 
                                       (quantity * price)) / new_quantity
                        
                        cursor.execute("""
                            UPDATE paper_trading_positions
                            SET quantity = ?, entry_price = ?, updated_at = ?
                            WHERE account_id = ? AND symbol = ? AND side = ?
                        """, (new_quantity, new_avg_price, datetime.now(), account_id, symbol, target_side))
                    else:
                        # Create new long position
                        position_id = f"POS_{account_id}_{symbol}_{uuid.uuid4()}"
                        cursor.execute("""
                            INSERT INTO paper_trading_positions
                            (position_id, account_id, symbol, side, quantity, entry_price, current_price)
                            VALUES (?, ?, ?, 'LONG', ?, ?, ?)
                        """, (position_id, account_id, symbol, quantity, price, price))
                
                elif action == TradeAction.SELL:
                    if existing:
                        new_quantity = existing['quantity'] - quantity
                        realized_pnl = (price - existing['entry_price']) * quantity
                        if new_quantity <= 0:
                            # Close position
                            cursor.execute("""
                                DELETE FROM paper_trading_positions
                                WHERE account_id = ? AND symbol = ? AND side = ?
                            """, (account_id, symbol, target_side))
                        else:
                            # Reduce position
                            cursor.execute("""
                                UPDATE paper_trading_positions
                                SET quantity = ?, updated_at = ?
                                WHERE account_id = ? AND symbol = ? AND side = ?
                            """, (new_quantity, datetime.now(), account_id, symbol, target_side))
                
                elif action == TradeAction.SHORT:
                    if existing:
                        # Add to existing short position
                        new_quantity = existing['quantity'] + quantity
                        new_avg_price = ((existing['quantity'] * existing['entry_price']) + 
                                       (quantity * price)) / new_quantity
                        
                        cursor.execute("""
                            UPDATE paper_trading_positions
                            SET quantity = ?, entry_price = ?, updated_at = ?
                            WHERE account_id = ? AND symbol = ? AND side = ?
                        """, (new_quantity, new_avg_price, datetime.now(), account_id, symbol, target_side))
                    else:
                        # Create new short position
                        position_id = f"POS_{account_id}_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        cursor.execute("""
                            INSERT INTO paper_trading_positions
                            (position_id, account_id, symbol, side, quantity, entry_price, current_price)
                            VALUES (?, ?, ?, 'SHORT', ?, ?, ?)
                        """, (position_id, account_id, symbol, quantity, price, price))
                
                elif action == TradeAction.COVER:
                    if existing:
                        new_quantity = existing['quantity'] - quantity
                        realized_pnl = (existing['entry_price'] - price) * quantity
                        if new_quantity <= 0:
                            # Close position
                            cursor.execute("""
                                DELETE FROM paper_trading_positions
                                WHERE account_id = ? AND symbol = ? AND side = ?
                            """, (account_id, symbol, target_side))
                        else:
                            # Reduce position
                            cursor.execute("""
                                UPDATE paper_trading_positions
                                SET quantity = ?, updated_at = ?
                                WHERE account_id = ? AND symbol = ? AND side = ?
                            """, (new_quantity, datetime.now(), account_id, symbol, target_side))
                
                conn.commit()
                return realized_pnl
        
        except Exception as e:
            logger.error(f"Error updating position: {e}")
            return 0.0
    
    def _update_account_balance(self, account_id: str, action: TradeAction, 
                               total_value: float, incoming_pnl: float = 0.0):
        """Update account balance and P&L metrics after trade."""
        try:
            with self.db_manager._get_connection() as conn:
                cursor = conn.cursor()
                
                if action in [TradeAction.BUY, TradeAction.SHORT]:
                    # Deduct from balance
                    cursor.execute("""
                        UPDATE paper_trading_accounts
                        SET current_balance = current_balance - ?,
                            updated_at = ?
                        WHERE account_id = ?
                    """, (total_value, datetime.now(), account_id))
                
                elif action in [TradeAction.SELL, TradeAction.COVER]:
                    # Add to balance
                    cursor.execute("""
                        UPDATE paper_trading_accounts
                        SET current_balance = current_balance + ?,
                            updated_at = ?
                        WHERE account_id = ?
                    """, (total_value, datetime.now(), account_id))
                
                # Update total P&L and P&L % for the account record
                cursor.execute("""
                    SELECT initial_balance, current_balance 
                    FROM paper_trading_accounts 
                    WHERE account_id = ?
                """, (account_id,))
                
                row = cursor.fetchone()
                if row:
                    initial = row['initial_balance']
                    current = row['current_balance']
                    cursor.execute("""
                        UPDATE paper_trading_accounts
                        SET realized_pnl = realized_pnl + ?,
                            updated_at = ?
                        WHERE account_id = ?
                    """, (incoming_pnl, datetime.now(), account_id))
                    
                    # Fetch updated realized_pnl for accuracy stats
                    cursor.execute("SELECT realized_pnl FROM paper_trading_accounts WHERE account_id = ?", (account_id,))
                    updated_realized = cursor.fetchone()['realized_pnl']
                    
                    total_pnl = updated_realized
                    total_pnl_pct = (total_pnl / initial * 100) if initial > 0 else 0
                    
                    cursor.execute("""
                        UPDATE paper_trading_accounts
                        SET total_pnl = ?, total_pnl_pct = ?
                        WHERE account_id = ?
                    """, (total_pnl, total_pnl_pct, account_id))
                
                conn.commit()
        
        except Exception as e:
            logger.error(f"Error updating account balance: {e}")
    
    def get_positions(self, account_id: str) -> List[Position]:
        """Get all open positions for an account."""
        try:
            with self.db_manager._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM paper_trading_positions
                    WHERE account_id = ?
                    ORDER BY entry_time DESC
                """, (account_id,))
                
                positions = []
                for row in cursor.fetchall():
                    # Update current price and P&L
                    current_price = self._get_current_price(row['symbol'])
                    if current_price:
                        entry_price = row['entry_price']
                        quantity = row['quantity']
                        
                        if row['side'] == 'LONG':
                            unrealized_pnl = (current_price - entry_price) * quantity
                        else:  # SHORT
                            unrealized_pnl = (entry_price - current_price) * quantity
                        
                        unrealized_pnl_pct = (unrealized_pnl / (entry_price * quantity)) * 100
                        
                        positions.append(Position(
                            position_id=row['position_id'],
                            account_id=row['account_id'],
                            symbol=row['symbol'],
                            side=row['side'],
                            quantity=row['quantity'],
                            entry_price=entry_price,
                            current_price=current_price,
                            unrealized_pnl=unrealized_pnl,
                            unrealized_pnl_pct=unrealized_pnl_pct,
                            entry_time=datetime.fromisoformat(row['entry_time']),
                            updated_at=datetime.now()
                        ))
                
                return positions
        
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_trade_history(self, account_id: str, limit: int = 100) -> List[Trade]:
        """Get trade history for an account."""
        try:
            with self.db_manager._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM paper_trading_trades
                    WHERE account_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (account_id, limit))
                
                trades = []
                for row in cursor.fetchall():
                    trades.append(Trade(
                        trade_id=row['trade_id'],
                        account_id=row['account_id'],
                        symbol=row['symbol'],
                        action=TradeAction(row['action']),
                        quantity=row['quantity'],
                        price=row['price'],
                        total_value=row['total_value'],
                        strategy_name=row['strategy_name'],
                        ai_reasoning=row['ai_reasoning'],
                        market_context=json.loads(row['market_context'] or '{}'),
                        timestamp=datetime.fromisoformat(row['timestamp'])
                    ))
                
                return trades
        
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []
            
    def execute_option_trade(self, account_id: str, underlying_symbol: str, 
                             contract_symbol: str = None, action=None, quantity: float = 0.0, 
                             price: float = 0.0, option_type: str = None, strike: float = None, 
                             expiration: str = None, strategy_name: str = "Manual",
                             ai_reasoning: str = "") -> Optional[Trade]:
        """Execute an options trade (BUY/SELL CALL/PUT).

        Robust to both full contract metadata and minimal callers:
        - ``action`` may be a :class:`TradeAction` enum or a raw string ('BUY'/'SELL').
        - When ``contract_symbol`` / ``option_type`` / ``strike`` / ``expiration`` are
          omitted they are derived from the underlying symbol so legacy callers that
          only pass ``(account_id, symbol, action, qty, price)`` keep working.
        """
        try:
            if isinstance(action, TradeAction):
                action = action.value
            action = str(action).upper()
            # COVER (closing a short option) is equivalent to buying the
            # contract back; normalize it so legacy callers that close short
            # positions with COVER keep working.
            if action == 'COVER':
                action = 'BUY'
            if action not in ('BUY', 'SELL'):
                raise ValueError(f"Option action must be BUY, SELL or COVER, got {action!r}")

            underlying = (underlying_symbol or '').strip().upper()
            contract_symbol = (contract_symbol or underlying or 'UNKNOWN').strip().upper()
            opt_type = str(option_type or 'CALL').upper()
            if opt_type not in ('CALL', 'PUT'):
                opt_type = 'CALL'
            strike = float(strike) if strike is not None else 0.0
            expiration = str(expiration or '') 
            quantity = float(quantity)
            price = float(price)

            total_value = quantity * price * 100  # Options multiplier (100 shares/contract)
            side = 'LONG' if action == 'BUY' else 'SHORT'
            
            trade_id = f"TR_OPT_{account_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            
            with self.lock:
                with self.db_manager._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO paper_trading_trades 
                        (trade_id, account_id, symbol, action, quantity, price, total_value, strategy_name, ai_reasoning)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (trade_id, account_id, contract_symbol, action, quantity, price, total_value, strategy_name, ai_reasoning))
                    conn.commit()
                
                # Update options positions and balance
                self._update_option_position(account_id, underlying, contract_symbol, 
                                            action, quantity, price, opt_type, strike, expiration)
                
                # Balance impact for options (Buy = Cash out, Sell = Cash in)
                trade_action = TradeAction.BUY if action == 'BUY' else TradeAction.SELL
                self._update_account_balance(account_id, trade_action, total_value)
                
                logger.info(f"Executed option trade {trade_id}: {action} {quantity} {contract_symbol} @ {price}")
                # Fetch as standard Trade object for consistency
                return Trade(trade_id, account_id, contract_symbol, trade_action, quantity, price, 
                             total_value, strategy_name, ai_reasoning, {}, datetime.now())
                                            
        except Exception as e:
            logger.error(f"Error executing option trade: {e}")
            return None

    def _update_option_position(self, account_id: str, underlying: str, contract: str, 
                               action: str, quantity: float, price: float, 
                               opt_type: str, strike: float, expiration: str):
        """Update options position in DB."""
        try:
            with self.db_manager._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM paper_trading_options_positions WHERE account_id = ? AND symbol = ?", 
                               (account_id, contract))
                existing = cursor.fetchone()
                
                side = 'LONG' if action == 'BUY' else 'SHORT'
                
                if existing:
                    # Side-aware position math:
                    #   LONG  position: BUY adds, SELL reduces (close)
                    #   SHORT position: SELL adds, BUY reduces (cover)
                    try:
                        existing_side = str(existing['side']).upper()
                    except (KeyError, IndexError, TypeError):
                        existing_side = 'LONG'
                    action_side = 'LONG' if action == 'BUY' else 'SHORT'
                    if existing_side == action_side:
                        new_quantity = existing['quantity'] + quantity
                    else:
                        new_quantity = existing['quantity'] - quantity
                    if new_quantity <= 0:
                        cursor.execute("DELETE FROM paper_trading_options_positions WHERE position_id = ?", (existing['position_id'],))
                    else:
                        cursor.execute("UPDATE paper_trading_options_positions SET quantity = ?, updated_at = ? WHERE position_id = ?", 
                                       (new_quantity, datetime.now(), existing['position_id']))
                else:
                    # New position
                    pos_id = f"OPOS_{account_id}_{contract}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    cursor.execute("""
                        INSERT INTO paper_trading_options_positions 
                        (position_id, account_id, symbol, underlying_symbol, option_type, strike, expiration, side, quantity, entry_price)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (pos_id, account_id, contract, underlying, opt_type, strike, expiration, side, quantity, price))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating option position: {e}")

    def get_option_positions(self, account_id: str) -> List[OptionPosition]:
        """Get all open option positions for an account."""
        try:
            with self.db_manager._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM paper_trading_options_positions WHERE account_id = ?", (account_id,))
                
                positions = []
                for row in cursor.fetchall():
                    # For now current price = entry price (placeholder until real-time opt prices integrated)
                    positions.append(OptionPosition(
                        position_id=row['position_id'],
                        account_id=row['account_id'],
                        symbol=row['symbol'],
                        underlying_symbol=row['underlying_symbol'],
                        option_type=row['option_type'],
                        strike=row['strike'],
                        expiration=row['expiration'],
                        side=row['side'],
                        quantity=row['quantity'],
                        entry_price=row['entry_price'],
                        current_price=row['entry_price'],
                        unrealized_pnl=0.0,
                        entry_time=datetime.fromisoformat(row['entry_time']) if isinstance(row['entry_time'], str) else row['entry_time'],
                        updated_at=datetime.now()
                    ))
                return positions
        except Exception as e:
            logger.error(f"Error getting option positions: {e}")
            return []
    
    def execute_optimization_order(self, account_id: str, symbol: str, 
                                 optimization_type: str, strategy: str,
                                 details: str) -> bool:
        """
        Execute a complex optimization order (e.g. roll, hedge, or convert to convex).
        This is a high-level command from the Position Optimizer.
        """
        try:
            logger.info(f"Executing optimization: {optimization_type} - {strategy} for {symbol}")
            
            # 1. Protection/Hedging (Add to existing)
            if optimization_type == "PROTECTION":
                # Typically suggests a Put Hedge. 
                # We extract strike from details if possible, or use a default 5% OTM
                price = self._get_current_price(symbol)
                strike = round(price * 0.95, 0)
                expiry = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
                
                self.execute_option_trade(
                    account_id=account_id,
                    underlying_symbol=symbol,
                    contract_symbol=f"{symbol}_HEDGE_{expiry}",
                    action="BUY",
                    quantity=1.0, # 1 lot per 100 shares assumed
                    price=price * 0.02, # Estimated 2% premium
                    option_type="PUT",
                    strike=strike,
                    expiration=expiry,
                    strategy_name="Optimizer Hedge",
                    ai_reasoning=details
                )
            
            # 2. Convexity Upgrade (Replace Stock with Spreads)
            elif optimization_type == "CONVEXITY_UPGRADE":
                # A: Close existing stock
                positions = self.get_positions(account_id)
                stock_pos = next((p for p in positions if p.symbol == symbol), None)
                if stock_pos:
                    self.execute_trade(
                        account_id=account_id,
                        symbol=symbol,
                        action=TradeAction.SELL if stock_pos.side == 'LONG' else TradeAction.COVER,
                        quantity=stock_pos.quantity,
                        strategy_name="Optimizer Rebalance",
                        ai_reasoning="Closing for convexity upgrade."
                    )
                
                # B: Open recommended strategy (e.g. Bull Call Spread)
                # Simplified: open a single ATM call as a proxy for the 'Spread' component for now
                price = self._get_current_price(symbol)
                expiry = (datetime.now() + timedelta(days=45)).strftime('%Y-%m-%d')
                self.execute_option_trade(
                    account_id=account_id,
                    underlying_symbol=symbol,
                    contract_symbol=f"{symbol}_SPREAD_{expiry}",
                    action="BUY",
                    quantity=1.0,
                    price=price * 0.05,
                    option_type="CALL",
                    strike=price,
                    expiration=expiry,
                    strategy_name=f"Optimizer {strategy}",
                    ai_reasoning=details
                )

            # 3. Profit Lock (Covered Call)
            elif optimization_type == "PROFIT_LOCK":
                price = self._get_current_price(symbol)
                strike = round(price * 1.05, 0) # 5% OTM
                expiry = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
                self.execute_option_trade(
                    account_id=account_id,
                    underlying_symbol=symbol,
                    contract_symbol=f"{symbol}_CC_{expiry}",
                    action="SELL",
                    quantity=1.0,
                    price=price * 0.015,
                    option_type="CALL",
                    strike=strike,
                    expiration=expiry,
                    strategy_name="Optimizer Profit Lock",
                    ai_reasoning=details
                )
                
            return True
        except Exception as e:
            logger.error(f"Optimization execution failed: {e}")
            return False

    def update_automation_settings(self, account_id: str, enabled: bool,
                                   config: Optional[Dict] = None) -> bool:
        """Update automation settings for an account."""
        try:
            with self.db_manager._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE paper_trading_accounts
                    SET automation_enabled = ?,
                        automation_config = ?,
                        updated_at = ?
                    WHERE account_id = ?
                """, (enabled, json.dumps(config or {}), datetime.now(), account_id))
                conn.commit()
                
                logger.info(f"Updated automation for account {account_id}: enabled={enabled}")
                return True
        
        except Exception as e:
            logger.error(f"Error updating automation settings: {e}")
            return False


# Global instance
_paper_trading_system = None


def get_paper_trading_system() -> PaperTradingSystem:
    """Get global paper trading system instance."""
    global _paper_trading_system
    if _paper_trading_system is None:
        _paper_trading_system = PaperTradingSystem()
    return _paper_trading_system
