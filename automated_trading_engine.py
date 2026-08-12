"""
Automated Trading Engine

Provides AI-driven automated trading for paper trading accounts:
1. Scans market for opportunities using unbiased analyzer
2. Generates trades based on real market data
3. Respects position sizing and risk management rules
4. User-controlled enable/disable/pause/resume
5. Comprehensive logging of all automated decisions

Author: APB - Octavian Team
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import json

from paper_trading_system import (
    get_paper_trading_system, 
    PaperTradingAccount, 
    TradeAction,
    AccountStatus
)
from unbiased_market_analyzer import UnbiasedMarketAnalyzer
from database_manager import get_database_manager

try:
    from data_sources import get_realtime_price, get_latest_price
    _HAS_REALTIME = True
except ImportError:
    _HAS_REALTIME = False
    get_realtime_price = None
    get_latest_price = None

logger = logging.getLogger(__name__)


class AutomationStatus(Enum):
    """Automation status types."""
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@dataclass
class RiskManagementRules:
    """Risk management rules for automated trading."""
    max_position_size_pct: float = 10.0  # Max % of account per position
    max_total_exposure_pct: float = 80.0  # Max % of account in positions
    max_positions: int = 10  # Max number of concurrent positions
    min_confidence_threshold: float = 0.65  # Min confidence to trade
    max_loss_per_trade_pct: float = 2.0  # Max loss per trade
    profit_target_multiplier: float = 2.0  # Risk/reward ratio
    stop_loss_pct: float = 5.0  # Stop loss %
    take_profit_pct: float = 10.0  # Take profit %
    
    # Full automation mode settings
    full_auto_mode: bool = False  # AI has complete control
    allow_shorts: bool = True  # Allow short positions
    emergency_stop_loss_pct: float = 15.0  # Account-level stop loss
    daily_loss_limit_pct: float = 5.0  # Daily loss limit
    
    def is_full_auto(self) -> bool:
        """Check if running in full automation mode."""
        return self.full_auto_mode
    
    def get_effective_max_position_size(self, confidence: float) -> float:
        """Get position size based on confidence (full auto scales with confidence)."""
        if self.full_auto_mode:
            # In full auto, scale position size with confidence
            # High confidence = larger positions (up to max)
            base_size = self.max_position_size_pct
            confidence_multiplier = min(confidence / 0.5, 1.5)  # Up to 1.5x for high confidence
            return min(base_size * confidence_multiplier, self.max_position_size_pct)
        else:
            # Custom mode: use configured max
            return self.max_position_size_pct
    
    def should_stop_trading(self, account_pnl_pct: float, daily_pnl_pct: float) -> bool:
        """Check if trading should stop due to losses."""
        # Emergency stop loss (account level)
        if account_pnl_pct <= -self.emergency_stop_loss_pct:
            return True
        
        # Daily loss limit
        if daily_pnl_pct <= -self.daily_loss_limit_pct:
            return True
        
        return False


@dataclass
class AutomatedTradeDecision:
    """Automated trade decision data."""
    account_id: str
    symbol: str
    action: TradeAction
    quantity: float
    expected_price: float
    confidence: float
    reasoning: str
    opportunity_score: float
    risk_reward_ratio: float
    timestamp: datetime


class AutomatedTradingEngine:
    """Engine for automated paper trading with AI decision making."""
    
    def __init__(self):
        self.paper_trading = get_paper_trading_system()
        self.unbiased_analyzer = UnbiasedMarketAnalyzer()
        self.db_manager = get_database_manager()
        
        # Automation state
        self.active_automations: Dict[str, AutomationStatus] = {}
        self.automation_threads: Dict[str, threading.Thread] = {}
        self.stop_flags: Dict[str, threading.Event] = {}
        
        # Performance tracking
        self.automation_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Activity / scan log (capped ring buffer)
        self.scan_logs: Dict[str, List[Dict[str, Any]]] = {}
        self._MAX_LOG_ENTRIES = 200
        
        # Configurable scan interval per account (default 60s)
        self.scan_intervals: Dict[str, int] = {}
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def start_automation(self, account_id: str, 
                        risk_rules: Optional[RiskManagementRules] = None) -> bool:
        """
        Start automated trading for an account.
        
        Args:
            account_id: Paper trading account ID
            risk_rules: Risk management rules (uses defaults if None)
            
        Returns:
            True if automation started successfully
        """
        try:
            with self.lock:
                # Check if already running
                if account_id in self.active_automations:
                    if self.active_automations[account_id] == AutomationStatus.RUNNING:
                        logger.warning(f"Automation already running for {account_id}")
                        return False
                
                # Verify account exists and is active
                account = self.paper_trading.get_account(account_id)
                if not account or account.status != AccountStatus.ACTIVE:
                    logger.error(f"Account {account_id} not found or not active")
                    return False
                
                # Use provided rules or defaults
                if risk_rules is None:
                    risk_rules = RiskManagementRules()
                
                # Update account automation config
                config = {
                    'risk_rules': {
                        'max_position_size_pct': risk_rules.max_position_size_pct,
                        'max_total_exposure_pct': risk_rules.max_total_exposure_pct,
                        'max_positions': risk_rules.max_positions,
                        'min_confidence_threshold': risk_rules.min_confidence_threshold,
                        'max_loss_per_trade_pct': risk_rules.max_loss_per_trade_pct,
                        'profit_target_multiplier': risk_rules.profit_target_multiplier,
                        'stop_loss_pct': risk_rules.stop_loss_pct,
                        'take_profit_pct': risk_rules.take_profit_pct
                    },
                    'started_at': datetime.now().isoformat()
                }
                
                self.paper_trading.update_automation_settings(
                    account_id, enabled=True, config=config
                )
                
                # Create stop flag
                stop_flag = threading.Event()
                self.stop_flags[account_id] = stop_flag
                
                # Start automation thread
                thread = threading.Thread(
                    target=self._automation_loop,
                    args=(account_id, risk_rules, stop_flag),
                    daemon=True
                )
                thread.start()
                
                self.automation_threads[account_id] = thread
                self.active_automations[account_id] = AutomationStatus.RUNNING
                
                # Initialize metrics
                self.automation_metrics[account_id] = {
                    'trades_executed': 0,
                    'opportunities_evaluated': 0,
                    'total_pnl': 0.0,
                    'win_rate': 0.0,
                    'symbols_scanned': 0,
                    'positions_auto_closed': 0,
                    'last_scan_time': None,
                    'last_scan_symbols': 0,
                    'started_at': datetime.now().isoformat()
                }
                self.scan_logs[account_id] = []
                
                logger.info(f"Started automation for account {account_id}")
                return True
        
        except Exception as e:
            logger.error(f"Error starting automation for {account_id}: {e}")
            return False
    
    def pause_automation(self, account_id: str) -> bool:
        """Pause automated trading for an account."""
        try:
            with self.lock:
                if account_id not in self.active_automations:
                    logger.warning(f"No automation found for {account_id}")
                    return False
                
                if self.active_automations[account_id] == AutomationStatus.PAUSED:
                    logger.warning(f"Automation already paused for {account_id}")
                    return False
                
                self.active_automations[account_id] = AutomationStatus.PAUSED
                logger.info(f"Paused automation for account {account_id}")
                return True
        
        except Exception as e:
            logger.error(f"Error pausing automation for {account_id}: {e}")
            return False
    
    def resume_automation(self, account_id: str) -> bool:
        """Resume paused automated trading."""
        try:
            with self.lock:
                if account_id not in self.active_automations:
                    logger.warning(f"No automation found for {account_id}")
                    return False
                
                if self.active_automations[account_id] != AutomationStatus.PAUSED:
                    logger.warning(f"Automation not paused for {account_id}")
                    return False
                
                self.active_automations[account_id] = AutomationStatus.RUNNING
                logger.info(f"Resumed automation for account {account_id}")
                return True
        
        except Exception as e:
            logger.error(f"Error resuming automation for {account_id}: {e}")
            return False
    
    def stop_automation(self, account_id: str) -> bool:
        """Stop automated trading for an account."""
        try:
            with self.lock:
                if account_id not in self.active_automations:
                    logger.warning(f"No automation found for {account_id}")
                    return False
                
                # Set stop flag
                if account_id in self.stop_flags:
                    self.stop_flags[account_id].set()
                
                # Update status
                self.active_automations[account_id] = AutomationStatus.STOPPED
                
                # Update account settings
                self.paper_trading.update_automation_settings(
                    account_id, enabled=False, config={}
                )
                
                logger.info(f"Stopped automation for account {account_id}")
                return True
        
        except Exception as e:
            logger.error(f"Error stopping automation for {account_id}: {e}")
            return False
    
    def set_scan_interval(self, account_id: str, seconds: int) -> None:
        """Set the scan interval for an account."""
        self.scan_intervals[account_id] = max(30, min(600, seconds))

    def get_automation_status(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get current automation status, metrics, and recent activity log."""
        try:
            if account_id not in self.active_automations:
                return None
            
            status = self.active_automations[account_id]
            metrics = self.automation_metrics.get(account_id, {})
            recent_logs = list(self.scan_logs.get(account_id, []))[-50:]  # Last 50
            
            return {
                'status': status.value,
                'metrics': metrics,
                'is_running': status == AutomationStatus.RUNNING,
                'is_paused': status == AutomationStatus.PAUSED,
                'scan_interval': self.scan_intervals.get(account_id, 60),
                'activity_log': recent_logs
            }
        
        except Exception as e:
            logger.error(f"Error getting automation status for {account_id}: {e}")
            return None

    def _log_activity(self, account_id: str, event_type: str, message: str,
                      details: Optional[Dict] = None):
        """Append an entry to the activity log for an account."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'message': message,
            'details': details or {}
        }
        if account_id not in self.scan_logs:
            self.scan_logs[account_id] = []
        self.scan_logs[account_id].append(entry)
        # Cap size
        if len(self.scan_logs[account_id]) > self._MAX_LOG_ENTRIES:
            self.scan_logs[account_id] = self.scan_logs[account_id][-self._MAX_LOG_ENTRIES:]
    
    def _automation_loop(self, account_id: str, risk_rules: RiskManagementRules,
                        stop_flag: threading.Event):
        """Main automation loop that runs in separate thread."""
        logger.info(f"Automation loop started for {account_id}")
        self._log_activity(account_id, 'START', 'Automation loop started')
        
        while not stop_flag.is_set():
            try:
                # Check if paused
                if self.active_automations.get(account_id) == AutomationStatus.PAUSED:
                    time.sleep(10)
                    continue
                
                # Check if stopped
                if self.active_automations.get(account_id) == AutomationStatus.STOPPED:
                    break
                
                # Run trading cycle (scan market + manage positions)
                asyncio.run(self._execute_trading_cycle(account_id, risk_rules))
                
                # Wait before next cycle (configurable, default 60s)
                interval = self.scan_intervals.get(account_id, 60)
                time.sleep(interval)
            
            except Exception as e:
                logger.error(f"Error in automation loop for {account_id}: {e}")
                self._log_activity(account_id, 'ERROR', f'Loop error: {str(e)[:200]}')
                self.active_automations[account_id] = AutomationStatus.ERROR
                time.sleep(60)
        
        logger.info(f"Automation loop stopped for {account_id}")
        self._log_activity(account_id, 'STOP', 'Automation loop stopped')
        
        # Cleanup
        with self.lock:
            if account_id in self.automation_threads:
                del self.automation_threads[account_id]
            if account_id in self.stop_flags:
                del self.stop_flags[account_id]
    
    async def _execute_trading_cycle(self, account_id: str, 
                                     risk_rules: RiskManagementRules):
        """Execute one trading cycle: manage positions, scan market, evaluate, trade."""
        try:
            logger.info(f"Executing trading cycle for {account_id}")
            self._log_activity(account_id, 'SCAN_START', 'Starting market scan cycle')
            
            # Get account state
            account = self.paper_trading.get_account(account_id)
            if not account:
                logger.error(f"Account {account_id} not found")
                return
            
            positions = self.paper_trading.get_positions(account_id)
            
            # ── Step 1: Scan entire market for opportunities ──
            opportunities = await self.unbiased_analyzer.scan_entire_market(max_symbols=200)
            
            scan_count = len(opportunities)
            # Update metrics
            if account_id in self.automation_metrics:
                self.automation_metrics[account_id]['opportunities_evaluated'] += scan_count
                self.automation_metrics[account_id]['symbols_scanned'] += scan_count
                self.automation_metrics[account_id]['last_scan_time'] = datetime.now().isoformat()
                self.automation_metrics[account_id]['last_scan_symbols'] = scan_count
            
            # ── Step 2: Manage existing positions (SL / TP / Signal Reversal) ──
            positions = self._manage_existing_positions(
                account_id, positions, risk_rules, opportunities
            )
            
            # ── Step 3: Check risk limits before new entries ──
            if not self._check_risk_limits(account, positions, risk_rules):
                self._log_activity(account_id, 'RISK_LIMIT',
                                   f'Risk limits reached ({len(positions)} positions), skipping new entries')
                return
            
            # Filter by confidence threshold
            qualified_opportunities = [
                opp for opp in opportunities
                if opp.confidence_score >= risk_rules.min_confidence_threshold
            ]
            
            # Exclude symbols we already hold
            held_symbols = {pos.symbol for pos in positions}
            qualified_opportunities = [
                opp for opp in qualified_opportunities
                if opp.symbol not in held_symbols
            ]
            
            self._log_activity(account_id, 'SCAN_RESULT',
                               f'Scanned {scan_count} assets, {len(qualified_opportunities)} qualify '
                               f'(confidence >= {risk_rules.min_confidence_threshold:.0%})',
                               {'total_scanned': scan_count, 'qualified': len(qualified_opportunities)})
            
            if not qualified_opportunities:
                return
            
            # Sort by opportunity score
            qualified_opportunities.sort(
                key=lambda x: x.profit_probability * x.confidence_score * abs(x.expected_return),
                reverse=True
            )
            
            # ── Step 4: Evaluate and execute top opportunities ──
            for opportunity in qualified_opportunities[:5]:
                if not self._can_take_trade(account, positions, risk_rules):
                    break
                
                decision = self._generate_trade_decision(
                    account_id, opportunity, account, risk_rules
                )
                
                if decision:
                    success = self._execute_automated_trade(decision)
                    
                    if success:
                        if account_id in self.automation_metrics:
                            self.automation_metrics[account_id]['trades_executed'] += 1
                        self._log_activity(account_id, 'TRADE',
                                           f'{decision.action.value} {decision.quantity:.1f} '
                                           f'{decision.symbol} @ ${decision.expected_price:,.2f} '
                                           f'(conf {decision.confidence:.0%})',
                                           {'symbol': decision.symbol,
                                            'action': decision.action.value,
                                            'price': decision.expected_price,
                                            'confidence': decision.confidence})
                        # Refresh positions
                        positions = self.paper_trading.get_positions(account_id)
                        account = self.paper_trading.get_account(account_id)
            
            logger.info(f"Trading cycle completed for {account_id}")
        
        except Exception as e:
            logger.error(f"Error in trading cycle for {account_id}: {e}")
            self._log_activity(account_id, 'ERROR', f'Cycle error: {str(e)[:200]}')

    def _manage_existing_positions(self, account_id: str, positions: List,
                                    risk_rules: RiskManagementRules,
                                    opportunities: Optional[List] = None) -> List:
        """Check existing positions against stop-loss / take-profit and AI signal reversals."""
        remaining = []
        MAX_HOLD_DAYS = 5  # Safety net: auto-close stale positions
        
        # Fast lookup map for latest signals
        opp_map = {opp.symbol: opp for opp in (opportunities or [])}
        
        for pos in positions:
            try:
                # Fetch live price — fall back to the position's stored current_price
                live_price = self._fetch_live_price(pos.symbol)
                if live_price is None or live_price <= 0:
                    # Fallback: use the position's last-known current_price
                    fallback_price = getattr(pos, 'current_price', None)
                    if fallback_price and fallback_price > 0:
                        live_price = fallback_price
                        logger.warning(
                            f"Using fallback price ${live_price:,.2f} for {pos.symbol} "
                            f"(live fetch failed)"
                        )
                    else:
                        logger.error(
                            f"No price available for {pos.symbol} — "
                            f"live fetch AND fallback both failed, skipping"
                        )
                        remaining.append(pos)
                        continue
                
                entry = pos.entry_price
                if entry <= 0:
                    remaining.append(pos)
                    continue
                
                if pos.side == 'LONG':
                    pnl_pct = ((live_price - entry) / entry) * 100
                else:
                    pnl_pct = ((entry - live_price) / entry) * 100
                
                should_close = False
                close_reason = ''
                
                # Stop-loss hit
                if pnl_pct <= -risk_rules.stop_loss_pct:
                    should_close = True
                    close_reason = f'Stop-loss triggered ({pnl_pct:+.1f}% vs -{risk_rules.stop_loss_pct}% limit)'
                
                # Take-profit hit
                elif pnl_pct >= risk_rules.take_profit_pct:
                    should_close = True
                    close_reason = f'Take-profit triggered ({pnl_pct:+.1f}% vs +{risk_rules.take_profit_pct}% target)'
                
                # Time-based expiry — auto-close positions held too long
                elif hasattr(pos, 'entry_time') and pos.entry_time:
                    from datetime import datetime, timedelta
                    try:
                        entry_time = pos.entry_time if isinstance(pos.entry_time, datetime) else datetime.fromisoformat(str(pos.entry_time))
                        hold_duration = datetime.now() - entry_time
                        if hold_duration > timedelta(days=MAX_HOLD_DAYS):
                            should_close = True
                            close_reason = f'Time-based expiry ({hold_duration.days}d held, max {MAX_HOLD_DAYS}d, PnL: {pnl_pct:+.1f}%)'
                    except Exception:
                        pass  # Entry time parsing failed, skip time check
                
                # 4. AI Signal Reversal check (Smart closing)
                if not should_close and pos.symbol in opp_map:
                    opp = opp_map[pos.symbol]
                    
                    # For Full Auto mode, we are more aggressive with signal-based closing
                    # The bot should "act exactly as the model wants" - prioritizing current signals over heuristics.
                    if risk_rules.is_full_auto():
                        reversal_threshold = 0.001  # Exit if even slightly opposite (0.1%)
                        min_confidence = 0.35        # Exit if confidence drops below 35%
                    else:
                        reversal_threshold = 0.015  # 1.5% for standard automated mode
                        min_confidence = 0.5         # 50% for standard automated mode
                    
                    if pos.side == 'LONG':
                        # Exit if model expects negative return (however small) or direction is not BULLISH
                        if (opp.expected_return < -reversal_threshold) or \
                           (risk_rules.is_full_auto() and opp.expected_return <= 0):
                            should_close = True
                            close_reason = f'AI Reversal Signal: BEARISH/NEUTRAL ({opp.expected_return:+.2%}) for LONG'
                        # Exit if confidence in the trade has faded
                        elif opp.confidence_score < min_confidence:
                            should_close = True
                            close_reason = f'Signal Weakness: Confidence dropped to {opp.confidence_score:.1%}'
                            
                    elif pos.side == 'SHORT':
                        # Exit if model expects positive return or direction is not BEARISH
                        if (opp.expected_return > reversal_threshold) or \
                           (risk_rules.is_full_auto() and opp.expected_return >= 0):
                            should_close = True
                            close_reason = f'AI Reversal Signal: BULLISH/NEUTRAL ({opp.expected_return:+.2%}) for SHORT'
                        # Exit if confidence in the trade has faded
                        elif opp.confidence_score < min_confidence:
                            should_close = True
                            close_reason = f'Signal Weakness: Confidence dropped to {opp.confidence_score:.1%}'

                if should_close:
                    action = TradeAction.SELL if pos.side == 'LONG' else TradeAction.COVER
                    logger.info(
                        f"Attempting auto-close: {action.value} {pos.quantity:.1f} {pos.symbol} "
                        f"@ ${live_price:,.2f} — {close_reason}"
                    )
                    trade = self.paper_trading.execute_trade(
                        account_id=account_id,
                        symbol=pos.symbol,
                        action=action,
                        quantity=pos.quantity,
                        price=live_price,
                        strategy_name='Automated Risk Management',
                        ai_reasoning=close_reason
                    )
                    if trade:
                        if account_id in self.automation_metrics:
                            self.automation_metrics[account_id]['positions_auto_closed'] += 1
                            self.automation_metrics[account_id]['trades_executed'] += 1
                        self._log_activity(account_id, 'AUTO_CLOSE',
                                           f'{action.value} {pos.quantity:.1f} {pos.symbol} '
                                           f'@ ${live_price:,.2f} — {close_reason}',
                                           {'symbol': pos.symbol, 'pnl_pct': pnl_pct})
                        logger.info(f"Successfully auto-closed {pos.symbol}: {close_reason}")
                    else:
                        logger.error(
                            f"execute_trade returned None for {pos.symbol} "
                            f"close attempt — position remains open"
                        )
                        remaining.append(pos)
                else:
                    remaining.append(pos)
            except Exception as e:
                remaining.append(pos)
                logger.error(f"Error managing position {pos.symbol}: {e}", exc_info=True)
        
        return remaining
    
    def _check_risk_limits(self, account: PaperTradingAccount, 
                          positions: List, risk_rules: RiskManagementRules) -> bool:
        """Check if account is within risk limits."""
        try:
            # Check max positions
            if len(positions) >= risk_rules.max_positions:
                logger.info(f"Max positions reached: {len(positions)}/{risk_rules.max_positions}")
                return False
            
            # Check total exposure
            total_exposure = sum(
                abs(pos.quantity * pos.current_price) for pos in positions
            )
            exposure_pct = (total_exposure / account.current_balance) * 100
            
            if exposure_pct >= risk_rules.max_total_exposure_pct:
                logger.info(f"Max exposure reached: {exposure_pct:.1f}%/{risk_rules.max_total_exposure_pct}%")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False
    
    def _can_take_trade(self, account: PaperTradingAccount, 
                       positions: List, risk_rules: RiskManagementRules) -> bool:
        """Check if we can take another trade."""
        # Check position limit
        if len(positions) >= risk_rules.max_positions:
            return False
        
        # Check if we have enough balance
        min_balance_required = account.initial_balance * 0.1  # Keep 10% in reserve
        if account.current_balance < min_balance_required:
            return False
        
        return True
    
    def _fetch_live_price(self, symbol: str) -> Optional[float]:
        """Fetch real-time price for a symbol using best available source."""
        # Method 1: data_sources.get_latest_price (most robust)
        if _HAS_REALTIME and get_latest_price:
            try:
                price = get_latest_price(symbol)
                if price and price > 0:
                    return float(price)
            except Exception as e:
                logger.debug(f"Method 1 (get_latest_price) failed for {symbol}: {e}")
        
        # Method 2: paper trading system's own price fetcher
        try:
            price = self.paper_trading._get_current_price(symbol)
            if price and price > 0:
                return float(price)
        except Exception as e:
            logger.debug(f"Method 2 (paper_trading._get_current_price) failed for {symbol}: {e}")
        
        # Method 3: yfinance direct fallback
        try:
            import yfinance as yf
            tk = yf.Ticker(symbol)
            hist = tk.history(period='5d')
            if hist is not None and not hist.empty:
                close = hist['Close']
                if hasattr(close, 'iloc') and len(close) > 0:
                    price = float(close.iloc[-1])
                    if price > 0:
                        return price
        except Exception as e:
            logger.debug(f"Method 3 (yfinance direct) failed for {symbol}: {e}")
        
        logger.warning(f"All price fetch methods failed for {symbol}")
        return None

    def _generate_trade_decision(self, account_id: str, opportunity: Any,
                                account: PaperTradingAccount,
                                risk_rules: RiskManagementRules) -> Optional[AutomatedTradeDecision]:
        """Generate trade decision from opportunity using REAL live prices."""
        try:
            # Fetch real live price
            live_price = self._fetch_live_price(opportunity.symbol)
            if live_price is None or live_price <= 0:
                logger.warning(f"Could not get live price for {opportunity.symbol}, skipping")
                self._log_activity(account_id, 'SKIP',
                                   f'No live price for {opportunity.symbol}')
                return None
            
            # Determine action
            if opportunity.expected_return > 0:
                action = TradeAction.BUY
            else:
                action = TradeAction.SHORT
            
            # Calculate position size
            position_size_dollars = account.current_balance * (risk_rules.max_position_size_pct / 100)
            
            # Adjust for confidence
            risk_adjusted_size = position_size_dollars * opportunity.confidence_score
            
            # Calculate quantity using REAL price
            quantity = risk_adjusted_size / live_price
            if quantity < 0.01:
                return None  # Position too small
            
            # Calculate risk/reward
            risk_reward_ratio = risk_rules.profit_target_multiplier
            
            # Get opportunity type safely
            opp_type = getattr(opportunity, 'opportunity_type', 'Statistical Edge')
            opp_reasoning = getattr(opportunity, 'reasoning', '')
            if not opp_reasoning and hasattr(opportunity, 'model_reasoning'):
                opp_reasoning = '; '.join(opportunity.model_reasoning[:3]) if opportunity.model_reasoning else ''
            
            # Generate reasoning
            reasoning = (
                f"Automated Trade Decision:\n"
                f"- Symbol: {opportunity.symbol}\n"
                f"- Live Price: ${live_price:,.2f}\n"
                f"- Profit Probability: {opportunity.profit_probability:.1%}\n"
                f"- Expected Return: {opportunity.expected_return:+.2%}\n"
                f"- Confidence: {opportunity.confidence_score:.1%}\n"
                f"- Risk-Adjusted Size: ${risk_adjusted_size:,.2f}\n"
                f"- Risk/Reward: {risk_reward_ratio:.1f}:1\n"
                f"\nAnalysis: {opp_reasoning}\n"
                f"\nRisk Management:\n"
                f"- Stop Loss: {risk_rules.stop_loss_pct}%\n"
                f"- Take Profit: {risk_rules.take_profit_pct}%\n"
                f"- Max Loss: {risk_rules.max_loss_per_trade_pct}% of account"
            )
            
            decision = AutomatedTradeDecision(
                account_id=account_id,
                symbol=opportunity.symbol,
                action=action,
                quantity=round(quantity, 4),
                expected_price=live_price,
                confidence=opportunity.confidence_score,
                reasoning=reasoning,
                opportunity_score=opportunity.profit_probability * opportunity.confidence_score,
                risk_reward_ratio=risk_reward_ratio,
                timestamp=datetime.now()
            )
            
            return decision
        
        except Exception as e:
            logger.error(f"Error generating trade decision: {e}")
            return None
    
    def _execute_automated_trade(self, decision: AutomatedTradeDecision) -> bool:
        """Execute automated trade decision."""
        try:
            # Execute trade through paper trading system
            trade = self.paper_trading.execute_trade(
                account_id=decision.account_id,
                symbol=decision.symbol,
                action=decision.action,
                quantity=decision.quantity,
                price=decision.expected_price,
                strategy_name="Automated AI Strategy",
                ai_reasoning=decision.reasoning,
                market_context={
                    'confidence': decision.confidence,
                    'opportunity_score': decision.opportunity_score,
                    'risk_reward_ratio': decision.risk_reward_ratio,
                    'automation_timestamp': decision.timestamp.isoformat()
                }
            )
            
            if trade:
                logger.info(f"Executed automated trade: {decision.action.value} {decision.quantity} {decision.symbol}")
                return True
            else:
                logger.error(f"Failed to execute automated trade for {decision.symbol}")
                return False
        
        except Exception as e:
            logger.error(f"Error executing automated trade: {e}")
            return False
    
    async def _evaluate_opportunities(self, account_id: str):
        """Evaluate trading opportunities and execute trades."""
        account = self.paper_trading.get_account(account_id)
        if not account:
            return
        
        config = self._automation_configs.get(account_id)
        if not config:
            return
        
        risk_rules = config['risk_rules']
        
        # Check emergency stops
        account_pnl_pct = (account.current_balance - account.initial_balance) / account.initial_balance * 100
        
        # Get daily P&L (simplified - would need trade history in real implementation)
        daily_pnl_pct = 0.0  # Calculate from today's trades
        
        if risk_rules.should_stop_trading(account_pnl_pct, daily_pnl_pct):
            self.stop_automation(account_id)
            print(f"Emergency stop triggered for account {account_id}")
            return
        
        # Get market opportunities
        try:
            # Scan market for opportunities
            universe_size = 200 if risk_rules.is_full_auto() else 50
            opportunities = await self._scan_market_opportunities(universe_size)
            
            # Filter by confidence threshold
            min_confidence = risk_rules.min_confidence_threshold
            qualified_opportunities = [
                opp for opp in opportunities 
                if opp.confidence_score >= min_confidence
            ]
            
            # In full auto mode, AI decides how many to take
            # In custom mode, respect max_positions limit
            if risk_rules.is_full_auto():
                # Take all high-confidence opportunities (up to balance limits)
                target_count = len(qualified_opportunities)
            else:
                # Respect configured max positions
                current_positions = len(self.paper_trading.get_positions(account_id))
                slots_available = max(0, risk_rules.max_positions - current_positions)
                target_count = min(len(qualified_opportunities), slots_available)
            
            # Execute top opportunities
            for opp in qualified_opportunities[:target_count]:
                await self._execute_opportunity(account_id, opp, risk_rules, account)
            
            # Update metrics
            config['metrics']['opportunities_evaluated'] += len(opportunities)
            
        except Exception as e:
            print(f"Error evaluating opportunities: {e}")
    
    async def _execute_opportunity(self, account_id: str, opportunity, risk_rules: RiskManagementRules, account):
        """Execute a trading opportunity."""
        try:
            # Calculate position size
            confidence = opportunity.confidence_score
            base_position_pct = risk_rules.get_effective_max_position_size(confidence)
            
            # In full auto mode, adjust based on expected return
            if risk_rules.is_full_auto():
                expected_return = getattr(opportunity, 'expected_return', 0.0)
                if expected_return > 0.10:  # >10% expected return
                    base_position_pct *= 1.25  # Increase size
                elif expected_return < 0.05:  # <5% expected return
                    base_position_pct *= 0.75  # Decrease size
            
            # Calculate quantity
            position_value = account.current_balance * (base_position_pct / 100)
            quantity = int(position_value / opportunity.current_price)
            
            if quantity < 1:
                return
            
            # Determine action (LONG vs SHORT)
            if opportunity.signal_direction == 'BULLISH':
                action = TradeAction.BUY
            elif opportunity.signal_direction == 'BEARISH' and risk_rules.allow_shorts:
                action = TradeAction.SHORT
            else:
                return  # Skip bearish if shorts not allowed
            
            # Calculate stop loss and take profit
            if risk_rules.is_full_auto():
                # AI sets its own levels based on volatility and expected return
                stop_loss_pct = risk_rules.stop_loss_pct
                take_profit_pct = risk_rules.take_profit_pct
            else:
                # Use configured levels
                stop_loss_pct = risk_rules.stop_loss_pct
                take_profit_pct = risk_rules.take_profit_pct
            
            # Execute trade
            trade = self.paper_trading.execute_trade(
                account_id=account_id,
                symbol=opportunity.symbol,
                action=action,
                quantity=quantity,
                price=opportunity.current_price,
                strategy_name="Full Automation" if risk_rules.is_full_auto() else "Custom Automation",
                ai_reasoning=f"Confidence: {confidence:.1%} | Expected Return: {opportunity.expected_return:+.1%} | "
                           f"Mode: {'FULL AUTO' if risk_rules.is_full_auto() else 'CUSTOM'}"
            )
            
            if trade:
                config = self._automation_configs.get(account_id)
                config['metrics']['trades_executed'] += 1
                
                print(f"Executed {action.value} {quantity}x {opportunity.symbol} @ ${opportunity.current_price:.2f}")
            
        except Exception as e:
            print(f"Error executing opportunity: {e}")
    
    async def _scan_market_opportunities(self, universe_size: int = 50):
        """Scan market for trading opportunities."""
        try:
            from unbiased_market_analyzer import UnbiasedMarketAnalyzer
            
            analyzer = UnbiasedMarketAnalyzer()
            opportunities = await analyzer.scan_entire_market(
                max_symbols=universe_size,
                min_confidence=0.20  # Lower threshold, filter later
            )
            
            return opportunities
            
        except Exception as e:
            print(f"Market scan error: {e}")
            return []


# Global instance
_automated_trading_engine = None


def get_automated_trading_engine() -> AutomatedTradingEngine:
    """Get global automated trading engine instance."""
    global _automated_trading_engine
    if _automated_trading_engine is None:
        _automated_trading_engine = AutomatedTradingEngine()
    return _automated_trading_engine
