# Automated Trading Engine Guide

## Overview

The Automated Trading Engine enables AI-driven paper trading that continuously scans the market for opportunities and executes trades automatically based on configurable risk management rules.

## Quick Start

### Basic Usage

```python
from automated_trading_engine import get_automated_trading_engine, RiskManagementRules
from paper_trading_system import get_paper_trading_system

# Get engine instance
engine = get_automated_trading_engine()

# Create a paper trading account first
paper_trading = get_paper_trading_system()
account = paper_trading.create_account(
    user_id="user123",
    account_name="AI Strategy Account",
    initial_balance=100000.0
)

# Start automation with default risk rules
success = engine.start_automation(account.account_id)

if success:
    print(f"Automation started for {account.account_id}")
```

### Custom Risk Rules

```python
# Define custom risk management rules
custom_rules = RiskManagementRules(
    max_position_size_pct=5.0,      # Max 5% per position
    max_total_exposure_pct=50.0,    # Max 50% total exposure
    max_positions=5,                 # Max 5 positions
    min_confidence_threshold=0.75,   # Min 75% confidence
    max_loss_per_trade_pct=1.0,     # Max 1% loss per trade
    profit_target_multiplier=3.0,    # 3:1 risk/reward
    stop_loss_pct=3.0,              # 3% stop loss
    take_profit_pct=9.0             # 9% take profit
)

# Start with custom rules
engine.start_automation(account.account_id, risk_rules=custom_rules)
```

## Control Functions

### Start Automation

```python
success = engine.start_automation(
    account_id="PT_user123_20260304120000",
    risk_rules=custom_rules  # Optional, uses defaults if None
)
```

**What happens:**
- Validates account exists and is active
- Saves risk rules to account config
- Starts background thread for continuous trading
- Begins scanning market every 5 minutes
- Returns `True` if started successfully

### Pause Automation

```python
success = engine.pause_automation(account_id)
```

**What happens:**
- Stops generating new trades
- Keeps existing positions open
- Thread continues running but skips trading
- Can be resumed later
- Returns `True` if paused successfully

### Resume Automation

```python
success = engine.resume_automation(account_id)
```

**What happens:**
- Resumes generating trades
- Continues from paused state
- Uses same risk rules as before
- Returns `True` if resumed successfully

### Stop Automation

```python
success = engine.stop_automation(account_id)
```

**What happens:**
- Completely stops automation
- Closes background thread
- Keeps existing positions (doesn't auto-close)
- Updates account automation settings
- Returns `True` if stopped successfully

### Get Status

```python
status = engine.get_automation_status(account_id)

if status:
    print(f"Status: {status['status']}")
    print(f"Is Running: {status['is_running']}")
    print(f"Trades Executed: {status['metrics']['trades_executed']}")
    print(f"Opportunities Evaluated: {status['metrics']['opportunities_evaluated']}")
```

**Returns:**
```python
{
    'status': 'RUNNING',  # or 'PAUSED', 'STOPPED', 'ERROR'
    'metrics': {
        'trades_executed': 15,
        'opportunities_evaluated': 250,
        'total_pnl': 2500.50,
        'win_rate': 0.67,
        'started_at': '2026-03-04T12:00:00'
    },
    'is_running': True,
    'is_paused': False
}
```

## Risk Management Rules

### RiskManagementRules Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_position_size_pct` | 10.0 | Maximum % of account per position |
| `max_total_exposure_pct` | 80.0 | Maximum % of account in all positions |
| `max_positions` | 10 | Maximum number of open positions |
| `min_confidence_threshold` | 0.65 | Minimum confidence to execute trade |
| `max_loss_per_trade_pct` | 2.0 | Maximum loss per trade as % of account |
| `profit_target_multiplier` | 2.0 | Risk/reward ratio (2:1 means 2x profit vs risk) |
| `stop_loss_pct` | 5.0 | Stop loss percentage from entry |
| `take_profit_pct` | 10.0 | Take profit percentage from entry |

### Conservative Strategy

```python
conservative = RiskManagementRules(
    max_position_size_pct=3.0,
    max_total_exposure_pct=30.0,
    max_positions=3,
    min_confidence_threshold=0.80,
    max_loss_per_trade_pct=0.5,
    profit_target_multiplier=4.0,
    stop_loss_pct=2.0,
    take_profit_pct=8.0
)
```

### Aggressive Strategy

```python
aggressive = RiskManagementRules(
    max_position_size_pct=15.0,
    max_total_exposure_pct=90.0,
    max_positions=15,
    min_confidence_threshold=0.55,
    max_loss_per_trade_pct=3.0,
    profit_target_multiplier=1.5,
    stop_loss_pct=7.0,
    take_profit_pct=10.5
)
```

### Balanced Strategy (Default)

```python
balanced = RiskManagementRules()  # Uses all defaults
```

## How It Works

### Trading Cycle (Every 5 Minutes)

1. **Check Status**
   - Verify automation is running (not paused/stopped)
   - Get current account state and positions

2. **Risk Limit Check**
   - Verify within max positions limit
   - Check total exposure percentage
   - Ensure sufficient balance reserves

3. **Market Scan**
   - Scan up to 100 symbols using unbiased analyzer
   - Get profit probability, confidence, expected return
   - Filter by minimum confidence threshold

4. **Opportunity Evaluation**
   - Sort by opportunity score (probability × confidence × return)
   - Evaluate top 5 opportunities
   - Check if each trade passes risk checks

5. **Trade Generation**
   - Determine action (BUY for positive return, SHORT for negative)
   - Calculate position size based on risk rules
   - Adjust size by confidence score
   - Generate comprehensive AI reasoning

6. **Trade Execution**
   - Execute through paper trading system
   - Log with full reasoning and market context
   - Update performance metrics
   - Refresh positions

### Position Sizing Logic

```python
# Base position size
base_size = account_balance * (max_position_size_pct / 100)

# Risk-adjusted size
adjusted_size = base_size * confidence_score

# Example:
# Account: $100,000
# Max position: 10%
# Confidence: 75%
# Position size: $100,000 * 0.10 * 0.75 = $7,500
```

### Opportunity Scoring

```python
opportunity_score = profit_probability * confidence_score * abs(expected_return)

# Example:
# Profit probability: 70%
# Confidence: 80%
# Expected return: 5%
# Score: 0.70 * 0.80 * 0.05 = 0.028
```

## Trade Logging

Every automated trade includes:

```python
{
    'trade_id': 'TR_PT_user123_...',
    'account_id': 'PT_user123_...',
    'symbol': 'AAPL',
    'action': 'BUY',
    'quantity': 50.0,
    'price': 175.50,
    'total_value': 8775.00,
    'strategy_name': 'Automated AI Strategy',
    'ai_reasoning': '''
        Automated Trade Decision:
        - Opportunity Type: momentum_breakout
        - Profit Probability: 72.5%
        - Expected Return: +4.8%
        - Confidence: 78.2%
        - Risk-Adjusted Position Size: $7,850.00
        - Risk/Reward Ratio: 2.0:1
        
        Analysis: Strong momentum indicators with volume confirmation...
        
        Risk Management:
        - Stop Loss: 5.0%
        - Take Profit: 10.0%
        - Max Loss: 2.0% of account
    ''',
    'market_context': {
        'confidence': 0.782,
        'opportunity_score': 0.0272,
        'risk_reward_ratio': 2.0,
        'automation_timestamp': '2026-03-04T14:30:00'
    },
    'timestamp': '2026-03-04T14:30:15'
}
```

## Safety Features

### Account Validation
- Verifies account exists before starting
- Checks account is ACTIVE status
- Validates sufficient balance

### Risk Limits
- Enforces max positions limit
- Checks total exposure before each trade
- Maintains balance reserves (10% minimum)
- Validates confidence threshold

### Thread Safety
- Uses locks for concurrent access
- Thread-safe status updates
- Atomic trade execution

### Error Handling
- Graceful error recovery
- Automatic retry on transient errors
- Sets ERROR status on critical failures
- Comprehensive error logging

### Stop Mechanisms
- Clean thread shutdown
- Stop flag for immediate halt
- Graceful cleanup on stop

## Monitoring

### Real-Time Status

```python
import time

while True:
    status = engine.get_automation_status(account_id)
    
    if status:
        print(f"\n=== Automation Status ===")
        print(f"Status: {status['status']}")
        print(f"Trades: {status['metrics']['trades_executed']}")
        print(f"Opportunities: {status['metrics']['opportunities_evaluated']}")
        print(f"P&L: ${status['metrics']['total_pnl']:.2f}")
        print(f"Win Rate: {status['metrics']['win_rate']:.1%}")
    
    time.sleep(60)  # Check every minute
```

### Performance Metrics

```python
# Get account to see overall performance
account = paper_trading.get_account(account_id)

print(f"Initial Balance: ${account.initial_balance:,.2f}")
print(f"Current Balance: ${account.current_balance:,.2f}")
print(f"Total P&L: ${account.total_pnl:,.2f} ({account.total_pnl_pct:.2f}%)")

# Get positions
positions = paper_trading.get_positions(account_id)
print(f"\nOpen Positions: {len(positions)}")

for pos in positions:
    print(f"  {pos.symbol}: {pos.quantity} @ ${pos.entry_price:.2f}")
    print(f"    Current: ${pos.current_price:.2f}")
    print(f"    P&L: ${pos.unrealized_pnl:.2f} ({pos.unrealized_pnl_pct:.2f}%)")

# Get trade history
trades = paper_trading.get_trade_history(account_id, limit=10)
print(f"\nRecent Trades: {len(trades)}")

for trade in trades[:5]:
    print(f"  {trade.timestamp}: {trade.action.value} {trade.quantity} {trade.symbol} @ ${trade.price:.2f}")
```

## Best Practices

### 1. Start Conservative
Begin with conservative risk rules and gradually increase as you gain confidence in the system.

### 2. Monitor Regularly
Check automation status and performance at least daily, especially in the first week.

### 3. Adjust Risk Rules
Fine-tune risk parameters based on observed performance and market conditions.

### 4. Use Pause Wisely
Pause automation during high volatility or uncertain market conditions rather than stopping completely.

### 5. Review Trade Logs
Regularly review AI reasoning for trades to understand decision-making patterns.

### 6. Set Realistic Expectations
Automated trading won't win every trade. Focus on overall win rate and risk-adjusted returns.

### 7. Test Thoroughly
Use paper trading extensively before considering any real money implementation.

## Troubleshooting

### Automation Won't Start

```python
# Check account status
account = paper_trading.get_account(account_id)
if not account:
    print("Account not found")
elif account.status != AccountStatus.ACTIVE:
    print(f"Account status is {account.status}, must be ACTIVE")
```

### No Trades Being Executed

```python
# Check automation status
status = engine.get_automation_status(account_id)
print(f"Status: {status['status']}")
print(f"Opportunities evaluated: {status['metrics']['opportunities_evaluated']}")

# If opportunities > 0 but trades = 0, check:
# 1. Confidence threshold might be too high
# 2. Position limits might be reached
# 3. Exposure limits might be reached
```

### Automation Stopped Unexpectedly

```python
status = engine.get_automation_status(account_id)
if status['status'] == 'ERROR':
    # Check logs for error details
    # Restart automation
    engine.stop_automation(account_id)
    time.sleep(5)
    engine.start_automation(account_id, risk_rules)
```

## Integration Example

### Complete Workflow

```python
from automated_trading_engine import get_automated_trading_engine, RiskManagementRules
from paper_trading_system import get_paper_trading_system
import time

# Initialize
engine = get_automated_trading_engine()
paper_trading = get_paper_trading_system()

# Create account
account = paper_trading.create_account(
    user_id="trader001",
    account_name="AI Momentum Strategy",
    initial_balance=50000.0
)

print(f"Created account: {account.account_id}")

# Define strategy
rules = RiskManagementRules(
    max_position_size_pct=8.0,
    max_total_exposure_pct=60.0,
    max_positions=8,
    min_confidence_threshold=0.70,
    stop_loss_pct=4.0,
    take_profit_pct=8.0
)

# Start automation
if engine.start_automation(account.account_id, risk_rules=rules):
    print("Automation started successfully")
    
    # Monitor for 1 hour
    for i in range(12):  # 12 * 5 minutes = 1 hour
        time.sleep(300)  # 5 minutes
        
        status = engine.get_automation_status(account.account_id)
        if status:
            print(f"\n[{i+1}/12] Trades: {status['metrics']['trades_executed']}, "
                  f"Opportunities: {status['metrics']['opportunities_evaluated']}")
    
    # Pause for review
    engine.pause_automation(account.account_id)
    print("\nAutomation paused for review")
    
    # Review performance
    account = paper_trading.get_account(account.account_id)
    print(f"P&L: ${account.total_pnl:.2f} ({account.total_pnl_pct:.2f}%)")
    
    # Resume or stop based on performance
    if account.total_pnl > 0:
        engine.resume_automation(account.account_id)
        print("Performance positive, resuming automation")
    else:
        engine.stop_automation(account.account_id)
        print("Performance negative, stopping automation")
```

## Next Steps

1. **UI Integration** - Add automation controls to Settings panel
2. **Trade Transparency** - Display automated trades in UI with reasoning
3. **Performance Dashboard** - Visualize automation metrics and P&L
4. **Advanced Strategies** - Add strategy templates (momentum, mean reversion, etc.)
5. **Backtesting** - Test strategies on historical data before live automation

## Files

- `automated_trading_engine.py` - Main engine implementation
- `paper_trading_system.py` - Paper trading backend
- `unbiased_market_analyzer.py` - Market opportunity scanner
- `AUTOMATED_TRADING_GUIDE.md` - This guide
