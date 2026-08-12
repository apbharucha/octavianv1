import sys, os, time
sys.path.insert(0, os.path.abspath('.'))
os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')

from paper_trading_system import get_paper_trading_system, TradeAction

pts = get_paper_trading_system()
uid = f'opt_user_{int(time.time())}'
acc = pts.create_account(user_id=uid, account_name='Opt Test', initial_balance=100000.0)

# 1) Legacy-style minimal call (5 args, enum action) — the portfolio_analyzer pattern
r1 = pts.execute_option_trade(acc.account_id, 'AAPL', None, TradeAction.SELL, 2, 5.0)
print('MINIMAL CALL (enum action):', r1 is not None)

# 2) Full metadata call
r2 = pts.execute_option_trade(acc.account_id, 'AAPL', 'AAPL260515C00150000', 'BUY', 3, 5.5,
                              option_type='CALL', strike=150.0, expiration='2026-05-15')
print('FULL CALL (str action):', r2 is not None)

# 3) Invalid action rejected
r3 = pts.execute_option_trade(acc.account_id, 'AAPL', None, 'HODL', 1, 1.0)
print('INVALID ACTION REJECTED:', r3 is None)

# 4) Positions tracked
ops = pts.get_option_positions(acc.account_id)
print('OPTION POSITIONS:', len(ops))
for p in ops:
    print(f'   {p.symbol} | {p.option_type} {p.strike} {p.expiration} | {p.side} qty={p.quantity}')

# 5) Balance impact check
a = pts.get_account(acc.account_id)
print('BALANCE:', a.current_balance)
print('EXPECTED 100k - 2*5*100 - 3*5.5*100 =', 100000 - 1000 - 1650)

pts.delete_account(acc.account_id)
print('CLEANUP OK')
