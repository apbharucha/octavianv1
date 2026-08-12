import sys, os, time
sys.path.insert(0, os.path.abspath('.'))
os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')

from paper_trading_system import get_paper_trading_system, TradeAction

pts = get_paper_trading_system()
uid = f'cov_{int(time.time())}'
acc = pts.create_account(user_id=uid, account_name='Cover Test', initial_balance=100000.0)

# Open a SHORT option (SELL 2 contracts)
r1 = pts.execute_option_trade(acc.account_id, 'AAPL', None, TradeAction.SELL, 2, 5.0)
print('OPEN SHORT:', r1 is not None)
print('POSITIONS AFTER OPEN:', [(p.side, p.quantity) for p in pts.get_option_positions(acc.account_id)])

# Close with COVER (should normalize to BUY)
r2 = pts.execute_option_trade(acc.account_id, 'AAPL', None, TradeAction.COVER, 2, 4.8)
print('CLOSE WITH COVER:', r2 is not None)
print('POSITIONS AFTER CLOSE:', [(p.side, p.quantity) for p in pts.get_option_positions(acc.account_id)])

# Balance: +1000 (sell premium) then -960 (buy back) => 100040
a = pts.get_account(acc.account_id)
print('BALANCE:', a.current_balance, '(expected 100040)')

pts.delete_account(acc.account_id)
print('CLEANUP OK')
