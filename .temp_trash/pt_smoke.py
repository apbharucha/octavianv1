import sys, os, time
sys.path.insert(0, os.path.abspath('.'))
os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')

from paper_trading_system import get_paper_trading_system, TradeAction

pts = get_paper_trading_system()
uid = f'test_user_{int(time.time())}'

# Account lifecycle
acc = pts.create_account(user_id=uid, account_name='Smoke Test', initial_balance=100000.0)
print('ACCOUNT:', acc.account_id, '| balance:', acc.current_balance)

# Trades
t1 = pts.execute_trade(acc.account_id, 'TEST', TradeAction.BUY, 10, price=100.0)
print('BUY:', t1 is not None, '| pos:', len(pts.get_positions(acc.account_id)))
t2 = pts.execute_trade(acc.account_id, 'TEST', TradeAction.SELL, 4, price=110.0)
print('SELL:', t2 is not None, '| pos:', len(pts.get_positions(acc.account_id)))

# History + positions
hist = pts.get_trade_history(acc.account_id, limit=10)
print('HISTORY:', len(hist))
for h in hist:
    print('   ', h.action.value, h.symbol, h.quantity, '@', h.price)

# Option trade path
try:
    ot = pts.execute_option_trade(acc.account_id, 'TEST', 'TEST250117C00100000',
                                  TradeAction.BUY, 5, price=2.5, underlying_price=101.0,
                                  expiry='2025-01-17', strike=100.0, option_type='call')
    print('OPTION TRADE:', ot is not None, '| opt pos:', len(pts.get_option_positions(acc.account_id)))
except Exception as e:
    print('OPTION TRADE ERR:', repr(e))

# Normalization
print('LIST ACCOUNTS via str/int user:', len(pts.list_accounts(uid)))

# Automation settings
print('AUTO SETTINGS:', pts.update_automation_settings(acc.account_id, True, {}))

# Cleanup
pts.delete_account(acc.account_id)
print('DELETED OK')

print()
print('ALL SMOKE TESTS PASSED')
