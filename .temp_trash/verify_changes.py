"""
Verification script to confirm all changes are present and working
Run this to verify before starting Streamlit
"""

print("="*70)
print("VERIFICATION SCRIPT - Checking All Implementations")
print("="*70)

# 1. Check paper_trading_ui.py has Breaking Trades
print("\n1. Checking paper_trading_ui.py...")
with open('paper_trading_ui.py', 'r') as f:
    ui_content = f.read()

checks = {
    "Breaking Trades tab": '"Breaking Trades"' in ui_content,
    "Performance Dashboard": 'Performance Dashboard & Grading' in ui_content,
    "Generate Breaking Trades button": 'Generate Breaking Trades' in ui_content,
    "Trade specifications section": 'Trade Specifications' in ui_content,
    "Confidence score display": 'CONFIDENCE' in ui_content,
}

for check, result in checks.items():
    status = "" if result else ""
    print(f"  {status} {check}")

# 2. Check modules import correctly
print("\n2. Checking module imports...")
try:
    from trading_system.risk_manager import AdvancedRiskManager
    print("   AdvancedRiskManager")
except Exception as e:
    print(f"   AdvancedRiskManager: {e}")

try:
    from trading_system.simulation_grader import SimulationGrader
    print("   SimulationGrader")
except Exception as e:
    print(f"   SimulationGrader: {e}")

try:
    from paper_trading_engine import PaperTradingEngine
    engine = PaperTradingEngine(100000)
    has_rm = hasattr(engine, 'risk_manager')
    has_gr = hasattr(engine, 'grader')
    print(f"   PaperTradingEngine (risk_manager: {has_rm}, grader: {has_gr})")
except Exception as e:
    print(f"   PaperTradingEngine: {e}")

try:
    from breaking_trades_generator import BreakingTradesGenerator
    print("   BreakingTradesGenerator")
except Exception as e:
    print(f"   BreakingTradesGenerator: {e}")

try:
    from advanced_backtester import AdvancedBacktester
    bt = AdvancedBacktester()
    has_run = hasattr(bt, 'run')
    print(f"   AdvancedBacktester (run method: {has_run})")
except Exception as e:
    print(f"   AdvancedBacktester: {e}")

# 3. Check forex pricing
print("\n3. Testing forex pricing...")
from paper_trading_ui import _get_live_price

test_pairs = ["EUR/USD", "USD/JPY", "GBP/USD"]
for pair in test_pairs:
    try:
        price = _get_live_price(pair)
        # Check if price is reasonable
        if 0.5 < price < 200:
            print(f"   {pair}: ${price:.4f}")
        else:
            print(f"   {pair}: ${price:.4f} (unusual)")
    except Exception as e:
        print(f"   {pair}: {e}")

# 4. Test grading system
print("\n4. Testing grading system...")
try:
    from trading_system.simulation_grader import SimulationGrader
    grader = SimulationGrader()

    # Test with good performance
    result = grader.calculate_grade(
        total_pnl_dollars=10000,
        total_return_pct=0.10,
        sharpe_ratio=2.0,
        max_drawdown_pct=0.05,
        win_rate=0.65,
        trade_count=50
    )
    print(f"   Good performance: Grade {result['final_grade']} (Score: {result['final_score']})")
    print(f"    - PnL Score: {result['breakdown']['pnl_score']} (60% weight)")

    # Test with poor performance
    result2 = grader.calculate_grade(
        total_pnl_dollars=-2000,
        total_return_pct=-0.02,
        sharpe_ratio=0.5,
        max_drawdown_pct=0.15,
        win_rate=0.40,
        trade_count=10
    )
    print(f"   Poor performance: Grade {result2['final_grade']} (Score: {result2['final_score']})")

except Exception as e:
    print(f"   Grading test failed: {e}")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
print("\nNext steps:")
print("1. Stop Streamlit (Ctrl+C)")
print("2. Clear cache: rm -rf ~/.streamlit/cache")
print("3. Restart: streamlit run main.py")
print("4. Navigate to Paper Trading page")
print("5. You should see 'Breaking Trades' as the first tab")
print("="*70)
