#!/bin/bash
#
# Octavian Market AI - Streamlit Startup Script
#
# Performance note: this script deliberately does NOT clear Python / Streamlit
# caches on every start. Clearing __pycache__ and .streamlit/cache forces a
# full cold re-import and re-fetch of every cached dataset on every restart,
# which makes the app load very slowly. Caches are only wiped when you pass
# --clean (e.g. after pulling new code where stale bytecode causes issues).

set -u

CLEAN=0
if [[ "${1:-}" == "--clean" ]]; then
  CLEAN=1
fi

echo "=============================================================================="
echo "OCTAVIAN MARKET AI - START"
echo "=============================================================================="

# 1. Stop any existing Streamlit processes (fresh environment)
echo "1. Stopping any running Streamlit processes..."
pkill -9 -f "streamlit run" 2>/dev/null
sleep 2
echo "Stopped"

# 2. Optionally clear caches (only with --clean; keeps normal restarts fast)
if [[ "$CLEAN" == "1" ]]; then
  echo "2. --clean requested: clearing Python and Streamlit caches..."
  rm -rf __pycache__ .streamlit/cache trading_system/__pycache__ 2>/dev/null
  find . -name "*.pyc" -delete 2>/dev/null
  echo "Caches cleared"
else
  echo "2. Caches preserved (fast restart). Use --clean to force a cold start."
fi

# 3. Verify key modules are present
echo "3. Verifying implementation..."
if grep -q "Breaking Trades" paper_trading_ui.py; then
  echo "Breaking Trades tab found"
else
  echo "WARNING: Breaking Trades tab not found"
fi

if grep -q "Performance Dashboard" paper_trading_ui.py; then
  echo "Performance Dashboard found"
else
  echo "WARNING: Performance Dashboard not found"
fi

# 4. Check modules load correctly
echo "4. Testing module imports..."
python3 -c"from paper_trading_system import PaperTradingSystem
from trading_system.risk_manager import AdvancedRiskManager
from trading_system.simulation_grader import SimulationGrader
from breaking_trades_generator import BreakingTradesGenerator
print('All modules import successfully')
"2>&1 | grep -v "WARNING"

echo ""
echo "=============================================================================="
echo "STARTING STREAMLIT"
echo "=============================================================================="
echo ""
echo "• Breaking Trades tab (high-confidence setups)"
echo "• Performance Dashboard with grading"
echo "• Risk Management system"
echo "• Enhanced PnL tracking"
echo ""
echo "Navigate to: Paper Trading"
echo "You should see 5 tabs: Breaking Trades | Equity | Options | Positions | History"
echo ""
echo "=============================================================================="
echo ""

# 5. Start Streamlit (headless)
streamlit run main.py --server.headless true
