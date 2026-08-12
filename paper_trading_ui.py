"""
Paper Trading UI Components

Provides comprehensive UI for paper trading system:
1. Account management (create, switch, view)
2. Position tracking and P&L display
3. Trade history with AI reasoning
4. Automated trading controls
5. Performance charts and metrics

Author: APB - Octavian Team
"""

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import json
import time

from paper_trading_system import (
    get_paper_trading_system,
    PaperTradingAccount,
    TradeAction,
    AccountStatus
)
from automated_trading_engine import (
    get_automated_trading_engine,
    RiskManagementRules,
    AutomationStatus
)
from data_sources import get_fx, get_futures_proxy, get_stock


def show_paper_trading_dashboard():
    """Main paper trading dashboard."""
    st.title(" Paper Trading Dashboard")
    
    # --- Handle Breaking Trade Import ---
    imported_trade = st.session_state.pop("import_breaking_trade", None)
    if imported_trade:
        st.session_state["_pending_import_trade"] = imported_trade
    pending_import = st.session_state.get("_pending_import_trade")
    if pending_import:
        sym = pending_import["symbol"]
        direction = pending_import["direction"]
        action_str = "BUY" if direction == "LONG" else "SHORT"
        with st.container():
            st.markdown(
                f'<div style="background:#1a2332;border:2px solid #DAA520;border-radius:8px;padding:16px;margin-bottom:16px;">'
                f'<div style="font-size:1.2rem;font-weight:700;color:#DAA520;">Imported Breaking Trade: {sym}</div>'
                f'<div style="color:#ccc;margin-top:4px;">{action_str} | {pending_import.get("setup_type","")} | '
                f'Confidence: {pending_import.get("confidence",0):.0f}% | R:R {pending_import.get("risk_reward",0):.1f}:1</div>'
                f'<div style="color:#888;margin-top:4px;">Entry ${pending_import.get("entry_price",0):,.2f} | '
                f'Stop ${pending_import.get("stop_loss",0):,.2f} | '
                f'TP1 ${pending_import.get("take_profit_1",0):,.2f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.info(f"**Reasoning:** {pending_import.get('reasoning', 'Breaking trade import')}")
    
    # Initialize systems
    paper_trading = get_paper_trading_system()
    automation_engine = get_automated_trading_engine()
    
    # Get or create user ID
    if 'user_id' not in st.session_state:
        st.session_state.user_id = "default_user"
    
    user_id = st.session_state.get("user_id", "default_user")
    
    # Get user accounts
    accounts = paper_trading.list_accounts(user_id)
    
    # Account selection/creation
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        if accounts:
            account_options = {
                f"{acc.account_name} (${acc.current_balance:,.2f})": acc.account_id
                for acc in accounts
            }
            selected_name = st.selectbox(
                "Select Account",
                options=list(account_options.keys()),
                key="account_selector"
            )
            selected_account_id = account_options[selected_name]
        else:
            st.info("No paper trading accounts yet. Create one to get started!")
            selected_account_id = None
    
    with col2:
        if st.button(" New Account", use_container_width=True):
            st.session_state.show_create_account = True
    
    with col3:
        if accounts and selected_account_id:
            if st.button(" Delete", use_container_width=True, type="secondary"):
                st.session_state.show_delete_confirm = True
                st.session_state.delete_account_id = selected_account_id
    
    # Create account dialog
    if st.session_state.get('show_create_account', False):
        with st.expander("Create New Account", expanded=True):
            account_name = st.text_input("Account Name", value="My Strategy")
            initial_balance = st.number_input(
                "Initial Balance ($)",
                min_value=1000.0,
                max_value=10000000.0,
                value=100000.0,
                step=10000.0
            )
            
            col_create1, col_create2 = st.columns(2)
            with col_create1:
                if st.button("Create", use_container_width=True):
                    new_account = paper_trading.create_account(
                        user_id=user_id,
                        account_name=account_name,
                        initial_balance=initial_balance
                    )
                    if new_account:
                        st.success(f"Created account: {account_name}")
                        st.session_state.show_create_account = False
                        st.rerun()
                    else:
                        st.error("Failed to create account")
            
            with col_create2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_create_account = False
                    st.rerun()
    
    # Delete account confirmation dialog
    if st.session_state.get('show_delete_confirm', False):
        delete_account_id = st.session_state.get('delete_account_id')
        delete_account = paper_trading.get_account(delete_account_id)
        
        if delete_account:
            with st.expander("Confirm Account Deletion", expanded=True):
                st.warning(
                    f"Are you sure you want to delete '{delete_account.account_name}'?\n\n"
                    f"This will permanently delete:\n"
                    f"- Account balance: ${delete_account.current_balance:,.2f}\n"
                    f"- All positions\n"
                    f"- All trade history\n\n"
                    f"This action cannot be undone."
                )
                
                col_del1, col_del2 = st.columns(2)
                with col_del1:
                    if st.button("Yes, Delete", use_container_width=True, type="primary"):
                        if paper_trading.delete_account(delete_account_id):
                            st.success(f"Deleted account: {delete_account.account_name}")
                            st.session_state.show_delete_confirm = False
                            st.session_state.delete_account_id = None
                            st.rerun()
                        else:
                            st.error("Failed to delete account")
                
                with col_del2:
                    if st.button("Cancel", use_container_width=True):
                        st.session_state.show_delete_confirm = False
                        st.session_state.delete_account_id = None
                        st.rerun()
    
    if not selected_account_id:
        return
    
    # Get selected account details
    account = paper_trading.get_account(selected_account_id)
    if not account:
        st.error("Account not found")
        return
    
    # --- Execute pending import trade ---
    pending_import = st.session_state.get("_pending_import_trade")
    if pending_import:
        sym = pending_import["symbol"]
        direction = pending_import["direction"]
        action_str = "BUY" if direction == "LONG" else "SHORT"
        
        col_imp1, col_imp2, col_imp3 = st.columns([2, 1, 1])
        with col_imp1:
            imp_qty = st.number_input(
                f"Quantity for {sym}",
                min_value=1, value=10, step=1,
                key="import_trade_qty",
            )
        with col_imp2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Execute Imported Trade", type="primary", key="exec_import_trade"):
                try:
                    live_price = _get_live_price_safe(sym)
                    if not live_price or live_price <= 0:
                        live_price = pending_import.get("entry_price", 0)
                    
                    if live_price and live_price > 0:
                        trade_action = TradeAction[action_str]
                        trade = paper_trading.execute_trade(
                            account_id=account.account_id,
                            symbol=sym,
                            action=trade_action,
                            quantity=int(imp_qty),
                            price=float(live_price),
                            strategy_name=f"Breaking Trade: {pending_import.get('setup_type', 'Import')}",
                            ai_reasoning=(
                                f"Imported from Breaking Trades | "
                                f"Confidence: {pending_import.get('confidence', 0):.0f}% | "
                                f"R:R: {pending_import.get('risk_reward', 0):.1f}:1 | "
                                f"{pending_import.get('reasoning', '')}"
                            ),
                        )
                        if trade:
                            st.success(
                                f"Trade executed: {action_str} {imp_qty} {sym} @ ${live_price:,.2f}"
                            )
                            st.session_state.pop("_pending_import_trade", None)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to execute trade")
                    else:
                        st.error(f"Cannot fetch price for {sym}")
                except Exception as e:
                    st.error(f"Error executing import trade: {e}")
        with col_imp3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Dismiss", key="dismiss_import_trade"):
                st.session_state.pop("_pending_import_trade", None)
                st.rerun()
        
        st.markdown("---")
    
    # Main dashboard tabs
    tabs = st.tabs([
        "Overview",
        "Positions",
        "Options Portfolio",
        "Trade History",
        "Automated Trading",
        "Performance"
    ])
    
    with tabs[0]:
        show_account_overview(account, paper_trading)
    
    with tabs[1]:
        show_positions(account, paper_trading)
        
    with tabs[2]:
        show_options_portfolio(account, paper_trading)
    
    with tabs[3]:
        show_trade_history(account, paper_trading)
    
    with tabs[4]:
        show_automation_controls(account, automation_engine)
    
    with tabs[5]:
        show_performance_charts(account, paper_trading)


def show_account_overview(account: PaperTradingAccount, paper_trading):
    """Display account overview with key metrics and theoretical (liquidation) values."""
    
    # Get positions for calculations
    positions = paper_trading.get_positions(account.account_id)
    option_positions = paper_trading.get_option_positions(account.account_id)
    
    unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
    # Add unrealized from options (simplified for now as already calculated in pos if available)
    unrealized_pnl += sum(pos.unrealized_pnl for pos in option_positions)
    
    # Calculate options market value
    options_value = sum(pos.current_price * pos.quantity * 100 for pos in option_positions)
    
    theoretical_value = account.current_balance + unrealized_pnl + sum(pos.entry_price * pos.quantity * 100 for pos in option_positions)
    # Actually simpler: balance already reflects cash spent. 
    # Theoretical value = Cash + Market Value of Stock + Market Value of Options
    stock_value = sum(pos.current_price * pos.quantity for pos in positions)
    theoretical_value = account.current_balance + stock_value + options_value
    
    # Total P&L if closed now = Realized so far + Unrealized from open positions
    theoretical_pnl = account.realized_pnl + unrealized_pnl
    theoretical_pnl_pct = (theoretical_pnl / account.initial_balance * 100) if account.initial_balance > 0 else 0
    
    # Header metrics (Existing Realized state)
    st.markdown("### Realized Account State")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Balance",
            f"${account.current_balance:,.2f}",
            delta=f"${account.realized_pnl:+.2f} Realized"
        )
    
    with col2:
        st.metric(
            "Realized P&L",
            f"${account.realized_pnl:,.2f}",
            delta=f"{(account.realized_pnl/account.initial_balance*100):+.2f}%"
        )
    
    with col3:
        total_exposure = sum(abs(pos.quantity * pos.current_price) for pos in positions)
        exposure_pct = (total_exposure / account.current_balance * 100) if account.current_balance > 0 else 0
        st.metric(
            "Total Exposure",
            f"${total_exposure:,.2f}",
            delta=f"{exposure_pct:.1f}%"
        )
    
    with col4:
        st.metric(
            "Open Positions",
            len(positions)
        )
    
    # Theoretical Liquidation Row
    st.markdown("### Theoretical Liquidation State (If Closed Now)")
    tcol1, tcol2 = st.columns(2)
    
    with tcol1:
        st.metric(
            "Total Account Balance (Liquidated)",
            f"${theoretical_value:,.2f}",
            delta=f"${unrealized_pnl:+.2f} vs Realized",
            delta_color="normal"
        )
    
    with tcol2:
        st.metric(
            "Total P&L (Liquidated)",
            f"${theoretical_pnl:,.2f}",
            delta=f"{theoretical_pnl_pct:+.2f}% vs Initial",
            delta_color="normal"
        )
        
    st.markdown("---")
    
    # Account details
    col_detail1, col_detail2 = st.columns(2)
    
    with col_detail1:
        st.subheader("Account Details")
        st.write(f"**Account ID:** `{account.account_id}`")
        st.write(f"**Name:** {account.account_name}")
        st.write(f"**Status:** {account.status.value}")
        st.write(f"**Created:** {account.created_at.strftime('%Y-%m-%d %H:%M')}")
    
    with col_detail2:
        st.subheader("Quick Actions")
        
        if st.button(" Manual Trade", use_container_width=True):
            st.session_state.show_manual_trade = True
        
        if st.button(" Export History", use_container_width=True):
            trades = paper_trading.get_trade_history(account.account_id, limit=1000)
            if trades:
                df = pd.DataFrame([{
                    'Timestamp': t.timestamp,
                    'Symbol': t.symbol,
                    'Action': t.action.value,
                    'Quantity': t.quantity,
                    'Price': t.price,
                    'Total Value': t.total_value,
                    'Strategy': t.strategy_name
                } for t in trades])
                
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"trades_{account.account_id}.csv",
                    "text/csv",
                    use_container_width=True
                )
    
    # Manual trade dialog
    if st.session_state.get('show_manual_trade', False):
        show_manual_trade_dialog(account, paper_trading)


def show_positions(account: PaperTradingAccount, paper_trading):
    """Display open positions."""
    st.subheader("Open Positions")
    
    positions = paper_trading.get_positions(account.account_id)
    
    if not positions:
        st.info("No open positions")
        return
    
    # Create positions dataframe
    positions_data = []
    for pos in positions:
        positions_data.append({
            'Symbol': pos.symbol,
            'Side': pos.side,
            'Quantity': pos.quantity,
            'Entry Price': f"${pos.entry_price:.2f}",
            'Current Price': f"${pos.current_price:.2f}",
            'Unrealized P&L': f"${pos.unrealized_pnl:.2f}",
            'P&L %': f"{pos.unrealized_pnl_pct:.2f}%",
            'Entry Time': pos.entry_time.strftime('%Y-%m-%d %H:%M')
        })
    
    df = pd.DataFrame(positions_data)
    
    # Style the dataframe
    def color_pnl(val):
        if isinstance(val, str) and '$' in val:
            num = float(val.replace('$', '').replace(',', ''))
            color = 'green' if num >= 0 else 'red'
            return f'color: {color}'
        elif isinstance(val, str) and '%' in val:
            num = float(val.replace('%', ''))
            color = 'green' if num >= 0 else 'red'
            return f'color: {color}'
        return ''
    
    styled_df = df.style.applymap(color_pnl, subset=['Unrealized P&L', 'P&L %'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Position actions
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_symbol = st.selectbox(
            "Select Position to Close",
            options=[pos.symbol for pos in positions]
        )
    
    with col2:
        if st.button("Close Position", use_container_width=True):
            # Find position
            pos = next((p for p in positions if p.symbol == selected_symbol), None)
            if pos:
                # Determine close action
                action = TradeAction.SELL if pos.side == 'LONG' else TradeAction.COVER
                
                # Execute close
                trade = paper_trading.execute_trade(
                    account.account_id,
                    selected_symbol,
                    action,
                    pos.quantity,
                    pos.current_price,
                    strategy_name="Manual Close",
                    ai_reasoning="User manually closed position"
                )
                if trade:
                    st.success(f"Successfully closed {selected_symbol}")
                    st.rerun()
                else:
                    st.error("Failed to close position")


def show_options_portfolio(account, paper_trading):
    """Display options-specific portfolio and greeks."""
    st.subheader("Options Portfolio")
    
    options = paper_trading.get_option_positions(account.account_id)
    if not options:
        st.info("No open options positions recorded.")
        return
        
    import options_engine
    import plotly.graph_objects as go
    engine = options_engine.get_options_engine()
    
    # Options summary with Greeks
    st.markdown("### Open Contracts & Greeks")
    
    rows = []
    total_delta = 0
    total_theta = 0
    
    for op in options:
        # In a real scenario, we'd fetch live IV and price
        # Using placeholders for the demo to show the engine in action
        iv = 0.30
        T = 0.1 # 10% of a year
        greeks = engine.black_scholes(op.current_price, op.strike, T, iv, op.option_type.lower())
        
        total_delta += greeks['delta'] * op.quantity * 100
        total_theta += greeks['theta'] * op.quantity * 100
        
        rows.append({
            "Contract": op.symbol,
            "Type": op.option_type,
            "Qty": op.quantity,
            "Strike": op.strike,
            "Delta": f"{greeks['delta']:.3f}",
            "Gamma": f"{greeks['gamma']:.4f}",
            "Theta": f"{greeks['theta']:.3f}",
            "Vega": f"{greeks['vega']:.3f}",
            "Value": f"${op.current_price * op.quantity * 100:,.2f}"
        })
    
    st.table(rows)
    
    # Portfolio exposure
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.metric("Net Portfolio Delta", f"{total_delta:.2f}", help="Shares of underlying equivalent")
    with exp_col2:
        st.metric("Daily Theta Decay", f"${total_theta:.2f}", help="Expected daily value erosion")
        
    # Strategy Builder / Payoff Simulator
    st.markdown("---")
    st.markdown("###  Strategy Payoff Simulator")
    
    selected_contract = st.selectbox("Select Strategy to Visualize", [o.symbol for o in options] + ["Custom Strategy"])
    
    if selected_contract == "Custom Strategy":
        st.caption("Custom strategy builder coming soon...")
    else:
        op = next(o for o in options if o.symbol == selected_contract)
        strategy = [{
            'strike': op.strike, 
            'type': op.option_type.lower(), 
            'side': 1 if op.side == 'LONG' else -1, 
            'cost': op.entry_price
        }]
        
        payoff = engine.get_strategy_pnl_map(strategy)
        
        fig = go.Figure()
        
        # Add payoff line
        fig.add_trace(go.Scatter(
            x=payoff['prices'], 
            y=payoff['pnls'], 
            mode='lines',
            line=dict(color='#00FFA3', width=3),
            name='P&L at Expiry',
            fill='tozeroy',
            fillcolor='rgba(0, 255, 163, 0.1)'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        
        # Add breakeven points
        for be in payoff['breakeven']:
            fig.add_vline(x=be, line_dash="dot", line_color="#FFB800", annotation_text=f"BE: {be:.2f}")
            
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Underlying Price ($)",
            yaxis_title="Profit / Loss ($)",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_trade_history(account: PaperTradingAccount, paper_trading):
    """Display trade history with AI reasoning."""
    st.subheader("Trade History")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        limit = st.selectbox("Show Last", [10, 25, 50, 100, 500], index=1)
    
    with col2:
        filter_symbol = st.text_input("Filter by Symbol", "")
    
    with col3:
        filter_action = st.selectbox("Filter by Action", ["All", "BUY", "SELL", "SHORT", "COVER"])
    
    # Get trades
    trades = paper_trading.get_trade_history(account.account_id, limit=limit)
    
    if not trades:
        st.info("No trades yet")
        return
    
    # Apply filters
    if filter_symbol:
        trades = [t for t in trades if filter_symbol.upper() in t.symbol.upper()]
    
    if filter_action != "All":
        trades = [t for t in trades if t.action.value == filter_action]
    
    # Display trades
    for trade in trades:
        with st.expander(
            f"{trade.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - "
            f"{trade.action.value} {trade.quantity} {trade.symbol} @ ${trade.price:.2f}"
        ):
            col_trade1, col_trade2 = st.columns(2)
            
            with col_trade1:
                st.write(f"**Trade ID:** `{trade.trade_id}`")
                st.write(f"**Symbol:** {trade.symbol}")
                st.write(f"**Action:** {trade.action.value}")
                st.write(f"**Quantity:** {trade.quantity}")
                st.write(f"**Price:** ${trade.price:.2f}")
                st.write(f"**Total Value:** ${trade.total_value:.2f}")
            
            with col_trade2:
                st.write(f"**Strategy:** {trade.strategy_name}")
                st.write(f"**Timestamp:** {trade.timestamp}")
                
                if trade.market_context:
                    st.write("**Market Context:**")
                    st.json(trade.market_context)
            
            if trade.ai_reasoning:
                st.markdown("** AI Reasoning:**")
                st.text(trade.ai_reasoning)


def show_automation_controls(account: PaperTradingAccount, automation_engine):
    """Display automation controls and status."""
    st.subheader("Automated Trading")
    
    # Get automation status
    status = automation_engine.get_automation_status(account.account_id)
    
    # Status display
    if status:
        status_color = {
            'RUNNING': '',
            'PAUSED': '',
            'STOPPED': '',
            'ERROR': ''
        }
        
        st.markdown(f"### {status_color.get(status['status'], '')} Status: {status['status']}")
        
        # Check if in full automation mode
        is_full_auto = st.session_state.get(f"full_auto_{account.account_id}", False)
        if is_full_auto:
            st.info(" **FULL AUTOMATION MODE ACTIVE**  AI has complete control")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Trades Executed", status['metrics']['trades_executed'])
        
        with col2:
            st.metric("Opportunities Evaluated", status['metrics']['opportunities_evaluated'])
        
        with col3:
            st.metric("Total P&L", f"${status['metrics']['total_pnl']:.2f}")
        
        with col4:
            st.metric("Win Rate", f"{status['metrics']['win_rate']:.1%}")
        
        st.caption(f"Started: {status['metrics']['started_at']}")
    else:
        st.info("Automation not running")
    
    st.markdown("---")
    
    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
    
    # Safely check status values with proper boolean conversion
    is_running = bool(status and status.get('is_running', False))
    is_paused = bool(status and status.get('is_paused', False))
    is_stopped = not status or status.get('status') == 'STOPPED'
    
    with col_ctrl1:
        if st.button(" Start", use_container_width=True, disabled=is_running):
            if is_stopped:
                # Show automation mode selection
                st.session_state.show_automation_mode_select = True
            else:
                st.warning("Automation already running")
    
    with col_ctrl2:
        if st.button(" Pause", use_container_width=True, disabled=not is_running or is_paused):
            if automation_engine.pause_automation(account.account_id):
                st.success("Automation paused")
                st.rerun()
    
    with col_ctrl3:
        if st.button(" Resume", use_container_width=True, disabled=not is_paused):
            if automation_engine.resume_automation(account.account_id):
                st.success("Automation resumed")
                st.rerun()
    
    with col_ctrl4:
        if st.button(" Stop", use_container_width=True, disabled=is_stopped):
            if automation_engine.stop_automation(account.account_id):
                st.success("Automation stopped")
                st.session_state.pop(f"full_auto_{account.account_id}", None)
                st.rerun()
    
    # Automation mode selection dialog
    if st.session_state.get('show_automation_mode_select', False):
        show_automation_mode_selection(account, automation_engine)
    
    # Risk configuration dialog (only for custom mode)
    if st.session_state.get('show_risk_config', False):
        show_risk_configuration_dialog(account, automation_engine)


def show_automation_mode_selection(account: PaperTradingAccount, automation_engine):
    """Show automation mode selection: Custom vs Full Auto."""
    with st.expander(" Select Automation Mode", expanded=True):
        st.markdown("""
        Choose how you want the AI to trade for you:
        
        **Custom Automation**  You set the rules (risk limits, position sizes, confidence thresholds)
        
        **Full Automation**  AI has complete control with minimal constraints (maximum profit focus)
        """)
        
        mode_tabs = st.tabs([" Custom Automation", " Full Automation"])
        
        with mode_tabs[0]:
            st.markdown("### Custom Automation")
            st.caption("Configure risk parameters and let the AI trade within your boundaries")
            
            col_custom1, col_custom2 = st.columns(2)
            
            with col_custom1:
                st.markdown("**Position Sizing**")
                max_position_size = st.slider(
                    "Max Position Size (%)",
                    min_value=1.0,
                    max_value=25.0,
                    value=10.0,
                    step=1.0,
                    help="Maximum % of account per position"
                )
                
                max_exposure = st.slider(
                    "Max Total Exposure (%)",
                    min_value=10.0,
                    max_value=100.0,
                    value=80.0,
                    step=5.0,
                    help="Maximum % of account in all positions"
                )
                
                max_positions = st.slider(
                    "Max Positions",
                    min_value=1,
                    max_value=20,
                    value=10,
                    help="Maximum number of open positions"
                )
            
            with col_custom2:
                st.markdown("**Risk Controls**")
                min_confidence = st.slider(
                    "Min Confidence (%)",
                    min_value=50,
                    max_value=95,
                    value=65,
                    help="Minimum confidence to execute trade"
                )
                
                stop_loss = st.slider(
                    "Stop Loss (%)",
                    min_value=1.0,
                    max_value=20.0,
                    value=5.0,
                    step=0.5,
                    help="Stop loss percentage from entry"
                )
                
                take_profit = st.slider(
                    "Take Profit (%)",
                    min_value=2.0,
                    max_value=50.0,
                    value=10.0,
                    step=1.0,
                    help="Take profit percentage from entry"
                )
            
            col_start_custom1, col_start_custom2 = st.columns(2)
            
            with col_start_custom1:
                if st.button("Start Custom Automation", type="primary", use_container_width=True):
                    risk_rules = RiskManagementRules(
                        max_position_size_pct=max_position_size,
                        max_total_exposure_pct=max_exposure,
                        max_positions=max_positions,
                        min_confidence_threshold=min_confidence / 100,
                        max_loss_per_trade_pct=2.0,
                        profit_target_multiplier=2.0,
                        stop_loss_pct=stop_loss,
                        take_profit_pct=take_profit
                    )
                    
                    if automation_engine.start_automation(account.account_id, risk_rules):
                        st.session_state[f"full_auto_{account.account_id}"] = False
                        st.session_state.show_automation_mode_select = False
                        st.success("Custom automation started!")
                        st.rerun()
                    else:
                        st.error("Failed to start automation")
            
            with col_start_custom2:
                if st.button("Cancel", use_container_width=True, key="cancel_custom_auto"):
                    st.session_state.show_automation_mode_select = False
                    st.rerun()
        
        with mode_tabs[1]:
            st.markdown("###  Full Automation Mode")
            st.caption("AI operates with complete autonomy  maximum profit focus, minimal constraints")
            
            st.warning("""
             **WARNING: HIGH AUTONOMY MODE**
            
            In Full Automation mode:
            - AI decides all position sizes (optimized for profit potential)
            - AI sets its own stop losses and take profits
            - AI can trade any asset in the universe
            - AI can take both long and short positions
            - AI adjusts strategy based on market conditions
            - Only hard limit: cannot exceed account balance
            
            **Use only if you:**
            - Trust the AI models completely
            - Accept high risk / high reward approach
            - Want maximum profit potential
            - Are comfortable with autonomous decisions
            """)
            
            st.markdown("---")
            
            st.markdown("### Full Auto Configuration")
            
            full_auto_col1, full_auto_col2 = st.columns(2)
            
            with full_auto_col1:
                st.markdown("**AI Autonomy Level**")
                autonomy_level = st.select_slider(
                    "Autonomy",
                    options=["Conservative", "Moderate", "Aggressive", "Maximum"],
                    value="Aggressive",
                    help="How aggressive the AI can be with position sizing and risk"
                )
                
                autonomy_map = {
                    "Conservative": {"max_pos": 15.0, "max_exp": 60.0, "min_conf": 0.60},
                    "Moderate": {"max_pos": 25.0, "max_exp": 80.0, "min_conf": 0.50},
                    "Aggressive": {"max_pos": 40.0, "max_exp": 95.0, "min_conf": 0.35},
                    "Maximum": {"max_pos": 50.0, "max_exp": 100.0, "min_conf": 0.25}
                }
                
                params = autonomy_map[autonomy_level]
                
                st.metric("Max Position Size", f"{params['max_pos']}%")
                st.metric("Max Total Exposure", f"{params['max_exp']}%")
                st.metric("Min Confidence", f"{params['min_conf']:.0%}")
            
            with full_auto_col2:
                st.markdown("**Emergency Controls**")
                
                emergency_stop_loss = st.slider(
                    "Emergency Stop Loss (Account Level)",
                    min_value=5.0,
                    max_value=30.0,
                    value=15.0,
                    step=1.0,
                    help="Stop ALL trading if account drops this % from peak"
                )
                
                daily_loss_limit = st.slider(
                    "Daily Loss Limit (%)",
                    min_value=2.0,
                    max_value=15.0,
                    value=5.0,
                    step=0.5,
                    help="Stop trading for the day if losses exceed this"
                )
                
                enable_shorts = st.checkbox(
                    "Allow Short Positions",
                    value=True,
                    help="Allow AI to short sell (not just long positions)"
                )
            
            st.markdown("---")
            
            # Acknowledgment checkbox
            acknowledge = st.checkbox(
                "I understand that Full Automation gives the AI complete control over trading decisions",
                value=False
            )
            
            col_start_full1, col_start_full2 = st.columns(2)
            
            with col_start_full1:
                if st.button(
                    " Start Full Automation", 
                    type="primary", 
                    use_container_width=True,
                    disabled=not acknowledge
                ):
                    # Create FULL AUTO risk rules (very loose constraints)
                    risk_rules = RiskManagementRules(
                        max_position_size_pct=params['max_pos'],
                        max_total_exposure_pct=params['max_exp'],
                        max_positions=50,  # AI can have many positions
                        min_confidence_threshold=params['min_conf'],
                        max_loss_per_trade_pct=10.0,  # Higher individual trade risk
                        profit_target_multiplier=1.5,  # AI sets its own targets
                        stop_loss_pct=15.0,  # Wider stops
                        take_profit_pct=25.0,  # Let profits run
                        # Full auto specific
                        allow_shorts=enable_shorts,
                        emergency_stop_loss_pct=emergency_stop_loss,
                        daily_loss_limit_pct=daily_loss_limit,
                        full_auto_mode=True
                    )
                    
                    if automation_engine.start_automation(account.account_id, risk_rules):
                        st.session_state[f"full_auto_{account.account_id}"] = True
                        st.session_state.show_automation_mode_select = False
                        st.success(" Full Automation activated! AI now has complete control.")
                        st.rerun()
                    else:
                        st.error("Failed to start full automation")
            
            with col_start_full2:
                if st.button("Cancel", use_container_width=True, key="cancel_full_auto"):
                    st.session_state.show_automation_mode_select = False
                    st.rerun()


def show_risk_configuration_dialog(account: PaperTradingAccount, automation_engine):
    """Show risk management configuration dialog (legacy - now handled in mode selection)."""
    # This function is now mostly deprecated in favor of show_automation_mode_selection
    # Keep for backward compatibility
    with st.expander(" Risk Management Configuration", expanded=True):
        st.write("Configure risk parameters for automated trading")
        
        col_risk1, col_risk2 = st.columns(2)
        
        with col_risk1:
            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=1.0,
                max_value=25.0,
                value=10.0,
                step=1.0,
                help="Maximum % of account per position"
            )
            
            max_exposure = st.slider(
                "Max Total Exposure (%)",
                min_value=10.0,
                max_value=100.0,
                value=80.0,
                step=5.0,
                help="Maximum % of account in all positions"
            )
            
            max_positions = st.slider(
                "Max Positions",
                min_value=1,
                max_value=20,
                value=10,
                help="Maximum number of open positions"
            )
            
            min_confidence = st.slider(
                "Min Confidence (%)",
                min_value=50,
                max_value=95,
                value=65,
                help="Minimum confidence to execute trade"
            )
        
        with col_risk2:
            stop_loss = st.slider(
                "Stop Loss (%)",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=0.5,
                help="Stop loss percentage from entry"
            )
            
            take_profit = st.slider(
                "Take Profit (%)",
                min_value=2.0,
                max_value=50.0,
                value=10.0,
                step=1.0,
                help="Take profit percentage from entry"
            )
            
            max_loss_per_trade = st.slider(
                "Max Loss Per Trade (%)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="Maximum loss per trade as % of account"
            )
            
            risk_reward = st.slider(
                "Risk/Reward Ratio",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="Minimum risk/reward ratio"
            )
        
        col_start1, col_start2 = st.columns(2)
        
        with col_start1:
            if st.button("Start Automation", use_container_width=True, key="start_legacy_auto"):
                # Create risk rules
                risk_rules = RiskManagementRules(
                    max_position_size_pct=max_position_size,
                    max_total_exposure_pct=max_exposure,
                    max_positions=max_positions,
                    min_confidence_threshold=min_confidence / 100,
                    max_loss_per_trade_pct=max_loss_per_trade,
                    profit_target_multiplier=risk_reward,
                    stop_loss_pct=stop_loss,
                    take_profit_pct=take_profit
                )
                
                # Start automation
                if automation_engine.start_automation(account.account_id, risk_rules):
                    st.success("Automation started successfully!")
                    st.session_state.show_risk_config = False
                    st.rerun()
                else:
                    st.error("Failed to start automation")
        
        with col_start2:
            if st.button("Cancel", use_container_width=True, key="cancel_legacy_auto"):
                st.session_state.show_risk_config = False
                st.rerun()


def show_performance_charts(account: PaperTradingAccount, paper_trading):
    """Display performance charts and analytics."""
    st.subheader("Performance Analytics")
    
    # Get trade history
    trades = paper_trading.get_trade_history(account.account_id, limit=500)
    
    if not trades:
        st.info("No trades to analyze yet")
        return
    
    # Create performance dataframe
    df_trades = pd.DataFrame([{
        'timestamp': t.timestamp,
        'symbol': t.symbol,
        'action': t.action.value,
        'quantity': t.quantity,
        'price': t.price,
        'total_value': t.total_value
    } for t in trades])
    
    # Calculate cumulative P&L (simplified)
    df_trades['pnl'] = 0.0  # Would need to match buys/sells for actual P&L
    df_trades['cumulative_pnl'] = df_trades['pnl'].cumsum()
    
    # P&L Chart
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=df_trades['timestamp'],
        y=df_trades['cumulative_pnl'],
        mode='lines',
        name='Cumulative P&L',
        line=dict(color='#DAA520', width=2)
    ))
    
    fig_pnl.update_layout(
        title="Cumulative P&L Over Time",
        xaxis_title="Date",
        yaxis_title="P&L ($)",
        height=400
    )
    
    st.plotly_chart(fig_pnl, use_container_width=True)
    
    # Trade distribution
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Actions distribution
        action_counts = df_trades['action'].value_counts()
        fig_actions = go.Figure(data=[go.Pie(
            labels=action_counts.index,
            values=action_counts.values,
            hole=0.4
        )])
        fig_actions.update_layout(title="Trade Actions Distribution", height=300)
        st.plotly_chart(fig_actions, use_container_width=True)
    
    with col_chart2:
        # Top symbols
        symbol_counts = df_trades['symbol'].value_counts().head(10)
        fig_symbols = go.Figure(data=[go.Bar(
            x=symbol_counts.values,
            y=symbol_counts.index,
            orientation='h',
            marker_color='#DAA520'
        )])
        fig_symbols.update_layout(title="Most Traded Symbols", height=300)
        st.plotly_chart(fig_symbols, use_container_width=True)


def show_manual_trade_dialog(account: PaperTradingAccount, paper_trading):
    """Show manual trade entry dialog. Trades always execute at live market price."""
    with st.expander(" Manual Trade Entry", expanded=True):
        col_trade1, col_trade2 = st.columns(2)
        
        with col_trade1:
            asset_type = st.radio("Asset Type", ["Stock", "Option", "Future"], horizontal=True, key="manual_asset_type")
            symbol = st.text_input("Symbol", value="AAPL" if asset_type != "Future" else "ES=F", key="manual_trade_symbol").strip().upper()
            
            if asset_type == "Stock":
                action = st.selectbox("Action", ["BUY", "SELL", "SHORT", "COVER"])
                quantity = st.number_input("Quantity (Shares)", min_value=1, value=10, step=1)
            elif asset_type == "Future":
                action = st.selectbox("Action", ["BUY", "SELL", "SHORT", "COVER"])
                quantity = st.number_input("Quantity (Contracts)", min_value=1, value=1, step=1)
            else:
                action = st.selectbox("Action", ["BUY", "SELL"])
                quantity = st.number_input("Quantity (Contracts)", min_value=1, value=1, step=1)
                opt_type = st.selectbox("Option Type", ["CALL", "PUT"])
                
                # Fetch typical strikes/expiry for the user or allow manual entry
                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    strike = st.number_input("Strike Price", min_value=0.01, value=150.0, step=1.0)
                with col_opt2:
                    # Default expiry: next Friday
                    from datetime import date
                    import datetime as dt
                    next_friday = date.today() + dt.timedelta((4 - date.today().weekday()) % 7)
                    if next_friday == date.today(): next_friday += dt.timedelta(7)
                    expiry = st.date_input("Expiration", value=next_friday)
        
        with col_trade2:
            # Fetch and display live price  this is the ONLY price the user can trade at
            live_price = None
            price_error = None
            
            if symbol:
                try:
                    # Try multiple methods to get live price
                    live_price = _get_live_price_safe(symbol)
                    
                    if not live_price or live_price <= 0:
                        price_error = f"Could not fetch valid price for {symbol}"
                except Exception as e:
                    price_error = f"Error fetching price: {str(e)}"
            
            if live_price and live_price > 0:
                st.markdown(
                    f'<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;'
                    f'padding:16px;margin:8px 0;text-align:center;">'
                    f'<div style="color:#8b949e;font-size:0.75rem;letter-spacing:1px;">LIVE MARKET PRICE</div>'
                    f'<div style="color:#00ff88;font-size:1.8rem;font-weight:700;">${live_price:,.2f}</div>'
                    f'<div style="color:#8b949e;font-size:0.7rem;margin-top:4px;">'
                    f'Trades execute at current market price only</div></div>',
                    unsafe_allow_html=True
                )
                estimated_cost = live_price * quantity
                st.caption(f"Estimated order value: **${estimated_cost:,.2f}**")
            else:
                if price_error:
                    st.error(price_error)
                else:
                    st.error(f"Cannot fetch live price for {symbol}")
            
            strategy = st.text_input("Strategy Name", value="Manual Trade", key="manual_trade_strategy")
            reasoning = st.text_area("Reasoning", value="Manual trade entry", key="manual_trade_reasoning", height=80)
        
        col_exec1, col_exec2 = st.columns(2)
        
        with col_exec1:
            can_trade = bool(live_price and live_price > 0)
            
            if st.button(
                "Execute Trade",
                use_container_width=True,
                disabled=not can_trade,
                type="primary"
            ):
                if not symbol:
                    st.error("Please enter a symbol")
                    return
                
                if quantity <= 0:
                    st.error("Quantity must be greater than 0")
                    return
                
                if not live_price or live_price <= 0:
                    st.error("Cannot execute trade without valid price")
                    return
                
                # Validate account has sufficient balance
                multiplier = 100 if asset_type == "Option" else (50 if asset_type == "Future" else 1)
                trade_cost = live_price * quantity * multiplier
                
                if action in ["BUY", "SHORT"] and trade_cost > account.current_balance:
                    st.error(
                        f"Insufficient balance!\n\n"
                        f"Required: ${trade_cost:,.2f}\n"
                        f"Available: ${account.current_balance:,.2f}"
                    )
                    return
                
                try:
                    if asset_type in ["Stock", "Future"]:
                        # Convert action string to TradeAction enum
                        trade_action = TradeAction[action]
                        
                        # Execute trade
                        trade = paper_trading.execute_trade(
                            account_id=account.account_id,
                            symbol=symbol,
                            action=trade_action,
                            quantity=int(quantity) * multiplier if asset_type == "Future" else int(quantity),
                            price=float(live_price),
                            strategy_name=strategy if strategy else f"Manual {asset_type} Trade",
                            ai_reasoning=reasoning if reasoning else f"User executed {action} manually"
                        )
                    else:
                        # Execute Option Trade
                        contract_symbol = f"{symbol}_{expiry.strftime('%y%m%d')}{opt_type[0]}{int(strike*1000):08d}"
                        trade = paper_trading.execute_option_trade(
                            account_id=account.account_id,
                            underlying_symbol=symbol,
                            contract_symbol=contract_symbol,
                            action=action,
                            quantity=int(quantity),
                            price=float(live_price),
                            option_type=opt_type,
                            strike=float(strike),
                            expiration=expiry.strftime('%Y-%m-%d'),
                            strategy_name=strategy if strategy else f"Manual {opt_type}",
                            ai_reasoning=reasoning if reasoning else f"User bought {opt_type} manually"
                        )
                    
                    if trade:
                        st.success(
                            f" Trade Executed Successfully!\n\n"
                            f"**{action}** {int(quantity)} {symbol} @ ${live_price:,.2f} (live market price)\n"
                            f"**Total Value:** ${trade_cost:,.2f}"
                        )
                        st.session_state.show_manual_trade = False
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(
                            f"Failed to execute trade. Please check:\n"
                            f" Symbol is valid: {symbol}\n"
                            f" Account has sufficient balance\n"
                            f" Price is available: ${live_price:,.2f}"
                        )
                
                except KeyError:
                    st.error(f"Invalid action: {action}")
                except ValueError as ve:
                    st.error(f"Invalid input values: {str(ve)}")
                except Exception as e:
                    st.error(f"Trade execution failed: {str(e)}")
        
        with col_exec2:
            if st.button("Cancel", use_container_width=True, key="cancel_manual_trade"):
                st.session_state.show_manual_trade = False
                st.rerun()


def _get_live_price_safe(symbol: str) -> Optional[float]:
    """Safely fetch live price with multiple fallback methods."""
    symbol = symbol.strip().upper()
    
    # Method 1: yfinance with timeout
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        
        price = getattr(info, 'last_price', None) or getattr(info, 'regularMarketPrice', None)
        if price and price > 0:
            return float(price)
    except Exception:
        pass
    
    # Method 2: History-based (more reliable for some assets)
    try:
        df = _fetch_data_safe(symbol, "5d")
        if df is not None and not df.empty and 'Close' in df.columns:
            close = df['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna().astype(float)
            
            if len(close) > 0:
                return float(close.iloc[-1])
    except Exception:
        pass
    
    # Method 3: data_sources functions
    try:
        # Forex
        if '/' in symbol:
            fx_pair = symbol.replace('/', '_')
            df = get_fx(fx_pair)
            if df is not None and not df.empty and 'Close' in df.columns:
                close = df['Close']
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                return float(close.dropna().astype(float).iloc[-1])
        
        # Stock
        df = get_stock(symbol, period="1d")
        if df is not None and not df.empty and 'Close' in df.columns:
            close = df['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            return float(close.dropna().astype(float).iloc[-1])
    except Exception:
        pass
    
    return None


def _fetch_data_safe(symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Safely fetch data with fallbacks."""
    try:
        if '/' in symbol:
            fx_pair = symbol.replace('/', '_')
            df = get_fx(fx_pair)
            if df is not None and not df.empty:
                return df
        
        if '=F' in symbol:
            return get_futures_proxy(symbol, period=period)
        
        return get_stock(symbol, period=period)
    except Exception:
        return None
