"""
Portfolio Analyzer UI with dual input modes:
1) Import from Paper Trading Account
2) Manual Entry
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except Exception:
        return None


def _position_multiplier(asset_type: str) -> float:
    if str(asset_type).lower() == "option":
        return 100.0
    return 1.0


def _fetch_live_price(symbol: str, asset_type: str) -> Optional[float]:
    try:
        from data_sources import get_fx, get_stock
    except Exception:
        return None

    try:
        at = str(asset_type).lower()
        if at == "fx":
            df = get_fx(symbol, period="5d")
        else:
            lookup_symbol = symbol
            if at == "crypto" and "-" not in symbol:
                lookup_symbol = f"{symbol}-USD"
            df = get_stock(lookup_symbol, period="5d")
        if df is None or df.empty:
            return None
        close_col = df.get("Close")
        if close_col is None or close_col.empty:
            return None
        return float(pd.to_numeric(close_col, errors="coerce").dropna().iloc[-1])
    except Exception:
        return None


def compute_portfolio_metrics(positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute portfolio metrics from a normalized positions list.

    Each position dict should contain:
    symbol, quantity, entry_price, current_price (optional), asset_type.
    """
    rows: List[Dict[str, Any]] = []
    known_market_total = 0.0
    known_entry_total = 0.0
    aggregate_pnl = 0.0

    for pos in positions:
        symbol = str(pos.get("symbol", "")).upper().strip()
        quantity = _coerce_float(pos.get("quantity")) or 0.0
        entry_price = _coerce_float(pos.get("entry_price")) or 0.0
        current_price = _coerce_float(pos.get("current_price"))
        asset_type = str(pos.get("asset_type", "equity")).lower()
        multiplier = _position_multiplier(asset_type)

        entry_value = quantity * entry_price * multiplier
        market_value = None
        unrealized_pnl = None
        return_pct = None
        price_status = "OK"

        if current_price is None:
            price_status = "Price unavailable"
        else:
            market_value = quantity * current_price * multiplier
            unrealized_pnl = market_value - entry_value
            return_pct = (unrealized_pnl / entry_value * 100.0) if entry_value != 0 else None
            known_market_total += market_value
            known_entry_total += entry_value
            aggregate_pnl += unrealized_pnl

        rows.append({
            "symbol": symbol,
            "asset_type": asset_type,
            "quantity": quantity,
            "entry_price": entry_price,
            "current_price": current_price,
            "entry_value": entry_value,
            "market_value": market_value,
            "unrealized_pnl": unrealized_pnl,
            "return_pct": return_pct,
            "position_weight_pct": 0.0,
            "price_status": price_status,
        })

    if known_market_total > 0:
        for row in rows:
            mv = row.get("market_value")
            row["position_weight_pct"] = (mv / known_market_total * 100.0) if mv is not None else 0.0

    overall_return_pct = (aggregate_pnl / known_entry_total * 100.0) if known_entry_total > 0 else 0.0
    return {
        "rows": rows,
        "summary": {
            "position_count": len(rows),
            "total_portfolio_value": known_market_total,
            "aggregate_unrealized_pnl": aggregate_pnl,
            "overall_return_pct": overall_return_pct,
            "priced_positions": sum(1 for r in rows if r.get("current_price") is not None),
            "unpriced_positions": sum(1 for r in rows if r.get("current_price") is None),
        },
    }


def _rows_for_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)[[
        "symbol",
        "asset_type",
        "quantity",
        "entry_price",
        "current_price",
        "entry_value",
        "market_value",
        "unrealized_pnl",
        "return_pct",
        "position_weight_pct",
        "price_status",
    ]]


def show_portfolio_analyzer():
    import streamlit as st

    st.title("Portfolio Analyzer")
    st.caption("Analyze imported paper-trading positions or manually entered portfolios.")

    if "user_id" not in st.session_state:
        st.session_state["user_id"] = st.session_state.get("username", "default_user")
    if "pa_manual_rows" not in st.session_state:
        st.session_state["pa_manual_rows"] = [{
            "symbol": "",
            "quantity": 0.0,
            "entry_price": 0.0,
            "current_price": "",
            "asset_type": "equity",
        }]

    tabs = st.tabs(["Import from Paper Trading Account", "Manual Portfolios", "Closed Positions History"])

    with tabs[0]:
        st.subheader("Import from Paper Trading Account")
        import_error = False
        try:
            from paper_trading_system import get_paper_trading_system
            pts = get_paper_trading_system()
            accounts = pts.list_accounts(st.session_state["user_id"])
        except Exception as e:
            import_error = True
            st.error(f"Could not load accounts: {e}")
            accounts = []

        if import_error or not accounts:
            st.warning("No paper trading accounts available. Use the Manual Entry tab.")
        else:
            account_map = {f"{a.account_name} ({a.account_id})": a.account_id for a in accounts}
            selected = st.selectbox("Select Account", list(account_map.keys()), key="pa_account_select")
            
            c1, c2 = st.columns([3, 1])
            if c1.button("Load Account Positions", type="primary", key="pa_load_account"):
                try:
                    account_id = account_map[selected]
                    st.session_state["pa_account_id"] = account_id
                    st.session_state["pa_account_name"] = selected
                    stock_positions = pts.get_positions(account_id)
                    option_positions = pts.get_option_positions(account_id)
                    imported_positions: List[Dict[str, Any]] = []

                    for p in stock_positions:
                        imported_positions.append({
                            "symbol": getattr(p, "symbol", ""),
                            "quantity": _coerce_float(getattr(p, "quantity", 0.0)) or 0.0,
                            "entry_price": _coerce_float(getattr(p, "entry_price", 0.0)) or 0.0,
                            "current_price": _coerce_float(getattr(p, "current_price", None)),
                            "asset_type": "equity",
                        })

                    for op in option_positions:
                        imported_positions.append({
                            "symbol": getattr(op, "symbol", ""),
                            "quantity": _coerce_float(getattr(op, "quantity", 0.0)) or 0.0,
                            "entry_price": _coerce_float(getattr(op, "entry_price", 0.0)) or 0.0,
                            "current_price": _coerce_float(getattr(op, "current_price", None)),
                            "asset_type": "option",
                        })

                    st.session_state["pa_loaded_positions"] = imported_positions
                    st.session_state["pa_loaded_source"] = "imported"
                    st.success(f"Loaded {len(imported_positions)} open positions from {selected}.")
                except Exception as e:
                    st.error(f"Failed to import positions: {e}. Use Manual Entry instead.")
            
            if c2.button("Delete Account", type="secondary", key="pa_delete_account"):
                account_id = account_map[selected]
                pts.delete_account(account_id)
                st.success(f"Deleted account {selected}.")
                st.rerun()

    with tabs[1]:
        st.subheader("Manual Portfolios")
        from manual_portfolio_system import get_manual_portfolio_system
        mps = get_manual_portfolio_system()
        
        portfolios = mps.list_portfolios()
        
        c1, c2 = st.columns([2, 1])
        if portfolios:
            port_opts = {f"{v['name']} ({k[:8]})": k for k, v in portfolios.items()}
            selected_port_label = c1.selectbox("Select Manual Portfolio", list(port_opts.keys()))
            selected_pid = port_opts[selected_port_label]
            active_port = portfolios[selected_pid]
        else:
            c1.info("No manual portfolios found.")
            selected_pid = None
            active_port = None

        with c2.expander("Create New Portfolio"):
            n_name = st.text_input("Name", "My Portfolio")
            n_goal = st.text_input("Goal", "Growth")
            n_risk = st.selectbox("Risk Level", ["Conservative", "Moderate", "Aggressive"])
            n_timeframe = st.selectbox("Timeframe", ["1 Year", "3 Years", "5 Years", "10+ Years"])
            n_target = st.number_input("Target Annual Growth (%)", min_value=0.0, value=10.0)
            if st.button("Create Portfolio"):
                mps.create_portfolio(n_name, n_goal, n_risk, n_timeframe, n_target)
                st.rerun()

        if active_port:
            st.markdown("---")
            col_goal, col_del = st.columns([5, 1])
            col_goal.markdown(f"**Goal:** {active_port['goal']} | **Target Growth:** {active_port.get('target_growth', 10.0)}% | **Timeframe:** {active_port.get('timeframe', '1 Year')} | **Risk:** {active_port['risk_level']}")
            if col_del.button("Delete Portfolio", type="secondary", key=f"del_port_{selected_pid}"):
                mps.delete_portfolio(selected_pid)
                st.rerun()
                
            st.markdown("#### Add Position")
            with st.form("add_manual_pos"):
                f1, f2, f3, f4 = st.columns(4)
                f_sym = f1.text_input("Symbol").strip().upper()
                f_qty = f2.number_input("Quantity", min_value=0.01, value=1.0)
                f_price = f3.number_input("Entry Price", min_value=0.01, value=100.0)
                f_type = f4.selectbox("Asset Type", ["equity", "etf", "option", "crypto", "fx"])
                if st.form_submit_button("Add Position"):
                    if f_sym:
                        mps.add_position(selected_pid, f_sym, f_qty, f_price, f_type)
                        st.rerun()
                        
            st.markdown("#### Active Positions")
            if active_port["positions"]:
                for pos in active_port["positions"]:
                    p1, p2, p3, p4, p5 = st.columns([2, 1, 2, 1, 1])
                    p1.write(f"**{pos['symbol']}** ({pos['asset_type']})")
                    p2.write(f"Qty: {pos['quantity']}")
                    p3.write(f"Entry: ${pos['entry_price']:.2f}")
                    if p4.button("Close", key=f"close_{pos['id']}"):
                        st.session_state["close_pos_id"] = pos['id']
                        st.session_state["close_pid"] = selected_pid
                    if p5.button("Delete", key=f"del_{pos['id']}"):
                        mps.delete_position(selected_pid, pos['id'])
                        st.rerun()
                
                if "close_pos_id" in st.session_state:
                    pos_id = st.session_state["close_pos_id"]
                    pid = st.session_state["close_pid"]
                    if pid == selected_pid: 
                        st.warning("Enter Exit Price to close position:")
                        exit_price = st.number_input("Exit Price", min_value=0.01, value=100.0)
                        cb1, cb2 = st.columns(2)
                        if cb1.button("Confirm Close"):
                            mps.close_position(pid, pos_id, exit_price)
                            del st.session_state["close_pos_id"]
                            st.rerun()
                        if cb2.button("Cancel"):
                            del st.session_state["close_pos_id"]
                            st.rerun()
            else:
                st.info("No active positions.")
                
            st.markdown("#### Closed Positions History")
            if active_port["history"]:
                hist_df = pd.DataFrame(active_port["history"])
                st.dataframe(hist_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No closed positions yet.")

            st.markdown("---")
            if st.button("Analyze Manual Portfolio", type="primary", key="pa_analyze_manual"):
                normalized: List[Dict[str, Any]] = []
                for pos in active_port["positions"]:
                    current_price = _fetch_live_price(pos["symbol"], pos["asset_type"])
                    normalized.append({
                        "symbol": pos["symbol"],
                        "quantity": float(pos["quantity"]),
                        "entry_price": float(pos["entry_price"]),
                        "current_price": current_price,
                        "asset_type": pos["asset_type"],
                    })

                st.session_state["pa_loaded_positions"] = normalized
                st.session_state["pa_loaded_source"] = f"manual - {active_port['name']}"
                st.success(f"Loaded {len(normalized)} manually entered positions for analysis.")

            st.markdown("---")
            st.subheader("AI Portfolio Advisor")
            st.caption("Get tailored intelligence on how to hit your portfolio goals.")
            if st.button("Generate AI Portfolio Strategy", type="secondary"):
                with st.spinner("Generating institutional-grade portfolio intelligence..."):
                    from financial_llm_engine import generate_portfolio_advice
                    advice = generate_portfolio_advice(active_port)
                    st.session_state[f"ai_advice_{selected_pid}"] = advice

            if f"ai_advice_{selected_pid}" in st.session_state:
                advice_text = st.session_state[f"ai_advice_{selected_pid}"]
                with st.expander("Portfolio Strategy Report", expanded=True):
                    st.markdown(f"```\n{advice_text}\n```")

    loaded_positions = st.session_state.get("pa_loaded_positions", [])
    if not loaded_positions:
        st.info("Load imported positions or submit manual entries to view analytics.")
        return

    st.markdown("---")
    st.subheader("Portfolio Metrics")
    c1, c2 = st.columns([5, 1])
    c1.caption(f"Source: {st.session_state.get('pa_loaded_source', 'unknown')} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if c2.button("Clear Loaded Data", type="secondary", key="pa_clear_loaded"):
        st.session_state.pop("pa_loaded_positions", None)
        st.session_state.pop("pa_loaded_source", None)
        st.rerun()

    metrics = compute_portfolio_metrics(loaded_positions)
    summary = metrics["summary"]
    rows = metrics["rows"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Portfolio Value", f"${summary['total_portfolio_value']:,.2f}")
    m2.metric("Aggregate Unrealized P&L", f"${summary['aggregate_unrealized_pnl']:+,.2f}")
    m3.metric("Overall Return", f"{summary['overall_return_pct']:+.2f}%")
    m4.metric("Priced / Unpriced", f"{summary['priced_positions']} / {summary['unpriced_positions']}")

    df = _rows_for_dataframe(rows)
    if not df.empty:
        st.dataframe(
            df.style.format({
                "entry_price": "${:,.4f}",
                "current_price": "${:,.4f}",
                "entry_value": "${:,.2f}",
                "market_value": "${:,.2f}",
                "unrealized_pnl": "${:+,.2f}",
                "return_pct": "{:+.2f}%",
                "position_weight_pct": "{:.2f}%",
            }, na_rep="Price unavailable"),
            use_container_width=True,
            hide_index=True,
        )

        export_df = df.copy()
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export Portfolio CSV",
            data=csv_data,
            file_name=f"portfolio_{st.session_state.get('pa_loaded_source', 'data')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    if st.session_state.get("pa_loaded_source") == "imported" and "pa_account_id" in st.session_state:
        st.markdown("---")
        st.subheader("Manage Paper Trading Positions")
        open_positions = []
        for p in loaded_positions:
            open_positions.append(f"{p['symbol']} ({p['quantity']} units)")
        
        if open_positions:
            c1, c2 = st.columns([3, 1])
            pos_to_close = c1.selectbox("Select Position to Close", open_positions)
            if c2.button("Close Position", type="primary"):
                sym_to_close = pos_to_close.split(" ")[0]
                pos = next((x for x in loaded_positions if x['symbol'] == sym_to_close), None)
                if pos:
                    try:
                        from paper_trading_system import get_paper_trading_system, TradeAction
                        pts = get_paper_trading_system()
                        acct_id = st.session_state["pa_account_id"]
                        
                        action = TradeAction.SELL if pos['quantity'] > 0 else TradeAction.COVER
                        curr_price = pos['current_price'] or _fetch_live_price(sym_to_close, pos['asset_type'])
                        
                        if pos['asset_type'] == 'option':
                            # Options close: sell a long, buy back a short. The engine
                            # derives contract metadata when not supplied by this caller.
                            close_action = TradeAction.SELL if action == TradeAction.SELL else TradeAction.COVER
                            trade = pts.execute_option_trade(
                                acct_id, sym_to_close, None, close_action,
                                abs(pos['quantity']), curr_price,
                                option_type=str(pos.get('option_type', 'CALL')),
                                strike=pos.get('strike'),
                                expiration=pos.get('expiration'),
                            )
                        else:
                            trade = pts.execute_trade(acct_id, sym_to_close, action, abs(pos['quantity']), curr_price)
                        
                        if trade is not None:
                            st.success(f"Successfully closed {sym_to_close} at {curr_price}")
                        else:
                            st.error(f"Failed to close {sym_to_close} — no trade was recorded.")
                        # Clean up
                        st.session_state["pa_loaded_positions"] = [x for x in loaded_positions if x['symbol'] != sym_to_close]
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to close position: {e}")

    with tabs[2]:
        st.subheader("Closed Positions History")
        if "pa_account_id" in st.session_state:
            try:
                from paper_trading_system import get_paper_trading_system
                pts = get_paper_trading_system()
                history = pts.get_trade_history(st.session_state["pa_account_id"])
                
                if not history:
                    st.info("No trade history found for this account.")
                else:
                    hist_data = []
                    for t in history:
                        hist_data.append({
                            "Date": t.timestamp,
                            "Symbol": t.symbol,
                            "Action": t.action.name,
                            "Quantity": t.quantity,
                            "Price": t.price,
                            "Details": t.details
                        })
                    st.dataframe(pd.DataFrame(hist_data), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Failed to load trade history: {e}")
        else:
            st.info("Please select and load a Paper Trading Account in the first tab to view closed positions.")

    # 25-Factor Advanced Institutional Analytics
    st.markdown("---")
    st.subheader("25-Factor Institutional Portfolio Analytics")

    if st.button("Run Comprehensive Grading & Analysis", type="primary", key="run_25_factor"):
        with st.spinner("Running Monte Carlo, Risk, and Factor Models..."):
            try:
                import asyncio
                from portfolio_analyzer_engine import get_portfolio_analyzer

                engine = get_portfolio_analyzer()

                # ── For manual portfolios: feed loaded_positions directly ──
                is_manual = st.session_state.get("pa_loaded_source", "").startswith("manual")
                acct_id   = st.session_state.get("pa_account_id")

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                if is_manual and loaded_positions:
                    analytics = loop.run_until_complete(
                        engine.analyze_portfolio_from_positions(loaded_positions)
                    )
                elif acct_id:
                    analytics = loop.run_until_complete(engine.analyze_portfolio(acct_id))
                else:
                    st.error("Please load a portfolio first.")
                    analytics = None

                if analytics:
                    # ── Display Grading ───────────────────────────────────
                    grade_color = "#4caf50" if analytics.health_score > 80 else "#ffaa00" if analytics.health_score > 60 else "#ef5350"
                    st.markdown(
                        f"<h2 style='text-align: center; color: {grade_color}'>"
                        f"Overall Health Grade: {analytics.health_grade} ({analytics.health_score:.1f}/100)</h2>",
                        unsafe_allow_html=True,
                    )

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("### Past Performance")
                        st.metric("Sharpe Ratio",  f"{analytics.sharpe_ratio:.2f}")
                        st.metric("Sortino Ratio", f"{analytics.sortino_ratio:.2f}")
                        st.metric("Max Drawdown",  f"{analytics.max_drawdown:.1f}%")
                        st.metric("Win Rate",      f"{analytics.win_rate:.1f}%")
                        st.metric("Profit Factor", f"{analytics.profit_factor:.2f}")

                    with c2:
                        st.markdown("### Present Risk & Allocation")
                        st.metric("Annual Volatility",   f"{analytics.volatility_annual:.2f}%")
                        st.metric("95% Daily VaR",       f"${analytics.var_95_daily:,.2f}")
                        st.metric("Beta to SPY",         f"{analytics.beta_to_spy:.2f}")
                        st.metric("Diversification Score", f"{analytics.diversification_score:.1f}/100")
                        st.metric("Avg Correlation",     f"{analytics.correlation_avg:.2f}")

                    with c3:
                        st.markdown("### Predictive & Macro")
                        st.metric("Macro Regime Alignment", analytics.regime_alignment)

                        st.markdown("**Scenario Impacts (Beta Adjusted)**")
                        for sc_name, sc_data in analytics.scenario_impacts.items():
                            impact_pct = sc_data.get('impact_pct', sc_data) if isinstance(sc_data, dict) else sc_data
                            prob       = sc_data.get('probability', '—')    if isinstance(sc_data, dict) else '—'
                            st.write(f"- **{sc_name}**: {impact_pct}% (Prob: {prob})")

                    st.markdown("### Institutional Improvement Ideas")
                    for idea in analytics.improvement_ideas:
                        title    = idea.get('title', '')
                        strategy = idea.get('strategy', '')
                        impact   = idea.get('impact', '')
                        # Build specific recommendation based on portfolio data
                        if analytics.volatility_annual > 0 and "Risk Adjusted" in title:
                            strategy = (f"Current Sharpe: {analytics.sharpe_ratio:.2f}. "
                                        "Improve by reducing low-conviction positions and trimming concentration risk.")
                        elif analytics.beta_to_spy > 0 and "Beta" in title:
                            strategy = (f"Current Beta: {analytics.beta_to_spy:.2f}. "
                                        "Add low-beta assets (defensive equities, bonds, gold) to stabilise market exposure.")
                        elif "Macro" in title:
                            strategy = (f"Current regime: {analytics.regime_alignment}. "
                                        "Tilt allocation toward sectors aligned with the prevailing macro regime.")
                        st.info(f"**{title}**\n\n{strategy}\n\n*Impact:* {impact}")

            except Exception as e:
                st.error(f"Advanced Analytics Error: {e}")
                import traceback; st.code(traceback.format_exc())
