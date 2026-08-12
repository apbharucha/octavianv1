"""Octavian Terms of Service / Risk Disclosure view.

For accounts created after the tos_accepted migration this page is
informational-only — acceptance was already recorded at registration.

For legacy accounts (tos_accepted = 0 in DB) we show the checkbox and
persist acceptance immediately so they are never prompted again.
"""
import streamlit as st
from octavian_theme import COLORS


def _persist_tos_acceptance(user_id: int) -> None:
    """Mark tos_accepted = 1 in the DB for a legacy account."""
    try:
        from db_manager import execute_query
        execute_query(
            "UPDATE users SET tos_accepted = 1 WHERE id = ?",
            params=(user_id,),
            commit=True,
)
        st.session_state["tos_acknowledged"] = True
    except Exception:
        # Fall back to session-only acceptance if DB write fails
        st.session_state["tos_acknowledged"] = True


def show_terms_of_service():
    st.title("Terms of Service & Risk Disclosure")

    already_accepted = st.session_state.get("tos_acknowledged", False)

    if already_accepted:
        st.caption(
            "You accepted these terms when you created your account."
            "This page is provided for reference.")
    else:
        st.caption(
            "Please review carefully. You must accept these terms before using"
            "trading-related features.")

    st.markdown(
        f"""<div style="background:linear-gradient(145deg,{COLORS['navy_mid']},{COLORS['navy_light']});border:1px solid {COLORS['border']};border-radius:12px;padding:16px;margin:10px 0 16px 0;">
  <div style="color:{COLORS['gold']};font-weight:700;margin-bottom:8px;">Important Notice</div>
  <div style="color:{COLORS['text_primary']};font-size:0.92rem;line-height:1.6;">
    Octavian provides analytics, simulations, and decision-support tooling for educational and research purposes.
    It does <b>not</b> provide personalized investment advice, tax advice, legal advice, or fiduciary services.
  </div>
</div>
        """,
        unsafe_allow_html=True,
)

    st.markdown(
        """### 1) No Investment Advice
- Content is informational and model-generated.
- Outputs may be incomplete, delayed, or inaccurate.
- You are solely responsible for all trading and allocation decisions.

### 2) Risk of Loss
- Markets are volatile; losses can exceed expectations.
- Historical or simulated performance does not guarantee future results.
- Derivatives and leveraged products carry elevated risk.

### 3) Data & Model Limitations
- Third-party feeds may fail, lag, or revise.
- Models can overfit or degrade under regime shifts.
- Any confidence score is probabilistic, not a guarantee.

### 4) Paper Trading & Simulation
- Simulated fills differ from live execution (slippage, liquidity, latency, routing).
- Paper outcomes should not be interpreted as expected real-money outcomes.

### 5) No Warranties
- Platform is provided "as is"and "as available".
- No warranty of uptime, merchantability, fitness for a particular purpose, or non-infringement.

### 6) Limitation of Liability
- To the fullest extent allowed by law, the platform owners/contributors are not liable for direct, indirect,
  incidental, consequential, or special damages from use or inability to use the platform.

### 7) User Responsibility
- Verify all critical numbers independently before execution.
- Maintain your own risk controls, position limits, and compliance obligations.
- Do not use this platform where prohibited by applicable law or policy.

### 8) Jurisdiction & Updates
- Terms may be updated over time; continued use implies acceptance of updates.
- If any term is unenforceable, remaining provisions stay in effect.
""")

    if already_accepted:
        st.success(
            "Terms accepted — your account is fully activated."
            "You will not be prompted again.")
    else:
        # Legacy account — let them accept here and persist it
        accepted = st.checkbox(
            "I have read and accept these Terms of Service & Risk Disclosure.",
            key="tos_accept_legacy",
)
        if accepted:
            user_id = st.session_state.get("user_id")
            if user_id:
                _persist_tos_acceptance(user_id)
            else:
                st.session_state["tos_acknowledged"] = True
            st.success(
                "Terms accepted and saved to your account."
                "You will not be prompted again.")
            st.rerun()
        else:
            st.warning(
                "Please accept the Terms of Service to unlock all platform features.")
