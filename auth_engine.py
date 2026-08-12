import streamlit as st
import bcrypt
import sqlite3
from db_manager import get_db_connection

def verify_password(plain_text_password, hashed_password):
    """Check a plain-text password against a hashed stored password."""
    # bcrypt expects bytes, so we encode the strings
    return bcrypt.checkpw(plain_text_password.encode('utf-8'), hashed_password.encode('utf-8'))

def hash_password(plain_text_password):
    """Hash a password with bcrypt for secure storage."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain_text_password.encode('utf-8'), salt).decode('utf-8')

def authenticate_user(username, password):
    """Verifies credentials against the db and updates session state."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, username, password_hash, tos_accepted FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()

    if row:
        user_id = row[0]
        stored_hash = row[2]
        if verify_password(password, stored_hash):
            st.session_state['authenticated'] = True
            st.session_state['user_id'] = user_id
            st.session_state['username'] = row[1]
            # Persist ToS acceptance across sessions — once accepted at registration,
            # the user should never need to accept again.
            st.session_state['tos_acknowledged'] = bool(row[3])
            return True
    return False

def register_user(username, email, password, tos_accepted: bool = False):
    """Creates a new user and generic default profile."""
    if not tos_accepted:
        return False, "You must accept the Terms of Service to create an account."
    
    conn = get_db_connection()
    c = conn.cursor()
    
    # Check if exists
    c.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
    if c.fetchone():
        conn.close()
        return False, "Username or Email already exists."
        
    try:
        hashed_pw = hash_password(password)
        c.execute("""
            INSERT INTO users (username, email, password_hash, tos_accepted)
            VALUES (?, ?, ?, ?)
        """, (username, email, hashed_pw, 1))
        
        user_id = c.lastrowid
        
        # Create empty profile
        c.execute("""
            INSERT INTO user_profiles (user_id, full_name, bio, portfolio_interests, risk_tolerance, preferred_timeframe)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, username, '', '', 'medium', 'medium-term'))
        
        # Set default notifications
        c.execute("""
            INSERT INTO user_settings (user_id, notif_in_app, frequency)
            VALUES (?, ?, ?)
        """, (user_id, 1, 'real-time'))
        
        conn.commit()
        
        # After successful registration, log them in with ToS already acknowledged
        st.session_state['authenticated'] = True
        st.session_state['user_id'] = user_id
        st.session_state['username'] = username
        st.session_state['tos_acknowledged'] = True
        
        return True, "Registration successful."
    except Exception as e:
        return False, f"Database error: {str(e)}"
    finally:
        conn.close()

def logout():
    """Clear session securely."""
    for key in ['authenticated', 'user_id', 'username', 'needs_onboarding']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

def show_auth_page():
    """Renders the secure Login / Sign Up portal."""
    st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>Octavian Terminal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Institutional Grade Agentic Intelligence</p>", unsafe_allow_html=True)
    st.write("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["Secure Login", "Register"])

        with tab_login:
            with st.form("login_form"):
                st.subheader("Sign In")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Access Terminal", use_container_width=True)
                
                if submit:
                    if authenticate_user(username, password):
                        st.success("Authentication successful. Initializing engines...")
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Access Denied.")

        with tab_signup:
            with st.form("signup_form"):
                st.subheader("Create Account")
                new_user = st.text_input("Username", key="reg_username")
                new_email = st.text_input("Email Address", key="reg_email")
                new_pwd = st.text_input("Password", type="password", key="reg_pwd")
                pwd_confirm = st.text_input("Confirm Password", type="password", key="reg_pwd_confirm")
                
                st.markdown(
                    "<small>By creating an account you agree to the "
                    "<b>Terms of Service &amp; Risk Disclosure</b> — viewable in the "
                    "sidebar under <em>Terms of Service</em> after login.</small>",
                    unsafe_allow_html=True,
                )
                agree = st.checkbox(
                    "I have read and accept the Terms of Service & Risk Disclosure.",
                    key="reg_tos_checkbox",
                )
                submit_new = st.form_submit_button("Register & Initialize Profile", use_container_width=True)
                
                if submit_new:
                    if not new_user or not new_pwd:
                        st.error("Username and Password are required.")
                    elif new_pwd != pwd_confirm:
                        st.error("Passwords do not match.")
                    elif not agree:
                        st.error("You must accept the Terms of Service to create an account.")
                    else:
                        success, desc = register_user(new_user, new_email, new_pwd, tos_accepted=True)
                        if success:
                            st.session_state['needs_onboarding'] = True
                            st.rerun()
                        else:
                            st.error(desc)

def show_onboarding_page():
    """Renders the onboarding sequence after a new user signs up."""
    st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>Welcome to Octavian</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Let's set up your Institutional Profile.</p>", unsafe_allow_html=True)
    st.write("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("onboarding_form"):
            st.subheader("Profile Configuration")
            full_name = st.text_input("Full Name / Alias")
            interests = st.text_input("Portfolio Interests (e.g. SPY, Crypto, AI)")
            
            st.subheader("Risk & Duration")
            risk = st.select_slider("Risk Tolerance", options=["Low", "Medium", "High"], value="Medium")
            timeframe = st.select_slider("Preferred Timeframe", options=["Short-term", "Medium-term", "Long-term"], value="Medium-term")
            
            st.subheader("Smart Notifications")
            st.caption("Would you like to receive actionable market alerts tailored to your profile?")
            notif_in_app = st.checkbox("Enable In-App Alerts", value=True)
            freq = st.selectbox("Alert Frequency", ["real-time", "daily", "weekly"])
            
            submit = st.form_submit_button("Complete Onboarding", type="primary", use_container_width=True)
            
            if submit:
                # Update DB
                conn = get_db_connection()
                try:
                    c = conn.cursor()
                    user_id = st.session_state['user_id']
                    
                    c.execute("""
                        UPDATE user_profiles 
                        SET full_name = ?, portfolio_interests = ?, risk_tolerance = ?, preferred_timeframe = ?
                        WHERE user_id = ?
                    """, (full_name, interests, risk.lower(), timeframe.lower(), user_id))
                    
                    c.execute("""
                        UPDATE user_settings
                        SET notif_in_app = ?, frequency = ?
                        WHERE user_id = ?
                    """, (int(notif_in_app), freq, user_id))
                    
                    conn.commit()
                finally:
                    conn.close()
                    
                # Complete onboarding
                del st.session_state['needs_onboarding']
                st.rerun()

