import streamlit as st
from db_manager import get_db_connection

def load_user_settings(user_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM user_settings WHERE user_id = ?", (user_id,))
    settings = c.fetchone()
    
    c.execute("SELECT full_name, bio, portfolio_interests, risk_tolerance, preferred_timeframe FROM user_profiles WHERE user_id = ?", (user_id,))
    profile = c.fetchone()
    
    conn.close()
    return dict(settings) if settings else {}, dict(profile) if profile else {}

def save_user_settings(user_id, settings_data, profile_data):
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute("""
        UPDATE user_settings         SET notif_email = ?, notif_sms = ?, notif_in_app = ?, frequency = ?,
             triggers = ?, quiet_hours_start = ?, quiet_hours_end = ?
        WHERE user_id = ?
    """, (
        int(settings_data['notif_email']),
        int(settings_data['notif_sms']),
        int(settings_data['notif_in_app']),
        settings_data['frequency'],
        settings_data['triggers'],
        settings_data['quiet_hours_start'],
        settings_data['quiet_hours_end'],
        user_id
    ))
    
    c.execute("""
        UPDATE user_profiles
        SET full_name = ?, bio = ?, portfolio_interests = ?, risk_tolerance = ?, preferred_timeframe = ?
        WHERE user_id = ?
    """, (
        profile_data['full_name'],
        profile_data['bio'],
        profile_data['portfolio_interests'],
        profile_data['risk_tolerance'],
        profile_data['preferred_timeframe'],
        user_id
    ))
    
    conn.commit()
    conn.close()

def show_notification_settings():
    st.title("Account & Notification Settings")
    st.markdown("Customize your institutional profile and smart alert delivery preferences.")
    
    user_id = st.session_state.get('user_id')
    if not user_id:
        st.error("No active user session found.")
        return
        
    settings, profile = load_user_settings(user_id)
    
    # Defaults just in case
    if not settings:
        settings = {'notif_email': 0, 'notif_sms': 0, 'notif_in_app': 1, 'frequency': 'real-time', 'triggers': 'all', 'quiet_hours_start': '22:00', 'quiet_hours_end': '06:00'}
    if not profile:
        profile = {'full_name': '', 'bio': '', 'portfolio_interests': '', 'risk_tolerance': 'medium', 'preferred_timeframe': 'medium-term'}
        
    tab1, tab2 = st.tabs(["Smart Notifications", "Identity & Portfolio"])
    
    with tab1:
        st.subheader("Notification Channels")
        c1, c2, c3 = st.columns(3)
        with c1:
            notif_in_app = st.toggle("In-App Notifications", value=bool(settings['notif_in_app']))
            st.caption("Delivered to your Octavian Inbox")
        with c2:
            notif_email = st.toggle("Email Digest", value=bool(settings['notif_email']))
            st.caption("Sent to registered email")
        with c3:
            notif_sms = st.toggle("SMS Text Alerts", value=bool(settings['notif_sms']))
            st.caption("Requires verified phone number")
            
        st.divider()
        st.subheader("Delivery Preferences")
        c4, c5 = st.columns(2)
        with c4:
            freq_opts = ["real-time", "daily", "weekly"]
            freq_idx = freq_opts.index(settings['frequency']) if settings['frequency'] in freq_opts else 0
            frequency = st.selectbox("Delivery Frequency", freq_opts, index=freq_idx)
            
            trigger_opts = ["all", "high-confidence-only", "portfolio-only"]
            trig_idx = trigger_opts.index(settings['triggers']) if settings['triggers'] in trigger_opts else 0
            triggers = st.selectbox("Alert Triggers", trigger_opts, index=trig_idx)
            
        with c5:
            st.markdown("**Quiet Hours (Do Not Disturb)**")
            qh_start = st.text_input("Start Time (HH:MM 24h)", value=settings['quiet_hours_start'])
            qh_end = st.text_input("End Time (HH:MM 24h)", value=settings['quiet_hours_end'])
            
    with tab2:
        st.subheader("Personal Identity")
        full_name = st.text_input("Full Name", value=profile['full_name'])
        bio = st.text_area("Bio / Trading Philosophy", value=profile['bio'])
        
        st.divider()
        st.subheader("Engine Targeting")
        st.caption("The Smart Engine uses these inputs to filter the firehose of alerts down to what you care about.")
        interests = st.text_input("Portfolio Interests (Comma separated, or 'ALL')", value=profile['portfolio_interests'])
        
        c6, c7 = st.columns(2)
        with c6:
            risk_opts = ["low", "medium", "high"]
            risk_idx = risk_opts.index(profile['risk_tolerance']) if profile['risk_tolerance'] in risk_opts else 1
            risk = st.selectbox("Engine Risk Tolerance", risk_opts, index=risk_idx)
            st.caption("Low risk = High confidence alerts only")
        with c7:
            tf_opts = ["short-term", "medium-term", "long-term"]
            tf_idx = tf_opts.index(profile['preferred_timeframe']) if profile['preferred_timeframe'] in tf_opts else 1
            tf = st.selectbox("Preferred Timeframe", tf_opts, index=tf_idx)
            
    if st.button("Save Account Settings", type="primary"):
        new_settings = {
            'notif_email': notif_email,
            'notif_sms': notif_sms,
            'notif_in_app': notif_in_app,
            'frequency': frequency,
            'triggers': triggers,
            'quiet_hours_start': qh_start,
            'quiet_hours_end': qh_end
}
        new_profile = {
            'full_name': full_name,
            'bio': bio,
            'portfolio_interests': interests,
            'risk_tolerance': risk,
            'preferred_timeframe': tf
}
        save_user_settings(user_id, new_settings, new_profile)
        st.success("Configuration saved successfully. The Smart Engine has been updated.")
