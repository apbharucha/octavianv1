import sqlite3
from datetime import datetime
from db_manager import get_db_connection

def dispatch_smart_alert(alert_type, title, message, asset_symbol=None, confidence_score=0):
    """
    Evaluates an alert against every user's personal risk profile and portfolio interests.
    Dispatches an in-app notification ONLY if it matches their specific configuration.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # Fetch all active users with their settings and profiles
    query = """
        SELECT u.id, p.portfolio_interests, p.risk_tolerance, s.notif_in_app, s.quiet_hours_start, s.quiet_hours_end
        FROM users u
        JOIN user_profiles p ON u.id = p.user_id
        JOIN user_settings s ON u.id = s.user_id
    """
    c.execute(query)
    users = c.fetchall()
    
    notifications_to_insert = []
    
    current_hour = datetime.now().hour
    current_time_str = datetime.now().strftime("%H:%M")
    
    for user in users:
        user_id = user["id"]
        interests = str(user["portfolio_interests"]).lower()
        risk = str(user["risk_tolerance"]).lower()
        in_app_enabled = bool(user["notif_in_app"])
        quiet_start = user["quiet_hours_start"] # e.g. "22:00"
        quiet_end = user["quiet_hours_end"]     # e.g. "06:00"
        
        # 1. Global setting check
        if not in_app_enabled:
            continue
            
        # 2. Quiet Hours check (Simplified logic for HH:MM strings)
        # Note: robust implementation requires datetime parsing, simple string compare works if zero-padded
        if quiet_start <= current_time_str or current_time_str <= quiet_end:
            if quiet_start > quiet_end: # Crosses midnight
                if current_time_str >= quiet_start or current_time_str <= quiet_end:
                    continue
        
        # 3. Portfolio Matching
        # If the alert has a specific asset, check if the user is interested
        is_interested = True
        if asset_symbol and asset_symbol.lower() not in interests and "all" not in interests:
            # Maybe it's a sector alert? Try basic heuristic
            is_interested = False
            
            # Simple fallback check
            if "crypto" in interests and asset_symbol.lower() in ['btc', 'eth', 'sol']:
                is_interested = True
            elif "tech" in interests and asset_symbol.lower() in ['aapl', 'msft', 'nvda', 'qqq']:
                is_interested = True
            elif "macro" in interests or "spx" in interests:
                is_interested = True

        if not is_interested:
            continue
            
        # 4. Risk Tolerance / Confidence Gate
        # Low risk = high confidence needed -> >= 85
        # Medium risk = medium confidence needed -> >= 70
        # High risk = any viable opportunity -> >= 50
        min_confidence = 70
        if risk == 'low': min_confidence = 85
        if risk == 'high': min_confidence = 50
        
        if confidence_score > 0 and confidence_score < min_confidence:
            continue
            
        # If passed all checks, queue notification
        notifications_to_insert.append((user_id, alert_type, title, message))
        
    if notifications_to_insert:
        c.executemany("""
            INSERT INTO notifications (user_id, type, title, message)
            VALUES (?, ?, ?, ?)
        """, notifications_to_insert)
        conn.commit()
        
    conn.close()
    return len(notifications_to_insert)


def get_user_notifications(user_id, unread_only=True, limit=10):
    """Retrieve notifications for a specific user."""
    conn = get_db_connection()
    c = conn.cursor()
    
    query = "SELECT id, type, title, message, created_at, is_read FROM notifications WHERE user_id = ?"
    if unread_only:
        query += " AND is_read = 0"
    query += " ORDER BY created_at DESC LIMIT ?"
    
    c.execute(query, (user_id, limit))
    notifs = []
    for row in c.fetchall():
        notifs.append({
            "id": row["id"],
            "type": row["type"],
            "title": row["title"],
            "message": row["message"],
            "created_at": row["created_at"],
            "is_read": row["is_read"]
        })
    conn.close()
    return notifs


def mark_notification_read(notif_id):
    """Mark a single notification as read."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE notifications SET is_read = 1 WHERE id = ?", (notif_id,))
    conn.commit()
    conn.close()

def mark_all_read(user_id):
    """Mark all notifications as read for a user."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE notifications SET is_read = 1 WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
