import sqlite3
import bcrypt
import os
from datetime import datetime

DATABASE_FILE = "octavian_users.db"

def get_db_connection():
    """Returns a dictionary-like SQLite connection."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database schemas and run migrations."""
    conn = get_db_connection()
    c = conn.cursor()

    # Users Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            tos_accepted BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Migration: add tos_accepted column if it doesn't exist yet (for existing DBs)
    try:
        c.execute("ALTER TABLE users ADD COLUMN tos_accepted BOOLEAN DEFAULT 0")
        conn.commit()
    except Exception:
        pass  # Column already exists

    # User Profiles Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id INTEGER PRIMARY KEY,
            full_name TEXT,
            profile_picture TEXT,
            bio TEXT,
            portfolio_interests TEXT,
            risk_tolerance TEXT,
            preferred_timeframe TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')

    # User Settings Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id INTEGER PRIMARY KEY,
            notif_email BOOLEAN DEFAULT 0,
            notif_sms BOOLEAN DEFAULT 0,
            notif_in_app BOOLEAN DEFAULT 1,
            frequency TEXT DEFAULT 'real-time',
            triggers TEXT DEFAULT 'all',
            quiet_hours_start TEXT DEFAULT '22:00',
            quiet_hours_end TEXT DEFAULT '06:00',
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')

    # Notifications Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            is_read BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')

    conn.commit()

    # Seed the Dev Account (apb:mavs07!) if it doesn't exist
    c.execute("SELECT id FROM users WHERE username = 'apb'")
    if c.fetchone() is None:
        salt = bcrypt.gensalt()
        hashed_pw = bcrypt.hashpw("mavs07!".encode('utf-8'), salt).decode('utf-8')
        
        c.execute("""
            INSERT INTO users (username, email, password_hash, tos_accepted) 
            VALUES (?, ?, ?, 1)
        """, ('apb', 'dev@octavian.ai', hashed_pw))
        
        user_id = c.lastrowid
        
        # Seed Profile
        c.execute("""
            INSERT INTO user_profiles (user_id, full_name, bio, portfolio_interests, risk_tolerance, preferred_timeframe)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, 'Admin Developer', 'System Architect for Octavian AI.', 'SPY, QQQ, AAPL', 'high', 'short-term'))
        
        # Seed Settings
        c.execute("""
            INSERT INTO user_settings (user_id, notif_in_app, frequency)
            VALUES (?, ?, ?)
        """, (user_id, 1, 'real-time'))
        
        conn.commit()

    conn.close()

# Automatically initialize on module load
init_db()

def execute_query(query, params=(), fetch_one=False, fetch_all=False, commit=False):
    """Helper method to execute queries safely."""
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(query, params)
        if commit:
            conn.commit()
            return c.lastrowid
        if fetch_one:
            return c.fetchone()
        if fetch_all:
            return c.fetchall()
    finally:
        conn.close()
