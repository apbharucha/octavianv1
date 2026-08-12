"""
Advanced Database Manager for AI Market Chatbot

This module provides comprehensive database management for:
1. Real-time market data storage and retrieval
2. AI conversation history and context
3. User analytics and preferences
4. Market analysis results caching
5. Performance metrics tracking

Author: AI Market Team
"""

import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import threading
from contextlib import contextmanager
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Comprehensive database manager for AI chatbot system."""
    
    def __init__(self, db_path: str = "market_ai_system.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._initialize_databases()
    
    def _initialize_databases(self):
        """Initialize all database tables with proper schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # ============= MARKET DATA TABLES =============
            
            # Real-time price data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    volume INTEGER,
                    data_source TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            """)
            
            # Technical indicators cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    rsi REAL,
                    ema20 REAL,
                    ema50 REAL,
                    macd REAL,
                    macd_signal REAL,
                    bb_upper REAL,
                    bb_middle REAL,
                    bb_lower REAL,
                    volume_sma REAL,
                    atr REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            """)
            
            # ML predictions and analysis results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bullish_prob REAL NOT NULL,
                    model_version TEXT,
                    prediction_horizon INTEGER DEFAULT 5,
                    signal_factors TEXT, -- JSON
                    model_scores TEXT, -- JSON
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Market regime and sentiment
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_regime (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    volatility_regime TEXT,
                    risk_mode TEXT,
                    vix_level REAL,
                    market_sentiment TEXT,
                    sector_rotation TEXT, -- JSON
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # ============= AI CHATBOT TABLES =============
            
            # Conversation history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    intent_detected TEXT, -- JSON array
                    symbols_mentioned TEXT, -- JSON array
                    charts_generated INTEGER DEFAULT 0,
                    response_time_ms INTEGER,
                    satisfaction_rating INTEGER, -- 1-5 scale
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # User preferences and profiles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    preferred_assets TEXT, -- JSON array
                    risk_tolerance TEXT,
                    trading_style TEXT,
                    notification_preferences TEXT, -- JSON
                    dashboard_layout TEXT, -- JSON
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # AI learning and feedback
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    feedback_type TEXT NOT NULL, -- 'positive', 'negative', 'correction'
                    feedback_text TEXT,
                    suggested_improvement TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            """)
            
            # ============= ANALYTICS TABLES =============
            
            # System performance metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_unit TEXT,
                    category TEXT, -- 'performance', 'accuracy', 'usage'
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Query analytics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_type TEXT NOT NULL,
                    symbol TEXT,
                    response_time_ms INTEGER,
                    success BOOLEAN,
                    error_message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Market data quality metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    data_source TEXT NOT NULL,
                    completeness_score REAL, -- 0-1
                    freshness_minutes INTEGER,
                    accuracy_score REAL, -- 0-1
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # ============= WATCHLISTS AND ALERTS =============
            
            # User watchlists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    symbols TEXT NOT NULL, -- JSON array
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Price alerts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL, -- 'price_above', 'price_below', 'change_percent'
                    threshold_value REAL NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    triggered_at DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # User activity logs for adaptive personalization
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_activity_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    action_type TEXT NOT NULL, -- 'page_view', 'symbol_search', 'btn_click', 'tool_usage'
                    action_detail TEXT, -- JSON or string
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data(symbol, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol ON ml_predictions(symbol, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_system_metrics_time ON system_metrics(timestamp, metric_name)",
                "CREATE INDEX IF NOT EXISTS idx_query_analytics_time ON query_analytics(timestamp, query_type)"
            ]
            
            # ============= NEW PROFESSIONAL FEATURES =============
            
            # Financial Models (DCF, 3-Statement)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pro_financial_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    model_type TEXT NOT NULL, -- 'DCF', 'LBO', '3_STATEMENT'
                    model_name TEXT,
                    assumptions TEXT, -- JSON
                    outputs TEXT, -- JSON
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Hypothetical Model Trades (Out-of-sample log)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pro_model_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    entry_time DATETIME NOT NULL,
                    exit_time DATETIME,
                    side TEXT NOT NULL, -- BUY/SELL/SHORT/COVER
                    entry_price REAL,
                    exit_price REAL,
                    pnl_pct REAL,
                    reasoning TEXT, -- Full AI explanation
                    regime_context TEXT, -- Bull/Bear when trade made
                    indicators_snapshot TEXT, -- JSON
                    is_hypothetical BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Daily Intelligence Reports
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pro_daily_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_date DATE NOT NULL,
                    asset_class TEXT, -- Stocks, Crypto, Forex, etc.
                    content TEXT, -- Markdown/HTML content
                    regime_summary TEXT,
                    top_picks TEXT, -- JSON
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(report_date, asset_class)
                )
            """)
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    @contextmanager
    def _get_connection(self):
        """Thread-safe database connection context manager."""
        with self.lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    # ============= MARKET DATA METHODS =============
    
    def store_market_data(self, symbol: str, data: pd.DataFrame, asset_type: str = "stock", 
                         data_source: str = "yfinance") -> bool:
        """Store market data with deduplication."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                records = []
                for timestamp, row in data.iterrows():
                    records.append((
                        symbol, asset_type, timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        float(row.get('Open', 0)), float(row.get('High', 0)),
                        float(row.get('Low', 0)), float(row.get('Close', 0)),
                        int(row.get('Volume', 0)), data_source
                    ))
                
                cursor.executemany("""
                    INSERT OR REPLACE INTO market_data 
                    (symbol, asset_type, timestamp, open_price, high_price, low_price, 
                     close_price, volume, data_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                
                conn.commit()
                logger.info(f"Stored {len(records)} records for {symbol}")
                return True
        except Exception as e:
            logger.error(f"Error storing market data for {symbol}: {e}")
            return False
    
    def get_market_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Retrieve market data for a symbol."""
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT timestamp, open_price, high_price, low_price, close_price, volume
                    FROM market_data 
                    WHERE symbol = ? AND timestamp >= datetime('now', '-{} days')
                    ORDER BY timestamp
                """.format(days)
                
                df = pd.read_sql_query(query, conn, params=(symbol,))
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                return df
        except Exception as e:
            logger.error(f"Error retrieving market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def store_technical_indicators(self, symbol: str, indicators: pd.DataFrame) -> bool:
        """Store technical indicators."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                records = []
                for timestamp, row in indicators.iterrows():
                    records.append((
                        symbol, timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        row.get('rsi'), row.get('ema20'), row.get('ema50'),
                        row.get('macd'), row.get('macd_signal'),
                        row.get('bb_upper'), row.get('bb_middle'), row.get('bb_lower'),
                        row.get('volume_sma'), row.get('atr')
                    ))
                
                cursor.executemany("""
                    INSERT OR REPLACE INTO technical_indicators 
                    (symbol, timestamp, rsi, ema20, ema50, macd, macd_signal,
                     bb_upper, bb_middle, bb_lower, volume_sma, atr)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error storing indicators for {symbol}: {e}")
            return False
    
    def store_ml_prediction(self, symbol: str, prediction: Dict[str, Any]) -> bool:
        """Store ML prediction results."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO ml_predictions 
                    (symbol, timestamp, signal, confidence, bullish_prob, model_version,
                     prediction_horizon, signal_factors, model_scores)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    prediction.get('signal', 'NEUTRAL'),
                    prediction.get('confidence', 0.5),
                    prediction.get('bullish_prob', 0.5),
                    prediction.get('model_version', 'v1.0'),
                    prediction.get('prediction_horizon', 5),
                    json.dumps(prediction.get('signal_factors', [])),
                    json.dumps(prediction.get('model_scores', {}))
                ))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error storing ML prediction for {symbol}: {e}")
            return False
    
    # ============= CONVERSATION METHODS =============
    
    def store_conversation(self, session_id: str, query: str, response: Dict[str, Any],
                          user_id: str = None, response_time_ms: int = None) -> int:
        """Store conversation with full context."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO conversations 
                    (session_id, user_id, query, response, intent_detected, symbols_mentioned,
                     charts_generated, response_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, user_id, query,
                    response.get('text', ''),
                    json.dumps(response.get('intents', [])),
                    json.dumps(response.get('symbols', [])),
                    len(response.get('charts', [])),
                    response_time_ms
                ))
                
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
            return -1
    
    def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Retrieve conversation history for context."""
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT query, response, intent_detected, symbols_mentioned, timestamp
                    FROM conversations 
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                
                cursor = conn.cursor()
                cursor.execute(query, (session_id, limit))
                
                history = []
                for row in cursor.fetchall():
                    history.append({
                        'query': row['query'],
                        'response': row['response'],
                        'intents': json.loads(row['intent_detected'] or '[]'),
                        'symbols': json.loads(row['symbols_mentioned'] or '[]'),
                        'timestamp': row['timestamp']
                    })
                
                return list(reversed(history))  # Return in chronological order
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    # ============= ANALYTICS METHODS =============
    
    def log_system_metric(self, metric_name: str, value: float, unit: str = None, 
                         category: str = "performance"):
        """Log system performance metrics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO system_metrics (metric_name, metric_value, metric_unit, category)
                    VALUES (?, ?, ?, ?)
                """, (metric_name, value, unit, category))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging metric {metric_name}: {e}")
    
    def log_query_analytics(self, query_type: str, symbol: str = None, 
                           response_time_ms: int = None, success: bool = True, 
                           error_message: str = None):
        """Log query analytics for performance monitoring."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO query_analytics 
                    (query_type, symbol, response_time_ms, success, error_message)
                    VALUES (?, ?, ?, ?, ?)
                """, (query_type, symbol, response_time_ms, success, error_message))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging query analytics: {e}")
    
    def get_analytics_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive analytics dashboard data."""
        try:
            with self._get_connection() as conn:
                analytics = {}
                
                # Query volume by type
                query = """
                    SELECT query_type, COUNT(*) as count, AVG(response_time_ms) as avg_time
                    FROM query_analytics 
                    WHERE timestamp >= datetime('now', '-7 days')
                    GROUP BY query_type
                    ORDER BY count DESC
                """
                df_queries = pd.read_sql_query(query, conn)
                analytics['query_volume'] = df_queries.to_dict('records')
                
                # Most requested symbols
                query = """
                    SELECT symbol, COUNT(*) as requests
                    FROM query_analytics 
                    WHERE symbol IS NOT NULL AND timestamp >= datetime('now', '-7 days')
                    GROUP BY symbol
                    ORDER BY requests DESC
                    LIMIT 10
                """
                df_symbols = pd.read_sql_query(query, conn)
                analytics['popular_symbols'] = df_symbols.to_dict('records')
                
                # System performance
                query = """
                    SELECT metric_name, AVG(metric_value) as avg_value, metric_unit
                    FROM system_metrics 
                    WHERE timestamp >= datetime('now', '-24 hours')
                    GROUP BY metric_name, metric_unit
                """
                df_metrics = pd.read_sql_query(query, conn)
                analytics['system_performance'] = df_metrics.to_dict('records')
                
                # Conversation stats
                query = """
                    SELECT 
                        COUNT(*) as total_conversations,
                        COUNT(DISTINCT session_id) as unique_sessions,
                        AVG(response_time_ms) as avg_response_time,
                        AVG(charts_generated) as avg_charts_per_query
                    FROM conversations 
                    WHERE timestamp >= datetime('now', '-7 days')
                """
                cursor = conn.cursor()
                cursor.execute(query)
                row = cursor.fetchone()
                analytics['conversation_stats'] = dict(row) if row else {}
                
                return analytics
        except Exception as e:
            logger.error(f"Error generating analytics dashboard: {e}")
            # Return empty structure to prevent dashboard crash
            return {
                'query_volume': [],
                'popular_symbols': [],
                'system_performance': [],
                'conversation_stats': {
                    'total_conversations': 0,
                    'unique_sessions': 0,
                    'avg_response_time': 0,
                    'avg_charts_per_query': 0
                }
            }
    
    # ============= USER MANAGEMENT METHODS =============
    
    def log_user_activity(self, user_id: str, action_type: str, action_detail: Any = None) -> bool:
        """Log user interaction for adaptive personalization."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                detail_str = json.dumps(action_detail) if action_detail is not None else None
                cursor.execute("""
                    INSERT INTO user_activity_logs (user_id, action_type, action_detail)
                    VALUES (?, ?, ?)
                """, (user_id, action_type, detail_str))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error logging user activity: {e}")
            return False

    def get_user_activity(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Retrieve recent user activity logs."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT action_type, action_detail, timestamp 
                    FROM user_activity_logs 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (user_id, limit))
                
                logs = []
                for row in cursor.fetchall():
                    logs.append({
                        'action_type': row['action_type'],
                        'detail': json.loads(row['action_detail']) if row['action_detail'] else None,
                        'timestamp': row['timestamp']
                    })
                return logs
        except Exception as e:
            logger.error(f"Error retrieving user activity: {e}")
            return []

    def create_user_profile(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Create or update user profile."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO user_profiles 
                    (user_id, preferred_assets, risk_tolerance, trading_style, 
                     notification_preferences, dashboard_layout, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    json.dumps(preferences.get('preferred_assets', [])),
                    preferences.get('risk_tolerance', 'medium'),
                    preferences.get('trading_style', 'balanced'),
                    json.dumps(preferences.get('notifications', {})),
                    json.dumps(preferences.get('dashboard_layout', {})),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error creating user profile: {e}")
            return False
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user profile and preferences."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM user_profiles WHERE user_id = ?
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'user_id': row['user_id'],
                        'preferred_assets': json.loads(row['preferred_assets'] or '[]'),
                        'risk_tolerance': row['risk_tolerance'],
                        'trading_style': row['trading_style'],
                        'notifications': json.loads(row['notification_preferences'] or '{}'),
                        'dashboard_layout': json.loads(row['dashboard_layout'] or '{}'),
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at']
                    }
                return {}
        except Exception as e:
            logger.error(f"Error retrieving user profile: {e}")
            return {}
    
    # ============= MAINTENANCE METHODS =============
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to maintain performance."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Clean old market data (keep more recent data)
                cursor.execute("""
                    DELETE FROM market_data 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(days_to_keep * 2))
                
                # Clean old conversations (keep less)
                cursor.execute("""
                    DELETE FROM conversations 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(days_to_keep))
                
                # Clean old analytics
                cursor.execute("""
                    DELETE FROM query_analytics 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(days_to_keep // 2))
                
                cursor.execute("""
                    DELETE FROM system_metrics 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(days_to_keep // 2))
                
                conn.commit()
                logger.info("Database cleanup completed")
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Table sizes
                tables = [
                    'market_data', 'technical_indicators', 'ml_predictions',
                    'conversations', 'user_profiles', 'system_metrics',
                    'query_analytics', 'watchlists', 'price_alerts'
                ]
                
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                
                # Database file size
                db_path = Path(self.db_path)
                if db_path.exists():
                    stats['db_size_mb'] = db_path.stat().st_size / (1024 * 1024)
                
                # Recent activity
                cursor.execute("""
                    SELECT COUNT(*) FROM conversations 
                    WHERE timestamp >= datetime('now', '-24 hours')
                """)
                stats['conversations_24h'] = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT COUNT(DISTINCT symbol) FROM market_data 
                    WHERE timestamp >= datetime('now', '-24 hours')
                """)
                stats['active_symbols_24h'] = cursor.fetchone()[0]
                
                return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}


# Global database manager instance
_db_manager = None

def get_database_manager() -> DatabaseManager:
    """Get or create database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager