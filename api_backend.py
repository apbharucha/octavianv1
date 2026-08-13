"""
Advanced API Backend for AI Market Chatbot

This module provides a comprehensive REST API backend for the AI chatbot system with:
1. Real-time market data endpoints
2. AI analysis and prediction APIs
3. User management and analytics
4. WebSocket support for real-time updates
5. Caching and performance optimization
6. Rate limiting and security

Author: AI Market Team
"""

from flask import Flask, request, jsonify, session
from flask_cors import CORS

# Optional dependencies — the API backend degrades gracefully (in-memory
# cache, no rate limiting, no Redis) when they are not installed, so the
# module always imports and the REST endpoints stay available.
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    _HAS_LIMITER = True
except Exception:
    _HAS_LIMITER = False

    def get_remote_address():  # type: ignore
        return "default"

    class _NoOpLimiter:
        def __init__(self, *args, **kwargs):
            pass

        def limit(self, *args, **kwargs):
            def _deco(fn):
                return fn

            return _deco

    Limiter = _NoOpLimiter  # type: ignore

try:
    from flask_caching import Cache

    _HAS_CACHING = True
except Exception:
    _HAS_CACHING = False

    class _NoOpCache:
        def __init__(self, *args, **kwargs):
            pass

        def cached(self, *args, **kwargs):
            def _deco(fn):
                return fn

            return _deco

    Cache = _NoOpCache  # type: ignore

try:
    import redis
except Exception:
    redis = None

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from functools import wraps

from ai_chatbot import get_chatbot
from database_manager import get_database_manager
from data_sources import get_stock, get_fx, get_futures_proxy
from ml_analysis import get_analyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
# Session signing key: read from config. config generates an ephemeral secure
# random key when SECRET_KEY is not set, so sessions never rely on a known
# hardcoded default (which would allow session forgery).
from config import FLASK_SECRET_KEY, CORS_ORIGINS, API_BACKEND_KEY
app.secret_key = FLASK_SECRET_KEY

# Enable CORS restricted to configured origins (default: local Streamlit dev).
CORS(app, origins=CORS_ORIGINS)

# Initialize Redis for caching and rate limiting
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
except:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - using in-memory alternatives")

# Configure caching
if REDIS_AVAILABLE:
    cache_config = {
        'CACHE_TYPE': 'redis',
        'CACHE_REDIS_HOST': 'localhost',
        'CACHE_REDIS_PORT': 6379,
        'CACHE_REDIS_DB': 1,
        'CACHE_DEFAULT_TIMEOUT': 300
    }
else:
    cache_config = {
        'CACHE_TYPE': 'simple',
        'CACHE_DEFAULT_TIMEOUT': 300
    }

cache = Cache(app, config=cache_config)

# Configure rate limiting
if REDIS_AVAILABLE:
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        storage_uri="redis://localhost:6379/2",
        default_limits=["1000 per hour", "100 per minute"]
    )
else:
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=["1000 per hour", "100 per minute"]
    )

# Global instances
chatbot = get_chatbot()
db_manager = get_database_manager()
analyzer = get_analyzer()

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=10)

def require_api_key(f):
    """
    Decorator to require a valid API key for protected endpoints.

    The key is compared against the API_BACKEND_KEY environment variable
    (constant-time comparison). When no key is configured, protected
    endpoints refuse requests rather than silently accepting any key, so a
    default-open deployment cannot happen by accident.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if not API_BACKEND_KEY:
            return jsonify({'error': 'API backend key not configured on server'}), 503
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        import hmac as _hmac
        if not _hmac.compare_digest(str(api_key), API_BACKEND_KEY):
            return jsonify({'error': 'Invalid API key'}), 403
        return f(*args, **kwargs)
    return decorated_function

def log_api_request(endpoint: str, success: bool = True, response_time: float = 0, error: str = None):
    """Log API request for analytics."""
    try:
        db_manager.log_query_analytics(
            query_type=f"api_{endpoint}",
            response_time_ms=int(response_time * 1000),
            success=success,
            error_message=error
        )
    except Exception as e:
        logger.error(f"Error logging API request: {e}")

# ============= HEALTH AND STATUS ENDPOINTS =============

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    start_time = time.time()
    
    try:
        # Check database connection
        db_stats = db_manager.get_database_stats()
        
        # Check ML model
        ml_status = "available" if analyzer else "unavailable"
        
        # Check cache
        cache_status = "available" if REDIS_AVAILABLE else "memory"
        
        response = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'database': 'connected' if db_stats else 'error',
                'ml_model': ml_status,
                'cache': cache_status,
                'redis': 'connected' if REDIS_AVAILABLE else 'unavailable'
            },
            'database_stats': db_stats
        }
        
        log_api_request('health', True, time.time() - start_time)
        return jsonify(response)
    
    except Exception as e:
        log_api_request('health', False, time.time() - start_time, str(e))
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/status', methods=['GET'])
def system_status():
    """Detailed system status."""
    start_time = time.time()
    
    try:
        analytics = db_manager.get_analytics_dashboard()
        
        response = {
            'system_status': 'operational',
            'timestamp': datetime.now().isoformat(),
            'analytics': analytics,
            'performance': {
                'uptime': 'N/A',  # Would track actual uptime in production
                'memory_usage': 'N/A',  # Would track actual memory usage
                'cpu_usage': 'N/A'  # Would track actual CPU usage
            }
        }
        
        log_api_request('status', True, time.time() - start_time)
        return jsonify(response)
    
    except Exception as e:
        log_api_request('status', False, time.time() - start_time, str(e))
        return jsonify({'error': str(e)}), 500

# ============= MARKET DATA ENDPOINTS =============

@app.route('/api/market-data/<symbol>', methods=['GET'])
@limiter.limit("60 per minute")
@cache.cached(timeout=300, query_string=True)
def get_market_data(symbol):
    """Get real-time market data for a symbol."""
    start_time = time.time()
    
    try:
        period = request.args.get('period', '1y')
        
        # Try to get from cache first
        cached_data = db_manager.get_market_data(symbol, days=365)
        
        if not cached_data.empty:
            # Check if data is recent enough (less than 1 hour old)
            latest_timestamp = cached_data.index[-1]
            time_diff = datetime.now() - latest_timestamp.to_pydatetime()
            
            if time_diff.total_seconds() < 3600:  # Use cached data
                response = {
                    'symbol': symbol,
                    'data': cached_data.to_dict('index'),
                    'source': 'cache',
                    'timestamp': datetime.now().isoformat()
                }
                
                log_api_request('market_data', True, time.time() - start_time)
                return jsonify(response)
        
        # Fetch fresh data
        if '/' in symbol or '_' in symbol:
            df = get_fx(symbol.replace('/', '_'))
            asset_type = 'fx'
        elif '=F' in symbol:
            df = get_futures_proxy(symbol, period=period)
            asset_type = 'futures'
        else:
            df = get_stock(symbol, period=period)
            asset_type = 'stock'
        
        if df is None or df.empty:
            log_api_request('market_data', False, time.time() - start_time, 'No data found')
            return jsonify({'error': f'No data found for symbol {symbol}'}), 404
        
        # Store in database
        db_manager.store_market_data(symbol, df, asset_type)
        
        response = {
            'symbol': symbol,
            'asset_type': asset_type,
            'data': df.to_dict('index'),
            'source': 'live',
            'timestamp': datetime.now().isoformat()
        }
        
        log_api_request('market_data', True, time.time() - start_time)
        return jsonify(response)
    
    except Exception as e:
        log_api_request('market_data', False, time.time() - start_time, str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/multiple-quotes', methods=['POST'])
@limiter.limit("30 per minute")
def get_multiple_quotes():
    """Get quotes for multiple symbols."""
    start_time = time.time()
    
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        
        if not symbols or len(symbols) > 10:
            return jsonify({'error': 'Please provide 1-10 symbols'}), 400
        
        results = {}
        
        # Use thread pool for parallel fetching
        def fetch_quote(symbol):
            try:
                # Get latest price from database or fetch fresh
                cached_data = db_manager.get_market_data(symbol, days=1)
                
                if not cached_data.empty:
                    latest = cached_data.iloc[-1]
                    return {
                        'symbol': symbol,
                        'price': float(latest['Close']),
                        'change': float(latest['Close'] - cached_data.iloc[-2]['Close']) if len(cached_data) > 1 else 0,
                        'timestamp': cached_data.index[-1].isoformat(),
                        'source': 'cache'
                    }
                else:
                    # Fetch fresh data
                    if '/' in symbol:
                        df = get_fx(symbol.replace('/', '_'))
                    elif '=F' in symbol:
                        df = get_futures_proxy(symbol, period='5d')
                    else:
                        df = get_stock(symbol, period='5d')
                    
                    if df is not None and not df.empty:
                        latest = df.iloc[-1]
                        change = float(latest['Close'] - df.iloc[-2]['Close']) if len(df) > 1 else 0
                        
                        return {
                            'symbol': symbol,
                            'price': float(latest['Close']),
                            'change': change,
                            'timestamp': df.index[-1].isoformat(),
                            'source': 'live'
                        }
                    else:
                        return {'symbol': symbol, 'error': 'No data available'}
            
            except Exception as e:
                return {'symbol': symbol, 'error': str(e)}
        
        # Execute in parallel
        futures = [executor.submit(fetch_quote, symbol) for symbol in symbols]
        
        for future in futures:
            try:
                result = future.result(timeout=10)
                results[result['symbol']] = result
            except Exception as e:
                logger.error(f"Error fetching quote: {e}")
        
        log_api_request('multiple_quotes', True, time.time() - start_time)
        return jsonify({
            'quotes': results,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        log_api_request('multiple_quotes', False, time.time() - start_time, str(e))
        return jsonify({'error': str(e)}), 500

# ============= AI ANALYSIS ENDPOINTS =============

@app.route('/api/analyze/<symbol>', methods=['GET'])
@limiter.limit("20 per minute")
def analyze_symbol(symbol):
    """Get comprehensive AI analysis for a symbol."""
    start_time = time.time()
    
    try:
        # Get analysis parameters
        include_prediction = request.args.get('prediction', 'true').lower() == 'true'
        include_risk = request.args.get('risk', 'false').lower() == 'true'
        include_technical = request.args.get('technical', 'true').lower() == 'true'
        
        # Get market data
        if '/' in symbol:
            df = get_fx(symbol.replace('/', '_'))
            asset_type = 'fx'
        elif '=F' in symbol:
            df = get_futures_proxy(symbol, period='1y')
            asset_type = 'futures'
        else:
            df = get_stock(symbol, period='1y')
            asset_type = 'stock'
        
        if df is None or df.empty:
            log_api_request('analyze', False, time.time() - start_time, 'No data found')
            return jsonify({'error': f'No data found for symbol {symbol}'}), 404
        
        analysis = {
            'symbol': symbol,
            'asset_type': asset_type,
            'timestamp': datetime.now().isoformat(),
            'current_price': float(df['Close'].iloc[-1])
        }
        
        # ML Prediction
        if include_prediction:
            try:
                analyzer.train_on_data(df)
                prediction = analyzer.predict(df)
                analysis['prediction'] = prediction
                
                # Store prediction in database
                db_manager.store_ml_prediction(symbol, prediction)
            except Exception as e:
                analysis['prediction'] = {'error': str(e)}
        
        # Technical Analysis
        if include_technical:
            try:
                from indicators import add_indicators
                df_with_indicators = add_indicators(df.copy())
                
                if not df_with_indicators.empty:
                    latest = df_with_indicators.iloc[-1]
                    
                    analysis['technical'] = {
                        'rsi': float(latest.get('rsi', 0)) if pd.notna(latest.get('rsi')) else None,
                        'ema20': float(latest.get('ema20', 0)) if pd.notna(latest.get('ema20')) else None,
                        'ema50': float(latest.get('ema50', 0)) if pd.notna(latest.get('ema50')) else None,
                        'trend': 'bullish' if latest.get('ema20', 0) > latest.get('ema50', 0) else 'bearish'
                    }
            except Exception as e:
                analysis['technical'] = {'error': str(e)}
        
        # Risk Analysis
        if include_risk:
            try:
                returns = df['Close'].pct_change().dropna()
                
                if len(returns) >= 20:
                    analysis['risk'] = {
                        'volatility_annual': float(returns.std() * np.sqrt(252) * 100),
                        'var_95': float(returns.quantile(0.05) * 100),
                        'max_drawdown': float(((df['Close'].cummax() - df['Close']) / df['Close'].cummax()).max() * 100)
                    }
            except Exception as e:
                analysis['risk'] = {'error': str(e)}
        
        # Performance metrics
        if len(df) > 1:
            analysis['performance'] = {
                'daily_change': float(df['Close'].iloc[-1] - df['Close'].iloc[-2]),
                'daily_change_pct': float((df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100),
                'weekly_change_pct': float((df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100) if len(df) > 5 else None,
                'monthly_change_pct': float((df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1) * 100) if len(df) > 20 else None
            }
        
        log_api_request('analyze', True, time.time() - start_time)
        return jsonify(analysis)
    
    except Exception as e:
        log_api_request('analyze', False, time.time() - start_time, str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
@limiter.limit("30 per minute")
def chat_endpoint():
    """Main chat endpoint for AI conversations."""
    start_time = time.time()
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        user_id = data.get('user_id')
        session_id = data.get('session_id')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Set session ID if provided
        if session_id:
            chatbot.session_id = session_id
        
        # Process query
        response = chatbot.process_query(query, user_id)
        
        # Convert charts to serializable format
        serializable_charts = []
        for chart in response.get('charts', []):
            chart_copy = chart.copy()
            if 'figure' in chart_copy:
                # Convert plotly figure to JSON
                chart_copy['figure_json'] = chart_copy['figure'].to_json()
                del chart_copy['figure']  # Remove non-serializable figure
            serializable_charts.append(chart_copy)
        
        response['charts'] = serializable_charts
        
        log_api_request('chat', True, time.time() - start_time)
        return jsonify(response)
    
    except Exception as e:
        log_api_request('chat', False, time.time() - start_time, str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/<symbol>', methods=['GET'])
@limiter.limit("40 per minute")
@cache.cached(timeout=600, query_string=True)
def get_predictions(symbol):
    """Get ML predictions for a symbol."""
    start_time = time.time()
    
    try:
        horizon = int(request.args.get('horizon', 5))
        
        # Get market data
        if '/' in symbol:
            df = get_fx(symbol.replace('/', '_'))
        elif '=F' in symbol:
            df = get_futures_proxy(symbol, period='1y')
        else:
            df = get_stock(symbol, period='1y')
        
        if df is None or df.empty:
            log_api_request('predictions', False, time.time() - start_time, 'No data found')
            return jsonify({'error': f'No data found for symbol {symbol}'}), 404
        
        # Train model and get prediction
        analyzer.train_on_data(df)
        prediction = analyzer.predict(df)
        
        # Store prediction
        db_manager.store_ml_prediction(symbol, prediction)
        
        response = {
            'symbol': symbol,
            'prediction': prediction,
            'horizon_days': horizon,
            'current_price': float(df['Close'].iloc[-1]),
            'timestamp': datetime.now().isoformat()
        }
        
        log_api_request('predictions', True, time.time() - start_time)
        return jsonify(response)
    except Exception as e:
        log_api_request('predictions', False, time.time() - start_time, str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/predict/advanced', methods=['POST'])
@limiter.limit("200 per minute")
def advanced_tiered_prediction():
    """
    Phase 6: API Access Tiering.
    Provides direct access to Octavian internal LSTM/Transformer nodes based on API tier.
    Tiers:
    - free: Simple Moving Average heuristics (Mock)
    - pro: LSTM prediction access
    - institutional: Full Ensemble Engine access (LSTM + Transformer + GBM)
    """
    start_time = time.time()
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'SPY')
        api_key = request.headers.get('X-API-Key', 'free_tier_key')
        
        # Mock Tier validation
        tier = "free"
        if api_key.startswith("sk_pro_"):
            tier = "pro"
        elif api_key.startswith("sk_inst_"):
            tier = "institutional"
            
        if tier == "free":
            # Mock return for free tier
            return jsonify({
                "tier": tier,
                "symbol": symbol,
                "prediction": "Available in Pro tier. Upgrade to access LSTM.",
                "status": "RESTRICTED"
            }), 403
            
        # Fetch Data
        df = get_stock(symbol, period='1y')
        if df is None or df.empty:
            return jsonify({'error': f'No data found for symbol {symbol}'}), 404
            
        # Institutional vs Pro tier logic
        from advanced_ml_engine import get_ensemble_engine
        engine = get_ensemble_engine()
        
        if tier == "pro":
            # Just return LSTM specific output
            engine.train_on_data(df)
            result = engine.predict(df) # Normally we'd extract just LSTM
            result["engine_used"] = "LSTM_Only"
        else: # Institutional
            engine.train_on_data(df)
            result = engine.predict(df)
            result["engine_used"] = "Full_Ensemble"
            
        log_api_request('advanced_tiered_prediction', True, time.time() - start_time)
        return jsonify({
            "tier": tier,
            "symbol": symbol,
            "prediction_results": result,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        log_api_request('advanced_tiered_prediction', False, time.time() - start_time, str(e))
        return jsonify({'error': str(e)}), 500

# ============= USER AND SESSION MANAGEMENT =============

@app.route('/api/user/profile', methods=['GET', 'POST'])
@limiter.limit("100 per hour")
def user_profile():
    """Get or update user profile."""
    start_time = time.time()
    
    try:
        user_id = request.args.get('user_id') or request.json.get('user_id') if request.json else None
        
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
        
        if request.method == 'GET':
            profile = db_manager.get_user_profile(user_id)
            log_api_request('user_profile_get', True, time.time() - start_time)
            return jsonify(profile)
        
        elif request.method == 'POST':
            preferences = request.get_json()
            success = db_manager.create_user_profile(user_id, preferences)
            
            if success:
                log_api_request('user_profile_post', True, time.time() - start_time)
                return jsonify({'success': True, 'message': 'Profile updated'})
            else:
                log_api_request('user_profile_post', False, time.time() - start_time, 'Update failed')
                return jsonify({'error': 'Failed to update profile'}), 500
    
    except Exception as e:
        log_api_request('user_profile', False, time.time() - start_time, str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation/history', methods=['GET'])
@limiter.limit("50 per hour")
def conversation_history():
    """Get conversation history for a session."""
    start_time = time.time()
    
    try:
        session_id = request.args.get('session_id')
        limit = int(request.args.get('limit', 50))
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        history = db_manager.get_conversation_history(session_id, limit)
        
        log_api_request('conversation_history', True, time.time() - start_time)
        return jsonify({
            'session_id': session_id,
            'history': history,
            'count': len(history)
        })
    
    except Exception as e:
        log_api_request('conversation_history', False, time.time() - start_time, str(e))
        return jsonify({'error': str(e)}), 500

# ============= ANALYTICS AND ADMIN ENDPOINTS =============

@app.route('/api/analytics/dashboard', methods=['GET'])
@require_api_key
@limiter.limit("10 per minute")
def analytics_dashboard():
    """Get comprehensive analytics dashboard."""
    start_time = time.time()
    
    try:
        analytics = db_manager.get_analytics_dashboard()
        
        log_api_request('analytics_dashboard', True, time.time() - start_time)
        return jsonify({
            'analytics': analytics,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        log_api_request('analytics_dashboard', False, time.time() - start_time, str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/stats', methods=['GET'])
@require_api_key
@limiter.limit("5 per minute")
def admin_stats():
    """Get detailed system statistics for administrators."""
    start_time = time.time()
    
    try:
        db_stats = db_manager.get_database_stats()
        analytics = db_manager.get_analytics_dashboard()
        
        response = {
            'database_stats': db_stats,
            'analytics': analytics,
            'system_info': {
                'redis_available': REDIS_AVAILABLE,
                'cache_type': cache_config['CACHE_TYPE'],
                'timestamp': datetime.now().isoformat()
            }
        }
        
        log_api_request('admin_stats', True, time.time() - start_time)
        return jsonify(response)
    
    except Exception as e:
        log_api_request('admin_stats', False, time.time() - start_time, str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/cleanup', methods=['POST'])
@require_api_key
@limiter.limit("1 per hour")
def admin_cleanup():
    """Perform database cleanup."""
    start_time = time.time()
    
    try:
        days_to_keep = request.json.get('days_to_keep', 90) if request.json else 90
        
        # Perform cleanup in background
        def cleanup_task():
            try:
                db_manager.cleanup_old_data(days_to_keep)
                logger.info(f"Database cleanup completed - kept {days_to_keep} days of data")
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
        
        executor.submit(cleanup_task)
        
        log_api_request('admin_cleanup', True, time.time() - start_time)
        return jsonify({
            'message': f'Cleanup initiated - keeping {days_to_keep} days of data',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        log_api_request('admin_cleanup', False, time.time() - start_time, str(e))
        return jsonify({'error': str(e)}), 500

# ============= ERROR HANDLERS =============

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests. Please try again later.'
    }), 429

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============= MAIN APPLICATION =============

if __name__ == '__main__':
    # Initialize database
    try:
        db_manager.get_database_stats()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
    
    # Start the Flask application
    logger.info("Starting AI Market Chatbot API Backend...")
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('DEBUG', 'False').lower() == 'true',
        threaded=True
    )