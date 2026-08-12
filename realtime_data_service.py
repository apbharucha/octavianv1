"""
Real-Time Market Data Streaming Service

This module provides real-time market data streaming capabilities:
1. WebSocket connections for live data feeds
2. Automated data collection and caching
3. Real-time price alerts and notifications
4. Market event detection and broadcasting
5. Performance monitoring and optimization

Author: AI Market Team
"""

import asyncio
import websockets
import json
import random
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import schedule
from dataclasses import dataclass
import numpy as np
import pandas as pd

from database_manager import get_database_manager
from data_sources import get_stock, get_fx, get_futures_proxy
from ai_chatbot import get_chatbot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PriceAlert:
    """Price alert configuration."""
    user_id: str
    symbol: str
    alert_type: str  # 'price_above', 'price_below', 'change_percent'
    threshold: float
    is_active: bool = True
    created_at: datetime = None

@dataclass
class MarketEvent:
    """Market event data structure."""
    event_type: str
    symbol: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    data: Dict[str, Any]
    timestamp: datetime

class RealTimeDataService:
    """Real-time market data streaming and alert service."""

    @staticmethod
    def _build_default_watchlist() -> Set[str]:
        """Default monitored symbols drawn from the dynamic ticker universe.

        A seeded, deterministic sample is used so the default set is stable
        between restarts while still reflecting current universe coverage.
        """
        symbols: Set[str] = {"^VIX"}
        try:
            from ticker_universe import get_ticker_universe
            u = get_ticker_universe()
            rng = random.Random(7)
            stocks = sorted(u.get_all_stocks())
            symbols.update(rng.sample(stocks, min(8, len(stocks))))
            symbols.update(sorted(u.get_etfs())[:5])
            symbols.update(sorted(u.get_crypto())[:3])
            symbols.update(sorted(u.get_forex())[:3])
        except Exception:
            pass
        return symbols

    def __init__(self):
        self.db_manager = get_database_manager()
        self.chatbot = get_chatbot()
        
        # WebSocket connections
        self.websocket_clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Subscriptions: symbol -> set of client websockets
        self.subscriptions: Dict[str, Set[websockets.WebSocketServerProtocol]] = {}
        
        # Price alerts
        self.price_alerts: List[PriceAlert] = []
        
        # Market data cache
        self.market_cache: Dict[str, Dict[str, Any]] = {}
        
        # Watchlist of symbols to monitor (drawn from the dynamic ticker universe)
        self.watchlist: Set[str] = self._build_default_watchlist()
        
        # Event detection thresholds
        self.event_thresholds = {
            'large_move': 0.05,  # 5% price change
            'volume_spike': 2.0,  # 2x average volume
            'volatility_spike': 1.5,  # 1.5x average volatility
            'gap_threshold': 0.02  # 2% gap
        }
        
        # Background tasks
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.running = False
        
        # Performance metrics
        self.metrics = {
            'data_updates': 0,
            'alerts_triggered': 0,
            'events_detected': 0,
            'websocket_messages': 0,
            'last_update': None
        }
    
    async def start_websocket_server(self, host: str = 'localhost', port: int = 8765):
        """Start WebSocket server for real-time data streaming."""
        logger.info(f"Starting WebSocket server on {host}:{port}")
        
        async def handle_client(websocket, path):
            """Handle individual WebSocket client connections."""
            self.websocket_clients.add(websocket)
            client_ip = websocket.remote_address[0] if websocket.remote_address else 'unknown'
            logger.info(f"New WebSocket client connected: {client_ip}")
            
            try:
                # Send welcome message
                await websocket.send(json.dumps({
                    'type': 'welcome',
                    'message': 'Connected to AI Market Data Stream',
                    'timestamp': datetime.now().isoformat(),
                    'available_symbols': list(self.watchlist)
                }))
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self.handle_websocket_message(websocket, data)
                    except json.JSONDecodeError:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'Invalid JSON format'
                        }))
                    except Exception as e:
                        logger.error(f"Error handling WebSocket message: {e}")
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': str(e)
                        }))
            
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"WebSocket client disconnected: {client_ip}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.websocket_clients.discard(websocket)
                # Remove from all subscriptions
                for symbol_subs in self.subscriptions.values():
                    symbol_subs.discard(websocket)
        
        # Start WebSocket server
        server = await websockets.serve(handle_client, host, port)
        logger.info("WebSocket server started successfully")
        return server
    
    async def handle_websocket_message(self, websocket, data: Dict[str, Any]):
        """Handle incoming WebSocket messages from clients."""
        message_type = data.get('type')
        
        if message_type == 'subscribe':
            symbols = data.get('symbols', [])
            for symbol in symbols:
                if symbol not in self.subscriptions:
                    self.subscriptions[symbol] = set()
                self.subscriptions[symbol].add(websocket)
                
                # Add to watchlist if not already there
                self.watchlist.add(symbol)
            
            await websocket.send(json.dumps({
                'type': 'subscription_confirmed',
                'symbols': symbols,
                'timestamp': datetime.now().isoformat()
            }))
            
            logger.info(f"Client subscribed to: {symbols}")
        
        elif message_type == 'unsubscribe':
            symbols = data.get('symbols', [])
            for symbol in symbols:
                if symbol in self.subscriptions:
                    self.subscriptions[symbol].discard(websocket)
                    if not self.subscriptions[symbol]:
                        del self.subscriptions[symbol]
            
            await websocket.send(json.dumps({
                'type': 'unsubscription_confirmed',
                'symbols': symbols,
                'timestamp': datetime.now().isoformat()
            }))
        
        elif message_type == 'get_quote':
            symbol = data.get('symbol')
            if symbol:
                quote_data = await self.get_real_time_quote(symbol)
                await websocket.send(json.dumps({
                    'type': 'quote',
                    'symbol': symbol,
                    'data': quote_data,
                    'timestamp': datetime.now().isoformat()
                }))
        
        elif message_type == 'create_alert':
            alert = PriceAlert(
                user_id=data.get('user_id', 'anonymous'),
                symbol=data.get('symbol'),
                alert_type=data.get('alert_type'),
                threshold=data.get('threshold'),
                created_at=datetime.now()
            )
            
            self.price_alerts.append(alert)
            
            await websocket.send(json.dumps({
                'type': 'alert_created',
                'alert_id': len(self.price_alerts) - 1,
                'message': f'Alert created for {alert.symbol}',
                'timestamp': datetime.now().isoformat()
            }))
        
        elif message_type == 'ping':
            await websocket.send(json.dumps({
                'type': 'pong',
                'timestamp': datetime.now().isoformat()
            }))
        
        self.metrics['websocket_messages'] += 1
    
    async def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a symbol."""
        try:
            # Check cache first
            if symbol in self.market_cache:
                cached_data = self.market_cache[symbol]
                cache_age = (datetime.now() - cached_data['timestamp']).total_seconds()
                
                if cache_age < 60:  # Use cache if less than 1 minute old
                    return cached_data['data']
            
            # Fetch fresh data
            if '/' in symbol or '_' in symbol:
                df = get_fx(symbol.replace('/', '_'))
            elif '=F' in symbol:
                df = get_futures_proxy(symbol, period='5d')
            else:
                df = get_stock(symbol, period='5d')
            
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest
                
                quote_data = {
                    'symbol': symbol,
                    'price': float(latest['Close']),
                    'change': float(latest['Close'] - prev['Close']),
                    'change_percent': float((latest['Close'] / prev['Close'] - 1) * 100) if prev['Close'] > 0 else 0,
                    'volume': int(latest.get('Volume', 0)),
                    'high': float(latest['High']),
                    'low': float(latest['Low']),
                    'open': float(latest['Open']),
                    'timestamp': df.index[-1].isoformat()
                }
                
                # Update cache
                self.market_cache[symbol] = {
                    'data': quote_data,
                    'timestamp': datetime.now()
                }
                
                return quote_data
            else:
                return {'error': f'No data available for {symbol}'}
        
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return {'error': str(e)}
    
    async def broadcast_to_subscribers(self, symbol: str, data: Dict[str, Any]):
        """Broadcast data to all subscribers of a symbol."""
        if symbol in self.subscriptions:
            message = json.dumps({
                'type': 'market_update',
                'symbol': symbol,
                'data': data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Send to all subscribers
            disconnected_clients = set()
            
            for websocket in self.subscriptions[symbol]:
                try:
                    await websocket.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(websocket)
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {e}")
                    disconnected_clients.add(websocket)
            
            # Clean up disconnected clients
            for client in disconnected_clients:
                self.subscriptions[symbol].discard(client)
                self.websocket_clients.discard(client)
    
    async def broadcast_market_event(self, event: MarketEvent):
        """Broadcast market event to all connected clients."""
        message = json.dumps({
            'type': 'market_event',
            'event': {
                'event_type': event.event_type,
                'symbol': event.symbol,
                'description': event.description,
                'severity': event.severity,
                'data': event.data,
                'timestamp': event.timestamp.isoformat()
            }
        })
        
        disconnected_clients = set()
        
        for websocket in self.websocket_clients:
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting event: {e}")
                disconnected_clients.add(websocket)
        
        # Clean up disconnected clients
        for client in disconnected_clients:
            self.websocket_clients.discard(client)
        
        self.metrics['events_detected'] += 1
    
    def start_background_tasks(self):
        """Start background data collection and monitoring tasks."""
        self.running = True
        
        # Schedule regular data updates
        schedule.every(1).minutes.do(self.update_market_data)
        schedule.every(5).minutes.do(self.check_price_alerts)
        schedule.every(10).minutes.do(self.detect_market_events)
        schedule.every(1).hours.do(self.cleanup_cache)
        
        # Start scheduler in background thread
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(1)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("Background tasks started")
    
    def update_market_data(self):
        """Update market data for all watchlist symbols."""
        logger.info(f"Updating market data for {len(self.watchlist)} symbols")
        
        def update_symbol(symbol):
            try:
                # Get fresh data
                quote_data = asyncio.run(self.get_real_time_quote(symbol))
                
                if 'error' not in quote_data:
                    # Store in database
                    df = pd.DataFrame([{
                        'Open': quote_data['open'],
                        'High': quote_data['high'],
                        'Low': quote_data['low'],
                        'Close': quote_data['price'],
                        'Volume': quote_data['volume']
                    }], index=[pd.Timestamp.now()])
                    
                    asset_type = 'fx' if '/' in symbol else 'futures' if '=F' in symbol else 'stock'
                    self.db_manager.store_market_data(symbol, df, asset_type)
                    
                    # Broadcast to subscribers
                    if symbol in self.subscriptions:
                        asyncio.run(self.broadcast_to_subscribers(symbol, quote_data))
                    
                    self.metrics['data_updates'] += 1
            
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")
        
        # Update symbols in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(update_symbol, list(self.watchlist))
        
        self.metrics['last_update'] = datetime.now()
    
    def check_price_alerts(self):
        """Check and trigger price alerts."""
        if not self.price_alerts:
            return
        
        logger.info(f"Checking {len(self.price_alerts)} price alerts")
        
        for alert in self.price_alerts:
            if not alert.is_active:
                continue
            
            try:
                # Get current price
                quote_data = asyncio.run(self.get_real_time_quote(alert.symbol))
                
                if 'error' in quote_data:
                    continue
                
                current_price = quote_data['price']
                triggered = False
                
                if alert.alert_type == 'price_above' and current_price > alert.threshold:
                    triggered = True
                elif alert.alert_type == 'price_below' and current_price < alert.threshold:
                    triggered = True
                elif alert.alert_type == 'change_percent':
                    change_pct = abs(quote_data.get('change_percent', 0))
                    if change_pct > alert.threshold:
                        triggered = True
                
                if triggered:
                    # Trigger alert
                    alert_message = {
                        'type': 'price_alert',
                        'alert': {
                            'symbol': alert.symbol,
                            'alert_type': alert.alert_type,
                            'threshold': alert.threshold,
                            'current_price': current_price,
                            'user_id': alert.user_id,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                    
                    # Broadcast alert
                    asyncio.run(self.broadcast_alert(alert_message))
                    
                    # Deactivate alert (one-time trigger)
                    alert.is_active = False
                    
                    self.metrics['alerts_triggered'] += 1
                    logger.info(f"Price alert triggered for {alert.symbol}")
            
            except Exception as e:
                logger.error(f"Error checking alert for {alert.symbol}: {e}")
    
    async def broadcast_alert(self, alert_message: Dict[str, Any]):
        """Broadcast price alert to all connected clients."""
        message = json.dumps(alert_message)
        
        for websocket in self.websocket_clients:
            try:
                await websocket.send(message)
            except Exception as e:
                logger.error(f"Error broadcasting alert: {e}")
    
    def detect_market_events(self):
        """Detect significant market events and broadcast them."""
        logger.info("Detecting market events")
        
        for symbol in list(self.watchlist)[:10]:  # Limit to top 10 symbols
            try:
                # Get recent data
                if '/' in symbol:
                    df = get_fx(symbol.replace('/', '_'))
                elif '=F' in symbol:
                    df = get_futures_proxy(symbol, period='5d')
                else:
                    df = get_stock(symbol, period='5d')
                
                if df is None or df.empty or len(df) < 2:
                    continue
                
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                
                # Calculate metrics
                price_change = (latest['Close'] / prev['Close'] - 1) * 100
                
                events = []
                
                # Large price movement
                if abs(price_change) > self.event_thresholds['large_move'] * 100:
                    direction = 'surge' if price_change > 0 else 'drop'
                    severity = 'high' if abs(price_change) > 10 else 'medium'
                    
                    events.append(MarketEvent(
                        event_type='large_price_move',
                        symbol=symbol,
                        description=f'{symbol} {direction} of {price_change:.2f}%',
                        severity=severity,
                        data={
                            'price_change_percent': price_change,
                            'current_price': float(latest['Close']),
                            'previous_price': float(prev['Close'])
                        },
                        timestamp=datetime.now()
                    ))
                
                # Volume spike (if volume data available)
                if 'Volume' in df.columns and len(df) >= 20:
                    avg_volume = df['Volume'].tail(20).mean()
                    current_volume = latest['Volume']
                    
                    if current_volume > avg_volume * self.event_thresholds['volume_spike']:
                        events.append(MarketEvent(
                            event_type='volume_spike',
                            symbol=symbol,
                            description=f'{symbol} volume spike: {current_volume/avg_volume:.1f}x average',
                            severity='medium',
                            data={
                                'current_volume': int(current_volume),
                                'average_volume': int(avg_volume),
                                'volume_ratio': float(current_volume / avg_volume)
                            },
                            timestamp=datetime.now()
                        ))
                
                # Gap detection
                gap_percent = (latest['Open'] / prev['Close'] - 1) * 100
                if abs(gap_percent) > self.event_thresholds['gap_threshold'] * 100:
                    gap_type = 'gap_up' if gap_percent > 0 else 'gap_down'
                    
                    events.append(MarketEvent(
                        event_type=gap_type,
                        symbol=symbol,
                        description=f'{symbol} {gap_type.replace("_", " ")} of {gap_percent:.2f}%',
                        severity='medium',
                        data={
                            'gap_percent': gap_percent,
                            'open_price': float(latest['Open']),
                            'previous_close': float(prev['Close'])
                        },
                        timestamp=datetime.now()
                    ))
                
                # Broadcast events
                for event in events:
                    asyncio.run(self.broadcast_market_event(event))
            
            except Exception as e:
                logger.error(f"Error detecting events for {symbol}: {e}")
    
    def cleanup_cache(self):
        """Clean up old cache entries."""
        current_time = datetime.now()
        expired_symbols = []
        
        for symbol, cache_data in self.market_cache.items():
            cache_age = (current_time - cache_data['timestamp']).total_seconds()
            if cache_age > 3600:  # Remove entries older than 1 hour
                expired_symbols.append(symbol)
        
        for symbol in expired_symbols:
            del self.market_cache[symbol]
        
        if expired_symbols:
            logger.info(f"Cleaned up {len(expired_symbols)} expired cache entries")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics."""
        return {
            'metrics': self.metrics.copy(),
            'active_connections': len(self.websocket_clients),
            'active_subscriptions': len(self.subscriptions),
            'watchlist_size': len(self.watchlist),
            'cache_size': len(self.market_cache),
            'active_alerts': len([a for a in self.price_alerts if a.is_active]),
            'timestamp': datetime.now().isoformat()
        }
    
    def stop(self):
        """Stop the real-time data service."""
        self.running = False
        logger.info("Real-time data service stopped")

# Global service instance
_realtime_service = None

def get_realtime_service() -> RealTimeDataService:
    """Get or create real-time data service instance."""
    global _realtime_service
    if _realtime_service is None:
        _realtime_service = RealTimeDataService()
    return _realtime_service

# Main execution for standalone service
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time Market Data Service')
    parser.add_argument('--host', default='localhost', help='WebSocket host')
    parser.add_argument('--port', type=int, default=8765, help='WebSocket port')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start service
    service = get_realtime_service()
    
    # Start background tasks
    service.start_background_tasks()
    
    # Start WebSocket server
    async def main():
        server = await service.start_websocket_server(args.host, args.port)
        
        logger.info(f"Real-time data service running on ws://{args.host}:{args.port}")
        logger.info("Press Ctrl+C to stop")
        
        try:
            await server.wait_closed()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            service.stop()
    
    # Run the service
    asyncio.run(main())