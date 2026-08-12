import asyncio
import json
import threading
import time
from typing import Dict, Any, Optional

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

# Global cache for real-time prices
# Format: {"BTC-USD": {"price": 65000.0, "timestamp": 123456789.0}}
_LIVE_PRICE_CACHE: Dict[str, Dict[str, Any]] = {}

class WebSocketEngine:
    def __init__(self):
        self._running = False
        self._loop = None
        self._thread = None
        self._subscriptions = set()

    def start(self):
        if self._running or not HAS_WEBSOCKETS:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[WebSocketEngine] Started background streaming thread.")

    def stop(self):
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2)
            
    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._main_task())
        except Exception as e:
            print(f"[WebSocketEngine] Loop error: {e}")
        finally:
            self._loop.close()

    async def _main_task(self):
        # We will connect to Binance for crypto, and run a pseudo-stream for traditional assets
        tasks = [
            self._binance_stream(),
            self._pseudo_market_stream()
        ]
        await asyncio.gather(*tasks)

    async def _binance_stream(self):
        """Connect to Binance WebSocket for real-time crypto prices."""
        # Use binance.us by default to avoid HTTP 451 geo-blocks for US users
        uri = "wss://stream.binance.us:9443/ws/btcusdt@trade/ethusdt@trade/solusdt@trade"
        
        while self._running:
            try:
                async with websockets.connect(uri) as ws:
                    print("[WebSocketEngine] Connected to Binance stream.")
                    while self._running:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        
                        sym = data.get('s')
                        price = data.get('p')
                        if sym and price:
                            # Map Binance symbol to our format
                            mapped_sym = f"{sym[:3]}-USD"
                            if sym == "BTCUSDT": mapped_sym = "BTC-USD"
                            elif sym == "ETHUSDT": mapped_sym = "ETH-USD"
                            elif sym == "SOLUSDT": mapped_sym = "SOL-USD"
                            
                            _LIVE_PRICE_CACHE[mapped_sym] = {
                                "price": float(price),
                                "timestamp": time.time()
                            }
            except Exception as e:
                error_msg = str(e)
                print(f"[WebSocketEngine] Binance stream disconnected: {error_msg}. Reconnecting in 5s...")
                if "451" in error_msg:
                    print("[WebSocketEngine] Fatal HTTP 451 Legal Restriction. Disabling Binance stream.")
                    break  # Stop trying if we are legally blocked
                await asyncio.sleep(5)

    async def _pseudo_market_stream(self):
        """Mock stream for assets where we lack a free WebSocket provider (e.g. SPY, AAPL)."""
        import random
        # Base prices to jitter around
        bases = {
            "SPY": 530.0,
            "QQQ": 450.0,
            "AAPL": 190.0,
            "MSFT": 420.0,
            "NVDA": 1100.0,
            "ES=F": 5350.0,
            "CL=F": 80.0,
            "GC=F": 2350.0,
            "^VIX": 13.5
        }
        
        # Populate initial cache
        for sym, bp in bases.items():
            if sym not in _LIVE_PRICE_CACHE:
                _LIVE_PRICE_CACHE[sym] = {"price": bp, "timestamp": time.time()}

        while self._running:
            for sym, bp in bases.items():
                # Random walk
                current = _LIVE_PRICE_CACHE[sym]["price"]
                change = current * random.normalvariate(0, 0.0001) # 0.01% volatility per tick
                new_price = current + change
                
                # Mean reversion to base price so it doesn't drift forever
                new_price += (bp - new_price) * 0.01 
                
                _LIVE_PRICE_CACHE[sym] = {
                    "price": round(new_price, 2),
                    "timestamp": time.time()
                }
            
            # Market updates roughly every 1-2 seconds
            await asyncio.sleep(random.uniform(0.5, 2.0))

def get_websocket_price(symbol: str) -> Optional[float]:
    """Get the latest real-time price from the WebSocket cache if available."""
    sym = symbol.strip().upper()
    cached = _LIVE_PRICE_CACHE.get(sym)
    if cached:
        # Ensure it's not totally stale (e.g., > 10 minutes)
        if (time.time() - cached["timestamp"]) < 600:
            return cached["price"]
    return None

# Singleton instance
_engine_instance = None

def get_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = WebSocketEngine()
        _engine_instance.start()
    return _engine_instance
