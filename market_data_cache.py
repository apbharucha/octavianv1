import time
import pandas as pd
import threading
from typing import Dict, Any, Optional

class MarketDataCache:
    """
    Institutional-grade thread-safe singleton cache for cross-engine market data.
    Provides centralized state management to prevent redundant API calls.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MarketDataCache, cls).__new__(cls)
                cls._instance._cache = {}
                cls._instance._stats = {"hits": 0, "misses": 0}
        return cls._instance

    def get(self, key: str, ttl_seconds: int = 300) -> Optional[Any]:
        """Fetch item from cache if not expired."""
        with self._lock:
            item = self._cache.get(key)
            if item:
                ts, value = item
                if time.time() - ts < ttl_seconds:
                    self._stats["hits"] += 1
                    return value
            self._stats["misses"] += 1
            return None

    def set(self, key: str, value: Any):
        """Set item in cache with current timestamp."""
        with self._lock:
            self._cache[key] = (time.time(), value)

    def clear(self):
        """Clear all cached data."""
        with self._lock:
            self._cache = {}

    def get_stats(self) -> Dict[str, int]:
        """Return cache performance statistics."""
        return self._stats

_global_cache = MarketDataCache()

def get_market_cache() -> MarketDataCache:
    return _global_cache
