"""
API Rate Limiter and Request Handler

Provides rate limiting, retry logic, and caching for external API calls.
Fixes Reddit 403 and Yahoo RSS 429 errors with proper throttling.

Author: APB - Octavian Team
"""

import time
import requests
from typing import Dict, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import logging
from functools import wraps
import hashlib
import json

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_window: int):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply rate limiting."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                
                # Remove old calls outside time window
                self.calls = [call_time for call_time in self.calls 
                             if now - call_time < self.time_window]
                
                # Check if we've hit the limit
                if len(self.calls) >= self.max_calls:
                    # Calculate wait time
                    oldest_call = self.calls[0]
                    wait_time = self.time_window - (now - oldest_call)
                    
                    if wait_time > 0:
                        logger.info(f"Rate limit reached, waiting {wait_time:.2f}s")
                        time.sleep(wait_time + 0.1)  # Add small buffer
                        
                        # Clean up again after waiting
                        now = time.time()
                        self.calls = [call_time for call_time in self.calls 
                                     if now - call_time < self.time_window]
                
                # Record this call
                self.calls.append(now)
            
            return func(*args, **kwargs)
        
        return wrapper


class ResponseCache:
    """Simple in-memory cache for API responses."""
    
    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize cache.
        
        Args:
            ttl_seconds: Time to live for cached responses
        """
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached response if not expired."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry['timestamp'] < self.ttl_seconds:
                    logger.debug(f"Cache hit for {key}")
                    return entry['data']
                else:
                    # Expired, remove it
                    del self.cache[key]
        return None
    
    def set(self, key: str, data: Any):
        """Store response in cache."""
        with self.lock:
            self.cache[key] = {
                'data': data,
                'timestamp': time.time()
            }
            logger.debug(f"Cached response for {key}")
    
    def clear(self):
        """Clear all cached data."""
        with self.lock:
            self.cache.clear()
    
    def cleanup_expired(self):
        """Remove expired entries."""
        with self.lock:
            now = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if now - entry['timestamp'] >= self.ttl_seconds
            ]
            for key in expired_keys:
                del self.cache[key]
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


class APIRequestHandler:
    """Handles API requests with rate limiting, retry logic, and caching."""
    
    def __init__(self):
        # Rate limiters for different services
        self.reddit_limiter = RateLimiter(max_calls=60, time_window=60)  # 60 calls per minute
        self.yahoo_limiter = RateLimiter(max_calls=30, time_window=60)   # 30 calls per minute
        self.generic_limiter = RateLimiter(max_calls=100, time_window=60)  # 100 calls per minute
        
        # Response caches
        self.reddit_cache = ResponseCache(ttl_seconds=300)  # 5 minutes
        self.yahoo_cache = ResponseCache(ttl_seconds=300)   # 5 minutes
        self.rss_cache = ResponseCache(ttl_seconds=600)     # 10 minutes
        
        # User agents for different services
        self.user_agents = {
            'reddit': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'yahoo': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'generic': 'Mozilla/5.0 (compatible; Octavian-AI/1.0; +https://octavian.ai)'
        }
    
    @staticmethod
    def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        """Calculate exponential backoff delay."""
        delay = min(base_delay * (2 ** attempt), max_delay)
        # Add jitter to prevent thundering herd
        jitter = delay * 0.1 * (0.5 - time.time() % 1)
        return delay + jitter
    
    def fetch_with_retry(self, url: str, service: str = 'generic', 
                        max_retries: int = 3, timeout: int = 10,
                        params: Optional[Dict] = None,
                        headers: Optional[Dict] = None) -> Optional[requests.Response]:
        """
        Fetch URL with retry logic and exponential backoff.
        
        Args:
            url: URL to fetch
            service: Service type ('reddit', 'yahoo', 'generic')
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            params: URL parameters
            headers: Additional headers
            
        Returns:
            Response object or None if all retries failed
        """
        # Generate cache key
        cache_key = hashlib.md5(f"{url}{json.dumps(params or {})}".encode()).hexdigest()
        
        # Check cache first
        if service == 'reddit':
            cached = self.reddit_cache.get(cache_key)
            if cached:
                return cached
        elif service == 'yahoo':
            cached = self.yahoo_cache.get(cache_key)
            if cached:
                return cached
        elif service == 'rss':
            cached = self.rss_cache.get(cache_key)
            if cached:
                return cached
        
        # Prepare headers
        request_headers = headers or {}
        if 'User-Agent' not in request_headers:
            request_headers['User-Agent'] = self.user_agents.get(service, self.user_agents['generic'])
        
        # Apply rate limiting
        if service == 'reddit':
            self.reddit_limiter(lambda: None)()
        elif service == 'yahoo':
            self.yahoo_limiter(lambda: None)()
        else:
            self.generic_limiter(lambda: None)()
        
        # Retry loop
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=request_headers,
                    timeout=timeout
                )
                
                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited by {service}, waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                
                # Check for forbidden
                if response.status_code == 403:
                    logger.warning(f"403 Forbidden from {service} on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        delay = self.exponential_backoff(attempt)
                        logger.info(f"Retrying after {delay:.2f}s")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"All retries exhausted for {service}")
                        return None
                
                # Success
                if response.status_code == 200:
                    # Cache successful response
                    if service == 'reddit':
                        self.reddit_cache.set(cache_key, response)
                    elif service == 'yahoo':
                        self.yahoo_cache.set(cache_key, response)
                    elif service == 'rss':
                        self.rss_cache.set(cache_key, response)
                    
                    return response
                
                # Other errors
                logger.warning(f"HTTP {response.status_code} from {service}")
                if attempt < max_retries - 1:
                    delay = self.exponential_backoff(attempt)
                    time.sleep(delay)
                    continue
                
                return None
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout fetching from {service} (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    delay = self.exponential_backoff(attempt)
                    time.sleep(delay)
                    continue
                return None
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error from {service}: {e}")
                if attempt < max_retries - 1:
                    delay = self.exponential_backoff(attempt)
                    time.sleep(delay)
                    continue
                return None
        
        return None
    
    def cleanup_caches(self):
        """Clean up expired cache entries."""
        self.reddit_cache.cleanup_expired()
        self.yahoo_cache.cleanup_expired()
        self.rss_cache.cleanup_expired()


# Global instance
_api_handler = None


def get_api_handler() -> APIRequestHandler:
    """Get global API request handler instance."""
    global _api_handler
    if _api_handler is None:
        _api_handler = APIRequestHandler()
    return _api_handler
