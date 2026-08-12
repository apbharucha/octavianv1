"""
Octavian High-Velocity Discovery Engine (OHVDE)
==============================================
The primary backbone for all market scanning and trade discovery within the Octavian Terminal.
Optimized for ultra-low latency, multi-asset class depth, and institutional fidelity.

Architecture:
1. Parallel I/O Layer: Async batch data retrieval.
2. Rapid Pulse Screener (RPS): Vectorized statistical filtering (NumPy).
3. Intelligence Nexus: Tiered analysis (Technical -> Bayesian ML).
4. Global Discovery Cache: Shared cross-component data persistence.
"""

import numpy as np
import pandas as pd
import asyncio
import concurrent.futures
import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import yfinance as yf

# Singleton access to specialized engines
from data_sources import get_stock
from quant_ensemble_model import get_quant_ensemble

class OctavianDiscoveryEngine:
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OctavianDiscoveryEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self.logger = logging.getLogger("OctavianDiscovery")
        self.quant = get_quant_ensemble()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=30)
        self._l1_cache = {} # Key: symbol, Value: {timestamp, data, scores}
        self._initialized = True

    @classmethod
    async def get_instance(cls):
        async with cls._lock:
            if cls._instance is None:
                cls._instance = OctavianDiscoveryEngine()
            return cls._instance

    async def scan_market_pulse(self, symbols: List[str], deep_scan: bool = False) -> List[Dict[str, Any]]:
        """
        Scans a list of symbols with ultra-high velocity.
        Tiered approach ensures we only run heavy models on high-potential candidates.
        """
        start_time = time.time()
        
        # 1. Parallel Batch Data Retrieval
        data_map = await self._fetch_batch_data_async(symbols)
        
        # 2. Rapid Pulse Screener (RPS) - Statistical Tier
        candidates = self._run_rapid_pulse_screener(data_map) if deep_scan else symbols
        
        # 3. Intelligence Nexus - Deep Analysis Tier
        tasks = []
        for s in candidates:
            df = data_map.get(s)
            if df is not None:
                tasks.append(self._deep_analyze_symbol(s, df, deep_scan))
        
        results = await asyncio.gather(*tasks)
        valid_results = [r for r in results if r is not None]
        
        elapsed = time.time() - start_time
        self.logger.info(f"Market Pulse Scan completed in {elapsed:.2f}s for {len(symbols)} symbols.")
        
        return valid_results

    async def _fetch_batch_data_async(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Asynchronously retrieves batch data using yfinance threads and caching."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._fetch_batch_data_sync, symbols)

    def _fetch_batch_data_sync(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Synchronous batch fetcher for use in executor.

        Downloads in chunks (150 symbols per yf.download call) so full-universe
        scans of thousands of symbols don't fail or hang on a single giant
        request. Each chunk is flattened, then merged into one data_map.
        """
        data_map = {}
        symbols = [s for s in symbols if s]
        self.logger.info(f"Fetching batch data for {len(symbols)} symbols (chunked)...")

        def _extract_chunk(chunk: List[str]) -> Dict[str, pd.DataFrame]:
            out = {}
            try:
                df_all = yf.download(chunk, period="3mo", interval="1d",
                                     group_by='ticker', threads=False, progress=False)
                if df_all is None or df_all.empty:
                    return out
                for s in chunk:
                    try:
                        if len(chunk) == 1:
                            sym_df = df_all.copy()
                        else:
                            if isinstance(df_all.columns, pd.MultiIndex):
                                if s in df_all.columns.get_level_values(0):
                                    sym_df = df_all[s].copy()
                                else:
                                    continue
                            elif s in df_all.columns:
                                sym_df = df_all[[s]].copy()
                            else:
                                continue
                        # Flatten any remaining MultiIndex columns
                        if isinstance(sym_df.columns, pd.MultiIndex):
                            sym_df.columns = sym_df.columns.get_level_values(-1)
                        sym_df = sym_df.dropna(how='all')
                        if not sym_df.empty and 'Close' in sym_df.columns:
                            out[s] = sym_df
                    except Exception as e:
                        self.logger.debug(f"Symbol {s} extraction failed: {e}")
            except Exception as e:
                self.logger.error(f"Chunk fetch error: {e}")
            return out

        # Download chunks in parallel (bounded) so full-universe scans of
        # thousands of symbols don't serialize into many slow sequential
        # yfinance requests. _extract_chunk is thread-safe (pure per-chunk work).
        _CHUNK = 150
        _MAX_CONCURRENT = 5
        chunks = [symbols[i:i + _CHUNK] for i in range(0, len(symbols), _CHUNK)]
        if len(chunks) <= 1:
            for chunk in chunks:
                data_map.update(_extract_chunk(chunk))
        else:
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=_MAX_CONCURRENT) as pool:
                futures = {pool.submit(_extract_chunk, c): c for c in chunks}
                for fut in _cf.as_completed(futures, timeout=180):
                    try:
                        data_map.update(fut.result(timeout=60))
                    except Exception:
                        continue

        # Individual fallback for symbols that chunked download missed
        if len(data_map) < len(symbols):
            missing = [s for s in symbols if s not in data_map]
            for s in missing[:50]:  # cap fallback to avoid pathological slowness
                try:
                    df = get_stock(s, period="3mo")
                    if df is not None and not df.empty:
                        data_map[s] = df
                except Exception:
                    pass
        return data_map

    @staticmethod
    def _safe_close(df: pd.DataFrame) -> Optional[np.ndarray]:
        """Return a clean 1-D float Close array, or None if unavailable.

        Handles yfinance MultiIndex columns, DataFrame-wrapped series, missing
        'Close' columns (e.g. after a partial/failed download), and empty data.
        This is the single guard used by every analysis path so a single bad
        symbol can never raise a KeyError and abort the whole scan.
        """
        if df is None or df.empty:
            return None
        close = None
        # MultiIndex columns (yfinance group_by='ticker' / single-ticker) —
        # flatten to the price-field level, then look for 'Close'.
        if isinstance(df.columns, pd.MultiIndex):
            _flat = df.copy()
            _FIELD_HINTS = ('close', 'open', 'high', 'low', 'volume', 'adj')
            _flat.columns = [
                str(next((p for p in c if isinstance(p, str) and p.lower() in _FIELD_HINTS), c[-1]))
                for c in _flat.columns
            ]
            if 'Close' in _flat.columns:
                close = _flat['Close']
            elif any('close' in str(c).lower() for c in _flat.columns):
                for _c in _flat.columns:
                    if 'close' in str(_c).lower():
                        close = _flat[_c]
                        break
        if close is None:
            close = df.get('Close')
        if close is None:
            # Fallback: first column whose name contains 'close'
            for col in df.columns:
                name = str(col)
                if 'close' in name.lower():
                    close = df[col]
                    break
        if close is None:
            return None
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        try:
            arr = pd.to_numeric(close, errors='coerce').dropna().astype(float).values
        except Exception:
            return None
        return arr if len(arr) else None

    def _run_rapid_pulse_screener(self, data_map: Dict[str, pd.DataFrame]) -> List[str]:
        """Tier 1: Nanosecond vectorized pre-filtering."""
        candidates = []
        for s, df in data_map.items():
            if df is None or df.empty or len(df) < 10:
                continue
            c = self._safe_close(df)
            if c is None or len(c) < 5:
                continue
            # Rapid Volatility/Momentum Check
            roc = (c[-1] / c[-5] - 1) * 100 if len(c) >= 5 else 0
            # Impulse check
            if abs(roc) > 1.5:
                candidates.append(s)
            elif 'Volume' in df.columns:
                v = df['Volume'].values
                if v[-1] > np.mean(v[-10:]) * 1.5:
                    candidates.append(s)
        return candidates

    async def _deep_analyze_symbol(self, symbol: str, df: pd.DataFrame, deep: bool) -> Optional[Dict[str, Any]]:
        """Tier 2: Technical and Bayesian ML synthesis."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._deep_analyze_sync, symbol, df, deep)

    def _deep_analyze_sync(self, symbol: str, df: pd.DataFrame, deep: bool) -> Optional[Dict[str, Any]]:
        """Deep analysis logic run in thread pool."""
        try:
            close = self._safe_close(df)
            if close is None or len(close) < 2:
                return None
            current_price = float(close[-1])
            prev_price = float(close[-2]) if len(close) >= 2 else current_price
            
            # 1. Technical Scorecard
            inds = self._calculate_technical_card(df)
            
            # 2. Heavy ML Inference (only if deep or technicals are hot)
            signal_data = {}
            if deep or abs(inds['score']) > 30:
                signal = self.quant.predict(close, symbol=symbol)
                signal_data = {
                    'direction': signal.direction,
                    'confidence': signal.confidence,
                    'expected_return': signal.expected_return,
                    'decision': signal.direction
                }
            
            # 3. Synthesis
            return {
                'symbol': symbol,
                'price': current_price,
                'change_1d': ((current_price / prev_price) - 1) * 100,
                'score': inds['score'],
                'indicators': inds,
                'signal': signal_data,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Deep analysis failed for {symbol}: {e}")
            return None

    def _calculate_technical_card(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Vectorized technical computation for score-carding."""
        c = self._safe_close(df)
        if c is None or len(c) < 10:
            return {'score': 0, 'rsi': 50, 'momentum': 0, 'volatility': 0,
                    'volume_spike': False, 'ema_trend': 'FLAT'}
        rsi = self._fast_rsi(c)
        
        score = 0
        if rsi > 70: score -= 20
        elif rsi < 30: score += 20
        
        # Momentum alignment
        m10 = (c[-1] / c[-10] - 1) * 100 if len(c) >= 10 else 0
        score += np.clip(m10 * 3, -40, 40)

        # Annualized realized volatility (20-day, %) — real measurement, not a placeholder
        vol_20 = 0.0
        if len(c) >= 21:
            rets = np.diff(c[-21:]) / c[-21:-1]
            vol_20 = float(np.std(rets, ddof=1) * np.sqrt(252) * 100) if len(rets) > 1 else 0.0

        return {
            'rsi': rsi,
            'momentum_10d': m10,
            'score': score,
            'above_sma50': c[-1] > np.mean(c[-50:]) if len(c) >= 50 else None,
            'volatility_20d': round(vol_20, 2)
        }

    def _fast_rsi(self, prices, n=14):
        if len(prices) < n+1: return 50.0
        deltas = np.diff(prices)
        seed = deltas[:n+1]
        up = seed[seed >= 0].sum()/n
        down = -seed[seed < 0].sum()/n
        if down == 0: return 100.0
        rs = up/down
        return 100. - 100./(1.+rs)

def get_discovery_engine():
    # Helper for synchronous call from Streamlit
    return OctavianDiscoveryEngine()
