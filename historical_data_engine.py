"""
Robust Historical Financial Data Engine
Multi-source data fetching with EODHD API and CSV fallbacks
Author: APB - Octavian Team

Data-integrity contract:
- Financial statements are ONLY sourced from real providers (EODHD / yfinance).
- Price-only CSV fallbacks are used for OHLCV market data, never to fabricate
  revenue, EBIT, or FCF figures.
- Every result carries a 'source' tag and a 'data_quality' flag so callers and
  the UI can distinguish REAL data from ESTIMATED/unavailable data.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from pathlib import Path
import requests
import warnings

warnings.filterwarnings('ignore')


class HistoricalDataEngine:
    """Multi-source historical financial data engine with robust fallbacks."""

    def __init__(self):
        # EODHD API key is read from environment / Streamlit secrets ONLY.
        from config import EODHD_API_KEY

        self.eodhd_api_key = EODHD_API_KEY
        self.eodhd_base_url = "https://eodhistoricaldata.com/api"

        # Data cache directory — project-relative so the app is portable.
        self.cache_dir = Path(__file__).resolve().parent / "data_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Dataset sources
        self.dataset_sources = {
            'nasdaq': self.cache_dir / "nasdaq_historical",
            'kaggle': self.cache_dir / "kaggle_stocks",
            'local': self.cache_dir / "local_cache",
        }

        # Create source directories
        for source_dir in self.dataset_sources.values():
            source_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #

    def get_historical_financials(self, ticker: str, years: int = 10) -> Dict[str, Any]:
        """
        Fetch historical financials with multiple fallbacks.
        Priority: EODHD API → yfinance (real statement data only).
        Price-only CSV datasets cannot produce financial statements and are
        therefore never used to fabricate them.
        """
        # Method 1: EODHD API (primary source)
        try:
            result = self._fetch_from_eodhd(ticker, years)
            if result and not result['data'].empty:
                return result
        except Exception as e:
            print(f"EODHD fetch failed for {ticker}: {e}")

        # Method 2: yfinance fallback (real statement data)
        try:
            result = self._fetch_from_yfinance(ticker, years)
            if result and not result['data'].empty:
                return result
        except Exception as e:
            print(f"yfinance fetch failed for {ticker}: {e}")

        # Method 3: Return an explicitly-labeled empty structure. No synthetic
        # financials are invented when real data is unavailable.
        return self._get_empty_historical()

    def get_historical_prices(self, ticker: str, years: int = 10) -> Dict[str, Any]:
        """
        Fetch historical OHLCV prices from any available source (EODHD, local
        CSVs, or yfinance). Prices are real data; provenance is always tagged.
        """
        # Method 1: local CSV price caches (fast, no network)
        try:
            result = self._fetch_price_from_csv(ticker, years)
            if result and not result['data'].empty:
                return result
        except Exception as e:
            print(f"CSV price fetch failed for {ticker}: {e}")

        # Method 2: EODHD historical endpoint
        try:
            result = self._fetch_price_from_eodhd(ticker, years)
            if result and not result['data'].empty:
                return result
        except Exception as e:
            print(f"EODHD price fetch failed for {ticker}: {e}")

        # Method 3: yfinance
        try:
            result = self._fetch_price_from_yfinance(ticker, years)
            if result and not result['data'].empty:
                return result
        except Exception as e:
            print(f"yfinance price fetch failed for {ticker}: {e}")

        return {'data': pd.DataFrame(), 'stats': {}, 'source': 'None (insufficient data)',
                'data_quality': 'UNAVAILABLE'}

    # ------------------------------------------------------------------ #
    # FINANCIALS — real statement data only
    # ------------------------------------------------------------------ #

    def _fetch_from_eodhd(self, ticker: str, years: int = 10) -> Dict[str, Any]:
        """Fetch real fundamentals from EODHD API (premium source)."""
        if not self.eodhd_api_key:
            return None
        try:
            # Clean ticker format for EODHD
            clean_ticker = ticker.replace('-USD', '').replace('=F', '').replace('=X', '')
            exchange = self._determine_exchange(ticker)

            # Fundamentals endpoint for financial data
            url = f"{self.eodhd_base_url}/fundamentals/{clean_ticker}.{exchange}"
            params = {
                'api_token': self.eodhd_api_key,
                'fmt': 'json'
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return None

            data = response.json()

            # Extract financials from EODHD response
            financials = data.get('Financials', {})
            income_statement = financials.get('Income_Statement', {})
            balance_sheet = financials.get('Balance_Sheet', {})
            cash_flow = financials.get('Cash_Flow', {})

            if not income_statement:
                return None

            # Parse annual data
            rows = []
            for date_str, values in income_statement.get('yearly', {}).items():
                try:
                    year = datetime.strptime(date_str, '%Y-%m-%d').year

                    revenue = values.get('totalRevenue', 0) or values.get('revenue', 0)
                    if revenue == 0:
                        continue

                    ebit = values.get('ebit', 0) or values.get('operatingIncome', 0)
                    ebit_margin = (ebit / revenue * 100) if revenue > 0 else 0

                    # Real cash flow from operations when available
                    fcf = None
                    if cash_flow:
                        for cdate, cvals in cash_flow.get('yearly', {}).items():
                            if cdate == date_str:
                                ocf = cvals.get('operatingCashFlow', 0) or cvals.get('cashFlowFromContinuingOperatingActivities', 0)
                                capex = cvals.get('capitalExpenditure', 0) or cvals.get('capex', 0)
                                if ocf:
                                    fcf = (ocf - abs(capex)) if capex else ocf
                                break

                    rows.append({
                        'Year': str(year),
                        'Revenue': revenue / 1e6,  # Millions
                        'EBIT Margin %': ebit_margin,
                        'FCF Margin %': (fcf / revenue * 100) if (fcf is not None and revenue > 0) else None,
                    })
                except Exception:
                    continue

            if not rows:
                return None

            # Sort by year and take last N years
            rows.sort(key=lambda x: x['Year'], reverse=True)
            rows = rows[:years]

            df = pd.DataFrame(rows)

            # Calculate growth rates
            if len(df) >= 2:
                df = df.sort_values('Year')
                df['Revenue Growth %'] = df['Revenue'].pct_change() * 100
                df = df.sort_values('Year', ascending=False)

            # FCF margin is only present when the provider reported real cash flow
            has_fcf = 'FCF Margin %' in df.columns and df['FCF Margin %'].notna().any()
            stats = {
                'avg_revenue_growth': df['Revenue Growth %'].mean() if 'Revenue Growth %' in df.columns else 0,
                'avg_ebit_margin': df['EBIT Margin %'].mean(),
                'avg_fcf_margin': df['FCF Margin %'].mean() if has_fcf else 0,
                'std_revenue_growth': df['Revenue Growth %'].std() if 'Revenue Growth %' in df.columns else 0,
                'has_real_fcf': has_fcf,
            }

            # Cache result for offline reuse
            self._cache_data(ticker, df)

            return {'data': df, 'stats': stats, 'source': 'EODHD API',
                    'data_quality': 'REAL' if has_fcf else 'REAL_PARTIAL'}

        except Exception as e:
            print(f"EODHD API error for {ticker}: {e}")
            return None

    def _fetch_from_yfinance(self, ticker: str, years: int = 10) -> Dict[str, Any]:
        """Fetch real financial statements via yfinance."""
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            financials = t.financials
            cashflow = getattr(t, 'cashflow', None)

            if financials is None or financials.empty:
                return None

            rows = []
            revenue_series = financials.loc['Total Revenue'] if 'Total Revenue' in financials.index else None
            ebit_series = None
            for label in ('EBIT', 'Operating Income'):
                if label in financials.index:
                    ebit_series = financials.loc[label]
                    break

            if revenue_series is None:
                return None

            for year_date in revenue_series.index[:years]:
                try:
                    rev = revenue_series.get(year_date, 0)
                    if rev is None or rev == 0:
                        continue

                    ebit_val = None
                    if ebit_series is not None:
                        ebit_val = ebit_series.get(year_date, 0)
                    ebit_margin = (ebit_val / rev * 100) if (ebit_val and rev > 0) else None

                    # Real free cash flow from the cash-flow statement when present
                    fcf_val = None
                    if cashflow is not None and not cashflow.empty:
                        for label in ('Free Cash Flow', 'Operating Cash Flow'):
                            if label in cashflow.index:
                                cf_series = cashflow.loc[label]
                                v = cf_series.get(year_date, 0)
                                if v and v != 0:
                                    fcf_val = v
                                    break

                    rows.append({
                        'Year': str(year_date)[:4],
                        'Revenue': rev / 1e6,
                        'EBIT Margin %': ebit_margin,
                        'FCF Margin %': (fcf_val / rev * 100) if (fcf_val is not None and rev > 0) else None,
                    })
                except Exception:
                    continue

            if not rows:
                return None

            df = pd.DataFrame(rows)
            df = df.sort_values('Year', ascending=False)

            if len(df) >= 2:
                temp_df = df.sort_values('Year')
                temp_df['Revenue Growth %'] = temp_df['Revenue'].pct_change() * 100
                df = temp_df.sort_values('Year', ascending=False)

            has_fcf = 'FCF Margin %' in df.columns and df['FCF Margin %'].notna().any()
            stats = {
                'avg_revenue_growth': df['Revenue Growth %'].mean() if 'Revenue Growth %' in df.columns else 0,
                'avg_ebit_margin': df['EBIT Margin %'].mean(),
                'avg_fcf_margin': df['FCF Margin %'].mean() if has_fcf else 0,
                'std_revenue_growth': df['Revenue Growth %'].std() if 'Revenue Growth %' in df.columns else 0,
                'has_real_fcf': has_fcf,
            }

            # Cache for future use
            self._cache_data(ticker, df)

            return {'data': df, 'stats': stats, 'source': 'yfinance',
                    'data_quality': 'REAL' if has_fcf else 'REAL_PARTIAL'}

        except Exception as e:
            print(f"yfinance error for {ticker}: {e}")
            return None

    # ------------------------------------------------------------------ #
    # PRICES — real OHLCV only
    # ------------------------------------------------------------------ #

    def _fetch_price_from_csv(self, ticker: str, years: int = 10) -> Dict[str, Any]:
        """Fetch real OHLCV price history from local CSV caches."""
        for source_name in ('local', 'nasdaq', 'kaggle'):
            src_dir = self.dataset_sources[source_name]
            candidates = [src_dir / f"{ticker}_historical.csv", src_dir / f"{ticker}.csv"]
            for path in candidates:
                if path.exists():
                    try:
                        df = pd.read_csv(path)
                        if 'Date' not in df.columns or 'Close' not in df.columns:
                            continue
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                        df = df.dropna(subset=['Date']).set_index('Date').sort_index()
                        if len(df) > 20:
                            return {'data': df, 'stats': {}, 'source': f'CSV ({source_name})',
                                    'data_quality': 'REAL'}
                    except Exception:
                        continue
        return None

    def _fetch_price_from_eodhd(self, ticker: str, years: int = 10) -> Dict[str, Any]:
        """Fetch real OHLCV price history from EODHD."""
        if not self.eodhd_api_key:
            return None
        try:
            clean_ticker = ticker.replace('-USD', '').replace('=F', '').replace('=X', '')
            exchange = self._determine_exchange(ticker)
            url = f"{self.eodhd_base_url}/eod/{clean_ticker}.{exchange}"
            params = {
                'api_token': self.eodhd_api_key,
                'fmt': 'json',
                'period': 'd',
                'from': (datetime.now() - timedelta(days=int(years * 365.25))).strftime('%Y-%m-%d'),
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return None
            payload = response.json()
            rows = []
            for item in payload:
                try:
                    rows.append({
                        'Date': item.get('date'),
                        'Open': item.get('open'),
                        'High': item.get('high'),
                        'Low': item.get('low'),
                        'Close': item.get('close'),
                        'Volume': item.get('volume'),
                    })
                except Exception:
                    continue
            if not rows:
                return None
            df = pd.DataFrame(rows)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).set_index('Date').sort_index()
            self._cache_data(ticker, df)
            return {'data': df, 'stats': {}, 'source': 'EODHD API', 'data_quality': 'REAL'}
        except Exception as e:
            print(f"EODHD price error for {ticker}: {e}")
            return None

    def _fetch_price_from_yfinance(self, ticker: str, years: int = 10) -> Dict[str, Any]:
        """Fetch real OHLCV price history via yfinance."""
        try:
            import yfinance as yf
            df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False, threads=False)
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).title() for c in df.columns]
            if 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            self._cache_data(ticker, df)
            return {'data': df, 'stats': {}, 'source': 'yfinance', 'data_quality': 'REAL'}
        except Exception as e:
            print(f"yfinance price error for {ticker}: {e}")
            return None

    # ------------------------------------------------------------------ #
    # HELPERS
    # ------------------------------------------------------------------ #

    def _determine_exchange(self, ticker: str) -> str:
        """Determine exchange code for EODHD API."""
        if any(x in ticker for x in ['-USD', 'BTC', 'ETH']):
            return 'CC'  # Crypto
        elif '=F' in ticker:
            return 'COMM'  # Commodities/Futures
        elif '=X' in ticker:
            return 'FOREX'
        else:
            return 'US'  # Default to US stocks

    def _cache_data(self, ticker: str, df: pd.DataFrame):
        """Cache data locally for future use."""
        try:
            cache_file = self.dataset_sources['local'] / f"{ticker}_historical.csv"
            df.to_csv(cache_file, index=False)
        except Exception:
            pass

    def _get_empty_historical(self) -> Dict[str, Any]:
        """Return an explicitly-unavailable structure when all real sources fail."""
        return {
            'data': pd.DataFrame(columns=['Year', 'Revenue', 'Revenue Growth %', 'EBIT Margin %', 'FCF Margin %']),
            'stats': {
                'avg_revenue_growth': 0,
                'avg_ebit_margin': 0,
                'avg_fcf_margin': 0,
                'std_revenue_growth': 0,
                'has_real_fcf': False,
            },
            'source': 'None (insufficient data)',
            'data_quality': 'UNAVAILABLE',
        }


# Global singleton
_historical_engine = None


def get_historical_engine() -> HistoricalDataEngine:
    """Get or create historical data engine singleton."""
    global _historical_engine
    if _historical_engine is None:
        _historical_engine = HistoricalDataEngine()
    return _historical_engine
