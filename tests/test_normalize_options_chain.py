"""
Unit tests for _normalize_options_chain (data_sources.py).

Covers:
- All required columns present (no-op path)
- 'vol' absent but 'volume' present → renamed to 'vol'
- Multiple columns missing → filled with 0
- Empty / None input → returns empty DataFrame

Requirements: 3.1, 3.2, 3.3
"""

import pandas as pd
import pytest

from data_sources import _normalize_options_chain

REQUIRED_COLS = ['vol', 'strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility', 'openInterest']


def _full_row() -> dict:
    """Return a dict with every required column populated."""
    return {
        'strike': 150.0,
        'lastPrice': 5.0,
        'bid': 4.9,
        'ask': 5.1,
        'impliedVolatility': 0.25,
        'openInterest': 1000,
        'vol': 200,
    }


# ---------------------------------------------------------------------------
# 1. All columns present — data should pass through unchanged
# ---------------------------------------------------------------------------

def test_all_columns_present_returns_same_values():
    df = pd.DataFrame([_full_row()])
    result = _normalize_options_chain(df, 'AAPL')

    assert list(result['vol']) == [200]
    assert list(result['strike']) == [150.0]
    assert list(result['lastPrice']) == [5.0]
    assert list(result['bid']) == [4.9]
    assert list(result['ask']) == [5.1]
    assert list(result['impliedVolatility']) == [0.25]
    assert list(result['openInterest']) == [1000]


def test_all_columns_present_schema_complete():
    df = pd.DataFrame([_full_row()])
    result = _normalize_options_chain(df, 'AAPL')
    for col in REQUIRED_COLS:
        assert col in result.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# 2. 'vol' absent but 'volume' present → rename, no zeros inserted
# ---------------------------------------------------------------------------

def test_volume_renamed_to_vol():
    row = _full_row()
    del row['vol']
    row['volume'] = 300
    df = pd.DataFrame([row])

    result = _normalize_options_chain(df, 'TSLA')

    assert 'vol' in result.columns
    assert 'volume' not in result.columns
    assert list(result['vol']) == [300]


def test_volume_renamed_preserves_other_columns():
    row = _full_row()
    del row['vol']
    row['volume'] = 300
    df = pd.DataFrame([row])

    result = _normalize_options_chain(df, 'TSLA')

    for col in REQUIRED_COLS:
        assert col in result.columns


# ---------------------------------------------------------------------------
# 3. 'vol' absent and 'volume' absent → zero-filled + warning logged
# ---------------------------------------------------------------------------

def test_vol_missing_no_volume_defaults_to_zero(caplog):
    row = _full_row()
    del row['vol']
    df = pd.DataFrame([row])

    import logging
    with caplog.at_level(logging.WARNING, logger='root'):
        result = _normalize_options_chain(df, 'SPY')

    assert 'vol' in result.columns
    assert list(result['vol']) == [0]
    assert any('vol column missing' in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# 4. Multiple required columns missing → all filled with 0
# ---------------------------------------------------------------------------

def test_multiple_columns_missing_filled_with_zero():
    # Only provide strike; everything else is absent
    df = pd.DataFrame([{'strike': 100.0}])
    result = _normalize_options_chain(df, 'NVDA')

    for col in REQUIRED_COLS:
        assert col in result.columns, f"Missing column after normalization: {col}"

    assert list(result['lastPrice']) == [0]
    assert list(result['bid']) == [0]
    assert list(result['ask']) == [0]
    assert list(result['impliedVolatility']) == [0.0]
    assert list(result['openInterest']) == [0]
    assert list(result['vol']) == [0]


def test_multiple_columns_missing_does_not_raise():
    df = pd.DataFrame([{'strike': 50.0, 'bid': 1.0}])
    # Should not raise KeyError or any other exception
    result = _normalize_options_chain(df, 'AMD')
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# 5. Empty / None input → returns empty DataFrame
# ---------------------------------------------------------------------------

def test_empty_dataframe_returns_empty():
    result = _normalize_options_chain(pd.DataFrame(), 'AAPL')
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_none_input_returns_empty():
    result = _normalize_options_chain(None, 'AAPL')  # type: ignore[arg-type]
    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ---------------------------------------------------------------------------
# 6. Original DataFrame is not mutated (copy semantics)
# ---------------------------------------------------------------------------

def test_original_dataframe_not_mutated():
    row = _full_row()
    del row['vol']
    row['volume'] = 500
    df = pd.DataFrame([row])
    original_cols = list(df.columns)

    _normalize_options_chain(df, 'MSFT')

    assert list(df.columns) == original_cols  # 'volume' still present in original
    assert 'vol' not in df.columns
