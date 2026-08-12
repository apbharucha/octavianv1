from portfolio_analyzer import compute_portfolio_metrics


def test_compute_metrics_all_prices_available():
    positions = [
        {"symbol": "AAPL", "quantity": 10, "entry_price": 100, "current_price": 110, "asset_type": "equity"},
        {"symbol": "MSFT", "quantity": 5, "entry_price": 200, "current_price": 190, "asset_type": "equity"},
    ]
    out = compute_portfolio_metrics(positions)
    summary = out["summary"]
    assert summary["position_count"] == 2
    assert summary["priced_positions"] == 2
    assert summary["unpriced_positions"] == 0
    assert summary["aggregate_unrealized_pnl"] == 50.0
    assert summary["total_portfolio_value"] == 2050.0


def test_compute_metrics_one_price_unavailable_excluded_from_aggregate():
    positions = [
        {"symbol": "AAPL", "quantity": 10, "entry_price": 100, "current_price": 110, "asset_type": "equity"},
        {"symbol": "XYZ", "quantity": 5, "entry_price": 200, "current_price": None, "asset_type": "equity"},
    ]
    out = compute_portfolio_metrics(positions)
    summary = out["summary"]
    assert summary["position_count"] == 2
    assert summary["priced_positions"] == 1
    assert summary["unpriced_positions"] == 1
    assert summary["aggregate_unrealized_pnl"] == 100.0
    assert summary["total_portfolio_value"] == 1100.0


def test_compute_metrics_zero_positions():
    out = compute_portfolio_metrics([])
    summary = out["summary"]
    assert summary["position_count"] == 0
    assert summary["priced_positions"] == 0
    assert summary["unpriced_positions"] == 0
    assert summary["aggregate_unrealized_pnl"] == 0.0
    assert summary["total_portfolio_value"] == 0.0


def test_imported_vs_manual_shape_consistency():
    imported = [
        {"symbol": "AAPL", "quantity": 10, "entry_price": 100, "current_price": 105, "asset_type": "equity"},
    ]
    manual = [
        {"symbol": "AAPL", "quantity": 10, "entry_price": 100, "current_price": 105, "asset_type": "equity"},
    ]
    out_imported = compute_portfolio_metrics(imported)
    out_manual = compute_portfolio_metrics(manual)
    assert set(out_imported.keys()) == set(out_manual.keys())
    assert set(out_imported["summary"].keys()) == set(out_manual["summary"].keys())
    assert set(out_imported["rows"][0].keys()) == set(out_manual["rows"][0].keys())
