"""
Manual Streamlit UI walkthrough of every main.py tab using AppTest,
with external providers (yfinance, news, LLM) mocked so the walkthrough
runs offline and deterministically.
"""
import os, sys, time
sys.path.insert(0, os.path.abspath('.'))
from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest

NAV = [
    "Dashboard", "Watchlist", "Market Scanner", "Symbol Analysis",
    "Chart Analysis", "Intelligence Center", "Market Heartbeat",
    "Financial Model Generator", "Daily Briefing", "Quant Portal",
    "Strategy Research Lab", "Paper Trading", "Simulation Hub",
    "Spreadsheet Generator", "Trader Profile", "Settings & Analytics",
]

def make_mock_df(n=90):
    import pandas as pd, numpy as np
    idx = pd.date_range('2025-01-01', periods=n, freq='D')
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.8, n))
    return pd.DataFrame({
        'Open': close * 0.99, 'High': close * 1.02, 'Low': close * 0.98,
        'Close': close, 'Adj Close': close, 'Volume': rng.integers(1e6, 5e6, n),
    }, index=idx)

MOCK_DF = None

def _mock_get_stock(symbol, *a, **k):
    global MOCK_DF
    if MOCK_DF is None:
        MOCK_DF = make_mock_df()
    return MOCK_DF.copy()

def run_tab(at, tab):
    at = AppTest.from_file('main.py', default_timeout=180)
    with patch('data_sources.get_stock', side_effect=_mock_get_stock), \
         patch('data_sources.get_fx', side_effect=_mock_get_stock), \
         patch('data_sources.get_futures_proxy', side_effect=_mock_get_stock), \
         patch('financial_llm_engine.check_llm_connectivity', return_value=False), \
         patch('financial_llm_engine._call_llm', return_value=''), \
         patch('unbiased_market_analyzer.UnbiasedMarketAnalyzer.scan_entire_market',
               return_value=[]), \
         patch('news_analysis_engine.NewsAnalysisEngine.fetch_and_process_news',
               return_value=[]), \
         patch('news_analysis_engine.NewsAnalysisEngine.get_market_sentiment',
               return_value={'avg_sentiment': 0.1, 'article_count': 0, 'sources': {}}), \
         patch('news_analysis_engine.NewsAnalysisEngine.get_market_whispers',
               return_value=[]), \
         patch('advanced_news_processor.AdvancedNewsProcessor.process_news_comprehensive',
               return_value=[]):
        at.run()
        # Select the tab via the Navigation radio
        for r in at.sidebar.radio:
            if r.label == 'Navigation':
                r.set_value(tab)
                at.run()
                break
        exc = [str(e.value) for e in at.exception]
        return at, exc
    return at, ['AppTest error']

def main():
    results = {}
    t0 = time.time()
    for tab in NAV:
        try:
            at, exc = run_tab(None, tab)
            ok = not exc
            results[tab] = {'ok': ok, 'exceptions': exc[:3],
                            'markdown': len(at.markdown),
                            'dataframes': len(at.dataframe),
                            'buttons': len(at.button)}
            print(f"[{'OK ' if ok else 'FAIL'}] {tab:28s} md={len(at.markdown):3d} df={len(at.dataframe):2d} btn={len(at.button):2d}")
            if exc:
                for e in exc[:2]:
                    print(f"      EXC: {e[:220]}")
        except Exception as e:
            results[tab] = {'ok': False, 'exceptions': [f'runner: {e}']}
            print(f"[ERR ] {tab:28s} runner: {str(e)[:220]}")
    print(f'\nWALKTHROUGH COMPLETE in {time.time()-t0:.1f}s')
    fails = {k: v for k, v in results.items() if not v.get('ok')}
    print(f'PASSED: {len(results)-len(fails)}/{len(results)}')
    for k, v in fails.items():
        print(f'  FAILED {k}: {v["exceptions"]}')

if __name__ == '__main__':
    main()
