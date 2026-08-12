import pandas as pd
import streamlit as st
import yfinance as yf
import sqlite3
from datetime import datetime
import sys
import os

# Ensure we run from the project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    from market_simulation_engine import MarketSimulationEngine
    SIM_ENGINE_AVAILABLE = True
    SIM_ENGINE_ERROR = None
except Exception as e:
    SIM_ENGINE_AVAILABLE = False
    SIM_ENGINE_ERROR = str(e)

FUTURES = {
    "ES (S&P 500)": "ES=F",
    "NQ (Nasdaq)": "NQ=F",
    "CL (Crude Oil)": "CL=F",
    "GC (Gold)": "GC=F",
    "SI (Silver)": "SI=F",
    "ZB (Bonds)": "ZB=F"
}

@st.cache_data(ttl=3600, show_spinner=False)
def futures_rank(lookback=21):
    rows = []
    
    tickers = list(FUTURES.values())
    
    try:
        # Bulk download
        data = yf.download(tickers, period="3mo", group_by='ticker', progress=False, threads=True)
        
        for name, ticker in FUTURES.items():
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker not in data.columns.levels[0]:
                        continue
                    df = data[ticker].dropna()
                else:
                    if ticker != tickers[0]: continue
                    df = data.dropna()
                    
                if df.empty or len(df) < lookback:
                    continue

                close_col = df["Close"]
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                
                current = float(close_col.iloc[-1])
                prev = float(close_col.iloc[-lookback])
                ret = ((current / prev) - 1) * 100 if prev > 0 else 0
                rows.append({"Contract": name, "TrendScore": round(ret, 2)})
            except Exception:
                continue
    except Exception:
         return pd.DataFrame(columns=["Contract", "TrendScore"])

    if len(rows) == 0:
        return pd.DataFrame(columns=["Contract", "TrendScore"])

    return pd.DataFrame(rows).sort_values("TrendScore", ascending=False)

def futures_data_health_check(lookback=21):
    """Quick diagnostic to verify futures data availability and basic inputs."""
    status = {"ok": True, "checked": [], "errors": []}
    tickers = list(FUTURES.values())
    try:
        data = yf.download(tickers, period="3mo", group_by="ticker", progress=False, threads=True)
        for name, ticker in FUTURES.items():
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker not in data.columns.levels[0]:
                        raise ValueError("missing_ticker_data")
                    df = data[ticker].dropna()
                else:
                    if ticker != tickers[0]:
                        continue
                    df = data.dropna()

                if df.empty or len(df) < lookback:
                    raise ValueError("insufficient_lookback")

                close_col = df["Close"]
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]

                current = float(close_col.iloc[-1])
                prev = float(close_col.iloc[-lookback])
                if prev <= 0:
                    raise ValueError("invalid_previous_close")

                status["checked"].append({"contract": name, "current": current, "prev": prev})
            except Exception as e:
                status["ok"] = False
                status["errors"].append({"contract": name, "error": str(e)})
    except Exception as e:
        status["ok"] = False
        status["errors"].append({"contract": "ALL", "error": str(e)})

    return status

def _db_path_for_today():
    today_str = datetime.now().strftime('%Y_%m_%d')
    return f"octavian_simulations_{today_str}.db"


def _ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS simulations (
            simulation_id TEXT PRIMARY KEY,
            start_time DATETIME,
            end_time DATETIME,
            market_regime TEXT,
            total_decisions INTEGER,
            successful_decisions INTEGER,
            failed_decisions INTEGER,
            total_return REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            win_rate REAL,
            avg_win REAL,
            avg_loss REAL,
            performance_data TEXT,
            insights_data TEXT
        );

        CREATE TABLE IF NOT EXISTS trading_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            simulation_id TEXT,
            timestamp DATETIME,
            symbol TEXT,
            action TEXT,
            quantity REAL,
            price REAL,
            confidence REAL,
            actual_outcome REAL,
            reasoning TEXT
        );

        CREATE TABLE IF NOT EXISTS simulated_news_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            simulation_id TEXT,
            timestamp DATETIME,
            headline TEXT,
            news_type TEXT,
            market_impact_score REAL,
            content TEXT,
            affected_symbols TEXT
        );

        CREATE TABLE IF NOT EXISTS learning_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            simulation_id TEXT,
            insight_type TEXT,
            insight_text TEXT,
            timestamp DATETIME
        );
    """)
    conn.commit()


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    return int(cur.fetchone()[0])


def run_simulation_health_check():
    report = {"ok": True, "errors": [], "warnings": [], "details": {}}

    if not SIM_ENGINE_AVAILABLE:
        report["ok"] = False
        report["errors"].append(f"Simulation engine unavailable: {SIM_ENGINE_ERROR}")
        return report

    # Run simulation
    try:
        engine = MarketSimulationEngine()
        engine.run_manual_simulation()
    except Exception as e:
        report["ok"] = False
        report["errors"].append(f"Simulation run failed: {e}")
        return report

    # Validate DB
    try:
        db_path = _db_path_for_today()
        conn = sqlite3.connect(db_path)

        counts = {
            "db_path": db_path,
            "simulations": _table_count(conn, "simulations"),
            "trading_decisions": _table_count(conn, "trading_decisions"),
            "simulated_news_events": _table_count(conn, "simulated_news_events"),
            "learning_insights": _table_count(conn, "learning_insights"),
        }
        report["details"] = counts

        if counts["simulations"] == 0:
            report["ok"] = False
            report["errors"].append("No simulations stored.")
        if counts["trading_decisions"] == 0:
            report["warnings"].append("No trading decisions recorded.")
        if counts["simulated_news_events"] == 0:
            report["warnings"].append("No news events generated.")
        if counts["learning_insights"] == 0:
            report["warnings"].append("No learning insights stored.")

        conn.close()
    except Exception as e:
        report["ok"] = False
        report["errors"].append(f"DB validation failed: {e}")

    return report


if __name__ == "__main__":
    result = run_simulation_health_check()

    status = "[OK] PASS" if result["ok"] else "[X] FAIL"
    print(f"\n{'='*50}")
    print(f"  Simulation Health Check: {status}")
    print(f"{'='*50}")

    d = result.get("details", {})
    print(f"  DB Path:            {d.get('db_path', 'N/A')}")
    print(f"  Simulations:        {d.get('simulations', 0)}")
    print(f"  Trading Decisions:  {d.get('trading_decisions', 0)}")
    print(f"  News Events:        {d.get('simulated_news_events', 0)}")
    print(f"  Learning Insights:  {d.get('learning_insights', 0)}")

    if result.get("errors"):
        print(f"\n  Errors:")
        for e in result["errors"]:
            print(f"    [X] {e}")
    if result.get("warnings"):
        print(f"\n  Warnings:")
        for w in result["warnings"]:
            print(f"    [WARN]  {w}")

    print(f"{'='*50}\n")
    sys.exit(0 if result["ok"] else 1)