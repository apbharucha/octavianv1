"""SQLite persistence for the chatbot evaluation harness.

Schema
------
evaluations     — one row per test query: query, category, extracted tickers,
                  full response text, overall score, criteria JSON, pass flag.
criteria_scores — one row per (evaluation, criterion) with a reason string.
issues          — issue log: aggregated failure patterns with example query,
                  affected count, severity, and status (open / fixed).

The database file lives in this package directory and is gitignored (*.db).
"""

import datetime
import json
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "octavian_chatbot_eval.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT,
    query             TEXT NOT NULL,
    category          TEXT,
    subcategory       TEXT,
    asset_class       TEXT,
    expected_intent   TEXT,
    extracted_tickers TEXT,
    intents_json      TEXT,
    response          TEXT,
    overall_score     REAL,
    criteria_json     TEXT,
    passed            INTEGER,
    issue_tags        TEXT,
    elapsed_ms        REAL,
    visuals_json      TEXT,
    created_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_evals_category ON evaluations(category);
CREATE INDEX IF NOT EXISTS idx_evals_passed   ON evaluations(passed);

CREATE TABLE IF NOT EXISTS criteria_scores (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    eval_id   INTEGER NOT NULL,
    criterion TEXT NOT NULL,
    score     REAL,
    reason    TEXT,
    FOREIGN KEY(eval_id) REFERENCES evaluations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS issues (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT,
    category       TEXT,
    description    TEXT,
    severity       TEXT,
    example_query  TEXT,
    affected_count INTEGER,
    status         TEXT DEFAULT 'open',
    fix_description TEXT,
    fixed_at       TEXT,
    created_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_issues_status ON issues(status);
"""


def get_conn(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 60000")
    return conn


def init_db(path: str = DB_PATH) -> sqlite3.Connection:
    conn = get_conn(path)
    # WAL journaling lets a second eval process read while the first writes —
    # parallel / overlapping bucket runs no longer hit "database is locked".
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    conn.executescript(SCHEMA)
    # Migration: add visuals_json to databases created before it existed.
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(evaluations)")]
    if "visuals_json" not in cols:
        conn.execute("ALTER TABLE evaluations ADD COLUMN visuals_json TEXT")
    conn.commit()
    return conn


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def insert_evaluation(conn, *, run_id, query, category, subcategory, asset_class,
                      expected_intent, extracted_tickers, intents, response,
                      overall, criteria, passed, issue_tags, elapsed_ms,
                      visuals=None) -> int:
    cur = conn.execute(
        """INSERT INTO evaluations
           (run_id, query, category, subcategory, asset_class, expected_intent,
            extracted_tickers, intents_json, response, overall_score,
            criteria_json, passed, issue_tags, elapsed_ms, visuals_json, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, query, category, subcategory, asset_class, expected_intent,
         json.dumps(extracted_tickers), json.dumps(intents), response,
         float(overall), json.dumps(criteria), 1 if passed else 0,
         ",".join(issue_tags) if issue_tags else "", float(elapsed_ms),
         json.dumps(visuals) if visuals else None, _now()),
    )
    eval_id = cur.lastrowid
    for criterion, (score, reason) in criteria.items():
        conn.execute(
            "INSERT INTO criteria_scores (eval_id, criterion, score, reason) VALUES (?,?,?,?)",
            (eval_id, criterion, float(score), reason or ""),
        )
    conn.commit()
    return eval_id


def insert_issue(conn, *, run_id, category, description, severity,
                 example_query, affected_count) -> int:
    cur = conn.execute(
        """INSERT INTO issues
           (run_id, category, description, severity, example_query,
            affected_count, status, created_at)
           VALUES (?,?,?,?,?,?,'open',?)""",
        (run_id, category, description, severity, example_query,
         int(affected_count), _now()),
    )
    conn.commit()
    return cur.lastrowid


def mark_issue_fixed(conn, issue_id: int, fix_description: str) -> None:
    conn.execute(
        "UPDATE issues SET status='fixed', fix_description=?, fixed_at=? WHERE id=?",
        (fix_description, _now(), issue_id),
    )
    conn.commit()


def failing_queries(conn, run_id: str = None, limit: int = 100000):
    if run_id:
        rows = conn.execute(
            "SELECT * FROM evaluations WHERE passed=0 AND run_id=? ORDER BY id",
            (run_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM evaluations WHERE passed=0 ORDER BY id",
        ).fetchall()
    return rows[:limit]


def summary(conn, run_id: str = None):
    q = "SELECT COUNT(*) AS n, AVG(overall_score) AS avg_o, SUM(passed) AS ok FROM evaluations"
    args = ()
    if run_id:
        q += " WHERE run_id=?"
        args = (run_id,)
    row = conn.execute(q, args).fetchone()
    n = row["n"] or 0
    avg = row["avg_o"] or 0.0
    ok = row["ok"] or 0
    per_cat = {}
    cat_q = ("SELECT category, COUNT(*) AS n, AVG(overall_score) AS avg_o, SUM(passed) AS ok "
             "FROM evaluations")
    if run_id:
        cat_q += " WHERE run_id=?"
        rows = conn.execute(cat_q + " GROUP BY category", (run_id,)).fetchall()
    else:
        rows = conn.execute(cat_q + " GROUP BY category").fetchall()
    for r in rows:
        per_cat[r["category"]] = {
            "n": r["n"] or 0,
            "avg": round(r["avg_o"] or 0.0, 2),
            "passed": r["ok"] or 0,
        }
    return {"total": n, "avg": round(avg, 2), "passed": ok, "per_category": per_cat}


def avg_criterion(conn, criterion: str, run_id: str = None):
    q = ("SELECT AVG(cs.score) AS s FROM criteria_scores cs "
         "JOIN evaluations e ON e.id = cs.eval_id WHERE cs.criterion=?")
    args = [criterion]
    if run_id:
        q += " AND e.run_id=?"
        args.append(run_id)
    row = conn.execute(q, args).fetchone()
    return round(row["s"] or 0.0, 2)
