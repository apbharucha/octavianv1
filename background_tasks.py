"""
Octavian Background Task Manager
================================

Lets long-running features (Breaking Trades scans, Daily Briefing
generation, market sweeps, …) keep executing in worker threads while the
user freely navigates the Streamlit app.

Why a module-level singleton?
-----------------------------
Streamlit reruns the whole script on every interaction and on every page
navigation, but all reruns for all sessions share one Python process.
Module-level state therefore survives page changes and reruns, so a task
submitted on the Dashboard keeps running while the user browses to Paper
Trading, the Financial Model Generator, or anywhere else.

Guarantees
----------
* **Never blocks the UI.** Tasks run in a bounded thread pool; the script
  returns immediately after submitting.
* **Concurrent but bounded.** All tasks share one pool (``_MAX_WORKERS``),
  so multiple processes run at the same time efficiently without spawning
  unbounded threads or starving the machine.
* **Session isolation.** Tasks are keyed by Streamlit session id, so users
  only ever see / get notified about their own jobs.
* **Exactly-once notification.** ``drain_completed`` atomically marks a
  finished task as "notified", so a main-script rerun and the idle poller
  fragment can never double-notify the same completion.
* **Bounded memory.** The registry keeps the most recent ``_MAX_TASKS``
  records and drops the oldest.
* **Thread safe.** Every mutation happens under one re-entrant lock.

Usage
-----
    from background_tasks import submit_task, drain_completed, get_task

    tid = submit_task("Breaking Trades scan", fn, session_id=sid,
                      result_key="dashboard_breaking_trades", *args, **kw)
    # ... user keeps navigating ...
    for task in drain_completed(sid):          # finished + not yet toasted
        st.session_state[task["result_key"]] = task["result"]
        st.toast(f"{task['name']} complete", icon="✅")
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

_LOCK = threading.RLock()
_EXECUTOR: ThreadPoolExecutor | None = None

# task_id -> record dict
_TASKS: dict[str, dict] = {}
# task ids in submission order, used for bounded cleanup
_ORDER: list[str] = []

# A few worker threads is enough: each task may itself parallelize
# internally (e.g. the discovery engine fans out its own fetches). Keeping
# the pool small means many background processes run concurrently without
# overwhelming the host.
_MAX_WORKERS = 4
_MAX_TASKS = 300


def _get_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    with _LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(
                max_workers=_MAX_WORKERS,
                thread_name_prefix="octavian-bg",
            )
        return _EXECUTOR


def _cleanup_locked() -> None:
    """Keep the registry bounded without dropping running tasks.

    Evicts the oldest finished / already-notified records first, so a task
    that is still executing can always report its result. Only if the cap is
    still exceeded (an extreme flood of concurrent jobs) are running records
    evicted as a last resort.
    """
    if len(_ORDER) <= _MAX_TASKS:
        return
    for tid in list(_ORDER):
        if len(_ORDER) <= _MAX_TASKS:
            break
        r = _TASKS.get(tid)
        if r is None or r["status"] != "running":
            _ORDER.remove(tid)
            _TASKS.pop(tid, None)
    while len(_ORDER) > _MAX_TASKS:
        oldest = _ORDER.pop(0)
        _TASKS.pop(oldest, None)


def submit_task(
    name: str,
    fn,
    session_id: str = "default",
    result_key: str | None = None,
    *args,
    **kwargs,
) -> str:
    """Run ``fn(*args, **kwargs)`` on a worker thread.

    Returns the task id immediately. The task record transitions
    ``running -> done | error``; on success ``record["result"]`` holds the
    return value and, if ``result_key`` is given, the UI can publish it
    straight into ``st.session_state[result_key]`` when it completes.
    """
    task_id = uuid.uuid4().hex[:12]
    record = {
        "task_id": task_id,
        "name": name,
        "session_id": session_id,
        "status": "running",
        "result": None,
        "error": None,
        "result_key": result_key,
        "notified": False,
        "submitted_at": time.time(),
        "finished_at": None,
    }
    with _LOCK:
        _TASKS[task_id] = record
        _ORDER.append(task_id)
        _cleanup_locked()

    def _run() -> None:
        try:
            result = fn(*args, **kwargs)
            with _LOCK:
                if task_id in _TASKS:
                    _TASKS[task_id]["status"] = "done"
                    _TASKS[task_id]["result"] = result
                    _TASKS[task_id]["finished_at"] = time.time()
        except Exception as exc:  # noqa: BLE001 - must never kill the worker
            with _LOCK:
                if task_id in _TASKS:
                    _TASKS[task_id]["status"] = "error"
                    _TASKS[task_id]["error"] = f"{type(exc).__name__}: {exc}"
                    _TASKS[task_id]["finished_at"] = time.time()

    _get_executor().submit(_run)
    return task_id


def get_task(task_id: str) -> dict | None:
    """Return a copy of the task record, or None if unknown."""
    with _LOCK:
        rec = _TASKS.get(task_id)
        return dict(rec) if rec else None


def tasks_for_session(session_id: str, limit: int = 20) -> list[dict]:
    """Most recent tasks (newest last) belonging to one session.

    Result payloads are stripped from the copies — callers (e.g. the sidebar
    status panel) only need status metadata, and the payload stays in the
    registry only until it is drained and published.
    """
    with _LOCK:
        out = []
        for tid in _ORDER:
            r = _TASKS.get(tid)
            if r and r["session_id"] == session_id:
                rec = dict(r)
                rec.pop("result", None)
                out.append(rec)
    return out[-limit:]


def running_tasks(session_id: str | None = None) -> list[dict]:
    """All currently-running tasks, optionally filtered by session."""
    with _LOCK:
        return [
            dict(r)
            for r in _TASKS.values()
            if r["status"] == "running"
            and (session_id is None or r["session_id"] == session_id)
        ]


def drain_completed(session_id: str) -> list[dict]:
    """Atomically return this session's finished-but-unnotified tasks.

    Each task is returned exactly once (marked ``notified`` under the lock),
    so concurrent callers (the main script and the idle poller fragment)
    can never double-notify or double-publish a completion.
    """
    with _LOCK:
        out = []
        for tid in _ORDER:
            r = _TASKS.get(tid)
            if (
                r
                and r["session_id"] == session_id
                and not r["notified"]
                and r["status"] in ("done", "error")
            ):
                r["notified"] = True
                out.append(dict(r))
                # Release the payload from the registry — the caller already
                # holds a copy and will publish it. Keeps memory bounded even
                # with large results (briefing dicts, trade objects).
                r["result"] = None
        return out


def mark_notified(session_id: str, task_id: str) -> None:
    """Mark a single task as notified (idempotent)."""
    with _LOCK:
        r = _TASKS.get(task_id)
        if r and r["session_id"] == session_id:
            r["notified"] = True


def reopen(session_id: str, task_id: str) -> None:
    """Re-queue a finished task for delivery.

    Used when the UI failed to publish/notify a completion the first time
    (e.g. a session-state write error) — the next drain will hand the result
    out again instead of losing it silently.
    """
    with _LOCK:
        r = _TASKS.get(task_id)
        if (
            r
            and r["session_id"] == session_id
            and r["status"] in ("done", "error")
        ):
            r["notified"] = False


def count() -> int:
    """Number of records currently held (for tests / diagnostics)."""
    with _LOCK:
        return len(_TASKS)


def shutdown(wait: bool = True) -> None:
    """Release the worker pool. Safe to call repeatedly; used by tests."""
    global _EXECUTOR
    with _LOCK:
        ex = _EXECUTOR
        _EXECUTOR = None
    if ex is not None:
        ex.shutdown(wait=wait)
