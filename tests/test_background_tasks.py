"""Unit tests for the background task manager (background_tasks.py).

Covers the guarantees the feature relies on:
* tasks run to completion on worker threads without blocking the caller
* failures are captured, never crash the worker
* sessions are isolated (users only see their own tasks)
* concurrent tasks genuinely run in parallel (bounded pool)
* completion notification is exactly-once (atomic drain)
* the registry stays bounded
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import background_tasks as bt


def _wait_until(pred, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _reset():
    """Release the pool and clear the registry between tests."""
    bt.shutdown(wait=True)
    with bt._LOCK:
        bt._TASKS.clear()
        bt._ORDER.clear()
        bt._EXECUTING.clear()


def test_successful_task_lifecycle():
    _reset()
    gate = threading.Event()

    def slow():
        # No timeout: the worker must stay blocked until the test releases
        # it, so status assertions are never racy under heavy suite load.
        gate.wait()
        return 42

    try:
        tid = bt.submit_task("t1", slow, session_id="s1", result_key="rk")
        # still running while the worker is blocked on the gate
        assert bt.get_task(tid)["status"] == "running"
        assert any(t["task_id"] == tid for t in bt.running_tasks("s1"))
    finally:
        gate.set()  # always release the worker, even if an assertion fails
    assert _wait_until(lambda: bt.get_task(tid)["status"] == "done")
    rec = bt.get_task(tid)
    assert rec["result"] == 42
    assert rec["result_key"] == "rk"
    assert rec["finished_at"] is not None
    # drain returns each task exactly once
    drained = bt.drain_completed("s1")
    assert len(drained) == 1 and drained[0]["task_id"] == tid
    assert bt.drain_completed("s1") == []
    _reset()


def test_error_task_is_captured():
    _reset()

    def boom():
        raise ValueError("bad input")

    tid = bt.submit_task("t2", boom, session_id="s2")
    assert _wait_until(lambda: bt.get_task(tid)["status"] == "error")
    assert "ValueError" in bt.get_task(tid)["error"]
    assert bt.get_task(tid)["result"] is None
    _reset()


def test_session_isolation():
    _reset()
    t1 = bt.submit_task("a", lambda: 1, session_id="s1")
    t2 = bt.submit_task("b", lambda: 2, session_id="s2")
    assert _wait_until(
        lambda: bt.get_task(t1)["status"] == "done"
        and bt.get_task(t2)["status"] == "done"
    )
    assert {t["task_id"] for t in bt.tasks_for_session("s1")} == {t1}
    assert {t["task_id"] for t in bt.tasks_for_session("s2")} == {t2}
    # each session drains only its own tasks
    assert bt.drain_completed("s1")[0]["task_id"] == t1
    assert bt.drain_completed("s2")[0]["task_id"] == t2
    _reset()


def test_concurrent_tasks_run_in_parallel():
    _reset()
    # A barrier of 3 can only be crossed if all 3 tasks are running at the
    # same time — proves the pool actually runs processes concurrently.
    barrier = threading.Barrier(3)

    def wait_all():
        barrier.wait(timeout=5)
        return "ok"

    tids = [bt.submit_task(f"c{i}", wait_all, session_id="s3") for i in range(3)]
    assert _wait_until(
        lambda: all(bt.get_task(t)["status"] == "done" for t in tids)
    )
    _reset()


def test_running_tasks_view():
    _reset()
    gate = threading.Event()

    def blocked():
        gate.wait()
        return "released"

    try:
        tid = bt.submit_task("blocked", blocked, session_id="s4")
        assert any(
            t["task_id"] == tid for t in bt.running_tasks("s4")
        ), "running task should be visible while it executes"
    finally:
        gate.set()  # always release the worker, even if an assertion fails
    assert _wait_until(lambda: bt.get_task(tid)["status"] == "done")
    assert bt.running_tasks("s4") == []
    _reset()


def test_registry_is_bounded():
    _reset()
    for i in range(350):
        bt.submit_task(f"x{i}", lambda i=i: i, session_id="s9")
    assert bt.count() <= 300, "registry must drop oldest records"
    _reset()


def test_eviction_prefers_to_keep_running_tasks():
    _reset()
    gate = threading.Event()

    def blocked():
        # No timeout: the task must still be running when eviction runs, or
        # the test would silently be testing the wrong thing (and flake
        # under full-suite load when 10s can elapse). Released via finally.
        gate.wait()
        return "done"

    try:
        tid_running = bt.submit_task("blocked", blocked, session_id="s12")
        # Deterministic: wait until the worker is genuinely executing the
        # blocked task, so the flood below can never evict it while queued.
        assert _wait_until(lambda: tid_running in bt._EXECUTING)
        for i in range(310):
            bt.submit_task(f"fast{i}", lambda i=i: i, session_id="s12")

        # Wait until every record that still exists is finished (the oldest
        # fast tasks may already have been evicted to stay within the cap —
        # that is the bounded-registry behavior we want).
        def all_survivors_done():
            with bt._LOCK:
                recs = list(bt._TASKS.values())
            return bool(recs) and all(
                r["status"] == "done"
                for r in recs
                if r["task_id"] != tid_running
            )

        assert _wait_until(all_survivors_done)
        # Push past the cap once more: cleanup must evict finished records,
        # never the still-running one.
        bt.submit_task("spill", lambda: 1, session_id="s12")
        assert bt.count() <= 300
        assert bt.get_task(tid_running) is not None, "running task was evicted"
    finally:
        gate.set()  # always release the worker, even if an assertion fails
    assert _wait_until(lambda: bt.get_task(tid_running)["status"] == "done")
    _reset()


def test_cleanup_evicts_queued_before_executing():
    """Regression: a flood of queued records must never evict a task whose
    worker thread is genuinely executing (the oldest record is the executing
    one — evicting it would silently drop the longest-pending job).
    """
    _reset()
    with bt._LOCK:
        for i in range(305):
            tid = f"t{i}"
            bt._TASKS[tid] = {
                "task_id": tid,
                "name": "x",
                "session_id": "s",
                "status": "running",  # queued: stamped at submit, no worker yet
                "result": None,
                "error": None,
                "result_key": None,
                "notified": False,
                "submitted_at": time.time(),
                "finished_at": None,
            }
            bt._ORDER.append(tid)
        executing = "t0"  # oldest record, the one a worker is actually running
        bt._EXECUTING.add(executing)
        bt._cleanup_locked()
        assert bt.get_task(executing) is not None, "executing task was evicted"
        assert bt.count() <= 300
    _reset()


def test_reopen_requeues_task_for_retry():
    _reset()
    tid = bt.submit_task("retry", lambda: "v", session_id="s11")
    assert _wait_until(lambda: bt.get_task(tid)["status"] == "done")
    assert len(bt.drain_completed("s11")) == 1
    assert bt.drain_completed("s11") == []  # notified -> exactly once
    bt.reopen("s11", tid)
    retried = bt.drain_completed("s11")
    assert len(retried) == 1 and retried[0]["task_id"] == tid
    assert bt.drain_completed("s11") == []
    _reset()


def test_result_payload_released_after_drain():
    _reset()
    payload = {"big": "data"}
    tid = bt.submit_task("payload", lambda p=payload: p, session_id="s13")
    assert _wait_until(lambda: bt.get_task(tid)["status"] == "done")
    assert bt.get_task(tid)["result"] is payload  # retained until delivered
    drained = bt.drain_completed("s13")
    assert drained[0]["result"] is payload  # caller still receives it
    assert bt.get_task(tid)["result"] is None  # registry releases it
    _reset()


def test_drain_is_atomic_under_contention():
    _reset()
    tid = bt.submit_task("race", lambda: "result", session_id="s10")
    assert _wait_until(lambda: bt.get_task(tid)["status"] == "done")
    # Two threads drain concurrently — the task must be returned exactly once.
    results = []
    lock = threading.Lock()

    def drainer():
        for _ in range(50):
            got = bt.drain_completed("s10")
            with lock:
                results.extend(got)

    threads = [threading.Thread(target=drainer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 1, f"expected exactly one drain, got {len(results)}"
    _reset()
