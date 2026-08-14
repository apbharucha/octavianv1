"""End-to-end integration test for the background-task UI pattern.

Runs a miniature Streamlit app (via Streamlit's own AppTest harness) that
mirrors the exact launch -> background-execute -> drain -> publish -> toast
flow wired into main.py for Breaking Trades and the Daily Briefing, so we
know the pattern works in a real script runtime, not just in the manager's
unit tests.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

APP_SCRIPT = """
import streamlit as st
import sys
sys.path.insert(0, {root!r})
import background_tasks as bt

def _sid():
    ctx = st.runtime.scriptrunner.get_script_run_ctx()
    return getattr(ctx, "session_id", None) or "default"

# 1) Launch: kick a job off on a worker thread and return immediately.
if st.button("Launch", key="launch"):
    tid = bt.submit_task(
        "Demo task",
        lambda: {{"value": 7, "symbol": "NVDA"}},
        session_id=_sid(),
        result_key="demo_result",
    )
    st.session_state["_bg_demo_result"] = tid
    st.toast("started in background")

# 2) Publish + notify: same code path as main.py._check_background_tasks().
for t in bt.drain_completed(_sid()):
    if t["status"] == "done" and t.get("result_key"):
        st.session_state[t["result_key"]] = t["result"]
    if t["status"] == "done":
        st.toast("Demo task complete")
    else:
        st.toast("Demo task failed: " + str(t.get("error"))[:80])

# 3) Show live status for the page.
tid = st.session_state.get("_bg_demo_result")
task = bt.get_task(tid) if tid else None
if task and task["status"] == "running":
    st.info("running in background")
elif task and task["status"] == "done":
    st.markdown("RESULT_READY: " + str(st.session_state.get("demo_result")))

st.write("MARKER")
"""


def test_app_runs_cleanly_before_any_launch():
    at = AppTest.from_string(APP_SCRIPT.format(root=_ROOT), default_timeout=30)
    at.run()
    assert len(at.exception) == 0
    assert len(at.button) == 1


def test_launch_runs_in_background_and_publishes_result():
    at = AppTest.from_string(APP_SCRIPT.format(root=_ROOT), default_timeout=30)
    at.run()
    at.button[0].click().run()  # launch the background task
    assert len(at.exception) == 0

    # The worker finishes in ms, so the click-rerun itself usually performs
    # the drain -> publish -> toast. If it was still running, give it a
    # moment and rerun (AppTest.toast only reflects the last run).
    if not any("Demo task" in t.value for t in at.toast):
        time.sleep(0.5)
        at.run()
        assert len(at.exception) == 0

    toasts = " | ".join(t.value for t in at.toast)
    assert "started in background" in toasts or "Demo task complete" in toasts

    body = " ".join(w.value for w in at.markdown)
    assert "RESULT_READY" in body, f"result never published: {body!r}"
    assert "NVDA" in body, f"published result missing payload: {body!r}"


def test_error_job_surfaces_failure_notification():
    at = AppTest.from_string(
        APP_SCRIPT.format(root=_ROOT).replace(
            'lambda: {"value": 7, "symbol": "NVDA"}', "lambda: 1 / 0"
        ),
        default_timeout=30,
    )
    at.run()
    at.button[0].click().run()
    assert len(at.exception) == 0
    if not any("Demo task" in t.value for t in at.toast):
        time.sleep(0.5)
        at.run()
        assert len(at.exception) == 0
    toasts = " | ".join(t.value for t in at.toast)
    assert "Demo task failed" in toasts, f"failure not surfaced: {toasts!r}"
