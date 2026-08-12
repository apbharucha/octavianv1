"""
chatbot_eval — Octavian AI Chatbot Evaluation Harness.

Mass-scale prompt testing with a persistent SQLite database:

  * prompt_factory.py  — generates thousands of prompts across every asset
                         class, plus macro, geopolitics, current events,
                         risk/hedging and trade-setup categories.
  * pipeline.py        — runs the real `financial_llm_engine` pipeline on a
                         deterministic, fully mocked data layer (no network,
                         ~10-30 ms/query).
  * rubric.py          — scores every response on 8 criteria, each out of 10,
                         plus a weighted overall score.
  * db.py              — SQLite persistence for evaluations, per-criterion
                         scores, and the issue log (open/fixed).
  * run_eval.py        — orchestrates: generate -> run -> score -> persist ->
                         summarize -> issue sync.

Run from the project root:

    python -m chatbot_eval.run_eval --limit 5000 --workers 8
    python -m chatbot_eval.run_eval --report
    python -m chatbot_eval.run_eval --rerun-failed --workers 8
    python -m chatbot_eval.run_eval --sync-issues
"""
