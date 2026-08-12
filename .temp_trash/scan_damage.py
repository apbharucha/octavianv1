#!/usr/bin/env python3
"""Scan for line-join damage caused by the emoji strip (quote/bracket + 2+ spaces + statement)."""
import re

files = [
    'document_analyzer.py',
    'financial_model_generator_ui.py',
    'futures_simulation_grader.py',
    'options_simulation_grader.py',
    'portfolio_chatbot_context.py',
    'terms_of_service.py',
    'notification_settings_ui.py',
    'data_downloader.py',
]

# Lines that look joined: a quote/bracket followed by 2+ spaces then statement-ish content
join_re = re.compile(r'["\'`([{]\s{2,}[A-Za-z_@#"\']')
# Also docstring-joined: line contains """ followed by content with big spacing
doc_re = re.compile(r'"""[^"]{0,60}\s{3,}')

for path in files:
    try:
        lines = open(path, encoding='utf-8').read().splitlines()
    except Exception as e:
        print(f'{path}: ERR {e}')
        continue
    hits = []
    for i, ln in enumerate(lines, 1):
        if join_re.search(ln) or doc_re.search(ln):
            hits.append((i, ln.strip()[:110]))
    print(f'=== {path}: {len(hits)} suspicious lines')
    for i, txt in hits[:25]:
        print(f'  {i}: {txt}')
