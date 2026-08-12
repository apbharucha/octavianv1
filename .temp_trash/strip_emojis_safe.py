#!/usr/bin/env python3
"""Remove emoji codepoints ONLY. Never touches whitespace or newlines."""
import os
import re

EMOJI_RE = re.compile(
    r"[\U0001F000-\U0001FAFF]"      # Misc pictographs, emoticons, transport, etc.
    r"|[\U0001F1E6-\U0001F1FF]"     # Regional indicator letters
    r"|[\u2600-\u27BF]"             # Misc symbols + dingbats (stars, check, warn...)
    r"|[\u2B00-\u2BFF]"             # Misc symbols and arrows (e.g. DOWN, star)
    r"|[\uFE0F\u200D]"              # Variation selector / ZWJ
)

TARGETS = [
    "ai_chatbot.py",
    "main.py",
    "quant_portal.py",
    "spreadsheet_generator.py",
    "data_downloader.py",
    "financial_model_generator_ui.py",
    "portfolio_chatbot_context.py",
    "document_analyzer.py",
    "terms_of_service.py",
    "futures_simulation_grader.py",
    "options_simulation_grader.py",
    "notification_settings_ui.py",
    "start_streamlit.sh",
    "README.md",
    "START_HERE.md",
    "IMPLEMENTATION_STATUS.md",
    "AUDIT_REPORT.md",
    "AI_RESPONSE_FORMATTING_GUIDE.md",
    "AGENTS.md",
    "memory/2026-02-21.md",
    "memory/2026-03-04.md",
]

total = 0
for path in TARGETS:
    if not os.path.exists(path):
        print(f"SKIP missing: {path}")
        continue
    with open(path, encoding="utf-8") as f:
        text = f.read()
    new_text = EMOJI_RE.sub("", text)
    n = len(EMOJI_RE.findall(text))
    if n:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_text)
        total += n
        print(f"STRIPPED {n:3d}: {path}")
    else:
        print(f"clean: {path}")
print(f"\nTotal removed: {total}")
