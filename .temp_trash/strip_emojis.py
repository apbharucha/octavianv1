#!/usr/bin/env python3
"""Remove emoji characters from Octavian source/docs files.

Strategy:
  * Remove emoji codepoints only (never ASCII, never meaningful unicode like
    arrows/bullets/dashes outside the emoji ranges).
  * Replace emoji + surrounding inline whitespace with a single space, then
    tidy obvious label artifacts: leading space right after an opening quote,
    trailing space right before a closing quote, and double spaces created by
    the removal.
  * Never touch leading indentation (line-start spaces are preserved).
"""
import os
import re

EMOJI_RE = re.compile(
    r"[\U0001F000-\U0001FAFF]"      # Misc pictographs, emoticons, transport, etc.
    r"|[\U0001F1E6-\U0001F1FF]"     # Regional indicator letters
    r"|[\u2600-\u27BF]"             # Misc symbols + dingbats (stars, check, warn...)
    r"|[\u2B00-\u2BFF]"             # Misc symbols and arrows (e.g. ⬇ ⭐)
    r"|[\uFE0F\u200D]"              # Variation selector / ZWJ
)

# Files that contain emojis (from the scan). Skip build artifacts (.next).
TARGETS = [
    "AGENTS.md",
    "AI_RESPONSE_FORMATTING_GUIDE.md",
    "AUDIT_REPORT.md",
    "IMPLEMENTATION_STATUS.md",
    "README.md",
    "START_HERE.md",
    "ai_chatbot.py",
    "data_downloader.py",
    "document_analyzer.py",
    "financial_model_generator_ui.py",
    "futures_simulation_grader.py",
    "main.py",
    "memory/2026-02-21.md",
    "memory/2026-03-04.md",
    "notification_settings_ui.py",
    "options_simulation_grader.py",
    "portfolio_chatbot_context.py",
    "quant_portal.py",
    "spreadsheet_generator.py",
    "start_streamlit.sh",
    "terms_of_service.py",
]

def strip_line(line: str) -> str:
    # Replace emoji clusters (with surrounding inline spaces) by a single space.
    line = EMOJI_RE.sub(lambda m: " ", line)
    # Tidy artifacts inside string literals / labels:
    #   " X" -> "X"   (space directly after an opening quote)
    #   "X " -> "X"   (space directly before a closing quote)
    #   "a  b" -> "a b"  (double spaces not at line start)
    line = re.sub(r'(?<=["\'`(\[{])\s+', "", line)
    line = re.sub(r'\s+(?=["\'`)\]}]$)', "", line)
    # Collapse multiple spaces on the *right* of content (never indentation).
    # Preserve line-start indentation: only collapse when at least one
    # non-space char precedes the run.
    line = re.sub(r"(\S)\s{2,}", r"\1 ", line)
    return line


def main() -> None:
    total_removed = 0
    for path in TARGETS:
        if not os.path.exists(path):
            print(f"SKIP (missing): {path}")
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        new_text = "".join(strip_line(line) for line in text.splitlines(keepends=True))
        removed = len(EMOJI_RE.findall(text))
        if removed:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_text)
            total_removed += removed
            print(f"STRIPPED {removed:3d} emoji(s): {path}")
        else:
            print(f"clean: {path}")
    print(f"\nTotal emoji instances removed: {total_removed}")


if __name__ == "__main__":
    main()
