#!/usr/bin/env python3
"""Repair line-join damage caused by the emoji strip.

The strip script removed whitespace (including the trailing newline) after
opening quotes/brackets and after commas, joining the next line onto the same
line.  The original indentation of the joined continuation is preserved in the
text, so we can deterministically re-split at any  quote/bracket/comma followed
by 2+ spaces then a non-space character.

We split from right to left so earlier splits don't shift later positions, and
we only treat brackets/quotes/commas as join points when the following run of
2+ spaces is immediately followed by a content character (r', ", ', #, etc.).
"""
import re
import sys

# Join point: one of " ' ` ( [ { ,  followed by 2+ spaces then a token char
JOIN_RE = re.compile(r'(["\'\`\(\[{,])\s{2,}(?=\S)')


def split_line(line: str) -> str:
    """Split a line at every join point, re-inserting a newline before the
    preserved indentation run."""
    matches = list(JOIN_RE.finditer(line))
    if not matches:
        return line
    # Build result from right to left.
    out = []
    end = len(line)
    for m in reversed(matches):
        out.append(line[m.end(1):end])          # continuation (spaces + content)
        out.append("\n")
        out.append(line[: m.end(1)])            # up to and including the char
        end = m.start(1)
    # matches[0] handled by loop; prepend the head
    out.reverse()
    head = line[: matches[0].start(1)]
    # Reassemble: head + first-split-part + \n + ...
    return head + "".join(out) if False else _reassemble(line, matches)


def _reassemble(line: str, matches: list) -> str:
    """Assemble: everything up to first char stays first."""
    parts = []
    pos = 0
    for m in matches:
        parts.append(line[pos:m.start(1)])   # text before char
        parts.append(line[m.start(1):m.end(1)])  # the char itself
        parts.append("\n")
        pos = m.end(1)
        # The continuation indentation is the run of spaces between end(1) and next content.
        # We must keep those spaces as the new line's leading whitespace.
        # Since we only matched when \S follows, pos points right after the char.
        # The spaces are already part of the tail; find where they end.
        mm = re.match(r'[ \t]+', line[pos:])
        if mm:
            parts.append(mm.group(0))   # keep indentation
            pos += mm.end()
        else:
            parts.append("")
    parts.append(line[pos:])
    return "".join(parts)


def repair_file(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines = text.splitlines(keepends=True)
    changed = 0
    new_lines = []
    for ln in lines:
        stripped = ln.rstrip("\n")
        if JOIN_RE.search(stripped):
            # Only attempt split if line actually looks like a join (2+ spaces).
            rebuilt = _reassemble(stripped, list(JOIN_RE.finditer(stripped)))
            rebuilt = rebuilt.rstrip("\n") + "\n"
            if rebuilt != stripped + "\n":
                changed += 1
            new_lines.append(rebuilt)
        else:
            new_lines.append(ln)
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(new_lines))
    return changed


if __name__ == "__main__":
    targets = [
        'document_analyzer.py',
        'financial_model_generator_ui.py',
        'futures_simulation_grader.py',
        'options_simulation_grader.py',
        'portfolio_chatbot_context.py',
        'terms_of_service.py',
        'notification_settings_ui.py',
    ]
    for p in targets:
        try:
            n = repair_file(p)
            print(f"{p}: {n} lines split")
        except Exception as e:
            print(f"{p}: ERROR {e}")
