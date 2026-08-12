import re
import os

EMOJI_RE = re.compile(
    r"[\U0001F000-\U0001FAFF]|[\u2600-\u27BF]|[\U0001F1E6-\U0001F1FF]|[\u2B00-\u2BFF]|[\u23E9-\u23FA]"
)
hits = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".venv", ".git", ".temp_trash", "nltk_data")]
    for fn in files:
        if fn.endswith(".py"):
            p = os.path.join(root, fn)
            try:
                for i, line in enumerate(open(p, encoding="utf-8"), 1):
                    if EMOJI_RE.search(line):
                        hits.append(f"{p}:{i}")
            except Exception:
                pass
print(f"PRODUCTION EMOJI HITS: {len(hits)}")
for h in hits[:20]:
    print(" ", h)
