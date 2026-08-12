import os, re, sys

EMOJI_RE = re.compile(
    r'[\U0001F000-\U0001FAFF]'      # Emoticons, symbols, misc pictographs
    r'|[\U00002600-\U000027BF]'     # Misc symbols, dingbats
    r'|[\U0001F1E6-\U0001F1FF]'     # Regional indicators
    r'|[\U00002B00-\U00002BFF]'     # Misc symbols and arrows
    r'|[\U0001F900-\U0001F9FF]'     # Supplemental symbols
    r'|[\U00002300-\U000023FF]'     # Misc technical
    r'|[\U00002E80-\U00002EFF]'     # CJK radicals (skip)
)

def scan(path):
    try:
        src = open(path, encoding='utf-8', errors='replace').read()
    except Exception as e:
        return 0
    return len(EMOJI_RE.findall(src))

hits = []
for root, dirs, files in os.walk('.'):
    if '.git' in root or '.venv' in root or 'node_modules' in root or '__pycache__' in root or 'nltk_data' in root:
        continue
    for fn in files:
        if fn.endswith('.py'):
            p = os.path.join(root, fn)
            n = scan(p)
            if n:
                hits.append((n, p))

hits.sort(reverse=True)
total = sum(n for n, _ in hits)
print(f'FILES WITH EMOJIS: {len(hits)}  TOTAL EMOJIS: {total}')
for n, p in hits:
    print(f'  {n:4d}  {p}')
