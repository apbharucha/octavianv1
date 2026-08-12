import re

for path in ['document_analyzer.py', 'financial_model_generator_ui.py']:
    lines = open(path, encoding='utf-8').read().splitlines()
    print(f'=== {path} ===')
    # Collapse-join: content char + 3+ spaces + content that looks like a new statement
    # (statement starts: identifier, st., self., return, for, if, #, ', ", {, f", r")
    pat = re.compile(r'(\S)\s{3,}([A-Za-z_#"\'`@])')
    for i, ln in enumerate(lines, 1):
        if pat.search(ln):
            print(f'  {i}: {ln.strip()[:130]}')
