import ast
import textwrap

# Working tree versions (nested inside show_advanced_chatbot)
src_wt = open('ai_chatbot.py', encoding='utf-8').read()
lines_wt = src_wt.split('\n')
tree_wt = ast.parse(src_wt)

targets = ['_get_real_time_data', '_create_advanced_price_chart', '_create_prediction_chart']

wt_versions = {}
for node in tree_wt.body:
    if isinstance(node, ast.FunctionDef) and node.name == 'show_advanced_chatbot':
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in targets:
                body = '\n'.join(lines_wt[child.lineno - 1:child.end_lineno])
                wt_versions[child.name] = textwrap.dedent(body)

for name in targets:
    wt = wt_versions.get(name)
    if wt is None:
        print(f"{name}: NOT FOUND in working tree show_advanced_chatbot")
        continue
    # HEAD version
    head = open(f'/tmp/method_{name}.py', encoding='utf-8').read()
    # strip the marker comment
    head_clean = head.replace('# --- restored from HEAD commit 3b2f657 ---\n', '')
    if head_clean.strip() == wt.strip():
        print(f"{name}: IDENTICAL between HEAD class version and working-tree nested version")
    else:
        print(f"{name}: DIFFERENT (len head={len(head_clean)}, len wt={len(wt)})")
        # quick first-diff
        import difflib
        d = list(difflib.unified_diff(head_clean.split('\n'), wt.split('\n'), lineterm='', n=1))
        print('  diff lines:', len(d))
        for x in d[:20]:
            print('   ', x)
