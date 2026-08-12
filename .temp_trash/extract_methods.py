import ast
src = open('/tmp/ai_chatbot_head.py', encoding='utf-8').read()
lines = src.split('\n')
tree = ast.parse(src)

targets = ['_get_real_time_data', '_create_advanced_price_chart', '_create_prediction_chart']

for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == 'OctavianEnhancedChatbot':
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in targets:
                # extract source lines: 1-based lineno
                start = child.lineno - 1
                end = child.end_lineno
                body = '\n'.join(lines[start:end])
                # dedent: the methods are indented 4 spaces within the class
                import textwrap
                body = textwrap.dedent(body)
                out = f"# --- restored from HEAD commit 3b2f657 ---\n{body}\n"
                fn = f'/tmp/method_{child.name}.py'
                with open(fn, 'w') as f:
                    f.write(out)
                print(f"Wrote {fn} ({child.end_lineno - child.lineno + 1} lines)")
