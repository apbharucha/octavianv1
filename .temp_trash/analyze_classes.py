import ast
src = open('ai_chatbot.py', encoding='utf-8').read()
tree = ast.parse(src)
for node in tree.body:
    if isinstance(node, ast.ClassDef):
        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        print(f"CLASS {node.name} @ line {node.lineno} - line {node.end_lineno}: {len(methods)} methods")
        names = [m.name for m in methods]
        print("  has _create_advanced_price_chart:", "_create_advanced_price_chart" in names)
        print("  has _create_prediction_chart:", "_create_prediction_chart" in names)
        print("  has _get_real_time_data:", "_get_real_time_data" in names)
        print("  first 3:", names[:3])
        print("  last 3:", names[-3:])
