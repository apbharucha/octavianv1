import ast
src = open('ai_chatbot.py', encoding='utf-8').read()
lines = src.split('\n')
tree = ast.parse(src)

targets = ['_get_real_time_data', '_create_advanced_price_chart', '_create_prediction_chart']

for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == 'show_advanced_chatbot':
        print(f"show_advanced_chatbot: {node.lineno}-{node.end_lineno}")
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in targets:
                print(f"  {child.name}: {child.lineno}-{child.end_lineno}")
        # what else is in show_advanced_chatbot body besides these?
        others = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name not in targets]
        print("  other nested defs:", others)
        # also find where the targets end relative to show_advanced_chatbot end
        last = node.body[-1]
        print(f"  last statement in fn: {type(last).__name__} @ {getattr(last, 'lineno', '?')}-{getattr(last, 'end_lineno', '?')}")

# Also: check the class end
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == 'OctavianEnhancedChatbot':
        print(f"\nClass: {node.lineno}-{node.end_lineno}")
        last = node.body[-1]
        print(f"  last class member: {type(last).__name__} {getattr(last, 'name', '')} @ {last.lineno}-{last.end_lineno}")
