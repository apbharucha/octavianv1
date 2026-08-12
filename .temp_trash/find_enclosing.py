import ast
src = open('ai_chatbot.py', encoding='utf-8').read()
lines = src.split('\n')
tree = ast.parse(src)

# Find the enclosing node for line 4790 (1-based)
target = 4790
def walk(node, depth=0):
    if hasattr(node, 'lineno'):
        start = getattr(node, 'lineno', 0)
        end = getattr(node, 'end_lineno', start)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Try, ast.If, ast.For, ast.While, ast.With)):
            if start <= target <= end:
                indent = '  ' * depth
                print(f"{indent}{type(node).__name__}: {getattr(node, 'name', '')} @ {start}-{end}")
                for child in ast.iter_child_nodes(node):
                    walk(child, depth + 1)

for node in tree.body:
    walk(node)

# Also check what's between 3175-3200 to see structure after class
print("\n=== lines 3175-3200 ===")
for i in range(3174, min(3200, len(lines))):
    print(f"{i+1}: {lines[i]!r}")
