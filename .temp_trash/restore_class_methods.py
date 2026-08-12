import ast

path = 'ai_chatbot.py'
lines = open(path, encoding='utf-8').read().split('\n')

# The three nested methods inside show_advanced_chatbot (1-based lines)
# _get_real_time_data: 4790-4865, _create_advanced_price_chart: 4867-5035, _create_prediction_chart: 5037-5161
start, end = 4790, 5161  # inclusive 1-based
methods_block = lines[start - 1:end]

# They are indented at 4 spaces (same as class methods) — but verify:
assert all(l.startswith('    ') or not l.strip() for l in methods_block), "unexpected indentation"

# The class ends at 3178 (1-based). Insert after line 3178.
insert_at = 3178  # 0-based index of the line AFTER which to insert = line 3178 itself in 0-based (line 3179)

# Validate boundaries
assert 'return levels' in lines[insert_at - 1], lines[insert_at - 1]
# find the 'def show_advanced_chatbot' line after insert_at
next_def = insert_at
while 'def show_advanced_chatbot' not in lines[next_def]:
    next_def += 1
    assert next_def < len(lines), 'def not found'
assert next_def - insert_at == 2, f"expected 2 blank lines gap, got {next_def - insert_at}: {lines[insert_at:next_def]!r}"

# Build new content: blank line + methods + blank lines
sep = [''] * 2
new_lines = lines[:insert_at] + sep + methods_block + sep + lines[insert_at:]

with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines))

# Validate
src = open(path, encoding='utf-8').read()
ast.parse(src)  # raises on syntax error
tree = ast.parse(src)
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == 'OctavianEnhancedChatbot':
        methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        print(f"Class now has {len(methods)} methods")
        for want in ['_get_real_time_data', '_create_advanced_price_chart', '_create_prediction_chart']:
            print(f"  has {want}: {want in methods}")
print("RESTORED OK")
