import ast, subprocess, sys

# 1. Compile check the recovered version
r = subprocess.run(["python3", "-m", "py_compile", "/tmp/spreadsheet_recovered.py"],
                   capture_output=True, text=True)
print("RECOVERED COMPILE:", "OK" if r.returncode == 0 else "FAIL")
if r.returncode != 0:
    print(r.stderr[:2000])

# 2. Compare top-level function signatures between recovered and current
def funcs(path):
    tree = ast.parse(open(path).read())
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in n.args.args]
            out[n.name] = args
    return out

rec = funcs("/tmp/spreadsheet_recovered.py")
cur = funcs("spreadsheet_generator.py")

print("\n=== Functions ONLY in recovered (missing from current) ===")
for name in sorted(set(rec) - set(cur)):
    print(f"  {name}({', '.join(rec[name])})")

print("\n=== Functions ONLY in current (not in recovered) ===")
for name in sorted(set(cur) - set(rec)):
    print(f"  {name}({', '.join(cur[name])})")

print("\n=== Signature differences ===")
for name in sorted(set(rec) & set(cur)):
    if rec[name] != cur[name]:
        print(f"  {name}: rec({rec[name]}) vs cur({cur[name]})")

# 3. Check line counts
print("\nRecovered lines:", len(open("/tmp/spreadsheet_recovered.py").readlines()))
print("Current lines:", len(open("spreadsheet_generator.py").readlines()))
