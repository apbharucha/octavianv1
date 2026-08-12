import subprocess, sys, os

COMMIT = "acf81dc1222ca50236a856247c9e25f4a501d007"

out = subprocess.run(["git", "show", f"{COMMIT}:spreadsheet_generator.py"],
                     capture_output=True, text=True)
if out.returncode != 0:
    print("FAILED to extract:", out.stderr)
    sys.exit(1)

content = out.stdout
with open("/tmp/spreadsheet_recovered.py", "w") as f:
    f.write(content)

print("Extracted to /tmp/spreadsheet_recovered.py, lines:", len(content.splitlines()))

# Find the pass stubs in the recovered version
lines = content.splitlines()
for i, l in enumerate(lines, 1):
    if l.strip() == "pass":
        # Show surrounding context
        start = max(0, i - 6)
        print(f"\n--- pass stub at line {i} (context) ---")
        for j in range(start, min(len(lines), i + 2)):
            print(f"{j+1:5d} | {lines[j]}")

# Check for IB engine wiring
print("\n=== ib_excel_engine / model engine refs in recovered ===")
for kw in ["ib_excel_engine", "build_mna_workbook", "build_lbo_workbook",
           "build_dcf_workbook", "build_comps_workbook", "get_dcf_engine",
           "get_lbo_engine", "get_mna_engine", "get_financial_model"]:
    print(f"  {kw}: {content.count(kw)}")

# Compare function list
import re
defs_rec = re.findall(r"def (_generate_\w+)\(", content)
defs_cur = re.findall(r"def (_generate_\w+)\(", open("spreadsheet_generator.py").read())
print("\n=== recovered _generate funcs ===")
print(defs_rec)
print("\n=== current _generate funcs ===")
print(defs_cur)
