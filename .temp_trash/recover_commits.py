import subprocess, sys, os, json

# Find which dangling commits contain a spreadsheet_generator.py with real implementations
r = subprocess.run(["git", "fsck", "--lost-found"], capture_output=True, text=True)
dangling = [l.split()[-1] for l in r.stdout.splitlines() if "dangling commit" in l]
print("dangling commits:", len(dangling))

results = []
for c in dangling:
    info = {}
    out = subprocess.run(["git", "show", f"{c}:spreadsheet_generator.py"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        info["has_file"] = False
    else:
        content = out.stdout
        info["has_file"] = True
        info["lines"] = len(content.splitlines())
        info["def_generate"] = content.count("def _generate")
        info["pass_stubs"] = sum(
            1 for l in content.splitlines() if l.strip() == "pass"
        )
        info["comps_workbook"] = "build_comps_workbook" in content
        info["ib_excel"] = "ib_excel_engine" in content
    # commit message
    msg = subprocess.run(
        ["git", "log", "-1", "--format=%s", c], capture_output=True, text=True
    ).stdout.strip()
    info["msg"] = msg
    results.append((c, info))

for c, info in results:
    if info.get("has_file"):
        print(f"=== {c} | {info['msg'][:60]}")
        print(f"    lines={info.get('lines')} def_generate={info.get('def_generate')} "
              f"pass_stubs={info.get('pass_stubs')} comps_workbook={info.get('comps_workbook')} "
              f"ib_excel={info.get('ib_excel')}")
    else:
        print(f"=== {c} | {info['msg'][:60]} | (no spreadsheet_generator.py)")
