import sys, os, subprocess
sys.path.insert(0, os.path.abspath('.'))

print('=== TEST FILES ===')
r = subprocess.run(['ls', 'tests/'], capture_output=True, text=True)
print(r.stdout)

print('=== STREAMLIT VERSION ===')
r = subprocess.run([sys.executable, '-c', 'import streamlit; print(streamlit.__version__)'], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[:200])

print('=== APP TEST AVAILABLE? ===')
r = subprocess.run([sys.executable, '-c', 'from streamlit.testing.v1 import AppTest; print("AppTest OK")'], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[:300])

print('=== MAIN.PY TOP-LEVEL STRUCTURE ===')
src = open('main.py', encoding='utf-8').read()
lines = src.split('\n')
print('lines:', len(lines))
for i, l in enumerate(lines, 1):
    s = l.strip()
    if s.startswith('def ') or s.startswith('st.set_page_config') or s.startswith('def main'):
        print(f'{i}: {s[:100]}')
