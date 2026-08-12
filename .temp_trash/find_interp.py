import subprocess
for pat in ['_call_lm_studio_interpreter', '_lm_studio_narrative', 'lm_studio_online', 'lm_studio_interpreter', 'call_lm_studio']:
    r = subprocess.run(['grep', '-rn', pat, '--include=*.py', '.'], capture_output=True, text=True)
    hits = [l for l in r.stdout.split('\n') if l and '__pycache__' not in l and '.temp_trash' not in l and '.venv' not in l]
    print(f'--- {pat} ({len(hits)})')
    for h in hits[:10]:
        print('   ', h)
    print()
