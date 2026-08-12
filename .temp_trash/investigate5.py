import subprocess

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

print("=" * 60)
print("A. Second process_unbiased_query (654 onwards) - first 100 lines")
print("=" * 60)
print(sh("sed -n '654,760p' ai_chatbot.py"))

print("=" * 60)
print("B. Where does the class end / which methods exist around 620-654")
print("=" * 60)
print(sh("sed -n '600,654p' ai_chatbot.py"))
